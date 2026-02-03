#!/usr/bin/env python3
"""
Train Refinement Network with Geometric Attention

This script extends the training pipeline to include geometric attention
in the training process, making the attention mechanism trainable.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from lapa.lapa_pipeline import LAPAPipeline
from lapa.models.track_reconstruction import TriangulationModule
from lapa.models.geometric_attention_fixed import TrackCorrespondence, GeometricAttention
from lapa.models.geometric_attention_sfm import TrackCorrespondenceSfM, GeometricAttentionSfM
from lapa.data.tap3d_loader import TAP3DLoader
from lapa.visualization.visualizer import LAPAVisualizer


class CombinedLoss(nn.Module):
    """
    Combined loss function for training both refinement and geometric attention.
    
    Includes:
    1. Reconstruction loss: How well the refined points match ground truth
    2. Identity loss: How well the refined points project to the original 2D points
    3. Attention loss: How well the attention mechanism captures point correspondences
    """
    
    def __init__(self, lambda_identity=1.0, lambda_temporal=0.5, lambda_attention=0.5):
        """
        Initialize the combined loss.
        
        Args:
            lambda_identity: Weight for identity loss
            lambda_temporal: Weight for temporal consistency loss
            lambda_attention: Weight for attention mechanism loss
        """
        super().__init__()
        self.lambda_identity = lambda_identity
        self.lambda_temporal = lambda_temporal
        self.lambda_attention = lambda_attention
        
    def compute_reconstruction_loss(self, pred_points, gt_points, visibility):
        """
        Compute reconstruction loss between predicted and ground truth 3D points.
        
        Args:
            pred_points: Predicted 3D points
            gt_points: Ground truth 3D points
            visibility: Visibility mask for ground truth
            
        Returns:
            Normalized reconstruction loss
        """
        # Only compute loss for visible points
        mask = visibility.bool()
        
        # MSE loss between predicted and ground truth 3D points
        loss = F.mse_loss(
            pred_points[mask], 
            gt_points[mask], 
            reduction='none'
        )
        
        # Normalize loss by dividing by the maximum range of ground truth points
        with torch.no_grad():
            valid_points = gt_points[mask]
            if valid_points.shape[0] > 0:
                min_vals = torch.min(valid_points, dim=0)[0]
                max_vals = torch.max(valid_points, dim=0)[0]
                ranges = max_vals - min_vals
                scale_factor = torch.max(ranges) + 1e-6
            else:
                scale_factor = torch.tensor(1.0, device=gt_points.device)
        
        # Apply normalization
        normalized_loss = loss / (scale_factor ** 2)
        
        return torch.mean(normalized_loss)
    
    def compute_identity_loss(self, pred_points, projection_matrices, tracks_2d):
        """
        Compute identity loss by projecting 3D points to 2D and comparing with original 2D tracks.
        
        Args:
            pred_points: Predicted 3D points
            projection_matrices: Camera projection matrices
            tracks_2d: Original 2D tracks
            
        Returns:
            Normalized identity loss
        """
        # Project predicted 3D points to 2D in each view
        batch_size, num_frames, num_points, _ = pred_points.shape
        num_views = len(tracks_2d)
        
        identity_loss = 0.0
        total_projections = 0
        
        for view_idx in range(num_views):
            proj_matrix = projection_matrices[view_idx]
            view_tracks_2d = tracks_2d[view_idx]
            
            # Get the number of points in this view
            view_num_points = view_tracks_2d.shape[2]
            
            # Only use the minimum number of points between predicted and ground truth
            min_points = min(num_points, view_num_points)
            
            # Project 3D points to 2D
            for frame_idx in range(num_frames):
                for batch_idx in range(batch_size):
                    # Get 3D points for this batch and frame (only up to min_points)
                    points_3d = pred_points[batch_idx, frame_idx, :min_points]
                    
                    # Get projection matrix for this batch
                    P = proj_matrix[batch_idx]
                    
                    # Add homogeneous coordinate
                    points_3d_homo = torch.cat(
                        [points_3d, torch.ones(points_3d.shape[0], 1, device=points_3d.device)], 
                        dim=1
                    )  # Shape: (num_points, 4)
                    
                    # Project to 2D
                    points_2d_homo = torch.matmul(P, points_3d_homo.t())  # Shape: (3, num_points)
                    points_2d_homo = points_2d_homo.t()  # Shape: (num_points, 3)
                    
                    # Convert to Cartesian coordinates
                    points_2d = points_2d_homo[:, :2] / (points_2d_homo[:, 2:] + 1e-10)
                    
                    # Get original 2D points for this view, batch, and frame
                    original_2d = view_tracks_2d[batch_idx, frame_idx, :min_points]
                    
                    # Get image dimensions to normalize the error
                    # Assuming the image width and height are around 1000 pixels
                    # Using view_tracks_2d to estimate image dimensions
                    with torch.no_grad():
                        # Find non-zero values to avoid skewing the normalization
                        valid_idx = torch.where(
                            torch.sum(torch.abs(original_2d), dim=1) > 1e-6
                        )[0]
                        
                        if len(valid_idx) > 0:
                            valid_points = original_2d[valid_idx]
                            
                            # Get approximate dimensions based on the range
                            x_min, y_min = torch.min(valid_points, dim=0)[0]
                            x_max, y_max = torch.max(valid_points, dim=0)[0]
                            
                            # Use range for normalization (add small epsilon)
                            width = max(224.0, x_max - x_min + 1e-6)
                            height = max(224.0, y_max - y_min + 1e-6)
                        else:
                            # Default normalization values if no valid points
                            width, height = 224.0, 224.0
                            
                    # Compute normalized MSE between projected and original 2D points
                    # Normalize by image dimensions to make the loss scale-invariant
                    error_x = ((points_2d[:, 0] - original_2d[:, 0]) / width) ** 2
                    error_y = ((points_2d[:, 1] - original_2d[:, 1]) / height) ** 2
                    
                    # Average error across points
                    view_loss = torch.mean(error_x + error_y)
                    identity_loss += view_loss
                    total_projections += 1
                    
        # Average across all projections
        if total_projections > 0:
            identity_loss = identity_loss / total_projections
        
        return identity_loss
    

    
    def compute_attention_loss(self, correspondence_matrices, gt_visibility):
        """
        Compute loss for the attention mechanism with enhanced diagonal bias.
        
        Args:
            correspondence_matrices: Correspondence matrices from the attention mechanism
            gt_visibility: Ground truth visibility masks of shape (batch_size, num_frames, num_points)
            
        Returns:
            Attention mechanism loss
        """
        # If there's no correspondence data, return zero loss
        if correspondence_matrices is None or len(correspondence_matrices) == 0:
            return torch.tensor(0.0, device=gt_visibility.device)
        
        # Get only the first frame's visibility since we're processing frames individually
        # Shape: (batch_size, num_points)
        frame_visibility = gt_visibility[:, 0]
        
        attention_loss = 0.0
        count = 0
        
        # For each view pair in the correspondence matrix
        for i, view_correspondences in enumerate(correspondence_matrices):
            for j, correspondence in enumerate(view_correspondences):
                # Skip self-attention (which is None)
                if correspondence is None:
                    continue
                
                # Get correspondence for this view pair
                # correspondence shape: (batch_size, num_points_i, num_points_j)
                
                batch_size = correspondence.shape[0]
                num_points_i = correspondence.shape[1]
                num_points_j = correspondence.shape[2]
                
                # Create a target correspondence matrix with stronger diagonal pattern
                for b in range(batch_size):
                    # Extract correspondence for this batch
                    corr = correspondence[b]  # (num_points_i, num_points_j)
                    
                    # Create enhanced ground truth correspondence matrix
                    gt_corr = torch.zeros_like(corr)
                    
                    # Set diagonal to 1.0 with a small window around it
                    # This creates a more robust pattern than just the diagonal
                    min_points = min(num_points_i, num_points_j)
                    for p in range(min_points):
                        if p < frame_visibility.shape[1] and frame_visibility[b, p]:
                            # Set the diagonal element
                            gt_corr[p, p] = 1.0
                            
                            # Add a small window around the diagonal (with lower weights)
                            window_size = 2
                            for offset in range(1, window_size + 1):
                                # Add near-diagonal elements with decaying weights
                                weight = 0.7 ** offset
                                
                                # Set points to the right (if in bounds)
                                if p + offset < min_points:
                                    gt_corr[p, p + offset] = weight
                                    gt_corr[p + offset, p] = weight
                                
                                # Set points to the left (if in bounds)
                                if p - offset >= 0:
                                    gt_corr[p, p - offset] = weight
                                    gt_corr[p - offset, p] = weight
                    
                    # Apply a focal loss-like weighting to emphasize learning positive correspondences
                    # This helps with the imbalanced nature of the correspondence problem
                    # (most elements should be 0, only a few should be 1)
                    pos_weight = 2.0  # Weighting for positive examples
                    weights = torch.ones_like(gt_corr) + gt_corr * pos_weight
                    
                    # Compute weighted BCE loss
                    loss = F.binary_cross_entropy_with_logits(
                        corr, gt_corr, weight=weights, reduction='mean'
                    )
                    
                    attention_loss += loss
                    count += 1
        
        # Average the loss
        if count > 0:
            attention_loss = attention_loss / count
        else:
            attention_loss = torch.tensor(0.0, device=gt_visibility.device)
        
        return attention_loss
    
    def forward(self, 
               refined_points, 
               gt_points, 
               visibility, 
               projection_matrices, 
               tracks_2d,
               correspondence_matrices=None):
        """
        Forward pass of the combined loss function.
        
        Args:
            refined_points: Refined 3D points from the network
            gt_points: Ground truth 3D points
            visibility: Visibility mask for ground truth
            projection_matrices: Camera projection matrices
            tracks_2d: Original 2D tracks
            correspondence_matrices: Correspondence matrices from attention mechanism
            
        Returns:
            Dictionary with loss components
        """
        # Compute reconstruction loss
        reconstruction_loss = self.compute_reconstruction_loss(
            refined_points, gt_points, visibility
        )
        
        # Compute identity loss
        identity_loss = self.compute_identity_loss(
            refined_points, projection_matrices, tracks_2d
        )
        
        # Compute attention loss if available
        attention_loss = torch.tensor(0.0, device=refined_points.device)
        if correspondence_matrices is not None:
            attention_loss = self.compute_attention_loss(
                correspondence_matrices, visibility
            )
        
        # Combine losses with weights
        total_loss = (
            1.0 * reconstruction_loss + 
            self.lambda_identity * identity_loss +
            self.lambda_attention * attention_loss
        )
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'identity_loss': identity_loss,
            'attention_loss': attention_loss
        }


class TrainerWithAttention:
    """
    Enhanced trainer for the Refinement Network with trainable Geometric Attention.
    """
    
    def __init__(self, 
                 data_dir: str, 
                 calibration_file: str,
                 target_size=(224, 224),
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 geometry_type='epipolar',
                 volume_size=16):
        """
        Initialize the enhanced trainer.
        
        Args:
            data_dir: Directory containing TAP3D data
            calibration_file: Path to the calibration file
            target_size: Target size for image resizing
            device: Device to run on
            geometry_type: Type of geometric constraint ('epipolar' or 'sfm')  
            volume_size: Size of the volumetric grid (16 or 24)
        """
        self.data_dir = data_dir
        self.calibration_file = calibration_file
        self.target_size = target_size
        self.device = device
        self.geometry_type = geometry_type
        self.volume_size = volume_size
        
        # Initialize data loader
        self.data_loader = TAP3DLoader(data_dir, calibration_file)
        
        # Initialize triangulation module for DLT
        self.triangulation = TriangulationModule().to(device)
        
        # Initialize the track correspondence module based on geometry type
        if geometry_type == 'epipolar':
            print(f"Using epipolar geometry with volume size {volume_size}")
            self.track_correspondence = TrackCorrespondence(
                feature_dim=128,
                volume_size=volume_size,
                num_heads=4
            ).to(device)
        elif geometry_type == 'sfm':
            print(f"Using SfM geometry with volume size {volume_size}")
            self.track_correspondence = TrackCorrespondenceSfM(
                feature_dim=128,
                volume_size=volume_size,
                num_heads=4
            ).to(device)
        else:
            raise ValueError(f"Unknown geometry type: {geometry_type}. Must be 'epipolar' or 'sfm'")
        
        # Initialize refinement network
        self.refinement_network = RefinementNetwork(
            hidden_dim=512,
            dropout=0.2,
            max_offset=0.3
        ).to(device)
        
        # Apply custom initialization to the attention mechanism
        self._initialize_attention_weights()
        
        # Initialize loss function with enhanced weights for attention
        self.loss_fn = CombinedLoss(
            lambda_identity=1.0,
            lambda_temporal=0.5,
            lambda_attention=2.0  # Increased weight for attention loss
        )
        
        # Initialize optimizer for both refinement and attention networks
        self.optimizer = torch.optim.Adam([
            {'params': self.refinement_network.parameters()},
            {'params': self.track_correspondence.parameters()}
        ], lr=0.001)
        
        # Add learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.7, patience=3, verbose=True)
            
        # Initialize pipeline (we'll use it for data preparation)
        self.pipeline = LAPAPipeline(
            data_dir=data_dir,
            calibration_file=calibration_file,
            target_size=target_size,
            device=device
        )
        
        # Replace pipeline's track_correspondence with our trainable one
        self.pipeline.track_correspondence = self.track_correspondence
        
        # Initialize visualizer for debugging
        self.visualizer = LAPAVisualizer(output_dir="outputs/attention_training")
        os.makedirs("outputs/attention_training", exist_ok=True)
    
    def _initialize_attention_weights(self):
        """
        Apply custom initialization to the attention mechanism to encourage
        stronger diagonal attention and better starting weights.
        """
        # Access the geometric attention module
        geo_attention = self.track_correspondence.geometric_attention
        
        # Initialize the track encoder with larger weights
        for name, param in geo_attention.track_encoder.named_parameters():
            if 'weight' in name:
                # Initialize with uniform distribution for better gradient flow
                nn.init.kaiming_uniform_(param, a=1.0)
            elif 'bias' in name:
                # Initialize bias with small positive values
                nn.init.constant_(param, 0.1)
        
        # For the multi-head attention, we'll keep PyTorch's default initialization
        # as it's specifically designed for attention mechanisms
        
        # Initialize the projection layer with larger weights
        nn.init.kaiming_uniform_(geo_attention.projection.weight, a=1.0)
        nn.init.constant_(geo_attention.projection.bias, 0.1)
        
        print("Applied custom initialization to attention mechanism")
    
    def prepare_ground_truth(self, data, view_names):
        """
        Prepare ground truth 3D tracks from the TAP3D dataset.
        
        Args:
            data: Dictionary with loaded data
            view_names: List of view names
            
        Returns:
            Ground truth 3D tracks tensor and visibility mask
        """
        # We'll use the 3D tracks from the first view as ground truth
        # These are already in world coordinates
        first_view = view_names[0]
        view_data = data['views'][first_view]
        
        # Get 3D tracks and visibility
        tracks_xyz = view_data['tracks_xyz']
        visibility = view_data['visibility']
        
        # Convert to tensor
        gt_tracks_3d = torch.from_numpy(tracks_xyz).float().to(self.device)
        gt_visibility = torch.from_numpy(visibility).bool().to(self.device)
        
        # Add batch dimension to match model output
        gt_tracks_3d = gt_tracks_3d.unsqueeze(0)
        gt_visibility = gt_visibility.unsqueeze(0)
        
        # Find the minimum number of points across all views to match the predicted shape
        min_points = min([data['views'][view]['tracks_xyz'].shape[1] for view in view_names])
        
        # Truncate ground truth to match the minimum number of points
        gt_tracks_3d = gt_tracks_3d[:, :, :min_points, :]
        gt_visibility = gt_visibility[:, :, :min_points]
        
        print(f"Ground truth 3D tracks shape: {gt_tracks_3d.shape} (truncated to {min_points} points)")
        
        return gt_tracks_3d, gt_visibility
    
    def train_epoch(self, view_names, num_frames=60):
        """
        Train for one epoch on the specified views.
        
        Args:
            view_names: List of view names to use
            num_frames: Number of frames to process
            
        Returns:
            Dictionary with training metrics
        """
        # Set models to training mode
        self.refinement_network.train()
        self.track_correspondence.train()
        
        # Load data
        data = self.data_loader.load_multi_view_data(view_names, self.target_size)
        
        # Prepare inputs
        tracks_2d, projection_matrices = self.pipeline.prepare_inputs(data)
        
        # Prepare ground truth
        gt_tracks_3d, gt_visibility = self.prepare_ground_truth(data, view_names)
        
        # Process frames
        total_loss = 0.0
        recon_loss_sum = 0.0
        identity_loss_sum = 0.0
        attention_loss_sum = 0.0
        metrics = {}
        
        # Ensure we don't exceed the available frames
        max_frames = min(num_frames, gt_tracks_3d.shape[1])
        
        # Process each frame
        for frame_idx in tqdm(range(max_frames), desc="Training"):
            # Get 2D tracks for this frame
            frame_tracks_2d = [track[:, frame_idx:frame_idx+1, :, :] for track in tracks_2d]
            
            # Run track correspondence (with attention mechanism)
            correspondence_result = self.track_correspondence(
                frame_tracks_2d, projection_matrices)
            
            # Extract correspondence matrices and store for loss calculation
            frame_correspondences = correspondence_result["correspondences"][0]
            
            # Apply DLT triangulation to get initial 3D points
            with torch.no_grad():
                # For DLT triangulation, we need to extract the 2D points from each view
                # and pass them directly to the triangulation module
                points_2d_for_triangulation = []
                for view_idx in range(len(frame_tracks_2d)):
                    # Get 2D points for this view and frame
                    # Shape: (batch_size, 1, num_points, 2)
                    points_2d_for_triangulation.append(frame_tracks_2d[view_idx][:, 0])
                
                # Triangulate points using DLT
                dlt_points = self.triangulation.triangulate_points_batch(
                    points_2d_for_triangulation,
                    projection_matrices
                )
                
                # Add frame dimension to match expected shape
                dlt_points = dlt_points.unsqueeze(1)
            
            # Apply refinement network
            refined_points = self.refinement_network(dlt_points)
            
            # Compute combined loss
            loss_dict = self.loss_fn(
                refined_points,
                gt_tracks_3d[:, frame_idx:frame_idx+1],
                gt_visibility[:, frame_idx:frame_idx+1],
                projection_matrices,
                frame_tracks_2d,
                frame_correspondences
            )
            
            # Backpropagation
            self.optimizer.zero_grad()
            loss_dict['total_loss'].backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss_dict['total_loss'].item()
            recon_loss_sum += loss_dict['reconstruction_loss'].item()
            identity_loss_sum += loss_dict['identity_loss'].item()
            attention_loss_sum += loss_dict['attention_loss'].item() if 'attention_loss' in loss_dict else 0.0
            
            # Log every 10 frames
            if frame_idx % 10 == 0:
                print(f"Frame {frame_idx}: ")
                print(f"  Total Loss: {loss_dict['total_loss'].item():.6f}")
                print(f"  Reconstruction Loss: {loss_dict['reconstruction_loss'].item():.6f}")
                print(f"  Identity Loss: {loss_dict['identity_loss'].item():.6f}")
                print(f"  Attention Loss: {loss_dict['attention_loss'].item():.6f}")
                
                # Visualize results occasionally
                if frame_idx % 30 == 0:
                    with torch.no_grad():
                        # Visualize DLT points vs refined points vs ground truth
                        self.visualize_comparison(
                            dlt_points[0, 0].cpu().numpy(),
                            refined_points[0, 0].detach().cpu().numpy(),
                            gt_tracks_3d[0, frame_idx].cpu().numpy(),
                            gt_visibility[0, frame_idx].cpu().numpy(),
                            frame_idx
                        )
                        
                        # Visualize attention grid (for debugging)
                        self.visualize_attention_grid(frame_idx)
        
        # Update learning rate based on average loss
        avg_loss = total_loss / max_frames
        self.scheduler.step(avg_loss)
        
        # Return average metrics
        return {
            'total_loss': avg_loss,
            'reconstruction_loss': recon_loss_sum / max_frames,
            'identity_loss': identity_loss_sum / max_frames,
            'attention_loss': attention_loss_sum / max_frames
        }
    
    def visualize_comparison(self, dlt_points, refined_points, gt_points, visibility, frame_idx):
        """
        Visualize comparison between DLT, refined, and ground truth points.
        
        Args:
            dlt_points: DLT triangulated points
            refined_points: Refined points from the network
            gt_points: Ground truth points
            visibility: Visibility mask
            frame_idx: Current frame index
        """
        # Create 3D plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Only plot visible points
        visible = visibility.astype(bool)
        
        # Plot DLT points
        ax.scatter(dlt_points[:, 0], dlt_points[:, 1], dlt_points[:, 2], 
                  c='blue', marker='o', label='DLT', alpha=0.6)
        
        # Plot refined points
        ax.scatter(refined_points[:, 0], refined_points[:, 1], refined_points[:, 2], 
                  c='red', marker='^', label='Refined', alpha=0.6)
        
        # Plot ground truth points (only visible ones)
        ax.scatter(gt_points[visible, 0], gt_points[visible, 1], gt_points[visible, 2], 
                  c='green', marker='x', label='Ground Truth', alpha=0.6)
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'3D Points Comparison - Frame {frame_idx}')
        ax.legend()
        
        # Save figure
        plt.savefig(f"outputs/attention_training/comparison_frame_{frame_idx}.png")
        plt.close(fig)
    
    def visualize_attention_grid(self, frame_idx):
        """
        Visualize the attention grid for debugging.
        
        Args:
            frame_idx: Current frame index
        """
        # Get the geometric attention module
        geo_attention = self.track_correspondence.geometric_attention
        
        # Create batch and device info
        batch_size = 1
        device = self.device
        
        # Create the 3D grid
        with torch.no_grad():
            grid = geo_attention.create_3d_grid(batch_size, device)
            
            # Get a slice of the grid (mid-plane)
            mid_idx = grid.shape[1] // 2
            grid_slice = grid[0, mid_idx].cpu().numpy()
            
            # Create figure
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot grid slice
            x = grid_slice[:, :, 0].flatten()
            y = grid_slice[:, :, 1].flatten()
            z = grid_slice[:, :, 2].flatten()
            
            ax.scatter(x, y, z, c=z, cmap='viridis', alpha=0.5, s=10)
            
            # Set labels and title
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'Attention Grid Slice - Frame {frame_idx}')
            
            # Save figure
            plt.savefig(f"outputs/attention_training/attention_grid_frame_{frame_idx}.png")
            plt.close(fig)


# Import RefinementNetwork for compatibility
from train_refinement import RefinementNetwork


def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Train volumetric attention model")
    
    # Data parameters
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing TAP3D data")
    parser.add_argument("--calibration_file", type=str, required=True,
                        help="Path to calibration file")
    parser.add_argument("--view_set", type=str, default="boxes",
                        help="View set to use for training")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs to train for")
    parser.add_argument("--num_frames", type=int, default=60,
                        help="Number of frames to process per epoch")
    parser.add_argument("--chunk_size", type=int, default=8,
                        help="Chunk size for processing frames")
    
    # Model parameters
    parser.add_argument("--geometry_type", type=str, default="epipolar", choices=["epipolar", "sfm"],
                        help="Type of geometric constraint to use (epipolar or sfm)")
    parser.add_argument("--grid_size", type=int, default=16, choices=[8, 16, 24],
                        help="Size of the volumetric grid (8, 16, or 24)")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="outputs/attention_training",
                        help="Directory to save outputs")
    
    return parser.parse_args()


def main():
    """
    Main function for training.
    """
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize trainer with specified geometry type and grid size
    trainer = TrainerWithAttention(
        data_dir=args.data_dir,
        calibration_file=args.calibration_file,
        target_size=(224, 224),
        device=device,
        geometry_type=args.geometry_type,
        volume_size=args.grid_size
    )
    
    # Select views based on view set
    if args.view_set == 'boxes':
        view_names = ['boxes_5', 'boxes_6', 'boxes_7']
    elif args.view_set == 'boxes_alt':
        view_names = ['boxes_11', 'boxes_12', 'boxes_17']
    elif args.view_set == 'boxes_alt2':
        view_names = ['boxes_19', 'boxes_22', 'boxes_27']
    elif args.view_set == 'basketball':
        view_names = ['basketball_3', 'basketball_4', 'basketball_5']
    elif args.view_set == 'basketball_alt':
        view_names = ['basketball_6', 'basketball_9', 'basketball_13']
    elif args.view_set == 'basketball_alt2':
        view_names = ['basketball_14', 'basketball_20', 'basketball_24']
    elif args.view_set == 'softball':
        view_names = ['softball_2', 'softball_9', 'softball_14']
    elif args.view_set == 'softball_alt':
        view_names = ['softball_19', 'softball_21', 'softball_23']
    elif args.view_set == 'tennis':
        view_names = ['tennis_2', 'tennis_4', 'tennis_5']
    elif args.view_set == 'tennis_alt':
        view_names = ['tennis_17', 'tennis_22', 'tennis_23']
    elif args.view_set == 'football':
        view_names = ['football_1', 'football_3', 'football_7']
    elif args.view_set == 'football_alt':
        view_names = ['football_16', 'football_19', 'football_21']
    elif args.view_set == 'juggle':
        view_names = ['juggle_4', 'juggle_5', 'juggle_7']
    elif args.view_set == 'juggle_alt':
        view_names = ['juggle_8', 'juggle_9', 'juggle_22']
    else:
        raise ValueError(f"Unknown view set: {args.view_set}")
        
    print(f"Training on view set: {args.view_set}")
    print(f"Views: {view_names}")
    print(f"Using geometry type: {args.geometry_type}")
    print(f"Grid size: {args.grid_size}")
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 40)
        
        # Train for one epoch
        train_metrics = trainer.train_epoch(view_names, args.num_frames)
        
        # Print metrics
        print(f"Train Loss: {train_metrics.get('total_loss', train_metrics.get('loss', 0.0)):.4f}")
        print(f"Reconstruction Loss: {train_metrics.get('reconstruction_loss', 0.0):.4f}")
        print(f"Identity Loss: {train_metrics.get('identity_loss', 0.0):.4f}")
        print(f"Temporal Loss: {train_metrics.get('temporal_loss', 0.0):.4f}")
        print(f"Attention Loss: {train_metrics.get('attention_loss', 0.0):.4f}")
        
        # Update learning rate
        trainer.scheduler.step(train_metrics.get('total_loss', train_metrics.get('loss', 0.0)))
        
        # Get the current loss value
        current_loss = train_metrics.get('total_loss', train_metrics.get('loss', float('inf')))
        
        # Save model if it's the best so far
        if current_loss < best_loss:
            best_loss = current_loss
            best_epoch = epoch + 1
            
            # Save attention model
            attention_path = os.path.join(args.output_dir, f"attention_model_{args.geometry_type}_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'attention_model': trainer.track_correspondence.state_dict(),
                'optimizer': trainer.optimizer.state_dict(),
                'loss': best_loss,
                'total_loss': best_loss,  # Add both keys for compatibility
                'geometry_type': args.geometry_type,
                'grid_size': args.grid_size
            }, attention_path)
            print(f"Saved best model to {attention_path}")
        
        # Always save the latest model
        latest_path = os.path.join(args.output_dir, f"attention_model_{args.geometry_type}_epoch_{args.epochs}.pth")
        torch.save({
            'epoch': epoch + 1,
            'attention_model': trainer.track_correspondence.state_dict(),
            'optimizer': trainer.optimizer.state_dict(),
            'loss': train_metrics.get('total_loss', train_metrics.get('loss', 0.0)),
            'total_loss': train_metrics.get('total_loss', train_metrics.get('loss', 0.0)),  # Add both keys for compatibility
            'geometry_type': args.geometry_type,
            'grid_size': args.grid_size
        }, latest_path)
    
    print("\nTraining complete!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Final model saved to {latest_path}")


if __name__ == "__main__":
    main()
