#!/usr/bin/env python3
"""
Train Refinement Network for 3D Track Reconstruction

This script implements a two-stage approach for 3D reconstruction:
1. Use DLT triangulation to get initial 3D points
2. Use a neural network to refine these points

The network is trained with reconstruction loss and identity loss,
both properly normalized to balance their contributions.
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
from lapa.models.geometric_attention_fixed import TrackCorrespondence
from lapa.models.geometric_attention_sfm import TrackCorrespondenceSfM
from lapa.data.tap3d_loader import TAP3DLoader
from lapa.visualization.visualizer import LAPAVisualizer


class RefinementNetwork(nn.Module):
    """
    Refinement network that takes DLT triangulated points and refines them.
    
    This network learns relative offsets to apply to the DLT points rather than absolute positions,
    ensuring that the refined points stay close to the original DLT points.
    """
    
    def __init__(self, hidden_dim=512, dropout=0.2, max_offset=0.3):
        """
        Initialize the refinement network.
        
        Args:
            hidden_dim: Dimension of hidden layers
            dropout: Dropout probability
            max_offset: Maximum allowed offset as a fraction of the input point magnitude
                       (limits how far the refined point can move from the DLT point)
        """
        super().__init__()
        
        self.max_offset = max_offset
        
        # Network that predicts offsets rather than absolute positions
        self.offset_network = nn.Sequential(
            nn.Linear(3, hidden_dim),  # Input: 3D point from DLT
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3),  # Output: Offset to apply to the DLT point
            nn.Tanh()  # Constrain offsets to [-1, 1] range
        )
        
    def forward(self, points_3d):
        """
        Forward pass of the refinement network.
        
        Args:
            points_3d: 3D points from DLT triangulation, shape (batch_size, num_frames, num_points, 3)
            
        Returns:
            Refined 3D points with the same shape
        """
        original_shape = points_3d.shape
        
        # Reshape to process all points at once
        flat_points = points_3d.reshape(-1, 3)
        
        # Compute point magnitudes for scaling offsets
        with torch.no_grad():
            # Add small epsilon to avoid division by zero
            point_magnitudes = torch.norm(flat_points, dim=1, keepdim=True) + 1e-6
        
        # Predict offsets (scaled to [-1, 1] by tanh)
        predicted_offsets = self.offset_network(flat_points)
        
        # Scale offsets by point magnitude and max_offset to keep them proportional
        scaled_offsets = predicted_offsets * point_magnitudes * self.max_offset
        
        # Apply offsets to original points
        refined_points = flat_points + scaled_offsets
        
        # Reshape back to original dimensions
        return refined_points.reshape(*original_shape)


class RefinementLoss(nn.Module):
    """
    Loss function for training the refinement network.
    
    Includes:
    1. Reconstruction loss: How well the refined points match ground truth
    2. Identity loss: How well the refined points project to the original 2D points
    3. Temporal loss: How well the refined points maintain temporal consistency
    """
    
    def __init__(self, lambda_identity=1.0, lambda_temporal=0.5, lambda_reconstruction=1.0):
        """
        Initialize the refinement loss.
        
        Args:
            lambda_identity: Weight for identity loss
            lambda_temporal: Weight for temporal consistency loss
            lambda_reconstruction: Weight for reconstruction loss
        """
        super().__init__()
        self.lambda_identity = lambda_identity
        self.lambda_temporal = lambda_temporal
        self.lambda_reconstruction = lambda_reconstruction
        
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
                        [points_3d, torch.ones(min_points, 1, device=points_3d.device)], 
                        dim=1
                    )
                    
                    # Project to 2D
                    points_2d_homo = torch.matmul(points_3d_homo, P.t())
                    points_2d = points_2d_homo[:, :2] / (points_2d_homo[:, 2:3] + 1e-10)
                    
                    # Get actual 2D points for this view (only up to min_points)
                    gt_points_2d = view_tracks_2d[batch_idx, frame_idx, :min_points]
                    
                    # Compute distance
                    dist = F.mse_loss(points_2d, gt_points_2d, reduction='none')
                    
                    # Normalize by image dimensions (224x224)
                    dist = dist / torch.tensor([224.0, 224.0], device=dist.device)
                    
                    # Debug output to show both normalized and pixel distances
                    with torch.no_grad():
                        pixel_dist = F.mse_loss(points_2d, gt_points_2d, reduction='mean')
                        if frame_idx % 20 == 0 and batch_idx == 0 and view_idx == 0:
                            print(f"View {view_idx}, Frame {frame_idx}: ")
                            print(f"  Normalized distance: {torch.mean(dist).item():.6f}")
                            print(f"  Pixel distance: {pixel_dist.item():.6f}")
                            print(f"  Using {min_points} points (min of {num_points} predicted and {view_num_points} in view)")
                    
                    identity_loss += torch.mean(dist)
                    total_projections += 1
        
        # Average across all projections
        if total_projections > 0:
            identity_loss = identity_loss / total_projections
        
        return identity_loss
        
    def compute_temporal_loss(self, pred_points):
        """
        Compute temporal consistency loss by measuring how smoothly points move over time.
        
        Args:
            pred_points: Predicted 3D points of shape (batch_size, num_frames, num_points, 3)
            
        Returns:
            Normalized temporal consistency loss
        """
        # Skip if we only have one frame
        if pred_points.shape[1] <= 1:
            return torch.tensor(0.0, device=pred_points.device)
        
        # Compute velocity (difference between consecutive frames)
        velocity = pred_points[:, 1:] - pred_points[:, :-1]  # (batch_size, num_frames-1, num_points, 3)
        
        # Compute acceleration (difference in velocity)
        if velocity.shape[1] <= 1:
            # If we only have one velocity measurement, penalize its magnitude
            acceleration = velocity
        else:
            acceleration = velocity[:, 1:] - velocity[:, :-1]  # (batch_size, num_frames-2, num_points, 3)
        
        # Compute mean squared acceleration (smoothness penalty)
        temporal_loss = torch.mean(torch.sum(acceleration ** 2, dim=-1))
        
        # Normalize by the maximum range of predicted points
        with torch.no_grad():
            min_vals = torch.min(pred_points.reshape(-1, 3), dim=0)[0]
            max_vals = torch.max(pred_points.reshape(-1, 3), dim=0)[0]
            ranges = max_vals - min_vals
            scale_factor = torch.max(ranges) + 1e-6
        
        # Apply normalization
        normalized_loss = temporal_loss / (scale_factor ** 2)
        
        return normalized_loss
        
    def forward(self, pred_points, gt_points, visibility, projection_matrices, tracks_2d):
        """
        Forward pass of the refinement loss.
        
        Args:
            pred_points: Predicted 3D points
            gt_points: Ground truth 3D points
            visibility: Visibility mask for ground truth
            projection_matrices: Camera projection matrices
            tracks_2d: Original 2D tracks
            
        Returns:
            Total loss and individual loss components
        """
        # Compute reconstruction loss
        recon_loss = self.compute_reconstruction_loss(pred_points, gt_points, visibility)
        
        # Compute identity loss
        identity_loss = self.compute_identity_loss(pred_points, projection_matrices, tracks_2d)
        
        # Compute temporal consistency loss
        temporal_loss = self.compute_temporal_loss(pred_points)
        
        # Combine losses with weights
        total_loss = (self.lambda_reconstruction * recon_loss + 
                      self.lambda_identity * identity_loss + 
                      self.lambda_temporal * temporal_loss)
        
        return {
            'loss': total_loss,
            'reconstruction_loss': recon_loss,
            'identity_loss': identity_loss,
            'temporal_loss': temporal_loss
        }


class RefinementTrainer:
    """Trainer for the Refinement Network using DLT triangulation as initialization.
    
    This trainer uses DLT triangulation as the initialization for 3D points and then
    trains a refinement network to improve these points.
    """
    
    def __init__(self, 
                 data_dir: str, 
                 calibration_file: str,
                 target_size=(224, 224),
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 attention_checkpoint=None,
                 lambda_reconstruction=1.0,
                 lambda_temporal=0.5,
                 lambda_identity=1.0):
        """
        Initialize the refinement trainer.
        
        Args:
            data_dir: Directory containing TAP3D data
            calibration_file: Path to the calibration file
            target_size: Target size for image resizing
            device: Device to run on
            attention_checkpoint: Path to pre-trained attention model checkpoint (optional)
            lambda_reconstruction: Weight for reconstruction loss
            lambda_temporal: Weight for temporal consistency loss
            lambda_identity: Weight for identity loss
        """
        self.data_dir = data_dir
        self.calibration_file = calibration_file
        self.target_size = target_size
        self.device = device
        self.use_attention = attention_checkpoint is not None
        
        # Initialize data loader
        self.data_loader = TAP3DLoader(data_dir, calibration_file)
        
        # Initialize pipeline (we'll use it for data preparation)
        self.pipeline = LAPAPipeline(
            data_dir=data_dir,
            calibration_file=calibration_file,
            target_size=target_size,
            device=device
        )
        
        # Initialize triangulation module for DLT
        self.triangulation = TriangulationModule().to(device)
        
        # Initialize refinement network
        self.refinement_network = RefinementNetwork(
            hidden_dim=512,
            dropout=0.2,
            max_offset=0.3
        ).to(device)
        
        # Initialize geometric attention model if checkpoint is provided
        if self.use_attention:
            print(f"Loading pre-trained attention model from {attention_checkpoint}")
            
            # Load checkpoint to determine geometry type
            checkpoint = torch.load(attention_checkpoint, map_location=device)
            geometry_type = checkpoint.get('geometry_type', 'epipolar')  # Default to epipolar if not specified
            grid_size = checkpoint.get('grid_size', 16)  # Default to 16 if not specified
            
            print(f"Using {geometry_type} geometry with grid size {grid_size}")
            
            # Initialize the appropriate track correspondence module based on geometry type
            if geometry_type == 'sfm':
                self.track_correspondence = TrackCorrespondenceSfM(
                    feature_dim=128,
                    volume_size=grid_size,
                    num_heads=4
                ).to(device)
            else:  # Default to epipolar
                self.track_correspondence = TrackCorrespondence(
                    feature_dim=128,
                    volume_size=grid_size,
                    num_heads=4
                ).to(device)
            
            # Load pre-trained weights
            self.track_correspondence.load_state_dict(checkpoint['attention_model'])
            
            # Freeze the attention model parameters
            for param in self.track_correspondence.parameters():
                param.requires_grad = False
                
            print("Attention model loaded and frozen for inference")
            
            # Replace pipeline's track_correspondence with our pre-trained one
            self.pipeline.track_correspondence = self.track_correspondence
        
        # Initialize loss function with customizable weights
        self.loss_fn = RefinementLoss(
            lambda_identity=lambda_identity,
            lambda_temporal=lambda_temporal,
            lambda_reconstruction=lambda_reconstruction
        )
        
        # Initialize optimizer with a moderate learning rate
        self.optimizer = torch.optim.Adam(self.refinement_network.parameters(), lr=0.001)
        
        # Add learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.7, patience=3, verbose=True)
        
        # Initialize visualizer for debugging
        self.visualizer = LAPAVisualizer(output_dir="outputs/refinement_training")
        os.makedirs("outputs/refinement_training", exist_ok=True)
    
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
        # Set model to training mode
        self.refinement_network.train()
        
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
        temporal_loss_sum = 0.0
        metrics = {}
        
        # Ensure we don't exceed the available frames
        max_frames = min(num_frames, gt_tracks_3d.shape[1])
        
        # Process each frame
        for frame_idx in tqdm(range(max_frames), desc="Training"):
            # Get 2D tracks for this frame
            frame_tracks_2d = [track[:, frame_idx:frame_idx+1, :, :] for track in tracks_2d]
            
            # Find correspondences for this frame
            correspondence_result = self.pipeline.track_correspondence(frame_tracks_2d, projection_matrices)
            
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
            
            # Compute loss
            loss_dict = self.loss_fn(
                refined_points,
                gt_tracks_3d[:, frame_idx:frame_idx+1],
                gt_visibility[:, frame_idx:frame_idx+1],
                projection_matrices,
                frame_tracks_2d
            )
            
            # Backpropagation
            self.optimizer.zero_grad()
            loss_dict['loss'].backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss_dict['loss'].item()
            recon_loss_sum += loss_dict['reconstruction_loss'].item()
            identity_loss_sum += loss_dict['identity_loss'].item()
            temporal_loss_sum += loss_dict['temporal_loss'].item()
            
            # Log every 10 frames
            if frame_idx % 10 == 0:
                print(f"Frame {frame_idx}: ")
                print(f"  Total Loss: {loss_dict['loss'].item():.6f}")
                print(f"  Reconstruction Loss: {loss_dict['reconstruction_loss'].item():.6f}")
                print(f"  Identity Loss: {loss_dict['identity_loss'].item():.6f}")
                print(f"  Temporal Loss: {loss_dict['temporal_loss'].item():.6f}")
                
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
        
        # Update learning rate based on average loss
        avg_loss = total_loss / max_frames
        self.scheduler.step(avg_loss)
        
        # Return average metrics
        return {
            'total_loss': avg_loss,
            'reconstruction_loss': recon_loss_sum / max_frames,
            'identity_loss': identity_loss_sum / max_frames,
            'temporal_loss': temporal_loss_sum / max_frames
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
        plt.savefig(f"outputs/refinement_training/comparison_frame_{frame_idx}.png")
        plt.close(fig)


def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Train refinement network")
    
    # Data parameters
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing TAP3D data")
    parser.add_argument("--calibration_file", type=str, required=True,
                        help="Path to calibration file")
    parser.add_argument("--view_set", type=str, default="boxes",
                        help="View set to use for training")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of epochs to train for")
    parser.add_argument("--num_frames", type=int, default=60,
                        help="Number of frames to process per epoch")
    parser.add_argument("--chunk_size", type=int, default=8,
                        help="Chunk size for processing frames")
    
    # Model parameters
    parser.add_argument("--attention_checkpoint", type=str, default=None,
                        help="Path to pre-trained attention model checkpoint (optional)")
    
    # Loss weights
    parser.add_argument("--lambda_reconstruction", type=float, default=1.0,
                        help="Weight for reconstruction loss")
    parser.add_argument("--lambda_temporal", type=float, default=0.5,
                        help="Weight for temporal consistency loss")
    parser.add_argument("--lambda_identity", type=float, default=1.0,
                        help="Weight for identity loss")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="outputs/refinement_training",
                        help="Directory to save outputs")
    
    return parser.parse_args()


def main():
    """
    Main function for training the refinement network.
    """
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Add device to args for compatibility
    args.device = device
    
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
    elif args.view_set == 'poster':
        view_names = ['poster_1', 'poster_2', 'poster_3', 'poster_4']
    elif args.view_set == 'custom' and args.custom_views:
        view_names = args.custom_views
    else:
        raise ValueError(f"Unknown view set: {args.view_set}")
    
    print(f"Training on views: {view_names}")
    print(f"Using device: {args.device}")
    
    # Initialize trainer with custom loss weights
    trainer = RefinementTrainer(
        data_dir=args.data_dir,
        calibration_file=args.calibration_file,
        target_size=(224, 224),
        device=args.device,
        attention_checkpoint=args.attention_checkpoint,
        lambda_reconstruction=args.lambda_reconstruction,
        lambda_temporal=args.lambda_temporal,
        lambda_identity=args.lambda_identity
    )
    
    if args.attention_checkpoint:
        print(f"Using pre-trained attention model from {args.attention_checkpoint}")
    else:
        print("Not using any pre-trained attention model")
    
    # Training loop
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_results = trainer.train_epoch(view_names, num_frames=args.num_frames)
        
        print(f"Epoch {epoch+1} completed. Metrics:")
        print(f"  Total Loss: {train_results['total_loss']:.6f}")
        print(f"  Reconstruction Loss: {train_results['reconstruction_loss']:.6f}")
        print(f"  Identity Loss: {train_results['identity_loss']:.6f}")
        print(f"  Temporal Loss: {train_results['temporal_loss']:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            checkpoint_path = os.path.join(args.output_dir, f"refinement_model_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': trainer.refinement_network.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'loss': train_results['total_loss'],
                'lambda_reconstruction': args.lambda_reconstruction,
                'lambda_temporal': args.lambda_temporal,
                'lambda_identity': args.lambda_identity
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    main()