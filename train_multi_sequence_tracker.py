#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training script for the multi-sequence learnable point tracker using TAP3D dataset.
This script loads multiple sequences and treats them as different views of the same scene.
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcmpt.data_utils.tap3d_dataset import TAP3DVidDataset
from mcmpt.modules.learnable_point_tracker import LearnablePointTracker
from mcmpt.losses.track_losses import TrackingLoss


class MultiSequenceDataset(Dataset):
    """
    Dataset wrapper that combines multiple sequences from TAP3D dataset
    and treats them as different views of the same scene.
    """
    def __init__(self, tap3d_dataset):
        self.tap3d_dataset = tap3d_dataset
        self.sequence_length = tap3d_dataset.sequence_length
        self.stride = tap3d_dataset.stride
        
        # Calculate the number of frames we can use in each sequence
        self.num_sequences = len(tap3d_dataset.tap3d_data_list)
        self.frames_per_sequence = min(tap3d_dataset.num_valid_frames)
        
        # Calculate continuous sequence chunks
        # Each chunk will be sequence_length frames long
        # We'll move by sequence_length each time (no overlap)
        # This ensures temporal continuity
        max_chunks = (self.frames_per_sequence - 1) // self.sequence_length
        self.chunks = [(i * self.sequence_length, (i + 1) * self.sequence_length) 
                      for i in range(max_chunks)]
        self.num_samples = len(self.chunks)
        
        print(f"MultiSequenceDataset: {self.num_sequences} sequences, {self.frames_per_sequence} frames per sequence")
        print(f"Sequence length: {self.sequence_length} (continuous chunks)")
        print(f"Created {self.num_samples} continuous chunks")
        print(f"First few chunks: {self.chunks[:3]}...")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Get multiple sequences at the same frame index and combine them as different views.
        Each returned chunk is a continuous sequence of frames.
        """
        if idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of bounds for dataset with {self.num_samples} samples")
        
        # Get the frame range for this chunk
        start_frame, end_frame = self.chunks[idx]
        
        # Get the corresponding sequence chunk from each view
        sequence_samples = []
        
        for seq_idx in range(self.num_sequences):
            # Get the sequence from this view
            sequence_data = self.tap3d_dataset.tap3d_data_list[seq_idx]
            
            # Extract the continuous chunk of frames
            sample = self.tap3d_dataset.get_sequence_at_index(
                sequence_data=sequence_data, 
                sequence_idx=seq_idx,
                start_frame=start_frame,
                end_frame=end_frame  # Ensure we get the full chunk
            )
            
            sequence_samples.append(sample)
        
        # Combine the samples into a multi-view sample
        combined_sample = self._combine_samples(sequence_samples)
        return combined_sample
    
    def _combine_samples(self, samples):
        """
        Combine multiple sequence samples into a single multi-view sample.
        
        Args:
            samples: List of samples from the TAP3D dataset
            
        Returns:
            Combined sample with multiple views
        """
        # Debug information about track queries
        print(f"\n==== TRACK QUERY DEBUG (MULTI SEQUENCE DATASET) ====\nProcessing {len(samples)} samples to combine track queries")
        for i, sample in enumerate(samples):
            print(f"Sample {i} keys: {list(sample.keys())}")
            if 'track_queries' in sample:
                print(f"View {i} has track_queries with {len(sample['track_queries'])} queries")
                if 'query_frame_map' in sample:
                    print(f"View {i} has query_frame_map with {len(sample['query_frame_map'])} frame entries")
            else:
                print(f"View {i} has NO track_queries")
        
        # Extract frames from each sample
        frames = [sample['frames'] for sample in samples]  # List of [T, 1, C, H, W]
        
        # Combine frames along the view dimension
        # Each sample has shape [T, 1, C, H, W], we want [T, V, C, H, W]
        combined_frames = torch.cat(frames, dim=1)  # [T, V, C, H, W]
        
        # Combine camera matrices
        combined_cam_matrices = []
        for sample in samples:
            combined_cam_matrices.extend(sample['cam_matrices'])
        
        # Combine camera names
        combined_camera_names = []
        for sample in samples:
            combined_camera_names.extend(sample['camera_names'])
        
        # Combine keypoints_2d
        combined_keypoints_2d = []
        for sample in samples:
            combined_keypoints_2d.extend(sample['keypoints_2d'])
        
        # Use the first sample's sequence indices and tracks_xyz
        seq_indices = samples[0]['seq_indices']
        tracks_xyz = samples[0]['tracks_xyz']
        
        # Create the combined sample
        combined_sample = {
            'seq_indices': seq_indices,
            'frames': combined_frames,
            'keypoints_2d': combined_keypoints_2d,
            'camera_names': combined_camera_names,
            'cam_matrices': combined_cam_matrices,
            'tracks_xyz': tracks_xyz
        }
        
        # Combine track queries if available
        combined_track_queries = []
        combined_query_frame_maps = []
        
        for view_idx, sample in enumerate(samples):
            if 'track_queries' in sample:
                # Add view index to help identify which view this query belongs to
                sample_queries = {'view_idx': view_idx, 'queries': sample['track_queries']}
                combined_track_queries.append(sample_queries)
                
                # Add query frame map if available
                if 'query_frame_map' in sample:
                    query_map = {'view_idx': view_idx, 'map': sample['query_frame_map']}
                    combined_query_frame_maps.append(query_map)
        
        if combined_track_queries:
            combined_sample['track_queries'] = combined_track_queries
            print(f"Added {len(combined_track_queries)} track query sets to combined sample")
        else:
            print(f"No track queries found in any sample")
        
        if combined_query_frame_maps:
            combined_sample['query_frame_maps'] = combined_query_frame_maps
            print(f"Added {len(combined_query_frame_maps)} query frame maps to combined sample")
        else:
            print(f"No query frame maps found in any sample")
        
        print(f"==== END TRACK QUERY DEBUG (MULTI SEQUENCE DATASET) ====\n")
        return combined_sample


def parse_args():
    parser = argparse.ArgumentParser(description='Train a multi-sequence learnable point tracker')
    parser.add_argument('--data_dir', type=str, default='./data/tap3d_boxes',
                        help='Path to TAP3D dataset')
    parser.add_argument('--output_dir', type=str, default='./outputs/multi_sequence_tracker',
                        help='Output directory for models and visualizations')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num_frames', type=int, default=5,
                        help='Number of frames per sequence')
    parser.add_argument('--max_points', type=int, default=100,
                        help='Maximum number of points to track')
    parser.add_argument('--grid_size', type=int, default=10,
                        help='Grid size for initial point sampling')
    parser.add_argument('--online_mode', action='store_true',
                        help='Use online mode for CoTracker')
    parser.add_argument('--visualize_every', type=int, default=5,
                        help='Visualize every N steps')
    parser.add_argument('--vis_skip_frames', type=int, default=5,
                        help='Number of frames to skip between visualized frames')
    parser.add_argument('--sequence_length', type=int, default=5,
                        help='Number of frames in sequence')
    parser.add_argument('--eval_only', action='store_true',
                        help='Run evaluation only (no training)')
    parser.add_argument('--recon_weight', type=float, default=1.0,
                        help='Weight for reconstruction loss')
    parser.add_argument('--temporal_weight', type=float, default=0.5,
                        help='Weight for temporal consistency loss')
    parser.add_argument('--buffer_size', type=int, default=50,
                        help='Buffer size for tracking (number of frames to keep in buffer). Should be at least sequence_length.')
    return parser.parse_args()


def create_timestamp():
    """Create a timestamp string for naming directories."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def setup_output_dirs(args):
    """Setup output directories for models and visualizations."""
    timestamp = create_timestamp()
    model_dir = os.path.join(args.output_dir, f'model_{timestamp}')
    vis_dir = os.path.join(args.output_dir, f'vis_{timestamp}')
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    # Create subdirectories for different visualization types
    os.makedirs(os.path.join(vis_dir, 'tracks'), exist_ok=True)
    os.makedirs(os.path.join(vis_dir, 'tracks_all_views'), exist_ok=True)
    os.makedirs(os.path.join(vis_dir, 'correspondences'), exist_ok=True)
    
    return model_dir, vis_dir


def prepare_data(args):
    """Prepare the TAP3D dataset and dataloader."""
    # Create config dictionary for TAP3DVidDataset
    # Use image size that's a multiple of 14 (patch size for DINO feature extraction)
    # 224 x 224 = 16 * 14 x 16 * 14 (standard size for many vision models)
    # Create dataset config with sequence_length matching buffer_size
    # This ensures we have enough frames for tracking
    dataset_config = {
        'dataset': {
            'tap3d_data_path': args.data_dir,
            'sequence_length': args.buffer_size,  # Use buffer_size as sequence length
            'image_size': (224, 224),
            'tap3d_sequences': 'boxes_5.npz,boxes_6.npz,boxes_7.npz',
            'start_frame': 0,
            'num_frames': 1000,
            'stride': 1
        }
    }
    print(f"Using sequence length of {args.buffer_size} frames (matching buffer size)")
    
    # Set up calibration file path
    calibration_file = os.path.join(args.data_dir, "calibration_161029_sports1.json")
    print(f"Using calibration file: {calibration_file}")
    
    # Check if calibration file exists
    if not os.path.exists(calibration_file):
        print(f"Warning: Calibration file not found: {calibration_file}")
        calibration_file = None
    
    # Create the TAP3D dataset
    tap3d_dataset = TAP3DVidDataset(dataset_config, split='train', calibration_file=calibration_file)
    
    # Create the multi-sequence dataset wrapper
    multi_seq_dataset = MultiSequenceDataset(tap3d_dataset)
    
    # Create the dataloader
    dataloader = DataLoader(
        multi_seq_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"Loaded TAP3D dataset with {len(multi_seq_dataset)} multi-view sequences")
    
    return dataloader


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, args, vis_dir):
    """Train the model for one epoch."""
    # Set model to train mode if we have an optimizer, otherwise eval mode
    model.train() if optimizer is not None else model.eval()
    total_loss = 0.0
    
    # Metrics to track
    metrics = {
        'total_loss': 0.0,
        'recon_loss': 0.0,
        'temporal_loss': 0.0,
        'identity_loss': 0.0  # Add identity loss metric
    }
    
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    progress_bar.set_description(f"Epoch {epoch+1}/{args.num_epochs}")
    
    # Log dataset size information
    print(f"Processing {len(dataloader)} batches in dataloader")
    
    # IMPORTANT: Reset state only ONCE at the beginning of an epoch
    model.reset_internal_state()
    
    iteration = 0
    for batch_idx, batch in enumerate(dataloader):
        # Log what batch we're processing
        print(f"\n==== Processing batch {batch_idx}/{len(dataloader)-1} ====")
        
        # Extract batch data
        frames = batch['frames'].to(device)
        cam_matrices = [m.to(device) for m in batch['cam_matrices']]
        gt_tracks_xyz = batch['tracks_xyz'].to(device) if 'tracks_xyz' in batch else None
        
        # Extract track queries if available
        track_queries = batch['track_queries'] if 'track_queries' in batch else None
        query_frame_maps = batch['query_frame_maps'] if 'query_frame_maps' in batch else None
        
        # Print frame shape for debugging
        if batch_idx == 0 or batch_idx % 10 == 0:
            print(f"Frame shape: {frames.shape} (B,T,V,C,H,W)")
            print(f"Number of views: {frames.shape[2]}")
            if gt_tracks_xyz is not None:
                print(f"Ground truth tracks shape: {gt_tracks_xyz.shape} (B,T,N,3)")
            if track_queries is not None:
                print(f"Track queries available: {len(track_queries)} sets")
            if query_frame_maps is not None:
                print(f"Query frame maps available: {len(query_frame_maps)} sets")
        
            # Debug print to check track_queries and query_frame_maps structure
        if track_queries is not None:
            print(f"Passing {len(track_queries)} track query sets to model")
            for i, query_set in enumerate(track_queries):
                print(f"  Query set {i}: view_idx={query_set['view_idx']}, {len(query_set['queries'])} queries")
        
        if query_frame_maps is not None:
            print(f"Passing {len(query_frame_maps)} query frame maps to model")
            for i, frame_map in enumerate(query_frame_maps):
                print(f"  Frame map {i}: view_idx={frame_map['view_idx']}, {len(frame_map['map'])} frame entries")
        
        outputs = model(
            frames, 
            cam_matrices, 
            batch_idx=batch_idx,
            track_queries=track_queries,
            query_frame_maps=query_frame_maps  # Note: fixed parameter name to match what the model expects
        )
        
        # Calculate loss using our new loss function
        if criterion is not None and gt_tracks_xyz is not None:
            # Pass track query maps to the loss function if available
            loss, loss_dict = criterion(
                model.current_batch_tracks,  # Dict of batch -> view -> tracks
                gt_tracks_xyz,               # Ground truth 3D tracks (B,T,M,3)
                batch_idx,                  # Current batch index
                model.track_query_maps if hasattr(model, 'track_query_maps') else None  # Track query maps
            )
            
            # Update metrics
            for k, v in loss_dict.items():
                metrics[k] += v
            
            # Print loss components
            print(f"Loss components: {loss_dict}")
        else:
            # Fallback to dummy loss if no criterion or ground truth
            loss = torch.tensor(0.0, requires_grad=True, device=device)
            print("Warning: Using dummy loss (criterion or ground truth not available)")
        
        # For visualization purposes
        if batch_idx % args.visualize_every == 0:
            # Visualize all views
            model.visualize_all_views(batch_idx=batch_idx, output_dir=os.path.join(vis_dir, 'tracks_all_views'), skip_frames=args.vis_skip_frames)
            
            # Visualize each view separately
            for view_id in range(frames.shape[2]):
                model.visualize_tracks(batch_idx=batch_idx, view_id=view_id, output_dir=os.path.join(vis_dir, 'tracks'), skip_frames=args.vis_skip_frames)
        
        # Backward pass and optimize only if we have an optimizer
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Update progress bar
        total_loss += loss.item()
        progress_bar.set_postfix(loss=total_loss/(batch_idx+1))
        progress_bar.update(1)
        
        iteration += 1
        
        # Removed iteration-level checkpointing to save only at epoch boundaries
        
        # In eval mode, only process a few batches to show visualizations
        if args.eval_only and batch_idx >= 5:
            print(f"Evaluation mode: stopping after {batch_idx+1} batches to show visualization results")
            break
        
        # Print batch boundaries to help with debugging
        print(f"==== Completed batch {batch_idx}/{len(dataloader)-1} ====")
    
    # Calculate average metrics
    batch_count = batch_idx + 1
    for k in metrics:
        metrics[k] /= batch_count
    
    print(f"Epoch {epoch+1} metrics: {metrics}")
    
    # Print trajectory summary at the end of epoch
    model.print_trajectory_summary()
    
    return total_loss / batch_count


def validate(model, dataloader, criterion, device, epoch, args, vis_dir):
    """Validate the model on the validation set."""
    model.eval()
    total_loss = 0.0
    
    # Metrics to track
    metrics = {
        'total_loss': 0.0,
        'recon_loss': 0.0,
        'temporal_loss': 0.0,
        'identity_loss': 0.0  # Include identity loss metric
    }
    
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    progress_bar.set_description(f"Validation Epoch {epoch+1}")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Extract batch data
            frames = batch['frames'].to(device)
            cam_matrices = [m.to(device) for m in batch['cam_matrices']]
            gt_tracks_xyz = batch['tracks_xyz'].to(device) if 'tracks_xyz' in batch else None
            
            # Extract track queries if available
            track_queries = batch['track_queries'] if 'track_queries' in batch else None
            query_frame_maps = batch['query_frame_maps'] if 'query_frame_maps' in batch else None
            
            # Reset tracking state for each batch
            model.reset_internal_state()
            
            # Forward pass with track queries
            outputs = model(
                frames, 
                cam_matrices, 
                batch_idx=batch_idx,
                reset_tracking=True,
                track_queries=track_queries,
                query_frame_maps=query_frame_maps
            )
            
            # Calculate loss using our loss function if available
            if criterion is not None and gt_tracks_xyz is not None:
                # Pass track query maps to the loss function if available
                loss, loss_dict = criterion(
                    model.current_batch_tracks,  # Dict of batch -> view -> tracks
                    gt_tracks_xyz,               # Ground truth 3D tracks (B,T,M,3)
                    batch_idx,                  # Current batch index
                    model.track_query_maps if hasattr(model, 'track_query_maps') else None  # Track query maps
                )
                
                # Update metrics
                for k, v in loss_dict.items():
                    metrics[k] += v
                    
                # Print loss components
                if batch_idx % 10 == 0:  # Print less frequently during validation
                    print(f"Validation loss components: {loss_dict}")
            else:
                # Fallback to dummy loss if no criterion or ground truth
                loss = torch.tensor(0.0, device=device)
            
            # For visualization purposes
            if batch_idx % args.visualize_every == 0:
                print(f"\nProcessing batch {batch_idx} for visualization")
                # Print frame shape for debugging
                print(f"Frame shape: {frames.shape}")
                print(f"Number of views: {frames.shape[2]}")
                
                # Visualize all views
                model.visualize_all_views(batch_idx=batch_idx, output_dir=os.path.join(vis_dir, 'tracks_all_views'), skip_frames=args.vis_skip_frames)
                
                # Visualize each view separately
                for view_id in range(frames.shape[2]):
                    model.visualize_tracks(batch_idx=batch_idx, view_id=view_id, output_dir=os.path.join(vis_dir, 'tracks'), skip_frames=args.vis_skip_frames)
            
            # Update progress bar
            total_loss += loss.item()
            progress_bar.set_postfix(loss=total_loss/(batch_idx+1))
            
            # In eval mode, only process a few batches to show visualizations
            if args.eval_only and batch_idx >= 5:
                print(f"\nEvaluation mode: stopping after {batch_idx+1} batches to show visualization results")
                break
    
    print(f"Validation Loss: {total_loss/len(dataloader):.4f}")
    return total_loss / len(dataloader)


def main():
    """Main function to run the training script."""
    # Parse command line arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directories
    model_dir, vis_dir = setup_output_dirs(args)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Prepare data
    dataloader = prepare_data(args)
    
    # Create model
    model_config = {
        'max_points': args.max_points,
        'grid_size': args.grid_size,
        'online_mode': args.online_mode,
        'visualization_enabled': True,
        'debug_mode': True,
        'image_size': 224,  # Fixed to DINO image size
        'buffer_size': args.buffer_size  # Pass buffer_size from command line args
    }
    model = LearnablePointTracker(model_config).to(device)
    print(f"Model configured with {args.max_points} points and grid size {args.grid_size}")
    
    # Create our custom loss function
    criterion = TrackingLoss(
        reconstruction_weight=1.0,
        temporal_weight=0.5,
        confidence_threshold=0.5
    ).to(device)
    print("Using custom TrackingLoss with reconstruction and temporal components")
    
    # By default, we're running in eval mode to visualize results
    # Only create optimizer if not in eval-only mode
    optimizer = None
    if not args.eval_only:
        # Get trainable parameters
        params = [p for p in model.parameters() if p.requires_grad]
        
        if len(params) > 0:
            optimizer = optim.Adam(params, lr=args.learning_rate)
            print(f"Optimizer created with {len(params)} trainable parameters")
            print(f"Using Adam optimizer with learning rate {args.learning_rate}")
            
            # Double-check that optimizer has parameters
            if len(optimizer.param_groups[0]['params']) > 0:
                print(f"Optimizer successfully initialized with {len(optimizer.param_groups[0]['params'])} parameter groups")
            else:
                print("WARNING: Optimizer has no parameters!")
                
            # Verify TrackCorrespondence parameters are included
            if hasattr(model, 'track_correspondence'):
                tc_params = sum(p.numel() for p in model.track_correspondence.parameters() if p.requires_grad)
                print(f"TrackCorrespondence module has {tc_params} trainable parameters in optimizer")
        else:
            print("No trainable parameters found, running in eval mode")
    else:
        print("Running in eval-only mode, no optimizer created")
    
    # Train or evaluate the model
    if not args.eval_only:
        print("Starting training...")
        for epoch in range(args.num_epochs):
            train_loss = train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, args, vis_dir)
            
            # Save model checkpoint with timestamp in model_dir
            checkpoint_path = os.path.join(model_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
                'train_loss': train_loss
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    else:
        # Just run one epoch in evaluation mode
        print("Running evaluation...")
        train_one_epoch(model, dataloader, criterion, None, device, 0, args, vis_dir)
    
    print("Training/evaluation completed!")


if __name__ == "__main__":
    main()
