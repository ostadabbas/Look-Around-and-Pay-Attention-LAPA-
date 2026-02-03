#!/usr/bin/env python3
"""
Visualize the trained attention with the fixed geometric attention mechanism.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import argparse

from lapa.models.geometric_attention_fixed import TrackCorrespondence
from lapa.lapa_pipeline import LAPAPipeline
from lapa.data.tap3d_loader import TAP3DLoader

def visualize_attention(checkpoint_path, data_dir, calibration_file, view_names, output_dir):
    """
    Visualize attention weights between views.
    
    Args:
        checkpoint_path: Path to the trained model checkpoint
        data_dir: Directory with TAP3D data
        calibration_file: Path to calibration file
        view_names: List of view names to visualize
        output_dir: Directory to save visualizations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize pipeline for data loading
    pipeline = LAPAPipeline(
        data_dir=data_dir,
        calibration_file=calibration_file,
        device=device
    )
    
    # Initialize attention model
    track_correspondence = TrackCorrespondence(
        feature_dim=128,
        volume_size=32,
        num_heads=4
    ).to(device)
    
    # Load trained weights
    print(f"Loading model from checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Try different keys that might contain the model
    if 'track_correspondence' in checkpoint:
        track_correspondence.load_state_dict(checkpoint['track_correspondence'])
        print("Loaded from 'track_correspondence' key")
    elif 'attention_model' in checkpoint:
        track_correspondence.load_state_dict(checkpoint['attention_model'])
        print("Loaded from 'attention_model' key")
    elif 'model_state_dict' in checkpoint:
        # Check if this is the combined model
        try:
            track_correspondence.load_state_dict({
                k.replace('track_correspondence.', ''): v 
                for k, v in checkpoint['model_state_dict'].items() 
                if k.startswith('track_correspondence.')
            })
            print("Loaded from 'model_state_dict' using prefix 'track_correspondence.'")
        except Exception as e:
            print(f"Failed to load using prefix: {e}")
            # Try without prefix
            try:
                track_correspondence.load_state_dict(checkpoint['model_state_dict'])
                print("Loaded from 'model_state_dict' directly")
            except Exception as e:
                print(f"Failed to load model: {e}")
                return
    else:
        print(f"No recognizable model found in checkpoint: {checkpoint_path}")
        print(f"Available keys: {list(checkpoint.keys())}")
        return
    
    # Set model to evaluation mode
    track_correspondence.eval()
    
    # Replace the pipeline's track correspondence with our loaded model
    pipeline.track_correspondence = track_correspondence
    
    # Load data
    print(f"Loading data for views: {view_names}")
    data = pipeline.data_loader.load_multi_view_data(view_names)
    
    # Prepare inputs with correct intrinsics and projection
    tracks_2d = []
    projection_matrices = []
    
    for view_name in view_names:
        # Access the data for this view from the 'views' dictionary
        view_data = data['views'][view_name]
        print(f"View {view_name}: Using original intrinsics: {view_data['intrinsics']}")
        
        # Create points from tracks
        track = view_data['tracks_xyz'][:1]  # Use only first frame for visualization
        visibility = view_data['visibility'][:1]
        
        # Convert 3D points to 2D
        points_2d = track[..., :2]  # Use only x, y coordinates
        
        # Add batch dimension if needed
        if len(points_2d.shape) == 3:  # [frames, points, 2]
            points_2d = torch.tensor(points_2d).unsqueeze(0).float().to(device)
        else:  # [batch, frames, points, 2]
            points_2d = torch.tensor(points_2d).float().to(device)
            
        tracks_2d.append(points_2d)
        
        # Get camera matrix from calibration data
        base_view_name = view_name.replace('.npz', '')
        if base_view_name in data['camera_matrices']:
            camera_matrix = data['camera_matrices'][base_view_name]
            print(f"View {view_name}: Loaded camera extrinsics matrix")
        else:
            print(f"Warning: No camera matrix found for view {base_view_name}")
            continue
        
        # Get camera intrinsics
        intrinsics = view_data['intrinsics']
        
        # Create projection matrix from camera matrix and intrinsics
        # Extract rotation and translation from camera matrix
        R = camera_matrix[:3, :3]  # 3x3 rotation matrix
        t = camera_matrix[:3, 3]   # 3x1 translation vector
        
        # Create camera extrinsics matrix [R|t]
        ext_matrix = np.eye(3, 4)
        ext_matrix[:3, :3] = R
        ext_matrix[:3, 3] = t
        
        # Create projection matrix using intrinsics and extrinsics
        fx, fy, cx, cy = intrinsics
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        
        # Compute projection matrix as P = K[R|t]
        proj_mat = K @ ext_matrix
        
        print(f"View {view_name}: Created projection matrix with intrinsics and extrinsics")
        
        # Add batch dimension and convert to tensor
        proj_mat = torch.tensor(proj_mat).unsqueeze(0).float().to(device)
        projection_matrices.append(proj_mat)
    
    # Run model to get attention
    print("Computing attention...")
    with torch.no_grad():
        # Use frame 0 for visualization
        frame_idx = 0
        frame_tracks_2d = [track[:, frame_idx:frame_idx+1, :, :].squeeze(1) for track in tracks_2d]
        
        # Get correspondence matrices and volume features
        correspondence_matrices, volume_features = track_correspondence.geometric_attention(
            frame_tracks_2d, projection_matrices
        )
    
    # Visualize attention weights between each pair of views
    for i, view_i in enumerate(view_names):
        for j, view_j in enumerate(view_names):
            if i == j:
                continue  # Skip self-attention
                
            corr_idx = j if j < i else j - 1  # Adjust index for skipped self-attention
            
            # Check if we have correspondence data for this pair
            if i >= len(correspondence_matrices) or corr_idx >= len(correspondence_matrices[i]):
                print(f"Warning: No correspondence data for views {view_i} to {view_j}")
                continue
                
            attention_weights = correspondence_matrices[i][corr_idx]
            
            # Skip if attention weights are None (self-attention)
            if attention_weights is None:
                print(f"Warning: Attention weights are None for views {view_i} to {view_j}")
                continue
            
            # Create figure for attention weights visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
            
            # Convert to numpy for visualization
            attn_numpy = attention_weights.detach().cpu().numpy()[0]
            
            # Attention weights as heatmap
            im = ax1.imshow(
                attn_numpy,
                cmap='viridis',
                aspect='auto'
            )
            ax1.set_title(f'Attention Weights: View {view_i} to {view_j}')
            ax1.set_xlabel(f'Points in View {view_j}')
            ax1.set_ylabel(f'Points in View {view_i}')
            plt.colorbar(im, ax=ax1)
            
            # Plot attention weights distribution
            attention_values = attn_numpy.flatten()
            ax2.hist(
                attention_values,
                bins=100,
                color='teal',
                alpha=0.7
            )
            
            # Add statistics lines
            max_val = np.max(attention_values)
            mean_val = np.mean(attention_values)
            median_val = np.median(attention_values)
            
            ax2.axvline(x=max_val, color='r', linestyle='--', label=f'Max: {max_val:.4f}')
            ax2.axvline(x=mean_val, color='g', linestyle='--', label=f'Mean: {mean_val:.4f}')
            ax2.axvline(x=median_val, color='b', linestyle='--', label=f'Median: {median_val:.4f}')
            
            ax2.set_title('Attention Weights Distribution')
            ax2.set_xlabel('Attention Weight Value')
            ax2.set_ylabel('Frequency')
            ax2.legend()
            ax2.grid(alpha=0.3)
            
            # Save the figure
            save_path = os.path.join(output_dir, f'fixed_attention_{view_i}_to_{view_j}.png')
            plt.savefig(save_path)
            print(f"Saved attention visualization to {save_path}")
            plt.close(fig)
    
    # Visualize 3D attention volume with multiple thresholds
    visualize_3d_attention_volume(
        volume_features,
        track_correspondence.geometric_attention.create_3d_grid(1, device),
        output_dir
    )

def visualize_3d_attention_volume(volume_features, grid, output_dir):
    """
    Visualize the 3D attention volume with multiple thresholds.
    
    Args:
        volume_features: Features in the volumetric grid
        grid: 3D grid for reference
        output_dir: Directory to save the visualization
    """
    feature_mag = torch.norm(volume_features, dim=-1).detach().cpu().numpy()
    
    # Create a single figure for all plots
    fig = plt.figure(figsize=(24, 16), facecolor='none')
    
    # Calculate thresholds
    max_val = np.max(feature_mag)
    min_val = np.min(feature_mag)
    mean_val = np.mean(feature_mag)
    print(f"Volume feature statistics: min={min_val:.6f}, max={max_val:.6f}, mean={mean_val:.6f}")
    
    # Create logarithmically spaced thresholds
    num_thresholds = 6
    if max_val > 0:
        base_threshold = max_val * 0.001
        thresholds = [base_threshold * (2**i) for i in range(num_thresholds)]
    else:
        thresholds = [0.001 * (10**i) for i in range(num_thresholds)]
    
    # For very small values, use linear thresholds
    if max_val < 0.01:
        thresholds = [max_val * i/num_thresholds for i in range(1, num_thresholds+1)]
        thresholds = [t if t > 0 else 0.0001 for t in thresholds]
    
    # Get grid coordinates and downsample for visualization
    grid_np = grid[0].detach().cpu().numpy()
    stride = 4
    grid_downsampled = grid_np[::stride, ::stride, ::stride]
    feature_mag_downsampled = feature_mag[0, ::stride, ::stride, ::stride]
    
    # Reshape for visualization
    x = grid_downsampled.reshape(-1, 3)[:, 0]
    y = grid_downsampled.reshape(-1, 3)[:, 1]
    z = grid_downsampled.reshape(-1, 3)[:, 2]
    c = feature_mag_downsampled.flatten()
    
    # Create subplots with only the points
    for i, threshold in enumerate(thresholds):
        if i >= 6:
            break
            
        # Create subplot with completely invisible axes
        ax = fig.add_subplot(2, 3, i+1, projection='3d')
        ax.set_axis_off()
        ax.patch.set_alpha(0)
        
        # Make panes invisible - using correct attribute names
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('none')
        ax.yaxis.pane.set_edgecolor('none')
        ax.zaxis.pane.set_edgecolor('none')
        
        # Only plot points with values above threshold
        mask = c > threshold
        points_above = np.sum(mask)
        
        if points_above > 0:
            # Use larger marker size for better visibility
            scatter = ax.scatter(
                x[mask], y[mask], z[mask],
                c=c[mask], 
                cmap='viridis',
                alpha=1.0,  # Fully opaque points
                s=400,  # Fixed large size
                marker='o',  # Circle markers
                edgecolors='none'  # No edge color
            )

    # Adjust layout and save with transparent background
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)

    # Save figure with transparent background
    save_path = os.path.join(output_dir, 'fixed_attention_volume_3d.png')
    plt.savefig(save_path, transparent=True, bbox_inches='tight', pad_inches=0)
    print(f"Saved 3D attention volume visualization to {save_path}")
    plt.close(fig)

    # Create histogram visualization
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot histogram of feature magnitudes
    hist_values, hist_bins, _ = ax.hist(c, bins=100, alpha=0.8, color='teal')
    
    # Mark the thresholds on the histogram
    for threshold in thresholds:
        ax.axvline(x=threshold, linestyle='--', alpha=0.7, 
                  label=f'Threshold: {threshold:.6f}, Points: {np.sum(c > threshold)}')
    
    # Add statistics
    ax.axvline(x=max_val, color='r', linestyle='-', label=f'Max: {max_val:.6f}')
    ax.axvline(x=mean_val, color='g', linestyle='-', label=f'Mean: {mean_val:.6f}')
    ax.axvline(x=np.median(c), color='b', linestyle='-', label=f'Median: {np.median(c):.6f}')
    
    ax.set_title('Attention Feature Magnitude Distribution')
    ax.set_xlabel('Feature Magnitude')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Save histogram figure
    hist_path = os.path.join(output_dir, 'fixed_attention_magnitude_histogram.png')
    plt.savefig(hist_path)
    print(f"Saved feature magnitude histogram to {hist_path}")
    plt.close(fig)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize trained attention weights')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='data/tap3d_boxes',
                        help='Path to the TAP3D dataset directory')
    parser.add_argument('--calibration_file', type=str,
                        default='data/tap3d_boxes/calibration_161029_sports1.json',
                        help='Path to calibration file')
    parser.add_argument('--view_set', type=str, default='boxes', help='View set to visualize')
    parser.add_argument('--output_dir', type=str, default='outputs/fixed_attention_viz', help='Output directory')
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Select views based on view set
    if args.view_set == 'boxes':
        view_names = ['boxes_5', 'boxes_6', 'boxes_7']
    elif args.view_set == 'basketball':
        view_names = ['basketball_3', 'basketball_4', 'basketball_5']
    else:
        raise ValueError(f"Unknown view set: {args.view_set}")
    
    visualize_attention(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        calibration_file=args.calibration_file,
        view_names=view_names,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()
