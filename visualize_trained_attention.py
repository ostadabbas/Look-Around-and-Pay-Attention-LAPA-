#!/usr/bin/env python3
"""
Visualize Attention Weights from Trained Model

This script visualizes the attention weights from a trained model
to see how the attention mechanism has improved after training.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import argparse

from lapa.models.geometric_attention import TrackCorrespondence
from lapa.lapa_pipeline import LAPAPipeline
from lapa.data.tap3d_loader import TAP3DLoader
from train_with_attention import TrainerWithAttention

def create_custom_colormaps():
    """Create custom colormaps for visualization."""
    # Attention weight colormap (red-yellow)
    attn_cmap = LinearSegmentedColormap.from_list(
        'attn', [(0.1, 0.1, 0.1), (0.8, 0.2, 0.2), (1.0, 0.8, 0.2)]
    )
    
    return attn_cmap

def visualize_attention(checkpoint_path, data_dir, calibration_file, view_names, output_dir):
    """
    Visualize attention weights from a trained model.
    
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
    
    # Initialize the pipeline for data loading
    pipeline = LAPAPipeline(
        data_dir=data_dir,
        calibration_file=calibration_file,
        device=device
    )
    
    # Initialize a new TrackCorrespondence module
    track_correspondence = TrackCorrespondence(
        feature_dim=128,
        volume_size=32,
        num_heads=4
    ).to(device)
    
    # Load the trained weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'attention_model' in checkpoint:
        track_correspondence.load_state_dict(checkpoint['attention_model'])
        print(f"Loaded attention model from checkpoint: {checkpoint_path}")
    else:
        print(f"No attention model found in checkpoint: {checkpoint_path}")
        return
    
    # Replace the pipeline's track correspondence with our loaded model
    pipeline.track_correspondence = track_correspondence
    
    # Load data
    print(f"Loading data for views: {view_names}")
    data = pipeline.data_loader.load_multi_view_data(view_names)
    
    # Prepare inputs
    tracks_2d, projection_matrices = pipeline.prepare_inputs(data)
    
    # Run track correspondence with gradient tracking disabled
    print("Computing attention...")
    with torch.no_grad():
        # Extract a single frame for visualization
        frame_idx = 0
        frame_tracks_2d = [track[:, frame_idx:frame_idx+1, :, :].squeeze(1) for track in tracks_2d]
        
        # Get geometric attention module
        geometric_attention = track_correspondence.geometric_attention
        
        # Compute attention
        correspondence_matrices, volume_features = geometric_attention(
            frame_tracks_2d, projection_matrices
        )
    
    # Create custom colormap
    attn_cmap = create_custom_colormaps()
    
    # Visualize attention weights between each pair of views
    print("Visualizing attention weights...")
    num_views = len(view_names)
    
    for i in range(num_views):
        for j in range(num_views):
            if i == j:
                continue  # Skip self-attention
            
            # Get correspondence matrix for this view pair
            correspondence = correspondence_matrices[i][j]
            
            # Skip if correspondence is None
            if correspondence is None:
                continue
            
            # Get tracks for both views (single frame, batch 0)
            track_i = tracks_2d[i][0, frame_idx].detach().cpu().numpy()
            track_j = tracks_2d[j][0, frame_idx].detach().cpu().numpy()
            
            # Use correspondence from first batch
            attn_weights = correspondence[0].detach().cpu().numpy()
            
            # Create figure
            fig = plt.figure(figsize=(18, 10))
            
            # 1. Plot attention heatmap
            ax1 = fig.add_subplot(221)
            im = ax1.imshow(attn_weights, cmap=attn_cmap, interpolation='nearest', vmin=0, vmax=0.3)
            ax1.set_title(f'Attention Weights: View {view_names[i]} to {view_names[j]}')
            ax1.set_xlabel(f'Points in View {view_names[j]}')
            ax1.set_ylabel(f'Points in View {view_names[i]}')
            plt.colorbar(im, ax=ax1)
            
            # 2. Plot attention weights distribution
            ax2 = fig.add_subplot(222)
            ax2.hist(attn_weights.flatten(), bins=50, alpha=0.8, color='teal')
            ax2.set_title('Attention Weights Distribution')
            ax2.set_xlabel('Attention Weight Value')
            ax2.set_ylabel('Frequency')
            ax2.grid(alpha=0.3)
            
            # Calculate some statistics
            max_weight = np.max(attn_weights)
            mean_weight = np.mean(attn_weights)
            median_weight = np.median(attn_weights)
            
            ax2.axvline(x=max_weight, color='r', linestyle='--', label=f'Max: {max_weight:.4f}')
            ax2.axvline(x=mean_weight, color='g', linestyle='--', label=f'Mean: {mean_weight:.4f}')
            ax2.axvline(x=median_weight, color='b', linestyle='--', label=f'Median: {median_weight:.4f}')
            ax2.legend()
            
            # 3. Plot tracks in view i
            ax3 = fig.add_subplot(223)
            ax3.scatter(track_i[:, 0], track_i[:, 1], c='blue', label=f'View {view_names[i]}')
            
            # Only annotate a subset of points for clarity
            step = max(1, len(track_i) // 50)
            for k in range(0, len(track_i), step):
                ax3.annotate(str(k), (track_i[k, 0], track_i[k, 1]))
                
            ax3.set_title(f'Track Points in View {view_names[i]}')
            ax3.set_xlabel('X')
            ax3.set_ylabel('Y')
            ax3.legend()
            
            # 4. Plot tracks in view j
            ax4 = fig.add_subplot(224)
            ax4.scatter(track_j[:, 0], track_j[:, 1], c='red', label=f'View {view_names[j]}')
            
            # Only annotate a subset of points for clarity
            for k in range(0, len(track_j), step):
                ax4.annotate(str(k), (track_j[k, 0], track_j[k, 1]))
                
            ax4.set_title(f'Track Points in View {view_names[j]}')
            ax4.set_xlabel('X')
            ax4.set_ylabel('Y')
            ax4.legend()
            
            plt.tight_layout()
            
            # Save figure
            save_path = os.path.join(output_dir, f'trained_attention_{view_names[i]}_to_{view_names[j]}.png')
            plt.savefig(save_path)
            print(f"Saved attention visualization to {save_path}")
            plt.close(fig)
    
    # Visualize 3D attention volume
    print("Visualizing 3D attention volume...")
    visualize_3d_attention_volume(volume_features, geometric_attention, projection_matrices, output_dir)

def visualize_3d_attention_volume(volume_features, geometric_attention, projection_matrices, output_dir, threshold=0.05):
    """
    Visualize the 3D attention volume.
    
    Args:
        volume_features: Volume features from geometric attention
        geometric_attention: The geometric attention module
        projection_matrices: List of projection matrices
        output_dir: Directory to save visualizations
        threshold: Threshold for attention feature magnitude
    """
    device = volume_features.device
    
    # Create 3D grid
    grid = geometric_attention.create_3d_grid(1, device)
    
    # Compute feature magnitude (as a proxy for attention strength)
    feature_mag = torch.norm(volume_features, dim=-1).detach().cpu().numpy()
    
    # Normalize for visualization
    max_mag = np.max(feature_mag)
    if max_mag > 0:
        feature_mag = feature_mag / max_mag
    
    # Downsample for visualization
    stride = 4
    grid_downsampled = grid[0, ::stride, ::stride, ::stride].detach().cpu().numpy()
    feature_mag_downsampled = feature_mag[0, ::stride, ::stride, ::stride]
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Reshape for visualization
    x = grid_downsampled.reshape(-1, 3)[:, 0]
    y = grid_downsampled.reshape(-1, 3)[:, 1]
    z = grid_downsampled.reshape(-1, 3)[:, 2]
    c = feature_mag_downsampled.flatten()
    
    # Only plot points with significant attention
    mask = c > threshold
    
    if np.sum(mask) > 0:
        scatter = ax.scatter(
            x[mask], y[mask], z[mask],
            c=c[mask], 
            cmap='viridis',
            alpha=0.7, 
            s=100*c[mask]  # Size proportional to attention
        )
        
        plt.colorbar(scatter, ax=ax, label='Attention Strength')
    else:
        ax.text(0, 0, 0, "No attention values above threshold",
                color='red', fontsize=15, ha='center')
    
    ax.set_title('3D Attention Volume (Feature Magnitude)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Add threshold info to plot
    plt.figtext(0.5, 0.01, f"Attention threshold: {threshold}", ha='center', fontsize=10)
    
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(output_dir, 'trained_attention_volume_3d.png')
    plt.savefig(save_path)
    print(f"Saved 3D attention volume visualization to {save_path}")
    plt.close(fig)
    
    # Try different thresholds if needed
    if threshold > 0.01 and np.sum(mask) < 10:
        lower_threshold = threshold / 2
        print(f"Few points above threshold {threshold}, trying lower threshold {lower_threshold}...")
        visualize_3d_attention_volume(volume_features, geometric_attention, projection_matrices, 
                                     output_dir, lower_threshold)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize attention from trained model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='data/tap3d_boxes',
                        help='Path to the TAP3D dataset directory')
    parser.add_argument('--calibration_file', type=str,
                        default='data/tap3d_boxes/calibration_161029_sports1.json',
                        help='Path to calibration file')
    parser.add_argument('--view_set', type=str, default='boxes', help='View set to visualize')
    parser.add_argument('--output_dir', type=str, default='outputs/trained_attention_viz', help='Output directory')
    
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
