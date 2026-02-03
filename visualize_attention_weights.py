#!/usr/bin/env python3
"""
Attention Weights Visualization

This script visualizes the attention weights and masks in 3D space
to help understand the geometric attention mechanism.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
import argparse

from lapa.models.geometric_attention import GeometricAttention, TrackCorrespondence
from lapa.lapa_pipeline import LAPAPipeline
from lapa.data.tap3d_loader import TAP3DLoader


class AttentionVisualizer:
    """Visualizer for attention weights and masks."""
    
    def __init__(self, data_dir, calibration_file, output_dir='outputs/attention_viz'):
        """
        Initialize the attention visualizer.
        
        Args:
            data_dir: Directory containing TAP3D data
            calibration_file: Path to calibration file
            output_dir: Directory to save visualizations
        """
        self.data_dir = data_dir
        self.calibration_file = calibration_file
        self.output_dir = output_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize pipeline
        self.pipeline = LAPAPipeline(
            data_dir=data_dir,
            calibration_file=calibration_file,
            device=self.device
        )
        
        # Access the geometric attention module
        self.track_correspondence = self.pipeline.track_correspondence
        self.geometric_attention = self.track_correspondence.geometric_attention
        
        # Create a custom colormap for attention weights
        colors = [(0.1, 0.1, 0.5), (0.5, 0.1, 0.1), (0.99, 0.7, 0.18)]
        self.attention_cmap = LinearSegmentedColormap.from_list('attention', colors)

    def load_sample_data(self, view_names):
        """
        Load sample data from the specified views.
        
        Args:
            view_names: List of view names to load
            
        Returns:
            Tuple of 2D tracks and projection matrices
        """
        # Load data
        data = self.pipeline.load_data(view_names)
        
        # Prepare inputs
        tracks_2d, projection_matrices = self.pipeline.prepare_inputs(data)
        
        return tracks_2d, projection_matrices, data
    
    def compute_attention(self, tracks_2d, projection_matrices):
        """
        Compute attention weights using the geometric attention module.
        
        Args:
            tracks_2d: List of 2D tracks tensors
            projection_matrices: List of projection matrices
            
        Returns:
            Tuple of correspondence matrices and volume features
        """
        # Extract a single frame for visualization
        frame_idx = 0
        frame_tracks_2d = [track[:, frame_idx:frame_idx+1, :, :].squeeze(1) for track in tracks_2d]
        
        # Run geometric attention with gradient tracking disabled
        with torch.no_grad():
            correspondence_matrices, volume_features = self.geometric_attention(
                frame_tracks_2d, projection_matrices
            )
        
        return correspondence_matrices, volume_features
    
    def visualize_attention_weights(self, correspondence_matrices, tracks_2d, projection_matrices, frame_idx=0):
        """
        Visualize attention weights between different views.
        
        Args:
            correspondence_matrices: Attention weights between views
            tracks_2d: List of 2D tracks
            projection_matrices: List of projection matrices
            frame_idx: Frame index to visualize
        """
        num_views = len(tracks_2d)
        
        # Create a figure for each view pair
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
                
                # Get shape of correspondence
                batch_size, num_points_i, num_points_j = correspondence.shape
                
                # Use correspondence from first batch
                # Detach from computation graph before converting to numpy
                attn_weights = correspondence[0].detach().cpu().numpy()
                
                # Create figure
                fig = plt.figure(figsize=(12, 10))
                
                # 1. Plot 2D heatmap of attention weights
                ax1 = fig.add_subplot(221)
                im = ax1.imshow(attn_weights, cmap='hot', interpolation='nearest')
                ax1.set_title(f'Attention Weights: View {i} to View {j}')
                ax1.set_xlabel(f'Points in View {j}')
                ax1.set_ylabel(f'Points in View {i}')
                plt.colorbar(im, ax=ax1)
                
                # 2. Plot 3D visualization of strongest attention connections
                ax2 = fig.add_subplot(222, projection='3d')
                
                # Get projection matrices
                P_i = projection_matrices[i][0].detach().cpu().numpy()
                P_j = projection_matrices[j][0].detach().cpu().numpy()
                
                # Threshold for showing connections (only show strongest connections)
                threshold = 0.5
                
                # Find strongest connections
                for pi in range(min(num_points_i, 10)):  # Limit to first 10 points for clarity
                    # Get strongest connections for this point
                    strongest_idx = np.argsort(attn_weights[pi])[::-1][:3]  # Top 3 connections
                    
                    # Plot each strong connection
                    for pj in strongest_idx:
                        if pj < len(track_j) and attn_weights[pi, pj] > threshold:
                            # Get 2D points
                            point_i = track_i[pi]
                            point_j = track_j[pj]
                            
                            # Plot points
                            weight = attn_weights[pi, pj]
                            color = plt.cm.hot(weight)
                            
                            # Draw connection between points
                            # We'll simulate depth by mapping to a range
                            z_i = 0.8
                            z_j = 0.2
                            
                            ax2.scatter(point_i[0], point_i[1], z_i, color=color, s=50)
                            ax2.scatter(point_j[0], point_j[1], z_j, color=color, s=50)
                            ax2.plot([point_i[0], point_j[0]], 
                                    [point_i[1], point_j[1]], 
                                    [z_i, z_j], color=color, alpha=weight)
                
                ax2.set_title(f'Strongest Attention Connections')
                ax2.set_xlabel('X')
                ax2.set_ylabel('Y')
                ax2.set_zlabel('View Depth')
                ax2.set_zlim(0, 1)
                
                # 3. Plot tracks in each view
                ax3 = fig.add_subplot(223)
                ax3.scatter(track_i[:, 0], track_i[:, 1], c='blue', label=f'View {i}')
                for k, point in enumerate(track_i):
                    ax3.annotate(str(k), (point[0], point[1]))
                ax3.set_title(f'Track Points in View {i}')
                ax3.set_xlabel('X')
                ax3.set_ylabel('Y')
                ax3.legend()
                
                ax4 = fig.add_subplot(224)
                ax4.scatter(track_j[:, 0], track_j[:, 1], c='red', label=f'View {j}')
                for k, point in enumerate(track_j):
                    ax4.annotate(str(k), (point[0], point[1]))
                ax4.set_title(f'Track Points in View {j}')
                ax4.set_xlabel('X')
                ax4.set_ylabel('Y')
                ax4.legend()
                
                plt.tight_layout()
                
                # Save figure
                save_path = os.path.join(self.output_dir, f'attention_view_{i}_to_{j}.png')
                plt.savefig(save_path)
                print(f"Saved attention visualization to {save_path}")
                plt.close(fig)
    
    def visualize_3d_attention_volume(self, volume_features, projection_matrices, downsample_factor=4):
        """
        Visualize the 3D attention volume features.
        
        Args:
            volume_features: Volume features from geometric attention
            projection_matrices: List of projection matrices
            downsample_factor: Factor to downsample the volume for visualization
        """
        # Check if volume features exist
        if volume_features is None:
            print("No volume features to visualize")
            return
        
        # Get volume shape and downsample for visualization
        batch_size, vol_size, _, _, feature_dim = volume_features.shape
        
        # Downsample for visualization
        stride = downsample_factor
        volume = volume_features[0, ::stride, ::stride, ::stride].detach().cpu().numpy()
        
        # Create 3D grid
        grid = self.geometric_attention.create_3d_grid(1, self.device)
        grid_downsampled = grid[0, ::stride, ::stride, ::stride].detach().cpu().numpy()
        
        # Project grid to views for reference
        grid_2d_views = self.geometric_attention.project_grid_to_views(
            grid, projection_matrices)
        
        # Compute feature magnitude (as a proxy for attention strength)
        feature_magnitude = np.linalg.norm(volume, axis=-1)
        
        # Normalize for visualization
        max_mag = np.max(feature_magnitude)
        if max_mag > 0:
            feature_magnitude = feature_magnitude / max_mag
        
        # Create figure
        fig = plt.figure(figsize=(16, 8))
        
        # Plot 3D attention volume with feature magnitude as color
        ax1 = fig.add_subplot(121, projection='3d')
        
        # Reshape for visualization
        x = grid_downsampled.reshape(-1, 3)[:, 0]
        y = grid_downsampled.reshape(-1, 3)[:, 1]
        z = grid_downsampled.reshape(-1, 3)[:, 2]
        c = feature_magnitude.flatten()
        
        # Only plot points with significant attention for clarity
        threshold = 0.1
        mask = c > threshold
        
        scatter = ax1.scatter(
            x[mask], y[mask], z[mask],
            c=c[mask], 
            cmap='viridis',
            alpha=0.7, 
            s=70*c[mask],  # Size proportional to attention
        )
        
        ax1.set_title('3D Attention Volume (Feature Magnitude)')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
        plt.colorbar(scatter, ax=ax1, label='Attention Strength')
        
        # Plot 3D attention volume with attention strength as size
        ax2 = fig.add_subplot(122, projection='3d')
        
        # Create different scalar values for color variation
        colors = grid_downsampled.reshape(-1, 3)[:, 2]  # Use Z as color
        
        scatter2 = ax2.scatter(
            x[mask], y[mask], z[mask],
            c=colors[mask], 
            cmap='coolwarm',
            alpha=0.7, 
            s=100*c[mask]  # Size proportional to attention
        )
        
        ax2.set_title('3D Attention Volume (Z-Depth Colored)')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        
        plt.colorbar(scatter2, ax=ax2, label='Z-Depth')
        
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(self.output_dir, 'attention_volume_3d.png')
        plt.savefig(save_path)
        print(f"Saved 3D attention volume visualization to {save_path}")
        plt.close(fig)
        
        # Create volumetric slices visualization
        self.visualize_volume_slices(volume, grid_downsampled, feature_magnitude)
    
    def visualize_volume_slices(self, volume, grid, feature_magnitude):
        """
        Visualize slices of the 3D attention volume.
        
        Args:
            volume: Volume features
            grid: The 3D grid
            feature_magnitude: Magnitude of features (proxy for attention strength)
        """
        # Create slices at different depth levels
        vol_size = grid.shape[0]
        slice_indices = [vol_size // 4, vol_size // 2, 3 * vol_size // 4]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': '3d'})
        
        for i, slice_idx in enumerate(slice_indices):
            ax = axes[i]
            
            # Get slice
            grid_slice = grid[slice_idx]
            magnitude_slice = feature_magnitude[slice_idx]
            
            # Plot slice
            x = grid_slice[:, :, 0].flatten()
            y = grid_slice[:, :, 1].flatten()
            z = grid_slice[:, :, 2].flatten()
            c = magnitude_slice.flatten()
            
            # Only plot points with significant attention
            threshold = 0.1
            mask = c > threshold
            
            scatter = ax.scatter(
                x[mask], y[mask], z[mask],
                c=c[mask], 
                cmap='plasma',
                alpha=0.7, 
                s=80*c[mask],  # Size proportional to attention
            )
            
            ax.set_title(f'Attention Volume Slice {slice_idx}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
        
        plt.colorbar(scatter, ax=axes[-1], label='Attention Strength')
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(self.output_dir, 'attention_volume_slices.png')
        plt.savefig(save_path)
        print(f"Saved volume slices visualization to {save_path}")
        plt.close(fig)
    
    def run_visualization(self, view_names):
        """
        Run the full visualization pipeline.
        
        Args:
            view_names: List of view names to visualize
        """
        print(f"Running attention visualization for views: {view_names}")
        
        # Load sample data
        tracks_2d, projection_matrices, data = self.load_sample_data(view_names)
        
        # Compute attention
        correspondence_matrices, volume_features = self.compute_attention(tracks_2d, projection_matrices)
        
        # Visualize attention weights
        self.visualize_attention_weights(correspondence_matrices, tracks_2d, projection_matrices)
        
        # Visualize 3D attention volume
        self.visualize_3d_attention_volume(volume_features, projection_matrices)
        
        print("Attention visualization complete!")
    """Parse command line arguments."""
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize attention weights and masks')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Path to the root dataset directory containing TAP3D data')
    parser.add_argument('--view_set', type=str, default='boxes', help='View set to visualize')
    parser.add_argument('--output_dir', type=str, default='outputs/attention_viz', help='Output directory')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Select views and calibration file based on view set
    if args.view_set == 'boxes':
        view_names = ['boxes_5', 'boxes_6', 'boxes_7']
        data_subdir = 'tap3d_boxes'
        calibration_file = os.path.join(args.data_dir, data_subdir, 'calibration_161029_sports1.json')
    elif args.view_set == 'basketball':
        view_names = ['basketball_3', 'basketball_4', 'basketball_5']
        data_subdir = 'tap3d_basketball'
        calibration_file = os.path.join(args.data_dir, data_subdir, 'calibration_161029_sports1.json')
    else:
        raise ValueError(f"Unknown view set: {args.view_set}")
    
    # Get full path to data directory
    data_dir = os.path.join(args.data_dir, data_subdir)
    
    print(f"Using data directory: {data_dir}")
    print(f"Using calibration file: {calibration_file}")
    
    # Initialize and run visualizer
    visualizer = AttentionVisualizer(
        data_dir=data_dir,
        calibration_file=calibration_file,
        output_dir=args.output_dir
    )
    
    visualizer.run_visualization(view_names)


if __name__ == "__main__":
    main()
