#!/usr/bin/env python3
"""
Visualize Refinement Tracks

This script visualizes the results of the refinement network using the same
visualization style as the original LAPA pipeline, showing tracks over time.
"""

import os
import argparse
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from lapa.lapa_pipeline import LAPAPipeline
from lapa.models.track_reconstruction import TriangulationModule
from lapa.data.tap3d_loader import TAP3DLoader
from lapa.visualization.visualizer import LAPAVisualizer
from train_refinement import RefinementNetwork


def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Visualize refinement tracks')
    parser.add_argument('--data_dir', type=str, default='data/tap3d_boxes', help='Directory with TAP3D data')
    parser.add_argument('--calibration_file', type=str, default='data/tap3d_boxes/calibration_161029_sports1.json', help='Path to calibration file')
    parser.add_argument('--checkpoint', type=str, default='outputs/refinement_training/refinement_model_epoch_2.pth', help='Path to model checkpoint')
    parser.add_argument('--view_set', type=str, default='boxes', help='View set to use (e.g., boxes, poster)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_dir', type=str, default='outputs/refinement_tracks', help='Output directory')
    parser.add_argument('--frame_idx', type=int, default=30, help='Frame index to visualize')
    parser.add_argument('--track_history', type=int, default=60, help='Number of frames to show in track history')
    parser.add_argument('--point_size', type=int, default=4, help='Size of points for visualization')
    parser.add_argument('--line_thickness', type=int, default=1, help='Thickness of lines for visualization')
    
    return parser.parse_args()


def visualize_refinement_tracks(args):
    """
    Visualize the refinement tracks.
    
    Args:
        args: Command line arguments
    """
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Select views based on view set
    if args.view_set == 'boxes':
        view_names = ['boxes_5', 'boxes_6', 'boxes_7']
    elif args.view_set == 'poster':
        view_names = ['poster_1', 'poster_2', 'poster_3', 'poster_4']
    else:
        raise ValueError(f"Unknown view set: {args.view_set}")
    
    print(f"Visualizing tracks for views: {view_names}")
    print(f"Using device: {args.device}")
    
    # Initialize data loader
    data_loader = TAP3DLoader(args.data_dir, args.calibration_file)
    
    # Initialize pipeline
    pipeline = LAPAPipeline(
        data_dir=args.data_dir,
        calibration_file=args.calibration_file,
        target_size=(224, 224),
        device=args.device
    )
    
    # Initialize triangulation module
    triangulation = TriangulationModule().to(args.device)
    
    # Initialize refinement network
    refinement_network = RefinementNetwork(
        hidden_dim=256,
        dropout=0.1
    ).to(args.device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    refinement_network.load_state_dict(checkpoint['model_state_dict'])
    refinement_network.eval()
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Initialize visualizer
    visualizer = LAPAVisualizer(output_dir=args.output_dir)
    
    # Load data
    data = data_loader.load_multi_view_data(view_names, (224, 224))
    
    # Prepare inputs
    tracks_2d, projection_matrices = pipeline.prepare_inputs(data)
    
    # Get ground truth
    first_view = view_names[0]
    view_data = data['views'][first_view]
    
    # Get 3D tracks and visibility
    gt_tracks_3d = torch.from_numpy(view_data['tracks_xyz']).float().to(args.device)
    gt_visibility = torch.from_numpy(view_data['visibility']).bool().to(args.device)
    
    # Add batch dimension
    gt_tracks_3d = gt_tracks_3d.unsqueeze(0)
    gt_visibility = gt_visibility.unsqueeze(0)
    
    # Find the minimum number of points across all views
    min_points = min([data['views'][view]['tracks_xyz'].shape[1] for view in view_names])
    
    # Truncate ground truth to match the minimum number of points
    gt_tracks_3d = gt_tracks_3d[:, :, :min_points, :]
    gt_visibility = gt_visibility[:, :, :min_points]
    
    # Get number of frames
    num_frames = min(gt_tracks_3d.shape[1], tracks_2d[0].shape[1])
    
    # Process all frames to build 3D tracks
    print("Processing frames to build 3D tracks...")
    dlt_tracks_3d = []
    refined_tracks_3d = []
    
    for frame_idx in tqdm(range(num_frames)):
        # Get 2D tracks for this frame
        frame_tracks_2d = [track[:, frame_idx:frame_idx+1, :, :] for track in tracks_2d]
        
        # Apply DLT triangulation to get initial 3D points
        with torch.no_grad():
            # For DLT triangulation, we need to extract the 2D points from each view
            points_2d_for_triangulation = []
            for view_idx in range(len(frame_tracks_2d)):
                # Get 2D points for this view and frame
                points_2d_for_triangulation.append(frame_tracks_2d[view_idx][:, 0])
            
            # Triangulate points using DLT
            dlt_points = triangulation.triangulate_points_batch(
                points_2d_for_triangulation,
                projection_matrices
            )
            
            # Add frame dimension to match expected shape
            dlt_points = dlt_points.unsqueeze(1)
            
            # Apply refinement network
            refined_points = refinement_network(dlt_points)
            
            # Store points for this frame
            dlt_tracks_3d.append(dlt_points.cpu().numpy())
            refined_tracks_3d.append(refined_points.cpu().numpy())
    
    # Concatenate tracks along frame dimension
    dlt_tracks_3d = np.concatenate(dlt_tracks_3d, axis=1)
    refined_tracks_3d = np.concatenate(refined_tracks_3d, axis=1)
    
    # Convert ground truth to numpy
    gt_tracks_3d_np = gt_tracks_3d.cpu().numpy()
    gt_visibility_np = gt_visibility.cpu().numpy()
    
    # Extract images for visualization
    images = []
    for view_name in view_names:
        view_images = []
        for frame_idx in range(num_frames):
            # Get JPEG bytes for this frame
            jpeg_bytes = data['views'][view_name]['images_jpeg_bytes'][frame_idx]
            # Decode JPEG bytes to image
            img = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            # Resize image
            img = cv2.resize(img, (224, 224))
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            view_images.append(img)
        images.append(view_images)
    
    # Visualize 3D tracks
    print("Visualizing 3D tracks...")
    
    # Create a dictionary similar to the original pipeline result
    result = {
        'tracks_3d': {
            'dlt': dlt_tracks_3d,
            'refined': refined_tracks_3d,
            'ground_truth': gt_tracks_3d_np
        },
        'visibility': gt_visibility_np,
        'tracks_2d': tracks_2d,
        'projection_matrices': projection_matrices,
        'images': images,
        'view_names': view_names
    }
    
    # Visualize 3D tracks comparison
    visualize_3d_tracks_comparison(
        result,
        args.frame_idx,
        args.track_history,
        args.output_dir
    )
    
    # Visualize 2D projections for each view
    for view_idx, view_name in enumerate(view_names):
        visualize_2d_projections(
            result,
            view_idx,
            args.frame_idx,
            args.track_history,
            args.point_size,
            args.line_thickness,
            args.output_dir
        )


def visualize_3d_tracks_comparison(result, frame_idx, track_history, output_dir):
    """
    Visualize comparison of 3D tracks.
    
    Args:
        result: Dictionary with results
        frame_idx: Frame index to visualize
        track_history: Number of frames to show in track history
        output_dir: Output directory
    """
    # Create 3D plot
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get tracks
    dlt_tracks = result['tracks_3d']['dlt'][0]  # Remove batch dimension
    refined_tracks = result['tracks_3d']['refined'][0]  # Remove batch dimension
    gt_tracks = result['tracks_3d']['ground_truth'][0]  # Remove batch dimension
    visibility = result['visibility'][0]  # Remove batch dimension
    
    # Determine track history range
    start_frame = max(0, frame_idx - track_history)
    end_frame = min(dlt_tracks.shape[0], frame_idx + 1)
    
    # Define colors for tracks
    dlt_color = 'blue'
    refined_color = 'red'
    gt_color = 'green'
    
    # Plot tracks with history
    for point_idx in range(dlt_tracks.shape[1]):
        # Plot DLT track history
        ax.plot(
            dlt_tracks[start_frame:end_frame, point_idx, 0],
            dlt_tracks[start_frame:end_frame, point_idx, 1],
            dlt_tracks[start_frame:end_frame, point_idx, 2],
            color=dlt_color, alpha=0.3, linewidth=1
        )
        
        # Plot refined track history
        ax.plot(
            refined_tracks[start_frame:end_frame, point_idx, 0],
            refined_tracks[start_frame:end_frame, point_idx, 1],
            refined_tracks[start_frame:end_frame, point_idx, 2],
            color=refined_color, alpha=0.3, linewidth=1
        )
        
        # Plot ground truth track history (only if visible)
        visible_frames = visibility[start_frame:end_frame, point_idx]
        if np.any(visible_frames):
            visible_indices = np.where(visible_frames)[0]
            ax.plot(
                gt_tracks[start_frame + visible_indices, point_idx, 0],
                gt_tracks[start_frame + visible_indices, point_idx, 1],
                gt_tracks[start_frame + visible_indices, point_idx, 2],
                color=gt_color, alpha=0.3, linewidth=1
            )
    
    # Plot current frame points
    # DLT points
    ax.scatter(
        dlt_tracks[frame_idx, :, 0],
        dlt_tracks[frame_idx, :, 1],
        dlt_tracks[frame_idx, :, 2],
        color=dlt_color, marker='o', label='DLT', s=30
    )
    
    # Refined points
    ax.scatter(
        refined_tracks[frame_idx, :, 0],
        refined_tracks[frame_idx, :, 1],
        refined_tracks[frame_idx, :, 2],
        color=refined_color, marker='^', label='Refined', s=30
    )
    
    # Ground truth points (only visible ones)
    visible_points = visibility[frame_idx]
    ax.scatter(
        gt_tracks[frame_idx, visible_points, 0],
        gt_tracks[frame_idx, visible_points, 1],
        gt_tracks[frame_idx, visible_points, 2],
        color=gt_color, marker='x', label='Ground Truth', s=30
    )
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'3D Tracks Comparison - Frame {frame_idx}')
    ax.legend()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, f"3d_tracks_comparison_frame_{frame_idx}.png"))
    plt.close(fig)
    
    # Also create a figure showing the coordinate differences
    fig = plt.figure(figsize=(15, 5))
    
    # For each coordinate (X, Y, Z)
    for i, coord in enumerate(['X', 'Y', 'Z']):
        ax = fig.add_subplot(1, 3, i+1)
        
        # Calculate differences
        dlt_gt_diff = np.abs(dlt_tracks[frame_idx, :, i] - gt_tracks[frame_idx, :, i])
        refined_gt_diff = np.abs(refined_tracks[frame_idx, :, i] - gt_tracks[frame_idx, :, i])
        
        # Plot histograms
        ax.hist(dlt_gt_diff, bins=20, alpha=0.5, color=dlt_color, label='DLT-GT')
        ax.hist(refined_gt_diff, bins=20, alpha=0.5, color=refined_color, label='Refined-GT')
        
        ax.set_xlabel(f'{coord} Difference')
        ax.set_ylabel('Count')
        ax.set_title(f'{coord} Coordinate Difference')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"coordinate_differences_frame_{frame_idx}.png"))
    plt.close(fig)


def visualize_2d_projections(result, view_idx, frame_idx, track_history, point_size, line_thickness, output_dir):
    """
    Visualize 2D projections of 3D tracks.
    
    Args:
        result: Dictionary with results
        view_idx: View index
        frame_idx: Frame index to visualize
        track_history: Number of frames to show in track history
        point_size: Size of points
        line_thickness: Thickness of lines
        output_dir: Output directory
    """
    # Get data
    view_name = result['view_names'][view_idx]
    images = result['images'][view_idx]
    projection_matrix = result['projection_matrices'][view_idx][0].cpu().numpy()  # Remove batch dimension
    
    # Get 3D tracks
    dlt_tracks = result['tracks_3d']['dlt'][0]  # Remove batch dimension
    refined_tracks = result['tracks_3d']['refined'][0]  # Remove batch dimension
    gt_tracks = result['tracks_3d']['ground_truth'][0]  # Remove batch dimension
    visibility = result['visibility'][0]  # Remove batch dimension
    
    # Get current frame image
    img = images[frame_idx].copy()
    
    # Determine track history range
    start_frame = max(0, frame_idx - track_history)
    end_frame = min(dlt_tracks.shape[0], frame_idx + 1)
    
    # Define colors
    dlt_color = (255, 0, 0)  # Blue in BGR
    refined_color = (0, 0, 255)  # Red in BGR
    gt_color = (0, 255, 0)  # Green in BGR
    
    # Project 3D tracks to 2D for each frame in history
    for frame in range(start_frame, end_frame):
        # Calculate alpha based on temporal distance
        alpha = 0.3 + 0.7 * (frame - start_frame) / (end_frame - start_frame)
        
        # Project DLT tracks
        for point_idx in range(dlt_tracks.shape[1]):
            # Project DLT point
            dlt_point = dlt_tracks[frame, point_idx]
            dlt_point_homo = np.append(dlt_point, 1.0)
            dlt_point_2d_homo = np.dot(dlt_point_homo, projection_matrix.T)
            dlt_point_2d = dlt_point_2d_homo[:2] / dlt_point_2d_homo[2]
            
            # Project refined point
            refined_point = refined_tracks[frame, point_idx]
            refined_point_homo = np.append(refined_point, 1.0)
            refined_point_2d_homo = np.dot(refined_point_homo, projection_matrix.T)
            refined_point_2d = refined_point_2d_homo[:2] / refined_point_2d_homo[2]
            
            # Project ground truth point (if visible)
            if visibility[frame, point_idx]:
                gt_point = gt_tracks[frame, point_idx]
                gt_point_homo = np.append(gt_point, 1.0)
                gt_point_2d_homo = np.dot(gt_point_homo, projection_matrix.T)
                gt_point_2d = gt_point_2d_homo[:2] / gt_point_2d_homo[2]
            
            # Draw points for current frame
            if frame == frame_idx:
                # Draw DLT point
                cv2.circle(img, tuple(dlt_point_2d.astype(int)), point_size, dlt_color, -1)
                
                # Draw refined point
                cv2.circle(img, tuple(refined_point_2d.astype(int)), point_size, refined_color, -1)
                
                # Draw ground truth point (if visible)
                if visibility[frame, point_idx]:
                    cv2.circle(img, tuple(gt_point_2d.astype(int)), point_size, gt_color, -1)
            
            # Draw track history
            if frame < frame_idx:
                # Get next frame points
                next_frame = frame + 1
                
                # DLT track
                next_dlt_point = dlt_tracks[next_frame, point_idx]
                next_dlt_point_homo = np.append(next_dlt_point, 1.0)
                next_dlt_point_2d_homo = np.dot(next_dlt_point_homo, projection_matrix.T)
                next_dlt_point_2d = next_dlt_point_2d_homo[:2] / next_dlt_point_2d_homo[2]
                
                # Draw DLT track line
                cv2.line(
                    img,
                    tuple(dlt_point_2d.astype(int)),
                    tuple(next_dlt_point_2d.astype(int)),
                    dlt_color,
                    line_thickness
                )
                
                # Refined track
                next_refined_point = refined_tracks[next_frame, point_idx]
                next_refined_point_homo = np.append(next_refined_point, 1.0)
                next_refined_point_2d_homo = np.dot(next_refined_point_homo, projection_matrix.T)
                next_refined_point_2d = next_refined_point_2d_homo[:2] / next_refined_point_2d_homo[2]
                
                # Draw refined track line
                cv2.line(
                    img,
                    tuple(refined_point_2d.astype(int)),
                    tuple(next_refined_point_2d.astype(int)),
                    refined_color,
                    line_thickness
                )
                
                # Ground truth track (if visible in both frames)
                if visibility[frame, point_idx] and visibility[next_frame, point_idx]:
                    next_gt_point = gt_tracks[next_frame, point_idx]
                    next_gt_point_homo = np.append(next_gt_point, 1.0)
                    next_gt_point_2d_homo = np.dot(next_gt_point_homo, projection_matrix.T)
                    next_gt_point_2d = next_gt_point_2d_homo[:2] / next_gt_point_2d_homo[2]
                    
                    # Draw ground truth track line
                    cv2.line(
                        img,
                        tuple(gt_point_2d.astype(int)),
                        tuple(next_gt_point_2d.astype(int)),
                        gt_color,
                        line_thickness
                    )
    
    # Add legend
    legend_y = 20
    cv2.putText(img, "DLT", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, dlt_color, 1)
    cv2.putText(img, "Refined", (10, legend_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, refined_color, 1)
    cv2.putText(img, "Ground Truth", (10, legend_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, gt_color, 1)
    
    # Add title
    cv2.putText(img, f"{view_name} - Frame {frame_idx}", (10, legend_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Save image
    cv2.imwrite(os.path.join(output_dir, f"2d_projection_{view_name}_frame_{frame_idx}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def main():
    """
    Main function for visualizing refinement tracks.
    """
    # Parse arguments
    args = parse_args()
    
    # Visualize refinement tracks
    visualize_refinement_tracks(args)


if __name__ == "__main__":
    main()
