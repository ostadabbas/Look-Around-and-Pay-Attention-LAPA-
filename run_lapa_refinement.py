#!/usr/bin/env python3
"""
LAPA Pipeline with Refinement Demo

This script demonstrates the LAPA pipeline with the refinement network
for multi-camera multi-point tracking using the TAP3D dataset.
"""

import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from lapa.lapa_pipeline import LAPAPipeline
from lapa.visualization.visualizer import LAPAVisualizer
from train_refinement import RefinementNetwork
from lapa.models.geometric_attention_fixed import TrackCorrespondence


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="LAPA Pipeline with Refinement Demo")
    
    # Data parameters
    parser.add_argument("--data_dir", type=str, default="data/tap3d_boxes",
                        help="Directory containing TAP3D data")
    parser.add_argument("--calib_file", type=str, default="data/tap3d_boxes/calibration_161029_sports1.json",
                        help="TAP3D calibration file")
    parser.add_argument("--output_dir", type=str, default="outputs/lapa_refinement_results",
                        help="Directory to save results")
    parser.add_argument("--checkpoint", type=str, default="outputs/refinement_training/refinement_model_epoch_5.pth",
                        help="Path to refinement model checkpoint")
    parser.add_argument("--attention_checkpoint", type=str, default=None,
                        help="Path to pre-trained attention model checkpoint (optional)")
    
    # View selection
    parser.add_argument("--views", type=str, nargs="+", default=["boxes_5", "boxes_6", "boxes_7"],
                        help="TAP3D views to process")
    parser.add_argument("--view_set", type=str, default=None,
                        help="Predefined view set to use instead of specifying individual views")
    parser.add_argument("--category", type=str, default=None,
                        help="Category name for organizing output (e.g., boxes, basketball)")
    
    # Processing parameters
    parser.add_argument("--target_size", type=int, default=448,
                        help="Target size for image resizing")
    parser.add_argument("--frame_idx", type=int, default=60,
                        help="Frame index to visualize")
    parser.add_argument("--track_history", type=int, default=60,
                        help="Number of frames to show in track history")
    parser.add_argument("--point_size", type=int, default=7,
                        help="Size of points for visualization")
    parser.add_argument("--line_thickness", type=int, default=3,
                        help="Thickness of lines for visualization")
    
    # Hardware parameters
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run on")
    
    return parser.parse_args()


def check_data_availability(data_dir, calib_file, views):
    """Check if the required data files are available."""
    # Check data directory
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        print("Please download the TAP3D dataset first.")
        return False
    
    # Check calibration file
    if not os.path.exists(calib_file):
        print(f"Calibration file not found: {calib_file}")
        print("Please make sure the calibration file is available.")
        return False
    
    # Check view files
    missing_views = []
    for view in views:
        view_file = os.path.join(data_dir, f"{view}.npz")
        if not os.path.exists(view_file):
            missing_views.append(view)
    
    if missing_views:
        print(f"Missing view files: {missing_views}")
        print("Please download the complete TAP3D dataset.")
        return False
    
    return True


def main():
    """Main function for the LAPA pipeline with refinement demo."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    if args.category:
        args.output_dir = os.path.join(args.output_dir, args.category)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Select views based on view set if specified
    if args.view_set:
        if args.view_set == 'boxes':
            args.views = ['boxes_5', 'boxes_6', 'boxes_7']
        elif args.view_set == 'boxes_alt':
            args.views = ['boxes_11', 'boxes_12', 'boxes_17']
        elif args.view_set == 'boxes_alt2':
            args.views = ['boxes_19', 'boxes_22', 'boxes_27']
        elif args.view_set == 'basketball':
            args.views = ['basketball_3', 'basketball_4', 'basketball_5']
        elif args.view_set == 'basketball_alt':
            args.views = ['basketball_6', 'basketball_9', 'basketball_13']
        elif args.view_set == 'basketball_alt2':
            args.views = ['basketball_14', 'basketball_20', 'basketball_24']
        elif args.view_set == 'softball':
            args.views = ['softball_2', 'softball_9', 'softball_14']
        elif args.view_set == 'softball_alt':
            args.views = ['softball_19', 'softball_21', 'softball_23']
        elif args.view_set == 'tennis':
            args.views = ['tennis_2', 'tennis_4', 'tennis_5']
        elif args.view_set == 'tennis_alt':
            args.views = ['tennis_17', 'tennis_22', 'tennis_23']
        elif args.view_set == 'football':
            args.views = ['football_1', 'football_3', 'football_7']
        elif args.view_set == 'football_alt':
            args.views = ['football_16', 'football_19', 'football_21']
        elif args.view_set == 'juggle':
            args.views = ['juggle_4', 'juggle_5', 'juggle_7']
        elif args.view_set == 'juggle_alt':
            args.views = ['juggle_8', 'juggle_9', 'juggle_22']
        else:
            print(f"Unknown view set: {args.view_set}")
            return
        
        # Set category if not already set
        if not args.category:
            args.category = args.view_set.split('_')[0]
            
    # Check data availability
    if not check_data_availability(args.data_dir, args.calib_file, args.views):
        return
    
    print("==" * 40)
    print("LAPA Pipeline with Refinement Demo")
    print("==" * 40)
    print(f"TAP3D data directory: {args.data_dir}")
    print(f"Calibration file: {args.calib_file}")
    print(f"Views: {args.views}")
    print(f"Target size: {args.target_size}")
    print(f"Frame index: {args.frame_idx}")
    print(f"Track history: {args.track_history}")
    print(f"Device: {args.device}")
    print(f"Output directory: {args.output_dir}")
    print(f"Refinement model checkpoint: {args.checkpoint}")
    if args.attention_checkpoint:
        print(f"Attention model checkpoint: {args.attention_checkpoint}")
    else:
        print("No attention model provided (using default pipeline)")
    print("=" * 80)
    
    # Initialize the LAPA pipeline
    print("Initializing LAPA pipeline...")
    if args.attention_checkpoint:
        print(f"Loading attention model from: {args.attention_checkpoint}")
        attention_model = TrackCorrespondence(
            feature_dim=128,
            volume_size=16,
            num_heads=4
        ).to(args.device)
        
        # Load pre-trained weights
        attention_checkpoint = torch.load(args.attention_checkpoint, map_location=args.device)
        # Check if the key is 'attention_model' or 'model_state_dict'
        if 'attention_model' in attention_checkpoint:
            attention_model.load_state_dict(attention_checkpoint['attention_model'])
            print(f"Loaded attention model from epoch {attention_checkpoint.get('epoch', 'unknown')}")
        else:  
            attention_model.load_state_dict(attention_checkpoint['model_state_dict'])
            print(f"Loaded attention model state dict directly")
            
        # Set model to evaluation mode
        attention_model.eval()
        print("Attention model loaded and frozen for inference")
        
        # Create pipeline with attention model
        pipeline = LAPAPipeline(
            data_dir=args.data_dir,
            calibration_file=args.calib_file,
            target_size=(args.target_size, args.target_size),
            device=args.device
        )
        
        # Replace pipeline's track_correspondence with our pre-trained one
        pipeline.track_correspondence = attention_model
        print("Set pipeline to use pre-trained attention model")
    else:
        # Create pipeline with default attention model
        pipeline = LAPAPipeline(
            data_dir=args.data_dir,
            calibration_file=args.calib_file,
            target_size=(args.target_size, args.target_size),
            device=args.device
        )
        print("Using default pipeline without pre-trained attention")
    
    # Initialize refinement network
    print("Initializing refinement network...")
    refinement_network = RefinementNetwork(
        hidden_dim=512,
        dropout=0.2,
        max_offset=0.3
    ).to(args.device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    refinement_network.load_state_dict(checkpoint['model_state_dict'])
    refinement_network.eval()
    print(f"Loaded refinement model checkpoint from epoch {checkpoint['epoch']}")
    
    # Initialize visualizers with separate output directories
    original_output_dir = os.path.join(args.output_dir, "original")
    refined_output_dir = os.path.join(args.output_dir, "refined")
    
    # Create output directories
    os.makedirs(original_output_dir, exist_ok=True)
    os.makedirs(refined_output_dir, exist_ok=True)
    
    # Initialize visualizers
    original_visualizer = LAPAVisualizer(output_dir=original_output_dir)
    refined_visualizer = LAPAVisualizer(output_dir=refined_output_dir)
    
    # Run the pipeline
    print("Running LAPA pipeline...")
    result = pipeline.run_pipeline(args.views)
    
    # Get the original 3D tracks from the result
    original_tracks_3d = result['tracks_3d'].clone()
    
    # Apply refinement to the 3D tracks
    print("Applying refinement to 3D tracks...")
    with torch.no_grad():
        # Add batch dimension for the refinement network
        tracks_3d_batch = original_tracks_3d.unsqueeze(0)
        
        # Apply refinement
        refined_tracks_3d_batch = refinement_network(tracks_3d_batch)
        
        # Remove batch dimension
        refined_tracks_3d = refined_tracks_3d_batch.squeeze(0)
        
        # Filter out invalid points from the refined tracks
        # This addresses the issue with weird points appearing in the visualization
        print("Filtering invalid points from refined tracks...")
        
        # Check for NaN or Inf values
        mask_invalid = torch.isnan(refined_tracks_3d) | torch.isinf(refined_tracks_3d)
        if torch.any(mask_invalid):
            # Replace invalid values with corresponding original values
            refined_tracks_3d = torch.where(mask_invalid, original_tracks_3d, refined_tracks_3d)
            print(f"Fixed {torch.sum(mask_invalid).item()} NaN/Inf values in refined tracks")
        
        # Check for points with near-zero values (these cause the weird points in the corner)
        # A point is near-zero if its norm is very small
        point_norms = torch.norm(refined_tracks_3d, dim=-1)
        near_zero_mask = point_norms < 1e-3
        if torch.any(near_zero_mask):
            # Replace near-zero points with corresponding original points
            refined_tracks_3d = torch.where(
                near_zero_mask.unsqueeze(-1).expand_as(refined_tracks_3d),
                original_tracks_3d,
                refined_tracks_3d
            )
            print(f"Fixed {torch.sum(near_zero_mask).item()} near-zero points in refined tracks")
        
        # Check for extreme outliers (values that are too large)
        extreme_mask = torch.abs(refined_tracks_3d) > 10.0
        if torch.any(extreme_mask):
            # Cap extreme values at ±10 units while preserving their sign
            refined_tracks_3d = torch.where(
                extreme_mask,
                torch.sign(refined_tracks_3d) * 10.0,  # Cap at ±10 units
                refined_tracks_3d
            )
            print(f"Capped {torch.sum(extreme_mask).item()} extreme values in refined tracks")
        
        # Debug the ranges after filtering
        with torch.no_grad():
            min_vals, _ = torch.min(refined_tracks_3d.reshape(-1, 3), dim=0)
            max_vals, _ = torch.max(refined_tracks_3d.reshape(-1, 3), dim=0)
            print(f"Refined tracks after filtering - X range: {min_vals[0].item()} to {max_vals[0].item()}")
            print(f"Refined tracks after filtering - Y range: {min_vals[1].item()} to {max_vals[1].item()}")
            print(f"Refined tracks after filtering - Z range: {min_vals[2].item()} to {max_vals[2].item()}")
    # Create a copy of the result dictionary with refined tracks
    refined_result = result.copy()
    refined_result['tracks_3d'] = refined_tracks_3d
    
    # Visualize refined pipeline results
    print(f"Visualizing refined results for frame {args.frame_idx}...")
    vis_results = refined_visualizer.visualize_results(
        refined_result,
        frame_idx=args.frame_idx,
        track_history=args.track_history,
        save=True,
        target_size=(args.target_size, args.target_size),
        point_size=args.point_size,
        line_thickness=args.line_thickness,
        max_points=50,
        show_occ=True
    )

    # Visualize original pipeline results
    print(f"Visualizing original results for frame {args.frame_idx}...")
    vis_results = original_visualizer.visualize_results(
        result,
        frame_idx=args.frame_idx,
        track_history=args.track_history,
        save=True,
        target_size=(args.target_size, args.target_size),
        point_size=args.point_size,
        line_thickness=args.line_thickness,
        max_points=50,
        show_occ=True
    )
    
    
    
    # Evaluate pipeline
    print("Evaluating pipeline performance...")
    original_metrics = pipeline.evaluate_pipeline(result)
    refined_metrics = pipeline.evaluate_pipeline(refined_result)
    
    print("\nOriginal Pipeline Metrics:")
    for key, value in original_metrics.items():
        print(f"  {key}: {value:.2f}")
    
    print("\nRefined Pipeline Metrics:")
    for key, value in refined_metrics.items():
        print(f"  {key}: {value:.2f}")
    
    print("\nResults saved to:", args.output_dir)
    print("\nLAPA pipeline with refinement completed successfully!")


# Removed 3D comparison visualization function


if __name__ == "__main__":
    main()
