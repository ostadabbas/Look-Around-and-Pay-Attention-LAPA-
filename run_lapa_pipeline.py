#!/usr/bin/env python3
"""
LAPA Pipeline Demo

This script demonstrates the complete LAPA (Look Around and Pay Attention)
pipeline for multi-camera multi-point tracking using the TAP3D dataset.
"""

import os
import argparse
import torch
import numpy as np
from tqdm import tqdm

from lapa.lapa_pipeline import LAPAPipeline
from lapa.visualization.visualizer import LAPAVisualizer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="LAPA Pipeline Demo")
    
    # Data parameters
    parser.add_argument("--data_dir", type=str, default="data/tap3d_boxes",
                        help="Directory containing TAP3D data")
    parser.add_argument("--calib_file", type=str, default="data/tap3d_boxes/calibration_161029_sports1.json",
                        help="TAP3D calibration file")
    parser.add_argument("--output_dir", type=str, default="outputs/lapa_results",
                        help="Directory to save results")
    
    # View selection
    parser.add_argument("--views", type=str, nargs="+", default=["boxes_5", "boxes_6", "boxes_7"],
                        help="TAP3D views to process")
    
    # Processing parameters
    parser.add_argument("--target_size", type=int, default=224,
                        help="Target size for image resizing")
    parser.add_argument("--frame_idx", type=int, default=30,
                        help="Frame index to visualize")
    parser.add_argument("--track_history", type=int, default=60,
                        help="Number of frames to show in track history")
    parser.add_argument("--point_size", type=int, default=4,
                        help="Size of points for 224x224 visualization")
    parser.add_argument("--line_thickness", type=int, default=1,
                        help="Thickness of lines for 224x224 visualization")
    
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
    """Main function for the LAPA pipeline demo."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check data availability
    if not check_data_availability(args.data_dir, args.calib_file, args.views):
        return
    
    print("=" * 80)
    print("LAPA: Look Around and Pay Attention - Multi-Camera Point Tracking Pipeline")
    print("=" * 80)
    print(f"Data directory: {args.data_dir}")
    print(f"Calibration file: {args.calib_file}")
    print(f"Views: {args.views}")
    print(f"Device: {args.device}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)
    
    # Initialize the LAPA pipeline
    print("Initializing LAPA pipeline...")
    pipeline = LAPAPipeline(
        data_dir=args.data_dir,
        calibration_file=args.calib_file,
        target_size=(args.target_size, args.target_size),
        device=args.device
    )
    
    # Initialize the visualizer
    visualizer = LAPAVisualizer(output_dir=args.output_dir)
    
    # Run the pipeline
    print("Running LAPA pipeline...")
    result = pipeline.run_pipeline(args.views)
    
    # Visualize results
    print(f"Visualizing results for frame {args.frame_idx} with track history of {args.track_history} frames...")
    vis_results = visualizer.visualize_results(
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
    metrics = pipeline.evaluate_pipeline(result)
    
    print("\nPipeline Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.2f}")
    
    print("\nResults saved to:", args.output_dir)
    print("\nLAPA pipeline completed successfully!")


if __name__ == "__main__":
    main()
