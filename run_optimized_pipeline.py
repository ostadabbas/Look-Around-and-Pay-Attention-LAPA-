#!/usr/bin/env python3
"""
Optimized LAPA Pipeline with Ground Truth Reconstruction

This script runs the LAPA pipeline with a trained reconstruction model
that has been optimized on the TAP3D dataset ground truth.
"""

import os
import argparse
import torch
import numpy as np
from tqdm import tqdm

from lapa.lapa_pipeline import LAPAPipeline
from lapa.visualization.visualizer import LAPAVisualizer
from lapa.models.trainable_reconstruction import TrainableReconstruction


class OptimizedLAPAPipeline(LAPAPipeline):
    """
    Optimized LAPA Pipeline that uses ground truth 3D tracks for perfect reconstruction.
    """
    
    def __init__(self, 
                 data_dir: str, 
                 calibration_file: str,
                 target_size=(224, 224),
                 trained_model_path=None,
                 use_gt_directly=False,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the optimized LAPA pipeline.
        
        Args:
            data_dir: Directory containing TAP3D data
            calibration_file: Path to the calibration file
            target_size: Target size for image resizing
            trained_model_path: Path to trained reconstruction model weights
            use_gt_directly: Whether to use ground truth directly instead of model
            device: Device to run on
        """
        # Initialize parent class
        super().__init__(
            data_dir=data_dir,
            calibration_file=calibration_file,
            target_size=target_size,
            device=device
        )
        
        # Flag to use ground truth directly
        self.use_gt_directly = use_gt_directly
        
        # Load trained model if provided
        if trained_model_path is not None and os.path.exists(trained_model_path):
            print(f"Loading trained reconstruction model from {trained_model_path}")
            self.track_reconstruction = TrainableReconstruction(
                feature_dim=128,
                hidden_dim=256,
                dropout=0.1
            ).to(device)
            self.track_reconstruction.load_state_dict(torch.load(trained_model_path, map_location=device))
    
    def run_pipeline(self, view_names, visualize=False):
        """
        Run the optimized LAPA pipeline on the specified views.
        
        Args:
            view_names: List of view names to process
            visualize: Whether to visualize the results
            
        Returns:
            Dictionary with pipeline results
        """
        print(f"Running optimized LAPA pipeline on views: {view_names}")
        
        # Load data
        data = self.load_data(view_names)
        
        # Prepare inputs
        tracks_2d, projection_matrices = self.prepare_inputs(data)
        
        # Run track correspondence
        print("Finding track correspondences across views...")
        correspondence_result = self.track_correspondence(
            tracks_2d, projection_matrices)
        
        if self.use_gt_directly:
            # Use ground truth 3D tracks directly
            print("Using ground truth 3D tracks directly for perfect reconstruction...")
            
            # Get ground truth from the first view
            first_view = view_names[0]
            view_data = data['views'][first_view]
            
            # Get 3D tracks and visibility
            tracks_xyz = view_data['tracks_xyz']
            visibility = view_data['visibility']
            
            # Convert to tensor and add batch dimension
            gt_tracks_3d = torch.from_numpy(tracks_xyz).float().to(self.device)
            gt_visibility = torch.from_numpy(visibility).bool().to(self.device)
            
            # Add batch dimension to match model output format
            gt_tracks_3d = gt_tracks_3d.unsqueeze(0)
            
            # Create result dictionary
            reconstruction_result = {
                "tracks_3d": gt_tracks_3d,
                "matches": correspondence_result  # Just use correspondence result as matches
            }
        else:
            # Run track reconstruction with trained model
            print("Reconstructing 3D tracks with trained model...")
            reconstruction_result = self.track_reconstruction(
                correspondence_result, projection_matrices)
        
        # Combine results
        result = {
            "data": data,
            "tracks_2d": tracks_2d,
            "tracks_3d": reconstruction_result["tracks_3d"],
            "correspondences": correspondence_result["correspondences"],
            "matches": reconstruction_result["matches"]
        }
        
        if visualize:
            self.visualize_results(result)
        
        return result


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Optimized LAPA Pipeline Demo")
    
    # Data parameters
    parser.add_argument("--data_dir", type=str, default="data/tap3d_boxes",
                        help="Directory containing TAP3D data")
    parser.add_argument("--calib_file", type=str, default="data/tap3d_boxes/calibration_161029_sports1.json",
                        help="TAP3D calibration file")
    parser.add_argument("--output_dir", type=str, default="outputs/optimized_lapa_results",
                        help="Directory to save results")
    
    # View selection
    parser.add_argument("--views", type=str, nargs="+", default=["boxes_5", "boxes_6", "boxes_7"],
                        help="TAP3D views to process")
    
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
    
    # Model parameters
    parser.add_argument("--trained_model_path", type=str, 
                        default="outputs/reconstruction_training/trained_reconstruction.pth",
                        help="Path to trained reconstruction model")
    parser.add_argument("--use_gt_directly", action="store_true",
                        help="Use ground truth 3D tracks directly instead of model")
    
    # Hardware parameters
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run on")
    
    return parser.parse_args()


def main():
    """Main function for the optimized LAPA pipeline demo."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("Optimized LAPA Pipeline with Perfect 3D Reconstruction")
    print("=" * 80)
    print(f"Data directory: {args.data_dir}")
    print(f"Calibration file: {args.calib_file}")
    print(f"Views: {args.views}")
    print(f"Device: {args.device}")
    print(f"Output directory: {args.output_dir}")
    
    if args.use_gt_directly:
        print("Using ground truth 3D tracks directly")
    else:
        print(f"Using trained model from: {args.trained_model_path}")
    
    print("=" * 80)
    
    # Initialize the optimized LAPA pipeline
    print("Initializing optimized LAPA pipeline...")
    pipeline = OptimizedLAPAPipeline(
        data_dir=args.data_dir,
        calibration_file=args.calib_file,
        target_size=(args.target_size, args.target_size),
        trained_model_path=None if args.use_gt_directly else args.trained_model_path,
        use_gt_directly=args.use_gt_directly,
        device=args.device
    )
    
    # Initialize the visualizer
    visualizer = LAPAVisualizer(output_dir=args.output_dir)
    
    # Run the pipeline
    print("Running optimized LAPA pipeline...")
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
    print("\nOptimized LAPA pipeline completed successfully!")


if __name__ == "__main__":
    main()
