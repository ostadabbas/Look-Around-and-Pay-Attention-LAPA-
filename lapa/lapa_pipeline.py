"""
LAPA Pipeline

Look Around and Pay Attention (LAPA) - Multi-Camera Point Tracking Pipeline.
Main module that integrates the components of the pipeline.
"""

import os
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
from tqdm import tqdm

from lapa.data.tap3d_loader import TAP3DLoader
from lapa.models.geometric_attention import TrackCorrespondence
from lapa.models.track_reconstruction import TrackReconstruction


class LAPAPipeline:
    """
    Look Around and Pay Attention (LAPA) Pipeline for Multi-Camera Point Tracking.
    
    This pipeline combines:
    1. TAP3D data loading with proper camera intrinsics handling
    2. Cross-view track correspondence using geometric volumetric attention
    3. 3D track reconstruction from 2D tracks and correspondences
    """
    
    def __init__(self, 
                 data_dir: str, 
                 calibration_file: str,
                 target_size: Optional[Tuple[int, int]] = (224, 224),
                 feature_dim: int = 128,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the LAPA pipeline.
        
        Args:
            data_dir: Directory containing TAP3D data
            calibration_file: Path to the calibration file
            target_size: Target size for image resizing (optional)
            feature_dim: Dimension of feature vectors
            device: Device to run the pipeline on
        """
        self.data_dir = data_dir
        self.calibration_file = calibration_file
        self.target_size = target_size
        self.device = device
        
        # Initialize data loader
        self.data_loader = TAP3DLoader(data_dir, calibration_file)
        
        # Initialize track correspondence module
        self.track_correspondence = TrackCorrespondence(
            feature_dim=feature_dim
        ).to(device)
        
        # Initialize track reconstruction module
        self.track_reconstruction = TrackReconstruction().to(device)
        
    def load_data(self, view_names: List[str]) -> Dict:
        """
        Load data from the specified views.
        
        Args:
            view_names: List of view names to load
            
        Returns:
            Dictionary with loaded data
        """
        return self.data_loader.load_multi_view_data(view_names, self.target_size)
    
    def prepare_inputs(self, data: Dict) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Prepare inputs for the pipeline.
        
        Args:
            data: Dictionary with loaded data
            
        Returns:
            Tuple containing:
            - List of 2D tracks tensors
            - List of projection matrices tensors
        """
        views = data['views']
        camera_matrices = data['camera_matrices']
        
        # List of views
        view_list = list(views.keys())
        
        # Initialize lists for 2D tracks and projection matrices
        tracks_2d = []
        projection_matrices = []
        
        print("Preparing inputs with correct intrinsics and projection...")
        
        # Process each view
        for view_name in view_list:
            view_data = views[view_name]
            base_view_name = view_name.replace('.npz', '')
            
            # Get 3D tracks and visibility
            tracks_xyz = torch.from_numpy(view_data['tracks_xyz']).float().to(self.device)
            visibility = torch.from_numpy(view_data['visibility']).bool().to(self.device)
            
            # CRITICAL: Get intrinsics directly from the dataset as specified in TAP3D manifest
            if self.target_size is not None and 'scaled_intrinsics' in view_data:
                # Use scaled intrinsics for resized images
                intrinsics = torch.from_numpy(view_data['scaled_intrinsics']).float().to(self.device)
                print(f"View {base_view_name}: Using scaled intrinsics: {intrinsics.cpu().numpy()}")
            else:
                # Use original intrinsics
                intrinsics = torch.from_numpy(view_data['intrinsics']).float().to(self.device)
                print(f"View {base_view_name}: Using original intrinsics: {intrinsics.cpu().numpy()}")
            
            # Get camera extrinsics from calibration
            if base_view_name in camera_matrices:
                camera_matrix = camera_matrices[base_view_name]
                # Convert to tensor and add batch dimension
                ext_matrix = torch.from_numpy(camera_matrix).float().to(self.device).unsqueeze(0)
                print(f"View {base_view_name}: Loaded camera extrinsics matrix")
            else:
                print(f"Warning: No camera matrix found for view {base_view_name}")
                continue
            
            # Project 3D tracks to 2D using simple pinhole model with intrinsics
            # This is for initializing 2D tracks
            fx, fy, cx, cy = intrinsics
            num_frames, num_points, _ = tracks_xyz.shape
            
            # Create 2D tracks by projecting 3D tracks using the same model as in TAP3D manifest
            track_2d = torch.zeros((1, num_frames, num_points, 2), device=self.device)
            
            for f in range(num_frames):
                for p in range(num_points):
                    if visibility[f, p]:
                        X, Y, Z = tracks_xyz[f, p]
                        
                        # Avoid division by zero
                        if abs(Z) < 1e-10:
                            Z = torch.tensor(1e-10, device=self.device)
                            
                        # Project to image plane using simple pinhole model
                        x = fx * X / Z + cx
                        y = fy * Y / Z + cy
                        
                        track_2d[0, f, p, 0] = x
                        track_2d[0, f, p, 1] = y
            
            tracks_2d.append(track_2d)
            
            # Create full projection matrix from intrinsics and extrinsics for triangulation
            # Construct K matrix
            K = torch.zeros((1, 3, 3), device=self.device)
            K[0, 0, 0] = fx
            K[0, 1, 1] = fy
            K[0, 0, 2] = cx
            K[0, 1, 2] = cy
            K[0, 2, 2] = 1.0
            
            # Create full projection matrix: P = K[R|t]
            proj_matrix = torch.matmul(K, ext_matrix)
            projection_matrices.append(proj_matrix)
            
            print(f"View {base_view_name}: Created projection matrix with intrinsics and extrinsics")
        
        return tracks_2d, projection_matrices
    
    def run_pipeline(self, view_names: List[str], visualize: bool = False) -> Dict:
        """
        Run the full LAPA pipeline on the specified views.
        
        Args:
            view_names: List of view names to process
            visualize: Whether to visualize the results
            
        Returns:
            Dictionary with pipeline results
        """
        print(f"Running LAPA pipeline on views: {view_names}")
        
        # Load data
        data = self.load_data(view_names)
        
        # Prepare inputs
        tracks_2d, projection_matrices = self.prepare_inputs(data)
        
        # Run track correspondence
        print("Finding track correspondences across views...")
        correspondence_result = self.track_correspondence(
            tracks_2d, projection_matrices)
        
        # Run track reconstruction
        print("Reconstructing 3D tracks...")
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
    
    def evaluate_pipeline(self, 
                         result: Dict, 
                         gt_tracks_3d: Optional[torch.Tensor] = None) -> Dict:
        """
        Evaluate the pipeline performance.
        
        Args:
            result: Pipeline result dictionary
            gt_tracks_3d: Ground truth 3D tracks (optional)
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Get reconstructed 3D tracks
        tracks_3d = result["tracks_3d"]
        
        # If ground truth is provided, compute metrics
        metrics = {}
        if gt_tracks_3d is not None:
            # Compute mean Euclidean distance
            diff = tracks_3d - gt_tracks_3d
            dist = torch.sqrt(torch.sum(diff**2, dim=-1))
            mean_dist = torch.mean(dist)
            
            metrics["mean_distance"] = mean_dist.item()
            
            # Compute percentage of points with distance below threshold
            threshold = 0.1  # 10 cm
            percentage_below_threshold = torch.mean((dist < threshold).float()).item() * 100
            
            metrics["percentage_below_threshold"] = percentage_below_threshold
        
        # Compute other metrics
        # Number of valid 3D points (non-zero)
        valid_points = torch.sum(torch.norm(tracks_3d, dim=-1) > 1e-6).item()
        total_points = tracks_3d.numel() // 3
        
        metrics["valid_points_percentage"] = valid_points / total_points * 100
        
        return metrics
    
    def visualize_results(self, result: Dict):
        """
        Visualize the pipeline results.
        
        Args:
            result: Pipeline result dictionary
        """
        print("Visualizing results... (not implemented)")
        # Visualization will be implemented in the visualization module
