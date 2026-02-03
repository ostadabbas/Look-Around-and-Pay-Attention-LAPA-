"""
TAP3D Dataset Loader

This module provides tools for loading and preprocessing data from the TAP3D dataset,
with special attention to proper camera intrinsics handling.
"""

import os
import numpy as np
import cv2
import torch
from typing import Dict, List, Tuple, Optional, Union

# Import the TAP3D calibration utilities
from mcmpt.data_utils.tap3d_calibration import TAP3DCalibration


class TAP3DLoader:
    """Loader for TAP3D dataset with proper handling of camera parameters."""
    
    def __init__(self, data_dir: str, calibration_file: str = None):
        """
        Initialize the TAP3D data loader.
        
        Args:
            data_dir: Directory containing TAP3D .npz files
            calibration_file: Path to the calibration file (optional)
        """
        self.data_dir = data_dir
        
        # Load calibration if provided
        self.calibration = None
        if calibration_file is not None and os.path.exists(calibration_file):
            self.calibration = TAP3DCalibration(calibration_file)
    
    def list_views(self) -> List[str]:
        """
        List all available view files in the data directory.
        
        Returns:
            List of view names (without path but with .npz extension)
        """
        return sorted([f for f in os.listdir(self.data_dir) 
                      if f.endswith('.npz') and f.startswith('boxes_')])
    
    def load_view(self, view_name: str) -> Dict:
        """
        Load a single TAP3D view file.
        
        Args:
            view_name: Name of the view file (e.g., 'boxes_5' or 'boxes_5.npz')
            
        Returns:
            Dictionary with the view data
        """
        # Ensure view_name has .npz extension
        if not view_name.endswith('.npz'):
            view_name = f"{view_name}.npz"
            
        view_path = os.path.join(self.data_dir, view_name)
        return self.load_view_from_path(view_path)
    
    def load_view_from_path(self, view_path: str) -> Dict:
        """
        Load a TAP3D view file from its full path.
        
        Args:
            view_path: Full path to the view file
            
        Returns:
            Dictionary with the view data
        """
        print(f"Loading TAP3D file: {view_path}")
        
        data = np.load(view_path)
        
        # Create a dictionary with the data
        result = {}
        
        # Extract image bytes if available
        if 'images_jpeg_bytes' in data:
            result['images_jpeg_bytes'] = data['images_jpeg_bytes']
            print(f"  images_jpeg_bytes shape: {result['images_jpeg_bytes'].shape}")
        
        # Extract 3D tracks if available
        if 'tracks_XYZ' in data:
            result['tracks_xyz'] = data['tracks_XYZ']
            print(f"  tracks_xyz shape: {result['tracks_xyz'].shape}")
        
        # Extract visibility flags if available
        if 'visibility' in data:
            result['visibility'] = data['visibility']
            print(f"  visibility shape: {result['visibility'].shape}")
        
        # Extract camera intrinsics if available - CRITICAL for TAP3D
        if 'fx_fy_cx_cy' in data:
            result['intrinsics'] = data['fx_fy_cx_cy']
            print(f"  intrinsics: {result['intrinsics']}")
        
        # Extract queries if available
        if 'queries_xyt' in data:
            result['queries_xyt'] = data['queries_xyt']
            print(f"  queries_xyt shape: {result['queries_xyt'].shape}")
        
        # Get view name from path
        result['view_name'] = os.path.basename(view_path).split('.')[0]
        
        return result
    
    def decode_images(self, 
                      images_jpeg_bytes: np.ndarray, 
                      target_size: Optional[Tuple[int, int]] = None) -> List[np.ndarray]:
        """
        Decode JPEG bytes to RGB images and optionally resize.
        
        Args:
            images_jpeg_bytes: Array of JPEG bytes
            target_size: Optional target size (width, height) to resize images
            
        Returns:
            List of RGB images
        """
        images = []
        for jpeg_bytes in images_jpeg_bytes:
            # Decode JPEG bytes to image
            img = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
            # Convert from BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Get original dimensions
            orig_height, orig_width = img.shape[:2]
            
            # Resize if target_size is specified
            if target_size is not None:
                img = cv2.resize(img, target_size)
            
            images.append(img)
        
        return images, (orig_width, orig_height)
    
    def scale_intrinsics(self, 
                        intrinsics: np.ndarray, 
                        orig_size: Tuple[int, int], 
                        target_size: Tuple[int, int]) -> np.ndarray:
        """
        Scale camera intrinsics when resizing images.
        
        Args:
            intrinsics: Original intrinsics [fx, fy, cx, cy]
            orig_size: Original image size (width, height)
            target_size: Target image size (width, height)
            
        Returns:
            Scaled intrinsics [fx, fy, cx, cy]
        """
        fx, fy, cx, cy = intrinsics
        
        # Calculate scaling factors
        width_scale = target_size[0] / orig_size[0]
        height_scale = target_size[1] / orig_size[1]
        
        # Scale intrinsics
        scaled_fx = fx * width_scale
        scaled_fy = fy * height_scale
        scaled_cx = cx * width_scale
        scaled_cy = cy * height_scale
        
        return np.array([scaled_fx, scaled_fy, scaled_cx, scaled_cy])
    
    def get_camera_matrices(self, view_names: List[str]) -> Dict[str, np.ndarray]:
        """
        Get camera matrices for the specified views.
        
        Args:
            view_names: List of view names
            
        Returns:
            Dictionary mapping view names to camera matrices
        """
        if self.calibration is None:
            raise ValueError("Calibration file is required for camera matrices")
        
        camera_matrices = {}
        for view_name in view_names:
            # Remove .npz extension if present
            base_view_name = view_name.replace('.npz', '')
            
            # Get camera parameters directly using the view name
            # The TAP3DCalibration class handles the mapping internally
            params = self.calibration.get_camera_params(base_view_name)
            
            # Get rotation and translation matrices
            R = params['R']
            t = params['t']
            
            # Construct camera matrix [R|t]
            camera_matrix = np.hstack((R, t))
            camera_matrices[base_view_name] = camera_matrix
        
        return camera_matrices
    
    def project_3d_to_2d(self, 
                         points_3d: np.ndarray, 
                         intrinsics: np.ndarray) -> np.ndarray:
        """
        Project 3D points to 2D using the simple pinhole camera model.
        
        Args:
            points_3d: 3D points of shape (num_frames, num_points, 3)
            intrinsics: Camera intrinsics [fx, fy, cx, cy] directly from dataset
            
        Returns:
            2D points of shape (num_frames, num_points, 2)
        """
        # CRITICAL: Use exactly the implementation from the TAP3D manifest
        # to ensure compatibility with the dataset
        num_frames, num_points, _ = points_3d.shape
        fx, fy, cx, cy = intrinsics
        
        points_2d = np.zeros((num_frames, num_points, 2))
        
        for f in range(num_frames):
            for p in range(num_points):
                X, Y, Z = points_3d[f, p]
                
                # Avoid division by zero
                if abs(Z) < 1e-10:
                    Z = 1e-10
                    
                # Project to image plane using simple pinhole model
                x = fx * X / Z + cx
                y = fy * Y / Z + cy
                
                points_2d[f, p, 0] = x
                points_2d[f, p, 1] = y
        
        return points_2d
    
    def calculate_infront_cameras(self, 
                                 points_3d: np.ndarray, 
                                 camera_matrix: np.ndarray) -> np.ndarray:
        """
        Calculate which points are in front of the camera.
        This implementation exactly matches the one in tap3d_combined_viz.py.
        
        Args:
            points_3d: 3D points of shape (num_frames, num_points, 3)
            camera_matrix: Camera matrix of shape (3, 4)
            
        Returns:
            Boolean array of shape (num_frames, num_points)
        """
        num_frames, num_points, _ = points_3d.shape
        
        # Get rotation and translation from camera matrix
        R = camera_matrix[:, :3]
        t = camera_matrix[:, 3]
        
        # Calculate camera center
        C = -np.linalg.inv(R) @ t
        
        # Calculate camera z-axis (third row of R)
        z_axis = R[2, :]
        
        # Reshape points for vectorized calculation
        points_flat = points_3d.reshape(-1, 3)
        
        # Calculate vectors from camera center to points
        vectors = points_flat - C
        
        # Calculate dot product with z-axis
        dot_products = np.sum(vectors * z_axis, axis=1)
        
        # Reshape back to original shape
        infront = (dot_products > 0).reshape(num_frames, num_points)
        
        return infront
    
    def load_multi_view_data(self, 
                            view_names: List[str], 
                            target_size: Optional[Tuple[int, int]] = None) -> Dict:
        """
        Load data from multiple views with proper scaling of intrinsics.
        
        Args:
            view_names: List of view names
            target_size: Optional target size for images
            
        Returns:
            Dictionary with multi-view data
        """
        multi_view_data = {
            'views': {},
            'camera_matrices': {}
        }
        
        for view_name in view_names:
            # Load view data
            view_data = self.load_view(view_name)
            
            # Decode images if available
            if 'images_jpeg_bytes' in view_data:
                images, orig_size = self.decode_images(
                    view_data['images_jpeg_bytes'], target_size)
                view_data['images'] = images
                view_data['orig_size'] = orig_size
                
                # Scale intrinsics if target_size is specified
                if target_size is not None and 'intrinsics' in view_data:
                    view_data['scaled_intrinsics'] = self.scale_intrinsics(
                        view_data['intrinsics'], orig_size, target_size)
            
            # Add view data to multi-view data
            multi_view_data['views'][view_name] = view_data
        
        # Get camera matrices if calibration is available
        if self.calibration is not None:
            multi_view_data['camera_matrices'] = self.get_camera_matrices(view_names)
        
        return multi_view_data
