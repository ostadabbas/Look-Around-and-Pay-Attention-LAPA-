"""
Geometric Volumetric Attention (Fixed)

This module implements the geometric volumetric attention mechanism for establishing
track correspondences across multiple camera views, with the critical fix for volume feature population.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import numpy as np

class GeometricAttention(nn.Module):
    """Geometric Volumetric Attention for multi-camera track correspondence.
    
    This module implements a volumetric attention mechanism that leverages
    geometric information to establish correspondences between tracks across
    multiple camera views.
    """
    
    def __init__(self, 
                 feature_dim: int = 128, 
                 volume_size: int = 32, 
                 num_heads: int = 4):
        """
        Initialize the geometric attention module.
        
        Args:
            feature_dim: Dimension of feature vectors
            volume_size: Size of the volumetric grid
            num_heads: Number of attention heads
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.volume_size = volume_size
        self.num_heads = num_heads
        
        # Feature extraction for 2D tracks
        self.track_encoder = nn.Sequential(
            nn.Linear(2, 64),  # Input: 2D coordinates
            nn.ReLU(),
            nn.Linear(64, feature_dim)
        )
        
        # Multi-head attention
        self.mha = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Final projection
        self.projection = nn.Linear(feature_dim, feature_dim)
        
        # 3D volumetric grid parameters
        self.grid_min = -1.0
        self.grid_max = 1.0
        
        # Additional layers for volume feature generation
        self.grid_encoder = nn.Sequential(
            nn.Linear(3, 64),  # Input: 3D coordinates
            nn.ReLU(),
            nn.Linear(64, feature_dim)
        )
        
        # Feature fusion network
        self.feature_fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
    
    def create_3d_grid(self, 
                      batch_size: int, 
                      device: torch.device) -> torch.Tensor:
        """
        Create a 3D volumetric grid.
        
        Args:
            batch_size: Batch size
            device: Device to create the grid on
            
        Returns:
            3D grid of shape (batch_size, volume_size, volume_size, volume_size, 3)
        """
        # Create 1D linspace for each dimension
        linspace = torch.linspace(self.grid_min, self.grid_max, self.volume_size, device=device)
        
        # Create meshgrid
        x, y, z = torch.meshgrid(linspace, linspace, linspace, indexing='ij')
        
        # Stack to create 3D grid
        grid = torch.stack([x, y, z], dim=-1)
        
        # Expand batch dimension
        grid = grid.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
        
        return grid
    
    def project_grid_to_views(self, 
                             grid: torch.Tensor, 
                             proj_matrices: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Project 3D grid points to multiple 2D views.
        
        Args:
            grid: 3D grid of shape (batch_size, volume_size, volume_size, volume_size, 3)
            proj_matrices: List of projection matrices, each of shape (batch_size, 3, 4)
            
        Returns:
            List of 2D grid projections, each of shape 
            (batch_size, volume_size, volume_size, volume_size, 2)
        """
        batch_size, vs, _, _, _ = grid.shape
        num_views = len(proj_matrices)
        
        # Reshape grid to (batch_size, volume_size^3, 3)
        grid_flat = grid.reshape(batch_size, vs * vs * vs, 3)
        
        # Homogeneous coordinates
        ones = torch.ones(batch_size, grid_flat.shape[1], 1, device=grid.device)
        grid_homo = torch.cat([grid_flat, ones], dim=-1)  # (batch_size, volume_size^3, 4)
        
        # Project to each view
        grid_2d_views = []
        for proj_mat in proj_matrices:
            # proj_mat: (batch_size, 3, 4)
            # grid_homo: (batch_size, volume_size^3, 4)
            # Transpose grid_homo for batch matrix multiplication
            grid_2d_homo = torch.bmm(proj_mat, grid_homo.transpose(1, 2))  # (batch_size, 3, volume_size^3)
            grid_2d_homo = grid_2d_homo.transpose(1, 2)  # (batch_size, volume_size^3, 3)
            
            # Convert homogeneous to pixel coordinates
            grid_2d = grid_2d_homo[:, :, :2] / (grid_2d_homo[:, :, 2:3] + 1e-10)
            
            # Reshape back to volumetric grid
            grid_2d = grid_2d.reshape(batch_size, vs, vs, vs, 2)
            grid_2d_views.append(grid_2d)
        
        return grid_2d_views
    
    def get_view_features(self, 
                         tracks_2d: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Extract features from 2D tracks.
        
        Args:
            tracks_2d: List of 2D tracks, each of shape (batch_size, num_points, 2)
            
        Returns:
            List of track features, each of shape (batch_size, num_points, feature_dim)
        """
        features = []
        for track in tracks_2d:
            # Extract features using the track encoder
            feature = self.track_encoder(track)
            features.append(feature)
        
        return features
    
    def compute_correspondence_matrix(self, 
                                     view_features: List[torch.Tensor]) -> List[List[torch.Tensor]]:
        """
        Compute correspondence matrix between views using attention.
        
        Args:
            view_features: List of view features, each of shape (batch_size, num_points, feature_dim)
            
        Returns:
            Correspondence matrix of shape (batch_size, num_views, num_points, num_views, num_points)
        """
        batch_size = view_features[0].shape[0]
        num_views = len(view_features)
        
        # Initialize correspondence matrix
        correspondence_matrix = []
        
        # Compute attention between each pair of views
        for i in range(num_views):
            query = view_features[i]  # (batch_size, num_points_i, feature_dim)
            
            view_correspondence = []
            for j in range(num_views):
                # Skip self-attention
                if i == j:
                    view_correspondence.append(None)
                    continue
                
                key = view_features[j]  # (batch_size, num_points_j, feature_dim)
                value = key
                
                # Multi-head attention
                attn_output, attn_weights = self.mha(
                    query=query, 
                    key=key, 
                    value=value,
                    need_weights=True
                )
                
                # attn_weights: (batch_size, num_points_i, num_points_j)
                view_correspondence.append(attn_weights)
            
            correspondence_matrix.append(view_correspondence)
        
        return correspondence_matrix
    
    def populate_volume_features(self,
                                grid: torch.Tensor,
                                grid_2d_views: List[torch.Tensor],
                                view_features: List[torch.Tensor],
                                correspondence_matrix: List[List[torch.Tensor]]) -> torch.Tensor:
        """
        Populate the 3D volume with features based on 2D projections and attention.
        
        Args:
            grid: 3D grid of shape (batch_size, volume_size, volume_size, volume_size, 3)
            grid_2d_views: List of 2D projections, each of shape (batch_size, volume_size, volume_size, volume_size, 2)
            view_features: List of view features, each of shape (batch_size, num_points, feature_dim)
            correspondence_matrix: Attention weights between views
            
        Returns:
            Populated volume features of shape (batch_size, volume_size, volume_size, volume_size, feature_dim)
        """
        batch_size = grid.shape[0]
        num_views = len(view_features)
        device = grid.device
        vs = self.volume_size
        
        # Initialize volume features with positional encoding from the grid
        # Process in chunks to save memory
        chunk_size = 8192  # Adjust based on available memory
        num_voxels = vs * vs * vs
        volume_features = torch.zeros(
            batch_size, vs, vs, vs, self.feature_dim, device=device
        )
        
        # Process grid encoding in chunks
        grid_flat = grid.reshape(batch_size, -1, 3)
        for i in range(0, num_voxels, chunk_size):
            end_idx = min(i + chunk_size, num_voxels)
            chunk = grid_flat[:, i:end_idx]
            encoded_chunk = self.grid_encoder(chunk)
            volume_features.reshape(batch_size, -1, self.feature_dim)[:, i:end_idx] = encoded_chunk
        
        # For each view
        for view_idx in range(num_views):
            # Get 2D projections for this view
            grid_2d = grid_2d_views[view_idx]  # (batch_size, vs, vs, vs, 2)
            grid_2d_flat = grid_2d.reshape(batch_size, -1, 2)  # (batch_size, vs^3, 2)
            
            # Get features for this view
            view_feature = view_features[view_idx]  # (batch_size, num_points, feature_dim)
            
            # For each batch
            for b in range(batch_size):
                # Only use the first 2 coords for distance computation
                points_2d = view_feature[b, :, :2]  # (num_points, 2)
                num_points = points_2d.shape[0]
                
                # Process in chunks to save memory
                for i in range(0, num_voxels, chunk_size):
                    end_idx = min(i + chunk_size, num_voxels)
                    
                    # Get grid projections for this chunk
                    grid_proj_chunk = grid_2d_flat[b, i:end_idx]  # (chunk_size, 2)
                    
                    # Initialize feature accumulator for this chunk
                    chunk_features = torch.zeros(end_idx - i, self.feature_dim, device=device)
                    
                    # Process points in smaller batches
                    point_batch_size = 64  # Process points in small batches
                    for j in range(0, num_points, point_batch_size):
                        end_j = min(j + point_batch_size, num_points)
                        
                        # Get points batch
                        points_batch = points_2d[j:end_j]  # (batch_size, 2)
                        features_batch = view_feature[b, j:end_j]  # (batch_size, feature_dim)
                        
                        # Compute distances efficiently
                        # Shape: (chunk_size, point_batch_size)
                        distances = torch.cdist(grid_proj_chunk, points_batch, p=2)
                        
                        # Convert to attention weights with temperature
                        temperature = 0.1
                        attn_weights = F.softmax(-distances / temperature, dim=1)
                        
                        # Weight features directly without expanding
                        # (chunk_size, point_batch_size) @ (point_batch_size, feature_dim)
                        weighted_feats = torch.matmul(attn_weights, features_batch)
                        
                        # Accumulate features
                        chunk_features += weighted_feats
                    
                    # Add to volume features
                    scale_factor = 1.0 / num_views
                    volume_features.reshape(batch_size, -1, self.feature_dim)[b, i:end_idx] += \
                        scale_factor * chunk_features
        
        # Apply feature projection in chunks to save memory
        volume_feats_flat = volume_features.reshape(-1, self.feature_dim)
        projected_feats = torch.zeros_like(volume_feats_flat)
        
        for i in range(0, volume_feats_flat.shape[0], chunk_size):
            end_idx = min(i + chunk_size, volume_feats_flat.shape[0])
            projected_feats[i:end_idx] = self.projection(volume_feats_flat[i:end_idx])
        
        volume_features = projected_feats.reshape(
            batch_size, vs, vs, vs, self.feature_dim)
        
        return volume_features
    
    def forward(self, 
               tracks_2d: List[torch.Tensor], 
               projection_matrices: List[torch.Tensor]) -> Tuple:
        """
        Forward pass of the geometric attention module.
        
        Args:
            tracks_2d: List of 2D tracks, each of shape (batch_size, num_points, 2)
            projection_matrices: List of projection matrices, each of shape (batch_size, 3, 4)
            
        Returns:
            Tuple containing:
            - correspondence_matrix: Attention weights between views
            - volume_features: Features in the volumetric grid
        """
        batch_size = tracks_2d[0].shape[0]
        device = tracks_2d[0].device
        
        # Extract features from 2D tracks
        view_features = self.get_view_features(tracks_2d)
        
        # Compute correspondence matrix between views
        correspondence_matrix = self.compute_correspondence_matrix(view_features)
        
        # Create 3D volumetric grid
        grid = self.create_3d_grid(batch_size, device)
        
        # Project grid to each view
        grid_2d_views = self.project_grid_to_views(grid, projection_matrices)
        
        # Populate volume features based on grid projections and view features
        volume_features = self.populate_volume_features(
            grid, grid_2d_views, view_features, correspondence_matrix)
        
        return correspondence_matrix, volume_features


class TrackCorrespondence(nn.Module):
    """Track Correspondence Module using Geometric Attention.
    
    This module establishes correspondences between 2D tracks across multiple 
    camera views using geometric volumetric attention.
    """
    
    def __init__(self, 
                 feature_dim: int = 128, 
                 volume_size: int = 32, 
                 num_heads: int = 4):
        """
        Initialize the track correspondence module.
        
        Args:
            feature_dim: Dimension of feature vectors
            volume_size: Size of the volumetric grid
            num_heads: Number of attention heads
        """
        super().__init__()
        
        # Geometric attention module
        self.geometric_attention = GeometricAttention(
            feature_dim=feature_dim,
            volume_size=volume_size,
            num_heads=num_heads
        )
        
        # Correspondence refinement
        self.refinement = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
    
    def forward(self, 
               tracks_2d: List[torch.Tensor], 
               projection_matrices: List[torch.Tensor],
               visibility: Optional[List[torch.Tensor]] = None) -> Dict:
        """
        Forward pass of the track correspondence module.
        
        Args:
            tracks_2d: List of 2D tracks, each of shape (batch_size, num_frames, num_points, 2)
            projection_matrices: List of projection matrices, each of shape (batch_size, 3, 4)
            visibility: Optional list of visibility flags, each of shape (batch_size, num_frames, num_points)
            
        Returns:
            Dictionary containing correspondence information
        """
        batch_size, num_frames = tracks_2d[0].shape[:2]
        num_views = len(tracks_2d)
        device = tracks_2d[0].device
        
        # Process each frame independently
        correspondence_matrices = []
        volume_features_list = []
        
        for frame_idx in range(num_frames):
            # Extract current frame data
            frame_tracks_2d = [track[:, frame_idx, :, :] for track in tracks_2d]
            
            # Run geometric attention
            correspondence_matrix, volume_features = self.geometric_attention(
                frame_tracks_2d, projection_matrices
            )
            
            # Store results
            correspondence_matrices.append(correspondence_matrix)
            volume_features_list.append(volume_features)
        
        # Concatenate frame results
        volume_features_tensor = torch.stack(volume_features_list, dim=1)  # (batch_size, num_frames, vs, vs, vs, feature_dim)
        
        # Apply refinement
        refined_features = self.refinement(volume_features_tensor.reshape(-1, self.geometric_attention.feature_dim))
        refined_features = refined_features.reshape(
            batch_size, num_frames, self.geometric_attention.volume_size, 
            self.geometric_attention.volume_size, self.geometric_attention.volume_size,
            self.geometric_attention.feature_dim
        )
        
        return {
            "tracks_2d": tracks_2d,  # Add the original tracks_2d to the return dict
            "correspondences": correspondence_matrices,
            "volume_features": volume_features_tensor,
            "refined_features": refined_features
        }
