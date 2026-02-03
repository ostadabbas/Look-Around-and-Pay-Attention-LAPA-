"""
Geometric Volumetric Attention with SfM Constraints

This module implements an alternative geometric volumetric attention mechanism
that uses Structure from Motion (SfM) constraints instead of epipolar geometry
for establishing track correspondences across multiple camera views.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import numpy as np

class GeometricAttentionSfM(nn.Module):
    """Geometric Volumetric Attention using SfM constraints for multi-camera track correspondence.
    
    This module implements a volumetric attention mechanism that leverages
    Structure from Motion constraints to establish correspondences between tracks across
    multiple camera views.
    """
    
    def __init__(self, 
                 feature_dim: int = 128, 
                 volume_size: int = 16, 
                 num_heads: int = 4):
        """
        Initialize the SfM-based geometric attention module.
        
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
        
        # SfM-specific layers for reprojection consistency
        self.reprojection_encoder = nn.Sequential(
            nn.Linear(4, 64),  # Input: 2D coordinates from two views
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # SfM consistency predictor (predicts if two points are consistent in 3D)
        self.sfm_consistency = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
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
    
    def compute_sfm_consistency(self, 
                              tracks_2d: List[torch.Tensor], 
                              projection_matrices: List[torch.Tensor]) -> List[List[torch.Tensor]]:
        """
        Compute SfM consistency between points in different views.
        
        Args:
            tracks_2d: List of 2D tracks, each of shape (batch_size, num_points, 2)
            projection_matrices: List of projection matrices, each of shape (batch_size, 3, 4)
            
        Returns:
            List of SfM consistency matrices, each of shape (batch_size, num_points_i, num_points_j)
        """
        batch_size = tracks_2d[0].shape[0]
        num_views = len(tracks_2d)
        device = tracks_2d[0].device
        
        # Initialize consistency matrix
        consistency_matrix = []
        
        # For each pair of views
        for i in range(num_views):
            view_i_tracks = tracks_2d[i]  # (batch_size, num_points_i, 2)
            proj_i = projection_matrices[i]  # (batch_size, 3, 4)
            
            view_consistency = []
            for j in range(num_views):
                # Skip self-consistency
                if i == j:
                    view_consistency.append(None)
                    continue
                
                view_j_tracks = tracks_2d[j]  # (batch_size, num_points_j, 2)
                proj_j = projection_matrices[j]  # (batch_size, 3, 4)
                
                # Get number of points in each view
                num_points_i = view_i_tracks.shape[1]
                num_points_j = view_j_tracks.shape[1]
                
                # Initialize consistency scores
                consistency_scores = torch.zeros(
                    batch_size, num_points_i, num_points_j, device=device
                )
                
                # For each batch
                for b in range(batch_size):
                    # Get projection matrices for this batch
                    P_i = proj_i[b]  # (3, 4)
                    P_j = proj_j[b]  # (3, 4)
                    
                    # Get camera centers (null space of projection matrices)
                    # For a projection matrix P = [M | -MC] where M is 3x3 and C is the camera center
                    # The camera center C can be computed as C = -M^-1 * b where P = [M | b]
                    M_i = P_i[:, :3]  # (3, 3)
                    b_i = P_i[:, 3]   # (3,)
                    M_j = P_j[:, :3]  # (3, 3)
                    b_j = P_j[:, 3]   # (3,)
                    
                    # Compute camera centers
                    C_i = -torch.linalg.solve(M_i, b_i)  # (3,)
                    C_j = -torch.linalg.solve(M_j, b_j)  # (3,)
                    
                    # For each point in view i
                    for pi in range(num_points_i):
                        # Get 2D point in view i
                        point_i = view_i_tracks[b, pi]  # (2,)
                        
                        # Create ray from camera center through the 2D point
                        # Convert 2D point to homogeneous coordinates
                        point_i_homo = torch.cat([point_i, torch.ones(1, device=device)])  # (3,)
                        
                        # Get 3D ray direction by multiplying with inverse of intrinsic matrix
                        # Approximate by using normalized homogeneous coordinates
                        ray_i = torch.linalg.solve(M_i, point_i_homo)  # (3,)
                        ray_i = ray_i / torch.norm(ray_i)
                        
                        # For each point in view j
                        for pj in range(num_points_j):
                            # Get 2D point in view j
                            point_j = view_j_tracks[b, pj]  # (2,)
                            
                            # Create ray from camera center through the 2D point
                            point_j_homo = torch.cat([point_j, torch.ones(1, device=device)])  # (3,)
                            ray_j = torch.linalg.solve(M_j, point_j_homo)  # (3,)
                            ray_j = ray_j / torch.norm(ray_j)
                            
                            # Compute closest points on the two rays
                            # This is a simplified SfM constraint - we check how close the rays come to intersecting
                            # Direction vectors of the rays
                            v_i = ray_i
                            v_j = ray_j
                            
                            # Vector between camera centers
                            w0 = C_i - C_j
                            
                            # Compute parameters for closest points
                            a = torch.dot(v_i, v_i)
                            b = torch.dot(v_i, v_j)
                            c = torch.dot(v_j, v_j)
                            d = torch.dot(v_i, w0)
                            e = torch.dot(v_j, w0)
                            
                            # Compute parameters for closest points
                            denom = a*c - b*b
                            
                            # Handle parallel rays
                            if abs(denom) < 1e-6:
                                # Parallel rays - use distance between camera centers
                                s_c = 0
                                t_c = 0
                            else:
                                s_c = (b*e - c*d) / denom
                                t_c = (a*e - b*d) / denom
                            
                            # Compute closest points on the rays
                            closest_i = C_i + s_c * v_i
                            closest_j = C_j + t_c * v_j
                            
                            # Compute distance between closest points
                            distance = torch.norm(closest_i - closest_j)
                            
                            # Convert distance to consistency score (closer is better)
                            # Use a soft threshold with sigmoid
                            consistency = torch.sigmoid(-distance * 5.0)  # Scale for sharper transition
                            
                            # Store consistency score - ensure indices are integers
                            consistency_scores[b, int(pi), int(pj)] = consistency
                
                view_consistency.append(consistency_scores)
            
            consistency_matrix.append(view_consistency)
        
        return consistency_matrix
    
    def compute_correspondence_matrix(self, 
                                     view_features: List[torch.Tensor],
                                     sfm_consistency: List[List[torch.Tensor]]) -> List[List[torch.Tensor]]:
        """
        Compute correspondence matrix between views using attention and SfM consistency.
        
        Args:
            view_features: List of view features, each of shape (batch_size, num_points, feature_dim)
            sfm_consistency: SfM consistency scores between views
            
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
                
                # Combine attention weights with SfM consistency
                # sfm_consistency[i][j]: (batch_size, num_points_i, num_points_j)
                # attn_weights: (batch_size, num_points_i, num_points_j)
                combined_weights = attn_weights * sfm_consistency[i][j]
                
                # Normalize combined weights
                combined_weights = F.normalize(combined_weights, p=1, dim=2)
                
                # Store combined weights
                view_correspondence.append(combined_weights)
            
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
        Forward pass of the SfM-based geometric attention module.
        
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
        
        # Compute SfM consistency between views
        sfm_consistency = self.compute_sfm_consistency(tracks_2d, projection_matrices)
        
        # Compute correspondence matrix between views using both attention and SfM consistency
        correspondence_matrix = self.compute_correspondence_matrix(view_features, sfm_consistency)
        
        # Create 3D volumetric grid
        grid = self.create_3d_grid(batch_size, device)
        
        # Project grid to each view
        grid_2d_views = self.project_grid_to_views(grid, projection_matrices)
        
        # Populate volume features based on grid projections and view features
        volume_features = self.populate_volume_features(
            grid, grid_2d_views, view_features, correspondence_matrix)
        
        return correspondence_matrix, volume_features


class TrackCorrespondenceSfM(nn.Module):
    """Track Correspondence Module using SfM-based Geometric Attention.
    
    This module establishes correspondences between 2D tracks across multiple 
    camera views using SfM-based geometric volumetric attention.
    """
    
    def __init__(self, 
                 feature_dim: int = 128, 
                 volume_size: int = 16, 
                 num_heads: int = 4):
        """
        Initialize the track correspondence module with SfM constraints.
        
        Args:
            feature_dim: Dimension of feature vectors
            volume_size: Size of the volumetric grid
            num_heads: Number of attention heads
        """
        super().__init__()
        
        # Geometric attention module with SfM constraints
        self.geometric_attention = GeometricAttentionSfM(
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
        Forward pass of the track correspondence module with SfM constraints.
        
        Args:
            tracks_2d: List of 2D tracks, each of shape (batch_size, num_frames, num_points, 2)
            projection_matrices: List of projection matrices, each of shape (batch_size, 3, 4)
            visibility: Optional list of visibility flags, each of shape (batch_size, num_frames, num_points)
            
        Returns:
            Dictionary containing correspondence information
        """
        batch_size, num_frames, _, _ = tracks_2d[0].shape
        num_views = len(tracks_2d)
        device = tracks_2d[0].device
        
        # Process each frame independently
        all_correspondences = []
        all_volume_features = []
        
        for frame_idx in range(num_frames):
            # Extract tracks for current frame
            frame_tracks = [
                track[:, frame_idx] for track in tracks_2d
            ]
            
            # Forward pass through geometric attention
            correspondence_matrix, volume_features = self.geometric_attention(
                frame_tracks, projection_matrices
            )
            
            all_correspondences.append(correspondence_matrix)
            all_volume_features.append(volume_features)
        
        # Combine results across frames
        result = {
            'correspondence_matrices': all_correspondences,
            'volume_features': all_volume_features
        }
        
        return result
