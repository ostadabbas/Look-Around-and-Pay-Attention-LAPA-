#!/usr/bin/env python3
"""
Simplified evaluation script for ablation studies

This script evaluates trained models and outputs metrics without visualizations,
specifically designed for the ablation study.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import random

# Import from local modules
from lapa.lapa_pipeline import LAPAPipeline
# Define our own TriangulationModule for evaluation
class TriangulationModule(nn.Module):
    """Simplified triangulation module for evaluation."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, points_2d, projection_matrices):
        """Triangulate 3D points from 2D points and projection matrices."""
        batch_size = points_2d[0].shape[0]
        device = points_2d[0].device
        
        # Find the minimum number of points across all views to avoid index errors
        min_points = min([pts.shape[1] for pts in points_2d])
        print(f"Using minimum point count across views: {min_points}")
        
        # Initialize output tensor
        points_3d = torch.zeros(batch_size, min_points, 3, device=device)
        
        # Process each batch element and point
        for b in range(batch_size):
            for p in range(min_points):
                # Extract 2D points from each view
                pts_2d = [pts[b, p] for pts in points_2d]
                
                # Skip if any point is at the default position (-1, -1) which indicates invalid points
                # Get dtype from the first point to ensure type consistency
                if any(torch.allclose(pt, torch.tensor([-1.0, -1.0], device=device, dtype=pts_2d[0].dtype)) for pt in pts_2d):
                    # Set to a default valid position for now
                    points_3d[b, p] = torch.tensor([0.0, 0.0, 5.0], device=device, dtype=points_3d.dtype)
                    continue
                
                # Extract projection matrices
                proj_matrices = [P[b] for P in projection_matrices]
                
                # Build the linear system for DLT
                A = torch.zeros(len(pts_2d) * 2, 4, device=device)
                
                for i, (pt, P) in enumerate(zip(pts_2d, proj_matrices)):
                    x, y = pt[0], pt[1]
                    p1, p2, p3 = P[0], P[1], P[2]
                    
                    A[i*2] = x * p3 - p1
                    A[i*2 + 1] = y * p3 - p2
                
                # Solve the system using SVD
                _, _, Vh = torch.linalg.svd(A)
                point_3d_homogeneous = Vh[-1]
                
                # Convert to inhomogeneous coordinates
                if point_3d_homogeneous[3] != 0:
                    point_3d = point_3d_homogeneous[:3] / point_3d_homogeneous[3]
                else:
                    point_3d = point_3d_homogeneous[:3]
                
                # Handle extreme values - TAP3D uses real-world coordinates in meters
                # Cap extremely large values to reasonable range (±10 meters)
                if torch.any(torch.abs(point_3d) > 10.0):
                    point_3d = torch.clamp(point_3d, min=-10.0, max=10.0)
                    
                points_3d[b, p] = point_3d
        
        # Scale the 3D points to match TAP3D's real-world coordinate system
        # Typically points should be within a few meters of the origin
        # Apply additional normalization if necessary
        max_abs_val = torch.max(torch.abs(points_3d))
        if max_abs_val > 10.0:
            print(f"Scaling down extreme values, max value: {max_abs_val.item()}")
            points_3d = points_3d * (5.0 / max_abs_val)
        
        return points_3d
from lapa.models.geometric_attention_fixed import TrackCorrespondence
from lapa.models.geometric_attention_sfm import TrackCorrespondenceSfM
from lapa.data.tap3d_loader import TAP3DLoader
from train_refinement import RefinementNetwork

def compute_jaccard_index(pred_visibility, gt_visibility):
    """
    Compute Jaccard Index (Intersection over Union) for visibility prediction.
    
    Args:
        pred_visibility: Predicted visibility mask (B, N)
        gt_visibility: Ground truth visibility mask (B, N)
        
    Returns:
        Jaccard Index value (0-1)
    """
    # Convert to binary
    pred_binary = (pred_visibility > 0.5).float()
    gt_binary = (gt_visibility > 0.5).float()
    
    # Compute intersection and union
    intersection = torch.sum(pred_binary * gt_binary)
    union = torch.sum(torch.clamp(pred_binary + gt_binary, 0, 1))
    
    # Compute Jaccard index
    if union > 0:
        jaccard = intersection / union
    else:
        jaccard = torch.tensor(1.0)  # Perfect score if both are empty
    
    return jaccard.item()

def compute_position_accuracy(pred_points, gt_points, gt_visibility, threshold=0.1):
    """
    Compute position accuracy (percentage of points within threshold).
    
    Args:
        pred_points: Predicted 3D points (B, N, 3)
        gt_points: Ground truth 3D points (B, N, 3)
        gt_visibility: Ground truth visibility mask (B, N)
        threshold: Distance threshold as fraction of scene size
        
    Returns:
        Position accuracy value (0-1)
    """
    # Only consider visible points
    mask = gt_visibility.bool()
    
    if torch.sum(mask) == 0:
        return 1.0  # Perfect score if no visible points
    
    # Align pred_points to gt_points to handle potential scale/translation differences
    # First, get centroids
    pred_centroid = torch.mean(pred_points[mask], dim=0, keepdim=True)
    gt_centroid = torch.mean(gt_points[mask], dim=0, keepdim=True)
    
    # Center both point sets
    centered_pred = pred_points[mask] - pred_centroid
    centered_gt = gt_points[mask] - gt_centroid
    
    # Calculate scale difference by comparing the average distance from centroid
    pred_scale = torch.mean(torch.norm(centered_pred, dim=1))
    gt_scale = torch.mean(torch.norm(centered_gt, dim=1))
    
    if pred_scale > 1e-6:  # Avoid division by zero
        scale_ratio = gt_scale / pred_scale
    else:
        scale_ratio = 1.0
    
    # Apply scaling and centering to align with ground truth
    aligned_pred = centered_pred * scale_ratio + gt_centroid
    
    # Now compute distances between aligned points
    distances = torch.norm(aligned_pred - gt_points[mask], dim=1)
    
    # Print some diagnostics
    print(f"Original pred range: {torch.min(pred_points[mask]).item():.4f} to {torch.max(pred_points[mask]).item():.4f}")
    print(f"GT range: {torch.min(gt_points[mask]).item():.4f} to {torch.max(gt_points[mask]).item():.4f}")
    print(f"Scale ratio applied: {scale_ratio:.4f}")
    print(f"Aligned pred range: {torch.min(aligned_pred).item():.4f} to {torch.max(aligned_pred).item():.4f}")
    print(f"Min distance: {torch.min(distances).item():.4f}, Max distance: {torch.max(distances).item():.4f}")
    
    # Compute scene size for normalization based on ground truth points
    with torch.no_grad():
        valid_points = gt_points[mask]
        if valid_points.shape[0] > 0:
            min_vals = torch.min(valid_points, dim=0)[0]
            max_vals = torch.max(valid_points, dim=0)[0]
            scene_size = torch.max(max_vals - min_vals)
            # Ensure a reasonable minimum scene size
            scene_size = max(scene_size.item(), 1.0)
        else:
            scene_size = 1.0
    
    # Normalize distances by scene size
    normalized_distances = distances / scene_size
    
    # Compute accuracy
    accuracy = torch.mean((normalized_distances < threshold).float()).item()
    print(f"Scene size: {scene_size:.4f}, Threshold: {threshold*scene_size:.4f} meters")
    print(f"Position accuracy: {accuracy:.4f} (threshold: {threshold})")
    
    return accuracy

def compute_occlusion_accuracy(pred_visibility, gt_visibility):
    """
    Compute occlusion prediction accuracy.
    
    Args:
        pred_visibility: Predicted visibility mask (B, N)
        gt_visibility: Ground truth visibility mask (B, N)
        
    Returns:
        Occlusion accuracy value (0-1)
    """
    # Convert to binary
    pred_binary = (pred_visibility > 0.5).float()
    gt_binary = (gt_visibility > 0.5).float()
    
    # Compute accuracy
    correct = torch.sum(pred_binary == gt_binary)
    total = pred_binary.numel()
    
    if total > 0:
        accuracy = correct / total
    else:
        accuracy = torch.tensor(1.0)  # Perfect score if no points
    
    return accuracy.item()

def compute_mpjpe(pred_points, gt_points, gt_visibility):
    """
    Compute Mean Per Joint Position Error.
    
    Args:
        pred_points: Predicted 3D points (B, N, 3)
        gt_points: Ground truth 3D points (B, N, 3)
        gt_visibility: Ground truth visibility mask (B, N)
        
    Returns:
        MPJPE value in normalized units
    """
    # Only consider visible points
    mask = gt_visibility.bool()
    
    if torch.sum(mask) == 0:
        return 0.0  # Perfect score if no visible points
    
    # Align pred_points to gt_points to handle potential scale/translation differences
    # First, get centroids
    pred_centroid = torch.mean(pred_points[mask], dim=0, keepdim=True)
    gt_centroid = torch.mean(gt_points[mask], dim=0, keepdim=True)
    
    # Center both point sets
    centered_pred = pred_points[mask] - pred_centroid
    centered_gt = gt_points[mask] - gt_centroid
    
    # Calculate scale difference by comparing the average distance from centroid
    pred_scale = torch.mean(torch.norm(centered_pred, dim=1))
    gt_scale = torch.mean(torch.norm(centered_gt, dim=1))
    
    if pred_scale > 1e-6:  # Avoid division by zero
        scale_ratio = gt_scale / pred_scale
    else:
        scale_ratio = 1.0
    
    # Apply scaling and centering to align with ground truth
    aligned_pred = centered_pred * scale_ratio + gt_centroid
    
    # Compute Euclidean distances using aligned points
    errors = torch.norm(aligned_pred - gt_points[mask], dim=1)
    
    # Compute MPJPE (average distance error)
    mpjpe = torch.mean(errors).item()
    print(f"MPJPE after alignment: {mpjpe:.4f} meters")
    
    return mpjpe

def compute_pck(pred_points, gt_points, gt_visibility, threshold=0.1):
    """
    Compute Percentage of Correct Keypoints.
    
    Args:
        pred_points: Predicted 3D points (B, N, 3)
        gt_points: Ground truth 3D points (B, N, 3)
        gt_visibility: Ground truth visibility mask (B, N)
        threshold: Distance threshold as fraction of scene size
        
    Returns:
        PCK value (0-100)
    """
    # Only consider visible points
    mask = gt_visibility.bool()
    
    if torch.sum(mask) == 0:
        return 100.0  # Perfect score if no visible points
    
    # Align pred_points to gt_points to handle potential scale/translation differences
    # First, get centroids
    pred_centroid = torch.mean(pred_points[mask], dim=0, keepdim=True)
    gt_centroid = torch.mean(gt_points[mask], dim=0, keepdim=True)
    
    # Center both point sets
    centered_pred = pred_points[mask] - pred_centroid
    centered_gt = gt_points[mask] - gt_centroid
    
    # Calculate scale difference by comparing the average distance from centroid
    pred_scale = torch.mean(torch.norm(centered_pred, dim=1))
    gt_scale = torch.mean(torch.norm(centered_gt, dim=1))
    
    if pred_scale > 1e-6:  # Avoid division by zero
        scale_ratio = gt_scale / pred_scale
    else:
        scale_ratio = 1.0
    
    # Apply scaling and centering to align with ground truth
    aligned_pred = centered_pred * scale_ratio + gt_centroid
    
    # Compute Euclidean distances using aligned points
    errors = torch.norm(aligned_pred - gt_points[mask], dim=1)
    
    # Compute scene size for normalization
    with torch.no_grad():
        valid_points = gt_points[mask]
        if valid_points.shape[0] > 0:
            min_vals = torch.min(valid_points, dim=0)[0]
            max_vals = torch.max(valid_points, dim=0)[0]
            scene_size = torch.max(max_vals - min_vals)
        else:
            scene_size = torch.tensor(1.0, device=gt_points.device)
    
    # Normalize distances by scene size
    normalized_distances = distances / scene_size
    
    # Compute MPJPE
    mpjpe = torch.mean(normalized_distances).item()
    
    return mpjpe

def project_points(points_3d, projection_matrix, visibility=None):
    """
    Project 3D points to 2D using a projection matrix.
    
    Args:
        points_3d: 3D points with shape (N, 3)
        projection_matrix: Camera projection matrix (3x4)
        visibility: Optional visibility mask for the points (N,)
        
    Returns:
        2D points with shape (N, 2)
    """
    # Ensure points_3d is a numpy array
    if isinstance(points_3d, torch.Tensor):
        points_3d = points_3d.detach().cpu().numpy()
    
    # Ensure projection_matrix is a numpy array
    if isinstance(projection_matrix, torch.Tensor):
        projection_matrix = projection_matrix.detach().cpu().numpy()
    
    # Add homogeneous coordinate (N, 3) -> (N, 4)
    N = points_3d.shape[0]
    points_homogeneous = np.ones((N, 4))
    points_homogeneous[:, :3] = points_3d
    
    # Project points: P * X = x
    # (3x4) * (4xN) -> (3xN)
    projected_points = np.dot(projection_matrix, points_homogeneous.T)
    
    # Convert to inhomogeneous coordinates
    # Avoid division by zero
    z = projected_points[2, :]
    z[np.abs(z) < 1e-10] = 1e-10  # Small epsilon to avoid division by zero
    
    # Get x, y coordinates
    x = projected_points[0, :] / z
    y = projected_points[1, :] / z
    
    # Stack x, y coordinates
    points_2d = np.vstack((x, y)).T  # (N, 2)
    
    # Apply visibility mask if provided
    if visibility is not None:
        # Set invisible points to (-1, -1) or some other invalid value
        invisible_mask = ~visibility.astype(bool)
        points_2d[invisible_mask] = -1.0
    
    return points_2d


def compute_pck(pred_points, gt_points, gt_visibility, threshold=0.1):
    """
    Compute Percentage of Correct Keypoints.
    
    Args:
        pred_points: Predicted 3D points (B, N, 3)
        gt_points: Ground truth 3D points (B, N, 3)
        gt_visibility: Ground truth visibility mask (B, N)
        threshold: Distance threshold as fraction of scene size
        
    Returns:
        PCK value (0-100)
    """
    # This is essentially the same as position accuracy but expressed as percentage
    accuracy = compute_position_accuracy(pred_points, gt_points, gt_visibility, threshold)
    return accuracy * 100.0

def evaluate_model(args, data_loader, model, device):
    """
    Evaluate the model on test data.
    
    Args:
        args: Command line arguments
        data_loader: Data loader for test data
        model: Trained model
        device: Computation device
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    # Initialize metric lists
    jaccard_indices = []
    position_accuracies = []
    occlusion_accuracies = []
    mpjpes = []
    pcks = []
    
    # Initialize metrics
    jaccard_sum = 0.0
    position_accuracy_sum = 0.0
    occlusion_accuracy_sum = 0.0
    mpjpe_sum = 0.0
    pck_sum = 0.0
    num_samples = 0
    
    # Process test data
    with torch.no_grad():
        # Select a random subset of frames for evaluation
        num_frames = min(args.num_frames, 20)  # Limit to 20 frames for evaluation
        frame_indices = random.sample(range(100), num_frames)  # Assuming 100 frames available
        
        # Get view names for the specified view set
        view_names = []
        if args.view_set == "boxes":
            view_names = ["boxes_5", "boxes_6", "boxes_7"]
        elif args.view_set == "boxes_all":
            view_names = ["boxes_5", "boxes_6", "boxes_7", "boxes_8", "boxes_9"]
        else:
            view_names = [f"{args.view_set}_{i}" for i in range(5, 8)]
            
        print(f"Using view names: {view_names}")
        
        for frame_idx in tqdm(frame_indices, desc="Evaluating"):
            # Prepare data structures
            tracks_2d = []
            projection_matrices = []
            
            # Process each view
            for view_name in view_names:
                # Load view data
                view_data = data_loader.load_view(view_name)
                
                # Get 2D tracks for this frame
                if 'tracks_2d' in view_data and frame_idx < len(view_data['tracks_2d']):
                    tracks = view_data['tracks_2d'][frame_idx]
                else:
                    # If 2D tracks not available, project 3D tracks
                    tracks_3d = view_data['tracks_xyz'][frame_idx]
                    visibility = view_data['visibility'][frame_idx]
                    
                    # Get camera intrinsics and extrinsics
                    intrinsics = view_data['intrinsics']
                    
                    # Get camera parameters from calibration
                    if data_loader.calibration:
                        # Get the projection matrix directly from calibration
                        P = data_loader.calibration.get_projection_matrix(view_name)
                        
                        # Scale the intrinsics for 224x224 images if needed
                        # This is based on the memory about TAP3D dataset handling
                        orig_width, orig_height = 640, 360  # Original TAP3D image dimensions
                        target_width, target_height = 224, 224  # Target dimensions
                        
                        width_scale = target_width / orig_width
                        height_scale = target_height / orig_height
                        
                        # Extract and scale the intrinsic components
                        fx, fy = intrinsics[0], intrinsics[1]
                        cx, cy = intrinsics[2], intrinsics[3]
                        
                        scaled_fx = fx * width_scale
                        scaled_fy = fy * height_scale
                        scaled_cx = cx * width_scale
                        scaled_cy = cy * height_scale
                        
                        # Create a scaled intrinsics matrix
                        K_scaled = np.array([
                            [scaled_fx, 0, scaled_cx],
                            [0, scaled_fy, scaled_cy],
                            [0, 0, 1]
                        ])
                        
                        # Extract R|t from the original projection matrix
                        # P = K[R|t], so [R|t] = K^-1 * P
                        K = np.array([
                            [fx, 0, cx],
                            [0, fy, cy],
                            [0, 0, 1]
                        ])
                        K_inv = np.linalg.inv(K)
                        Rt = K_inv @ P
                        
                        # Create new projection matrix with scaled intrinsics
                        P = K_scaled @ Rt
                    else:
                        # Fallback if no calibration
                        fx, fy, cx, cy = intrinsics
                        K = np.array([
                            [fx, 0, cx],
                            [0, fy, cy],
                            [0, 0, 1]
                        ])
                        RT = np.eye(3, 4)  # Identity rotation and zero translation
                        P = K @ RT
                    
                    # Project 3D points to 2D using our own implementation
                    tracks = project_points(tracks_3d, P, visibility)
                
                tracks_2d.append(torch.tensor(tracks, device=device).unsqueeze(0))  # Add batch dimension
                projection_matrices.append(torch.tensor(P, device=device).unsqueeze(0))  # Add batch dimension
            
            # Get ground truth 3D points and visibility from the first view's data
            # We need to get the 3D tracks from the view data we loaded earlier
            first_view_data = data_loader.load_view(view_names[0])
            
            # Get the 3D tracks for this frame
            gt_points_3d = first_view_data['tracks_xyz'][frame_idx]
            gt_visibility_mask = first_view_data['visibility'][frame_idx]
            
            # Find the minimum number of points across all views
            min_points = min([pts.shape[1] for pts in tracks_2d])
            print(f"Limiting ground truth to {min_points} points to match prediction")
            
            # Truncate ground truth to match the minimum number of points
            gt_points_3d = gt_points_3d[:min_points]
            gt_visibility_mask = gt_visibility_mask[:min_points]
            
            # Convert to tensors and add batch dimension
            gt_points = torch.tensor(gt_points_3d, device=device).unsqueeze(0)  # Add batch dimension
            gt_visibility = torch.tensor(gt_visibility_mask, device=device).unsqueeze(0)  # Add batch dimension
            
            # Triangulate initial 3D points using DLT
            triangulation_module = TriangulationModule().to(device)
            dlt_points = triangulation_module(tracks_2d, projection_matrices)
            
            # Refine 3D points using the model
            refined_points = model(dlt_points)
            
            # Compute metrics
            jaccard = compute_jaccard_index(torch.ones_like(gt_visibility), gt_visibility)  # Assuming all points visible in prediction
            position_accuracy = compute_position_accuracy(refined_points, gt_points, gt_visibility)
            occlusion_accuracy = compute_occlusion_accuracy(torch.ones_like(gt_visibility), gt_visibility)  # Assuming all points visible in prediction
            mpjpe = compute_mpjpe(refined_points, gt_points, gt_visibility)
            pck = compute_pck(refined_points, gt_points, gt_visibility)
            
            # Append to metric lists
            jaccard_indices.append(jaccard)
            position_accuracies.append(position_accuracy)
            occlusion_accuracies.append(occlusion_accuracy)
            mpjpes.append(mpjpe)
            pcks.append(pck)
            
            # Also update metrics sums (for backward compatibility)
            jaccard_sum += jaccard
            position_accuracy_sum += position_accuracy
            occlusion_accuracy_sum += occlusion_accuracy
            mpjpe_sum += mpjpe
            pck_sum += pck
            num_samples += 1
    
    # Compute overall metrics
    avg_jaccard = np.mean(jaccard_indices)
    avg_position_accuracy = np.mean(position_accuracies)
    avg_occlusion_accuracy = np.mean(occlusion_accuracies)
    avg_mpjpe = np.mean(mpjpes)
    avg_pck = np.mean(pcks)
    
    # Print metrics in a consistent format for extraction
    print("\n" + "=" * 50)
    print("EVALUATION METRICS SUMMARY")
    print("=" * 50)
    print(f"Jaccard Index: {avg_jaccard:.4f}")
    print(f"Position Accuracy: {avg_position_accuracy:.4f}")
    print(f"Occlusion Accuracy: {avg_occlusion_accuracy:.4f}")
    print(f"MPJPE: {avg_mpjpe:.4f}")
    print(f"PCK: {avg_pck:.2f}")
    print("=" * 50)
    
    # Save metrics to output directory if provided
    if hasattr(args, 'output_dir') and args.output_dir:
        metrics_file = os.path.join(args.output_dir, 'metrics.txt')
        with open(metrics_file, 'w') as f:
            f.write(f"Jaccard Index: {avg_jaccard:.4f}\n")
            f.write(f"Position Accuracy: {avg_position_accuracy:.4f}\n")
            f.write(f"Occlusion Accuracy: {avg_occlusion_accuracy:.4f}\n")
            f.write(f"MPJPE: {avg_mpjpe:.4f}\n")
            f.write(f"PCK: {avg_pck:.2f}\n")
    
    # Return metrics dictionary
    return {
        "jaccard_index": avg_jaccard,
        "position_accuracy": avg_position_accuracy,
        "occlusion_accuracy": avg_occlusion_accuracy,
        "mpjpe": avg_mpjpe,
        "pck": avg_pck
    }

def main():
    """Main function for evaluation."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate trained model for ablation study")
    
    # Data parameters
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing TAP3D data")
    parser.add_argument("--calibration_file", type=str, required=True,
                        help="Path to calibration file")
    parser.add_argument("--view_set", type=str, default="boxes",
                        help="View set to use for evaluation")
    
    # Model parameters
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--attention_checkpoint", type=str, default=None,
                        help="Path to trained attention model checkpoint (optional)")
    
    # Evaluation parameters
    parser.add_argument("--num_frames", type=int, default=20,
                        help="Number of frames to evaluate")
    parser.add_argument("--skip_visualization", action="store_true",
                        help="Skip visualization and only compute metrics")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="outputs/evaluation",
                        help="Directory to save evaluation results")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize data loader
    data_loader = TAP3DLoader(args.data_dir, args.calibration_file)
    
    # Load refinement model
    refinement_model = RefinementNetwork(
        hidden_dim=512,
        dropout=0.2,
        max_offset=0.3
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    refinement_model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded refinement model from {args.checkpoint}")
    
    # Evaluate model
    metrics = evaluate_model(args, data_loader, refinement_model, device)
    
    # Save metrics to file
    metrics_file = os.path.join(args.output_dir, "metrics.txt")
    with open(metrics_file, "w") as f:
        f.write("Evaluation Metrics:\n")
        f.write(f"Jaccard Index (AJ): {metrics['jaccard_index']:.4f}\n")
        f.write(f"Average Position Accuracy: {metrics['position_accuracy']:.4f}\n")
        f.write(f"Occlusion Accuracy (OA): {metrics['occlusion_accuracy']:.4f}\n")
        f.write(f"MPJPE: {metrics['mpjpe']:.4f}\n")
        f.write(f"PCK: {metrics['pck']:.4f}\n")
    
    print(f"Metrics saved to {metrics_file}")

if __name__ == "__main__":
    main()
