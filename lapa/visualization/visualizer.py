"""
LAPA Visualization Module

Tools for visualizing the results of the LAPA pipeline, including:
- 2D tracks with correspondences
- 3D reconstructed tracks
- Multi-view visualization
"""

import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from typing import Dict, List, Tuple, Optional, Union


class LAPAVisualizer:
    """
    Visualizer for LAPA pipeline results.
    """
    
    def __init__(self, output_dir: str = None):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save visualizations (optional)
        """
        self.output_dir = output_dir
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
        
        # Define colors for visualization
        self.colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (128, 0, 0),    # Maroon
            (0, 128, 0),    # Dark Green
            (0, 0, 128),    # Navy
            (128, 128, 0),  # Olive
            (128, 0, 128),  # Purple
            (0, 128, 128),  # Teal
            (255, 165, 0),  # Orange
            (255, 192, 203),# Pink
            (165, 42, 42),  # Brown
            (0, 255, 127)   # Spring Green
        ]
    
    def visualize_2d_tracks(self, 
                           images: List[np.ndarray], 
                           tracks_2d: torch.Tensor, 
                           visibility: Optional[torch.Tensor] = None,
                           infront_cameras: Optional[torch.Tensor] = None,
                           frame_idx: int = 0, 
                           track_history: int = 16, 
                           point_size: int = 7, 
                           line_thickness: int = 2,
                           show_occ: bool = True,
                           max_points: Optional[int] = None) -> np.ndarray:
        """
        Visualize 2D tracks on images using the same style as tap3d_combined_viz.py.
        
        Args:
            images: List of images
            tracks_2d: 2D tracks of shape (batch_size, num_frames, num_points, 2)
            visibility: Visibility flags of shape (batch_size, num_frames, num_points)
            infront_cameras: Boolean flags indicating if points are in front of cameras
            frame_idx: Frame index to visualize
            track_history: Number of frames to show in track history
            point_size: Size of points
            line_thickness: Thickness of lines
            show_occ: Whether to show occluded points
            max_points: Maximum number of points to visualize
            
        Returns:
            Visualized image
        """
        # Get data from batch
        tracks = tracks_2d[0]  # Remove batch dimension
        
        # Get image for current frame
        if isinstance(images, list) and len(images) > frame_idx:
            img = images[frame_idx].copy()
        elif isinstance(images, np.ndarray) and images.shape[0] > frame_idx:
            img = images[frame_idx].copy()
        else:
            raise ValueError("Invalid images format or frame_idx out of bounds")
        
        # Get image dimensions
        h, w = img.shape[:2]
        
        # Get dimensions
        num_frames, num_points, _ = tracks.shape
        
        # Limit the number of points if specified
        if max_points is not None:
            num_points = min(num_points, max_points)
            
        # Ensure frame_idx is valid
        frame_idx = min(frame_idx, num_frames - 1)
        
        # Move tensor to CPU and convert to numpy
        tracks_np = tracks.detach().cpu().numpy()
        
        # Get visibility if provided
        if visibility is not None:
            visibility_np = visibility[0].detach().cpu().numpy()
        else:
            visibility_np = np.ones((num_frames, num_points), dtype=bool)
            
        # Use infront_cameras if provided, otherwise assume all points are in front of camera
        if infront_cameras is not None:
            infront_np = infront_cameras[0].detach().cpu().numpy()
        else:
            infront_np = np.ones_like(visibility_np).astype(bool)
        
        # Use bright colors for better visibility - exactly matching tap3d_combined_viz.py
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (255, 128, 0),  # Orange
            (128, 0, 255),  # Purple
            (0, 255, 128)   # Lime
        ]
        
        # Draw tracks (lines from previous frames)
        start_frame = max(0, frame_idx - track_history)
        
        for i in range(num_points):
            color = colors[i % len(colors)]
            
            for f in range(start_frame, frame_idx):
                # Calculate line thickness based on position in track history
                # Thickness increases as we get closer to the current frame
                progress = (f - start_frame + 1) / (frame_idx - start_frame + 1)  # 0.0 to 1.0
                current_thickness = max(1, int(line_thickness * progress))  # Ensure minimum thickness of 1
                
                if visibility_np[f, i] and visibility_np[f+1, i]:
                    x1, y1 = int(round(tracks_np[f, i, 0])), int(round(tracks_np[f, i, 1]))
                    x2, y2 = int(round(tracks_np[f+1, i, 0])), int(round(tracks_np[f+1, i, 1]))
                    
                    # Calculate distance between consecutive points
                    distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    
                    # Only draw line if distance is reasonable (prevents trajectory breaks)
                    # For 224x224 images, a reasonable threshold might be 20-30% of image width
                    max_distance = 0.3 * w  # 30% of image width
                    
                    # Check if points are within image bounds and not too far apart
                    if (0 <= x1 < w and 0 <= y1 < h and 
                        0 <= x2 < w and 0 <= y2 < h and
                        distance < max_distance):
                        cv2.line(img, (x1, y1), (x2, y2), color, current_thickness)
                elif show_occ and infront_np[f, i] and infront_np[f+1, i]:
                    x1, y1 = int(round(tracks_np[f, i, 0])), int(round(tracks_np[f, i, 1]))
                    x2, y2 = int(round(tracks_np[f+1, i, 0])), int(round(tracks_np[f+1, i, 1]))
                    
                    # Calculate distance between consecutive points
                    distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    max_distance = 0.3 * w  # 30% of image width
                    
                    # Check if points are within image bounds and not too far apart
                    if (0 <= x1 < w and 0 <= y1 < h and 
                        0 <= x2 < w and 0 <= y2 < h and
                        distance < max_distance):
                        cv2.line(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)  # Thinner line for occluded points
        
        # Draw each point with a unique color
        for i in range(num_points):
            # Skip if we've reached max_points
            if max_points is not None and i >= max_points:
                break
                
            # Get color for this point
            color = self.colors[i % len(self.colors)]
            
            # Get current point position
            if visibility_np[frame_idx, i]:
                x, y = int(round(tracks_np[frame_idx, i, 0])), int(round(tracks_np[frame_idx, i, 1]))
                
                # Filter out points at the origin or corners (likely errors)
                # Skip points at (0,0) or very close to corners
                corner_margin = 10
                is_at_corner = ((x < corner_margin and y < corner_margin) or  # Top-left
                               (x < corner_margin and y > h - corner_margin) or  # Bottom-left
                               (x > w - corner_margin and y < corner_margin) or  # Top-right
                               (x > w - corner_margin and y > h - corner_margin))  # Bottom-right
                
                # Check if point is within image bounds and not at corners
                if 0 <= x < w and 0 <= y < h and not is_at_corner:
                    cv2.circle(img, (x, y), point_size, color, -1)  # Filled circle
                    
            # Draw occluded points if requested
            elif show_occ and infront_np is not None and infront_np[frame_idx, i]:
                x, y = int(round(tracks_np[frame_idx, i, 0])), int(round(tracks_np[frame_idx, i, 1]))
                
                # Filter out points at the origin or corners (likely errors)
                corner_margin = 10
                is_at_corner = ((x < corner_margin and y < corner_margin) or  # Top-left
                               (x < corner_margin and y > h - corner_margin) or  # Bottom-left
                               (x > w - corner_margin and y < corner_margin) or  # Top-right
                               (x > w - corner_margin and y > h - corner_margin))  # Bottom-right
                
                # Check if point is within image bounds and not at corners
                if 0 <= x < w and 0 <= y < h and not is_at_corner:
                    cv2.circle(img, (x, y), point_size-2, color, 1)  # Hollow circle for occluded points       
        return img
    
    def visualize_3d_tracks(self, 
                           tracks_3d: torch.Tensor, 
                           visibility: Optional[torch.Tensor] = None,
                           frame_idx: int = 0, 
                           track_history: int = 16) -> plt.Figure:
        """
        Visualize 3D tracks.
        
        Args:
            tracks_3d: 3D tracks of shape (batch_size, num_frames, num_points, 3)
            visibility: Visibility flags of shape (batch_size, num_frames, num_points)
            frame_idx: Frame index to visualize
            track_history: Number of frames to show in track history
            
        Returns:
            Matplotlib figure with 3D tracks
        """
        # Get data from batch
        tracks = tracks_3d[0]  # Remove batch dimension
        
        # Get dimensions
        num_frames, num_points, _ = tracks.shape
        
        # Move tensor to CPU and convert to numpy
        tracks_np = tracks.detach().cpu().numpy()
        
        # Get visibility if provided
        if visibility is not None:
            visibility_np = visibility[0].detach().cpu().numpy()
        else:
            visibility_np = np.ones((num_frames, num_points), dtype=bool)
        
        # Debug: Check for points near bottom right corner
        print(f"\nDEBUGGING 3D TRACKS:\n{'='*30}")
        print(f"Total frames: {num_frames}, points: {num_points}")
        
        # Create figure
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Calculate start frame for track history
        start_frame = max(1, frame_idx - track_history)  # Start from frame 1 (not 0) to skip initialization
        
        # Draw tracks with history - using the approach from tap3d_combined_viz.py
        for p in range(num_points):
            # Get color for this point (convert to matplotlib format)
            color = tuple(c / 255.0 for c in self.colors[p % len(self.colors)])
            
            # Draw track history
            points_to_plot = []
            
            for f in range(start_frame, frame_idx + 1):
                if visibility_np[f, p]:
                    # Make sure the point isn't zero (as we zeroed out frame 0 earlier)
                    point = tracks_np[f, p]
                    if not np.allclose(point, 0):
                        points_to_plot.append(point)
            
            if len(points_to_plot) > 1:  # Need at least 2 points to draw a line
                points_to_plot = np.array(points_to_plot)
                
                # Draw track history with variable line thickness
                # Calculate line segments with increasing thickness
                for i in range(len(points_to_plot) - 1):
                    # Calculate thickness based on position in track history
                    # Thickness increases as we get closer to the current frame
                    progress = (i + 1) / len(points_to_plot)  # 0.0 to 1.0
                    thickness = 0.5 + 2.5 * progress  # Thickness from 0.5 to 3.0
                    
                    # Draw line segment with calculated thickness
                    ax.plot([points_to_plot[i, 0], points_to_plot[i+1, 0]],
                            [points_to_plot[i, 1], points_to_plot[i+1, 1]],
                            [points_to_plot[i, 2], points_to_plot[i+1, 2]],
                            color=color, alpha=0.7, linewidth=thickness)
                
                # Draw current point
                if visibility_np[frame_idx, p] and not np.allclose(tracks_np[frame_idx, p], 0):
                    current_point = tracks_np[frame_idx, p]
                    ax.scatter(current_point[0], current_point[1], current_point[2],
                              color=color, s=70, edgecolors='black')
        
        # Collect data points for ranges
        all_x, all_y, all_z = [], [], []
        for p in range(num_points):
            if visibility_np[frame_idx, p] and not np.allclose(tracks_np[frame_idx, p], 0):
                point = tracks_np[frame_idx, p]
                all_x.append(point[0])
                all_y.append(point[1])
                all_z.append(point[2])
        
        # Print ranges
        if all_x and all_y and all_z:
            print(f"X range: {min(all_x)} to {max(all_x)}")
            print(f"Y range: {min(all_y)} to {max(all_y)}")
            print(f"Z range: {min(all_z)} to {max(all_z)}")
        
        # Set equal aspect ratio with equal scaling
        ax.set_box_aspect([1, 1, 1])
        
        # Remove grid and ticks for cleaner visualization
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        
        # Set title
        # ax.set_title(f'3D Tracks - Frame {frame_idx}')
        
        return fig
    
    def visualize_correspondences(self, 
                                 images: List[List[np.ndarray]], 
                                 correspondence_result: Dict,
                                 frame_idx: int = 0) -> List[np.ndarray]:
        """
        Visualize track correspondences across views.
        
        Args:
            images: List of image sequences for each view
            correspondence_result: Dictionary with correspondence information
            frame_idx: Frame index to visualize
            
        Returns:
            List of visualized images with correspondences
        """
        # Extract correspondences
        correspondences = correspondence_result["correspondences"][frame_idx]
        tracks_2d = correspondence_result["tracks_2d"]
        
        # Number of views
        num_views = len(images)
        
        # Create copies of images for visualization
        vis_images = [images[i][frame_idx].copy() for i in range(num_views)]
        
        # Draw correspondences
        for i in range(num_views):
            for j in range(num_views):
                if i == j or correspondences[i][j] is None:
                    continue
                
                # Get correspondence matrix
                corr_matrix = correspondences[i][j]
                
                # Get 2D tracks for both views
                tracks_i = tracks_2d[i][0, frame_idx].detach().cpu().numpy()
                tracks_j = tracks_2d[j][0, frame_idx].detach().cpu().numpy()
                
                # Find matches using argmax along dimension 2
                matches_i_to_j = torch.argmax(corr_matrix, dim=2)[0].detach().cpu().numpy()
                
                # Draw matches
                for p_i, p_j in enumerate(matches_i_to_j):
                    # Get points in both views
                    pt_i = tuple(tracks_i[p_i].astype(int))
                    pt_j = tuple(tracks_j[p_j].astype(int))
                    
                    # Get color for this match
                    color = self.colors[p_i % len(self.colors)]
                    
                    # Draw points in both views
                    cv2.circle(vis_images[i], pt_i, 7, color, -1)
                    cv2.circle(vis_images[j], pt_j, 7, color, -1)
                    
                    # Add text with match index
                    cv2.putText(vis_images[i], str(p_i), 
                               (pt_i[0] + 10, pt_i[1] + 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    cv2.putText(vis_images[j], str(p_i), 
                               (pt_j[0] + 10, pt_j[1] + 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return vis_images
    
    def visualize_results(self, 
                         result: Dict, 
                         frame_idx: int = 0, 
                         track_history: int = 16,
                         save: bool = True,
                         target_size: Tuple[int, int] = (448, 448),
                         point_size: int = 9,
                         line_thickness: int = 2,
                         max_points: int = 50,
                         show_occ: bool = True,
                         create_combined: bool = True) -> Dict:
        """
        Visualize all results from the LAPA pipeline using the style from tap3d_combined_viz.py.
        
        Args:
            result: Pipeline result dictionary
            frame_idx: Frame index to visualize
            track_history: Number of frames to show in track history
            save: Whether to save visualizations to disk
            target_size: Target size for image resizing (448x448 by default)
            point_size: Size of points to draw
            line_thickness: Thickness of lines to draw
            max_points: Maximum number of points to visualize
            show_occ: Whether to show occluded points
            create_combined: Whether to create a combined 4x1 visualization
            
        Returns:
            Dictionary with visualization images
        """
        vis_results = {}
        
        # Extract data
        data = result["data"]
        tracks_2d = result["tracks_2d"]
        tracks_3d = result["tracks_3d"]
        correspondences = result["correspondences"]
        
        # Get views
        views = data["views"]
        view_names = list(views.keys())
        camera_matrices = data.get("camera_matrices", {})
        
        print(f"Visualizing results for {len(view_names)} views at frame {frame_idx}...")
        
        # Decode images for each view with proper scaling
        images_by_view = {}
        scaled_intrinsics_by_view = {}
        infront_cameras_by_view = {}
        
        for view_name in view_names:
            base_view_name = view_name.replace('.npz', '')
            view_data = views[view_name]
            
            if 'images_jpeg_bytes' in view_data:
                # Decode and resize images
                images_jpeg = view_data['images_jpeg_bytes']
                images = []
                for img_bytes in images_jpeg:
                    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Get original dimensions
                    orig_height, orig_width = img.shape[:2]
                    
                    # Resize if target_size is specified
                    if target_size is not None:
                        img = cv2.resize(img, target_size)
                    
                    images.append(img)
                
                images_by_view[base_view_name] = images
                
                # Scale intrinsics properly - critical for correct visualization
                intrinsics = view_data['intrinsics']
                fx, fy, cx, cy = intrinsics
                
                width_scale = target_size[0] / orig_width
                height_scale = target_size[1] / orig_height
                
                scaled_fx = fx * width_scale
                scaled_fy = fy * height_scale
                scaled_cx = cx * width_scale
                scaled_cy = cy * height_scale
                
                scaled_intrinsics = np.array([scaled_fx, scaled_fy, scaled_cx, scaled_cy])
                scaled_intrinsics_by_view[base_view_name] = scaled_intrinsics
                
                print(f"View {base_view_name}: Original size {orig_width}x{orig_height}, scaled to {target_size}")
                print(f"View {base_view_name}: Original intrinsics: {intrinsics}")
                print(f"View {base_view_name}: Scaled intrinsics: {scaled_intrinsics}")
                
                # Calculate infront_cameras
                if base_view_name in camera_matrices:
                    points_3d = view_data['tracks_xyz']
                    camera_matrix = camera_matrices[base_view_name]
                    
                    from lapa.data.tap3d_loader import TAP3DLoader
                    loader = TAP3DLoader(None)
                    infront_cameras = loader.calculate_infront_cameras(points_3d, camera_matrix)
                    infront_cameras_by_view[base_view_name] = infront_cameras
                    
                    print(f"View {base_view_name}: Calculated infront_cameras with shape {infront_cameras.shape}")
        
        # Visualize 2D tracks for each view
        tracks_2d_vis = {}
        for i, view_name in enumerate(view_names):
            base_view_name = view_name.replace('.npz', '')
            if base_view_name in images_by_view:
                # Get visibility if available
                visibility = None
                if 'visibility' in views[view_name]:
                    visibility = torch.from_numpy(views[view_name]['visibility']).unsqueeze(0)
                
                # Get infront_cameras if available
                infront_cameras = None
                if base_view_name in infront_cameras_by_view:
                    infront_cameras = torch.from_numpy(infront_cameras_by_view[base_view_name]).unsqueeze(0)
                
                # Visualize 2D tracks with proper parameters
                try:
                    vis_img = self.visualize_2d_tracks(
                        images_by_view[base_view_name],
                        tracks_2d[i],
                        visibility,
                        infront_cameras,
                        frame_idx,
                        track_history,
                        point_size,
                        line_thickness,
                        show_occ,
                        max_points
                    )
                    
                    tracks_2d_vis[base_view_name] = vis_img
                    
                    # Save visualization
                    if save and self.output_dir is not None:
                        out_path = os.path.join(self.output_dir, f"{base_view_name}_2d_tracks_frame_{frame_idx}.jpg")
                        cv2.imwrite(out_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
                        print(f"Saved 2D visualization to {out_path}")
                except Exception as e:
                    print(f"Error visualizing 2D tracks for view {base_view_name}: {e}")
        
        vis_results["tracks_2d"] = tracks_2d_vis
        
        # Visualize 3D tracks
        try:
            # Get visibility from result
            visibility = None
            if "visibility" in result and result["visibility"] is not None:
                visibility = result["visibility"]
            
            fig_3d = self.visualize_3d_tracks(
                result["tracks_3d"], 
                visibility,
                frame_idx=frame_idx,
                track_history=track_history
            )
            
            vis_results["tracks_3d"] = fig_3d
            
            # Save visualization
            if save and self.output_dir is not None:
                out_path = os.path.join(self.output_dir, f"3d_tracks_frame_{frame_idx}.jpg")
                fig_3d.savefig(out_path, bbox_inches='tight')
                print(f"Saved 3D visualization to {out_path}")
        except Exception as e:
            print(f"Error visualizing 3D tracks: {e}")
        
        # Visualize correspondences
        try:
            all_images = [images_by_view[view_name.replace('.npz', '')] for view_name in view_names 
                          if view_name.replace('.npz', '') in images_by_view]
            
            corr_result = {
                "correspondences": correspondences,
                "tracks_2d": tracks_2d
            }
            
            corr_vis = self.visualize_correspondences(
                all_images,
                corr_result,
                frame_idx
            )
            
            vis_results["correspondences"] = corr_vis
            
            # Save visualization
            if save and self.output_dir is not None:
                for i, view_name in enumerate(view_names):
                    base_view_name = view_name.replace('.npz', '')
                    if i < len(corr_vis):
                        out_path = os.path.join(self.output_dir, f"{base_view_name}_correspondences_frame_{frame_idx}.jpg")
                        cv2.imwrite(out_path, cv2.cvtColor(corr_vis[i], cv2.COLOR_RGB2BGR))
                        print(f"Saved correspondence visualization to {out_path}")
        except Exception as e:
            print(f"Error visualizing correspondences: {e}")
        
        # Create combined 4x1 visualization if requested
        if create_combined and len(tracks_2d_vis) > 0:
            try:
                # Get all 2D track visualizations
                view_2d_images = []
                for view_name in view_names:
                    base_view_name = view_name.replace('.npz', '')
                    if base_view_name in tracks_2d_vis:
                        view_2d_images.append(tracks_2d_vis[base_view_name])
                
                # Make sure we have at least one 2D visualization
                if len(view_2d_images) > 0:
                    # Get 3D visualization as image
                    if 'tracks_3d' in vis_results:
                        # Convert matplotlib figure to image
                        fig_3d = vis_results['tracks_3d']
                        canvas = FigureCanvasAgg(fig_3d)
                        canvas.draw()
                        
                        # Convert to numpy array
                        fig_w, fig_h = fig_3d.get_size_inches() * fig_3d.get_dpi()
                        viz_3d = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(int(fig_h), int(fig_w), 4)
                        viz_3d = cv2.cvtColor(viz_3d, cv2.COLOR_RGBA2RGB)
                        
                        # Resize all images to the same height
                        h, w = view_2d_images[0].shape[:2]
                        resized_2d_images = []
                        
                        for img in view_2d_images:
                            resized_2d_images.append(cv2.resize(img, (w, h)))
                        
                        # Resize 3D visualization to match 2D image height
                        viz_3d = cv2.resize(viz_3d, (int(h * viz_3d.shape[1] / viz_3d.shape[0]), h))
                        
                        # Combine images horizontally
                        combined_img = np.hstack(resized_2d_images + [viz_3d])
                        
                        # Add to results
                        vis_results['combined'] = combined_img
                        
                        # Save combined visualization
                        if save and self.output_dir is not None:
                            view_str = '_'.join([name.replace('.npz', '').replace('boxes_', '') for name in view_names])
                            combined_path = os.path.join(self.output_dir, f'combined_viz_{view_str}_frame{frame_idx}.png')
                            cv2.imwrite(combined_path, cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR))
                            print(f'Saved combined visualization to {combined_path}')
            except Exception as e:
                print(f'Error creating combined visualization: {e}')
        
        return vis_results
    
    def decode_images(self, images_jpeg_bytes: np.ndarray) -> Tuple[List[np.ndarray], Tuple[int, int]]:
        """
        Decode JPEG bytes to RGB images.
        
        Args:
            images_jpeg_bytes: Array of JPEG bytes
            
        Returns:
            Tuple containing:
            - List of RGB images
            - Tuple with original image dimensions (width, height)
        """
        images = []
        orig_size = None
        
        for jpeg_bytes in images_jpeg_bytes:
            # Decode JPEG bytes to image
            img = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
            # Convert from BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Get original dimensions
            if orig_size is None:
                orig_height, orig_width = img.shape[:2]
                orig_size = (orig_width, orig_height)
            
            images.append(img)
        
        return images, orig_size
