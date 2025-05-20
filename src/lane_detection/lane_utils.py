"""
Lane Detection Utilities

Helper functions and utilities for lane detection processing.
"""

import cv2
import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt


class LaneUtils:
    """Utility class for lane detection operations."""
    
    @staticmethod
    def create_perspective_transform_matrix(image_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create perspective transform matrix for bird's eye view.
        
        Args:
            image_shape: (height, width) of the image
            
        Returns:
            Tuple of (transform_matrix, inverse_transform_matrix)
        """
        height, width = image_shape
        
        # Source points (trapezoid in original image)
        src_points = np.float32([
            [width * 0.2, height],           # bottom left
            [width * 0.4, height * 0.6],    # top left
            [width * 0.6, height * 0.6],    # top right
            [width * 0.8, height]           # bottom right
        ])
        
        # Destination points (rectangle in bird's eye view)
        dst_points = np.float32([
            [width * 0.2, height],          # bottom left
            [width * 0.2, 0],               # top left
            [width * 0.8, 0],               # top right
            [width * 0.8, height]           # bottom right
        ])
        
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        M_inv = cv2.getPerspectiveTransform(dst_points, src_points)
        
        return M, M_inv
    
    @staticmethod
    def apply_perspective_transform(image: np.ndarray, 
                                   transform_matrix: np.ndarray) -> np.ndarray:
        """
        Apply perspective transformation to image.
        
        Args:
            image: Input image
            transform_matrix: Perspective transformation matrix
            
        Returns:
            Transformed image
        """
        height, width = image.shape[:2]
        return cv2.warpPerspective(image, transform_matrix, (width, height))
    
    @staticmethod
    def calculate_lane_curvature(lane_points: np.ndarray, 
                                ym_per_pix: float = 30/720, 
                                xm_per_pix: float = 3.7/700) -> float:
        """
        Calculate lane curvature in meters.
        
        Args:
            lane_points: Array of (x, y) points defining the lane
            ym_per_pix: Meters per pixel in y direction
            xm_per_pix: Meters per pixel in x direction
            
        Returns:
            Radius of curvature in meters
        """
        if len(lane_points) < 3:
            return float('inf')
        
        # Extract coordinates
        x_coords = lane_points[:, 0] * xm_per_pix
        y_coords = lane_points[:, 1] * ym_per_pix
        
        # Fit polynomial in world space
        poly_coeffs = np.polyfit(y_coords, x_coords, 2)
        A, B, C = poly_coeffs
        
        # Calculate curvature at the bottom of the image
        y_eval = np.max(y_coords)
        curvature = ((1 + (2*A*y_eval + B)**2)**1.5) / np.absolute(2*A)
        
        return curvature
    
    @staticmethod
    def calculate_vehicle_position(left_lane_points: np.ndarray,
                                  right_lane_points: np.ndarray,
                                  image_width: int,
                                  xm_per_pix: float = 3.7/700) -> float:
        """
        Calculate vehicle position relative to lane center.
        
        Args:
            left_lane_points: Points defining left lane
            right_lane_points: Points defining right lane
            image_width: Width of the image
            xm_per_pix: Meters per pixel in x direction
            
        Returns:
            Vehicle position offset from center in meters (negative = left, positive = right)
        """
        if len(left_lane_points) == 0 or len(right_lane_points) == 0:
            return 0.0
        
        # Get bottom points of lanes
        left_bottom_x = left_lane_points[-1, 0]
        right_bottom_x = right_lane_points[-1, 0]
        
        # Calculate lane center
        lane_center = (left_bottom_x + right_bottom_x) / 2
        
        # Calculate vehicle center (assumed to be image center)
        vehicle_center = image_width / 2
        
        # Calculate offset in meters
        offset = (vehicle_center - lane_center) * xm_per_pix
        
        return offset
    
    @staticmethod
    def smooth_lane_detection(current_lanes: List[np.ndarray],
                            previous_lanes: List[List[np.ndarray]],
                            smoothing_factor: float = 0.8) -> List[np.ndarray]:
        """
        Apply temporal smoothing to lane detection results.
        
        Args:
            current_lanes: Current frame lane detection results
            previous_lanes: List of previous frame results
            smoothing_factor: Weight for current frame vs. history
            
        Returns:
            Smoothed lane detection results
        """
        if not previous_lanes:
            return current_lanes
        
        smoothed_lanes = []
        
        for i, current_lane in enumerate(current_lanes):
            if len(previous_lanes) > i and len(previous_lanes[i]) > 0:
                # Average with previous detections
                prev_lane = previous_lanes[i][-1]  # Most recent
                
                if len(current_lane) == len(prev_lane):
                    smoothed_lane = (smoothing_factor * current_lane + 
                                   (1 - smoothing_factor) * prev_lane)
                    smoothed_lanes.append(smoothed_lane)
                else:
                    smoothed_lanes.append(current_lane)
            else:
                smoothed_lanes.append(current_lane)
        
        return smoothed_lanes
    
    @staticmethod
    def visualize_detection_pipeline(image: np.ndarray,
                                   edges: np.ndarray,
                                   masked: np.ndarray,
                                   result: np.ndarray) -> plt.Figure:
        """
        Create visualization of the detection pipeline.
        
        Args:
            image: Original image
            edges: Edge detection result
            masked: Masked image
            result: Final result with lanes
            
        Returns:
            Matplotlib figure with subplots
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Edge detection
        axes[0, 1].imshow(edges, cmap='gray')
        axes[0, 1].set_title('Edge Detection')
        axes[0, 1].axis('off')
        
        # Masked image
        axes[1, 0].imshow(masked, cmap='gray')
        axes[1, 0].set_title('Region of Interest')
        axes[1, 0].axis('off')
        
        # Final result
        axes[1, 1].imshow(result)
        axes[1, 1].set_title('Lane Detection Result')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def validate_lane_detection(left_lane: np.ndarray,
                              right_lane: np.ndarray,
                              image_width: int,
                              min_lane_width: float = 0.3,
                              max_lane_width: float = 0.8) -> bool:
        """
        Validate that detected lanes are reasonable.
        
        Args:
            left_lane: Left lane points
            right_lane: Right lane points
            image_width: Image width in pixels
            min_lane_width: Minimum lane width as fraction of image width
            max_lane_width: Maximum lane width as fraction of image width
            
        Returns:
            True if lanes are valid, False otherwise
        """
        if len(left_lane) == 0 or len(right_lane) == 0:
            return False
        
        # Check lane width
        left_bottom_x = left_lane[-1, 0]
        right_bottom_x = right_lane[-1, 0]
        lane_width = abs(right_bottom_x - left_bottom_x) / image_width
        
        if lane_width < min_lane_width or lane_width > max_lane_width:
            return False
        
        # Check lane order (left should be left of right)
        if left_bottom_x >= right_bottom_x:
            return False
        
        return True