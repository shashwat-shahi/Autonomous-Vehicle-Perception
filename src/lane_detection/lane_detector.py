"""
Lane Detection Implementation

This module implements lane detection using classical computer vision techniques
and deep learning approaches.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt


class LaneDetector:
    """
    Lane detection system using computer vision techniques.
    
    Combines traditional image processing (Canny edge detection, Hough transforms)
    with perspective transformation for robust lane detection.
    """
    
    def __init__(self, 
                 canny_low_threshold: int = 50,
                 canny_high_threshold: int = 150,
                 hough_rho: int = 2,
                 hough_theta: float = np.pi/180,
                 hough_threshold: int = 15,
                 hough_min_line_len: int = 40,
                 hough_max_line_gap: int = 20):
        """
        Initialize lane detector with configurable parameters.
        
        Args:
            canny_low_threshold: Lower threshold for Canny edge detection
            canny_high_threshold: Upper threshold for Canny edge detection
            hough_rho: Distance resolution in pixels for Hough transform
            hough_theta: Angle resolution in radians for Hough transform
            hough_threshold: Accumulator threshold for Hough transform
            hough_min_line_len: Minimum line length for Hough transform
            hough_max_line_gap: Maximum gap between line segments for Hough transform
        """
        self.canny_low = canny_low_threshold
        self.canny_high = canny_high_threshold
        self.hough_rho = hough_rho
        self.hough_theta = hough_theta
        self.hough_threshold = hough_threshold
        self.hough_min_line_len = hough_min_line_len
        self.hough_max_line_gap = hough_max_line_gap
        
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for lane detection.
        
        Args:
            image: Input RGB image
            
        Returns:
            Preprocessed grayscale image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        return blur
    
    def apply_roi_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Apply region of interest mask to focus on lane area.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Masked image
        """
        height, width = image.shape
        
        # Define region of interest polygon (trapezoid focusing on road)
        vertices = np.array([
            [0, height],
            [width * 0.4, height * 0.6],
            [width * 0.6, height * 0.6],
            [width, height]
        ], dtype=np.int32)
        
        # Create mask
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, [vertices], 255)
        
        # Apply mask
        masked_image = cv2.bitwise_and(image, mask)
        
        return masked_image
    
    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        """
        Detect edges using Canny edge detector.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Edge-detected binary image
        """
        edges = cv2.Canny(image, self.canny_low, self.canny_high)
        return edges
    
    def detect_lines(self, edges: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect lines using Hough transform.
        
        Args:
            edges: Edge-detected binary image
            
        Returns:
            Array of detected lines or None if no lines found
        """
        lines = cv2.HoughLinesP(
            edges,
            self.hough_rho,
            self.hough_theta,
            self.hough_threshold,
            minLineLength=self.hough_min_line_len,
            maxLineGap=self.hough_max_line_gap
        )
        
        return lines
    
    def separate_left_right_lanes(self, lines: np.ndarray, 
                                  image_width: int) -> Tuple[List, List]:
        """
        Separate detected lines into left and right lane candidates.
        
        Args:
            lines: Array of detected lines
            image_width: Width of the image
            
        Returns:
            Tuple of (left_lines, right_lines)
        """
        left_lines = []
        right_lines = []
        
        if lines is None:
            return left_lines, right_lines
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate slope
            if x2 - x1 == 0:  # Vertical line
                continue
                
            slope = (y2 - y1) / (x2 - x1)
            
            # Filter by slope and position
            if slope < -0.5 and x1 < image_width * 0.6:  # Left lane
                left_lines.append(line[0])
            elif slope > 0.5 and x1 > image_width * 0.4:  # Right lane
                right_lines.append(line[0])
        
        return left_lines, right_lines
    
    def fit_lane_polynomial(self, lines: List) -> Optional[np.poly1d]:
        """
        Fit a polynomial to lane lines.
        
        Args:
            lines: List of line segments
            
        Returns:
            Polynomial fit or None if insufficient data
        """
        if len(lines) < 2:
            return None
        
        # Extract points from lines
        x_coords = []
        y_coords = []
        
        for line in lines:
            x1, y1, x2, y2 = line
            x_coords.extend([x1, x2])
            y_coords.extend([y1, y2])
        
        # Fit polynomial (degree 1 for straight lines, degree 2 for curves)
        if len(x_coords) >= 2:
            poly = np.polyfit(y_coords, x_coords, 1)
            return np.poly1d(poly)
        
        return None
    
    def draw_lanes(self, image: np.ndarray, 
                   left_poly: Optional[np.poly1d],
                   right_poly: Optional[np.poly1d]) -> np.ndarray:
        """
        Draw detected lanes on the image.
        
        Args:
            image: Original RGB image
            left_poly: Left lane polynomial
            right_poly: Right lane polynomial
            
        Returns:
            Image with lanes drawn
        """
        result = image.copy()
        height, width = image.shape[:2]
        
        # Create lane overlay
        lane_overlay = np.zeros_like(image)
        
        # Generate y coordinates for drawing
        y_coords = np.linspace(height * 0.6, height, 100)
        
        # Draw left lane
        if left_poly is not None:
            x_left = left_poly(y_coords)
            # Filter valid coordinates
            valid_indices = (x_left >= 0) & (x_left < width)
            x_left = x_left[valid_indices]
            y_left = y_coords[valid_indices]
            
            if len(x_left) > 1:
                points = np.column_stack((x_left.astype(int), y_left.astype(int)))
                cv2.polylines(lane_overlay, [points], False, (255, 0, 0), 10)
        
        # Draw right lane
        if right_poly is not None:
            x_right = right_poly(y_coords)
            # Filter valid coordinates
            valid_indices = (x_right >= 0) & (x_right < width)
            x_right = x_right[valid_indices]
            y_right = y_coords[valid_indices]
            
            if len(x_right) > 1:
                points = np.column_stack((x_right.astype(int), y_right.astype(int)))
                cv2.polylines(lane_overlay, [points], False, (0, 0, 255), 10)
        
        # Fill lane area if both lanes detected
        if left_poly is not None and right_poly is not None:
            y_fill = np.linspace(height * 0.6, height, 50)
            x_left_fill = left_poly(y_fill)
            x_right_fill = right_poly(y_fill)
            
            # Create lane fill points
            left_points = np.column_stack((x_left_fill.astype(int), y_fill.astype(int)))
            right_points = np.column_stack((x_right_fill.astype(int), y_fill.astype(int)))
            
            # Combine points for polygon
            lane_points = np.concatenate([left_points, right_points[::-1]])
            cv2.fillPoly(lane_overlay, [lane_points], (0, 255, 0))
        
        # Combine with original image
        result = cv2.addWeighted(result, 0.8, lane_overlay, 0.2, 0)
        
        return result
    
    def detect(self, image: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Detect lanes in the input image.
        
        Args:
            image: Input RGB image
            
        Returns:
            Tuple of (result_image, detection_info)
        """
        # Preprocess image
        preprocessed = self.preprocess_image(image)
        
        # Apply ROI mask
        masked = self.apply_roi_mask(preprocessed)
        
        # Detect edges
        edges = self.detect_edges(masked)
        
        # Detect lines
        lines = self.detect_lines(edges)
        
        # Separate left and right lanes
        left_lines, right_lines = self.separate_left_right_lanes(lines, image.shape[1])
        
        # Fit polynomials
        left_poly = self.fit_lane_polynomial(left_lines)
        right_poly = self.fit_lane_polynomial(right_lines)
        
        # Draw lanes on image
        result = self.draw_lanes(image, left_poly, right_poly)
        
        # Prepare detection info
        detection_info = {
            'left_lane_detected': left_poly is not None,
            'right_lane_detected': right_poly is not None,
            'num_left_lines': len(left_lines),
            'num_right_lines': len(right_lines),
            'edges': edges,
            'masked_image': masked
        }
        
        return result, detection_info
    
    def detect_from_path(self, image_path: str) -> Tuple[np.ndarray, dict]:
        """
        Detect lanes from image file path.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Tuple of (result_image, detection_info)
        """
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return self.detect(image)