"""
Image Processing Utilities

Common image processing functions for autonomous vehicle perception.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt
from PIL import Image


class ImageUtils:
    """Collection of image processing utilities."""
    
    @staticmethod
    def resize_image(image: np.ndarray, target_size: Tuple[int, int], 
                    maintain_aspect: bool = True) -> np.ndarray:
        """
        Resize image to target size.
        
        Args:
            image: Input image
            target_size: (width, height) target size
            maintain_aspect: Whether to maintain aspect ratio
            
        Returns:
            Resized image
        """
        if maintain_aspect:
            h, w = image.shape[:2]
            target_w, target_h = target_size
            
            # Calculate scaling factor
            scale = min(target_w / w, target_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            
            # Resize image
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # Pad to target size
            top = (target_h - new_h) // 2
            bottom = target_h - new_h - top
            left = (target_w - new_w) // 2
            right = target_w - new_w - left
            
            result = cv2.copyMakeBorder(resized, top, bottom, left, right, 
                                      cv2.BORDER_CONSTANT, value=[0, 0, 0])
        else:
            result = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
        
        return result
    
    @staticmethod
    def normalize_image(image: np.ndarray, mean: List[float] = [0.485, 0.456, 0.406],
                       std: List[float] = [0.229, 0.224, 0.225]) -> np.ndarray:
        """
        Normalize image using ImageNet statistics.
        
        Args:
            image: Input image in range [0, 255]
            mean: Channel means
            std: Channel standard deviations
            
        Returns:
            Normalized image
        """
        # Convert to float and normalize to [0, 1]
        normalized = image.astype(np.float32) / 255.0
        
        # Apply mean and std normalization
        mean = np.array(mean).reshape(1, 1, 3)
        std = np.array(std).reshape(1, 1, 3)
        
        normalized = (normalized - mean) / std
        
        return normalized
    
    @staticmethod
    def denormalize_image(image: np.ndarray, mean: List[float] = [0.485, 0.456, 0.406],
                         std: List[float] = [0.229, 0.224, 0.225]) -> np.ndarray:
        """
        Denormalize image back to [0, 255] range.
        
        Args:
            image: Normalized image
            mean: Channel means used for normalization
            std: Channel standard deviations used for normalization
            
        Returns:
            Denormalized image in [0, 255] range
        """
        mean = np.array(mean).reshape(1, 1, 3)
        std = np.array(std).reshape(1, 1, 3)
        
        # Reverse normalization
        denormalized = image * std + mean
        
        # Clip to [0, 1] and convert to [0, 255]
        denormalized = np.clip(denormalized, 0, 1) * 255
        
        return denormalized.astype(np.uint8)
    
    @staticmethod
    def apply_brightness_contrast(image: np.ndarray, brightness: float = 0,
                                 contrast: float = 1.0) -> np.ndarray:
        """
        Apply brightness and contrast adjustments.
        
        Args:
            image: Input image
            brightness: Brightness adjustment (-100 to 100)
            contrast: Contrast multiplier (0.5 to 3.0)
            
        Returns:
            Adjusted image
        """
        adjusted = image.astype(np.float32)
        adjusted = adjusted * contrast + brightness
        adjusted = np.clip(adjusted, 0, 255)
        
        return adjusted.astype(np.uint8)
    
    @staticmethod
    def apply_gamma_correction(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        """
        Apply gamma correction to image.
        
        Args:
            image: Input image
            gamma: Gamma value (0.5 to 2.0)
            
        Returns:
            Gamma corrected image
        """
        # Build lookup table
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                         for i in np.arange(0, 256)]).astype(np.uint8)
        
        # Apply gamma correction
        return cv2.LUT(image, table)
    
    @staticmethod
    def convert_color_space(image: np.ndarray, conversion: str) -> np.ndarray:
        """
        Convert image between color spaces.
        
        Args:
            image: Input image
            conversion: Conversion type ('rgb2hsv', 'hsv2rgb', 'rgb2gray', etc.)
            
        Returns:
            Converted image
        """
        conversion_map = {
            'rgb2hsv': cv2.COLOR_RGB2HSV,
            'hsv2rgb': cv2.COLOR_HSV2RGB,
            'rgb2gray': cv2.COLOR_RGB2GRAY,
            'gray2rgb': cv2.COLOR_GRAY2RGB,
            'rgb2lab': cv2.COLOR_RGB2LAB,
            'lab2rgb': cv2.COLOR_LAB2RGB,
            'bgr2rgb': cv2.COLOR_BGR2RGB,
            'rgb2bgr': cv2.COLOR_RGB2BGR
        }
        
        if conversion not in conversion_map:
            raise ValueError(f"Unsupported conversion: {conversion}")
        
        return cv2.cvtColor(image, conversion_map[conversion])
    
    @staticmethod
    def apply_gaussian_blur(image: np.ndarray, kernel_size: int = 5,
                           sigma: float = 1.0) -> np.ndarray:
        """
        Apply Gaussian blur to image.
        
        Args:
            image: Input image
            kernel_size: Size of Gaussian kernel (odd number)
            sigma: Standard deviation of Gaussian kernel
            
        Returns:
            Blurred image
        """
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    @staticmethod
    def apply_motion_blur(image: np.ndarray, size: int = 15, 
                         angle: float = 0) -> np.ndarray:
        """
        Apply motion blur to simulate camera motion.
        
        Args:
            image: Input image
            size: Size of motion blur kernel
            angle: Angle of motion in degrees
            
        Returns:
            Motion blurred image
        """
        # Create motion blur kernel
        kernel = np.zeros((size, size))
        
        # Calculate kernel values based on angle
        angle_rad = np.deg2rad(angle)
        cos_val = np.cos(angle_rad)
        sin_val = np.sin(angle_rad)
        
        for i in range(size):
            x = int(i * cos_val)
            y = int(i * sin_val)
            
            if 0 <= x < size and 0 <= y < size:
                kernel[y, x] = 1
        
        # Normalize kernel
        kernel = kernel / np.sum(kernel)
        
        return cv2.filter2D(image, -1, kernel)
    
    @staticmethod
    def add_noise(image: np.ndarray, noise_type: str = 'gaussian',
                 intensity: float = 0.1) -> np.ndarray:
        """
        Add noise to image for data augmentation.
        
        Args:
            image: Input image
            noise_type: Type of noise ('gaussian', 'salt_pepper', 'uniform')
            intensity: Noise intensity (0.0 to 1.0)
            
        Returns:
            Noisy image
        """
        if noise_type == 'gaussian':
            noise = np.random.normal(0, intensity * 255, image.shape)
            noisy = image.astype(np.float32) + noise
        elif noise_type == 'salt_pepper':
            noisy = image.copy().astype(np.float32)
            # Salt noise (white pixels)
            salt_coords = tuple([np.random.randint(0, i - 1, int(intensity * image.size * 0.5))
                                for i in image.shape[:2]])
            noisy[salt_coords] = 255
            
            # Pepper noise (black pixels)
            pepper_coords = tuple([np.random.randint(0, i - 1, int(intensity * image.size * 0.5))
                                  for i in image.shape[:2]])
            noisy[pepper_coords] = 0
        elif noise_type == 'uniform':
            noise = np.random.uniform(-intensity * 255, intensity * 255, image.shape)
            noisy = image.astype(np.float32) + noise
        else:
            raise ValueError(f"Unsupported noise type: {noise_type}")
        
        return np.clip(noisy, 0, 255).astype(np.uint8)
    
    @staticmethod
    def create_roi_mask(image_shape: Tuple[int, int], 
                       vertices: List[Tuple[int, int]]) -> np.ndarray:
        """
        Create region of interest mask.
        
        Args:
            image_shape: (height, width) of image
            vertices: List of (x, y) vertices defining the ROI
            
        Returns:
            Binary mask
        """
        mask = np.zeros(image_shape, dtype=np.uint8)
        vertices_array = np.array([vertices], dtype=np.int32)
        cv2.fillPoly(mask, vertices_array, 255)
        
        return mask
    
    @staticmethod
    def crop_image(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Crop image using bounding box.
        
        Args:
            image: Input image
            bbox: (x1, y1, x2, y2) bounding box coordinates
            
        Returns:
            Cropped image
        """
        x1, y1, x2, y2 = bbox
        return image[y1:y2, x1:x2]
    
    @staticmethod
    def paste_image(background: np.ndarray, foreground: np.ndarray,
                   position: Tuple[int, int]) -> np.ndarray:
        """
        Paste foreground image onto background at specified position.
        
        Args:
            background: Background image
            foreground: Foreground image to paste
            position: (x, y) position to paste at
            
        Returns:
            Combined image
        """
        x, y = position
        h, w = foreground.shape[:2]
        
        # Ensure we don't go out of bounds
        y_end = min(y + h, background.shape[0])
        x_end = min(x + w, background.shape[1])
        fg_h = y_end - y
        fg_w = x_end - x
        
        if fg_h > 0 and fg_w > 0:
            result = background.copy()
            result[y:y_end, x:x_end] = foreground[:fg_h, :fg_w]
            return result
        
        return background
    
    @staticmethod
    def calculate_histogram(image: np.ndarray, bins: int = 256) -> np.ndarray:
        """
        Calculate image histogram.
        
        Args:
            image: Input image
            bins: Number of histogram bins
            
        Returns:
            Histogram array
        """
        if len(image.shape) == 3:
            # Color image - calculate for each channel
            histograms = []
            for i in range(image.shape[2]):
                hist = cv2.calcHist([image], [i], None, [bins], [0, 256])
                histograms.append(hist.flatten())
            return np.array(histograms)
        else:
            # Grayscale image
            hist = cv2.calcHist([image], [0], None, [bins], [0, 256])
            return hist.flatten()
    
    @staticmethod
    def equalize_histogram(image: np.ndarray) -> np.ndarray:
        """
        Apply histogram equalization to improve contrast.
        
        Args:
            image: Input image
            
        Returns:
            Histogram equalized image
        """
        if len(image.shape) == 3:
            # Color image - apply to each channel
            result = np.zeros_like(image)
            for i in range(image.shape[2]):
                result[:, :, i] = cv2.equalizeHist(image[:, :, i])
            return result
        else:
            # Grayscale image
            return cv2.equalizeHist(image)