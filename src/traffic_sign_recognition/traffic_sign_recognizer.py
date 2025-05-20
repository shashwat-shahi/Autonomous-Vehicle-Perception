"""
Traffic Sign Recognition Implementation

This module implements traffic sign detection and classification using
deep learning models including YOLO for detection and CNN for classification.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from PIL import Image


class TrafficSignRecognizer:
    """
    Traffic sign recognition system using deep learning.
    
    Combines object detection for sign localization with classification
    for sign type identification.
    """
    
    def __init__(self, 
                 confidence_threshold: float = 0.5,
                 nms_threshold: float = 0.4,
                 device: str = 'cpu'):
        """
        Initialize traffic sign recognizer.
        
        Args:
            confidence_threshold: Minimum confidence for detections
            nms_threshold: Non-maximum suppression threshold
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.device = device
        
        # German Traffic Sign class names (GTSRB dataset)
        self.class_names = [
            'Speed limit 20', 'Speed limit 30', 'Speed limit 50', 'Speed limit 60',
            'Speed limit 70', 'Speed limit 80', 'End of speed limit 80', 'Speed limit 100',
            'Speed limit 120', 'No passing', 'No passing for vehicles over 3.5 metric tons',
            'Right-of-way at the next intersection', 'Priority road', 'Yield', 'Stop',
            'No vehicles', 'Vehicles over 3.5 metric tons prohibited', 'No entry',
            'General caution', 'Dangerous curve to the left', 'Dangerous curve to the right',
            'Double curve', 'Bumpy road', 'Slippery road', 'Road narrows on the right',
            'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing',
            'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing',
            'End of all speed and passing limits', 'Turn right ahead', 'Turn left ahead',
            'Ahead only', 'Go straight or right', 'Go straight or left', 'Keep right',
            'Keep left', 'Roundabout mandatory', 'End of no passing',
            'End of no passing by vehicles over 3.5 metric tons'
        ]
        
        # Image preprocessing transform
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize models (placeholders - would load actual trained models)
        self.detector_model = None  # YOLO model would be loaded here
        self.classifier_model = None  # CNN classifier would be loaded here
        
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for detection.
        
        Args:
            image: Input RGB image
            
        Returns:
            Preprocessed image
        """
        # Resize to standard input size for detection model
        resized = cv2.resize(image, (640, 640))
        
        # Normalize pixel values
        normalized = resized.astype(np.float32) / 255.0
        
        return normalized
    
    def detect_signs_classical(self, image: np.ndarray) -> List[Dict]:
        """
        Detect traffic signs using classical computer vision (fallback method).
        
        Args:
            image: Input RGB image
            
        Returns:
            List of detection dictionaries
        """
        detections = []
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Define color ranges for typical traffic signs
        # Red signs (stop, speed limits, etc.)
        red_lower1 = np.array([0, 50, 50])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 50, 50])
        red_upper2 = np.array([180, 255, 255])
        
        # Blue signs (mandatory signs)
        blue_lower = np.array([100, 50, 50])
        blue_upper = np.array([130, 255, 255])
        
        # Yellow signs (warning signs)
        yellow_lower = np.array([20, 50, 50])
        yellow_upper = np.array([30, 255, 255])
        
        # Create masks
        red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        
        # Combine masks
        combined_mask = cv2.bitwise_or(red_mask, cv2.bitwise_or(blue_mask, yellow_mask))
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Filter by area
            area = cv2.contourArea(contour)
            if area < 500:  # Minimum area threshold
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by aspect ratio (signs are roughly square or circular)
            aspect_ratio = w / h
            if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                continue
            
            # Add detection
            detection = {
                'bbox': [x, y, x + w, y + h],
                'confidence': 0.7,  # Default confidence for classical detection
                'class_id': 0,  # Unknown class
                'class_name': 'Traffic Sign'
            }
            detections.append(detection)
        
        return detections
    
    def classify_sign(self, sign_image: np.ndarray) -> Tuple[int, float]:
        """
        Classify a detected traffic sign.
        
        Args:
            sign_image: Cropped sign image
            
        Returns:
            Tuple of (class_id, confidence)
        """
        # For demonstration, return a mock classification
        # In real implementation, this would use a trained CNN
        
        # Convert to PIL Image for preprocessing
        if sign_image.dtype != np.uint8:
            sign_image = (sign_image * 255).astype(np.uint8)
        
        pil_image = Image.fromarray(sign_image)
        
        # Apply preprocessing
        input_tensor = self.transform(pil_image).unsqueeze(0)
        
        # Mock classification (would use actual model here)
        # For now, return a random valid class
        class_id = np.random.randint(0, len(self.class_names))
        confidence = np.random.uniform(0.5, 0.95)
        
        return class_id, confidence
    
    def apply_nms(self, detections: List[Dict]) -> List[Dict]:
        """
        Apply Non-Maximum Suppression to remove duplicate detections.
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            Filtered detections after NMS
        """
        if not detections:
            return []
        
        # Extract bounding boxes and scores
        boxes = []
        scores = []
        
        for det in detections:
            bbox = det['bbox']
            boxes.append([bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]])
            scores.append(det['confidence'])
        
        boxes = np.array(boxes)
        scores = np.array(scores)
        
        # Apply NMS using OpenCV
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), 
            scores.tolist(), 
            self.confidence_threshold, 
            self.nms_threshold
        )
        
        # Filter detections
        filtered_detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                filtered_detections.append(detections[i])
        
        return filtered_detections
    
    def recognize(self, image: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Recognize traffic signs in the input image.
        
        Args:
            image: Input RGB image
            
        Returns:
            Tuple of (result_image, detections)
        """
        # Detect signs
        detections = self.detect_signs_classical(image)
        
        # Classify each detection
        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            
            # Extract sign region
            sign_roi = image[y1:y2, x1:x2]
            
            if sign_roi.size > 0:
                # Classify the sign
                class_id, confidence = self.classify_sign(sign_roi)
                
                # Update detection with classification results
                detection['class_id'] = class_id
                detection['class_name'] = self.class_names[class_id]
                detection['classification_confidence'] = confidence
        
        # Apply NMS
        filtered_detections = self.apply_nms(detections)
        
        # Draw results on image
        result_image = self.draw_detections(image, filtered_detections)
        
        return result_image, filtered_detections
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw detection results on the image.
        
        Args:
            image: Input image
            detections: List of detection dictionaries
            
        Returns:
            Image with detections drawn
        """
        result = image.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            
            # Draw bounding box
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{detection['class_name']}: {detection['confidence']:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Draw label background
            cv2.rectangle(result, 
                         (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), 
                         (0, 255, 0), -1)
            
            # Draw label text
            cv2.putText(result, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return result
    
    def recognize_from_path(self, image_path: str) -> Tuple[np.ndarray, List[Dict]]:
        """
        Recognize traffic signs from image file path.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Tuple of (result_image, detections)
        """
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return self.recognize(image)
    
    def get_sign_statistics(self, detections: List[Dict]) -> Dict:
        """
        Get statistics about detected traffic signs.
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            Statistics dictionary
        """
        stats = {
            'total_signs': len(detections),
            'sign_types': {},
            'average_confidence': 0.0,
            'max_confidence': 0.0,
            'min_confidence': 1.0
        }
        
        if not detections:
            return stats
        
        confidences = []
        
        for detection in detections:
            # Count sign types
            class_name = detection['class_name']
            if class_name in stats['sign_types']:
                stats['sign_types'][class_name] += 1
            else:
                stats['sign_types'][class_name] = 1
            
            # Collect confidences
            confidence = detection['confidence']
            confidences.append(confidence)
        
        # Calculate confidence statistics
        stats['average_confidence'] = np.mean(confidences)
        stats['max_confidence'] = np.max(confidences)
        stats['min_confidence'] = np.min(confidences)
        
        return stats