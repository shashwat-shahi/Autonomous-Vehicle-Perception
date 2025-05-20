"""
Traffic Sign Classifier

Deep learning classifier for traffic sign recognition using CNN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from typing import Tuple, List
import numpy as np


class SignClassifier(nn.Module):
    """
    Convolutional Neural Network for traffic sign classification.
    
    Based on LeNet-5 architecture but adapted for traffic sign recognition
    with additional layers and modern techniques.
    """
    
    def __init__(self, num_classes: int = 43):
        """
        Initialize the classifier.
        
        Args:
            num_classes: Number of traffic sign classes (43 for GTSRB)
        """
        super(SignClassifier, self).__init__()
        
        self.num_classes = num_classes
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 32, 32)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # First convolutional block
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Second convolutional block
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Third convolutional block
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten for fully connected layers
        x = x.view(-1, 128 * 4 * 4)
        
        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        return x
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get class probabilities from logits.
        
        Args:
            x: Input tensor
            
        Returns:
            Probability tensor
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=1)
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with confidence scores.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (predicted_classes, confidence_scores)
        """
        probs = self.predict_proba(x)
        confidence, predicted = torch.max(probs, 1)
        
        return predicted, confidence


class ImprovedSignClassifier(nn.Module):
    """
    Improved traffic sign classifier using ResNet-inspired architecture.
    """
    
    def __init__(self, num_classes: int = 43):
        """
        Initialize the improved classifier.
        
        Args:
            num_classes: Number of traffic sign classes
        """
        super(ImprovedSignClassifier, self).__init__()
        
        self.num_classes = num_classes
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.fc = nn.Linear(256, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
    
    def _make_layer(self, in_channels: int, out_channels: int, blocks: int, stride: int = 1):
        """
        Create a residual layer.
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            blocks: Number of residual blocks
            stride: Stride for first block
            
        Returns:
            Sequential layer
        """
        layers = []
        
        # First block might have stride
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the improved network.
        
        Args:
            x: Input tensor
            
        Returns:
            Output logits
        """
        # Initial convolution
        x = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        
        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Classifier
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


class ResidualBlock(nn.Module):
    """
    Residual block for improved classifier.
    """
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        """
        Initialize residual block.
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            stride: Convolution stride
        """
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through residual block.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += self.shortcut(residual)
        out = F.relu(out)
        
        return out


def create_data_transforms(input_size: int = 32) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Create data transformation pipelines for training and validation.
    
    Args:
        input_size: Input image size
        
    Returns:
        Tuple of (train_transform, val_transform)
    """
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Validation transforms without augmentation
    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def initialize_weights(model: nn.Module):
    """
    Initialize model weights using Xavier/He initialization.
    
    Args:
        model: PyTorch model
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)