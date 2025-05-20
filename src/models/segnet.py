"""
SegNet Implementation for Semantic Segmentation

SegNet architecture for road scene understanding and lane detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


class SegNet(nn.Module):
    """
    SegNet architecture for semantic segmentation.
    
    Based on the VGG-16 encoder with symmetric decoder using
    max pooling indices for upsampling.
    """
    
    def __init__(self, input_channels: int = 3, output_channels: int = 12):
        """
        Initialize SegNet model.
        
        Args:
            input_channels: Number of input image channels
            output_channels: Number of segmentation classes
        """
        super(SegNet, self).__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        # Encoder layers (based on VGG-16)
        self.encoder = nn.ModuleList([
            # Stage 1
            nn.Sequential(
                nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            ),
            # Stage 2
            nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            ),
            # Stage 3
            nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ),
            # Stage 4
            nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ),
            # Stage 5
            nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            )
        ])
        
        # Decoder layers (symmetric to encoder)
        self.decoder = nn.ModuleList([
            # Stage 5 decoder
            nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ),
            # Stage 4 decoder
            nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ),
            # Stage 3 decoder
            nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            ),
            # Stage 2 decoder
            nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            ),
            # Stage 1 decoder
            nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, output_channels, kernel_size=3, padding=1)
            )
        ])
        
        # Max pooling and upsampling layers
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, 2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through SegNet.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Segmentation output
        """
        # Store pooling indices for upsampling
        pool_indices = []
        
        # Encoder forward pass
        for i, encoder_stage in enumerate(self.encoder):
            x = encoder_stage(x)
            x, indices = self.pool(x)
            pool_indices.append(indices)
        
        # Decoder forward pass (reverse order)
        for i, decoder_stage in enumerate(self.decoder):
            # Get corresponding pooling indices
            indices = pool_indices[-(i+1)]
            x = self.unpool(x, indices)
            x = decoder_stage(x)
        
        return x
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions with softmax activation.
        
        Args:
            x: Input tensor
            
        Returns:
            Class probability predictions
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=1)


class SegNetBasic(nn.Module):
    """
    Simplified SegNet architecture with fewer layers.
    """
    
    def __init__(self, input_channels: int = 3, output_channels: int = 12):
        """Initialize basic SegNet."""
        super(SegNetBasic, self).__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        # Encoder
        self.enc_conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.enc_conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=7, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.enc_conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=7, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.enc_conv4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=7, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.dec_conv4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=7, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.dec_conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=7, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.dec_conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=7, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.dec_conv1 = nn.Conv2d(64, output_channels, kernel_size=7, padding=3)
        
        # Pooling layers
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, 2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through basic SegNet."""
        # Encoder
        x = self.enc_conv1(x)
        x, indices1 = self.pool(x)
        
        x = self.enc_conv2(x)
        x, indices2 = self.pool(x)
        
        x = self.enc_conv3(x)
        x, indices3 = self.pool(x)
        
        x = self.enc_conv4(x)
        x, indices4 = self.pool(x)
        
        # Decoder
        x = self.unpool(x, indices4)
        x = self.dec_conv4(x)
        
        x = self.unpool(x, indices3)
        x = self.dec_conv3(x)
        
        x = self.unpool(x, indices2)
        x = self.dec_conv2(x)
        
        x = self.unpool(x, indices1)
        x = self.dec_conv1(x)
        
        return x


class SegNetForLanes(SegNet):
    """
    SegNet specifically configured for lane detection.
    
    Uses fewer output classes optimized for lane segmentation.
    """
    
    def __init__(self, input_channels: int = 3):
        """
        Initialize lane detection SegNet.
        
        Args:
            input_channels: Number of input channels
        """
        # 3 classes: background, left lane, right lane
        super(SegNetForLanes, self).__init__(input_channels=input_channels, 
                                           output_channels=3)
    
    def predict_lanes(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Predict lane segmentation masks.
        
        Args:
            x: Input tensor
            threshold: Probability threshold for lane detection
            
        Returns:
            Lane segmentation mask
        """
        probs = self.predict(x)
        
        # Combine left and right lane probabilities
        lane_probs = probs[:, 1:3, :, :].sum(dim=1, keepdim=True)
        
        return (lane_probs > threshold).float()


def initialize_segnet_weights(model: SegNet):
    """
    Initialize SegNet weights using He initialization.
    
    Args:
        model: SegNet model to initialize
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def create_segnet_loss(ignore_index: int = -1, weight: torch.Tensor = None) -> nn.Module:
    """
    Create loss function for SegNet training.
    
    Args:
        ignore_index: Index to ignore in loss calculation
        weight: Class weights for imbalanced datasets
        
    Returns:
        Cross entropy loss function
    """
    return nn.CrossEntropyLoss(ignore_index=ignore_index, weight=weight)