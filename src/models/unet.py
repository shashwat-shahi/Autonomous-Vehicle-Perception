"""
U-Net Implementation for Semantic Segmentation

U-Net architecture for lane detection and road segmentation tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class DoubleConv(nn.Module):
    """Double convolution block used in U-Net."""
    
    def __init__(self, in_channels: int, out_channels: int):
        """
        Initialize double convolution block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super(DoubleConv, self).__init__()
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through double convolution."""
        return self.double_conv(x)


class Down(nn.Module):
    """Downsampling block with max pooling and double convolution."""
    
    def __init__(self, in_channels: int, out_channels: int):
        """
        Initialize down block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super(Down, self).__init__()
        
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through down block."""
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upsampling block with transpose convolution and double convolution."""
    
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        """
        Initialize up block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            bilinear: Whether to use bilinear upsampling
        """
        super(Up, self).__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through up block.
        
        Args:
            x1: Feature map from decoder path
            x2: Feature map from encoder path (skip connection)
            
        Returns:
            Upsampled and concatenated feature map
        """
        x1 = self.up(x1)
        
        # Handle size mismatch between x1 and x2
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                       diff_y // 2, diff_y - diff_y // 2])
        
        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        
        return self.conv(x)


class OutConv(nn.Module):
    """Output convolution layer."""
    
    def __init__(self, in_channels: int, out_channels: int):
        """
        Initialize output convolution.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super(OutConv, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through output convolution."""
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net architecture for semantic segmentation.
    
    Originally designed for biomedical image segmentation but adapted
    for lane detection and road segmentation tasks.
    """
    
    def __init__(self, n_channels: int = 3, n_classes: int = 2, bilinear: bool = True):
        """
        Initialize U-Net model.
        
        Args:
            n_channels: Number of input channels
            n_classes: Number of output classes
            bilinear: Whether to use bilinear upsampling
        """
        super(UNet, self).__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # Encoder (contracting path)
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        # Decoder (expanding path)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        # Output layer
        self.outc = OutConv(64, n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through U-Net.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Segmentation output of shape (batch_size, n_classes, height, width)
        """
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder path
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output
        logits = self.outc(x)
        
        return logits
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions with softmax activation.
        
        Args:
            x: Input tensor
            
        Returns:
            Probability predictions
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=1)


class UNetForLanes(UNet):
    """
    U-Net specifically configured for lane detection.
    
    Uses binary segmentation (lane vs. background).
    """
    
    def __init__(self, n_channels: int = 3):
        """
        Initialize lane detection U-Net.
        
        Args:
            n_channels: Number of input channels
        """
        super(UNetForLanes, self).__init__(n_channels=n_channels, n_classes=2)
    
    def predict_lanes(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Predict lane masks with thresholding.
        
        Args:
            x: Input tensor
            threshold: Probability threshold for lane detection
            
        Returns:
            Binary lane mask
        """
        probs = self.predict(x)
        lane_probs = probs[:, 1:2, :, :]  # Extract lane probability channel
        
        return (lane_probs > threshold).float()


class AttentionUNet(nn.Module):
    """
    U-Net with attention mechanism for improved segmentation.
    """
    
    def __init__(self, n_channels: int = 3, n_classes: int = 2):
        """
        Initialize Attention U-Net.
        
        Args:
            n_channels: Number of input channels
            n_classes: Number of output classes
        """
        super(AttentionUNet, self).__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        
        # Decoder with attention
        self.up1 = AttentionUp(1024, 512)
        self.up2 = AttentionUp(512, 256)
        self.up3 = AttentionUp(256, 128)
        self.up4 = AttentionUp(128, 64)
        
        # Output
        self.outc = OutConv(64, n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Attention U-Net."""
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder with attention
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        return self.outc(x)


class AttentionBlock(nn.Module):
    """Attention mechanism for U-Net."""
    
    def __init__(self, F_g: int, F_l: int, F_int: int):
        """
        Initialize attention block.
        
        Args:
            F_g: Number of feature maps in gating signal
            F_l: Number of feature maps in input feature map
            F_int: Number of intermediate feature maps
        """
        super(AttentionBlock, self).__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through attention block."""
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi


class AttentionUp(nn.Module):
    """Upsampling block with attention mechanism."""
    
    def __init__(self, in_channels: int, out_channels: int):
        """Initialize attention upsampling block."""
        super(AttentionUp, self).__init__()
        
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.attention = AttentionBlock(in_channels // 2, in_channels // 2, in_channels // 4)
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Forward pass through attention up block."""
        x1 = self.up(x1)
        
        # Apply attention
        x2 = self.attention(x1, x2)
        
        # Handle size mismatch
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                       diff_y // 2, diff_y - diff_y // 2])
        
        # Concatenate
        x = torch.cat([x2, x1], dim=1)
        
        return self.conv(x)