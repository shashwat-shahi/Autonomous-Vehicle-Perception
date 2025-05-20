"""
Lane Detection Module

This module provides lane detection capabilities using computer vision techniques
including Hough transforms, edge detection, and polynomial fitting.
"""

from .lane_detector import LaneDetector
from .lane_utils import LaneUtils

__all__ = ['LaneDetector', 'LaneUtils']