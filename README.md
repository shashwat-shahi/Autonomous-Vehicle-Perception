# Autonomous Vehicle Perception System

This repository contains a comprehensive computer vision system for autonomous vehicles, including lane detection and traffic sign recognition capabilities.

## Features

### Core Capabilities
- **Lane Detection**: Advanced computer vision algorithms for detecting and tracking road lanes
- **Traffic Sign Recognition**: Deep learning models for identifying and classifying traffic signs
- **Real-time Processing**: Optimized for real-time inference on automotive hardware

### Supported Models
- **YOLO**: For object detection and traffic sign recognition
- **SegNet**: For semantic segmentation and lane detection
- **U-Net**: For precise lane boundary segmentation

## Tech Stack

- **Frameworks**: PyTorch, OpenCV
- **Models**: YOLO, SegNet, U-Net
- **Processing**: NumPy, scikit-image
- **Visualization**: Matplotlib, OpenCV

## Supported Datasets

- **German Traffic Signs**: GTSRB dataset for traffic sign recognition
- **Cityscapes**: Urban scene understanding dataset
- **KITTI**: Autonomous driving dataset with stereo vision and LiDAR

## Installation

1. Clone the repository:
```bash
git clone https://github.com/shashwat-shahi/Autonomous-Vehicle-Perception.git
cd Autonomous-Vehicle-Perception
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Lane Detection
```python
from src.lane_detection import LaneDetector

detector = LaneDetector()
lanes = detector.detect(image_path)
```

### Traffic Sign Recognition
```python
from src.traffic_sign_recognition import TrafficSignRecognizer

recognizer = TrafficSignRecognizer()
signs = recognizer.recognize(image_path)
```

## Project Structure

```
├── src/
│   ├── lane_detection/          # Lane detection modules
│   ├── traffic_sign_recognition/ # Traffic sign recognition modules
│   ├── models/                  # Model implementations
│   ├── utils/                   # Utility functions
│   └── datasets/                # Dataset loading and preprocessing
├── tests/                       # Unit tests
├── examples/                    # Example scripts and demos
└── requirements.txt             # Python dependencies
```

## Examples

See the `examples/` directory for demonstration scripts showing how to use the various components of the system.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
