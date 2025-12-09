"""
CAD ML Module - Machine Learning for Computer-Aided Design

This module provides ML capabilities for CAD applications including:
- Shape classification and recognition
- Geometric feature extraction
- Design parameter optimization
- Synthetic CAD data generation

Typical CAD ML use cases:
- Classifying 2D/3D shapes from point cloud or mesh data
- Predicting manufacturing parameters
- Design similarity search
- Automated design validation
"""

from src.cad_ml.shape_data_generator import CADShapeDataGenerator
from src.cad_ml.cad_feature_extractor import CADFeatureExtractor
from src.cad_ml.cad_shape_classifier import CADShapeClassifier

__all__ = [
    'CADShapeDataGenerator',
    'CADFeatureExtractor', 
    'CADShapeClassifier'
]
