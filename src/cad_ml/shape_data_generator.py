"""
CAD Shape Data Generator

Generates synthetic CAD shape data for training ML models.
Supports 2D geometric primitives commonly found in CAD applications:
- Circle, Rectangle, Triangle, Hexagon, Ellipse

Each shape is represented by computed geometric features:
- Area and perimeter measurements
- Aspect ratio and compactness metrics
- Number of vertices (0 for curved shapes)
"""

import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional

from sklearn.model_selection import train_test_split

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.exception import CustomException
from src.logger import logging


@dataclass
class CADDataConfig:
    """Configuration for CAD data generation and storage"""
    cad_data_path: str = os.path.join('artifacts', 'cad_ml', 'cad_shapes.csv')
    cad_train_path: str = os.path.join('artifacts', 'cad_ml', 'cad_train.csv')
    cad_test_path: str = os.path.join('artifacts', 'cad_ml', 'cad_test.csv')
    n_samples_per_class: int = 200
    noise_level: float = 0.05


class CADShapeDataGenerator:
    """
    Generates synthetic CAD shape data with geometric features.
    
    This simulates real-world CAD data where shapes are represented
    by their geometric properties - commonly used for:
    - Shape classification in design automation
    - Manufacturing parameter prediction
    - Design similarity analysis
    """
    
    SHAPE_CLASSES = ['circle', 'rectangle', 'triangle', 'hexagon', 'ellipse']
    
    def __init__(self, config: Optional[CADDataConfig] = None):
        self.config = config if config else CADDataConfig()
        
    def _generate_circle_features(self, n_samples: int) -> np.ndarray:
        """Generate features for circle shapes"""
        # Random radii between 1 and 10
        radii = np.random.uniform(1, 10, n_samples)
        
        # Geometric features
        areas = np.pi * radii ** 2
        perimeters = 2 * np.pi * radii
        aspect_ratios = np.ones(n_samples)  # Circle has aspect ratio 1
        compactness = 4 * np.pi * areas / (perimeters ** 2)  # Always ~1 for circles
        n_vertices = np.zeros(n_samples)  # Infinite vertices (represented as 0)
        
        # Add noise to simulate measurement uncertainty
        noise = np.random.normal(0, self.config.noise_level, (n_samples, 5))
        
        features = np.column_stack([
            areas, perimeters, aspect_ratios, compactness, n_vertices
        ])
        
        return features + noise * np.abs(features)
    
    def _generate_rectangle_features(self, n_samples: int) -> np.ndarray:
        """Generate features for rectangle shapes"""
        widths = np.random.uniform(1, 10, n_samples)
        heights = np.random.uniform(1, 10, n_samples)
        
        areas = widths * heights
        perimeters = 2 * (widths + heights)
        aspect_ratios = np.maximum(widths, heights) / np.minimum(widths, heights)
        compactness = 4 * np.pi * areas / (perimeters ** 2)
        n_vertices = np.full(n_samples, 4)
        
        noise = np.random.normal(0, self.config.noise_level, (n_samples, 5))
        
        features = np.column_stack([
            areas, perimeters, aspect_ratios, compactness, n_vertices
        ])
        
        return features + noise * np.abs(features)
    
    def _generate_triangle_features(self, n_samples: int) -> np.ndarray:
        """Generate features for triangle shapes"""
        # Random triangle sides
        a = np.random.uniform(2, 10, n_samples)
        b = np.random.uniform(2, 10, n_samples)
        # Ensure valid triangle (c < a + b)
        c = np.random.uniform(np.abs(a - b) + 0.5, a + b - 0.5)
        
        # Heron's formula for area
        s = (a + b + c) / 2
        areas = np.sqrt(s * (s - a) * (s - b) * (s - c))
        perimeters = a + b + c
        
        # Aspect ratio approximation (longest side / shortest side)
        sides = np.column_stack([a, b, c])
        aspect_ratios = np.max(sides, axis=1) / np.min(sides, axis=1)
        
        compactness = 4 * np.pi * areas / (perimeters ** 2)
        n_vertices = np.full(n_samples, 3)
        
        noise = np.random.normal(0, self.config.noise_level, (n_samples, 5))
        
        features = np.column_stack([
            areas, perimeters, aspect_ratios, compactness, n_vertices
        ])
        
        return features + noise * np.abs(features)
    
    def _generate_hexagon_features(self, n_samples: int) -> np.ndarray:
        """Generate features for regular hexagon shapes"""
        # Side length
        sides = np.random.uniform(1, 10, n_samples)
        
        areas = (3 * np.sqrt(3) / 2) * sides ** 2
        perimeters = 6 * sides
        aspect_ratios = np.ones(n_samples)  # Regular hexagon
        compactness = 4 * np.pi * areas / (perimeters ** 2)
        n_vertices = np.full(n_samples, 6)
        
        noise = np.random.normal(0, self.config.noise_level, (n_samples, 5))
        
        features = np.column_stack([
            areas, perimeters, aspect_ratios, compactness, n_vertices
        ])
        
        return features + noise * np.abs(features)
    
    def _generate_ellipse_features(self, n_samples: int) -> np.ndarray:
        """Generate features for ellipse shapes"""
        # Semi-major and semi-minor axes
        a = np.random.uniform(2, 10, n_samples)
        b = np.random.uniform(1, 8, n_samples)
        # Ensure a >= b
        a, b = np.maximum(a, b), np.minimum(a, b)
        
        areas = np.pi * a * b
        # Ramanujan approximation for ellipse perimeter
        h = ((a - b) ** 2) / ((a + b) ** 2)
        perimeters = np.pi * (a + b) * (1 + 3 * h / (10 + np.sqrt(4 - 3 * h)))
        
        aspect_ratios = a / b
        compactness = 4 * np.pi * areas / (perimeters ** 2)
        n_vertices = np.zeros(n_samples)  # Infinite vertices
        
        noise = np.random.normal(0, self.config.noise_level, (n_samples, 5))
        
        features = np.column_stack([
            areas, perimeters, aspect_ratios, compactness, n_vertices
        ])
        
        return features + noise * np.abs(features)
    
    def generate_dataset(self) -> pd.DataFrame:
        """
        Generate complete CAD shape dataset with features and labels.
        
        Returns:
            DataFrame with columns:
            - area: Shape area
            - perimeter: Shape perimeter
            - aspect_ratio: Width/height ratio
            - compactness: 4*pi*area / perimeter^2 (circularity measure)
            - n_vertices: Number of vertices (0 for curved shapes)
            - shape_class: Shape label
        """
        logging.info("Generating synthetic CAD shape dataset")
        
        try:
            all_features = []
            all_labels = []
            
            generators = {
                'circle': self._generate_circle_features,
                'rectangle': self._generate_rectangle_features,
                'triangle': self._generate_triangle_features,
                'hexagon': self._generate_hexagon_features,
                'ellipse': self._generate_ellipse_features
            }
            
            for shape_class, generator in generators.items():
                features = generator(self.config.n_samples_per_class)
                labels = [shape_class] * self.config.n_samples_per_class
                
                all_features.append(features)
                all_labels.extend(labels)
                
                logging.info(f"Generated {self.config.n_samples_per_class} samples for {shape_class}")
            
            # Combine all features
            features_array = np.vstack(all_features)
            
            # Create DataFrame
            df = pd.DataFrame(features_array, columns=[
                'area', 'perimeter', 'aspect_ratio', 'compactness', 'n_vertices'
            ])
            df['shape_class'] = all_labels
            
            # Shuffle the dataset
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            logging.info(f"Generated CAD dataset with {len(df)} samples, {len(self.SHAPE_CLASSES)} classes")
            
            return df
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def save_dataset(self, df: pd.DataFrame) -> Tuple[str, str, str]:
        """
        Save generated dataset to CSV files with train/test split.
        
        Returns:
            Tuple of (raw_data_path, train_data_path, test_data_path)
        """
        logging.info("Saving CAD dataset to files")
        
        try:
            # Create directory
            os.makedirs(os.path.dirname(self.config.cad_data_path), exist_ok=True)
            
            # Save raw data
            df.to_csv(self.config.cad_data_path, index=False)
            logging.info(f"Saved raw CAD data to {self.config.cad_data_path}")
            
            # Train/test split
            train_df, test_df = train_test_split(df, test_size=0.2, 
                                                  stratify=df['shape_class'],
                                                  random_state=42)
            
            train_df.to_csv(self.config.cad_train_path, index=False)
            test_df.to_csv(self.config.cad_test_path, index=False)
            
            logging.info(f"Train set: {len(train_df)} samples, Test set: {len(test_df)} samples")
            
            return (self.config.cad_data_path, 
                    self.config.cad_train_path, 
                    self.config.cad_test_path)
            
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Demo: Generate and save CAD shape data
    generator = CADShapeDataGenerator()
    df = generator.generate_dataset()
    
    print("\n=== CAD Shape Dataset Summary ===")
    print(f"Total samples: {len(df)}")
    print(f"\nShape class distribution:")
    print(df['shape_class'].value_counts())
    print(f"\nFeature statistics:")
    print(df.describe())
    
    # Save to files
    raw_path, train_path, test_path = generator.save_dataset(df)
    print(f"\nData saved to:")
    print(f"  Raw: {raw_path}")
    print(f"  Train: {train_path}")
    print(f"  Test: {test_path}")
