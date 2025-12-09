"""
CAD Feature Extractor

Extracts and transforms geometric features from CAD shape data
for use in machine learning models.

Features computed:
- Normalized geometric properties
- Shape descriptors (Fourier descriptors, moments)
- Derived features (circularity, rectangularity, etc.)
"""

import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Optional

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class CADFeatureConfig:
    """Configuration for CAD feature extraction"""
    preprocessor_path: str = os.path.join('artifacts', 'cad_ml', 'cad_preprocessor.pkl')
    label_encoder_path: str = os.path.join('artifacts', 'cad_ml', 'cad_label_encoder.pkl')


class CADFeatureExtractor:
    """
    Extracts and transforms features from CAD shape data.
    
    Handles:
    - Feature scaling and normalization
    - Derived feature computation
    - Label encoding for shape classes
    """
    
    FEATURE_COLUMNS = ['area', 'perimeter', 'aspect_ratio', 'compactness', 'n_vertices']
    TARGET_COLUMN = 'shape_class'
    
    def __init__(self, config: Optional[CADFeatureConfig] = None):
        self.config = config if config else CADFeatureConfig()
        self.preprocessor = None
        self.label_encoder = None
        
    def _compute_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute additional derived features useful for shape classification.
        
        Derived features:
        - circularity: How close to a perfect circle (compactness normalized)
        - log_area: Log of area (helps with scale invariance)
        - perimeter_area_ratio: Perimeter / sqrt(area)
        - is_polygonal: Binary flag for shapes with discrete vertices
        """
        df = df.copy()
        
        # Log transform for better distribution
        df['log_area'] = np.log1p(df['area'])
        df['log_perimeter'] = np.log1p(df['perimeter'])
        
        # Perimeter to area ratio (scale-invariant shape descriptor)
        df['perimeter_area_ratio'] = df['perimeter'] / np.sqrt(df['area'] + 1e-6)
        
        # Binary feature: is the shape polygonal (has discrete vertices)?
        df['is_polygonal'] = (df['n_vertices'] > 0).astype(float)
        
        # Normalized compactness (0-1 range)
        df['circularity'] = df['compactness'].clip(0, 1)
        
        return df
    
    def get_feature_transformer(self) -> ColumnTransformer:
        """
        Create preprocessing pipeline for CAD features.
        
        Returns:
            ColumnTransformer that scales all numeric features
        """
        # All CAD features are numeric, so we use StandardScaler
        feature_cols = self.FEATURE_COLUMNS + [
            'log_area', 'log_perimeter', 'perimeter_area_ratio', 
            'is_polygonal', 'circularity'
        ]
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('scaler', StandardScaler(), feature_cols)
            ],
            remainder='drop'
        )
        
        return preprocessor
    
    def fit_transform(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit preprocessor on training data and transform both train and test.
        
        Args:
            train_df: Training DataFrame with features and target
            test_df: Test DataFrame with features and target
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logging.info("Starting CAD feature extraction and transformation")
        
        try:
            # Compute derived features
            train_df = self._compute_derived_features(train_df)
            test_df = self._compute_derived_features(test_df)
            
            # Get feature columns (original + derived)
            feature_cols = self.FEATURE_COLUMNS + [
                'log_area', 'log_perimeter', 'perimeter_area_ratio',
                'is_polygonal', 'circularity'
            ]
            
            # Separate features and target
            X_train = train_df[feature_cols]
            X_test = test_df[feature_cols]
            y_train = train_df[self.TARGET_COLUMN]
            y_test = test_df[self.TARGET_COLUMN]
            
            # Fit and transform features
            self.preprocessor = self.get_feature_transformer()
            X_train_scaled = self.preprocessor.fit_transform(X_train)
            X_test_scaled = self.preprocessor.transform(X_test)
            
            logging.info(f"Transformed features shape: {X_train_scaled.shape}")
            
            # Encode labels
            self.label_encoder = LabelEncoder()
            y_train_encoded = self.label_encoder.fit_transform(y_train)
            y_test_encoded = self.label_encoder.transform(y_test)
            
            logging.info(f"Classes: {list(self.label_encoder.classes_)}")
            
            # Save preprocessor and label encoder
            os.makedirs(os.path.dirname(self.config.preprocessor_path), exist_ok=True)
            
            save_object(self.config.preprocessor_path, self.preprocessor)
            save_object(self.config.label_encoder_path, self.label_encoder)
            
            logging.info(f"Saved preprocessor to {self.config.preprocessor_path}")
            
            return X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using fitted preprocessor.
        
        Args:
            df: DataFrame with CAD features
            
        Returns:
            Transformed feature array
        """
        if self.preprocessor is None:
            raise CustomException("Preprocessor not fitted. Call fit_transform first.", sys)
        
        try:
            df = self._compute_derived_features(df)
            
            feature_cols = self.FEATURE_COLUMNS + [
                'log_area', 'log_perimeter', 'perimeter_area_ratio',
                'is_polygonal', 'circularity'
            ]
            
            return self.preprocessor.transform(df[feature_cols])
            
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Demo: Load CAD data and extract features
    from src.cad_ml.shape_data_generator import CADShapeDataGenerator, CADDataConfig
    
    # Generate data
    generator = CADShapeDataGenerator()
    df = generator.generate_dataset()
    generator.save_dataset(df)
    
    # Load train/test
    train_df = pd.read_csv(CADDataConfig().cad_train_path)
    test_df = pd.read_csv(CADDataConfig().cad_test_path)
    
    # Extract features
    extractor = CADFeatureExtractor()
    X_train, X_test, y_train, y_test = extractor.fit_transform(train_df, test_df)
    
    print("\n=== CAD Feature Extraction Results ===")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"Number of classes: {len(extractor.label_encoder.classes_)}")
    print(f"Classes: {list(extractor.label_encoder.classes_)}")
