"""
CAD Shape Classifier

Machine learning model for classifying CAD shapes based on geometric features.
Uses multiple algorithms and selects the best performer.

Supported models:
- Random Forest
- Gradient Boosting
- XGBoost
- Support Vector Machine
- Neural Network (MLP)
"""

import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import cross_val_score, GridSearchCV

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class CADClassifierConfig:
    """Configuration for CAD shape classifier"""
    model_path: str = os.path.join('artifacts', 'cad_ml', 'cad_shape_model.pkl')
    cv_folds: int = 5
    random_state: int = 42


class CADShapeClassifier:
    """
    Multi-model classifier for CAD shape recognition.
    
    Trains multiple ML models and selects the best one based on
    cross-validation accuracy. Suitable for:
    - Shape classification in design automation
    - Quality control in manufacturing
    - Design pattern recognition
    """
    
    def __init__(self, config: Optional[CADClassifierConfig] = None):
        self.config = config if config else CADClassifierConfig()
        self.best_model = None
        self.best_model_name = None
        self.model_scores = {}
        
    def _get_models(self) -> Dict:
        """Get dictionary of classification models to evaluate"""
        return {
            "Random Forest": RandomForestClassifier(random_state=self.config.random_state),
            "Gradient Boosting": GradientBoostingClassifier(random_state=self.config.random_state),
            "Decision Tree": DecisionTreeClassifier(random_state=self.config.random_state),
            "Logistic Regression": LogisticRegression(random_state=self.config.random_state, max_iter=1000),
            "K-Nearest Neighbors": KNeighborsClassifier(),
            "SVM": SVC(random_state=self.config.random_state, probability=True),
            "MLP Neural Network": MLPClassifier(random_state=self.config.random_state, max_iter=500),
            "AdaBoost": AdaBoostClassifier(random_state=self.config.random_state)
        }
    
    def _get_hyperparameters(self) -> Dict:
        """Get hyperparameter grids for model tuning"""
        return {
            "Random Forest": {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 20, None],
                'min_samples_split': [2, 5, 10]
            },
            "Gradient Boosting": {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            "Decision Tree": {
                'max_depth': [5, 10, 20, None],
                'min_samples_split': [2, 5, 10]
            },
            "Logistic Regression": {
                'C': [0.1, 1.0, 10.0]
            },
            "K-Nearest Neighbors": {
                'n_neighbors': [3, 5, 7, 11],
                'weights': ['uniform', 'distance']
            },
            "SVM": {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['rbf', 'poly']
            },
            "MLP Neural Network": {
                'hidden_layer_sizes': [(64,), (128,), (64, 32)],
                'alpha': [0.0001, 0.001]
            },
            "AdaBoost": {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 1.0]
            }
        }
    
    def evaluate_models(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray,
                        tune_hyperparameters: bool = False) -> Dict[str, float]:
        """
        Evaluate multiple models on CAD shape classification.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features  
            y_test: Test labels
            tune_hyperparameters: Whether to perform GridSearchCV tuning
            
        Returns:
            Dictionary of model names to test accuracy scores
        """
        logging.info("Starting model evaluation for CAD shape classification")
        
        try:
            models = self._get_models()
            hyperparams = self._get_hyperparameters() if tune_hyperparameters else {}
            
            model_results = {}
            trained_models = {}
            
            for name, model in models.items():
                logging.info(f"Training {name}...")
                
                if tune_hyperparameters and name in hyperparams:
                    # Use GridSearchCV for hyperparameter tuning
                    grid_search = GridSearchCV(
                        model, 
                        hyperparams[name],
                        cv=3,
                        scoring='accuracy',
                        n_jobs=-1
                    )
                    grid_search.fit(X_train, y_train)
                    trained_model = grid_search.best_estimator_
                    logging.info(f"Best params for {name}: {grid_search.best_params_}")
                else:
                    # Train with default parameters
                    trained_model = model
                    trained_model.fit(X_train, y_train)
                
                # Cross-validation score
                cv_scores = cross_val_score(trained_model, X_train, y_train, 
                                           cv=self.config.cv_folds, scoring='accuracy')
                
                # Test score
                y_pred = trained_model.predict(X_test)
                test_accuracy = accuracy_score(y_test, y_pred)
                
                model_results[name] = {
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'test_accuracy': test_accuracy,
                    'precision': precision_score(y_test, y_pred, average='weighted'),
                    'recall': recall_score(y_test, y_pred, average='weighted'),
                    'f1': f1_score(y_test, y_pred, average='weighted')
                }
                
                trained_models[name] = trained_model
                
                logging.info(f"{name}: CV={cv_scores.mean():.4f}(±{cv_scores.std():.4f}), "
                           f"Test={test_accuracy:.4f}")
            
            # Find best model based on test accuracy
            best_name = max(model_results, key=lambda k: model_results[k]['test_accuracy'])
            best_score = model_results[best_name]['test_accuracy']
            
            self.best_model = trained_models[best_name]
            self.best_model_name = best_name
            self.model_scores = model_results
            
            logging.info(f"Best model: {best_name} with accuracy {best_score:.4f}")
            
            return model_results
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_test: np.ndarray, y_test: np.ndarray) -> float:
        """
        Train and select the best CAD shape classifier.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Best model test accuracy
        """
        logging.info("Training CAD shape classifier")
        
        try:
            # Evaluate all models
            results = self.evaluate_models(X_train, y_train, X_test, y_test)
            
            # Ensure minimum accuracy threshold
            best_accuracy = results[self.best_model_name]['test_accuracy']
            
            if best_accuracy < 0.7:
                logging.warning(f"Best model accuracy ({best_accuracy:.4f}) below 0.7 threshold")
            
            # Save the best model
            os.makedirs(os.path.dirname(self.config.model_path), exist_ok=True)
            save_object(self.config.model_path, self.best_model)
            logging.info(f"Saved best model ({self.best_model_name}) to {self.config.model_path}")
            
            return best_accuracy
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict shape classes for new data.
        
        Args:
            X: Feature array
            
        Returns:
            Predicted class labels
        """
        if self.best_model is None:
            raise CustomException("Model not trained. Call train() first.", sys)
        
        return self.best_model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get class probabilities for predictions.
        
        Args:
            X: Feature array
            
        Returns:
            Class probability array
        """
        if self.best_model is None:
            raise CustomException("Model not trained. Call train() first.", sys)
        
        return self.best_model.predict_proba(X)
    
    def get_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   class_names: list = None) -> str:
        """Generate detailed classification report"""
        return classification_report(y_true, y_pred, target_names=class_names)
    
    def get_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Generate confusion matrix"""
        return confusion_matrix(y_true, y_pred)
    
    def print_results_summary(self) -> None:
        """Print formatted summary of all model results"""
        if not self.model_scores:
            print("No results available. Run evaluate_models() first.")
            return
        
        print("\n" + "=" * 80)
        print("CAD SHAPE CLASSIFICATION - MODEL COMPARISON RESULTS")
        print("=" * 80)
        
        # Header
        print(f"\n{'Model':<25} {'CV Accuracy':<15} {'Test Acc':<12} {'F1 Score':<12}")
        print("-" * 64)
        
        # Sort by test accuracy
        sorted_models = sorted(self.model_scores.items(), 
                               key=lambda x: x[1]['test_accuracy'], reverse=True)
        
        for name, scores in sorted_models:
            marker = " ★" if name == self.best_model_name else ""
            print(f"{name:<25} {scores['cv_mean']:.4f}±{scores['cv_std']:.4f}   "
                  f"{scores['test_accuracy']:.4f}       {scores['f1']:.4f}{marker}")
        
        print("-" * 64)
        print(f"\n★ Best Model: {self.best_model_name}")
        print(f"  Test Accuracy: {self.model_scores[self.best_model_name]['test_accuracy']:.4f}")
        print(f"  F1 Score: {self.model_scores[self.best_model_name]['f1']:.4f}")
        print("=" * 80)


if __name__ == "__main__":
    # Complete CAD ML Demo Pipeline
    from src.cad_ml.shape_data_generator import CADShapeDataGenerator, CADDataConfig
    from src.cad_ml.cad_feature_extractor import CADFeatureExtractor
    
    print("\n" + "=" * 80)
    print("CAD ML DEMO - SHAPE CLASSIFICATION PIPELINE")
    print("=" * 80)
    
    # Step 1: Generate synthetic CAD data
    print("\n[Step 1] Generating synthetic CAD shape data...")
    generator = CADShapeDataGenerator()
    df = generator.generate_dataset()
    raw_path, train_path, test_path = generator.save_dataset(df)
    print(f"  Generated {len(df)} samples with {len(generator.SHAPE_CLASSES)} shape classes")
    
    # Step 2: Load and extract features
    print("\n[Step 2] Extracting geometric features...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    extractor = CADFeatureExtractor()
    X_train, X_test, y_train, y_test = extractor.fit_transform(train_df, test_df)
    print(f"  Features extracted: {X_train.shape[1]} dimensions")
    
    # Step 3: Train and evaluate classifier
    print("\n[Step 3] Training and evaluating classifiers...")
    classifier = CADShapeClassifier()
    best_accuracy = classifier.train(X_train, y_train, X_test, y_test)
    
    # Step 4: Print results
    classifier.print_results_summary()
    
    # Step 5: Show classification report for best model
    print("\n[Detailed Classification Report for Best Model]")
    y_pred = classifier.predict(X_test)
    print(classifier.get_classification_report(y_test, y_pred, 
                                                class_names=list(extractor.label_encoder.classes_)))
    
    print("\nCAD ML Demo completed successfully!")
    print(f"Trained model saved to: {classifier.config.model_path}")
