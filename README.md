## End to End ML pipeline


# Machine Learning Pipeline

A complete machine learning project featuring:
1. **Student Performance Prediction** - Predicting academic outcomes
2. **CAD Shape Classification** - ML for Computer-Aided Design applications

## Project Structure

```
U_ML/
├── src/
│   ├── components/
│   │   ├── data_ingestion.py      # Data loading and train-test split
│   │   ├── data_transformation.py # Feature engineering and preprocessing
│   │   └── model_trainer.py       # Model training and evaluation
│   ├── cad_ml/                    # CAD Machine Learning module
│   │   ├── __init__.py
│   │   ├── shape_data_generator.py    # Synthetic CAD shape data generation
│   │   ├── cad_feature_extractor.py   # Geometric feature extraction
│   │   └── cad_shape_classifier.py    # Multi-model shape classifier
│   ├── exception.py               # Custom exception handling
│   ├── logger.py                  # Logging configuration
│   └── utils.py                   # Utility functions
├── notebook/
│   └── data/
│       └── stud.csv               # Student dataset
├── artifacts/                      # Generated train/test/raw data
├── requirements.txt               # Project dependencies
└── README.md
```

## Features

### Student Performance Module
- **Data Ingestion**: Load data and perform train-test split (80-20)
- **Data Transformation**: Feature scaling, encoding, and preprocessing
- **Model Training**: Multiple algorithms (Scikit-learn, XGBoost, CatBoost)
- **Model Evaluation**: Performance metrics and model comparison
- **Logging & Exception Handling**: Custom logger and exception classes

### CAD ML Module (NEW)
Machine learning for Computer-Aided Design applications:
- **Synthetic CAD Data Generation**: Creates geometric shape datasets with 5 shape classes
- **Geometric Feature Extraction**: Computes area, perimeter, compactness, aspect ratio
- **Multi-Model Classification**: Evaluates 8 ML algorithms for shape recognition
- **Shape Classes**: Circle, Rectangle, Triangle, Hexagon, Ellipse
- **Typical Accuracy**: ~96% on shape classification tasks

**CAD ML Use Cases:**
- Shape classification in design automation
- Quality control in manufacturing
- Design pattern recognition
- Automated design validation

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```powershell
   python -m venv venv
   ```

3. Activate virtual environment:
   ```powershell
   # On Windows
   venv\Scripts\activate
   ```

4. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

## Dependencies

- pandas
- numpy
- scikit-learn
- xgboost
- catboost
- seaborn
- matplotlib
- ipykernel
- dill

## Usage

### Student Performance Pipeline

Run the complete pipeline:

```powershell
cd e:\U_ML
python src/components/data_ingestion.py
```

This will:
1. Load student data from `notebook/data/stud.csv`
2. Save raw data to `artifacts/data.csv`
3. Split into train/test sets
4. Transform features
5. Train and evaluate models

### CAD ML Demo Pipeline

Run the CAD shape classification demo:

```bash
python -m src.cad_ml.cad_shape_classifier
```

This will:
1. Generate synthetic CAD shape data (1000 samples, 5 classes)
2. Extract geometric features (area, perimeter, compactness, etc.)
3. Train and evaluate 8 different ML classifiers
4. Select and save the best performing model
5. Display detailed classification report

**Sample Output:**
```
================================================================================
CAD SHAPE CLASSIFICATION - MODEL COMPARISON RESULTS
================================================================================

Model                     CV Accuracy     Test Acc     F1 Score    
----------------------------------------------------------------
Gradient Boosting         0.9587±0.0135   0.9650       0.9649 ★
Random Forest             0.9625±0.0088   0.9600       0.9600
...

★ Best Model: Gradient Boosting
  Test Accuracy: 0.9650
  F1 Score: 0.9649
================================================================================
```

## Project Workflow

```
Data Ingestion
    ↓
Data Transformation
    ↓
Model Training
    ↓
Model Evaluation
```

## Output

### Student Performance
- `artifacts/data.csv` - Raw dataset
- `artifacts/train.csv` - Training data
- `artifacts/test.csv` - Test data
- Console output with model performance metrics

### CAD ML
- `artifacts/cad_ml/cad_shapes.csv` - Raw CAD shape dataset
- `artifacts/cad_ml/cad_train.csv` - CAD training data
- `artifacts/cad_ml/cad_test.csv` - CAD test data
- `artifacts/cad_ml/cad_shape_model.pkl` - Trained classifier model
- `artifacts/cad_ml/cad_preprocessor.pkl` - Feature preprocessing pipeline
- `artifacts/cad_ml/cad_label_encoder.pkl` - Shape label encoder

## CAD ML Technical Details

The CAD ML module demonstrates machine learning techniques applicable to Computer-Aided Design:

### Geometric Features Extracted
| Feature | Description |
|---------|-------------|
| Area | Shape area (computed from geometric properties) |
| Perimeter | Shape perimeter/circumference |
| Aspect Ratio | Width to height ratio |
| Compactness | Circularity measure: 4π × Area / Perimeter² |
| N Vertices | Number of vertices (0 for curved shapes) |
| Log Area | Log-transformed area for scale invariance |
| Perimeter/Area Ratio | Scale-invariant shape descriptor |
| Is Polygonal | Binary flag for discrete vertex shapes |
| Circularity | Normalized compactness (0-1) |

### Models Evaluated
- Random Forest Classifier
- Gradient Boosting Classifier
- Decision Tree Classifier
- Logistic Regression
- K-Nearest Neighbors
- Support Vector Machine (SVM)
- Multi-Layer Perceptron (Neural Network)
- AdaBoost Classifier

## License

MIT
