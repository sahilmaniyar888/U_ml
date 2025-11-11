## End to End ML pipeline


# Student Performance Prediction - Machine Learning Pipeline

A complete machine learning project for predicting student performance using data ingestion, transformation, and model training with multiple algorithms.

## Project Structure

```
U_ML/
├── src/
│   ├── components/
│   │   ├── data_ingestion.py      # Data loading and train-test split
│   │   ├── data_transformation.py # Feature engineering and preprocessing
│   │   └── model_trainer.py       # Model training and evaluation
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

- **Data Ingestion**: Load data and perform train-test split (80-20)
- **Data Transformation**: Feature scaling, encoding, and preprocessing
- **Model Training**: Multiple algorithms (Scikit-learn, XGBoost, CatBoost)
- **Model Evaluation**: Performance metrics and model comparison
- **Logging & Exception Handling**: Custom logger and exception classes

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

- `artifacts/data.csv` - Raw dataset
- `artifacts/train.csv` - Training data
- `artifacts/test.csv` - Test data
- Console output with model performance metrics

## License

MIT
