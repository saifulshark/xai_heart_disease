"""
Data Loader for UCI Heart Disease Dataset
Loads the processed Cleveland dataset and assigns proper column names
"""

import pandas as pd
import numpy as np
from pathlib import Path


class HeartDiseaseDataLoader:
    """
    Loader for the UCI Heart Disease dataset (Cleveland database)
    
    Attributes:
        data_path (Path): Path to the dataset file
        column_names (list): List of feature names according to documentation
    """
    
    def __init__(self, data_path: str):
        """
        Initialize the data loader
        
        Args:
            data_path: Path to processed.cleveland.data file
        """
        self.data_path = Path(data_path)
        
        # Column names from heart-disease.names documentation
        self.column_names = [
            'age',              # Age in years
            'sex',              # Sex (1 = male; 0 = female)
            'cp',               # Chest pain type (1-4)
            'trestbps',         # Resting blood pressure (mm Hg)
            'chol',             # Serum cholesterol (mg/dl)
            'fbs',              # Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
            'restecg',          # Resting electrocardiographic results (0-2)
            'thalach',          # Maximum heart rate achieved
            'exang',            # Exercise induced angina (1 = yes; 0 = no)
            'oldpeak',          # ST depression induced by exercise relative to rest
            'slope',            # Slope of the peak exercise ST segment (1-3)
            'ca',               # Number of major vessels (0-3) colored by fluoroscopy
            'thal',             # Thalassemia (3 = normal; 6 = fixed defect; 7 = reversible defect)
            'target'            # Diagnosis of heart disease (0-4, where 0 = no disease)
        ]
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the Cleveland heart disease dataset
        
        Returns:
            DataFrame with proper column names
        """
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {self.data_path}. "
                f"Please ensure the file exists at the specified location."
            )
        
        # Load data (no header in original file)
        df = pd.read_csv(
            self.data_path,
            names=self.column_names,
            na_values='?'  # Missing values are represented as '?'
        )
        
        print(f"Dataset loaded successfully from {self.data_path}")
        print(f"Shape: {df.shape}")
        print(f"Missing values:\n{df.isnull().sum()}")
        
        return df
    
    def get_feature_descriptions(self) -> dict:
        """
        Get clinical descriptions of features
        
        Returns:
            Dictionary mapping feature names to descriptions
        """
        descriptions = {
            'age': 'Age in years',
            'sex': 'Sex (1=male, 0=female)',
            'cp': 'Chest pain type (1=typical angina, 2=atypical angina, 3=non-anginal pain, 4=asymptomatic)',
            'trestbps': 'Resting blood pressure (mm Hg on admission)',
            'chol': 'Serum cholesterol (mg/dl)',
            'fbs': 'Fasting blood sugar > 120 mg/dl (1=true, 0=false)',
            'restecg': 'Resting ECG results (0=normal, 1=ST-T wave abnormality, 2=left ventricular hypertrophy)',
            'thalach': 'Maximum heart rate achieved',
            'exang': 'Exercise induced angina (1=yes, 0=no)',
            'oldpeak': 'ST depression induced by exercise relative to rest',
            'slope': 'Slope of peak exercise ST segment (1=upsloping, 2=flat, 3=downsloping)',
            'ca': 'Number of major vessels (0-3) colored by fluoroscopy',
            'thal': 'Thalassemia (3=normal, 6=fixed defect, 7=reversible defect)',
            'target': 'Heart disease diagnosis (0=no disease, 1-4=disease present)'
        }
        return descriptions


if __name__ == "__main__":
    # Example usage
    loader = HeartDiseaseDataLoader("../data/processed.cleveland.data")
    df = loader.load_data()
    print("\nFirst few rows:")
    print(df.head())
    print("\nDataset info:")
    print(df.info())
