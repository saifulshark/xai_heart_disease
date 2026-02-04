"""
Data Preprocessing Module
Handles missing values, feature scaling, and train/validation/test splitting
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
from pathlib import Path


class HeartDiseasePreprocessor:
    """
    Preprocessing pipeline for heart disease dataset
    
    Handles:
    - Binary target transformation (0 vs 1-4)
    - Missing value imputation
    - Feature standardization
    - Stratified data splitting
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize preprocessor
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.imputer = None
        self.scaler = None
        self.feature_names = None
        
    def transform_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert multi-class target (0-4) to binary classification
        0 = no heart disease
        1-4 = heart disease present
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with binary target
        """
        df = df.copy()
        df['target'] = (df['target'] > 0).astype(int)
        
        print("Target distribution:")
        print(df['target'].value_counts())
        print(f"Class balance: {df['target'].value_counts(normalize=True)}")
        
        return df
    
    def handle_missing_values(self, X_train: pd.DataFrame, X_val: pd.DataFrame = None, 
                             X_test: pd.DataFrame = None) -> tuple:
        """
        Impute missing values using median strategy
        Fit on training data, transform all sets
        
        Args:
            X_train: Training features
            X_val: Validation features (optional)
            X_test: Test features (optional)
            
        Returns:
            Tuple of imputed dataframes
        """
        self.imputer = SimpleImputer(strategy='median')
        
        # Fit on training data only
        X_train_imputed = pd.DataFrame(
            self.imputer.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        result = [X_train_imputed]
        
        if X_val is not None:
            X_val_imputed = pd.DataFrame(
                self.imputer.transform(X_val),
                columns=X_val.columns,
                index=X_val.index
            )
            result.append(X_val_imputed)
        
        if X_test is not None:
            X_test_imputed = pd.DataFrame(
                self.imputer.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
            result.append(X_test_imputed)
        
        print(f"Missing values imputed using median strategy")
        return tuple(result) if len(result) > 1 else result[0]
    
    def standardize_features(self, X_train: pd.DataFrame, X_val: pd.DataFrame = None,
                           X_test: pd.DataFrame = None) -> tuple:
        """
        Standardize features using StandardScaler
        Fit on training data, transform all sets
        
        Args:
            X_train: Training features
            X_val: Validation features (optional)
            X_test: Test features (optional)
            
        Returns:
            Tuple of scaled dataframes
        """
        self.scaler = StandardScaler()
        self.feature_names = X_train.columns.tolist()
        
        # Fit on training data only
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        result = [X_train_scaled]
        
        if X_val is not None:
            X_val_scaled = pd.DataFrame(
                self.scaler.transform(X_val),
                columns=X_val.columns,
                index=X_val.index
            )
            result.append(X_val_scaled)
        
        if X_test is not None:
            X_test_scaled = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
            result.append(X_test_scaled)
        
        print("Features standardized (mean=0, std=1)")
        return tuple(result) if len(result) > 1 else result[0]
    
    def split_data(self, df: pd.DataFrame, train_size: float = 0.7, 
                   val_size: float = 0.15, test_size: float = 0.15) -> dict:
        """
        Perform stratified train/validation/test split
        
        Args:
            df: Input dataframe
            train_size: Proportion for training (default: 0.7)
            val_size: Proportion for validation (default: 0.15)
            test_size: Proportion for testing (default: 0.15)
            
        Returns:
            Dictionary containing split datasets
        """
        assert abs(train_size + val_size + test_size - 1.0) < 1e-6, \
            "Split proportions must sum to 1.0"
        
        # Separate features and target
        X = df.drop('target', axis=1)
        y = df['target']
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=test_size,
            stratify=y,
            random_state=self.random_state
        )
        
        # Second split: separate train and validation
        val_ratio = val_size / (train_size + val_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio,
            stratify=y_temp,
            random_state=self.random_state
        )
        
        print(f"\nData split completed:")
        print(f"  Training set:   {len(X_train)} samples ({len(X_train)/len(df)*100:.1f}%)")
        print(f"  Validation set: {len(X_val)} samples ({len(X_val)/len(df)*100:.1f}%)")
        print(f"  Test set:       {len(X_test)} samples ({len(X_test)/len(df)*100:.1f}%)")
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test
        }
    
    def preprocess_pipeline(self, df: pd.DataFrame) -> dict:
        """
        Complete preprocessing pipeline
        
        Args:
            df: Raw dataframe
            
        Returns:
            Dictionary with preprocessed train/val/test sets
        """
        # Transform target
        df = self.transform_target(df)
        
        # Split data
        splits = self.split_data(df)
        
        # Handle missing values
        X_train, X_val, X_test = self.handle_missing_values(
            splits['X_train'],
            splits['X_val'],
            splits['X_test']
        )
        
        # Standardize features
        X_train, X_val, X_test = self.standardize_features(
            X_train, X_val, X_test
        )
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': splits['y_train'],
            'y_val': splits['y_val'],
            'y_test': splits['y_test']
        }
    
    def save_preprocessor(self, save_dir: str):
        """
        Save imputer and scaler for future use
        
        Args:
            save_dir: Directory to save preprocessor objects
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        if self.imputer is not None:
            joblib.dump(self.imputer, save_dir / 'imputer.pkl')
        
        if self.scaler is not None:
            joblib.dump(self.scaler, save_dir / 'scaler.pkl')
        
        print(f"Preprocessor saved to {save_dir}")


if __name__ == "__main__":
    # Example usage
    from data_loader import HeartDiseaseDataLoader
    
    loader = HeartDiseaseDataLoader("../data/processed.cleveland.data")
    df = loader.load_data()
    
    preprocessor = HeartDiseasePreprocessor(random_state=42)
    data = preprocessor.preprocess_pipeline(df)
    
    print("\nPreprocessed data shapes:")
    for key, value in data.items():
        print(f"{key}: {value.shape}")
