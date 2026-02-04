"""
Model Training Module
Implements multiple classifiers for heart disease prediction
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, roc_curve)
import joblib
from pathlib import Path
from typing import Dict, Any


class HeartDiseaseModelTrainer:
    """
    Train and manage multiple ML models for heart disease prediction
    
    Models:
    - Logistic Regression (interpretable baseline)
    - Decision Tree
    - Random Forest
    - Gradient Boosting
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize model trainer
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.results = {}
        
    def initialize_models(self) -> Dict[str, Any]:
        """
        Initialize all models with appropriate hyperparameters
        
        Returns:
            Dictionary of initialized models
        """
        models = {
            'Logistic Regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                solver='lbfgs'
            ),
            'Decision Tree': DecisionTreeClassifier(
                random_state=self.random_state,
                max_depth=5,
                min_samples_split=10,
                min_samples_leaf=5
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=self.random_state,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8
            )
        }
        
        return models
    
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Train all models
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        self.models = self.initialize_models()
        
        print("Training models...")
        for name, model in self.models.items():
            print(f"  Training {name}...")
            model.fit(X_train, y_train)
            print(f"  ✓ {name} trained")
        
        print("\nAll models trained successfully!")
    
    def evaluate_model(self, model, X: pd.DataFrame, y: pd.Series, 
                      dataset_name: str = "Test") -> Dict[str, float]:
        """
        Evaluate a single model
        
        Args:
            model: Trained model
            X: Features
            y: True labels
            dataset_name: Name of dataset (for display)
            
        Returns:
            Dictionary of metrics
        """
        # Predictions
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1_score': f1_score(y, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y, y_pred_proba)
        }
        
        return metrics
    
    def evaluate_all_models(self, X_val: pd.DataFrame, y_val: pd.Series,
                          X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        """
        Evaluate all models on validation and test sets
        
        Args:
            X_val: Validation features
            y_val: Validation labels
            X_test: Test features
            y_test: Test labels
            
        Returns:
            DataFrame with all evaluation metrics
        """
        results = []
        
        for name, model in self.models.items():
            print(f"\nEvaluating {name}...")
            
            # Validation metrics
            val_metrics = self.evaluate_model(model, X_val, y_val, "Validation")
            
            # Test metrics
            test_metrics = self.evaluate_model(model, X_test, y_test, "Test")
            
            # Combine results
            result = {
                'Model': name,
                'Val_Accuracy': val_metrics['accuracy'],
                'Val_Precision': val_metrics['precision'],
                'Val_Recall': val_metrics['recall'],
                'Val_F1': val_metrics['f1_score'],
                'Val_ROC_AUC': val_metrics['roc_auc'],
                'Test_Accuracy': test_metrics['accuracy'],
                'Test_Precision': test_metrics['precision'],
                'Test_Recall': test_metrics['recall'],
                'Test_F1': test_metrics['f1_score'],
                'Test_ROC_AUC': test_metrics['roc_auc']
            }
            
            results.append(result)
            
            # Print test results
            print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
            print(f"  Test Recall:   {test_metrics['recall']:.4f}")
            print(f"  Test F1-Score: {test_metrics['f1_score']:.4f}")
            print(f"  Test ROC-AUC:  {test_metrics['roc_auc']:.4f}")
        
        results_df = pd.DataFrame(results)
        self.results = results_df
        
        return results_df
    
    def get_roc_curves(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Calculate ROC curves for all models
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary containing ROC curve data for each model
        """
        roc_data = {}
        
        for name, model in self.models.items():
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            roc_data[name] = {
                'fpr': fpr,
                'tpr': tpr,
                'thresholds': thresholds,
                'auc': auc
            }
        
        return roc_data
    
    def save_models(self, save_dir: str):
        """
        Save all trained models
        
        Args:
            save_dir: Directory to save models
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for name, model in self.models.items():
            filename = name.lower().replace(' ', '_') + '.pkl'
            joblib.dump(model, save_dir / filename)
        
        print(f"\nModels saved to {save_dir}")
    
    def load_models(self, load_dir: str):
        """
        Load pre-trained models
        
        Args:
            load_dir: Directory containing saved models
        """
        load_dir = Path(load_dir)
        
        model_files = {
            'Logistic Regression': 'logistic_regression.pkl',
            'Decision Tree': 'decision_tree.pkl',
            'Random Forest': 'random_forest.pkl',
            'Gradient Boosting': 'gradient_boosting.pkl'
        }
        
        for name, filename in model_files.items():
            filepath = load_dir / filename
            if filepath.exists():
                self.models[name] = joblib.load(filepath)
                print(f"Loaded {name}")


if __name__ == "__main__":
    # Example usage
    from data_loader import HeartDiseaseDataLoader
    from preprocessing import HeartDiseasePreprocessor
    
    # Load and preprocess data
    loader = HeartDiseaseDataLoader("../data/processed.cleveland.data")
    df = loader.load_data()
    
    preprocessor = HeartDiseasePreprocessor(random_state=42)
    data = preprocessor.preprocess_pipeline(df)
    
    # Train models
    trainer = HeartDiseaseModelTrainer(random_state=42)
    trainer.train_models(data['X_train'], data['y_train'])
    
    # Evaluate models
    results = trainer.evaluate_all_models(
        data['X_val'], data['y_val'],
        data['X_test'], data['y_test']
    )
    
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(results.to_string(index=False))
