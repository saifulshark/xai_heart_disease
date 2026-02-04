"""
Partial Dependence Plots (PDP) Module
Visualize the relationship between features and predicted outcomes
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay, partial_dependence
from pathlib import Path
from typing import Any, List, Tuple


class PartialDependenceExplainer:
    """
    Partial Dependence Plot (PDP) explainability
    
    Shows how predictions change as individual features vary
    while marginalizing over all other features
    """
    
    def __init__(self, model: Any, feature_names: list,
                 results_dir: str = "../results"):
        """
        Initialize PDP explainer
        
        Args:
            model: Trained model
            feature_names: List of feature names
            results_dir: Directory to save results
        """
        self.model = model
        self.feature_names = feature_names
        self.results_dir = Path(results_dir)
        self.figures_dir = self.results_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Set plot style
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
    
    def plot_single_pdp(self, X: pd.DataFrame, feature: str,
                       model_name: str = "model", filename: str = None):
        """
        Plot PDP for a single feature
        
        Args:
            X: Feature dataset
            feature: Feature name
            model_name: Name of model for title
            filename: Custom filename (optional)
        """
        if feature not in self.feature_names:
            raise ValueError(f"Feature '{feature}' not found in feature names")
        
        feature_idx = self.feature_names.index(feature)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Generate PDP
        display = PartialDependenceDisplay.from_estimator(
            self.model,
            X,
            features=[feature_idx],
            feature_names=self.feature_names,
            ax=ax,
            kind='average',
            grid_resolution=50
        )
        
        # Customize
        ax.set_title(f'Partial Dependence Plot - {feature} - {model_name}',
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel(feature, fontsize=12, fontweight='bold')
        ax.set_ylabel('Partial Dependence', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        # Save
        if filename is None:
            filename = f"pdp_{feature}_{model_name.lower().replace(' ', '_')}.png"
        
        filepath = self.figures_dir / filename
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        print(f"PDP for {feature} saved to {filepath}")
        plt.close()
    
    def plot_multiple_pdp(self, X: pd.DataFrame, features: List[str],
                         model_name: str = "model", filename: str = None,
                         n_cols: int = 2):
        """
        Plot PDPs for multiple features in a grid
        
        Args:
            X: Feature dataset
            features: List of feature names
            model_name: Name of model
            filename: Custom filename (optional)
            n_cols: Number of columns in grid
        """
        # Validate features
        invalid_features = [f for f in features if f not in self.feature_names]
        if invalid_features:
            raise ValueError(f"Features not found: {invalid_features}")
        
        # Get feature indices
        feature_indices = [self.feature_names.index(f) for f in features]
        
        # Calculate grid dimensions
        n_features = len(features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 6*n_rows))
        axes = axes.flatten() if n_features > 1 else [axes]
        
        # Generate PDPs
        for idx, (feature, feature_idx, ax) in enumerate(zip(features, feature_indices, axes)):
            display = PartialDependenceDisplay.from_estimator(
                self.model,
                X,
                features=[feature_idx],
                feature_names=self.feature_names,
                ax=ax,
                kind='average',
                grid_resolution=50
            )
            
            ax.set_title(f'{feature}', fontsize=12, fontweight='bold')
            ax.set_xlabel(feature, fontsize=11, fontweight='bold')
            ax.set_ylabel('Partial Dependence', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
        
        # Hide extra subplots
        for idx in range(n_features, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle(f'Partial Dependence Plots - {model_name}',
                    fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        # Save
        if filename is None:
            filename = f"pdp_multiple_{model_name.lower().replace(' ', '_')}.png"
        
        filepath = self.figures_dir / filename
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        print(f"Multiple PDPs saved to {filepath}")
        plt.close()
    
    def plot_2d_pdp(self, X: pd.DataFrame, features: Tuple[str, str],
                   model_name: str = "model", filename: str = None):
        """
        Plot 2D PDP showing interaction between two features
        
        Args:
            X: Feature dataset
            features: Tuple of two feature names
            model_name: Name of model
            filename: Custom filename (optional)
        """
        if len(features) != 2:
            raise ValueError("Exactly two features required for 2D PDP")
        
        # Validate features
        for feature in features:
            if feature not in self.feature_names:
                raise ValueError(f"Feature '{feature}' not found")
        
        # Get feature indices
        feature_indices = [self.feature_names.index(f) for f in features]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Generate 2D PDP
        display = PartialDependenceDisplay.from_estimator(
            self.model,
            X,
            features=[feature_indices],
            feature_names=self.feature_names,
            ax=ax,
            kind='average',
            grid_resolution=20
        )
        
        # Customize
        ax.set_title(
            f'2D Partial Dependence - {features[0]} vs {features[1]} - {model_name}',
            fontsize=14, fontweight='bold', pad=20
        )
        
        plt.tight_layout()
        
        # Save
        if filename is None:
            safe_feat0 = features[0].replace(' ', '_')
            safe_feat1 = features[1].replace(' ', '_')
            filename = f"pdp_2d_{safe_feat0}_{safe_feat1}_{model_name.lower().replace(' ', '_')}.png"
        
        filepath = self.figures_dir / filename
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        print(f"2D PDP for {features} saved to {filepath}")
        plt.close()
    
    def plot_clinical_pdps(self, X: pd.DataFrame, model_name: str = "model"):
        """
        Plot PDPs for clinically important features
        
        Args:
            X: Feature dataset
            model_name: Name of model
        """
        # Define clinically important features
        clinical_features = [
            'age', 'trestbps', 'chol', 'thalach', 'oldpeak'
        ]
        
        # Filter to available features
        available_features = [f for f in clinical_features if f in self.feature_names]
        
        if not available_features:
            print("No clinical features found in dataset")
            return
        
        print(f"\nGenerating PDPs for clinical features: {available_features}")
        
        # Plot multiple PDPs
        self.plot_multiple_pdp(
            X,
            available_features,
            model_name,
            filename=f"pdp_clinical_{model_name.lower().replace(' ', '_')}.png"
        )
    
    def plot_categorical_pdps(self, X: pd.DataFrame, model_name: str = "model"):
        """
        Plot PDPs for categorical features
        
        Args:
            X: Feature dataset
            model_name: Name of model
        """
        # Define categorical features
        categorical_features = [
            'sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'
        ]
        
        # Filter to available features
        available_features = [f for f in categorical_features if f in self.feature_names]
        
        if not available_features:
            print("No categorical features found in dataset")
            return
        
        print(f"\nGenerating PDPs for categorical features: {available_features}")
        
        # Plot multiple PDPs
        self.plot_multiple_pdp(
            X,
            available_features,
            model_name,
            filename=f"pdp_categorical_{model_name.lower().replace(' ', '_')}.png",
            n_cols=3
        )
    
    def generate_all_pdps(self, X: pd.DataFrame, model_name: str,
                         interaction_pairs: List[Tuple[str, str]] = None):
        """
        Generate comprehensive set of PDPs
        
        Args:
            X: Feature dataset
            model_name: Name of model
            interaction_pairs: List of feature pairs for 2D PDPs
        """
        print(f"\nGenerating Partial Dependence Plots for {model_name}...")
        
        # Clinical features
        self.plot_clinical_pdps(X, model_name)
        
        # Categorical features
        self.plot_categorical_pdps(X, model_name)
        
        # 2D interactions if specified
        if interaction_pairs:
            for pair in interaction_pairs:
                if all(f in self.feature_names for f in pair):
                    self.plot_2d_pdp(X, pair, model_name)
        
        print(f"✓ PDPs generated for {model_name}")


if __name__ == "__main__":
    print("Partial Dependence Plot module loaded successfully")
