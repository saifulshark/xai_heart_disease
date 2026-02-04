"""
LIME (Local Interpretable Model-agnostic Explanations) Module
Generate local explanations for individual predictions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lime import lime_tabular
from pathlib import Path
from typing import Any, List


class LIMEExplainer:
    """
    LIME-based local model explainability
    
    Generates local explanations showing which features
    contributed most to individual predictions
    """
    
    def __init__(self, model: Any, X_train: pd.DataFrame,
                 feature_names: List[str], class_names: List[str] = None,
                 results_dir: str = "../results"):
        """
        Initialize LIME explainer
        
        Args:
            model: Trained model with predict_proba method
            X_train: Training data for reference
            feature_names: List of feature names
            class_names: List of class names (default: ['No Disease', 'Disease'])
            results_dir: Directory to save results
        """
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        self.class_names = class_names or ['No Disease', 'Disease']
        
        self.results_dir = Path(results_dir)
        self.figures_dir = self.results_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize LIME explainer
        self.explainer = lime_tabular.LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=feature_names,
            class_names=self.class_names,
            mode='classification',
            random_state=42
        )
        
        # Set plot style
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        
        print("LIME explainer initialized")
    
    def explain_instance(self, instance: np.ndarray, num_features: int = 10) -> Any:
        """
        Generate LIME explanation for a single instance
        
        Args:
            instance: Single data instance to explain
            num_features: Number of top features to show
            
        Returns:
            LIME Explanation object
        """
        explanation = self.explainer.explain_instance(
            data_row=instance,
            predict_fn=self.model.predict_proba,
            num_features=num_features,
            num_samples=5000
        )
        
        return explanation
    
    def plot_explanation(self, X: pd.DataFrame, index: int,
                        model_name: str = "model", num_features: int = 10,
                        filename: str = None):
        """
        Plot LIME explanation for a specific sample
        
        Args:
            X: Dataset
            index: Index of sample to explain
            model_name: Name of model for filename
            num_features: Number of features to display
            filename: Custom filename (optional)
        """
        # Get instance
        instance = X.iloc[index].values
        
        # Generate explanation
        explanation = self.explain_instance(instance, num_features)
        
        # Get prediction
        prediction = self.model.predict(instance.reshape(1, -1))[0]
        proba = self.model.predict_proba(instance.reshape(1, -1))[0]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get explanation as list
        exp_list = explanation.as_list()
        
        # Separate features and values
        features = [item[0] for item in exp_list]
        values = [item[1] for item in exp_list]
        
        # Create color array (positive = green, negative = red)
        colors = ['green' if v > 0 else 'red' for v in values]
        
        # Create horizontal bar plot
        y_pos = np.arange(len(features))
        ax.barh(y_pos, values, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features, fontsize=10)
        ax.set_xlabel('Feature Contribution', fontsize=12, fontweight='bold')
        ax.set_title(
            f'LIME Explanation - Sample {index} - {model_name}\n'
            f'Prediction: {self.class_names[prediction]} '
            f'(Probability: {proba[prediction]:.3f})',
            fontsize=13, fontweight='bold', pad=20
        )
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax.grid(True, axis='x', alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.7, label='Supports Disease'),
            Patch(facecolor='red', alpha=0.7, label='Opposes Disease')
        ]
        ax.legend(handles=legend_elements, loc='best', fontsize=10)
        
        plt.tight_layout()
        
        # Save figure
        if filename is None:
            filename = f"lime_{model_name.lower().replace(' ', '_')}_sample_{index}.png"
        
        filepath = self.figures_dir / filename
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        print(f"LIME explanation saved to {filepath}")
        plt.close()
    
    def save_explanation_text(self, X: pd.DataFrame, index: int,
                             model_name: str = "model", num_features: int = 10,
                             filename: str = None):
        """
        Save LIME explanation as text file
        
        Args:
            X: Dataset
            index: Index of sample to explain
            model_name: Name of model
            num_features: Number of features to include
            filename: Custom filename (optional)
        """
        # Get instance
        instance = X.iloc[index].values
        
        # Generate explanation
        explanation = self.explain_instance(instance, num_features)
        
        # Get prediction
        prediction = self.model.predict(instance.reshape(1, -1))[0]
        proba = self.model.predict_proba(instance.reshape(1, -1))[0]
        
        # Create text file
        if filename is None:
            filename = f"lime_{model_name.lower().replace(' ', '_')}_sample_{index}.txt"
        
        filepath = self.results_dir / "tables" / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"LIME EXPLANATION - SAMPLE {index} - {model_name}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Prediction: {self.class_names[prediction]}\n")
            f.write(f"Probability: {proba[prediction]:.4f}\n\n")
            
            f.write("Feature Contributions:\n")
            f.write("-" * 80 + "\n")
            
            for feature, contribution in explanation.as_list():
                direction = "→ Disease" if contribution > 0 else "→ No Disease"
                f.write(f"{feature:40s}: {contribution:+.4f} {direction}\n")
            
            f.write("=" * 80 + "\n")
        
        print(f"LIME text explanation saved to {filepath}")
    
    def generate_explanations(self, X: pd.DataFrame, model_name: str,
                            sample_indices: List[int] = [0, 1, 2],
                            num_features: int = 10):
        """
        Generate LIME explanations for multiple samples
        
        Args:
            X: Dataset to explain
            model_name: Name of model
            sample_indices: Indices of samples to explain
            num_features: Number of features to show per explanation
        """
        print(f"\nGenerating LIME explanations for {model_name}...")
        
        for idx in sample_indices:
            if idx < len(X):
                self.plot_explanation(X, idx, model_name, num_features)
                self.save_explanation_text(X, idx, model_name, num_features)
        
        print(f"✓ LIME explanations generated for {model_name}")
    
    def compare_samples(self, X: pd.DataFrame, indices: List[int],
                       model_name: str = "model", num_features: int = 8,
                       filename: str = None):
        """
        Create comparison plot of LIME explanations for multiple samples
        
        Args:
            X: Dataset
            indices: List of sample indices to compare
            model_name: Name of model
            num_features: Number of features per sample
            filename: Custom filename (optional)
        """
        n_samples = len(indices)
        fig, axes = plt.subplots(1, n_samples, figsize=(8*n_samples, 6))
        
        if n_samples == 1:
            axes = [axes]
        
        for i, (ax, idx) in enumerate(zip(axes, indices)):
            instance = X.iloc[idx].values
            explanation = self.explain_instance(instance, num_features)
            
            # Get prediction
            prediction = self.model.predict(instance.reshape(1, -1))[0]
            proba = self.model.predict_proba(instance.reshape(1, -1))[0]
            
            # Get explanation as list
            exp_list = explanation.as_list()
            features = [item[0] for item in exp_list]
            values = [item[1] for item in exp_list]
            colors = ['green' if v > 0 else 'red' for v in values]
            
            # Plot
            y_pos = np.arange(len(features))
            ax.barh(y_pos, values, color=colors, alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features, fontsize=9)
            ax.set_xlabel('Contribution', fontsize=10, fontweight='bold')
            ax.set_title(
                f'Sample {idx}\n{self.class_names[prediction]} ({proba[prediction]:.2f})',
                fontsize=11, fontweight='bold'
            )
            ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
            ax.grid(True, axis='x', alpha=0.3)
        
        plt.suptitle(f'LIME Explanations Comparison - {model_name}',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save
        if filename is None:
            filename = f"lime_comparison_{model_name.lower().replace(' ', '_')}.png"
        
        filepath = self.figures_dir / filename
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        print(f"LIME comparison saved to {filepath}")
        plt.close()


if __name__ == "__main__":
    print("LIME explainability module loaded successfully")
