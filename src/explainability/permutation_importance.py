"""
Permutation Feature Importance Module
Calculate feature importance by measuring performance degradation
when feature values are randomly permuted
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from pathlib import Path
from typing import Any


class PermutationImportanceExplainer:
    """
    Permutation-based feature importance
    
    Measures importance by evaluating model performance
    degradation when each feature is randomly shuffled
    """
    
    def __init__(self, model: Any, feature_names: list,
                 results_dir: str = "../results"):
        """
        Initialize permutation importance explainer
        
        Args:
            model: Trained model
            feature_names: List of feature names
            results_dir: Directory to save results
        """
        self.model = model
        self.feature_names = feature_names
        self.results_dir = Path(results_dir)
        self.figures_dir = self.results_dir / "figures"
        self.tables_dir = self.results_dir / "tables"
        
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        
        self.importance_result = None
        
        # Set plot style
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
    
    def calculate_importance(self, X: pd.DataFrame, y: pd.Series,
                           n_repeats: int = 10, random_state: int = 42,
                           scoring: str = 'roc_auc'):
        """
        Calculate permutation importance
        
        Args:
            X: Feature dataset
            y: Target labels
            n_repeats: Number of times to permute each feature
            random_state: Random seed
            scoring: Scoring metric ('roc_auc', 'accuracy', etc.)
        """
        print(f"Calculating permutation importance (n_repeats={n_repeats})...")
        
        self.importance_result = permutation_importance(
            self.model,
            X,
            y,
            n_repeats=n_repeats,
            random_state=random_state,
            scoring=scoring,
            n_jobs=-1
        )
        
        print("Permutation importance calculated")
    
    def get_importance_dataframe(self) -> pd.DataFrame:
        """
        Get importance results as DataFrame
        
        Returns:
            DataFrame with feature importance statistics
        """
        if self.importance_result is None:
            raise ValueError("Importance not calculated. Call calculate_importance() first.")
        
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance_Mean': self.importance_result.importances_mean,
            'Importance_Std': self.importance_result.importances_std
        })
        
        # Sort by mean importance
        importance_df = importance_df.sort_values('Importance_Mean', ascending=False)
        
        return importance_df
    
    def plot_importance(self, model_name: str = "model",
                       top_n: int = None, filename: str = None):
        """
        Plot permutation importance with error bars
        
        Args:
            model_name: Name of model for title
            top_n: Number of top features to display (None = all)
            filename: Custom filename (optional)
        """
        if self.importance_result is None:
            raise ValueError("Importance not calculated. Call calculate_importance() first.")
        
        # Get importance DataFrame
        importance_df = self.get_importance_dataframe()
        
        # Limit to top N if specified
        if top_n is not None:
            importance_df = importance_df.head(top_n)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Sort for plotting (ascending for horizontal bar chart)
        importance_df = importance_df.sort_values('Importance_Mean', ascending=True)
        
        # Plot horizontal bars with error bars
        y_pos = np.arange(len(importance_df))
        ax.barh(
            y_pos,
            importance_df['Importance_Mean'],
            xerr=importance_df['Importance_Std'],
            alpha=0.7,
            color='steelblue',
            error_kw={'elinewidth': 2, 'capsize': 5}
        )
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(importance_df['Feature'], fontsize=11)
        ax.set_xlabel('Permutation Importance (ROC-AUC decrease)',
                     fontsize=12, fontweight='bold')
        ax.set_title(f'Permutation Feature Importance - {model_name}',
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, axis='x', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        # Save figure
        if filename is None:
            filename = f"permutation_importance_{model_name.lower().replace(' ', '_')}.png"
        
        filepath = self.figures_dir / filename
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        print(f"Permutation importance plot saved to {filepath}")
        plt.close()
    
    def save_importance_table(self, model_name: str = "model",
                             filename: str = None):
        """
        Save importance values as CSV
        
        Args:
            model_name: Name of model
            filename: Custom filename (optional)
        """
        importance_df = self.get_importance_dataframe()
        
        if filename is None:
            filename = f"permutation_importance_{model_name.lower().replace(' ', '_')}.csv"
        
        filepath = self.tables_dir / filename
        importance_df.to_csv(filepath, index=False, float_format='%.6f')
        print(f"Permutation importance table saved to {filepath}")
    
    def generate_all_outputs(self, X: pd.DataFrame, y: pd.Series,
                           model_name: str, n_repeats: int = 10,
                           top_n: int = None):
        """
        Calculate and save all permutation importance outputs
        
        Args:
            X: Feature dataset
            y: Target labels
            model_name: Name of model
            n_repeats: Number of permutation repeats
            top_n: Number of top features to plot
        """
        print(f"\nGenerating permutation importance for {model_name}...")
        
        # Calculate importance
        self.calculate_importance(X, y, n_repeats=n_repeats)
        
        # Generate outputs
        self.plot_importance(model_name, top_n=top_n)
        self.save_importance_table(model_name)
        
        print(f"✓ Permutation importance generated for {model_name}")


class ComparePermutationImportance:
    """
    Compare permutation importance across multiple models
    """
    
    def __init__(self, results_dir: str = "../results"):
        """
        Initialize comparator
        
        Args:
            results_dir: Directory to save results
        """
        self.results_dir = Path(results_dir)
        self.figures_dir = self.results_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
    
    def plot_comparison(self, importance_dfs: dict, top_n: int = 10,
                       filename: str = "permutation_importance_comparison.png"):
        """
        Create comparison plot across models
        
        Args:
            importance_dfs: Dictionary mapping model names to importance DataFrames
            top_n: Number of top features to display
            filename: Output filename
        """
        # Get unique features across all models
        all_features = set()
        for df in importance_dfs.values():
            all_features.update(df.head(top_n)['Feature'].tolist())
        
        # Create figure
        n_models = len(importance_dfs)
        fig, axes = plt.subplots(1, n_models, figsize=(8*n_models, 8), sharey=True)
        
        if n_models == 1:
            axes = [axes]
        
        for ax, (model_name, importance_df) in zip(axes, importance_dfs.items()):
            # Get top features
            top_features = importance_df.head(top_n).copy()
            top_features = top_features.sort_values('Importance_Mean', ascending=True)
            
            # Plot
            y_pos = np.arange(len(top_features))
            ax.barh(
                y_pos,
                top_features['Importance_Mean'],
                xerr=top_features['Importance_Std'],
                alpha=0.7,
                color='steelblue',
                error_kw={'elinewidth': 1.5, 'capsize': 3}
            )
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_features['Feature'], fontsize=10)
            ax.set_xlabel('Importance', fontsize=11, fontweight='bold')
            ax.set_title(model_name, fontsize=12, fontweight='bold')
            ax.grid(True, axis='x', alpha=0.3)
        
        plt.suptitle('Permutation Feature Importance Comparison',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save
        filepath = self.figures_dir / filename
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        print(f"Comparison plot saved to {filepath}")
        plt.close()


if __name__ == "__main__":
    print("Permutation importance module loaded successfully")
