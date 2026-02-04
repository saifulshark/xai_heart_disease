"""
SHAP (SHapley Additive exPlanations) Module
Generate global and local explanations using SHAP values
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from pathlib import Path
from typing import Any, List


class SHAPExplainer:
    """
    SHAP-based model explainability
    
    Generates:
    - Global feature importance (summary plots)
    - Local explanations for individual predictions
    """
    
    def __init__(self, model: Any, X_background: pd.DataFrame, 
                 feature_names: List[str], results_dir: str = "../results"):
        """
        Initialize SHAP explainer
        
        Args:
            model: Trained model
            X_background: Background dataset for SHAP (typically training data)
            feature_names: List of feature names
            results_dir: Directory to save results
        """
        self.model = model
        self.X_background = X_background
        self.feature_names = feature_names
        self.results_dir = Path(results_dir)
        self.figures_dir = self.results_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize SHAP explainer
        self.explainer = None
        self.shap_values = None
        self.expected_value = None
        self.class_index = None
        
        # Set plot style
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
    
    def initialize_explainer(self, model_type: str = "tree"):
        """
        Initialize appropriate SHAP explainer based on model type
        
        Args:
            model_type: Type of model ('tree', 'linear', 'kernel')
        """
        if model_type == "tree":
            # For tree-based models (RandomForest, GradientBoosting, DecisionTree)
            self.explainer = shap.TreeExplainer(self.model)
        elif model_type == "linear":
            # For linear models (LogisticRegression)
            self.explainer = shap.LinearExplainer(self.model, self.X_background)
        else:
            # For general models (kernel explainer - slower)
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba, 
                shap.sample(self.X_background, 100)
            )
        
        print(f"SHAP {model_type} explainer initialized")
    
    def calculate_shap_values(self, X: pd.DataFrame):
        """
        Calculate SHAP values for dataset
        
        Args:
            X: Dataset to explain
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call initialize_explainer() first.")
        
        print("Calculating SHAP values...")
        sv = self.explainer.shap_values(X)
        ev = getattr(self.explainer, 'expected_value', None)
        
        # Normalize shap values to a 2D array (n_samples, n_features) when possible,
        # selecting the positive class for binary classification
        if isinstance(sv, list):
            if len(sv) >= 2:
                self.class_index = 1
                self.shap_values = sv[1]
            else:
                self.class_index = 0
                self.shap_values = sv[0]
            # pick expected value for the same class when available
            if isinstance(ev, (list, np.ndarray)):
                try:
                    self.expected_value = ev[self.class_index]
                except Exception:
                    self.expected_value = ev
            else:
                self.expected_value = ev
        elif isinstance(sv, np.ndarray):
            if sv.ndim == 3:
                # handle shapes like (n_samples, n_features, n_classes)
                if sv.shape[0] == X.shape[0] and sv.shape[2] >= 2:
                    self.class_index = 1
                    self.shap_values = sv[..., self.class_index]
                # handle shapes like (n_classes, n_samples, n_features)
                elif sv.shape[0] >= 2 and sv.shape[1] == X.shape[0]:
                    self.class_index = 1
                    self.shap_values = sv[self.class_index]
                else:
                    # fallback: try to squeeze last axis
                    if sv.shape[-1] == 1:
                        self.shap_values = sv.squeeze(-1)
                        self.class_index = None
                    else:
                        # pick last class as default
                        self.class_index = sv.shape[-1] - 1
                        self.shap_values = sv[..., self.class_index]
                # pick expected value similarly if available
                if isinstance(ev, (list, np.ndarray)):
                    try:
                        self.expected_value = ev[self.class_index]
                    except Exception:
                        self.expected_value = ev
                else:
                    self.expected_value = ev
            elif sv.ndim == 2:
                self.shap_values = sv
                self.class_index = None
                self.expected_value = ev
            else:
                raise ValueError(f"Unrecognized shap_values array shape: {sv.shape}")
        else:
            raise ValueError("Unrecognized shap_values type")
        
        print("SHAP values calculated")
    
    def plot_summary(self, X: pd.DataFrame, model_name: str = "model",
                    filename: str = None):
        """
        Create SHAP summary plot (global feature importance)
        
        Args:
            X: Dataset
            model_name: Name of model for filename
            filename: Custom filename (optional)
        """
        if self.shap_values is None:
            self.calculate_shap_values(X)
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values, 
            X,
            feature_names=self.feature_names,
            show=False,
            plot_size=(10, 8)
        )
        plt.title(f'SHAP Feature Importance - {model_name}', 
                 fontsize=14, fontweight='bold', pad=20)
        
        if filename is None:
            filename = f"shap_summary_{model_name.lower().replace(' ', '_')}.png"
        
        filepath = self.figures_dir / filename
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        print(f"SHAP summary plot saved to {filepath}")
        plt.close()
    
    def plot_bar(self, X: pd.DataFrame, model_name: str = "model",
                filename: str = None):
        """
        Create SHAP bar plot (mean absolute SHAP values)
        
        Args:
            X: Dataset
            model_name: Name of model for filename
            filename: Custom filename (optional)
        """
        if self.shap_values is None:
            self.calculate_shap_values(X)
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values,
            X,
            feature_names=self.feature_names,
            plot_type="bar",
            show=False
        )
        plt.title(f'SHAP Global Feature Importance - {model_name}',
                 fontsize=14, fontweight='bold', pad=20)
        
        if filename is None:
            filename = f"shap_bar_{model_name.lower().replace(' ', '_')}.png"
        
        filepath = self.figures_dir / filename
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        print(f"SHAP bar plot saved to {filepath}")
        plt.close()
    
    def plot_waterfall(self, X: pd.DataFrame, index: int, 
                      model_name: str = "model", filename: str = None):
        """
        Create SHAP waterfall plot for a single prediction
        
        Args:
            X: Dataset
            index: Index of sample to explain
            model_name: Name of model for filename
            filename: Custom filename (optional)
        """
        if self.shap_values is None:
            self.calculate_shap_values(X)
        
        # Get values for the chosen sample and ensure 1D array
        vals = self.shap_values[index]
        if isinstance(vals, np.ndarray) and vals.ndim == 2:
            # select class column if known, else pick last column
            if self.class_index is not None:
                vals = vals[:, self.class_index]
            else:
                vals = vals[:, -1]
        
        # Determine base value (scalar) from expected_value or explainer
        base_value = getattr(self, 'expected_value', None)
        if isinstance(base_value, (list, np.ndarray)):
            try:
                base_value = base_value[self.class_index] if self.class_index is not None else float(base_value[0])
            except Exception:
                base_value = float(base_value[0])
        if base_value is None:
            base_value = 0
        
        explanation = shap.Explanation(
            values=vals,
            base_values=base_value,
            data=X.iloc[index].values,
            feature_names=self.feature_names
        )
        
        plt.figure(figsize=(10, 8))
        shap.waterfall_plot(explanation, show=False)
        plt.title(f'SHAP Waterfall Plot - Sample {index} - {model_name}',
                 fontsize=14, fontweight='bold', pad=20)
        
        if filename is None:
            filename = f"shap_waterfall_{model_name.lower().replace(' ', '_')}_sample_{index}.png"
        
        filepath = self.figures_dir / filename
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        print(f"SHAP waterfall plot saved to {filepath}")
        plt.close()
    
    def plot_force(self, X: pd.DataFrame, index: int,
                  model_name: str = "model", filename: str = None):
        """
        Create SHAP force plot for a single prediction
        
        Args:
            X: Dataset
            index: Index of sample to explain
            model_name: Name of model for filename
            filename: Custom filename (optional)
        """
        if self.shap_values is None:
            self.calculate_shap_values(X)
        
        # Determine expected (base) value using calculated expected_value if available
        expected_value = getattr(self, 'expected_value', None)
        if isinstance(expected_value, (list, np.ndarray)):
            try:
                expected_value = expected_value[self.class_index] if self.class_index is not None else float(expected_value[0])
            except Exception:
                expected_value = float(expected_value[0])
        if expected_value is None:
            expected_value = (self.explainer.expected_value[1]
                              if isinstance(getattr(self.explainer, 'expected_value', None), (list, np.ndarray))
                              else getattr(self.explainer, 'expected_value', 0))
        
        # Get values for the chosen sample and ensure correct shape
        vals = self.shap_values[index]
        if isinstance(vals, np.ndarray) and vals.ndim == 2:
            if self.class_index is not None:
                vals = vals[:, self.class_index]
            else:
                vals = vals[:, -1]
        
        # Create force plot
        shap.force_plot(
            expected_value,
            vals,
            X.iloc[index],
            feature_names=self.feature_names,
            matplotlib=True,
            show=False
        )
        
        if filename is None:
            filename = f"shap_force_{model_name.lower().replace(' ', '_')}_sample_{index}.png"
        
        filepath = self.figures_dir / filename
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        print(f"SHAP force plot saved to {filepath}")
        plt.close()
    
    def get_top_features(self, n: int = 10) -> pd.DataFrame:
        """
        Get top N most important features by mean absolute SHAP value
        
        Args:
            n: Number of top features to return
            
        Returns:
            DataFrame with feature importance rankings
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not calculated")
        
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Mean_Abs_SHAP': mean_abs_shap
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Mean_Abs_SHAP', ascending=False)
        
        return importance_df.head(n)
    
    def generate_all_plots(self, X: pd.DataFrame, model_name: str,
                          sample_indices: List[int] = [0, 1, 2]):
        """
        Generate all SHAP visualizations
        
        Args:
            X: Dataset to explain
            model_name: Name of model
            sample_indices: Indices of samples for local explanations
        """
        print(f"\nGenerating SHAP explanations for {model_name}...")
        
        # Calculate SHAP values once
        if self.shap_values is None:
            self.calculate_shap_values(X)
        
        # Global explanations
        self.plot_summary(X, model_name)
        self.plot_bar(X, model_name)
        
        # Local explanations for selected samples
        for idx in sample_indices:
            if idx < len(X):
                self.plot_waterfall(X, idx, model_name)
        
        print(f"✓ SHAP explanations generated for {model_name}")


if __name__ == "__main__":
    print("SHAP explainability module loaded successfully")
