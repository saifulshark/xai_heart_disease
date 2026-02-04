"""
Evaluation and Visualization Module
Generate performance tables and ROC curves for publication
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict


class ModelEvaluator:
    """
    Generate publication-quality evaluation outputs
    """
    
    def __init__(self, results_dir: str = "../results"):
        """
        Initialize evaluator
        
        Args:
            results_dir: Directory to save results
        """
        self.results_dir = Path(results_dir)
        self.tables_dir = self.results_dir / "tables"
        self.figures_dir = self.results_dir / "figures"
        
        # Create directories
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Set publication-quality plotting style
        self._set_plot_style()
    
    def _set_plot_style(self):
        """Set matplotlib style for publication-quality figures"""
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['xtick.labelsize'] = 11
        plt.rcParams['ytick.labelsize'] = 11
        plt.rcParams['legend.fontsize'] = 11
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['savefig.bbox'] = 'tight'
        sns.set_palette("husl")
    
    def save_performance_table(self, results_df: pd.DataFrame, 
                               filename: str = "model_performance.csv"):
        """
        Save performance comparison table
        
        Args:
            results_df: DataFrame with model performance metrics
            filename: Output filename
        """
        filepath = self.tables_dir / filename
        results_df.to_csv(filepath, index=False, float_format='%.4f')
        print(f"Performance table saved to {filepath}")
        
        # Also save formatted version for LaTeX
        latex_filepath = self.tables_dir / filename.replace('.csv', '.tex')
        results_df.to_latex(latex_filepath, index=False, float_format='%.4f')
        print(f"LaTeX table saved to {latex_filepath}")
    
    def plot_roc_curves(self, roc_data: Dict, 
                       filename: str = "roc_curves.png"):
        """
        Plot ROC curves for all models
        
        Args:
            roc_data: Dictionary containing ROC curve data
            filename: Output filename
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot each model's ROC curve
        colors = plt.cm.tab10(np.linspace(0, 1, len(roc_data)))
        
        for (name, data), color in zip(roc_data.items(), colors):
            ax.plot(
                data['fpr'], 
                data['tpr'],
                label=f"{name} (AUC = {data['auc']:.3f})",
                linewidth=2.5,
                color=color
            )
        
        # Plot diagonal (random classifier)
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
        
        # Formatting
        ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')
        ax.set_title('ROC Curves - Heart Disease Classification', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='lower right', fontsize=12, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        
        # Save figure
        filepath = self.figures_dir / filename
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        print(f"ROC curves saved to {filepath}")
        plt.close()
    
    def plot_metric_comparison(self, results_df: pd.DataFrame,
                              metrics: list = ['Test_Accuracy', 'Test_Recall', 
                                             'Test_F1', 'Test_ROC_AUC'],
                              filename: str = "metric_comparison.png"):
        """
        Plot bar chart comparing metrics across models
        
        Args:
            results_df: DataFrame with model performance
            metrics: List of metrics to compare
            filename: Output filename
        """
        # Prepare data
        data_to_plot = results_df[['Model'] + metrics].copy()
        data_melted = pd.melt(
            data_to_plot, 
            id_vars=['Model'], 
            var_name='Metric', 
            value_name='Score'
        )
        
        # Clean metric names
        data_melted['Metric'] = data_melted['Metric'].str.replace('Test_', '')
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        sns.barplot(
            data=data_melted,
            x='Metric',
            y='Score',
            hue='Model',
            ax=ax
        )
        
        ax.set_xlabel('Evaluation Metric', fontsize=14, fontweight='bold')
        ax.set_ylabel('Score', fontsize=14, fontweight='bold')
        ax.set_title('Model Performance Comparison on Test Set', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.legend(title='Model', fontsize=11, title_fontsize=12)
        ax.set_ylim([0, 1.05])
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', fontsize=9, padding=3)
        
        # Save figure
        filepath = self.figures_dir / filename
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        print(f"Metric comparison saved to {filepath}")
        plt.close()
    
    def create_summary_report(self, results_df: pd.DataFrame,
                             filename: str = "summary_report.txt"):
        """
        Create text summary of results
        
        Args:
            results_df: DataFrame with model performance
            filename: Output filename
        """
        filepath = self.tables_dir / filename
        
        with open(filepath, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("HEART DISEASE CLASSIFICATION - MODEL EVALUATION SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            # Best models by metric
            f.write("BEST PERFORMING MODELS BY METRIC:\n")
            f.write("-" * 80 + "\n")
            
            metrics = {
                'Test_Accuracy': 'Accuracy',
                'Test_Precision': 'Precision',
                'Test_Recall': 'Recall (Sensitivity)',
                'Test_F1': 'F1-Score',
                'Test_ROC_AUC': 'ROC-AUC'
            }
            
            for col, name in metrics.items():
                best_idx = results_df[col].idxmax()
                best_model = results_df.loc[best_idx, 'Model']
                best_score = results_df.loc[best_idx, col]
                f.write(f"{name:25s}: {best_model:20s} ({best_score:.4f})\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("DETAILED RESULTS (TEST SET):\n")
            f.write("=" * 80 + "\n\n")
            
            # Detailed results for each model
            for _, row in results_df.iterrows():
                f.write(f"\n{row['Model']}:\n")
                f.write(f"  Accuracy:  {row['Test_Accuracy']:.4f}\n")
                f.write(f"  Precision: {row['Test_Precision']:.4f}\n")
                f.write(f"  Recall:    {row['Test_Recall']:.4f}\n")
                f.write(f"  F1-Score:  {row['Test_F1']:.4f}\n")
                f.write(f"  ROC-AUC:   {row['Test_ROC_AUC']:.4f}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            
        print(f"Summary report saved to {filepath}")
    
    def generate_all_outputs(self, results_df: pd.DataFrame, roc_data: Dict):
        """
        Generate all evaluation outputs
        
        Args:
            results_df: DataFrame with model performance
            roc_data: Dictionary with ROC curve data
        """
        print("\nGenerating evaluation outputs...")
        
        # Save performance table
        self.save_performance_table(results_df)
        
        # Plot ROC curves
        self.plot_roc_curves(roc_data)
        
        # Plot metric comparison
        self.plot_metric_comparison(results_df)
        
        # Create summary report
        self.create_summary_report(results_df)
        
        print("\n✓ All evaluation outputs generated successfully!")


if __name__ == "__main__":
    # This would typically be called from the main experiment script
    print("Evaluation module loaded successfully")
