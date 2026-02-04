"""
Utility Functions
Helper functions for the heart disease prediction project
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


def create_project_structure(base_dir: str = "."):
    """
    Create complete project directory structure
    
    Args:
        base_dir: Base directory for project
    """
    base_path = Path(base_dir)
    
    directories = [
        "data",
        "src",
        "src/explainability",
        "experiments",
        "results/tables",
        "results/figures",
        "paper_assets",
        "models"
    ]
    
    for directory in directories:
        dir_path = base_path / directory
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Project structure created at {base_path.absolute()}")


def save_experiment_config(config: Dict[str, Any], save_dir: str = "../results"):
    """
    Save experiment configuration to JSON
    
    Args:
        config: Configuration dictionary
        save_dir: Directory to save configuration
    """
    save_path = Path(save_dir) / "tables"
    save_path.mkdir(parents=True, exist_ok=True)
    
    filepath = save_path / "experiment_config.json"
    
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Experiment configuration saved to {filepath}")


def load_experiment_config(config_path: str) -> Dict[str, Any]:
    """
    Load experiment configuration from JSON
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config


def print_section_header(title: str, width: int = 80):
    """
    Print formatted section header
    
    Args:
        title: Section title
        width: Width of header
    """
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width + "\n")


def print_subsection_header(title: str, width: int = 80):
    """
    Print formatted subsection header
    
    Args:
        title: Subsection title
        width: Width of header
    """
    print("\n" + "-" * width)
    print(title)
    print("-" * width)


def format_metric(value: float, decimals: int = 4) -> str:
    """
    Format metric value for display
    
    Args:
        value: Metric value
        decimals: Number of decimal places
        
    Returns:
        Formatted string
    """
    return f"{value:.{decimals}f}"


def get_timestamp() -> str:
    """
    Get current timestamp string
    
    Returns:
        Timestamp in YYYYMMDD_HHMMSS format
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_model_artifact(obj: Any, name: str, save_dir: str = "../models"):
    """
    Save a model or preprocessor artifact
    
    Args:
        obj: Object to save
        name: Name for the saved file
        save_dir: Directory to save to
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    filepath = save_path / f"{name}.pkl"
    joblib.dump(obj, filepath)
    
    return filepath


def load_model_artifact(filepath: str) -> Any:
    """
    Load a saved model or preprocessor artifact
    
    Args:
        filepath: Path to saved artifact
        
    Returns:
        Loaded object
    """
    return joblib.load(filepath)


def ensure_reproducibility(seed: int = 42):
    """
    Set random seeds for reproducibility
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    # Note: For scikit-learn, pass random_state parameter to each model
    print(f"Random seed set to {seed}")


def validate_data_path(data_path: str) -> bool:
    """
    Validate that data path exists
    
    Args:
        data_path: Path to data file
        
    Returns:
        True if path exists, False otherwise
    """
    path = Path(data_path)
    
    if not path.exists():
        print(f"ERROR: Data file not found at {data_path}")
        return False
    
    print(f"✓ Data file found at {data_path}")
    return True


def create_latex_table(df: pd.DataFrame, caption: str, label: str,
                      save_path: str = None) -> str:
    """
    Create LaTeX table from DataFrame
    
    Args:
        df: DataFrame to convert
        caption: Table caption
        label: Table label for referencing
        save_path: Path to save LaTeX file (optional)
        
    Returns:
        LaTeX table string
    """
    latex_str = df.to_latex(
        index=False,
        float_format='%.4f',
        caption=caption,
        label=label,
        position='htbp'
    )
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(latex_str)
        print(f"LaTeX table saved to {save_path}")
    
    return latex_str


def summarize_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate dataset summary statistics
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        'n_samples': len(df),
        'n_features': len(df.columns) - 1,  # Exclude target
        'missing_values': df.isnull().sum().to_dict(),
        'target_distribution': df['target'].value_counts().to_dict() if 'target' in df.columns else None,
        'numeric_summary': df.describe().to_dict()
    }
    
    return summary


def print_dataset_info(df: pd.DataFrame, name: str = "Dataset"):
    """
    Print formatted dataset information
    
    Args:
        df: DataFrame to describe
        name: Name of dataset
    """
    print(f"\n{name} Information:")
    print(f"  Shape: {df.shape}")
    print(f"  Features: {df.shape[1]}")
    print(f"  Samples: {df.shape[0]}")
    
    if 'target' in df.columns:
        print(f"\n  Target Distribution:")
        print(f"    {df['target'].value_counts().to_dict()}")
    
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"\n  Missing Values:")
        for col, count in missing[missing > 0].items():
            print(f"    {col}: {count} ({count/len(df)*100:.1f}%)")
    else:
        print(f"\n  Missing Values: None")


class ExperimentLogger:
    """
    Logger for experiment tracking
    """
    
    def __init__(self, log_dir: str = "../results/tables"):
        """
        Initialize logger
        
        Args:
            log_dir: Directory for log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = get_timestamp()
        self.log_file = self.log_dir / f"experiment_log_{timestamp}.txt"
        
        # Initialize log
        with open(self.log_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("EXPERIMENT LOG\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
    
    def log(self, message: str, print_console: bool = True):
        """
        Log a message
        
        Args:
            message: Message to log
            print_console: Whether to also print to console
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {message}\n"
        
        with open(self.log_file, 'a') as f:
            f.write(log_entry)
        
        if print_console:
            print(message)
    
    def log_section(self, title: str):
        """
        Log a section header
        
        Args:
            title: Section title
        """
        self.log("\n" + "="*80)
        self.log(title)
        self.log("="*80 + "\n")
    
    def log_results(self, results_dict: Dict[str, Any]):
        """
        Log results dictionary
        
        Args:
            results_dict: Dictionary of results
        """
        for key, value in results_dict.items():
            if isinstance(value, float):
                self.log(f"{key}: {value:.4f}")
            else:
                self.log(f"{key}: {value}")


if __name__ == "__main__":
    print("Utility functions loaded successfully")
    
    # Example: Create project structure
    # create_project_structure()
