"""
Explainability Module for Heart Disease Classification

This module provides multiple explainability methods for interpreting
machine learning models in the context of medical diagnosis.

Available methods:
- SHAP (SHapley Additive exPlanations)
- LIME (Local Interpretable Model-agnostic Explanations)  
- Permutation Feature Importance
- Partial Dependence Plots (PDPs)
"""

from .shap_explain import SHAPExplainer
from .lime_explain import LIMEExplainer
from .permutation_importance import PermutationImportanceExplainer
from .pdp import PartialDependenceExplainer

__all__ = [
    'SHAPExplainer',
    'LIMEExplainer',
    'PermutationImportanceExplainer',
    'PartialDependenceExplainer'
]

__version__ = '1.0.0'
__author__ = 'Heart Disease XAI Research Team'
