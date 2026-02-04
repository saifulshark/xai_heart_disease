"""
Main Experiment Script
End-to-end pipeline for heart disease classification with XAI
"""

import sys
import warnings
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np

from data_loader import HeartDiseaseDataLoader
from preprocessing import HeartDiseasePreprocessor
from models import HeartDiseaseModelTrainer
from evaluation import ModelEvaluator
from explainability.shap_explain import SHAPExplainer
from explainability.lime_explain import LIMEExplainer
from explainability.permutation_importance import PermutationImportanceExplainer
from explainability.pdp import PartialDependenceExplainer
from utils import (print_section_header, print_subsection_header,
                   save_experiment_config, ExperimentLogger)

warnings.filterwarnings('ignore')


def run_complete_experiment(data_path: str, results_dir: str = "../results",
                           random_state: int = 42):
    """
    Run complete experimental pipeline
    
    Args:
        data_path: Path to processed.cleveland.data
        results_dir: Directory to save results
        random_state: Random seed for reproducibility
    """
    # Initialize logger
    logger = ExperimentLogger(f"{results_dir}/tables")
    logger.log_section("HEART DISEASE CLASSIFICATION WITH EXPLAINABLE AI")
    
    # =========================================================================
    # 1. DATA LOADING
    # =========================================================================
    print_section_header("STEP 1: DATA LOADING")
    logger.log_section("DATA LOADING")
    
    loader = HeartDiseaseDataLoader(data_path)
    df = loader.load_data()
    
    logger.log(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
    
    # =========================================================================
    # 2. DATA PREPROCESSING
    # =========================================================================
    print_section_header("STEP 2: DATA PREPROCESSING")
    logger.log_section("DATA PREPROCESSING")
    
    preprocessor = HeartDiseasePreprocessor(random_state=random_state)
    data = preprocessor.preprocess_pipeline(df)
    
    logger.log(f"Training set: {data['X_train'].shape}")
    logger.log(f"Validation set: {data['X_val'].shape}")
    logger.log(f"Test set: {data['X_test'].shape}")
    
    # Save preprocessor
    preprocessor.save_preprocessor(f"{results_dir}/../models")
    
    # =========================================================================
    # 3. MODEL TRAINING
    # =========================================================================
    print_section_header("STEP 3: MODEL TRAINING")
    logger.log_section("MODEL TRAINING")
    
    trainer = HeartDiseaseModelTrainer(random_state=random_state)
    trainer.train_models(data['X_train'], data['y_train'])
    
    logger.log("All models trained successfully")
    
    # Save models
    trainer.save_models(f"{results_dir}/../models")
    
    # =========================================================================
    # 4. MODEL EVALUATION
    # =========================================================================
    print_section_header("STEP 4: MODEL EVALUATION")
    logger.log_section("MODEL EVALUATION")
    
    # Evaluate all models
    results_df = trainer.evaluate_all_models(
        data['X_val'], data['y_val'],
        data['X_test'], data['y_test']
    )
    
    logger.log("\nTest Set Results:")
    for _, row in results_df.iterrows():
        logger.log(f"\n{row['Model']}:")
        logger.log(f"  Accuracy: {row['Test_Accuracy']:.4f}")
        logger.log(f"  Precision: {row['Test_Precision']:.4f}")
        logger.log(f"  Recall: {row['Test_Recall']:.4f}")
        logger.log(f"  F1-Score: {row['Test_F1']:.4f}")
        logger.log(f"  ROC-AUC: {row['Test_ROC_AUC']:.4f}")
    
    # Get ROC curve data
    roc_data = trainer.get_roc_curves(data['X_test'], data['y_test'])
    
    # Generate evaluation outputs
    evaluator = ModelEvaluator(results_dir)
    evaluator.generate_all_outputs(results_df, roc_data)
    
    # =========================================================================
    # 5. EXPLAINABILITY - SHAP
    # =========================================================================
    print_section_header("STEP 5: EXPLAINABILITY ANALYSIS - SHAP")
    logger.log_section("SHAP ANALYSIS")
    
    feature_names = data['X_train'].columns.tolist()
    
    # SHAP for Random Forest (tree-based)
    print_subsection_header("SHAP: Random Forest")
    logger.log("Generating SHAP explanations for Random Forest...")
    
    shap_rf = SHAPExplainer(
        trainer.models['Random Forest'],
        data['X_train'],
        feature_names,
        results_dir
    )
    shap_rf.initialize_explainer(model_type='tree')
    shap_rf.generate_all_plots(data['X_test'], 'Random Forest', [0, 1, 2])
    
    # SHAP for Gradient Boosting (tree-based)
    print_subsection_header("SHAP: Gradient Boosting")
    logger.log("Generating SHAP explanations for Gradient Boosting...")
    
    shap_gb = SHAPExplainer(
        trainer.models['Gradient Boosting'],
        data['X_train'],
        feature_names,
        results_dir
    )
    shap_gb.initialize_explainer(model_type='tree')
    shap_gb.generate_all_plots(data['X_test'], 'Gradient Boosting', [0, 1, 2])
    
    # =========================================================================
    # 6. EXPLAINABILITY - LIME
    # =========================================================================
    print_section_header("STEP 6: EXPLAINABILITY ANALYSIS - LIME")
    logger.log_section("LIME ANALYSIS")
    
    # LIME for Random Forest
    print_subsection_header("LIME: Random Forest")
    logger.log("Generating LIME explanations for Random Forest...")
    
    lime_rf = LIMEExplainer(
        trainer.models['Random Forest'],
        data['X_train'],
        feature_names,
        results_dir=results_dir
    )
    lime_rf.generate_explanations(data['X_test'], 'Random Forest', [0, 1, 2, 3, 4])
    
    # =========================================================================
    # 7. EXPLAINABILITY - PERMUTATION IMPORTANCE
    # =========================================================================
    print_section_header("STEP 7: EXPLAINABILITY ANALYSIS - PERMUTATION IMPORTANCE")
    logger.log_section("PERMUTATION IMPORTANCE ANALYSIS")
    
    # Permutation importance for each model
    for model_name, model in trainer.models.items():
        print_subsection_header(f"Permutation Importance: {model_name}")
        logger.log(f"Calculating permutation importance for {model_name}...")
        
        perm_explainer = PermutationImportanceExplainer(
            model,
            feature_names,
            results_dir
        )
        perm_explainer.generate_all_outputs(
            data['X_test'],
            data['y_test'],
            model_name,
            n_repeats=10,
            top_n=13  # All features
        )
    
    # =========================================================================
    # 8. EXPLAINABILITY - PARTIAL DEPENDENCE PLOTS
    # =========================================================================
    print_section_header("STEP 8: EXPLAINABILITY ANALYSIS - PARTIAL DEPENDENCE PLOTS")
    logger.log_section("PARTIAL DEPENDENCE PLOT ANALYSIS")
    
    # PDPs for Random Forest
    print_subsection_header("PDPs: Random Forest")
    logger.log("Generating PDPs for Random Forest...")
    
    pdp_rf = PartialDependenceExplainer(
        trainer.models['Random Forest'],
        feature_names,
        results_dir
    )
    
    # Important 2D interactions
    interaction_pairs = [
        ('age', 'thalach'),
        ('chol', 'trestbps'),
        ('oldpeak', 'thalach')
    ]
    
    pdp_rf.generate_all_pdps(data['X_test'], 'Random Forest', interaction_pairs)
    
    # =========================================================================
    # 9. SAVE EXPERIMENT CONFIGURATION
    # =========================================================================
    print_section_header("STEP 9: SAVING EXPERIMENT CONFIGURATION")
    
    config = {
        'data_path': data_path,
        'random_state': random_state,
        'train_size': len(data['X_train']),
        'val_size': len(data['X_val']),
        'test_size': len(data['X_test']),
        'n_features': len(feature_names),
        'feature_names': feature_names,
        'models': list(trainer.models.keys()),
        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    save_experiment_config(config, results_dir)
    logger.log("Experiment configuration saved")
    
    # =========================================================================
    # 10. FINAL SUMMARY
    # =========================================================================
    print_section_header("EXPERIMENT COMPLETED SUCCESSFULLY!")
    logger.log_section("EXPERIMENT COMPLETED")
    
    print("\n✓ All models trained and evaluated")
    print("✓ Performance metrics saved to results/tables/")
    print("✓ Visualizations saved to results/figures/")
    print("✓ Models saved to models/")
    print("\nCheck the results directory for all outputs!")
    
    logger.log("All experiment outputs generated successfully")
    logger.log(f"Results saved to: {Path(results_dir).absolute()}")
    
    # Print best model
    best_model_idx = results_df['Test_ROC_AUC'].idxmax()
    best_model = results_df.loc[best_model_idx, 'Model']
    best_auc = results_df.loc[best_model_idx, 'Test_ROC_AUC']
    
    print(f"\n🏆 Best Model: {best_model} (ROC-AUC: {best_auc:.4f})")
    logger.log(f"\nBest performing model: {best_model} (ROC-AUC: {best_auc:.4f})")
    
    return results_df, trainer, data


if __name__ == "__main__":
    # Configuration
    DATA_PATH = "../data/processed.cleveland.data"
    RESULTS_DIR = "../results"
    RANDOM_STATE = 42
    
    # Check if custom data path provided
    if len(sys.argv) > 1:
        DATA_PATH = sys.argv[1]
    
    # Run experiment
    try:
        results, trainer, data = run_complete_experiment(
            DATA_PATH,
            RESULTS_DIR,
            RANDOM_STATE
        )
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease ensure:")
        print("1. The dataset file exists at the specified path")
        print("2. You're running from the correct directory")
        print(f"\nExpected data path: {Path(DATA_PATH).absolute()}")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()
