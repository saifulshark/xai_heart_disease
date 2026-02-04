# Heart Disease Classification with Explainable AI

A complete, publication-quality research codebase for explainable medical diagnosis support using the UCI Heart Disease dataset.

## 🎯 Project Overview

This project implements an end-to-end machine learning pipeline for heart disease classification with comprehensive explainability analysis. It is designed for Q1 journal publication (e.g., IEEE JBHI, Artificial Intelligence in Medicine, Expert Systems with Applications).

### Key Features

- **Multiple ML Models**: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Explainability Methods**:
  - SHAP (SHapley Additive exPlanations)
  - LIME (Local Interpretable Model-agnostic Explanations)
  - Permutation Feature Importance
  - Partial Dependence Plots (PDPs)
- **Publication-Ready Outputs**: High-quality figures, tables, and LaTeX exports
- **Reproducible Pipeline**: Automated execution scripts with fixed random seeds

## 📁 Project Structure

```
xai_heart_disease/
│
├── data/
│   └── processed.cleveland.data          # UCI Heart Disease dataset
│
├── src/
│   ├── data_loader.py                    # Dataset loading and column naming
│   ├── preprocessing.py                  # Data cleaning and preparation
│   ├── models.py                         # ML model training and management
│   ├── evaluation.py                     # Performance evaluation and visualization
│   ├── utils.py                          # Utility functions
│   ├── main_experiment.py                # Main experimental pipeline
│   │
│   └── explainability/
│       ├── shap_explain.py               # SHAP explanations
│       ├── lime_explain.py               # LIME explanations
│       ├── permutation_importance.py     # Permutation-based importance
│       └── pdp.py                        # Partial dependence plots
│
├── experiments/
│   ├── run_experiments.ps1               # PowerShell script (Windows)
│   └── run_experiments.sh                # Bash script (Linux/macOS)
│
├── results/
│   ├── tables/                           # CSV and LaTeX tables
│   └── figures/                          # Publication-quality figures
│
├── models/                               # Saved trained models
│
├── paper_assets/
│   ├── methodology_notes.md              # Methodology section template
│   ├── clinical_interpretation.md        # Clinical interpretation guide
│   └── references.bib                    # BibTeX references
│
├── requirements.txt                      # Python dependencies
└── README.md                             # This file
```

## 🚀 Quick Start

### Prerequisites

- **Python**: 3.8 or higher
- **Operating System**: Windows, Linux, or macOS
- **Dataset**: UCI Heart Disease dataset (processed.cleveland.data)

### Installation and Execution

#### Option 1: Windows (PowerShell)

1. **Update the data path** in `experiments/run_experiments.ps1`:
   ```powershell
   $DATA_PATH = "C:\Users\mushfiq\Downloads\heart+disease\processed.cleveland.data"
   ```

2. **Run the script**:
   ```powershell
   cd experiments
   .\run_experiments.ps1
   ```

#### Option 2: Linux/macOS (Bash)

1. **Make the script executable**:
   ```bash
   chmod +x experiments/run_experiments.sh
   ```

2. **Update the data path** (if needed) in `experiments/run_experiments.sh`:
   ```bash
   DATA_PATH="../data/processed.cleveland.data"
   ```

3. **Run the script**:
   ```bash
   cd experiments
   ./run_experiments.sh
   ```

#### Option 3: Manual Execution

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Copy the dataset**:
   ```bash
   # Copy processed.cleveland.data to data/ directory
   cp /path/to/processed.cleveland.data data/
   ```

4. **Run the experiment**:
   ```bash
   cd src
   python main_experiment.py ../data/processed.cleveland.data
   ```

## 📊 Pipeline Overview

The experimental pipeline consists of the following steps:

### 1. Data Loading
- Loads `processed.cleveland.data` from the specified path
- Assigns proper column names according to UCI documentation
- Handles missing values (represented as '?')

### 2. Data Preprocessing
- Converts multi-class target (0-4) to binary (0 = no disease, 1-4 = disease)
- Imputes missing values using median strategy
- Standardizes features using StandardScaler
- Performs stratified split: 70% train, 15% validation, 15% test

### 3. Model Training
Trains four classifiers:
- **Logistic Regression**: Interpretable baseline
- **Decision Tree**: Simple tree-based model
- **Random Forest**: Ensemble method
- **Gradient Boosting**: Advanced ensemble

### 4. Model Evaluation
Evaluates all models on validation and test sets using:
- Accuracy
- Precision
- Recall (sensitivity)
- F1-Score
- ROC-AUC

Generates:
- Performance comparison table (CSV and LaTeX)
- ROC curves (publication-quality PNG)
- Metric comparison visualizations

### 5. Explainability Analysis

#### SHAP
- Global feature importance (summary plots)
- Local explanations for individual predictions (waterfall plots)
- Feature contribution visualization

#### LIME
- Local explanations for selected test samples
- Feature contribution breakdown
- Comparison across multiple samples

#### Permutation Importance
- Model-agnostic feature importance
- Statistical significance with error bars
- Cross-model comparison

#### Partial Dependence Plots
- Relationship between features and predictions
- Clinical feature analysis (age, blood pressure, cholesterol, etc.)
- 2D interaction plots for feature pairs

### 6. Results Export
All outputs are saved to `results/`:
- **Tables** (CSV + LaTeX): `results/tables/`
- **Figures** (PNG, 300 DPI): `results/figures/`
- **Models** (pickle): `models/`

## 📈 Expected Outputs

After running the pipeline, you will find:

### Tables
- `model_performance.csv`: Complete evaluation metrics
- `model_performance.tex`: LaTeX table for paper
- `permutation_importance_*.csv`: Feature importance rankings
- `experiment_config.json`: Experiment configuration
- `summary_report.txt`: Text summary of results

### Figures

#### Model Evaluation
- `roc_curves.png`: ROC curves for all models
- `metric_comparison.png`: Bar chart comparing metrics

#### SHAP Explanations
- `shap_summary_*.png`: Global feature importance
- `shap_bar_*.png`: Mean absolute SHAP values
- `shap_waterfall_*.png`: Individual prediction explanations

#### LIME Explanations
- `lime_*.png`: Local explanations for test samples
- `lime_comparison_*.png`: Multi-sample comparison

#### Permutation Importance
- `permutation_importance_*.png`: Feature importance with error bars

#### Partial Dependence Plots
- `pdp_clinical_*.png`: Clinical features (age, BP, cholesterol)
- `pdp_categorical_*.png`: Categorical features
- `pdp_2d_*.png`: 2D interaction plots

## 🔬 For Researchers

### Adapting for Your Research

#### Using Different Datasets
1. Modify `data_loader.py` to load your dataset
2. Update `column_names` to match your features
3. Adjust preprocessing in `preprocessing.py` as needed

#### Adding New Models
1. Add model to `initialize_models()` in `models.py`
2. The pipeline will automatically train and evaluate it

#### Customizing Explainability
- **SHAP**: Modify `generate_all_plots()` in `shap_explain.py`
- **LIME**: Adjust `num_features` and `sample_indices`
- **PDPs**: Add features to `clinical_features` or `categorical_features`

### Clinical Interpretation

The results should be interpreted with clinical context:

- **High SHAP/Importance Features**: Strong predictors (e.g., chest pain type, max heart rate)
- **Positive SHAP Values**: Increase disease probability
- **Negative SHAP Values**: Decrease disease probability
- **PDPs**: Show how risk changes with feature values (e.g., age vs. risk)

See `paper_assets/clinical_interpretation.md` for detailed guidance.

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{heart_disease_xai,
  title = {Heart Disease Classification with Explainable AI},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/xai_heart_disease}
}
```

## 📚 Key References

- **SHAP**: Lundberg & Lee (2017) - "A Unified Approach to Interpreting Model Predictions"
- **LIME**: Ribeiro et al. (2016) - "Why Should I Trust You?"
- **UCI Dataset**: Detrano et al. (1989) - "International application of a new probability algorithm..."

Full BibTeX references available in `paper_assets/references.bib`

## 🤝 Contributing

This is a research codebase designed for reproducibility and publication. If you find issues or have suggestions:

1. Document the issue clearly
2. Provide minimal reproducible examples
3. Suggest improvements with justification

## 📄 License

This project is intended for academic research. Please cite appropriately if used in publications.

## ⚠️ Important Notes

### Reproducibility
- All random seeds are fixed (default: 42)
- Results should be identical across runs
- Virtual environment ensures dependency consistency

### Dataset Path
- **Windows**: Update `DATA_PATH` in `run_experiments.ps1`
- **Linux/macOS**: Update `DATA_PATH` in `run_experiments.sh`
- **Manual**: Pass path as argument to `main_experiment.py`

### Computational Requirements
- **CPU-only** (no GPU required)
- **RAM**: 4GB minimum, 8GB recommended
- **Runtime**: ~5-15 minutes depending on hardware

### Troubleshooting

**Issue**: `FileNotFoundError: Dataset not found`
- **Solution**: Verify the data path in the execution script

**Issue**: Package installation errors
- **Solution**: Use Python 3.8-3.10 (3.11+ may have compatibility issues)

**Issue**: SHAP takes too long
- **Solution**: Reduce number of background samples in `shap_explain.py`

**Issue**: Out of memory
- **Solution**: Reduce `n_repeats` in permutation importance or use subset of data

## 📞 Support

For questions or issues:
1. Check this README
2. Review error messages carefully
3. Verify all paths and dependencies
4. Consult the paper assets for methodology details

## 🎓 Academic Use

This codebase is designed for:
- Q1 journal submissions
- Master's/PhD research
- Reproducible ML research
- Explainable AI education

All code is commented and modular for easy understanding and modification.

---

**Happy Researching! 🚀**
