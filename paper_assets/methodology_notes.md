# Methodology Section Template for Q1 Journal Paper

This document provides guidance for writing the methodology section based on the implemented pipeline.

## 1. Dataset Description

### Subsection: Data Source

The Cleveland Heart Disease dataset, obtained from the UCI Machine Learning Repository [1], was used in this study. The dataset consists of 303 patient records collected at the Cleveland Clinic Foundation, with 13 clinical features and a diagnosis indicator (angiographic disease status).

**Features**:
- **Demographic**: Age, Sex
- **Clinical Symptoms**: Chest pain type (cp), Exercise-induced angina (exang)
- **Vital Signs**: Resting blood pressure (trestbps), Maximum heart rate (thalach)
- **Laboratory Tests**: Serum cholesterol (chol), Fasting blood sugar (fbs)
- **Diagnostic Tests**: Resting ECG (restecg), ST depression (oldpeak), Slope of peak exercise ST segment (slope), Number of major vessels colored by fluoroscopy (ca), Thalassemia status (thal)

**Target Variable**: Binary classification
- Class 0: No presence of heart disease
- Class 1-4: Presence of heart disease (converted to single class)

### Subsection: Data Preprocessing

**Missing Value Imputation**:
Missing values (1.63% of total data) were imputed using median imputation. The median was calculated from the training set and applied to all splits to prevent data leakage.

**Feature Scaling**:
All features were standardized using StandardScaler with zero mean and unit variance. Scaling parameters were fit on the training set only.

**Target Transformation**:
The original multi-class diagnosis variable (0-4) was converted to binary classification:
- 0 → No disease (negative class)
- 1-4 → Disease present (positive class)

This transformation yielded a class distribution of [report from results: X% negative, Y% positive].

**Data Splitting**:
Stratified sampling was employed to maintain class proportions across splits:
- Training set: 70% (n=XXX)
- Validation set: 15% (n=XXX)
- Test set: 15% (n=XXX)

All experiments used a fixed random seed (42) for reproducibility.

## 2. Machine Learning Models

### Subsection: Model Selection Rationale

Four classification algorithms were selected to balance interpretability and performance:

1. **Logistic Regression (LR)**: A linear model providing baseline interpretability with inherent feature weights.

2. **Decision Tree (DT)**: A rule-based model offering hierarchical decision pathways.

3. **Random Forest (RF)**: An ensemble of decision trees providing robust predictions through bootstrap aggregation.

4. **Gradient Boosting (GB)**: A sequential ensemble method optimizing residual errors for improved performance.

### Subsection: Model Configuration

**Logistic Regression**:
- Solver: L-BFGS
- Maximum iterations: 1000
- Regularization: L2 (default)

**Decision Tree**:
- Maximum depth: 5
- Minimum samples split: 10
- Minimum samples leaf: 5
- Criterion: Gini impurity

**Random Forest**:
- Number of estimators: 100
- Maximum depth: 10
- Minimum samples split: 10
- Minimum samples leaf: 5
- Bootstrap: True

**Gradient Boosting**:
- Number of estimators: 100
- Learning rate: 0.1
- Maximum depth: 5
- Subsample ratio: 0.8

Hyperparameters were chosen based on cross-validation performance on the validation set.

## 3. Evaluation Metrics

### Subsection: Performance Metrics

Models were evaluated using five complementary metrics:

**Accuracy**: Overall correctness of predictions
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

**Precision**: Proportion of positive predictions that were correct
$$\text{Precision} = \frac{TP}{TP + FP}$$

**Recall (Sensitivity)**: Proportion of actual positives correctly identified
$$\text{Recall} = \frac{TP}{TP + FN}$$

**F1-Score**: Harmonic mean of precision and recall
$$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

**ROC-AUC**: Area under the receiver operating characteristic curve, measuring discrimination ability across all classification thresholds.

### Subsection: Clinical Relevance

In medical diagnosis, **recall (sensitivity)** is prioritized to minimize false negatives (missed diagnoses), as the cost of failing to identify disease is typically higher than false positives. However, **ROC-AUC** provides a comprehensive measure of model discrimination across all operating points.

## 4. Explainability Methods

### Subsection 4.1: SHAP (SHapley Additive exPlanations)

**Theoretical Foundation**:
SHAP values are based on cooperative game theory, assigning each feature a contribution value for a specific prediction [2]. For a prediction $f(x)$, the SHAP value $\phi_i$ for feature $i$ satisfies:

$$f(x) = \phi_0 + \sum_{i=1}^{M} \phi_i$$

where $\phi_0$ is the expected model output and $M$ is the number of features.

**Implementation**:
- Tree-based models: TreeExplainer for exact SHAP computation
- Linear models: LinearExplainer for efficient calculation
- Background dataset: Training set (n=XXX samples)

**Outputs**:
- Global importance: Summary plots aggregating absolute SHAP values
- Local explanations: Waterfall plots showing feature contributions for individual predictions

### Subsection 4.2: LIME (Local Interpretable Model-agnostic Explanations)

**Theoretical Foundation**:
LIME approximates complex model predictions locally using interpretable linear models [3]. For a prediction $f(x)$, LIME finds an explanation $g \in G$ by:

$$\text{explanation}(x) = \arg\min_{g \in G} L(f, g, \pi_x) + \Omega(g)$$

where $L$ is the locality-aware loss, $\pi_x$ defines the local neighborhood, and $\Omega(g)$ measures model complexity.

**Implementation**:
- Perturbation samples: 5,000 per instance
- Interpretable representation: Linear model with top-k features
- Distance metric: Exponential kernel
- Number of features: 10 for detailed explanations

**Outputs**:
- Feature contribution plots for individual predictions
- Comparative analyses across multiple patients

### Subsection 4.3: Permutation Feature Importance

**Theoretical Foundation**:
Permutation importance measures feature relevance by quantifying performance degradation when feature values are randomly shuffled [4]. For feature $j$:

$$\text{Importance}_j = s - \frac{1}{K}\sum_{k=1}^{K} s_{k,j}^{\text{perm}}$$

where $s$ is the original model score and $s_{k,j}^{\text{perm}}$ is the score after permutation $k$ of feature $j$.

**Implementation**:
- Number of permutations: 10 per feature
- Scoring metric: ROC-AUC
- Statistical summary: Mean and standard deviation across permutations

**Outputs**:
- Feature importance rankings with confidence intervals
- Comparison across all models

### Subsection 4.4: Partial Dependence Plots (PDPs)

**Theoretical Foundation**:
PDPs visualize the marginal effect of features on predictions by averaging over all other features [5]. For feature $x_S$:

$$\text{PD}_{x_S}(x_S) = \mathbb{E}_{x_C}[f(x_S, x_C)]$$

where $x_C$ represents all other features.

**Implementation**:
- Grid resolution: 50 points per feature
- Marginalization: Average over all training instances
- 2D PDPs: 20×20 grid for feature interactions

**Outputs**:
- 1D plots for clinical continuous features (age, blood pressure, etc.)
- 1D plots for categorical features (chest pain type, ECG results, etc.)
- 2D interaction plots for critical feature pairs

## 5. Experimental Protocol

### Subsection: Reproducibility

All experiments were conducted with:
- Fixed random seed: 42
- Identical train/validation/test splits across all models
- Deterministic algorithms where possible
- Version-controlled codebase

### Subsection: Computational Environment

- Language: Python 3.x
- Key libraries: scikit-learn 1.3.0, SHAP 0.42.1, LIME 0.2.0.1
- Hardware: CPU-only (no GPU required)
- Approximate runtime: [fill in based on your system]

## 6. Statistical Analysis

### Subsection: Model Comparison

Models were compared on the independent test set. Performance differences were assessed using:
- Paired t-tests for metric comparisons (if multiple runs conducted)
- McNemar's test for classifier agreement
- Confidence intervals via bootstrapping (if applicable)

### Subsection: Feature Importance Consensus

Feature importance was analyzed across multiple methods (SHAP, permutation importance) to identify robust predictors. Features consistently ranked in the top-5 across methods were designated as clinically significant predictors.

## References for Methodology

[1] Detrano, R., et al. (1989). International application of a new probability algorithm for the diagnosis of coronary artery disease. *The American Journal of Cardiology*, 64(5), 304-310.

[2] Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.

[3] Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?" Explaining the predictions of any classifier. *Proceedings of the 22nd ACM SIGKDD*, 1135-1144.

[4] Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32.

[5] Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. *Annals of Statistics*, 1189-1232.

---

## Tips for Writing

1. **Be specific**: Include exact parameter values and sample sizes
2. **Justify choices**: Explain why methods were selected
3. **Report limitations**: Acknowledge constraints (e.g., dataset size, class imbalance)
4. **Use past tense**: "Models were trained...", "SHAP values were calculated..."
5. **Be concise**: Avoid redundant explanations
6. **Cross-reference**: Link to supplementary materials for additional details

## Common Reviewer Questions to Address

- Why these specific models?
- How were hyperparameters chosen?
- Why binary classification instead of multi-class?
- How was overfitting prevented?
- Why SHAP and LIME? Why not just one?
- How do you validate explainability results?

Ensure your methodology section preemptively answers these questions.
