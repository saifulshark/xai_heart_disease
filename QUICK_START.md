# Quick Start Guide

## 🚀 Running the Complete Pipeline in 5 Minutes

This guide will get you from zero to results in under 5 minutes.

## Prerequisites Checklist

- [ ] Python 3.8+ installed (`python --version`)
- [ ] Dataset file at: `C:\Users\mushfiq\Downloads\heart+disease\processed.cleveland.data`
- [ ] Terminal/PowerShell access
- [ ] ~500MB free disk space for dependencies and results

## Step-by-Step Instructions

### Windows Users (PowerShell)

```powershell
# 1. Navigate to the experiments directory
cd path\to\xai_heart_disease\experiments

# 2. Run the automated script
.\run_experiments.ps1
```

**That's it!** The script will:
- ✅ Check Python installation
- ✅ Create virtual environment
- ✅ Install all dependencies
- ✅ Verify dataset location
- ✅ Run complete pipeline
- ✅ Generate all results

### Linux/macOS Users (Bash)

```bash
# 1. Navigate to the experiments directory
cd path/to/xai_heart_disease/experiments

# 2. Make script executable
chmod +x run_experiments.sh

# 3. Run the automated script
./run_experiments.sh
```

## What Happens During Execution?

### Phase 1: Setup (1-2 minutes)
```
[1/6] Checking Python installation...        ✓
[2/6] Setting up virtual environment...      ✓
[3/6] Installing dependencies...             ✓ (may take 1-2 min)
[4/6] Verifying data file...                 ✓
```

### Phase 2: Experiments (3-10 minutes)
```
[5/6] Running experiment pipeline...

STEP 1: DATA LOADING                         ✓
  - Loaded 303 samples with 13 features

STEP 2: DATA PREPROCESSING                   ✓
  - Training: 212 samples
  - Validation: 45 samples
  - Test: 46 samples

STEP 3: MODEL TRAINING                       ✓
  - Logistic Regression                      ✓
  - Decision Tree                            ✓
  - Random Forest                            ✓
  - Gradient Boosting                        ✓

STEP 4: MODEL EVALUATION                     ✓
  - Performance metrics calculated
  - ROC curves generated

STEP 5: SHAP ANALYSIS                        ✓
  - Random Forest explanations               ✓
  - Gradient Boosting explanations           ✓

STEP 6: LIME ANALYSIS                        ✓
  - Local explanations for 5 samples         ✓

STEP 7: PERMUTATION IMPORTANCE               ✓
  - All models analyzed                      ✓

STEP 8: PARTIAL DEPENDENCE PLOTS             ✓
  - Clinical features                        ✓
  - Categorical features                     ✓
  - 2D interactions                          ✓

STEP 9: SAVING CONFIGURATION                 ✓

✓ EXPERIMENT COMPLETED SUCCESSFULLY!
```

### Phase 3: Summary
```
[6/6] Experiment Summary

Results Location: C:\...\xai_heart_disease\results
  - Figures generated: ~40
  - Tables generated: ~15

Key Outputs:
  📊 results\tables\model_performance.csv
  📈 results\figures\roc_curves.png
  🔍 results\figures\shap_*.png
  🔍 results\figures\lime_*.png
  📉 results\figures\pdp_*.png

🏆 Best Model: Random Forest (ROC-AUC: 0.XXXX)
```

## Expected Runtime

| Hardware          | Time    |
|-------------------|---------|
| Modern CPU (4+ cores) | 3-5 min  |
| Standard CPU (2 cores) | 5-10 min |
| Older CPU         | 10-15 min |

*Note: First run takes longer due to dependency installation*

## Verifying Results

### Check 1: Results Directory
```bash
ls results/
# Should see: tables/ and figures/
```

### Check 2: Number of Outputs
```bash
# Windows
(Get-ChildItem results\figures\*.png).Count   # Should be ~35-45
(Get-ChildItem results\tables\*.*).Count      # Should be ~10-15

# Linux/macOS
ls results/figures/*.png | wc -l              # Should be ~35-45
ls results/tables/* | wc -l                   # Should be ~10-15
```

### Check 3: Performance Table
```bash
# View the main results
cat results/tables/model_performance.csv
```

Expected columns:
- Model
- Val_Accuracy, Val_Precision, Val_Recall, Val_F1, Val_ROC_AUC
- Test_Accuracy, Test_Precision, Test_Recall, Test_F1, Test_ROC_AUC

### Check 4: Best Model
```bash
# Should be at the end of the log
tail results/tables/experiment_log_*.txt
```

## Manual Verification (Optional)

If you want to verify the pipeline manually:

```python
cd src
python

>>> from data_loader import HeartDiseaseDataLoader
>>> loader = HeartDiseaseDataLoader("../data/processed.cleveland.data")
>>> df = loader.load_data()
>>> print(df.shape)  # Should be (303, 14)
>>> exit()
```

## Troubleshooting

### Problem: "Python not found"

**Windows**:
```powershell
# Install Python from python.org
# Or via Windows Store
winget install Python.Python.3.11
```

**Linux**:
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

**macOS**:
```bash
brew install python@3.11
```

### Problem: "Dataset not found"

**Solution**:
```powershell
# Verify file exists
Test-Path "C:\Users\mushfiq\Downloads\heart+disease\processed.cleveland.data"

# If False, update path in experiments/run_experiments.ps1
# Line ~15: $DATA_PATH = "YOUR\PATH\HERE"
```

### Problem: "Permission denied" (run_experiments.sh)

**Solution**:
```bash
chmod +x experiments/run_experiments.sh
```

### Problem: Package installation fails

**Solution 1**: Upgrade pip
```bash
python -m pip install --upgrade pip
```

**Solution 2**: Use older Python (3.8-3.10)
```bash
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Solution 3**: Install packages one by one
```bash
pip install numpy pandas scikit-learn
pip install shap lime matplotlib seaborn
```

### Problem: "Out of memory"

**Solution**: Reduce dataset size or increase swap space
```python
# In src/main_experiment.py, modify:
# Line ~85: Change n_repeats from 10 to 5
# Line ~120: Use fewer background samples for SHAP
```

### Problem: Scripts run but no figures generated

**Check**:
1. matplotlib backend: `python -c "import matplotlib; print(matplotlib.get_backend())"`
2. Write permissions: `ls -la results/`
3. Disk space: `df -h`

**Solution**:
```python
# In src code, add at top:
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
```

## What's Next?

After successful execution:

### 1. Review Results
```bash
# Open performance table
start results\tables\model_performance.csv        # Windows
open results/tables/model_performance.csv         # macOS
xdg-open results/tables/model_performance.csv     # Linux
```

### 2. View Figures
```bash
# Open figures directory
start results\figures\                            # Windows
open results/figures/                             # macOS
nautilus results/figures/                         # Linux
```

### 3. Read Summary
```bash
cat results/tables/summary_report.txt
```

### 4. Start Writing Paper
- See `paper_assets/methodology_notes.md`
- See `paper_assets/clinical_interpretation.md`
- Use `paper_assets/references.bib`

## Common Next Steps

### Customize Experiments

**Change models**:
Edit `src/models.py`, function `initialize_models()`

**Adjust hyperparameters**:
Edit `src/models.py`, modify model configurations

**Change explainability samples**:
Edit `src/main_experiment.py`, modify `sample_indices`

**Add more XAI methods**:
Create new file in `src/explainability/`

### Re-run After Changes

```bash
# Full re-run
cd experiments
.\run_experiments.ps1  # Windows
./run_experiments.sh   # Linux/macOS

# Or manually
cd src
python main_experiment.py ../data/processed.cleveland.data
```

### Export for Paper

**LaTeX tables**:
```
results/tables/model_performance.tex
```

**High-res figures** (300 DPI):
```
results/figures/*.png
```

**BibTeX references**:
```
paper_assets/references.bib
```

## Performance Tips

### Speed up execution:
1. Reduce `n_repeats` in permutation importance (line ~85 in main_experiment.py)
2. Use fewer SHAP background samples
3. Skip some explainability methods temporarily
4. Use fewer models (comment out in models.py)

### Increase quality:
1. Increase `n_repeats` for permutation importance
2. Increase `num_samples` in LIME (line in lime_explain.py)
3. Higher `grid_resolution` for PDPs
4. Generate more individual explanations

## Getting Help

1. **Check logs**: `results/tables/experiment_log_*.txt`
2. **Error messages**: Read carefully, usually self-explanatory
3. **README.md**: Full documentation
4. **Source code**: All files are heavily commented

## Success Indicators

You know everything worked if you see:

✅ All 6 steps completed without errors
✅ ~35-45 PNG files in `results/figures/`
✅ ~10-15 files in `results/tables/`
✅ Best model reported with ROC-AUC score
✅ No error messages in terminal

## Ready for Publication

After successful run, you have:

- ✅ Trained and evaluated models
- ✅ Performance metrics (tables + LaTeX)
- ✅ Publication-quality figures (300 DPI)
- ✅ Explainability analysis (SHAP, LIME, PDP)
- ✅ Reproducible pipeline (fixed seeds)
- ✅ Complete documentation
- ✅ BibTeX references

**You're ready to write your Q1 journal paper!** 🎉

---

**Estimated time from start to finish: 5-10 minutes**

Good luck with your research! 🚀
