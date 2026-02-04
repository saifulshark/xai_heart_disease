# Complete Setup Guide for Windows

## 📋 Quick Start (5 minutes)

### Step 1: Create Project Folder

Open PowerShell and run these commands:

```powershell
# Navigate to where you want to create the project (change as needed)
cd C:\Users\mushfiq\Documents

# Create project structure
mkdir heart_disease_xai
cd heart_disease_xai
mkdir data
mkdir results
mkdir results\figures
mkdir results\tables

# Verify structure
tree /F
```

### Step 2: Copy Your Dataset

You already have the dataset! Just copy it:

```powershell
# Copy the Cleveland dataset to your project
copy "C:\Users\mushfiq\Downloads\heart+disease\processed.cleveland.data" "C:\Users\mushfiq\Documents\heart_disease_xai\data\"

# Verify it copied
dir data
```

### Step 3: Create the Python Script

1. Open Notepad or your favorite text editor
2. Copy the entire code from `standalone_pipeline.py` (provided separately)
3. Save as: `C:\Users\mushfiq\Documents\heart_disease_xai\standalone_pipeline.py`

**Important:** Make sure it's saved as `.py` not `.py.txt`

### Step 4: Run the Pipeline

```powershell
# Make sure you're in the project directory
cd C:\Users\mushfiq\Documents\heart_disease_xai

# Run the pipeline
python standalone_pipeline.py
```

**Expected runtime:** 30-60 seconds

---

## 📁 Final Folder Structure

After setup, your folder should look like this:

```
C:\Users\mushfiq\Documents\heart_disease_xai\
│
├── data\
│   └── processed.cleveland.data          ← Your dataset
│
├── results\
│   ├── figures\                          ← Generated images (after running)
│   │   ├── roc_curves.png
│   │   ├── confusion_matrices.png
│   │   ├── permutation_importance.png
│   │   ├── partial_dependence.png
│   │   ├── importance_comparison.png
│   │   └── decision_tree_viz.png
│   │
│   ├── tables\                           ← Generated CSV files (after running)
│   │   ├── model_comparison.csv
│   │   ├── permutation_importance.csv
│   │   └── importance_comparison.csv
│   │
│   └── SUMMARY_REPORT.txt                ← Text summary (after running)
│
└── standalone_pipeline.py                ← Main Python script
```

---

## ⚙️ Prerequisites

### Check if Python is installed:

```powershell
python --version
```

You should see something like `Python 3.8.x` or higher.

**If Python is not installed:**
1. Download from: https://www.python.org/downloads/
2. During installation, check "Add Python to PATH"
3. Restart PowerShell

### Check if required packages are installed:

```powershell
python -c "import numpy, pandas, sklearn, matplotlib, seaborn; print('All packages installed!')"
```

**If you get an error:**

```powershell
pip install numpy pandas scikit-learn matplotlib seaborn
```

---

## 🚀 Running the Pipeline

### Option 1: From PowerShell

```powershell
cd C:\Users\mushfiq\Documents\heart_disease_xai
python standalone_pipeline.py
```

### Option 2: Double-click (Windows Explorer)

1. Navigate to `C:\Users\mushfiq\Documents\heart_disease_xai\`
2. Right-click `standalone_pipeline.py`
3. Select "Open with" → "Python"

### Option 3: From any Python IDE

1. Open `standalone_pipeline.py` in your IDE (VS Code, PyCharm, Spyder, etc.)
2. Click "Run" or press F5

---

## 📊 What Happens When You Run It

You'll see output like this:

```
================================================================================
EXPLAINABLE AI PIPELINE FOR HEART DISEASE DIAGNOSIS
================================================================================

Script directory: C:\Users\mushfiq\Documents\heart_disease_xai
Looking for data at: C:\Users\mushfiq\Documents\heart_disease_xai\data\processed.cleveland.data
Results will be saved to: C:\Users\mushfiq\Documents\heart_disease_xai\results

SECTION 1: DATA LOADING
--------------------------------------------------------------------------------
✓ Dataset loaded from: C:\Users\mushfiq\Documents\heart_disease_xai\data\processed.cleveland.data
  Shape: (303, 14)
  Missing values: 6
  Class 0 (No disease): 164
  Class 1 (Disease): 139

SECTION 2: DATA PREPARATION
--------------------------------------------------------------------------------
✓ Missing values imputed using median strategy
✓ Data split: Train=212 (70.0%) | Val=45 (14.9%) | Test=46 (15.2%)
✓ Features scaled using StandardScaler

SECTION 3: MODEL TRAINING
--------------------------------------------------------------------------------
Training Logistic Regression...
✓ Logistic Regression trained
Training Decision Tree...
✓ Decision Tree trained
Training Random Forest...
✓ Random Forest trained
Training Gradient Boosting...
✓ Gradient Boosting trained

✓ All 4 models trained successfully

SECTION 4: MODEL EVALUATION
--------------------------------------------------------------------------------

Test Set Performance:
              Model  Accuracy  Precision  Recall  F1-Score  ROC-AUC
Logistic Regression     0.826      0.810   0.895     0.850    0.895
      Decision Tree     0.761      0.714   0.882     0.789    0.731
      Random Forest     0.891      0.944   0.895     0.919    0.939
  Gradient Boosting     0.870      0.850   0.944     0.895    0.942

✓ Results saved to: ...\results\tables\model_comparison.csv

✓ Best model (by Recall): Gradient Boosting

[... continues with visualizations and analysis ...]
```

---

## 📈 Viewing Results

### Generated Figures (PNG images, 300 DPI)

Open with any image viewer or web browser:

```
results\figures\roc_curves.png
results\figures\confusion_matrices.png
results\figures\permutation_importance.png
results\figures\partial_dependence.png
results\figures\importance_comparison.png
results\figures\decision_tree_viz.png
```

### Data Tables (CSV files)

Open with Excel, Google Sheets, or any text editor:

```
results\tables\model_comparison.csv
results\tables\permutation_importance.csv
results\tables\importance_comparison.csv
```

### Summary Report (Text file)

Open with Notepad or any text editor:

```
results\SUMMARY_REPORT.txt
```

---

## 🔧 Customization

### Change Data Path

Edit line 37 in `standalone_pipeline.py`:

```python
# Default (auto-detects)
DATA_PATH = os.path.join(SCRIPT_DIR, "data", "processed.cleveland.data")

# Custom path example
DATA_PATH = r"C:\MyData\heart_data.csv"
```

### Modify Model Parameters

Example - Change Random Forest settings (around line 175):

```python
# Default
rf = RandomForestClassifier(
    n_estimators=100,      # Number of trees
    random_state=RANDOM_SEED,
    max_depth=10,          # Maximum tree depth
    min_samples_split=20,  # Minimum samples to split
    n_jobs=-1              # Use all CPU cores
)

# More trees, deeper
rf = RandomForestClassifier(
    n_estimators=200,      # More trees
    random_state=RANDOM_SEED,
    max_depth=15,          # Deeper trees
    min_samples_split=10,
    n_jobs=-1
)
```

### Change Train/Test Split

Modify lines 136-141:

```python
# Default: 70/15/15 split
X_train, X_temp, y_train, y_temp = train_test_split(
    X_imputed, y, test_size=0.30, ...  # 30% for val+test
)

# Change to 80/10/10
X_train, X_temp, y_train, y_temp = train_test_split(
    X_imputed, y, test_size=0.20, ...  # 20% for val+test
)
```

---

## ❓ Troubleshooting

### Problem: "python is not recognized"

**Solution:**
- Ensure Python is installed
- Add Python to PATH:
  - Search "Environment Variables" in Windows
  - Edit "Path" variable
  - Add: `C:\Python3X\` (your Python install location)

### Problem: "No module named 'sklearn'"

**Solution:**
```powershell
pip install scikit-learn
```

### Problem: "Permission denied" or "Access denied"

**Solution:**
- Run PowerShell as Administrator
- Or choose a different location (e.g., Desktop)

### Problem: "FileNotFoundError: data\processed.cleveland.data"

**Solution:**
- Verify the data file is in the correct location:
  ```powershell
  dir data
  ```
- Make sure it's named exactly `processed.cleveland.data`
- Or the pipeline will auto-generate sample data

### Problem: Script runs but no figures appear

**Solution:**
- Figures are saved to disk, not displayed
- Check: `results\figures\` folder
- Open `.png` files with any image viewer

### Problem: Slow execution

**Solution:**
- Reduce `n_estimators` in models (lines 175, 182)
- Reduce `n_repeats` in permutation importance (line 523)
- Example: Change `n_estimators=100` to `n_estimators=50`

---

## 📚 Understanding the Output

### Model Performance Metrics

- **Accuracy**: Overall correctness (TP+TN)/(Total)
- **Precision**: Of predicted positives, how many are actually positive
- **Recall**: Of actual positives, how many were caught (most important for medical!)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve (higher = better)

### Clinical Focus: RECALL

For medical diagnosis, **RECALL (Sensitivity)** is most important because:
- High recall = fewer missed diagnoses
- We want to catch all potential disease cases
- False positives (extra testing) better than false negatives (missed disease)

### Feature Importance

The pipeline identifies which patient measurements are most predictive:
- **oldpeak**: ST depression during exercise (often most important)
- **ca**: Number of major vessels (fluoroscopy)
- **cp**: Chest pain type
- **thalach**: Maximum heart rate
- **thal**: Thalassemia status

---

## 💡 Pro Tips

### Tip 1: Run Multiple Times

The random seed is fixed, so you'll get the same results each run. To see variation:

```python
# Line 32 - change the seed
RANDOM_SEED = 42  # Change to 123, 456, etc.
```

### Tip 2: Save Script Output

```powershell
python standalone_pipeline.py > output.txt 2>&1
```

This saves all console output to `output.txt`

### Tip 3: Use Jupyter Notebook

For interactive exploration:

```powershell
pip install jupyter
jupyter notebook
```

Then create a notebook and run the pipeline cell-by-cell

### Tip 4: Compare Multiple Datasets

Run the pipeline on different UCI datasets:
- Hungarian data: `processed.hungarian.data`
- Switzerland data: `processed.switzerland.data`

Just change the `DATA_PATH` variable

---

## 🎓 Next Steps

1. **Review the results** in `results\SUMMARY_REPORT.txt`
2. **Examine the figures** to understand model performance
3. **Check feature importance** to see what predicts disease
4. **Experiment** with different model parameters
5. **Try other datasets** from the UCI repository

---

## 📞 Need Help?

Common resources:
- **Python documentation**: https://docs.python.org/3/
- **scikit-learn**: https://scikit-learn.org/stable/
- **UCI Dataset info**: https://archive.ics.uci.edu/ml/datasets/heart+Disease

---

## ✅ Quick Verification

After running, you should have:

```powershell
# Check files exist
dir results\figures
dir results\tables
dir results\SUMMARY_REPORT.txt
```

Expected output:
- ✓ 6 PNG files in `figures\`
- ✓ 3 CSV files in `tables\`
- ✓ 1 TXT file in `results\`

If all present: **Success!** 🎉

---

## 📋 Summary Checklist

- [ ] Python installed (3.8+)
- [ ] Packages installed (numpy, pandas, sklearn, matplotlib, seaborn)
- [ ] Project folder created
- [ ] Dataset copied to `data\` folder
- [ ] `standalone_pipeline.py` created
- [ ] Pipeline executed successfully
- [ ] Results generated in `results\` folder

**All checked? You're done!** 🚀
