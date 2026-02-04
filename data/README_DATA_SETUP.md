# Data File Setup Guide

## Dataset Information

This project uses the **UCI Heart Disease Dataset**, specifically the **processed Cleveland database**.

### Dataset Source

- **Repository**: UCI Machine Learning Repository
- **Dataset Name**: Heart Disease Data Set
- **Direct Link**: https://archive.ics.uci.edu/ml/datasets/heart+disease
- **Citation**: Detrano et al. (1989)

### Required File

**Filename**: `processed.cleveland.data`

**Location on Your System**: 
```
C:\Users\mushfiq\Downloads\heart+disease\processed.cleveland.data
```

## Setup Instructions

### Option 1: Use Your Current Location (Windows)

If your data is already at:
```
C:\Users\mushfiq\Downloads\heart+disease\processed.cleveland.data
```

**No action needed!** The scripts are pre-configured for this path.

### Option 2: Copy to Project Directory

1. **Create the data directory** (if running manually):
   ```bash
   mkdir data
   ```

2. **Copy the file**:
   ```bash
   # Windows (PowerShell)
   Copy-Item "C:\Users\mushfiq\Downloads\heart+disease\processed.cleveland.data" -Destination "data\"

   # Linux/macOS
   cp ~/Downloads/heart+disease/processed.cleveland.data data/
   ```

3. **Update the script path**:
   
   In `experiments/run_experiments.ps1`, change:
   ```powershell
   $DATA_PATH = "C:\Users\mushfiq\Downloads\heart+disease\processed.cleveland.data"
   ```
   to:
   ```powershell
   $DATA_PATH = "$PROJECT_ROOT\data\processed.cleveland.data"
   ```

### Option 3: Custom Location

If your data is in a different location:

1. **Edit the execution script**:
   
   **Windows** (`experiments/run_experiments.ps1`):
   ```powershell
   $DATA_PATH = "YOUR\CUSTOM\PATH\processed.cleveland.data"
   ```
   
   **Linux/macOS** (`experiments/run_experiments.sh`):
   ```bash
   DATA_PATH="/your/custom/path/processed.cleveland.data"
   ```

2. **Or pass as command-line argument**:
   ```bash
   cd src
   python main_experiment.py /path/to/processed.cleveland.data
   ```

## Dataset Format

The `processed.cleveland.data` file has the following characteristics:

### Structure
- **Format**: CSV (comma-separated values)
- **Header**: None (column names assigned in code)
- **Rows**: 303 patient records
- **Columns**: 14 (13 features + 1 target)

### Columns (in order)
1. age
2. sex
3. cp (chest pain type)
4. trestbps (resting blood pressure)
5. chol (serum cholesterol)
6. fbs (fasting blood sugar)
7. restecg (resting ECG results)
8. thalach (maximum heart rate)
9. exang (exercise induced angina)
10. oldpeak (ST depression)
11. slope (slope of peak exercise ST segment)
12. ca (number of major vessels)
13. thal (thalassemia)
14. target (diagnosis: 0-4)

### Missing Values
- Represented as: `?`
- Handled by: Median imputation in preprocessing

### Example Rows
```
63,1,1,145,233,1,2,150,0,2.3,3,0,6,0
67,1,4,160,286,0,2,108,1,1.5,2,3,3,2
67,1,4,120,229,0,2,129,1,2.6,2,2,7,1
```

## Verification

To verify your dataset is correctly formatted:

```python
import pandas as pd

# Load the file
df = pd.read_csv('path/to/processed.cleveland.data', header=None)

print(f"Shape: {df.shape}")  # Should be (303, 14)
print(f"Missing values: {df.isin(['?']).sum().sum()}")  # Should be ~6
print(f"First row:\n{df.iloc[0]}")
```

Expected output:
```
Shape: (303, 14)
Missing values: 6
First row:
0     63.0
1      1.0
2      1.0
...
13     0.0
```

## Troubleshooting

### Error: "FileNotFoundError: Dataset not found"

**Cause**: The data file is not at the expected location.

**Solution**:
1. Check the file exists: `ls C:\Users\mushfiq\Downloads\heart+disease\`
2. Verify filename spelling: `processed.cleveland.data` (exact match)
3. Update the path in the script or pass as argument

### Error: "Permission denied"

**Cause**: No read access to the file.

**Solution**:
```bash
# Windows (PowerShell - run as Administrator)
icacls "path\to\file" /grant YourUsername:R

# Linux/macOS
chmod 644 path/to/file
```

### Warning: "Missing values detected"

**This is expected!** The dataset has missing values (represented as '?'), which are handled automatically by the preprocessing pipeline.

## Alternative Datasets (Optional)

The UCI Heart Disease repository includes data from multiple locations:
- `processed.cleveland.data` (used in this project)
- `processed.hungarian.data`
- `processed.switzerland.data`
- `processed.va.data` (Long Beach VA)

To use a different dataset:
1. Download the alternative file
2. Update the path in the execution script
3. Verify the format matches (same 14 columns)

**Note**: Results may differ as these datasets have different characteristics and sample sizes.

## Data Privacy

This is a publicly available, anonymized dataset from 1988. No patient identifiers are included. However, when publishing results:

1. Cite the original source properly (see `references.bib`)
2. Acknowledge the data providers (Cleveland Clinic Foundation)
3. Note the dataset's age and geographic limitations

## Ready to Run?

Once your data file is in place:

1. ✅ File exists at the correct location
2. ✅ Path is updated in the execution script (if changed)
3. ✅ File format verified (303 rows, 14 columns)

You're ready to run the experiments! See `README.md` for execution instructions.
