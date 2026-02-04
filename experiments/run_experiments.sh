#!/bin/bash

# ============================================================================
# Heart Disease Classification with Explainable AI
# Bash Execution Script for Ubuntu/Linux
# ============================================================================

echo "==============================================================================="
echo "  HEART DISEASE CLASSIFICATION WITH EXPLAINABLE AI"
echo "  End-to-End Experimental Pipeline"
echo "==============================================================================="
echo ""

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_PATH="$PROJECT_ROOT/data/processed.cleveland.data"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Step 1: Check Python installation
echo -e "${YELLOW}[1/4] Checking Python installation...${NC}"
if command -v python3.12 &> /dev/null; then
    PYTHON_CMD=python3.12
    PYTHON_VERSION=$(python3.12 --version)
    echo -e "${GREEN}  ✓ Python found: $PYTHON_VERSION${NC}"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
    PYTHON_VERSION=$(python3 --version)
    echo -e "${GREEN}  ✓ Python found: $PYTHON_VERSION${NC}"
else
    echo -e "${RED}  ✗ Python not found! Please install Python 3.8+${NC}"
    exit 1
fi

# Step 2: Verify data file exists
echo -e "\n${YELLOW}[2/4] Verifying data file...${NC}"
if [ -f "$DATA_PATH" ]; then
    echo -e "${GREEN}  ✓ Data file found: $DATA_PATH${NC}"
else
    echo -e "${RED}  ✗ Data file not found at: $DATA_PATH${NC}"
    echo -e "${YELLOW}\n  Please ensure processed.cleveland.data exists in the data/ directory${NC}"
    echo -e "${YELLOW}  Current PROJECT_ROOT: $PROJECT_ROOT${NC}"
    exit 1
fi

# Step 3: Prepare output directories
echo -e "\n${YELLOW}[3/4] Preparing output directories...${NC}"
mkdir -p "$PROJECT_ROOT/results/figures"
mkdir -p "$PROJECT_ROOT/results/tables"
mkdir -p "$PROJECT_ROOT/models"
echo -e "${GREEN}  ✓ Output directories ready${NC}"

# Step 4: Run the experiment
echo -e "\n${YELLOW}[4/4] Running experiment pipeline...${NC}"
echo -e "${CYAN}  This will:${NC}"
echo -e "${CYAN}    - Load and preprocess the data${NC}"
echo -e "${CYAN}    - Train 4 machine learning models${NC}"
echo -e "${CYAN}    - Evaluate model performance${NC}"
echo -e "${CYAN}    - Generate explainability visualizations (SHAP, LIME, PDP)${NC}"
echo -e "${CYAN}    - Save all results to results/ directory${NC}"
echo ""

# Change to src directory
cd "$PROJECT_ROOT/src"

# Run the experiment
$PYTHON_CMD main_experiment.py "$DATA_PATH"

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}  ✓ Experiment completed successfully!${NC}"
else
    echo -e "\n${RED}  ✗ Experiment failed with errors${NC}"
    exit 1
fi

# Summary
echo -e "\n${YELLOW}Experiment Summary${NC}"
echo "==============================================================================="

RESULTS_PATH="$PROJECT_ROOT/results"
FIGURES_COUNT=$(find "$RESULTS_PATH/figures" -name "*.png" 2>/dev/null | wc -l)
TABLES_COUNT=$(find "$RESULTS_PATH/tables" -type f 2>/dev/null | wc -l)

echo -e "\nResults Location: $RESULTS_PATH"
echo "  - Figures generated: $FIGURES_COUNT"
echo "  - Tables generated: $TABLES_COUNT"
echo -e "\nKey Outputs:"
echo "  📊 Performance metrics: results/tables/model_performance.csv"
echo "  📈 ROC curves: results/figures/roc_curves.png"
echo "  🔍 SHAP explanations: results/figures/shap_*.png"
echo "  🔍 LIME explanations: results/figures/lime_*.png"
echo "  📉 Partial Dependence: results/figures/pdp_*.png"

echo -e "\n==============================================================================="
echo -e "${GREEN}  ✓ PIPELINE COMPLETED SUCCESSFULLY!${NC}"
echo "==============================================================================="
echo ""

# Return to original directory
cd "$PROJECT_ROOT"
