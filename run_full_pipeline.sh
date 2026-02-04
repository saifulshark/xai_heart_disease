#!/bin/bash

################################################################################
# Heart Disease Classification with Explainable AI - Full Pipeline Runner
# Standalone execution script for Linux/macOS/WSL
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="Heart Disease XAI Pipeline"
VERSION="1.0.0"

################################################################################
# Print banner
################################################################################
print_banner() {
    echo -e "${CYAN}"
    echo "================================================================================"
    echo "  $PROJECT_NAME"
    echo "  Version: $VERSION"
    echo "  Complete End-to-End Experimental Pipeline"
    echo "================================================================================"
    echo -e "${NC}"
}

################################################################################
# Print section header
################################################################################
print_section() {
    echo -e "\n${YELLOW}[$1] $2${NC}"
}

################################################################################
# Print success message
################################################################################
print_success() {
    echo -e "${GREEN}  ✓ $1${NC}"
}

################################################################################
# Print error message
################################################################################
print_error() {
    echo -e "${RED}  ✗ $1${NC}"
}

################################################################################
# Print info message
################################################################################
print_info() {
    echo -e "${CYAN}  → $1${NC}"
}

################################################################################
# Check if command exists
################################################################################
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

################################################################################
# Detect Python command
################################################################################
detect_python() {
    if command_exists python3; then
        PYTHON_CMD=python3
        PIP_CMD=pip3
    elif command_exists python; then
        PYTHON_CMD=python
        PIP_CMD=pip
    else
        print_error "Python not found! Please install Python 3.8+"
        exit 1
    fi
}

################################################################################
# Main execution
################################################################################
main() {
    print_banner
    
    # Determine project root (where this script is located)
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$SCRIPT_DIR"
    
    # Change to project root
    cd "$PROJECT_ROOT"
    
    # =========================================================================
    # Step 1: Check Python
    # =========================================================================
    print_section "1/7" "Checking Python installation"
    detect_python
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
    print_success "Python found: $PYTHON_VERSION"
    
    # =========================================================================
    # Step 2: Setup virtual environment
    # =========================================================================
    print_section "2/7" "Setting up virtual environment"
    VENV_PATH="$PROJECT_ROOT/venv"
    
    if [ ! -d "$VENV_PATH" ]; then
        print_info "Creating new virtual environment..."
        $PYTHON_CMD -m venv "$VENV_PATH"
        print_success "Virtual environment created"
    else
        print_success "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source "$VENV_PATH/bin/activate"
    print_success "Virtual environment activated"
    
    # =========================================================================
    # Step 3: Install dependencies
    # =========================================================================
    print_section "3/7" "Installing dependencies"
    
    if [ -f "$PROJECT_ROOT/requirements.txt" ]; then
        print_info "Upgrading pip..."
        $PYTHON_CMD -m pip install --upgrade pip --quiet
        
        print_info "Installing required packages (this may take a few minutes)..."
        pip install -r "$PROJECT_ROOT/requirements.txt" --quiet
        
        if [ $? -eq 0 ]; then
            print_success "All dependencies installed"
        else
            print_error "Failed to install dependencies"
            exit 1
        fi
    else
        print_error "requirements.txt not found!"
        exit 1
    fi
    
    # =========================================================================
    # Step 4: Locate dataset
    # =========================================================================
    print_section "4/7" "Locating dataset"
    
    # Possible dataset locations (in order of priority)
    DATASET_LOCATIONS=(
        "$PROJECT_ROOT/data/processed.cleveland.data"
        "$HOME/Downloads/heart+disease/processed.cleveland.data"
        "$HOME/Downloads/processed.cleveland.data"
        "/mnt/c/Users/mushfiq/Downloads/heart+disease/processed.cleveland.data"  # WSL path
    )
    
    DATA_PATH=""
    for location in "${DATASET_LOCATIONS[@]}"; do
        if [ -f "$location" ]; then
            DATA_PATH="$location"
            break
        fi
    done
    
    if [ -z "$DATA_PATH" ]; then
        print_error "Dataset not found in standard locations!"
        echo ""
        echo -e "${YELLOW}Please specify the dataset path:${NC}"
        echo "  1. Place processed.cleveland.data in: $PROJECT_ROOT/data/"
        echo "  2. Or run: $0 /path/to/processed.cleveland.data"
        exit 1
    fi
    
    print_success "Dataset found: $DATA_PATH"
    
    # =========================================================================
    # Step 5: Create output directories
    # =========================================================================
    print_section "5/7" "Creating output directories"
    
    mkdir -p "$PROJECT_ROOT/results/tables"
    mkdir -p "$PROJECT_ROOT/results/figures"
    mkdir -p "$PROJECT_ROOT/models"
    
    print_success "Output directories created"
    
    # =========================================================================
    # Step 6: Run the experiment
    # =========================================================================
    print_section "6/7" "Running experimental pipeline"
    
    echo ""
    print_info "This will:"
    print_info "  • Load and preprocess the data"
    print_info "  • Train 4 machine learning models"
    print_info "  • Evaluate model performance"
    print_info "  • Generate SHAP explanations"
    print_info "  • Generate LIME explanations"
    print_info "  • Calculate permutation importance"
    print_info "  • Create partial dependence plots"
    print_info "  • Save all results to results/ directory"
    echo ""
    
    # Change to src directory
    if [ -d "$PROJECT_ROOT/src" ]; then
        cd "$PROJECT_ROOT/src"
    else
        print_error "src/ directory not found!"
        exit 1
    fi
    
    # Run the main experiment
    print_info "Starting experiment (this may take 5-15 minutes)..."
    echo ""
    echo -e "${BLUE}=================================================================================${NC}"
    
    $PYTHON_CMD main_experiment.py "$DATA_PATH"
    EXPERIMENT_STATUS=$?
    
    echo -e "${BLUE}=================================================================================${NC}"
    echo ""
    
    if [ $EXPERIMENT_STATUS -eq 0 ]; then
        print_success "Experiment completed successfully!"
    else
        print_error "Experiment failed with errors"
        exit 1
    fi
    
    # Return to project root
    cd "$PROJECT_ROOT"
    
    # =========================================================================
    # Step 7: Summary
    # =========================================================================
    print_section "7/7" "Pipeline Summary"
    
    echo ""
    echo -e "${CYAN}================================================================================${NC}"
    
    # Count outputs
    FIGURES_COUNT=$(find "$PROJECT_ROOT/results/figures" -name "*.png" 2>/dev/null | wc -l)
    TABLES_COUNT=$(find "$PROJECT_ROOT/results/tables" -type f 2>/dev/null | wc -l)
    MODELS_COUNT=$(find "$PROJECT_ROOT/models" -name "*.pkl" 2>/dev/null | wc -l)
    
    echo -e "\n${GREEN}✓ Pipeline Completed Successfully!${NC}\n"
    echo "Results Location: $PROJECT_ROOT/results"
    echo "  • Figures generated: $FIGURES_COUNT"
    echo "  • Tables generated: $TABLES_COUNT"
    echo "  • Models saved: $MODELS_COUNT"
    
    echo ""
    echo "Key Outputs:"
    echo "  📊 Performance metrics: results/tables/model_performance.csv"
    echo "  📈 ROC curves: results/figures/roc_curves.png"
    echo "  🔍 SHAP explanations: results/figures/shap_*.png"
    echo "  🔍 LIME explanations: results/figures/lime_*.png"
    echo "  📉 Partial Dependence: results/figures/pdp_*.png"
    
    echo ""
    echo -e "${CYAN}================================================================================${NC}"
    
    # Display best model if summary file exists
    SUMMARY_FILE=$(find "$PROJECT_ROOT/results/tables" -name "summary_report.txt" 2>/dev/null | head -n 1)
    if [ -f "$SUMMARY_FILE" ]; then
        echo ""
        echo -e "${YELLOW}Best Model:${NC}"
        grep -A 1 "BEST PERFORMING MODELS BY METRIC:" "$SUMMARY_FILE" | tail -n 1 | head -n 1 || true
    fi
    
    echo ""
    echo -e "${GREEN}✓ All outputs saved and ready for analysis!${NC}"
    echo ""
    
    # =========================================================================
    # Optional: Open results directory
    # =========================================================================
    echo -e "${YELLOW}Would you like to open the results directory? (y/n)${NC}"
    read -r OPEN_RESULTS
    
    if [[ "$OPEN_RESULTS" =~ ^[Yy]$ ]]; then
        if command_exists xdg-open; then
            xdg-open "$PROJECT_ROOT/results" 2>/dev/null &
        elif command_exists open; then
            open "$PROJECT_ROOT/results" 2>/dev/null &
        elif command_exists explorer.exe; then
            # WSL
            explorer.exe "$(wslpath -w "$PROJECT_ROOT/results")" 2>/dev/null &
        else
            echo "Please manually open: $PROJECT_ROOT/results"
        fi
    fi
    
    echo ""
    echo -e "${CYAN}================================================================================${NC}"
    echo -e "${GREEN}Thank you for using the Heart Disease XAI Pipeline!${NC}"
    echo -e "${CYAN}================================================================================${NC}"
    echo ""
}

################################################################################
# Handle command line arguments
################################################################################
if [ $# -eq 1 ]; then
    # Custom dataset path provided
    if [ -f "$1" ]; then
        DATA_PATH="$1"
        export DATA_PATH
        main
    else
        echo -e "${RED}Error: File not found: $1${NC}"
        exit 1
    fi
else
    # Use default locations
    main
fi
