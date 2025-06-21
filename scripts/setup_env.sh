#!/bin/bash

# ==============================================================================
# Elisa-3 HD Ring Attractor Network - Comprehensive Environment Setup
# Optimized for NVIDIA B200 GPU with CUDA 12.4+
# Last updated: December 2024
# ==============================================================================

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[STATUS]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ==============================================================================
# Main Setup Function
# ==============================================================================

main() {
    echo "===================================================================="
    echo "🧠 Elisa-3 HD Ring Attractor Network - Environment Setup"
    echo "===================================================================="
    echo ""
    
    # Check Python version
    print_status "Checking Python version..."
    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    
    if [[ $(echo "$PYTHON_VERSION < 3.8" | bc) -eq 1 ]]; then
        print_error "Python 3.8+ required. Current version: $PYTHON_VERSION"
        exit 1
    fi
    print_success "Python $PYTHON_VERSION detected"
    
    # Check if in virtual environment
    if [[ -z "$VIRTUAL_ENV" ]]; then
        print_warning "Not in a virtual environment. It's recommended to use a virtual environment."
        echo "To create one: python3 -m venv elisa3_env && source elisa3_env/bin/activate"
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        print_success "Virtual environment detected: $VIRTUAL_ENV"
    fi
    
    # Upgrade pip, wheel, and setuptools
    print_status "Upgrading pip, wheel, and setuptools..."
    python -m pip install --upgrade pip wheel setuptools
    print_success "Package managers upgraded"
    
    # Check CUDA availability
    print_status "Checking CUDA availability..."
    CUDA_AVAILABLE=$(python -c "import subprocess; result = subprocess.run(['nvidia-smi'], capture_output=True); print('True' if result.returncode == 0 else 'False')")
    
    if [[ "$CUDA_AVAILABLE" == "True" ]]; then
        print_success "NVIDIA GPU detected"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
        
        # Check for B200
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
        if [[ "$GPU_NAME" == *"B200"* ]]; then
            print_warning "NVIDIA B200 GPU detected"
            print_warning "B200 (sm_100) has limited PyTorch support. Code will use CPU fallback."
        fi
    else
        print_warning "No NVIDIA GPU detected. Installation will proceed for CPU-only usage."
    fi
    
    # Install PyTorch with CUDA 12.4 support
    print_status "Installing PyTorch 2.6.0 with CUDA 12.4 support..."
    if pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124; then
        print_success "PyTorch installed successfully"
    else
        print_error "PyTorch installation failed"
        exit 1
    fi
    
    # Verify PyTorch installation
    print_status "Verifying PyTorch installation..."
    python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'CUDA device count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('CUDA not available - using CPU')
" || {
    print_error "PyTorch verification failed"
    exit 1
}
    
    # Install core requirements
    print_status "Installing core requirements..."
    
    # Install in batches to avoid conflicts
    print_status "Installing scientific computing packages..."
    pip install numpy>=1.24.1 scipy>=1.15.0 pandas>=2.3.0 scikit-learn>=1.7.0
    
    print_status "Installing visualization packages..."
    pip install matplotlib>=3.10.0 seaborn>=0.13.2 plotly>=5.17.0
    
    print_status "Installing Jupyter ecosystem..."
    pip install jupyter>=1.1.0 jupyterlab>=4.0.8 notebook>=6.5.5 ipywidgets>=8.1.0
    pip install jupyter-contrib-nbextensions>=0.7.0 ipykernel>=6.0.0
    
    print_status "Installing additional packages..."
    pip install tqdm>=4.67.0 rich>=13.0.0
    
    # Install remaining requirements
    print_status "Installing remaining requirements from requirements.txt..."
    if [[ -f "requirements.txt" ]]; then
        # Filter out already installed packages and PyTorch
        grep -v "^#" requirements.txt | grep -v "^torch" | grep -v "^$" > temp_requirements.txt || true
        if [[ -s temp_requirements.txt ]]; then
            pip install -r temp_requirements.txt || print_warning "Some packages failed to install. This may be OK."
        fi
        rm -f temp_requirements.txt
    else
        print_warning "requirements.txt not found"
    fi
    
    # Install Jupyter extensions
    print_status "Setting up Jupyter extensions..."
    jupyter contrib nbextension install --user || print_warning "Jupyter extensions installation had issues"
    jupyter nbextension enable varInspector/main || true
    jupyter nbextension enable execute_time/ExecuteTime || true
    
    # Test imports
    print_status "Testing critical imports..."
    python -c "
import numpy as np
import torch
import matplotlib.pyplot as plt
import jupyter
import tqdm
print('✅ All critical imports successful')
" || {
    print_error "Some imports failed"
    exit 1
}
    
    # Create necessary directories
    print_status "Creating project directories..."
    mkdir -p hd_ring_attractor/src
    mkdir -p hd_ring_attractor/notebooks
    mkdir -p hd_ring_attractor/data
    mkdir -p hd_ring_attractor/models
    mkdir -p hd_ring_attractor/results
    mkdir -p documentation
    print_success "Project directories created"
    
    # Final system info
    echo ""
    echo "===================================================================="
    echo "📊 SYSTEM INFORMATION"
    echo "===================================================================="
    python -c "
import platform
import torch
import numpy as np
import sys

print(f'Platform: {platform.platform()}')
print(f'Python: {sys.version.split()[0]}')
print(f'PyTorch: {torch.__version__}')
print(f'NumPy: {np.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'cuDNN Version: {torch.backends.cudnn.version()}')
"
    
    # B200 specific notes
    if [[ "$GPU_NAME" == *"B200"* ]]; then
        echo ""
        echo "===================================================================="
        echo "🔧 NVIDIA B200 GPU NOTES"
        echo "===================================================================="
        echo "• B200 GPU (sm_100) detected but not fully supported by PyTorch 2.6.0"
        echo "• The code will automatically fall back to CPU execution"
        echo "• This is normal and expected behavior"
        echo "• Full B200 support will come in future PyTorch versions"
        echo "• CPU execution is fully functional for all experiments"
    fi
    
    echo ""
    echo "===================================================================="
    print_success "✅ ENVIRONMENT SETUP COMPLETE!"
    echo "===================================================================="
    echo ""
    echo "📚 Next steps:"
    echo "1. Run validation: python test_environment.py"
    echo "2. Launch Jupyter: jupyter lab"
    echo "3. Open notebook: hd_ring_attractor/notebooks/comprehensive_single_peak_analysis.ipynb"
    echo ""
    echo "🧠 For the single-peak solution demo:"
    echo "   python comprehensive_validation_test.py"
    echo ""
}

# Run main function
main "$@"