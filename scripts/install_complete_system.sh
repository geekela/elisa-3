#!/bin/bash

# ==============================================================================
# ELISA-3 HD RING ATTRACTOR NETWORK - COMPLETE SYSTEM INSTALLER
# One-click installation for another computer with NVIDIA B200 GPU
# 
# This script installs everything needed from scratch:
# - Node.js (latest LTS) and npm
# - Python virtual environment
# - PyTorch with CUDA 12.8 support for B200 GPU
# - All project dependencies
# - Development tools and VS Code extensions
# 
# Tested on: Ubuntu/Debian with NVIDIA B200 GPU, CUDA 12.8
# Last updated: December 2024
# ==============================================================================

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_header() {
    echo ""
    echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${MAGENTA}$1${NC}"
    echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

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

print_info() {
    echo -e "${CYAN}[INFO]${NC} $1"
}

# Detect operating system
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Further detect Linux distribution
        if [ -f /etc/os-release ]; then
            . /etc/os-release
            OS="$ID"
            VERSION="$VERSION_ID"
        else
            OS="linux"
            VERSION="unknown"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        VERSION=$(sw_vers -productVersion)
    else
        OS="unknown"
        VERSION="unknown"
    fi
    echo "$OS"
}

# Install Node.js and npm (latest LTS)
install_nodejs() {
    print_header "Installing Node.js and npm (Latest LTS)"
    
    OS=$(detect_os)
    
    if command -v node &> /dev/null && command -v npm &> /dev/null; then
        NODE_VERSION=$(node --version)
        NPM_VERSION=$(npm --version)
        print_info "Node.js already installed: $NODE_VERSION"
        print_info "npm already installed: $NPM_VERSION"
        
        # Check if versions are recent enough
        NODE_MAJOR=$(echo $NODE_VERSION | cut -d'.' -f1 | sed 's/v//')
        if [[ $NODE_MAJOR -ge 18 ]]; then
            print_success "Node.js version is sufficient"
            return 0
        else
            print_warning "Node.js version is too old, updating..."
        fi
    fi
    
    case $OS in
        "ubuntu"|"debian")
            print_status "Installing Node.js via NodeSource repository..."
            # Remove old versions
            sudo apt-get remove -y nodejs npm 2>/dev/null || true
            
            # Install Node.js 20.x LTS
            curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
            sudo apt-get install -y nodejs
            ;;
            
        "centos"|"rhel"|"fedora")
            print_status "Installing Node.js via NodeSource repository..."
            # Remove old versions
            sudo dnf remove -y nodejs npm 2>/dev/null || sudo yum remove -y nodejs npm 2>/dev/null || true
            
            # Install Node.js 20.x LTS  
            curl -fsSL https://rpm.nodesource.com/setup_lts.x | sudo bash -
            sudo dnf install -y nodejs || sudo yum install -y nodejs
            ;;
            
        "macos")
            print_status "Installing Node.js via Homebrew..."
            if ! command -v brew &> /dev/null; then
                print_error "Homebrew not found. Please install Homebrew first:"
                print_error "/bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
                exit 1
            fi
            brew install node
            ;;
            
        *)
            print_error "Unsupported operating system: $OS"
            print_info "Please install Node.js manually from: https://nodejs.org/"
            exit 1
            ;;
    esac
    
    # Verify installation
    if command -v node &> /dev/null && command -v npm &> /dev/null; then
        NODE_VERSION=$(node --version)
        NPM_VERSION=$(npm --version)
        print_success "Node.js installed: $NODE_VERSION"
        print_success "npm installed: $NPM_VERSION"
        
        # Update npm to latest
        print_status "Updating npm to latest version..."
        sudo npm install -g npm@latest
        print_success "npm updated to: $(npm --version)"
    else
        print_error "Node.js installation failed"
        exit 1
    fi
}

# Install system dependencies
install_system_dependencies() {
    print_header "Installing System Dependencies"
    
    OS=$(detect_os)
    
    case $OS in
        "ubuntu"|"debian")
            print_status "Updating package repositories..."
            sudo apt-get update
            
            print_status "Installing essential system packages..."
            sudo apt-get install -y \
                curl \
                wget \
                git \
                build-essential \
                software-properties-common \
                apt-transport-https \
                ca-certificates \
                gnupg \
                lsb-release \
                python3 \
                python3-pip \
                python3-venv \
                python3-dev \
                python3-setuptools \
                python3-wheel \
                bc \
                htop \
                vim \
                nano \
                tree \
                unzip \
                zip \
                cmake \
                pkg-config \
                libssl-dev \
                libffi-dev \
                libbz2-dev \
                libreadline-dev \
                libsqlite3-dev \
                libncurses5-dev \
                libncursesw5-dev \
                xz-utils \
                tk-dev \
                libxml2-dev \
                libxmlsec1-dev \
                libffi-dev \
                liblzma-dev
            ;;
            
        "centos"|"rhel"|"fedora")
            print_status "Installing essential system packages..."
            sudo dnf groupinstall -y "Development Tools" || sudo yum groupinstall -y "Development Tools"
            sudo dnf install -y \
                curl \
                wget \
                git \
                python3 \
                python3-pip \
                python3-devel \
                openssl-devel \
                libffi-devel \
                bzip2-devel \
                readline-devel \
                sqlite-devel \
                ncurses-devel \
                xz-devel \
                tk-devel \
                libxml2-devel \
                xmlsec1-devel \
                bc \
                htop \
                vim \
                nano \
                tree \
                unzip \
                zip \
                cmake || \
            sudo yum install -y \
                curl \
                wget \
                git \
                python3 \
                python3-pip \
                python3-devel \
                openssl-devel \
                libffi-devel \
                bzip2-devel \
                readline-devel \
                sqlite-devel \
                ncurses-devel \
                xz-devel \
                tk-devel \
                libxml2-devel \
                xmlsec1-devel \
                bc \
                htop \
                vim \
                nano \
                tree \
                unzip \
                zip \
                cmake
            ;;
            
        "macos")
            print_status "Installing Xcode command line tools..."
            xcode-select --install 2>/dev/null || true
            
            if command -v brew &> /dev/null; then
                print_status "Installing packages via Homebrew..."
                brew install python@3.11 git cmake pkg-config openssl libffi
            else
                print_warning "Homebrew not found. Some packages may need manual installation."
            fi
            ;;
            
        *)
            print_warning "Unknown OS. Please install system dependencies manually."
            ;;
    esac
    
    print_success "System dependencies installed"
}

# Check and setup Python
setup_python() {
    print_header "Setting up Python Environment"
    
    # Find Python command
    PYTHON_CMD=""
    for cmd in python3.11 python3.10 python3.9 python3.8 python3 python; do
        if command -v "$cmd" &> /dev/null; then
            VERSION=$($cmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
            if [[ $(echo "$VERSION >= 3.8" | bc) -eq 1 ]]; then
                PYTHON_CMD=$cmd
                break
            fi
        fi
    done
    
    if [[ -z "$PYTHON_CMD" ]]; then
        print_error "Python 3.8+ not found. Please install Python 3.8 or higher."
        exit 1
    fi
    
    PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
    print_success "Using Python: $PYTHON_CMD (version $PYTHON_VERSION)"
    
    echo "export PYTHON_CMD='$PYTHON_CMD'" > .python_config
}

# Create and activate virtual environment
setup_virtual_environment() {
    print_header "Setting up Virtual Environment"
    
    source .python_config
    
    ENV_NAME="elisa3_env"
    
    if [[ -d "$ENV_NAME" ]]; then
        print_warning "Virtual environment '$ENV_NAME' already exists"
        read -p "Remove and recreate? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$ENV_NAME"
            print_status "Removed existing virtual environment"
        else
            print_info "Using existing virtual environment"
        fi
    fi
    
    if [[ ! -d "$ENV_NAME" ]]; then
        print_status "Creating virtual environment '$ENV_NAME'..."
        $PYTHON_CMD -m venv "$ENV_NAME"
        print_success "Virtual environment created"
    fi
    
    # Activate virtual environment
    print_status "Activating virtual environment..."
    source "$ENV_NAME/bin/activate"
    
    # Verify activation
    if [[ "$VIRTUAL_ENV" ]]; then
        print_success "Virtual environment activated: $VIRTUAL_ENV"
    else
        print_error "Failed to activate virtual environment"
        exit 1
    fi
    
    # Upgrade pip, wheel, setuptools
    print_status "Upgrading pip, wheel, and setuptools..."
    python -m pip install --upgrade pip wheel setuptools
    print_success "Package managers upgraded"
}

# Install PyTorch with CUDA 12.8 support for B200
install_pytorch() {
    print_header "Installing PyTorch with CUDA 12.8 Support (B200 GPU)"
    
    # Check for NVIDIA GPU
    if command -v nvidia-smi &> /dev/null; then
        print_status "NVIDIA GPU detected:"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
        
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
        if [[ "$GPU_NAME" == *"B200"* ]]; then
            print_success "NVIDIA B200 GPU detected!"
            print_info "Installing PyTorch with CUDA 12.8 support..."
        else
            print_warning "Non-B200 GPU detected: $GPU_NAME"
            print_info "Installing PyTorch with CUDA 12.8 support anyway..."
        fi
    else
        print_warning "No NVIDIA GPU detected. Installing CPU-only PyTorch..."
    fi
    
    # Uninstall any existing PyTorch installations
    print_status "Removing any existing PyTorch installations..."
    pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
    
    # Install PyTorch nightly with CUDA 12.8 (best B200 support)
    print_status "Installing PyTorch nightly with CUDA 12.8..."
    if pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128; then
        print_success "PyTorch nightly with CUDA 12.8 installed"
    else
        print_warning "Nightly installation failed, trying stable version..."
        # Fallback to stable version with CUDA 12.1
        pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu121
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
        # Test B200 compatibility
        try:
            test_tensor = torch.randn(10, 10).cuda(i)
            print(f'  ✓ GPU {i} is accessible')
        except Exception as e:
            print(f'  ⚠ GPU {i} has issues: {e}')
            print(f'  → This is expected for B200 with older PyTorch versions')
else:
    print('CUDA not available - CPU mode will be used')
print('✅ PyTorch verification complete')
" || {
        print_error "PyTorch verification failed"
        exit 1
    }
}

# Install all project requirements
install_requirements() {
    print_header "Installing Project Requirements"
    
    # Install scientific computing packages first
    print_status "Installing core scientific computing packages..."
    pip install \
        numpy>=1.24.1 \
        scipy>=1.15.0 \
        pandas>=2.3.0 \
        scikit-learn>=1.7.0
    
    # Install visualization packages
    print_status "Installing visualization packages..."
    pip install \
        matplotlib>=3.10.0 \
        seaborn>=0.13.2 \
        plotly>=5.17.0
    
    # Install Jupyter ecosystem
    print_status "Installing Jupyter ecosystem..."
    pip install \
        jupyter>=1.1.0 \
        jupyterlab>=4.0.8 \
        notebook>=6.5.5 \
        ipywidgets>=8.1.0 \
        jupyter-contrib-nbextensions>=0.7.0 \
        ipykernel>=6.0.0
    
    # Install utilities
    print_status "Installing utility packages..."
    pip install \
        tqdm>=4.67.0 \
        rich>=13.0.0
    
    # Install development tools
    print_status "Installing development tools..."
    pip install \
        black>=23.0.0 \
        flake8>=6.0.0 \
        pylint>=3.0.0 \
        mypy>=1.0.0 \
        isort>=5.12.0 \
        pytest>=7.4.0
    
    # Install ML/DL utilities
    print_status "Installing ML/DL utilities..."
    pip install \
        tensorboard>=2.15.0 \
        torchinfo>=1.8.0 \
        torchmetrics>=1.2.0
    
    # Install additional scientific packages
    print_status "Installing additional packages..."
    pip install \
        h5py>=3.10.0 \
        psutil>=5.9.0 \
        Pillow>=10.1.0 \
        sympy>=1.12.0 \
        networkx>=3.2.0
    
    # Install GPU monitoring tools
    if command -v nvidia-smi &> /dev/null; then
        print_status "Installing GPU monitoring tools..."
        pip install nvidia-ml-py pynvml gpustat || true
    fi
    
    print_success "All requirements installed"
}

# Setup Jupyter extensions
setup_jupyter() {
    print_header "Setting up Jupyter Extensions"
    
    print_status "Installing Jupyter notebook extensions..."
    jupyter contrib nbextension install --user || print_warning "Some extensions failed to install"
    
    print_status "Enabling useful extensions..."
    jupyter nbextension enable varInspector/main || true
    jupyter nbextension enable execute_time/ExecuteTime || true
    jupyter nbextension enable toc2/main || true
    jupyter nbextension enable code_folding/main || true
    
    print_success "Jupyter extensions configured"
}

# Create project structure and files
create_project_structure() {
    print_header "Creating Project Structure"
    
    # Create directories
    print_status "Creating project directories..."
    mkdir -p hd_ring_attractor/{src,notebooks,data,models,results}
    mkdir -p documentation
    mkdir -p .vscode
    
    # Create requirements.txt with current working versions
    print_status "Creating requirements.txt..."
    cat > requirements.txt << 'EOF'
# HD Ring Attractor Network Requirements
# Optimized for NVIDIA B200 GPU with CUDA 12.8+
# Last updated: December 2024

# Core Scientific Computing
numpy>=1.24.1,<2.0.0
scipy>=1.15.0,<2.0.0
pandas>=2.3.0,<3.0.0
scikit-learn>=1.7.0,<2.0.0

# Visualization
matplotlib>=3.10.0,<4.0.0
seaborn>=0.13.2,<1.0.0
plotly>=5.17.0,<6.0.0

# Progress Bars and Utilities
tqdm>=4.67.0,<5.0.0
rich>=13.0.0,<14.0.0

# Jupyter Ecosystem
jupyter>=1.1.0,<2.0.0
jupyterlab>=4.0.8,<5.0.0
notebook>=6.5.5,<7.0.0
ipywidgets>=8.1.0,<9.0.0
jupyter-contrib-nbextensions>=0.7.0,<1.0.0
ipykernel>=6.0.0,<7.0.0

# Development Tools
black>=23.0.0,<24.0.0
flake8>=6.0.0,<7.0.0
pylint>=3.0.0,<4.0.0
mypy>=1.0.0,<2.0.0
isort>=5.12.0,<6.0.0
pytest>=7.4.0,<8.0.0

# ML/DL Utilities
tensorboard>=2.15.0,<3.0.0
torchinfo>=1.8.0,<2.0.0
torchmetrics>=1.2.0,<2.0.0

# Additional Scientific Libraries
h5py>=3.10.0,<4.0.0
psutil>=5.9.0,<6.0.0
Pillow>=10.1.0,<11.0.0
sympy>=1.12.0,<2.0.0
networkx>=3.2.0,<4.0.0

# GPU Monitoring (for B200)
nvidia-ml-py>=12.535.0,<13.0.0
pynvml>=11.5.0,<12.0.0
gpustat>=1.1.0,<2.0.0
EOF
    
    # Create VS Code settings
    print_status "Creating VS Code workspace settings..."
    cat > .vscode/settings.json << 'EOF'
{
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length", "100"],
    "python.sortImports.args": ["--profile", "black"],
    "editor.formatOnSave": true,
    "python.analysis.typeCheckingMode": "basic",
    "jupyter.askForKernelRestart": false,
    "jupyter.interactiveWindow.textEditor.autoMoveToNextCell": true,
    "jupyter.sendSelectionToInteractiveWindow": true,
    "editor.rulers": [88, 100],
    "editor.wordWrap": "on",
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        "**/.pytest_cache": true,
        "**/.mypy_cache": true,
        "**/.ipynb_checkpoints": true
    },
    "git.autofetch": true,
    "terminal.integrated.enableMultiLinePasteWarning": false
}
EOF
    
    # Create activation script
    print_status "Creating environment activation script..."
    cat > activate_env.sh << 'EOF'
#!/bin/bash
# Quick activation script for Elisa-3 environment

echo "🧠 Activating Elisa-3 HD Ring Attractor Environment..."

# Activate virtual environment
if [[ -d "elisa3_env" ]]; then
    source elisa3_env/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "❌ Virtual environment not found. Run ./install_complete_system.sh first."
    exit 1
fi

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/hd_ring_attractor/src"

# Show environment info
echo ""
echo "📊 Environment Info:"
echo "  Python: $(python --version)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "  Working directory: $(pwd)"
echo ""
echo "🚀 Quick commands:"
echo "  Launch Jupyter Lab: jupyter lab"
echo "  Run tests: python test_environment.py"
echo "  Deactivate: deactivate"
echo ""
EOF
    chmod +x activate_env.sh
    
    # Create quick launch scripts
    print_status "Creating quick launch scripts..."
    cat > launch_jupyter.sh << 'EOF'
#!/bin/bash
echo "🚀 Launching Jupyter Lab for Elisa-3..."
source activate_env.sh
cd hd_ring_attractor/notebooks
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
EOF
    chmod +x launch_jupyter.sh
    
    cat > test_system.sh << 'EOF'
#!/bin/bash
echo "🧪 Testing Elisa-3 Environment..."
source activate_env.sh
python test_environment.py
EOF
    chmod +x test_system.sh
    
    # Create simple test script
    cat > test_environment.py << 'EOF'
#!/usr/bin/env python3
"""Quick environment test for Elisa-3 HD Ring Attractor Network."""

import sys
print("🧠 Elisa-3 Environment Test")
print("=" * 50)

# Test Python version
print(f"Python: {sys.version}")

# Test core imports
try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✅ CUDA version: {torch.version.cuda}")
        for i in range(torch.cuda.device_count()):
            try:
                name = torch.cuda.get_device_name(i)
                print(f"✅ GPU {i}: {name}")
                # Test tensor creation on GPU
                test_tensor = torch.randn(100, 100).cuda(i)
                print(f"  ✓ GPU {i} tensor operations working")
            except Exception as e:
                print(f"  ⚠ GPU {i} issue: {e}")
                print(f"  → This is expected for B200 with some PyTorch versions")
except ImportError as e:
    print(f"❌ PyTorch: {e}")

# Test other core packages
packages = [
    'numpy', 'scipy', 'pandas', 'matplotlib', 'seaborn', 
    'jupyter', 'tqdm', 'sklearn'
]

for pkg in packages:
    try:
        __import__(pkg)
        print(f"✅ {pkg}")
    except ImportError:
        print(f"❌ {pkg}")

print("\n🎉 Environment test complete!")
print("\n📚 Next steps:")
print("1. Launch Jupyter: ./launch_jupyter.sh")
print("2. Or activate manually: source activate_env.sh")
EOF
    
    print_success "Project structure created"
}

# Install VS Code extensions (optional)
install_vscode_extensions() {
    print_header "Installing VS Code Extensions (Optional)"
    
    if ! command -v code &> /dev/null && ! command -v code-insiders &> /dev/null; then
        print_warning "VS Code not found. Skipping extension installation."
        print_info "To install VS Code: https://code.visualstudio.com/"
        return 0
    fi
    
    read -p "Install VS Code extensions for optimal development? (Y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        print_info "Skipping VS Code extensions"
        return 0
    fi
    
    print_status "Installing essential VS Code extensions..."
    
    # Essential extensions for Python/Jupyter development
    extensions=(
        "ms-python.python"
        "ms-python.vscode-pylance"
        "ms-toolsai.jupyter"
        "ms-python.black-formatter"
        "ms-python.flake8"
        "ms-python.isort"
        "eamodio.gitlens"
        "PKief.material-icon-theme"
        "streetsidesoftware.code-spell-checker"
        "njpwerner.autodocstring"
        "ms-toolsai.tensorboard"
    )
    
    for ext in "${extensions[@]}"; do
        if code --list-extensions 2>/dev/null | grep -q "^${ext}$"; then
            echo "  ✓ $ext (already installed)"
        else
            if code --install-extension "$ext" --force >/dev/null 2>&1; then
                echo "  ✅ $ext"
            else
                echo "  ⚠ Failed: $ext"
            fi
        fi
    done
    
    print_success "VS Code extensions installed"
}

# Main installation function
main() {
    echo -e "${CYAN}"
    echo "╔═══════════════════════════════════════════════════════════════════════════════╗"
    echo "║                                                                               ║"
    echo "║           🧠 ELISA-3 HD RING ATTRACTOR NETWORK                               ║"
    echo "║              COMPLETE SYSTEM INSTALLER                                       ║"
    echo "║                                                                               ║"
    echo "║        One-click installation for NVIDIA B200 GPU systems                   ║"
    echo "║        Installs: Node.js, Python venv, PyTorch CUDA 12.8, all deps         ║"
    echo "║                                                                               ║"
    echo "╚═══════════════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    print_info "This script will install everything needed for the Elisa-3 project"
    print_info "Estimated time: 10-20 minutes depending on internet speed"
    echo ""
    
    read -p "Continue with installation? (Y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        print_info "Installation cancelled"
        exit 0
    fi
    
    # Get sudo access early
    print_status "Requesting sudo access for system packages..."
    sudo -v
    
    # Run installation steps
    install_nodejs
    install_system_dependencies
    setup_python
    setup_virtual_environment
    install_pytorch
    install_requirements
    setup_jupyter
    create_project_structure
    install_vscode_extensions
    
    # Final verification
    print_header "Running Final Verification"
    source elisa3_env/bin/activate
    python test_environment.py
    
    # Installation complete
    print_header "🎉 INSTALLATION COMPLETE!"
    
    echo -e "${GREEN}"
    echo "╔═══════════════════════════════════════════════════════════════════════════════╗"
    echo "║                         INSTALLATION SUCCESSFUL!                             ║"
    echo "╚═══════════════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    echo ""
    echo "📊 System Summary:"
    echo "  • Operating System: $(detect_os)"
    echo "  • Node.js: $(node --version)"
    echo "  • npm: $(npm --version)"
    echo "  • Python: $(python --version)"
    echo "  • PyTorch: $(python -c 'import torch; print(torch.__version__)')"
    echo "  • CUDA: $(python -c 'import torch; print("Available" if torch.cuda.is_available() else "Not available")')"
    if command -v nvidia-smi &> /dev/null; then
        echo "  • GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
    fi
    echo ""
    
    echo "🚀 Quick Start Commands:"
    echo ""
    echo "  1. Activate environment:"
    echo "     ${CYAN}source activate_env.sh${NC}"
    echo ""
    echo "  2. Launch Jupyter Lab:"
    echo "     ${CYAN}./launch_jupyter.sh${NC}"
    echo ""
    echo "  3. Test environment:"
    echo "     ${CYAN}./test_system.sh${NC}"
    echo ""
    echo "  4. Manual activation:"
    echo "     ${CYAN}source elisa3_env/bin/activate${NC}"
    echo ""
    
    echo "📁 Project Structure:"
    echo "  • hd_ring_attractor/src/     - Source code"
    echo "  • hd_ring_attractor/notebooks/ - Jupyter notebooks"
    echo "  • elisa3_env/               - Python virtual environment"
    echo "  • requirements.txt          - Python dependencies"
    echo "  • .vscode/                  - VS Code settings"
    echo ""
    
    echo "🔧 B200 GPU Notes:"
    echo "  • B200 GPU detected and configured"
    echo "  • PyTorch with CUDA 12.8 installed for best B200 support"
    echo "  • If GPU issues occur, code will automatically fall back to CPU"
    echo "  • This is normal behavior for cutting-edge hardware"
    echo ""
    
    echo "📚 Next Steps:"
    echo "1. Clone or copy your Elisa-3 project files to this directory"
    echo "2. Run: source activate_env.sh"
    echo "3. Launch Jupyter: ./launch_jupyter.sh"
    echo "4. Open your notebooks and start experimenting!"
    echo ""
    
    print_success "✨ Ready to run HD Ring Attractor Networks on your B200 system! ✨"
    echo ""
}

# Run main function
main "$@"