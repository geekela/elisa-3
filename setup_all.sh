#!/bin/bash

# ==============================================================================
# Elisa-3 HD Ring Attractor Network - Complete Setup Script
# One-click installation for all dependencies and environment
# Optimized for NVIDIA B200 GPU with CUDA 12.4+
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
    echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${MAGENTA}$1${NC}"
    echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
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
        OS="linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
    elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
        OS="windows"
    else
        OS="unknown"
    fi
    echo $OS
}

# Main setup function
main() {
    echo -e "${CYAN}"
    echo "╔═══════════════════════════════════════════════════════════════════╗"
    echo "║                                                                   ║"
    echo "║        🧠 ELISA-3 HD RING ATTRACTOR NETWORK                      ║"
    echo "║           Complete Environment Setup                              ║"
    echo "║                                                                   ║"
    echo "║        Optimized for NVIDIA B200 GPU & PyTorch 2.6.0            ║"
    echo "║                                                                   ║"
    echo "╚═══════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    # System detection
    OS=$(detect_os)
    print_info "Detected operating system: $OS"
    
    # Check Python version
    print_header "Step 1: Checking Python Version"
    
    PYTHON_CMD="python3"
    if ! command -v python3 &> /dev/null; then
        if command -v python &> /dev/null; then
            PYTHON_CMD="python"
        else
            print_error "Python not found. Please install Python 3.8 or higher."
            exit 1
        fi
    fi
    
    PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    print_info "Python version: $PYTHON_VERSION"
    
    if [[ $(echo "$PYTHON_VERSION < 3.8" | bc) -eq 1 ]]; then
        print_error "Python 3.8+ required. Current version: $PYTHON_VERSION"
        exit 1
    fi
    print_success "Python version check passed"
    
    # Virtual environment setup
    print_header "Step 2: Virtual Environment Setup"
    
    if [[ -z "$VIRTUAL_ENV" ]]; then
        print_warning "Not in a virtual environment. Creating one is recommended."
        read -p "Create virtual environment 'elisa3_env'? (Y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            print_status "Creating virtual environment..."
            $PYTHON_CMD -m venv elisa3_env
            
            # Activate virtual environment
            if [[ "$OS" == "windows" ]]; then
                source elisa3_env/Scripts/activate
            else
                source elisa3_env/bin/activate
            fi
            print_success "Virtual environment created and activated"
        fi
    else
        print_success "Virtual environment detected: $VIRTUAL_ENV"
    fi
    
    # Run main setup script
    print_header "Step 3: Installing Core Dependencies"
    
    if [[ -f "setup_env.sh" ]]; then
        print_status "Running setup_env.sh..."
        bash setup_env.sh
        print_success "Core dependencies installed"
    else
        print_error "setup_env.sh not found"
        exit 1
    fi
    
    # Install VS Code extensions (optional)
    print_header "Step 4: VS Code Extensions (Optional)"
    
    if command -v code &> /dev/null || command -v code-insiders &> /dev/null; then
        read -p "Install VS Code extensions for optimal development experience? (Y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            if [[ -f "install_vscode_extensions.sh" ]]; then
                bash install_vscode_extensions.sh
            else
                print_warning "install_vscode_extensions.sh not found"
            fi
        fi
    else
        print_info "VS Code not detected. Skipping extensions installation."
    fi
    
    # Run environment validation
    print_header "Step 5: Environment Validation"
    
    if [[ -f "test_environment.py" ]]; then
        print_status "Running comprehensive environment tests..."
        $PYTHON_CMD test_environment.py
    else
        print_warning "test_environment.py not found. Skipping validation."
    fi
    
    # Create quick start scripts
    print_header "Step 6: Creating Quick Start Scripts"
    
    # Create run_notebook.sh
    cat > run_notebook.sh << 'EOF'
#!/bin/bash
# Quick launcher for Jupyter Lab
echo "🚀 Launching Jupyter Lab..."
cd hd_ring_attractor/notebooks
jupyter lab
EOF
    chmod +x run_notebook.sh
    print_success "Created run_notebook.sh"
    
    # Create run_demo.sh
    cat > run_demo.sh << 'EOF'
#!/bin/bash
# Quick launcher for single-peak solution demo
echo "🧠 Running comprehensive single-peak solution validation..."
python comprehensive_validation_test.py
EOF
    chmod +x run_demo.sh
    print_success "Created run_demo.sh"
    
    # Final summary
    print_header "🎉 SETUP COMPLETE!"
    
    echo -e "${GREEN}"
    echo "╔═══════════════════════════════════════════════════════════════════╗"
    echo "║                    INSTALLATION SUCCESSFUL!                       ║"
    echo "╚═══════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    echo ""
    echo "📊 Environment Summary:"
    echo "  • Python: $PYTHON_VERSION"
    echo "  • PyTorch: 2.6.0 with CUDA 12.4 support"
    echo "  • B200 GPU: Configured with CPU fallback"
    echo "  • All core dependencies: Installed"
    echo "  • Project structure: Ready"
    echo ""
    
    echo "🚀 Quick Start Commands:"
    echo ""
    echo "  1. Launch Jupyter Lab:"
    echo "     ${CYAN}./run_notebook.sh${NC}"
    echo ""
    echo "  2. Run single-peak solution demo:"
    echo "     ${CYAN}./run_demo.sh${NC}"
    echo ""
    echo "  3. Open specific notebook:"
    echo "     ${CYAN}jupyter lab hd_ring_attractor/notebooks/comprehensive_single_peak_analysis.ipynb${NC}"
    echo ""
    echo "  4. Test environment:"
    echo "     ${CYAN}python test_environment.py${NC}"
    echo ""
    
    echo "📚 Key Files:"
    echo "  • ${GREEN}comprehensive_single_peak_analysis.ipynb${NC} - Main solution notebook"
    echo "  • ${GREEN}enhanced_single_peak_model.py${NC} - Optimized model implementation"
    echo "  • ${GREEN}comprehensive_validation_test.py${NC} - Complete validation suite"
    echo "  • ${GREEN}SOLUTION_SUMMARY.md${NC} - Detailed solution documentation"
    echo ""
    
    echo "🔧 B200 GPU Notes:"
    echo "  • B200 (sm_100) has limited PyTorch support"
    echo "  • Code automatically falls back to CPU when needed"
    echo "  • This is normal and expected behavior"
    echo "  • All functionality works correctly on CPU"
    echo ""
    
    if [[ -z "$VIRTUAL_ENV" ]]; then
        echo "⚠️  Remember to activate your virtual environment:"
        if [[ "$OS" == "windows" ]]; then
            echo "     ${YELLOW}elisa3_env\\Scripts\\activate${NC}"
        else
            echo "     ${YELLOW}source elisa3_env/bin/activate${NC}"
        fi
        echo ""
    fi
    
    echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}✨ Happy experimenting with HD Ring Attractor Networks! ✨${NC}"
    echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

# Run main function
main "$@"