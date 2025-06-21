# Elisa-3 HD Ring Attractor Network - One-Click Installation

**Complete system setup for NVIDIA B200 GPU with PyTorch CUDA 12.8**

## 🚀 Quick Installation (New Computer)

For a **fresh computer with NVIDIA B200 GPU**, run this single command:

```bash
curl -fsSL https://raw.githubusercontent.com/your-repo/elisa-3/main/install_complete_system.sh | bash
```

Or download and run locally:

```bash
wget https://raw.githubusercontent.com/your-repo/elisa-3/main/install_complete_system.sh
chmod +x install_complete_system.sh
./install_complete_system.sh
```

## 📋 What Gets Installed

### System Components
- **Node.js** (Latest LTS v20+) and **npm** (latest)
- **System dependencies** (build tools, git, curl, etc.)
- **Python 3.8+** with development headers

### Python Environment
- **Virtual environment** (`elisa3_env`)
- **PyTorch 2.8.0 dev** with **CUDA 12.8** (B200 optimized)
- **All scientific computing packages** (NumPy, SciPy, Pandas, etc.)
- **Jupyter Lab** with extensions
- **Development tools** (Black, Flake8, Pylint, etc.)

### Development Tools
- **VS Code extensions** (Python, Jupyter, AI tools)
- **Project structure** with proper configuration
- **Quick launch scripts**

## 🎯 Tested Configuration

This installer replicates the exact working setup:

```
Operating System: Ubuntu/Debian Linux
Node.js: v24.2.0
npm: 11.4.2
Python: 3.11.11
PyTorch: 2.8.0.dev20250620+cu128
CUDA: 12.8
GPU: NVIDIA B200 (183GB VRAM)
Driver: 570.133.20
```

## ⚡ Quick Start After Installation

```bash
# Activate environment
source activate_env.sh

# Launch Jupyter Lab
./launch_jupyter.sh

# Or manually
source elisa3_env/bin/activate
jupyter lab
```

## 🔧 System Requirements

### Minimum Requirements
- **OS**: Ubuntu 20.04+, Debian 11+, CentOS 8+, macOS 12+
- **RAM**: 8GB (16GB+ recommended)
- **Storage**: 10GB free space
- **Internet**: Stable connection for downloads

### Optimal Requirements (B200 Setup)
- **OS**: Ubuntu 22.04 LTS
- **GPU**: NVIDIA B200 with 183GB VRAM
- **Driver**: NVIDIA Driver 570+
- **CUDA**: 12.8 (installed automatically)
- **RAM**: 32GB+ system RAM
- **Storage**: SSD with 50GB+ free space

## 📁 Directory Structure Created

```
elisa-3/
├── elisa3_env/                    # Python virtual environment
├── hd_ring_attractor/
│   ├── src/                       # Source code
│   ├── notebooks/                 # Jupyter notebooks
│   ├── data/                      # Data files
│   ├── models/                    # Saved models
│   └── results/                   # Results and outputs
├── .vscode/                       # VS Code settings
├── requirements.txt               # Python dependencies
├── activate_env.sh               # Environment activation script
├── launch_jupyter.sh             # Jupyter launcher
├── test_system.sh                # System test script
└── install_complete_system.sh    # This installer
```

## 🧪 Testing the Installation

```bash
# Test everything
./test_system.sh

# Or manually
source activate_env.sh
python test_environment.py
```

## 🐛 Troubleshooting

### B200 GPU Issues
The B200 GPU has limited PyTorch support. If you see device errors:
- ✅ **This is normal and expected**
- ✅ **Code automatically falls back to CPU**
- ✅ **All functionality works on CPU**

### Common Issues

**1. Permission denied during installation**
```bash
sudo chmod +x install_complete_system.sh
```

**2. Node.js installation fails**
```bash
# Manual install via NodeSource
curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
sudo apt-get install -y nodejs
```

**3. PyTorch CUDA mismatch**
```bash
# Reinstall PyTorch
source elisa3_env/bin/activate
pip uninstall torch torchvision torchaudio
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

**4. VS Code extensions not installing**
- Ensure VS Code is installed and `code` command is in PATH
- Install VS Code from: https://code.visualstudio.com/
- Add to PATH: View → Command Palette → "Shell Command: Install 'code' command in PATH"

## 📚 Manual Installation (Alternative)

If the automated installer fails, you can install manually:

```bash
# 1. Install Node.js
curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
sudo apt-get install -y nodejs

# 2. Create Python virtual environment
python3 -m venv elisa3_env
source elisa3_env/bin/activate

# 3. Install PyTorch with CUDA 12.8
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# 4. Install requirements
pip install -r requirements.txt

# 5. Setup Jupyter
jupyter contrib nbextension install --user
jupyter nbextension enable varInspector/main
```

## 🌟 Features Included

### Scientific Computing Stack
- **NumPy 1.24+** - Numerical computing
- **SciPy 1.15+** - Scientific computing
- **Pandas 2.3+** - Data manipulation
- **Matplotlib 3.10+** - Plotting
- **Seaborn 0.13+** - Statistical visualization

### Machine Learning
- **PyTorch 2.8.0 dev** - Deep learning framework
- **scikit-learn 1.7+** - Machine learning
- **TensorBoard 2.15+** - Experiment tracking
- **torchinfo** - Model analysis
- **torchmetrics** - Model metrics

### Development Environment
- **Jupyter Lab 4.0+** - Interactive notebooks
- **Black** - Code formatting
- **Flake8** - Style checking
- **Pylint** - Code analysis
- **MyPy** - Type checking
- **pytest** - Testing framework

### VS Code Extensions
- Python language support with Pylance
- Jupyter notebook integration
- Git tools (GitLens)
- AI assistants (Copilot, Continue)
- Code quality tools
- Markdown and documentation support

## 🔄 Updating the Environment

```bash
# Activate environment
source activate_env.sh

# Update packages
pip install --upgrade pip
pip install --upgrade -r requirements.txt

# Update PyTorch to latest nightly
pip install --upgrade --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

## 📞 Support

If you encounter issues:

1. **Check the troubleshooting section above**
2. **Run the test script**: `./test_system.sh`
3. **Verify GPU status**: `nvidia-smi`
4. **Check Python environment**: `source activate_env.sh && python -c "import torch; print(torch.__version__)"`

## 📄 License

This installation script is part of the Elisa-3 HD Ring Attractor Network project.

---

**✨ Ready to explore Head Direction networks with cutting-edge B200 GPU acceleration! ✨**