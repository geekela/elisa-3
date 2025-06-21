#!/usr/bin/env python3
"""
Comprehensive test script for the Elisa-3 HD Ring Attractor Network environment.
This script validates that all dependencies are installed and working properly.
Optimized for B200 GPU with automatic CPU fallback.
"""

import sys
import pathlib
import platform
from typing import List, Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"{title.upper()}")
    print(f"{'='*60}")


def print_subsection(title: str):
    """Print a formatted subsection header."""
    print(f"\n{title}:")
    print(f"{'-'*40}")


def test_system_info() -> Tuple[bool, List[str]]:
    """Get and display system information."""
    errors = []
    
    try:
        import platform
        import multiprocessing
        import os
        
        print(f"Platform: {platform.platform()}")
        print(f"Python: {platform.python_version()}")
        print(f"CPU cores: {multiprocessing.cpu_count()}")
        print(f"Memory: {os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024**3):.1f} GB" if hasattr(os, 'sysconf') else "N/A")
        
    except Exception as e:
        errors.append(f"System info failed: {e}")
    
    return len(errors) == 0, errors


def test_core_imports() -> Tuple[bool, List[str]]:
    """Test all core required imports."""
    errors = []
    
    print_subsection("Core Scientific Libraries")
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError as e:
        errors.append(f"NumPy import failed: {e}")
    
    try:
        import scipy
        print(f"✓ SciPy {scipy.__version__}")
    except ImportError as e:
        errors.append(f"SciPy import failed: {e}")
    
    try:
        import pandas as pd
        print(f"✓ Pandas {pd.__version__}")
    except ImportError as e:
        errors.append(f"Pandas import failed: {e}")
    
    try:
        import sklearn
        print(f"✓ Scikit-learn {sklearn.__version__}")
    except ImportError as e:
        errors.append(f"Scikit-learn import failed: {e}")
    
    print_subsection("Visualization Libraries")
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend for testing
        import matplotlib.pyplot as plt
        print(f"✓ Matplotlib {matplotlib.__version__}")
    except ImportError as e:
        errors.append(f"Matplotlib import failed: {e}")
    
    try:
        import seaborn as sns
        print(f"✓ Seaborn {sns.__version__}")
    except ImportError as e:
        errors.append(f"Seaborn import failed: {e}")
    
    try:
        import plotly
        print(f"✓ Plotly {plotly.__version__}")
    except ImportError as e:
        errors.append(f"Plotly import failed: {e}")
    
    print_subsection("Utilities")
    try:
        import tqdm
        print(f"✓ tqdm {tqdm.__version__}")
    except ImportError as e:
        errors.append(f"tqdm import failed: {e}")
    
    import rich
    try:
        print(f"✓ Rich {rich.__version__}")
    except AttributeError:
        # Rich doesn't expose __version__ directly
        print("✓ Rich (version check not available)")
    
    return len(errors) == 0, errors


def test_pytorch_b200() -> Tuple[bool, List[str]]:
    """Test PyTorch functionality with B200 GPU considerations."""
    errors = []
    info = {}
    
    print_subsection("PyTorch Configuration")
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"cuDNN version: {torch.backends.cudnn.version()}")
            print(f"CUDA device count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                device_name = torch.cuda.get_device_name(i)
                device_props = torch.cuda.get_device_properties(i)
                print(f"\nGPU {i}: {device_name}")
                print(f"  Compute capability: {device_props.major}.{device_props.minor}")
                print(f"  Memory: {device_props.total_memory / (1024**3):.1f} GB")
                
                # Check for B200
                if "B200" in device_name or device_props.major >= 10:
                    print("  ⚠️  B200 GPU detected - will use CPU fallback")
                    info['b200_detected'] = True
        else:
            print("No CUDA devices available - using CPU")
        
        # Test tensor operations with automatic fallback
        print_subsection("Testing Tensor Operations")
        device = torch.device('cpu')  # Default to CPU for B200 compatibility
        
        if torch.cuda.is_available() and not info.get('b200_detected', False):
            try:
                # Try GPU first if not B200
                test_device = torch.device('cuda:0')
                test_tensor = torch.randn(100, 100, device=test_device)
                result = torch.matmul(test_tensor, test_tensor.T)
                device = test_device
                print(f"✓ GPU tensor operations successful on {device}")
            except Exception as e:
                print(f"⚠️  GPU operations failed, using CPU: {e}")
        
        # CPU operations (always test)
        cpu_tensor = torch.randn(100, 100, device='cpu')
        cpu_result = torch.matmul(cpu_tensor, cpu_tensor.T)
        print(f"✓ CPU tensor operations successful")
        
        # Test autograd
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True, device=device)
        y = x ** 2
        y.sum().backward()
        print(f"✓ Autograd working on {device}")
        
        # Test neural network layers
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 1)
        ).to(device)
        
        test_input = torch.randn(5, 10, device=device)
        output = model(test_input)
        print(f"✓ Neural network layers working on {device}")
        
    except Exception as e:
        errors.append(f"PyTorch test failed: {e}")
    
    return len(errors) == 0, errors


def test_jupyter_environment() -> Tuple[bool, List[str]]:
    """Test Jupyter and notebook-related functionality."""
    errors = []
    
    print_subsection("Jupyter Ecosystem")
    try:
        import jupyter
        print("✓ Jupyter core installed")
    except ImportError as e:
        errors.append(f"Jupyter import failed: {e}")
    
    try:
        import jupyterlab
        print("✓ JupyterLab installed")
    except ImportError as e:
        errors.append(f"JupyterLab import failed: {e}")
    
    try:
        import notebook
        print("✓ Jupyter Notebook installed")
    except ImportError as e:
        errors.append(f"Notebook import failed: {e}")
    
    try:
        import ipykernel
        print("✓ IPython kernel installed")
    except ImportError as e:
        errors.append(f"IPython kernel import failed: {e}")
    
    try:
        import ipywidgets
        print("✓ IPython widgets installed")
    except ImportError as e:
        warnings.warn(f"IPython widgets not installed (optional): {e}")
    
    try:
        import nbformat
        import nbconvert
        print("✓ Notebook format/convert tools installed")
    except ImportError as e:
        warnings.warn(f"Notebook tools not installed (optional): {e}")
    
    return len(errors) == 0, errors


def test_custom_modules() -> Tuple[bool, List[str]]:
    """Test custom ring attractor modules."""
    errors = []
    
    print_subsection("Custom HD Ring Attractor Modules")
    try:
        import torch
        
        # Add src to path
        current_dir = pathlib.Path(__file__).parent
        src_path = current_dir / "hd_ring_attractor" / "src"
        
        if not src_path.exists():
            errors.append(f"Source directory not found: {src_path}")
            print(f"❌ Source directory not found: {src_path}")
            return False, errors
        
        sys.path.insert(0, str(src_path))
        
        # Test core imports
        try:
            from models import RingAttractorNetwork
            print("✓ models.py imported")
        except ImportError as e:
            errors.append(f"models.py import failed: {e}")
        
        try:
            from utils import generate_trajectory, angle_to_input, compute_error
            print("✓ utils.py imported")
        except ImportError as e:
            errors.append(f"utils.py import failed: {e}")
        
        try:
            from single_peak_model import SinglePeakRingAttractor, create_single_peak_model
            print("✓ single_peak_model.py imported")
        except ImportError as e:
            warnings.warn(f"single_peak_model.py not found (optional): {e}")
        
        try:
            from enhanced_single_peak_model import EnhancedSinglePeakRingAttractor, create_enhanced_single_peak_model
            print("✓ enhanced_single_peak_model.py imported")
        except ImportError as e:
            warnings.warn(f"enhanced_single_peak_model.py not found (optional): {e}")
        
        # Test model creation
        if 'RingAttractorNetwork' in locals():
            device = torch.device('cpu')  # Use CPU for B200 compatibility
            model = RingAttractorNetwork(n_exc=64, n_inh=16, device=device)
            print("✓ Basic model creation successful")
            
            # Test forward pass
            test_input = torch.randn(64, device=device)
            output = model(test_input, steps=1)
            print("✓ Model forward pass successful")
        
        # Test utilities
        if 'generate_trajectory' in locals():
            test_angles, _ = generate_trajectory(10, dt=0.1)
            print("✓ Trajectory generation working")
        
        if 'angle_to_input' in locals():
            test_input = angle_to_input(torch.tensor(0.0, device=device), n_exc=64, device=device)
            print("✓ Input generation working")
        
    except Exception as e:
        errors.append(f"Custom modules test failed: {e}")
        import traceback
        traceback.print_exc()
    
    return len(errors) == 0, errors


def test_visualization() -> Tuple[bool, List[str]]:
    """Test visualization capabilities."""
    errors = []
    
    print_subsection("Testing Visualization Pipeline")
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Create test plot
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.linspace(0, 2*np.pi, 100)
        y = np.sin(x)
        ax.plot(x, y, 'b-', linewidth=2)
        ax.set_title("Test Sine Wave")
        ax.set_xlabel("x")
        ax.set_ylabel("sin(x)")
        ax.grid(True, alpha=0.3)
        
        # Save to verify it works
        test_path = pathlib.Path("/tmp/test_plot.png")
        plt.savefig(test_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        if test_path.exists():
            print("✓ Matplotlib plotting and saving working")
            test_path.unlink()  # Clean up
        else:
            errors.append("Failed to save test plot")
        
        # Test seaborn
        import seaborn as sns
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(6, 4))
        data = np.random.randn(100)
        sns.histplot(data, ax=ax)
        plt.close(fig)
        print("✓ Seaborn plotting working")
        
    except Exception as e:
        errors.append(f"Visualization test failed: {e}")
    
    return len(errors) == 0, errors


def test_optional_packages() -> Dict[str, bool]:
    """Test optional packages and report availability."""
    optional_status = {}
    
    print_subsection("Optional Packages")
    
    # Development tools
    for package in ['black', 'flake8', 'pylint', 'mypy', 'pytest']:
        try:
            __import__(package)
            optional_status[package] = True
            print(f"✓ {package} installed")
        except ImportError:
            optional_status[package] = False
            print(f"○ {package} not installed (optional)")
    
    # ML/DL tools
    for package in ['tensorboard', 'wandb', 'torchinfo', 'torchmetrics']:
        try:
            __import__(package)
            optional_status[package] = True
            print(f"✓ {package} installed")
        except ImportError:
            optional_status[package] = False
            print(f"○ {package} not installed (optional)")
    
    # Performance tools
    for package in ['numba', 'ray', 'dask']:
        try:
            __import__(package)
            optional_status[package] = True
            print(f"✓ {package} installed")
        except ImportError:
            optional_status[package] = False
            print(f"○ {package} not installed (optional)")
    
    return optional_status


def main():
    """Run all tests and report results."""
    print_section("ELISA-3 HD RING ATTRACTOR ENVIRONMENT VALIDATION")
    print("Comprehensive test for B200 GPU optimized environment")
    
    all_tests_passed = True
    all_errors = []
    
    # Run tests
    tests = [
        ("System Information", test_system_info),
        ("Core Imports", test_core_imports),
        ("PyTorch B200 Compatibility", test_pytorch_b200),
        ("Jupyter Environment", test_jupyter_environment),
        ("Custom Modules", test_custom_modules),
        ("Visualization", test_visualization),
    ]
    
    for test_name, test_func in tests:
        print_section(test_name)
        try:
            passed, errors = test_func()
            if passed:
                print(f"\n✅ {test_name} PASSED")
            else:
                print(f"\n❌ {test_name} FAILED")
                all_tests_passed = False
                all_errors.extend(errors)
        except Exception as e:
            print(f"\n❌ {test_name} CRASHED: {e}")
            all_tests_passed = False
            all_errors.append(f"{test_name} crashed: {e}")
            import traceback
            traceback.print_exc()
    
    # Test optional packages
    print_section("Optional Packages Status")
    optional_status = test_optional_packages()
    
    # Final report
    print_section("FINAL VALIDATION REPORT")
    
    if all_tests_passed:
        print("🎉 ALL REQUIRED TESTS PASSED!")
        print("\n✅ Environment is ready for HD Ring Attractor experiments")
        print("✅ B200 GPU compatibility configured (CPU fallback enabled)")
        print("✅ All core dependencies installed and working")
        
        print("\n📚 Quick Start Guide:")
        print("1. Launch Jupyter: jupyter lab")
        print("2. Navigate to: hd_ring_attractor/notebooks/")
        print("3. Open: comprehensive_single_peak_analysis.ipynb")
        print("\n🧠 For single-peak solution validation:")
        print("   python comprehensive_validation_test.py")
        
    else:
        print("❌ SOME REQUIRED TESTS FAILED")
        print("\nErrors encountered:")
        for i, error in enumerate(all_errors, 1):
            print(f"  {i}. {error}")
        
        print("\n🔧 Troubleshooting:")
        print("1. Run setup script: bash setup_env.sh")
        print("2. Check Python version: python --version (need 3.8+)")
        print("3. Verify virtual environment is activated")
        print("4. For B200 GPU issues, the code will use CPU automatically")
    
    # Optional packages summary
    optional_installed = sum(optional_status.values())
    print(f"\n📦 Optional packages: {optional_installed}/{len(optional_status)} installed")
    
    print("\n" + "="*60)
    
    return 0 if all_tests_passed else 1


if __name__ == "__main__":
    sys.exit(main())