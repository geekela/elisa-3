#!/bin/bash

# ==============================================================================
# VS Code Extensions Installer for Elisa-3 HD Ring Attractor Network
# Comprehensive setup for Python, Jupyter, and scientific computing
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

# Check if VS Code is available
check_vscode() {
    if command -v code &> /dev/null; then
        print_success "VS Code 'code' command found"
        return 0
    elif command -v code-insiders &> /dev/null; then
        print_warning "VS Code Insiders found. Using 'code-insiders' command"
        alias code='code-insiders'
        return 0
    else
        print_error "VS Code 'code' command not found in PATH"
        echo ""
        echo "Please ensure VS Code is installed and the 'code' command is available:"
        echo "1. Install VS Code from: https://code.visualstudio.com/"
        echo "2. Enable command line: View > Command Palette > 'Shell Command: Install code command in PATH'"
        echo ""
        echo "For VS Code Insiders: https://code.visualstudio.com/insiders/"
        return 1
    fi
}

# Install extension with error handling
install_extension() {
    local extension=$1
    local description=$2
    
    if code --list-extensions 2>/dev/null | grep -q "^${extension}$"; then
        echo "  ✓ ${extension} (already installed)"
    else
        if code --install-extension "${extension}" --force >/dev/null 2>&1; then
            print_success "${extension} - ${description}"
        else
            print_warning "Failed to install ${extension} - ${description}"
        fi
    fi
}

# Main installation function
main() {
    echo "===================================================================="
    echo "🧩 VS Code Extensions Installer for Elisa-3 HD Ring Attractor"
    echo "===================================================================="
    echo ""
    
    # Check VS Code availability
    if ! check_vscode; then
        exit 1
    fi
    
    # Get VS Code version
    print_status "VS Code version:"
    code --version | head -n1 || true
    echo ""
    
    # Core Python Development Extensions
    print_status "Installing Core Python Development Extensions..."
    install_extension "ms-python.python" "Python language support"
    install_extension "ms-python.vscode-pylance" "Pylance language server"
    install_extension "ms-python.pylint" "Pylint linting"
    install_extension "ms-python.flake8" "Flake8 style checking"
    install_extension "ms-python.black-formatter" "Black code formatter"
    install_extension "ms-python.isort" "Import sorting"
    install_extension "ms-python.mypy-type-checker" "MyPy type checking"
    
    # Jupyter Notebook Support
    print_status "Installing Jupyter Notebook Extensions..."
    install_extension "ms-toolsai.jupyter" "Jupyter notebook support"
    install_extension "ms-toolsai.jupyter-keymap" "Jupyter keybindings"
    install_extension "ms-toolsai.jupyter-renderers" "Jupyter output renderers"
    install_extension "ms-toolsai.vscode-jupyter-cell-tags" "Jupyter cell tags"
    install_extension "ms-toolsai.vscode-jupyter-slideshow" "Jupyter slideshow"
    
    # Scientific Computing and Data Science
    print_status "Installing Scientific Computing Extensions..."
    install_extension "njpwerner.autodocstring" "Automatic docstring generation"
    install_extension "tushortz.python-extended-snippets" "Python snippets"
    install_extension "kevinrose.vsc-python-indent" "Python indentation"
    install_extension "littlefoxteam.vscode-python-test-adapter" "Python test explorer"
    
    # AI/ML Development Tools
    print_status "Installing AI/ML Development Extensions..."
    install_extension "ms-toolsai.tensorboard" "TensorBoard integration"
    install_extension "GitHub.copilot" "GitHub Copilot (if licensed)"
    install_extension "continue.continue" "Continue AI assistant"
    install_extension "TabNine.tabnine-vscode" "TabNine AI autocomplete"
    
    # Git and Version Control
    print_status "Installing Git Extensions..."
    install_extension "eamodio.gitlens" "GitLens - Git supercharged"
    install_extension "mhutchie.git-graph" "Git Graph visualization"
    install_extension "donjayamanne.githistory" "Git History"
    
    # Documentation and Markdown
    print_status "Installing Documentation Extensions..."
    install_extension "yzhang.markdown-all-in-one" "Markdown All in One"
    install_extension "shd101wyy.markdown-preview-enhanced" "Markdown Preview Enhanced"
    install_extension "DavidAnson.vscode-markdownlint" "Markdown linting"
    install_extension "bierner.markdown-mermaid" "Mermaid diagram support"
    
    # Code Quality and Productivity
    print_status "Installing Code Quality Extensions..."
    install_extension "streetsidesoftware.code-spell-checker" "Code spell checker"
    install_extension "usernamehw.errorlens" "Error Lens - inline errors"
    install_extension "SonarSource.sonarlint-vscode" "SonarLint code quality"
    install_extension "mechatroner.rainbow-csv" "Rainbow CSV"
    install_extension "janisdd.vscode-edit-csv" "Edit CSV files"
    
    # File and Project Management
    print_status "Installing File Management Extensions..."
    install_extension "alefragnani.project-manager" "Project Manager"
    install_extension "sleistner.vscode-fileutils" "File Utils"
    install_extension "mkxml.vscode-filesize" "File size display"
    install_extension "alexcvzz.vscode-sqlite" "SQLite viewer"
    
    # Remote Development (useful for GPU servers)
    print_status "Installing Remote Development Extensions..."
    install_extension "ms-vscode-remote.remote-ssh" "Remote - SSH"
    install_extension "ms-vscode-remote.remote-containers" "Remote - Containers"
    install_extension "ms-vscode-remote.remote-wsl" "Remote - WSL"
    install_extension "ms-vscode-remote.vscode-remote-extensionpack" "Remote Development Pack"
    
    # UI and Theme Enhancements
    print_status "Installing UI Enhancement Extensions..."
    install_extension "PKief.material-icon-theme" "Material Icon Theme"
    install_extension "zhuangtongfa.material-theme" "One Dark Pro theme"
    install_extension "oderwat.indent-rainbow" "Indent Rainbow"
    install_extension "CoenraadS.bracket-pair-colorizer-2" "Bracket Pair Colorizer"
    install_extension "vscode-icons-team.vscode-icons" "VSCode Icons"
    
    # Utilities
    print_status "Installing Utility Extensions..."
    install_extension "christian-kohler.path-intellisense" "Path Intellisense"
    install_extension "formulahendry.code-runner" "Code Runner"
    install_extension "humao.rest-client" "REST Client"
    install_extension "ms-vscode.hexeditor" "Hex Editor"
    install_extension "redhat.vscode-yaml" "YAML support"
    install_extension "ms-azuretools.vscode-docker" "Docker support"
    
    # Data Visualization
    print_status "Installing Data Visualization Extensions..."
    install_extension "RandomFractalsInc.vscode-data-preview" "Data Preview"
    install_extension "GrapeCity.gc-excelviewer" "Excel Viewer"
    install_extension "tomoki1207.pdf" "PDF viewer"
    
    # Create VS Code settings for the project
    print_status "Creating VS Code workspace settings..."
    mkdir -p .vscode
    
    cat > .vscode/settings.json << 'EOF'
{
    // Python Configuration
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length", "100"],
    "python.sortImports.args": ["--profile", "black"],
    "editor.formatOnSave": true,
    "python.analysis.typeCheckingMode": "basic",
    
    // Jupyter Configuration
    "jupyter.askForKernelRestart": false,
    "jupyter.interactiveWindow.textEditor.autoMoveToNextCell": true,
    "jupyter.sendSelectionToInteractiveWindow": true,
    "notebook.cellToolbarLocation": {
        "default": "right",
        "jupyter-notebook": "right"
    },
    
    // Editor Configuration
    "editor.rulers": [88, 100],
    "editor.wordWrap": "on",
    "editor.minimap.enabled": true,
    "editor.suggestSelection": "first",
    "editor.snippetSuggestions": "top",
    
    // File Configuration
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        "**/.pytest_cache": true,
        "**/.mypy_cache": true,
        "**/.ipynb_checkpoints": true
    },
    
    // Git Configuration
    "git.autofetch": true,
    "git.confirmSync": false,
    
    // Terminal Configuration
    "terminal.integrated.enableMultiLinePasteWarning": false,
    
    // Spell Checker Configuration
    "cSpell.words": [
        "elisa",
        "numpy",
        "scipy",
        "matplotlib",
        "seaborn",
        "pytorch",
        "torch",
        "cuda",
        "cudnn",
        "attractor",
        "ipynb",
        "jupyter",
        "nbformat",
        "nbconvert"
    ]
}
EOF
    
    print_success "VS Code workspace settings created in .vscode/settings.json"
    
    # Create recommended extensions file
    cat > .vscode/extensions.json << 'EOF'
{
    "recommendations": [
        "ms-python.python",
        "ms-python.vscode-pylance",
        "ms-toolsai.jupyter",
        "ms-toolsai.tensorboard",
        "eamodio.gitlens",
        "njpwerner.autodocstring",
        "streetsidesoftware.code-spell-checker",
        "PKief.material-icon-theme"
    ]
}
EOF
    
    print_success "Recommended extensions list created in .vscode/extensions.json"
    
    # Summary
    echo ""
    echo "===================================================================="
    print_success "✅ VS CODE EXTENSIONS INSTALLATION COMPLETE!"
    echo "===================================================================="
    echo ""
    echo "📊 Installed extension categories:"
    echo "  ✓ Python development (language server, linting, formatting)"
    echo "  ✓ Jupyter notebook support with enhanced features"
    echo "  ✓ Scientific computing and data science tools"
    echo "  ✓ AI/ML development assistants"
    echo "  ✓ Git version control with GitLens"
    echo "  ✓ Documentation and Markdown tools"
    echo "  ✓ Code quality and productivity enhancers"
    echo "  ✓ Remote development capabilities"
    echo "  ✓ UI enhancements and themes"
    echo "  ✓ Data visualization tools"
    echo ""
    echo "🔧 Project-specific VS Code settings created in:"
    echo "  • .vscode/settings.json - Workspace settings"
    echo "  • .vscode/extensions.json - Recommended extensions"
    echo ""
    echo "📚 Next steps:"
    echo "1. Restart VS Code to activate all extensions"
    echo "2. Select Python interpreter: Ctrl/Cmd+Shift+P > 'Python: Select Interpreter'"
    echo "3. Open a notebook: hd_ring_attractor/notebooks/*.ipynb"
    echo "4. Run cells with Shift+Enter"
    echo ""
    echo "💡 Tips:"
    echo "  • Use Ctrl/Cmd+Shift+P to open Command Palette"
    echo "  • Press F1 for quick access to commands"
    echo "  • Check View > Extensions to manage installed extensions"
    echo ""
}

# Run main function
main "$@"