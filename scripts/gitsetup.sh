#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Installing Node Version Manager (nvm)...${NC}"

# Download and install nvm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash

# Load nvm into current shell
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"

echo -e "${GREEN}Installing latest Node.js (which includes npm)...${NC}"

# Install latest Node.js (npm is included automatically)
nvm install node  # 'node' is an alias for the latest version
nvm use node
nvm alias default node  # Set as default for new shells

# Verify both Node.js and npm are installed
echo -e "${YELLOW}Verifying installation...${NC}"
echo -e "${GREEN}Node.js version:${NC} $(node --version)"
echo -e "${GREEN}npm version:${NC} $(npm --version)"

# Update npm to latest version (optional but recommended)
echo -e "${GREEN}Updating npm to latest version...${NC}"
npm install -g npm@latest

# Install Claude Code globally
echo -e "${GREEN}Installing @anthropic-ai/claude-code globally...${NC}"
npm install -g @anthropic-ai/claude-code

# Check if installation was successful
if command -v claude &> /dev/null; then
    echo -e "${GREEN}✓ Claude Code installed successfully!${NC}"
    echo -e "${GREEN}You can now use 'claude' command from your terminal.${NC}"
else
    echo -e "${RED}Claude Code installed but 'claude' command not immediately available${NC}"
    echo -e "${YELLOW}Try one of these:${NC}"
    echo -e "  1. Restart your terminal"
    echo -e "  2. Run: source ~/.bashrc"
    echo -e "  3. Run: export PATH=\"\$PATH:\$HOME/.nvm/versions/node/\$(node --version)/bin\""
fi

echo -e "${GREEN}✓ Setup complete!${NC}"
echo -e "${YELLOW}Note: For new terminal sessions, nvm will be automatically loaded${NC}"
git config --global user.name "Elisa"
git config --global user.email "eladerrico@gmail.com"
apt update
apt install gh
gh auth login