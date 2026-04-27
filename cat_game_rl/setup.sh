#!/usr/bin/env bash
set -euo pipefail

echo "=== Cat Game Crafting RL - Setup ==="

# Install uv if not present
if ! command -v uv &>/dev/null; then
  echo "Installing uv..."
  if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  else
    curl -LsSf https://astral.sh/uv/install.sh | sh
  fi
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

echo "uv version: $(uv --version)"

# Create venv and install all deps (including dev)
echo "Creating virtual environment and installing dependencies..."
uv sync --extra dev

echo ""
echo "Setup complete. Activate the environment with:"
echo "  source .venv/bin/activate    # Linux/macOS"
echo "  .venv\\Scripts\\activate       # Windows"
echo ""
echo "Or run commands directly via:"
echo "  uv run python scripts/train.py"
echo "  uv run pytest"
