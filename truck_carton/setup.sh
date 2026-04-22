#!/usr/bin/env bash
# Set up the truck-carton development environment using UV.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_VERSION="3.10"

echo "=== truck-carton environment setup ==="
echo ""

# --- Check UV ---
if ! command -v uv &>/dev/null; then
  echo "UV not found. Installing..."
  if command -v pip &>/dev/null; then
    pip install uv
  elif command -v pipx &>/dev/null; then
    pipx install uv
  else
    echo "ERROR: Neither pip nor pipx available."
    echo "Install UV manually: https://docs.astral.sh/uv/"
    exit 1
  fi
fi
echo "UV version: $(uv --version)"

# --- Check Python version ---
echo ""
echo "Checking Python ${PYTHON_VERSION}..."
if ! uv python list 2>/dev/null | grep -q "${PYTHON_VERSION}"; then
  echo "Installing Python ${PYTHON_VERSION} via UV..."
  uv python install "${PYTHON_VERSION}" || true
fi

# --- Create/sync virtual environment ---
echo ""
echo "Syncing project dependencies..."
uv sync --dev

echo ""
echo "=== Verifying installation ==="

# Core imports
uv run python -c "
import gymnasium; print(f'  gymnasium {gymnasium.__version__}')
import stable_baselines3; print(f'  stable-baselines3 {stable_baselines3.__version__}')
import sb3_contrib; print(f'  sb3-contrib {sb3_contrib.__version__}')
import torch; print(f'  torch {torch.__version__}')
import numpy; print(f'  numpy {numpy.__version__}')
import networkx; print(f'  networkx {networkx.__version__}')
import matplotlib; print(f'  matplotlib {matplotlib.__version__}')
import PIL; print(f'  Pillow {PIL.__version__}')
"

# Optional dashboard dependencies
echo ""
echo "Checking optional dashboard dependencies..."
uv run python -c "
try:
  import streamlit; print(f'  streamlit {streamlit.__version__}')
except ImportError:
  print('  streamlit NOT installed (run: uv add streamlit plotly)')
try:
  import plotly; print(f'  plotly {plotly.__version__}')
except ImportError:
  print('  plotly NOT installed')
"

# Package import
echo ""
echo "Checking truck_carton package..."
uv run python -c "
from truck_carton.config import AppConfig
c = AppConfig()
print(f'  AppConfig loaded: {len(c.curriculum.stages)} curriculum stages')
print(f'  Action space: Discrete({c.env.max_candidates + c.env.max_routing_actions})')
print('  Package OK')
"

# Run tests
echo ""
echo "=== Running test suite ==="
uv run python -m pytest tests/ -v --tb=short

echo ""
echo "=== Setup complete ==="
echo "Activate with: source .venv/bin/activate (or use 'uv run')"
