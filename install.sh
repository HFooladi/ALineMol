#!/bin/bash
# install.sh - ALineMol Installation Script
# Usage: ./install.sh [cpu|cu118|cu121|cu124]

set -e

# Default to CPU
DEVICE="${1:-cpu}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

# PyTorch wheel URLs
case "$DEVICE" in
    cpu)
        PYTORCH_INDEX="https://download.pytorch.org/whl/cpu"
        echo "Installing with CPU-only PyTorch..."
        ;;
    cu118)
        PYTORCH_INDEX="https://download.pytorch.org/whl/cu118"
        echo "Installing with CUDA 11.8 PyTorch..."
        ;;
    cu121)
        PYTORCH_INDEX="https://download.pytorch.org/whl/cu121"
        echo "Installing with CUDA 12.1 PyTorch..."
        ;;
    cu124)
        PYTORCH_INDEX="https://download.pytorch.org/whl/cu124"
        echo "Installing with CUDA 12.4 PyTorch..."
        ;;
    *)
        echo "Usage: $0 [cpu|cu118|cu121|cu124]"
        echo "  cpu   - CPU-only PyTorch (default)"
        echo "  cu118 - CUDA 11.8"
        echo "  cu121 - CUDA 12.1"
        echo "  cu124 - CUDA 12.4"
        exit 1
        ;;
esac

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Create virtual environment
echo "Creating virtual environment with Python $PYTHON_VERSION..."
uv venv --python "$PYTHON_VERSION"

# Activate venv
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
uv pip install -e ".[dev,test]" \
    -f "$PYTORCH_INDEX" \
    -f https://data.dgl.ai/wheels/repo.html

# Install compatible torchdata version for DGL
echo "Installing compatible torchdata version..."
uv pip install "torchdata<0.8" -f "$PYTORCH_INDEX"

echo ""
echo "Installation complete!"
echo "Activate the environment with: source .venv/bin/activate"
