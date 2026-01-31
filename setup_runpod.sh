#!/bin/bash
# Setup script for RunPod environment
# Run this after git clone on RunPod

set -e  # Exit on error

echo "=========================================="
echo "Setting up environment on RunPod"
echo "=========================================="

# 1. Install uv if not already installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
else
    echo "✓ uv already installed"
fi

# 2. Sync dependencies
echo ""
echo "Installing dependencies..."
uv sync

# 3. Download model
echo ""
echo "Downloading Qwen2.5-Math-1.5B model..."
uv run python download_model.py

# 4. Verify setup
echo ""
echo "Verifying installation..."
uv run python -c "from vllm import LLM; from cs336_alignment.drgrpo_grader import r1_zero_reward_fn; print('✓ All imports successful')"

echo ""
echo "=========================================="
echo "✓ Setup complete!"
echo "=========================================="
echo ""
echo "You can now run:"
echo "  uv run python math_baseline.py --max-samples 10"
echo ""
