#!/usr/bin/env bash
# setup.sh — one-shot environment setup for XFGU
set -euo pipefail

PYTHON="${PYTHON:-python3}"

echo "=== XFGU Setup ==="
echo "Creating virtual environment..."
"$PYTHON" -m venv .venv

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "=== Setup complete ==="
echo "Activate the environment with: source .venv/bin/activate"
echo "Run the full experiment with:  python scripts/run_full_experiment.py configs/default.yaml"
