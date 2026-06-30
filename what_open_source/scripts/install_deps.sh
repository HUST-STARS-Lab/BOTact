#!/usr/bin/env bash
# Install CUDA extensions for Visuo-Tactile Depth Fusion Framework
# Follows PointMamba (https://github.com/LMD0311/PointMamba) setup flow.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "[1/3] Installing Python requirements..."
pip install -r requirements.txt

echo "[2/3] Installing causal-conv1d (required by mamba-ssm)..."
pip install "causal-conv1d>=1.2.0" ${PIP_EXTRA_ARGS:-}

echo "[3/3] Installing mamba-ssm from bundled source..."
pip install -e mamba/ ${PIP_EXTRA_ARGS:-} || pip install -e mamba/ --no-build-isolation

echo ""
echo "Done. Verify with:"
echo "  python -c \"from mamba_ssm.modules.mamba_simple import Mamba; print('mamba OK')\""
echo ""
echo "Also install PointNet++ ops and KNN_CUDA — see README.md § Installation."
