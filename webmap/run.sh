#!/bin/bash
# Launcher script for interactive webmap
# Usage: bash webmap/run.sh (from project root)

set -e

# Get project root (parent of webmap/)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "WEBMAP LAUNCHER"
echo "=========================================="
echo "Project root: $PROJECT_ROOT"
echo ""

# 1. Run QA check
echo "[1] Running quality assurance check..."
micromamba run -n geo python scripts/qa_webmap.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ QA check failed. Fix issues and try again."
    exit 1
fi

# 2. Launch Streamlit
echo ""
echo "[2] Launching Streamlit app..."
echo "    → http://localhost:8501"
echo ""
micromamba run -n geo streamlit run webmap/app.py
