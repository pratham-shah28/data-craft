#!/bin/bash
# Start MLflow UI Server
# Usage: ./start_mlflow_ui.sh

echo "============================================"
echo "Starting MLflow UI Server"
echo "============================================"

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MLFLOW_DIR="$(dirname "$SCRIPT_DIR")"

echo "MLflow directory: $MLFLOW_DIR"
echo "Storage location: $MLFLOW_DIR/mlruns"
echo ""

# Check if mlflow is installed
if ! command -v mlflow &> /dev/null; then
    echo "❌ MLflow is not installed!"
    echo "   Install with: pip install mlflow"
    exit 1
fi

echo "✓ MLflow found"
echo ""

# Start MLflow UI
echo "Starting MLflow UI..."
echo "Access at: http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop"
echo "============================================"
echo ""

cd "$MLFLOW_DIR"
mlflow ui --backend-store-uri "file://$MLFLOW_DIR/mlruns" --port 5000
