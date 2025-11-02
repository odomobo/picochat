#!/bin/bash
# Initialize a nanochat training run
# Usage: source init_run.sh <run_name>

if [ -z "$1" ]; then
    echo "Error: No run name provided"
    echo "Usage: source init_run.sh <run_name>"
    return 1
fi

# Enable debug mode to show each command as it executes
set -x

RUN_NAME="$1"

# Set directory paths
export NANOCHAT_DATA_DIR="$PWD/working/data"
export NANOCHAT_RUN_DIR="$PWD/working/runs/${RUN_NAME}"
export WANDB_RUN="${RUN_NAME}"
export OMP_NUM_THREADS=1

# Create directories
mkdir -p "$NANOCHAT_DATA_DIR"
mkdir -p "$NANOCHAT_RUN_DIR"

# Disable debug mode for venv operations (too noisy)
set +x

# Deactivate current venv if active, then activate the correct one
if [ -n "$VIRTUAL_ENV" ]; then
    echo "Deactivating current virtual environment..."
    deactivate
fi
if [ -f ".venv/bin/activate" ]; then
    echo "Activating .venv virtual environment..."
    source .venv/bin/activate
else
    echo "Warning: .venv/bin/activate not found, skipping venv activation"
fi

# Re-enable debug mode
set -x

# Initialize the report (will error if already exists)
python -m nanochat.report init

# Disable debug mode
set +x
