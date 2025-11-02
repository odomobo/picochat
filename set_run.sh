#!/bin/bash
# Switch to an existing nanochat training run
# Usage: source set_run.sh <run_name>

if [ -z "$1" ]; then
    echo "Error: No run name provided"
    echo "Usage: source set_run.sh <run_name>"
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

# Disable debug mode
set +x

# Check if run directory exists
if [ ! -d "$NANOCHAT_RUN_DIR" ]; then
    echo "Error: Run directory does not exist: $NANOCHAT_RUN_DIR"
    echo "This run has not been initialized yet."
    echo "To create a new run, use: source init_run.sh <run_name>"
    return 1
fi

# Check if config.py exists
if [ ! -f "$NANOCHAT_RUN_DIR/config.py" ]; then
    echo "Warning: config.py not found in $NANOCHAT_RUN_DIR"
    echo "This run may not be properly configured."
fi

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

echo ""
echo "=========================================="
echo "Switched to run: $RUN_NAME"
echo "=========================================="
echo "NANOCHAT_DATA_DIR: $NANOCHAT_DATA_DIR"
echo "NANOCHAT_RUN_DIR: $NANOCHAT_RUN_DIR"
echo ""

# Show config summary if it exists
if [ -f "$NANOCHAT_RUN_DIR/config.py" ]; then
    echo "Configuration summary:"
    grep -E "^(run|depth|model_dim|device_batch_size|total_batch_size|max_seq_len|target_param_data_ratio) = " "$NANOCHAT_RUN_DIR/config.py" 2>/dev/null || echo "  (unable to parse config)"
    echo ""
fi

# Show checkpoint status
if [ -f "$NANOCHAT_RUN_DIR/base.pt" ]; then
    echo "Checkpoints found:"
    ls -lh "$NANOCHAT_RUN_DIR"/*.pt 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
    echo ""
else
    echo "No checkpoints found yet."
    echo ""
fi
