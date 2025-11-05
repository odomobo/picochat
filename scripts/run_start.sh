#!/bin/bash
# Execute a nanochat training run
# Usage: bash start_run.sh
#
# Prerequisites:
# - Must have run: source scripts/run_init.sh <run_name>
# - NANOCHAT_RUN_DIR and NANOCHAT_DATA_DIR must be set
# - Configuration must exist at $NANOCHAT_RUN_DIR/config.py

# Verify environment is set up
if [ -z "$NANOCHAT_RUN_DIR" ]; then
    echo "ERROR: NANOCHAT_RUN_DIR is not set"
    echo "Please run: source scripts/run_init.sh <run_name>"
    exit 1
fi

if [ -z "$NANOCHAT_DATA_DIR" ]; then
    echo "ERROR: NANOCHAT_DATA_DIR is not set"
    echo "Please run: source scripts/run_init.sh <run_name>"
    exit 1
fi

# Verify configuration exists
if [ ! -f "$NANOCHAT_RUN_DIR/config.py" ]; then
    echo "ERROR: Configuration file not found: $NANOCHAT_RUN_DIR/config.py"
    echo "Please run: source scripts/run_init.sh <run_name>"
    exit 1
fi

# Train the base model
# Note: config.py is automatically loaded by base_train.py from NANOCHAT_RUN_DIR
python -m scripts.base_train

# Evaluate loss on train/val splits
python -m scripts.base_loss

# Evaluate CORE metric
python -m scripts.base_eval

# Generate final report
python -m nanochat.report generate

echo ""
echo "=" * 80
echo "Training run complete!"
echo "Results saved to: $NANOCHAT_RUN_DIR"
echo "Report: $NANOCHAT_RUN_DIR/report/report.md"
echo "=" * 80
