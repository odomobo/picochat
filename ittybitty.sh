#!/bin/bash

# ittybitty.sh - Tiny experimental model for testing
# Target: ~38M parameters (untied weights), 500M token pretraining only
#
# Planned architecture (from CHECKPOINT.md):
# vocab size   : 24576 (3 * 2^13)
# depth        : 4
# dim          : 512
# head dim     : 64
# heads        : 8
#
# Parameter breakdown:
# wte:         12,582,912 (33.33%)
# transformer: 12,582,912 (33.33%)
# lm_head:     12,582,912 (33.33%)
# Total:       37,748,736 params
#
# TODO: Current architecture derivation in base_train.py doesn't support our target:
#   - model_dim is calculated as depth * 64 (line 89), so depth=4 gives 256, not 512
#   - num_heads is calculated from model_dim with head_dim=128 (line 90), not 64
#   - We need to modify base_train.py to either:
#     a) Add --model_dim and --head_dim as configurable parameters, OR
#     b) Change the aspect ratio formula from 64 to 128 (depth * 128), OR
#     c) Change head_dim from 128 to 64 and adjust num_heads calculation
#   For now, this script will run with depth=4, which gives:
#     - model_dim = 4 * 64 = 256
#     - num_heads = max(1, (256 + 127) // 128) = 2
#   This is NOT our target architecture!

# we don't want this script to run on its own... for now!
exit

# Default intermediate artifacts directory
export OMP_NUM_THREADS=1
export NANOCHAT_RUN_DIR="$HOME/.cache/nanochat"
export NANOCHAT_DATA_DIR="$HOME/.cache/nanochat/data"
mkdir -p $NANOCHAT_RUN_DIR
mkdir -p $NANOCHAT_DATA_DIR

# -----------------------------------------------------------------------------
# Python venv setup with uv

# install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# install the repo dependencies
uv sync --extra gpu
# activate venv
source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb setup (optional)
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# Initialize report
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer

# Install Rust / Cargo
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Build the rustbpe Tokenizer
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# Download initial data for tokenizer training (~2B chars = 8 shards)
python -m nanochat.dataset -n 8

# Start downloading data for pretraining in background
# For 500M tokens at ~4.8 chars/token = ~2.4B chars
# At 250M chars/shard, we need ~10 shards
# Download 15 for safety
python -m nanochat.dataset -n 15 &
DATASET_DOWNLOAD_PID=$!

# Train tokenizer with custom vocab size (24576 = 3 * 2^13)
# Using ~2B characters from the initial 8 shards
python -m scripts.tok_train --vocab_size=24576 --max_chars=2000000000

# Evaluate the tokenizer
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model (pretraining only)

# Download eval bundle for CORE metric evaluation
EVAL_BUNDLE_URL=https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip
if [ ! -d "$NANOCHAT_DATA_DIR/eval_bundle" ]; then
    curl -L -o eval_bundle.zip $EVAL_BUNDLE_URL
    unzip -q eval_bundle.zip
    rm eval_bundle.zip
    mv eval_bundle $NANOCHAT_DATA_DIR
fi

# Wait for dataset download to complete
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

# Calculate training iterations for exactly 500M tokens
# total_batch_size defaults to 524,288 tokens
# num_iterations = 500,000,000 / 524,288 ≈ 954 steps
# We disable target_param_data_ratio (Chinchilla) by setting it to -1
# and explicitly set num_iterations instead

echo "Training configuration:"
echo "  Target tokens: 500M"
echo "  Total batch size: 524,288 tokens"
echo "  Device batch size: 48 (optimized for RTX 3090 24GB)"
echo "  Gradient accumulation steps: ~11 (524,288 / (48 × 2048))"
echo "  Iterations: ~954 steps"
echo "  Expected VRAM: ~20.8 GB"
echo "  Expected model: depth=4, model_dim=256, num_heads=2"
echo "  WARNING: This is NOT the target architecture (need dim=512, heads=8)"

# Pretrain the model
# Using depth=4 which gives model_dim=256, num_heads=2 (NOT our target 512/8!)
# Single GPU (RTX 3090 24GB): using batch_size=48 (20.8GB VRAM, 86.7% utilization)
python -m scripts.base_train \
  --depth=4 \
  --num_iterations=954 \
  --target_param_data_ratio=-1 \
  --device_batch_size=48 \
  --run=$WANDB_RUN

# Evaluate the model on train/val data
python -m scripts.base_loss

# Evaluate on CORE tasks
python -m scripts.base_eval

# -----------------------------------------------------------------------------
# Generate report
python -m nanochat.report generate

echo ""
echo "Pretraining complete!"
echo "Training summary:"
echo "  - Device batch size: 48 (RTX 3090 optimized)"
echo "  - VRAM usage: ~20.8 GB / 24 GB"
echo "  - Model architecture: depth=4, dim=256, heads=2"
echo "  - NOTE: This differs from target architecture (dim=512, heads=8)"
echo "  - See TODO comments at top of script for required changes."
