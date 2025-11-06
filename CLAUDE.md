# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

nanochat is a minimal, hackable codebase for training small-scale language models from scratch. This fork focuses on rapid experimentation with pretraining at the <100M parameter scale to understand scaling laws, architecture choices, and optimization dynamics.

**Philosophy**: nanochat is not an exhaustively configurable LLM "framework" - it's a cohesive, minimal, readable, hackable, maximally-forkable "strong baseline" codebase. Avoid adding giant configuration objects, model factories, or complex if-then-else logic.

**Current Focus**: The codebase currently focuses on base model pretraining only. Midtraining, supervised fine-tuning (SFT), and reinforcement learning (RL) stages are legacy components not used in the current experimental workflow.

## Ticket Workflow

When processing tickets, follow these steps **in order**:

1. **Present understanding and implementation plan FIRST**
   - In your very first response, explain what you think the ticket is asking for
   - Present a complete implementation plan with all technical details
   - Do NOT ask questions yet, do NOT implement yet
   - If you have uncertainties, note them in the plan but still present a complete picture

2. **Ask questions based on uncertainty** (in the same first response)
   - Only ask questions when genuinely uncertain about requirements
   - If you have confident understanding, skip questions entirely
   - Questions should be specific and actionable

3. **Present revised plan after questions are answered**
   - When user answers your questions, present an updated implementation plan
   - Incorporate all the answers and clarifications into a concrete, detailed plan
   - Show exactly what you will implement based on their feedback
   - This revised plan is REQUIRED before proceeding - do not skip it

4. **Get explicit authorization before implementing**
   - Wait for explicit approval like "yes", "proceed", "go ahead", "implement this"
   - Answering your questions is NOT authorization
   - Providing clarifications is NOT authorization
   - You must see clear permission to begin implementation after presenting the revised plan

**CRITICAL**: Never write code, create files, or make changes until you receive explicit authorization. Clarifying questions and discussing the plan does not constitute permission to implement.

**Note**: The "status" field in tickets is irrelevant to processing. This workflow applies to all types of tickets (features, bugs, refactoring, research, etc.).

## Build & Development Commands

### Environment Setup
```bash
# Install uv package manager (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
uv sync --extra gpu  # For GPU (CUDA 12.8)
# OR
uv sync --extra cpu  # For CPU only

# Activate virtual environment
source .venv/bin/activate

# Install Rust tokenizer (requires Rust/Cargo)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
```

### Running Tests
```bash
# Run tokenizer tests
python -m pytest tests/test_rustbpe.py -v -s

# Run specific test markers
python -m pytest -m "not slow"  # Exclude slow tests
```

### Training Commands

**Note**: The `speedrun.sh` script is from the original nanochat and has been removed. For current experiments, use the run initialization workflow described in the "Experimental Workflow" section below.

**Distributed training (multi-GPU)**:
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=20 --device_batch_size=32
```

**Single GPU training** (omit torchrun, automatically uses gradient accumulation):
```bash
python -m scripts.base_train -- --depth=20 --device_batch_size=32
```

**CPU/MPS training** (much smaller models, see dev/runcpu.sh for example hyperparameters):
```bash
python -m scripts.base_train -- --depth=4 --max_seq_len=512 --device_batch_size=1 --eval_tokens=512 --core_metric_every=-1 --total_batch_size=512 --num_iterations=20
```

### Inference & Chat

**Note**: Chat interfaces (chat_cli.py, chat_web.py) from the original nanochat have been removed. This fork focuses on pretraining base models, not chat applications.

### Individual Pipeline Stages
```bash
# 1. Tokenizer
python -m scripts.tok_train --max_chars=2000000000
python -m scripts.tok_eval

# 2. Base model (pretraining)
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=20
torchrun --standalone --nproc_per_node=8 -m scripts.base_loss
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval
```

**Note**: This codebase focuses on base model pretraining only. The original nanochat included midtraining, SFT, and RL stages, but those have been removed from this fork.

### Data Management
```bash
# Download pretraining data shards (each ~250M chars, ~100MB compressed)
python -m nanochat.dataset -n 240  # Downloads 240 shards for d20 model
```

## Architecture

### Training Pipeline Stages

**Current workflow (pretraining only):**
1. **Tokenizer Training** (rustbpe): Custom Rust BPE tokenizer
   - Default vocab size: 65536 (2^16) for full-scale models
   - Tiny model vocab size: 24576 (3 × 2^13) - appropriate for experimental models <100M parameters
2. **Base Model Pretraining**: GPT-style autoregressive language modeling on web text

The original nanochat included additional stages (Midtraining, SFT, RL) for training chat models, but these have been removed from this fork as they're not relevant for small-scale pretraining experiments.

### Model Architecture (nanochat/gpt.py)

The GPT implementation uses a modern transformer with:
- **Rotary Positional Embeddings (RoPE)**: No learned positional embeddings
- **QK Normalization**: Normalizes queries and keys in attention
- **Weight Tying (Optional)**: Configurable via `tie_weights` parameter
  - `tie_weights=False` (default): Separate wte and lm_head (original behavior)
  - `tie_weights=True`: Tie wte and lm_head to share weights (reduces params by ~50%)
- **ReLU^2 Activation**: In MLP layers instead of GELU
- **RMSNorm**: Functional normalization without learnable parameters, applied after token embedding and after each block
- **No Bias**: Linear layers have no bias terms
- **Multi-Query Attention (MQA) / Grouped Query Attention (GQA)**: Configurable n_kv_head for efficient inference
- **Logits Softcapping**: 15.0 * tanh(logits / 15.0)

Model size is determined by `--depth` parameter:
- `depth=20` → 561M params (d20, ~$100 speedrun) - with vocab_size=65536
- `depth=26` → ~800M params (d26, ~$300, GPT-2 grade) - with vocab_size=65536
- `depth=32` → 1.9B params (d32, ~$800) - with vocab_size=65536
- `model_dim = depth * 64` (aspect ratio of 64, now configurable)
- `num_heads = max(1, (model_dim + 127) // 128)` (head dim of 128)

For tiny experimental models (<100M params), use vocab_size=24576 (3 × 2^13) to reduce embedding overhead.

### Dual Optimizer Setup (nanochat/gpt.py:222-270)

**Critical architectural pattern**: The model uses two separate optimizers:
1. **AdamW** for embeddings and unembedding (wte, lm_head)
   - Learning rates scaled by `(model_dim / 768)^-0.5`
   - **When tie_weights=False** (untied, default):
     - embedding_lr=0.2 for wte
     - unembedding_lr=0.004 for lm_head
   - **When tie_weights=True** (tied):
     - tied_weights_lr=0.2 for shared wte/lm_head weights
     - Only one parameter group (wte only, lm_head doesn't exist)
2. **Muon** for all transformer matrix parameters (attention, MLP)
   - Default LR: matrix_lr=0.02

Both have distributed variants (DistAdamW, DistMuon) for multi-GPU training.

### Configuration System (nanochat/configurator.py)

**"Poor Man's Configurator"** - a unique approach that directly modifies script globals():
- Config files are Python scripts executed with `exec()`
- CLI arguments override via `--key=value` syntax
- No config objects or prefixes needed
- Example: `python script.py --depth=26 --device_batch_size=16`

### Data Loading (nanochat/dataloader.py)

**Tokenizing Distributed Data Loader**:
- Streams text from parquet files on-the-fly
- Tokenizes in batches during iteration
- Each process in DDP reads different shards (stride by world_size)
- Uses deque buffer to accumulate tokens
- Prepends BOS token to each document
- Supports multi-threaded tokenization

### Checkpoint Management (nanochat/checkpoint_manager.py)

Models are saved to `$NANOCHAT_RUN_DIR/`:
- `base.pt`: Pretrained base model

(The original nanochat also saved `mid.pt`, `sft.pt`, `rl.pt` checkpoints, but those pipeline stages have been removed from this fork.)

**Backward Compatibility Pattern**: When adding new features to the model architecture, follow this pattern to ensure old checkpoints continue to work:

1. **Add to GPTConfig dataclass with backward-compatible default:**
   ```python
   @dataclass
   class GPTConfig:
       # ...existing fields...
       new_feature: bool = False  # False = old behavior (original)
   ```

2. **Add fallback logic in checkpoint_manager.py:build_model():**
   ```python
   if "new_feature" not in model_config_kwargs:
       model_config_kwargs["new_feature"] = False  # fallback to old behavior
       log0("Warning: new_feature not found in checkpoint, assuming False (old behavior)")
   ```

3. **The fallback value must preserve original behavior:**
   - Old checkpoints → work exactly as before (no breaking changes)
   - New checkpoints → can use the new feature
   - Example: `tie_weights` defaults to `False` (untied, original behavior)

This pattern enables continuous feature additions while maintaining full backward compatibility with all previously trained models.

### Inference Engine (nanochat/engine.py)

**KVCache**: Efficient autoregressive generation with key-value caching
- Maintains cache across layers: shape `(num_layers, 2, batch_size, num_heads, seq_len, head_dim)`
- Position tracking with automatic advancement
- Supports prefill from another cache

**Calculator Tool**: Built-in tool for safe Python expression evaluation
- Math expressions: `eval_with_timeout()` with 3-second timeout
- String operations: `.count()` method supported for tasks like "count r in strawberry"
- Sandbox restrictions prevent dangerous operations

### Evaluation Tasks (tasks/)

Task framework with `TaskMixture` and `TaskSequence` abstractions:
- **CORE**: Comprehensive evaluation from DCLM paper (base model quality) - **primary benchmark for current workflow**
- **ARC**: AI2 Reasoning Challenge (science questions)
- **GSM8K**: Grade School Math (8K problems)
- **HumanEval**: Python coding task
- **MMLU**: Massive Multitask Language Understanding

Other tasks like SmolTalk, SpellingBee, CustomJSON exist in the tasks/ directory (inherited from original nanochat) but are designed for chat/instruct models and not actively used in the pretraining-only workflow.

### Report Generation (nanochat/report.py)

Generates `report.md` with:
- System info and timestamps
- Codebase statistics (lines, files, dependencies)
- Evaluation metrics across training stages
- Total wall clock time

## Memory Management & Scaling

### Handling OOM Errors

If you run out of VRAM:
1. Reduce `--device_batch_size` (32 → 16 → 8 → 4 → 2 → 1)
   - Code automatically increases gradient accumulation to maintain total_batch_size
   - This trades parallel compute for sequential compute
2. Reduce model size with `--depth` (20 → 16 → 12 → 8 → 4)
3. Reduce `--max_seq_len` (2048 → 1024 → 512)

### Computing Data Requirements

For Chinchilla-optimal training (20 tokens per parameter):
```python
# Example for d20 (561M params)
num_tokens = 561e6 * 20 = 11.2B tokens
# Assuming 4.8 chars/token compression
num_chars = 11.2B * 4.8 ≈ 54B chars
# At 250M chars/shard
num_shards = 54B / 250M ≈ 216 shards (round to 240)
```

This is why speedrun.sh downloads 240 shards (~24GB).

## Training Larger Models

For experimental purposes with small models (<100M params), the default configuration uses `depth=4` and `vocab_size=24576`. To scale up:

```bash
# For larger pretraining runs, download more data shards
python -m nanochat.dataset -n 450 &

# Increase depth (model size) and adjust batch size for memory
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=26 --device_batch_size=16
```

**Note**: The original nanochat supported training larger chat models (d20/d26/d32) with a full chat model pipeline. This fork has removed those components and focuses on small-scale pretraining experiments.

## Computing Environment Notes

- **Tested platforms**: 8XH100, 8XA100 (Ampere is slower but works)
- **Single GPU**: Works fine, 8x slower due to gradient accumulation
- **GPU VRAM**: <80GB requires tuning device_batch_size
- **CPU/MPS**: Supported as of Oct 21, 2025 (see dev/runcpu.sh for small model configs)
- **Auto-detection**: Code auto-detects CUDA > MPS > CPU if device_type not specified

## Rust Tokenizer (rustbpe/)

Custom BPE tokenizer written in Rust for performance:
- Built with PyO3 bindings to Python
- Trained on ~2B characters
- Vocab size: 2^16 = 65536
- Compiled with `maturin develop --release`
- See rustbpe/README.md for rationale

## WandB Integration

Optional but recommended:
```bash
# Login once
wandb login

# Use with runs
WANDB_RUN=my_run_name bash speedrun.sh
```

Default is `WANDB_RUN=dummy` which disables logging.

## Contributing Guidelines

From README:
- Maintain cognitive simplicity - no giant config objects, model factories, or if-else monsters
- Single cohesive baseline, not an exhaustive framework
- **LLM disclosure policy**: Declare any substantial LLM contributions in PRs that you don't fully understand

## File Organization

```
nanochat/           # Core library modules
  gpt.py            # Transformer architecture
  dataloader.py     # Streaming tokenized data loader
  engine.py         # Inference engine with KV cache
  tokenizer.py      # Tokenizer wrapper
  adamw.py          # Distributed AdamW optimizer
  muon.py           # Distributed Muon optimizer
  configurator.py   # CLI/config file argument parser
  checkpoint_manager.py  # Save/load checkpoints
  core_eval.py      # CORE metric evaluation
  loss_eval.py      # Bits per byte evaluation
  dataset.py        # Download/read pretraining data
  execution.py      # Python code execution tool
  report.py         # Report generation

scripts/            # Entry point scripts
  base_train.py     # Pretrain base model
  base_eval.py      # Evaluate CORE score
  base_loss.py      # Evaluate bits per byte
  base_complete.py  # Model completion utility
  tok_train.py      # Train tokenizer
  tok_eval.py       # Evaluate tokenizer
  configure.py      # Configuration wizard for experiments
  estimate_model_size.py  # Model size calculator
  run_init.sh       # Initialize new experiment run
  run_set.sh        # Switch to existing experiment run
  run_start.sh      # Start training run
  queue_runs.sh     # Queue multiple experimental runs

tasks/              # Evaluation task definitions
  common.py         # TaskMixture, TaskSequence
  arc.py, gsm8k.py, mmlu.py, humaneval.py, etc.

rustbpe/            # Rust BPE tokenizer
  src/lib.rs        # Rust implementation
  Cargo.toml        # Rust dependencies

dev/                # Development utilities
  gen_synthetic_data.py    # Example identity data generation
  repackage_data_reference.py  # Pretraining shard generation
  runcpu.sh         # Example CPU/MPS config
```

## Common Patterns

### Running Distributed Commands
Always use `torchrun` for multi-GPU, or omit for single GPU:
```bash
# Multi-GPU (8 GPUs)
torchrun --standalone --nproc_per_node=8 -m scripts.base_train

# Single GPU (automatic gradient accumulation)
python -m scripts.base_train
```

### Accessing Artifacts
Directory structure has been split for running multiple experiments:
- `NANOCHAT_DATA_DIR`: Shared immutable data (training shards, tokenizer, eval bundles)
- `NANOCHAT_RUN_DIR`: Run-specific outputs (checkpoints, reports, configs)

Set via `scripts/run_init.sh` or manually:
```bash
export NANOCHAT_DATA_DIR="$PWD/working/data"
export NANOCHAT_RUN_DIR="$PWD/working/runs/my_experiment"
```

### Debugging Print Statements
Use `print0()` from nanochat.common - only prints from rank 0 in distributed training.

## Experimental Workflow (Pretraining Focus)

**Philosophy**: This codebase is being adapted for rapid experimentation with small pretraining models. The focus is on understanding scaling laws and architecture choices at the <100M parameter scale, where instruct finetuning provides minimal value. The workflow prioritizes explicit configuration, reproducibility, and isolation of experimental runs.

**Note on vocab size**: For tiny models (<100M params), use vocab_size=24576 (3 × 2^13) instead of the default 65536. This reduces embedding overhead significantly - at 24576, embeddings are ~37.5% of the size compared to 65536, making more parameters available for the transformer layers where actual learning happens.

### Three-Step Run Initialization

The experimental workflow enforces upfront configuration for reproducibility:

1. **One-time setup** (already done):
   - Download training data shards to shared `NANOCHAT_DATA_DIR`
   - Train tokenizer once, shared across all experiments

2. **Per-experiment initialization**:
   ```bash
   source scripts/run_init.sh experiment_name
   # - Sets NANOCHAT_RUN_DIR=working/runs/experiment_name
   # - Sets NANOCHAT_DATA_DIR=working/data
   # - Activates venv
   # - Initializes report (errors if already exists)
   # - Runs interactive configuration wizard (scripts/configure.py)
   # - Creates config.py in run directory
   ```

3. **Execute training**:
   ```bash
   bash start_run.sh
   # - Runs base_train.py (requires config.py)
   # - Evaluates model
   # - Generates report
   ```

**Switching between runs**:
```bash
source scripts/run_set.sh experiment_name
# - Sets environment variables to point to existing run
# - Activates venv
# - Shows config summary and checkpoint status
# - Use this to switch between different experiments
```

### Configuration Philosophy

**Explicit over implicit**: Training scripts (currently `base_train.py`) require a `config.py` file in `$NANOCHAT_RUN_DIR`. This prevents accidental runs with default parameters and ensures every experiment is documented.

Configuration is handled by a configuration wizard, `scripts/configure.py`. It sets the configuration that will be used by the current run.

### Why Pretraining Only?

This fork focuses exclusively on base model pretraining at small scales (<100M parameters). At this scale:
- Models lack the capacity to benefit meaningfully from instruct finetuning
- The goal is to understand fundamental properties: architecture efficiency, scaling laws, tokenizer impact, optimization dynamics
- Chat capabilities (midtraining/SFT/RL) are irrelevant to these research questions

The original nanochat included midtraining, SFT, and RL training stages for building chat models. These have been completely removed from this fork, which focuses exclusively on base model pretraining for scaling research.

### Codebase Adaptation Notes

The codebase is being "hacked" (adapted) for experimental purposes rather than maintained as a general framework:
- Configuration is now required, not optional
- Focus on single pipeline stage (pretraining)
- Tooling optimized for rapid iteration and comparison
- Expect some features to be temporarily bypassed for experimentation

This is intentional and aligned with the nanochat philosophy of "hackable baseline" over "configurable framework".
