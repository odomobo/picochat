# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

nanochat is a full-stack implementation of an LLM like ChatGPT in a single, clean, minimal, hackable codebase. It trains ChatGPT-style models from scratch including tokenization, pretraining, finetuning, evaluation, and inference. The entire pipeline is designed to run on a single 8XH100 node.

**Philosophy**: nanochat is not an exhaustively configurable LLM "framework" - it's a cohesive, minimal, readable, hackable, maximally-forkable "strong baseline" codebase. Avoid adding giant configuration objects, model factories, or complex if-then-else logic.

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

**Quick speedrun (~$100, 4 hours on 8XH100)**:
```bash
bash speedrun.sh
# Or in a screen session with logging:
screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh
```

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
```bash
# Chat via CLI (interactive)
python -m scripts.chat_cli

# Chat via CLI (single prompt)
python -m scripts.chat_cli -p "Why is the sky blue?"

# Chat via web UI (ChatGPT-style interface)
python -m scripts.chat_web
# Then visit the URL shown (e.g., http://YOUR_IP:8000/)
```

### Individual Pipeline Stages
```bash
# 1. Tokenizer
python -m scripts.tok_train --max_chars=2000000000
python -m scripts.tok_eval

# 2. Base model (pretraining)
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=20
torchrun --standalone --nproc_per_node=8 -m scripts.base_loss
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval

# 3. Midtraining (conversation tokens, tool use, multiple choice)
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i mid

# 4. Supervised Fine-tuning (SFT)
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft

# 5. Reinforcement Learning (optional, GSM8K only)
torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i rl -a GSM8K
```

### Data Management
```bash
# Download pretraining data shards (each ~250M chars, ~100MB compressed)
python -m nanochat.dataset -n 240  # Downloads 240 shards for d20 model
```

## Architecture

### Training Pipeline Stages
1. **Tokenizer Training** (rustbpe): Custom Rust BPE tokenizer with 2^16 (65536) vocab size
2. **Base Model Pretraining**: GPT-style autoregressive language modeling on web text
3. **Midtraining**: Teach conversation format, special tokens, tool use, multiple choice
4. **Supervised Fine-tuning (SFT)**: Domain adaptation per-sequence
5. **Reinforcement Learning (RL)**: Optional post-training on specific tasks (currently GSM8K)

### Model Architecture (nanochat/gpt.py)

The GPT implementation uses a modern transformer with:
- **Rotary Positional Embeddings (RoPE)**: No learned positional embeddings
- **QK Normalization**: Normalizes queries and keys in attention
- **Untied Weights**: Separate token embedding (wte) and language model head (lm_head)
- **ReLU^2 Activation**: In MLP layers instead of GELU
- **RMSNorm**: Functional normalization without learnable parameters, applied after token embedding and after each block
- **No Bias**: Linear layers have no bias terms
- **Multi-Query Attention (MQA) / Grouped Query Attention (GQA)**: Configurable n_kv_head for efficient inference
- **Logits Softcapping**: 15.0 * tanh(logits / 15.0)

Model size is determined by `--depth` parameter:
- `depth=20` → 561M params (d20, ~$100 speedrun)
- `depth=26` → ~800M params (d26, ~$300, GPT-2 grade)
- `depth=32` → 1.9B params (d32, ~$800)
- `model_dim = depth * 64` (aspect ratio of 64)
- `num_heads = max(1, (model_dim + 127) // 128)` (head dim of 128)

### Dual Optimizer Setup (nanochat/gpt.py:213-242)

**Critical architectural pattern**: The model uses two separate optimizers:
1. **AdamW** for embeddings and unembedding (wte, lm_head)
   - Learning rates scaled by `(model_dim / 768)^-0.5`
   - Default LRs: embedding_lr=0.2, unembedding_lr=0.004
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

Models are saved to `~/.cache/nanochat/{model_tag}/` by default:
- `base.pt`: Pretrained base model
- `mid.pt`: After midtraining
- `sft.pt`: After supervised fine-tuning
- `rl.pt`: After reinforcement learning (optional)

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
- **CORE**: Comprehensive evaluation from DCLM paper (base model quality)
- **ARC**: AI2 Reasoning Challenge (science questions)
- **GSM8K**: Grade School Math (8K problems)
- **HumanEval**: Python coding task
- **MMLU**: Massive Multitask Language Understanding
- **SmolTalk**: Conversational dataset from HuggingFace
- **SpellingBee**: Custom task for teaching letter counting/spelling
- **CustomJSON**: Load arbitrary conversational datasets from JSONL

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

## Customization

### Adding Custom Personality/Identity

See GitHub Discussions guide: "infusing identity to your nanochat"
1. Generate synthetic conversation data (see dev/gen_synthetic_data.py)
2. Save as JSONL format
3. Mix into midtraining and SFT stages

### Adding New Capabilities

See GitHub Discussions guide: "counting r in strawberry (and how to add abilities generally)"
1. Create custom task in tasks/ directory
2. Add to TaskMixture for midtraining
3. Evaluate with chat_eval

### Training Larger Models (d26, d32)

To train GPT-2 grade d26 model from speedrun.sh:
```bash
# Download more data shards (450 for d26)
python -m nanochat.dataset -n 450 &

# Increase depth, decrease batch size to fit in memory
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=26 --device_batch_size=16

# Use same batch size in midtraining
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train -- --device_batch_size=16
```

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
  mid_train.py      # Midtraining
  chat_sft.py       # Supervised fine-tuning
  chat_rl.py        # Reinforcement learning
  chat_eval.py      # Evaluate chat models
  chat_cli.py       # CLI chat interface
  chat_web.py       # Web UI chat interface
  tok_train.py      # Train tokenizer
  tok_eval.py       # Evaluate tokenizer

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

Set via `init_run.sh` or manually:
```bash
export NANOCHAT_DATA_DIR="$PWD/working/data"
export NANOCHAT_RUN_DIR="$PWD/working/runs/my_experiment"
```

### Debugging Print Statements
Use `print0()` from nanochat.common - only prints from rank 0 in distributed training.

## Experimental Workflow (Pretraining Focus)

**Philosophy**: This codebase is being adapted for rapid experimentation with small pretraining models. The focus is on understanding scaling laws and architecture choices at the <100M parameter scale, where instruct finetuning provides minimal value. The workflow prioritizes explicit configuration, reproducibility, and isolation of experimental runs.

### Three-Step Run Initialization

The experimental workflow enforces upfront configuration for reproducibility:

1. **One-time setup** (already done):
   - Download training data shards to shared `NANOCHAT_DATA_DIR`
   - Train tokenizer once, shared across all experiments

2. **Per-experiment initialization**:
   ```bash
   source init_run.sh experiment_name
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

### Configuration Philosophy

**Explicit over implicit**: Training scripts (currently `base_train.py`) require a `config.py` file in `$NANOCHAT_RUN_DIR`. This prevents accidental runs with default parameters and ensures every experiment is documented.

Configuration is handled by a configuration wizard, `scripts/configure.py`. It sets the configuration that will be used by the current run.

### Why Pretraining Only?

At small scales (<100M parameters), the models lack the capacity to benefit meaningfully from instruct finetuning. The experimental focus is on understanding:
- Architecture efficiency at different scales
- Data requirements and scaling laws
- Tokenizer impact on model performance
- Training dynamics and optimization

Midtraining, SFT, and RL stages remain in the codebase but are not part of the current experimental workflow.

### Codebase Adaptation Notes

The codebase is being "hacked" (adapted) for experimental purposes rather than maintained as a general framework:
- Configuration is now required, not optional
- Focus on single pipeline stage (pretraining)
- Tooling optimized for rapid iteration and comparison
- Expect some features to be temporarily bypassed for experimentation

This is intentional and aligned with the nanochat philosophy of "hackable baseline" over "configurable framework".
