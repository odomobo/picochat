# nanochat Training Checkpoint

**Hardware**: RTX 3090 (24GB VRAM), Single GPU
**Status**: Successfully trained 15.7M param base model, produces text completions
**Goal**: Build tiny 38M param model for experimentation

---

## Current Status

✅ **Completed:**
- Tokenizer trained (vocab_size=24576)
- Base model trained (15.7M params, 500M tokens)
- Model generates text completions (gibberish, but works!)
- Interactive completion tool working (`base_complete.py`)

⚠️ **Known Issue:**
- Current model is 15.7M params (depth=4, dim=256, heads=2)
- Target model is 37.75M params (depth=4, dim=512, heads=8)
- Architecture mismatch due to hardcoded formulas in base_train.py

---

## Files Created This Session

1. **ittybitty.sh** - Training script for tiny model (pretraining only, 500M tokens)
2. **calculate_vram_usage.py** - VRAM calculator for different batch sizes
3. **calculate_38M_architecture.py** - Architecture parameter calculator
4. **base_complete.py** - Interactive text completion tool
5. **CLAUDE.md** - Comprehensive repo guide
6. **MODEL_SPECIFICATIONS.md** - d20/d26/d32 model specs with references

---

## Target Architecture

### Specifications

| Parameter | Value |
|-----------|-------|
| **Depth (layers)** | 4 |
| **Model dimension** | 512 |
| **Num heads** | 8 |
| **Head dimension** | 64 |
| **Vocabulary size** | 24,576 (3 × 2^13) |
| **Total parameters** | 37.75M |
| **Params per layer** | 3.15M |

### Parameter Distribution (1:1:1 split)

| Component | Parameters | Percentage |
|-----------|------------|------------|
| Token embedding (wte) | 12,582,912 | 33.33% |
| Transformer layers | 12,582,912 | 33.33% |
| LM head (lm_head) | 12,582,912 | 33.33% |
| **TOTAL** | **37,748,736** | **100.00%** |

### Design Rationale

- **1:1:1 split**: Sets up for future weight tying → 1:1 (embeddings:transformer)
- **4 layers**: Deep enough to be practical, not so deep layers become tiny
- **512 dim**: Strong representational capacity
- **8 heads**: Maximum attention diversity for this width (512/64 = 8)
- **64-dim heads**: More flexibility than standard 128-dim heads for tiny models

---

## Current vs Target Architecture

### What We Actually Built (15.7M params)

Based on nanochat's hardcoded formulas:
- **model_dim** = depth × 64 = 4 × 64 = **256** (not 512!)
- **num_heads** = (256 + 127) // 128 = **2** (not 8!)
- **head_dim** = 128 (not 64!)

**Parameter breakdown:**
- wte: 24,576 × 256 = 6.29M
- Transformer: 4 layers × 786K = 3.15M
- lm_head: 256 × 24,576 = 6.29M
- **Total: 15.73M params**

### What Needs to Change

From `ittybitty.sh` TODO comments:

**Problem**: `base_train.py` derives architecture from depth:
```python
# Line 89:
model_dim = depth * 64  # aspect ratio 64

# Line 90:
num_heads = max(1, (model_dim + 127) // 128)  # head_dim 128
```

**Solutions** (pick one):
1. Add `--model_dim` and `--head_dim` as CLI parameters
2. Change aspect ratio from 64 to 128 (depth × 128)
3. Change head_dim calculation from 128 to 64

**Recommended**: Option 1 (add CLI parameters) for maximum flexibility.

---

## Hardware Configuration (RTX 3090)

### VRAM Usage Analysis

**Theoretical vs Real-world:**
- Calculator estimate for batch_size=32: 4.4 GB
- Actual usage: ~7 GB
- Difference: ~2.6 GB overhead

**Real-world overhead includes:**
- PyTorch CUDA context (~1-1.5 GB)
- `torch.compile()` cache (~800 MB)
- Gradient accumulation buffers (~400 MB)
- Mixed precision overhead (~300 MB)
- Memory fragmentation (~300 MB)

**Rule of thumb**: Add 2-3 GB to calculator estimates for real usage.

### Optimal Batch Size: 48

| Batch Size | Theoretical VRAM | Real VRAM (est) | Fits? |
|------------|------------------|-----------------|-------|
| 32 | 14.0 GB | ~16.6 GB | ✓ YES |
| 40 | 17.4 GB | ~20.0 GB | ✓ YES |
| **48** | **20.8 GB** | **~23.4 GB** | **✓ YES** |
| 56 | 24.2 GB | ~27 GB | ✗ NO |

**Configuration in ittybitty.sh:**
```bash
python -m scripts.base_train \
  --depth=4 \
  --num_iterations=954 \
  --target_param_data_ratio=-1 \
  --device_batch_size=48 \
  --run=$WANDB_RUN
```

---

## Training Configuration

### Environment Variables

**NANOCHAT_BASE_DIR** (default: `~/.cache/nanochat`)
- Where all artifacts are saved (tokenizer, checkpoints, data shards, reports)
- Override: `export NANOCHAT_BASE_DIR="/your/path"`

**WANDB_RUN** (default: `"dummy"`)
- Name for Weights & Biases logging
- Use `"dummy"` to disable wandb logging
- Example: `export WANDB_RUN="ittybitty_rtx3090_v1"`

**OMP_NUM_THREADS=1**
- Limits OpenMP threads to prevent CPU oversubscription
- Important for multi-GPU (less critical for single GPU)
- Reduces CPU overhead and context switching

### Single GPU Training (No torchrun)

**Multi-GPU** (8 GPUs):
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=4
```

**Single GPU** (RTX 3090):
```bash
python -m scripts.base_train --depth=4
# Note: NO '--' separator needed without torchrun!
```

**Key difference**: The `--` separator is only needed with `torchrun` to separate launcher args from script args.

### Training Data

**For 500M tokens:**
- Characters needed: 500M × 4.8 = 2.4B chars
- Data shards: 2.4B / 250M = ~10 shards
- Downloaded: 15 shards for safety
- Disk space: ~1.5 GB compressed

**Chinchilla optimal (20:1):**
- For 37.75M params: 755M tokens (not used for this test run)

---

## Tools Created

### 1. calculate_vram_usage.py

Calculates VRAM usage for different batch sizes.

**Usage:**
```bash
python3 calculate_vram_usage.py
```

**Includes:**
- Model parameters (bf16)
- Optimizer states (AdamW + Muon, fp32)
- Gradients (bf16)
- Activations (scales with batch size)
- 20% overhead estimate

**Limitation**: Underestimates by 2-3 GB due to PyTorch runtime overhead.

### 2. base_complete.py

Interactive text completion tool for base models.

**Usage:**
```bash
# Default (auto-detect device)
python -m scripts.base_complete

# Faster startup on CPU
python -m scripts.base_complete --device-type cpu

# With custom parameters
python -m scripts.base_complete --device-type cpu -t 0.5 -m 200
```

**Features:**
- Loads model once, stays in memory
- Interactive prompt with `> ` on each line
- Type `###` on new line to trigger completion
- Input shown in gray, completion in normal color
- Ctrl+C to exit
- Supports newlines and arbitrary text

**Why not chat_cli/chat_web?**
- Those need midtraining/SFT for conversation format
- Base model only does raw text continuation
- base_complete is minimal and fast for testing

---

## Key Technical Learnings

### 1. Vocabulary Size Impact

Vocabulary size directly determines embedding parameter count:

```
For model_dim = 512:
- vocab_size = 65,536 → wte + lm_head = 67.1M params
- vocab_size = 24,576 → wte + lm_head = 25.2M params
- vocab_size = 8,192  → wte + lm_head = 8.4M params
```

**For tiny models**: Smaller vocab allocates more params to transformer layers (computation) vs embeddings (lookup tables).

### 2. Head Dimension Trade-off

**Standard nanochat**: 128-dim heads
- Works well for large models (d20+)
- `num_heads = (model_dim + 127) // 128`

**For tiny models**: 64-dim heads
- More flexibility at small model_dim
- Allows model_dim=512 with 8 heads (512/64=8)
- More attention diversity

### 3. Untied Weights (wte vs lm_head)

nanochat uses **separate** matrices for training benefits:
- Different learning rates (embedding_lr=0.2 vs unembedding_lr=0.004 = 50× difference!)
- Different precisions (wte can be bf16, lm_head stays fp32)
- Better training dynamics

**Future plan**: Tie them after proving functionality to save 12.5M params.

### 4. rustbpe Tokenizer

- Trains BPE in Rust (fast, parallel)
- Exports to tiktoken for inference
- Fills gap: tiktoken (inference only) vs HuggingFace (bloated)
- Uses GPT-4 style regex splitting pattern

### 5. Training Pipeline Stages

1. **base.pt** (what we have): Raw language modeling, no chat awareness
2. **mid.pt**: Learns conversation tokens (`<|user_start|>`, etc.) and tool use
3. **sft.pt**: Supervised fine-tuning, better instruction following
4. **rl.pt**: Optional RL on specific tasks (GSM8K)

**chat_cli/chat_web require at least mid.pt** to understand conversation format.

---

## Next Steps

### Immediate: Fix Architecture Mismatch

**Goal**: Modify `base_train.py` to build target architecture (dim=512, heads=8).

**Recommended approach**:
1. Add `--model_dim` CLI parameter (overrides depth × 64 formula)
2. Add `--head_dim` CLI parameter (overrides hardcoded 128)
3. Update derived calculations:
   ```python
   # Current:
   model_dim = depth * 64
   num_heads = max(1, (model_dim + 127) // 128)

   # Modified:
   model_dim = model_dim if model_dim else depth * 64
   num_heads = max(1, (model_dim + head_dim - 1) // head_dim)
   ```

**Then retrain**:
```bash
python -m scripts.base_train \
  --depth=4 \
  --model_dim=512 \
  --head_dim=64 \
  --num_iterations=954 \
  --target_param_data_ratio=-1 \
  --device_batch_size=48 \
  --run=$WANDB_RUN
```

### Future Work

1. **Weight tying**: After proving 38M model works, tie wte/lm_head to save 12.5M params
2. **Evaluation**: Run base_eval to get CORE score
3. **Optional**: Run midtraining/SFT if wanting to test chat capabilities
4. **Experiments**: Try different architectures using calculate_38M_architecture.py

---

## Quick Reference Commands

### Setup
```bash
# Set working directory
export NANOCHAT_BASE_DIR="./nanochat_workspace"

# Set wandb run name (or "dummy" to disable)
export WANDB_RUN="ittybitty_v1"

# Activate venv
source .venv/bin/activate
```

### Training Pipeline (Current)
```bash
# 1. Train tokenizer
python -m scripts.tok_train --vocab_size=24576 --max_chars=2000000000

# 2. Evaluate tokenizer
python -m scripts.tok_eval

# 3. Download data
python -m nanochat.dataset -n 15

# 4. Train model (produces 15.7M params - wrong architecture!)
python -m scripts.base_train \
  --depth=4 \
  --num_iterations=954 \
  --target_param_data_ratio=-1 \
  --device_batch_size=48 \
  --run=$WANDB_RUN

# 5. Evaluate
python -m scripts.base_loss
python -m scripts.base_eval
```

### Testing
```bash
# Interactive text completion (fast on CPU)
python -m scripts.base_complete --device-type cpu

# Calculate VRAM for batch sizes
python3 calculate_vram_usage.py

# Calculate architecture parameters
python3 calculate_38M_architecture.py
```

---

## Session Notes

**What worked:**
- ✅ Successfully trained a tiny model (even if wrong size)
- ✅ Model generates text (gibberish, but structurally correct)
- ✅ Interactive completion tool is fast and responsive on CPU
- ✅ VRAM calculations were helpful for planning batch size
- ✅ Single GPU setup works smoothly on RTX 3090

**What to fix:**
- ⚠️ Architecture mismatch (15.7M vs 37.75M params)
- ⚠️ Need to modify base_train.py to support custom model_dim/head_dim
- 📝 Document the changes needed in base_train.py

**Observations:**
- Tiny 15.7M model produces text that looks like English but makes no sense (expected)
- Loading on CPU is faster for quick testing/iteration
- Real VRAM usage ~60% higher than calculator estimates (good to know!)
- The `--` separator confusion with torchrun vs python (now understood)

---

## File Locations

**Scripts:**
- `ittybitty.sh` - Main training script
- `scripts/base_complete.py` - Interactive completion tool
- `scripts/base_train.py` - Training script (needs modification)
- `calculate_vram_usage.py` - VRAM calculator
- `calculate_38M_architecture.py` - Architecture calculator

**Checkpoints:** (in `$NANOCHAT_BASE_DIR/d4/`)
- `base.pt` - Trained base model (15.7M params)

**Data:** (in `$NANOCHAT_BASE_DIR/`)
- `tokenizer/` - Trained tokenizer (vocab=24576)
- `tokenized_data/` - FineWeb shards
- `eval_bundle/` - Evaluation datasets
