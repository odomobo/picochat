# nanochat Session Checkpoint

**Date**: Current session
**Goal**: Design a tiny transformer architecture (~38M params) for experimentation

---

## Session Summary

Analyzed the nanochat codebase and designed a custom 38M parameter architecture with balanced parameter distribution between embeddings and transformer layers.

---

## Files Created

1. **CLAUDE.md** - Comprehensive guide for Claude Code when working in this repo
2. **MODEL_SPECIFICATIONS.md** - Detailed specs for nanochat's d20, d26, d32 models with source references
3. **calculate_38M_architecture.py** - Python script to calculate balanced architectures
   - Run with: `python3 calculate_38M_architecture.py`

---

## Key Technical Insights

### rustbpe (Rust BPE Tokenizer)
- Trains BPE tokenizer in Rust (fast, parallel)
- Exports to tiktoken for inference
- Fills gap between tiktoken (inference only) and HuggingFace (bloated)

### Vocabulary Size Impact
**Critical insight**: Vocabulary size determines embedding parameter count!

For `model_dim = 512`:
- `vocab_size = 65,536` → wte + lm_head = 67.1M params (just embeddings!)
- `vocab_size = 24,576` → wte + lm_head = 25.2M params ✅

**For tiny models**: Small vocab allocates more params to transformer layers (actual computation) vs embeddings (lookup tables).

### Head Dimension for Tiny Models
**Standard nanochat**: 128-dim heads (from base_train.py:90)

**For tiny models**: 64-dim heads gives more flexibility
- Allows more heads at smaller model_dim
- Can build with model_dim=512 and 8 heads (512/64=8)

### Untied Weights (wte vs lm_head)
nanochat uses **separate** wte and lm_head matrices (not tied) for:
- Different learning rates (embedding_lr=0.2 vs unembedding_lr=0.004)
- Different precisions (wte can be bf16, lm_head stays fp32)
- Better training dynamics

**Our plan**: Start untied, then tie them later once model proves functional.

---

## Final Architecture: d4, vocab=24576

```bash
--depth=4 --vocab_size=24576
```

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

### Parameter Distribution

| Component | Parameters | Percentage |
|-----------|------------|------------|
| Token embedding (wte) | 12,582,912 | **33.33%** |
| Transformer layers | 12,582,912 | **33.33%** |
| LM head (lm_head) | 12,582,912 | **33.33%** |
| **TOTAL** | **37,748,736** | **100.00%** |

Mathematically perfect 1:1:1 split.

### Why This Architecture?

**Design philosophy:**
- **1:1:1 split** sets up for future weight tying → 1:1 (embeddings:transformer = 50:50)
- **4 layers**: Deep enough to be practical, not so deep layers become too small
- **512 dim**: Strong representational capacity
- **8 heads**: Maximum attention diversity for this width (512/64 = 8)
- **3.15M params/layer**: Substantial enough to learn meaningful patterns

**Depth/width ratio:** 512:4 = 128:1 (similar to GPT-2 Small's 768:12 = 64:1)

### Design Constraints (from discussion)

> "The goal is to have hyperparameters which are balanced, not mathematically perfect percentages."

- d2 is too shallow to be practical
- d7+ is too deep (layers too small to learn practical patterns)
- d4 is the sweet spot
- Eventually will **tie wte and lm_head** → converts to 1:1 split
- At tiny scale, 1:1 embeddings:transformer is maximum acceptable ratio

---

## Implementation

### Modify nanochat for 64-dim heads

Edit `scripts/base_train.py:90`:

```python
# Standard nanochat (128-dim heads):
num_heads = max(1, (model_dim + 127) // 128)

# Change to 64-dim heads:
num_heads = max(1, (model_dim + 63) // 64)
```

### Training Commands

```bash
# 1. Setup environment
source .venv/bin/activate

# 2. Train custom tokenizer with 24K vocab
python -m scripts.tok_train --vocab_size=24576 --max_chars=4000000000

# 3. Evaluate tokenizer
python -m scripts.tok_eval

# 4. Download training data (~15 shards)
python -m nanochat.dataset -n 15

# 5. Train the model
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
  --depth=4 \
  --device_batch_size=32

# Note: vocab_size is read from the trained tokenizer automatically
```

### Training Data Requirements (Chinchilla 20:1)

For 37.75M params:
- **Tokens**: 755M (0.76B)
- **Characters**: ~3.6B @ 4.8 chars/token
- **Data shards**: ~15 @ 250M chars/shard
- **Disk space**: ~1.5GB compressed

---

## Next Steps

1. ✅ **Architecture designed**: depth=4, vocab=24576, dim=512, heads=8
2. **Modify base_train.py** for 64-dim heads
3. **Train tokenizer** with vocab_size=24576
4. **Run training** with --depth=4
5. **Evaluate performance**
6. **If successful**: Implement weight tying between wte and lm_head
   - Reduces params from ~38M to ~25M
   - Changes distribution from 1:1:1 to 1:1 (embeddings:transformer = 50:50)

---

## Quick Reference

### Architecture Summary
```
d4: depth=4, vocab=24576, dim=512, heads=8 (head_dim=64)
Total: 37.75M params
Split: 33.33% wte / 33.33% transformer / 33.33% lm_head
```

### Key Files
- Architecture calculator: `calculate_38M_architecture.py`
- Training script to modify: `scripts/base_train.py` (line 90)
- Tokenizer training: `scripts/tok_train.py`

### Useful Commands
```bash
# Run architecture calculator
python3 calculate_38M_architecture.py

# Train model
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=4

# Chat with model (after training)
python -m scripts.chat_cli
python -m scripts.chat_web
```
