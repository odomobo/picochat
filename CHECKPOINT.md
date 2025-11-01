# nanochat Session Checkpoint

**Date**: Current session
**Goal**: Design a tiny transformer architecture (~38M params) for experimentation

---

## Session Summary

We analyzed the nanochat codebase and designed custom architectures for tiny language models with balanced parameter distributions between embeddings and transformer layers.

---

## Files Created

1. **CLAUDE.md** - Comprehensive guide for Claude Code when working in this repo
   - Build commands, architecture details, training pipeline
   - Memory management, scaling laws, customization guides

2. **MODEL_SPECIFICATIONS.md** - Detailed specs for d20, d26, d32 models
   - All architectural parameters with explicit source references
   - Training data requirements, FLOPs calculations
   - Hardware requirements and performance metrics

3. **calculate_38M_architecture.py** - Python script to calculate balanced architectures
   - Target: 38M parameters
   - Goal: Balanced distribution between wte, lm_head, and transformer layers
   - Uses 64-dim heads (vs. 128-dim in standard nanochat)
   - Run with: `python3 calculate_38M_architecture.py`

---

## Key Technical Insights

### 1. rustbpe (Rust BPE Tokenizer)

**What it does:**
- Fills the gap between tiktoken (inference only) and HuggingFace tokenizers (bloated)
- Trains BPE tokenizer in Rust (fast, parallel)
- Exports to tiktoken for inference
- GPT-4 style regex splitting pattern for text chunks

**Key point**: nanochat uses **untied weights** (separate wte and lm_head) for modern training practices:
- Different learning rates (embedding_lr=0.2 vs unembedding_lr=0.004)
- Different precisions (wte can be bf16, lm_head stays fp32)
- Better training dynamics for large models

### 2. Vocabulary Size Impact

**Critical insight**: Vocabulary size determines embedding parameter count!

For a model with `model_dim = 384`:
- `vocab_size = 65,536` → wte + lm_head = 50.3M params (just embeddings!)
- `vocab_size = 32,768` → wte + lm_head = 25.2M params
- `vocab_size = 16,384` → wte + lm_head = 12.6M params

**For tiny models (25-40M params):**
- Large vocab wastes parameters on embeddings (lookup tables)
- Small vocab allocates more params to transformer layers (actual computation)
- Better to learn fewer tokens deeply than many tokens shallowly

### 3. Head Dimension Trade-off

**Standard nanochat**: 128-dim heads
- From base_train.py: `num_heads = max(1, (model_dim + 127) // 128)`

**For tiny models**: 64-dim heads gives more flexibility
- Allows more heads at smaller model_dim
- More architectural options at small scales
- Can build deeper models with narrower width

### 4. Scaling Laws (Chinchilla)

**Optimal ratio**: 20 tokens per parameter
- d20 (561M params) → 11.2B tokens
- d32 (1.88B params) → 38B tokens

**FLOPs budget determines capability:**
- 4e19 FLOPs → "kindergartener" level
- 4.5e20 FLOPs → GPT-2 level

---

## Architecture Design Philosophy

### Original Goal: 25M params, 500M tokens

**Recommendations explored:**
- Conservative: depth=3, vocab=65536, dim=192, heads=2 (26.5M params)
- **Balanced**: depth=6, vocab=16384, dim=384, heads=3 (23.2M params) ⭐
- Efficient: depth=7, vocab=8192, dim=448, heads=4 (24.2M params)

### Updated Goal: 38M params with balanced distribution

**Target**: ~33% wte, ~33% lm_head, ~33% transformer layers

**Why this split?**
- Plan to **tie wte and lm_head later** → becomes 50% embeddings, 50% transformer
- At tiny scale, embeddings might dwarf transformer, but 1:1 is maximum acceptable ratio
- Goal is BALANCED hyperparameters, not mathematically perfect percentages
- Too shallow (d2) = not practical
- Too deep (d7+) = layers too small to learn anything practical

---

## Final Architecture Candidates (64-dim heads)

### Option 1: "The Extremist" (d29)
```bash
--depth=29 --vocab_size=65536
```

**Specs:**
- Layers: 29, Model dim: 192, Heads: 3, Head dim: 64
- Total: 38.0M params (0.01% off target)
- Distribution: 33.12% / 33.76% / 33.12%
- Params per layer: 442K

**Philosophy:** Maximum depth, minimum width
- 29 computational steps
- Full vocabulary (65K tokens)
- Very narrow (192 dim)
- Only 3 attention heads

**Assessment:**
- ✅ Nearly perfect parameter count
- ✅ Full vocabulary (no custom tokenizer)
- ❌ Unprecedented depth for 38M model (risky)
- ❌ Very narrow width may limit representations
- ❌ Only 3 heads limits attention diversity
- ❌ Tiny layers (442K each) may not learn well

---

### Option 2: "The Goldilocks" (d4) ⭐ RECOMMENDED
```bash
--depth=4 --vocab_size=24576
```

**Specs:**
- Layers: 4, Model dim: 512, Heads: 8, Head dim: 64
- Total: 37.7M params (0.66% under target)
- Distribution: **exactly 33.33% / 33.33% / 33.33%**
- Params per layer: 3.1M

**Philosophy:** Balanced width and depth
- 4 computational steps (practical depth)
- Medium vocabulary (24K tokens = 3 × 2^13)
- Wide model (512 dim)
- 8 attention heads (maximum for this width)

**Assessment:**
- ✅ Mathematically perfect distribution
- ✅ 8 heads = excellent attention diversity
- ✅ 512 dim = strong representational capacity
- ✅ 3.1M params/layer = substantial, should train well
- ✅ 4 layers = proven depth, stable training
- ⚠️ Requires custom tokenizer (24K vocab)

**Depth/width ratio:** 512:4 = 128:1
- Similar to GPT-2 Small (768:12 = 64:1)
- Well-tested scaling pattern

---

### Option 3: "The Conventional" (d7)
```bash
--depth=7 --vocab_size=32768
```

**Specs:**
- Layers: 7, Model dim: 384, Heads: 6, Head dim: 64
- Total: 37.6M params (1.18% under target)
- Distribution: 33.51% / 32.98% / 33.51%
- Params per layer: 1.8M

**Philosophy:** Conventional transformer scaling
- 7 computational steps
- Medium vocabulary (32K tokens)
- Moderate width (384 dim)
- 6 attention heads

**Assessment:**
- ✅ Respectable depth
- ✅ Good attention diversity (6 heads)
- ✅ Balanced proportions
- ⚠️ User considers this "too deep" - layers too small to be practical
- ⚠️ 1.8M params/layer may be on the small side

---

## Decision Framework

**Key insight from user:**
> "mathematically perfect doesn't matter at all. It's a target to shoot for, not the goal. The goal is to have hyperparameters which are balanced."

**Practical constraints:**
- d2 is too shallow to be practical
- d7+ is too deep (layers too small to learn practical patterns)
- d4 is the sweet spot: deep enough to be useful, wide enough to learn

**Future plan:**
- Once model proves functional, **tie wte and lm_head** together
- This converts 1:1:1 split → 1:1 split (embeddings:transformer = 50:50)
- At tiny scale, embeddings > transformer might help performance
- But 1:1 is maximum acceptable ratio

---

## Implementation Notes

### To use 64-dim heads in nanochat:

Modify `scripts/base_train.py:90`:

```python
# Standard nanochat (128-dim heads):
num_heads = max(1, (model_dim + 127) // 128)

# For 64-dim heads:
num_heads = max(1, (model_dim + 63) // 64)
```

### To train with custom vocab size:

```bash
# 1. Train tokenizer
python -m scripts.tok_train --vocab_size=24576 --max_chars=4000000000

# 2. Eval tokenizer
python -m scripts.tok_eval

# 3. Download data (~15 shards for 38M model)
python -m nanochat.dataset -n 15

# 4. Train model
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
  --depth=4 \
  --device_batch_size=32
```

### Data requirements (Chinchilla 20:1):

For 37.7M params (d4 model):
- Tokens: 754M (0.75B)
- Characters: ~3.6B @ 4.8 chars/token
- Data shards: ~15 @ 250M chars/shard
- Disk space: ~1.5GB compressed

---

## Next Steps

1. **Decide on final architecture** (likely d4, vocab=24576)
2. **Implement 64-dim head modification** in base_train.py
3. **Train custom tokenizer** with vocab_size=24576
4. **Run training** with --depth=4
5. **Evaluate performance**
6. **If successful**: Implement weight tying between wte and lm_head
   - This will reduce params from ~38M to ~25M
   - Changes distribution from 1:1:1 to 1:1 (embeddings:transformer = 50:50)

---

## Open Questions

1. Will 4 layers be deep enough for practical performance?
2. How well will 64-dim heads work (vs. standard 128-dim)?
3. Should we start with d4 or try d7 for comparison?
4. Performance comparison: tied vs. untied weights at this scale?

---

## Key Files Reference

- **Architecture calculator**: `calculate_38M_architecture.py`
- **Model specs**: `MODEL_SPECIFICATIONS.md`
- **Training script**: `scripts/base_train.py` (needs modification for 64-dim heads)
- **Tokenizer training**: `scripts/tok_train.py`
- **Main training scripts**: `speedrun.sh` (d20), `run1000.sh` (d32)

---

## Useful Commands

```bash
# Run architecture calculator
python3 calculate_38M_architecture.py

# Train tokenizer with custom vocab
python -m scripts.tok_train --vocab_size=24576

# Train model with custom depth
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=4

# Chat with model (after training)
python -m scripts.chat_cli
python -m scripts.chat_web
```

---

## Context for Next Session

**Current recommendation**: depth=4, vocab=24576, dim=512, heads=8 (64-dim each)

This gives:
- 37.7M total parameters
- Perfect 33.33% / 33.33% / 33.33% split
- 8 attention heads (maximum diversity)
- 512 model dimension (strong capacity)
- 3.1M params per layer (substantial)
- Practical depth (not too shallow, not too deep)

**Philosophy**: Balanced, practical architecture that should train well and perform reasonably for a tiny model. The 1:1:1 split sets us up for future weight tying (→ 1:1 split).
