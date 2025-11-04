# Implementation Report: Output Projection Layer for Tied Weights

## Overview
Add an optional `hidden_dim × hidden_dim` linear projection layer between the final transformer output and lm_head when using tied weights. This layer provides a transformation space to decouple the dual role of tied embeddings.

---

## 1. Architecture Changes (nanochat/gpt.py)

### 1.1 GPTConfig Dataclass (line ~26-34)
**Add new field:**
```python
@dataclass
class GPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    tie_weights: bool = False
    use_output_projection: bool = False  # NEW: add projection before lm_head when tie_weights=True
```

**Backward compatibility default:** `False` (no projection, original behavior)

### 1.2 GPT.__init__() (line ~140-153)
**Modify lm_head creation logic:**

Current:
```python
if config.tie_weights:
    self.lm_head = None
else:
    self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
```

New:
```python
if config.tie_weights:
    if config.use_output_projection:
        # Add projection layer before using tied weights
        self.output_projection = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.lm_head = None  # Will use wte via projection
    else:
        # Original: use wte directly
        self.output_projection = None
        self.lm_head = None
else:
    # Untied: no projection needed, separate lm_head
    self.output_projection = None
    self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
```

### 1.3 GPT.init_weights() (line ~164-180)
**Add initialization for output_projection:**

After the c_proj zero-initialization block (line ~171-173):
```python
# zero out c_proj weights in all blocks
for block in self.transformer.h:
    torch.nn.init.zeros_(block.mlp.c_proj.weight)
    torch.nn.init.zeros_(block.attn.c_proj.weight)

# NEW: Don't zero out output_projection - use standard init
# It's initialized by self.apply(self._init_weights) above
```

**Note:** The output_projection will get standard initialization from `_init_weights()` (fan-in/fan-out scaled normal distribution). Don't special-case it.

### 1.4 GPT.forward() (line ~272-290)
**Modify logits computation:**

Current:
```python
# Compute logits: use tied weights if enabled, otherwise use separate lm_head
if self.config.tie_weights:
    logits = F.linear(x, self.transformer.wte.weight)
else:
    logits = self.lm_head(x)
```

New:
```python
# Compute logits
if self.config.tie_weights:
    if self.config.use_output_projection:
        # Apply projection then use tied weights
        x = self.output_projection(x)
        logits = F.linear(x, self.transformer.wte.weight)
    else:
        # Original: use tied weights directly
        logits = F.linear(x, self.transformer.wte.weight)
else:
    # Untied: use separate lm_head
    logits = self.lm_head(x)
```

### 1.5 GPT.setup_optimizers() (line ~222-270)
**Add output_projection to Muon optimizer:**

Current:
```python
# Separate out all parameters into groups
matrix_params = list(self.transformer.h.parameters())
embedding_params = list(self.transformer.wte.parameters())
```

New:
```python
# Separate out all parameters into groups
matrix_params = list(self.transformer.h.parameters())

# NEW: Add output_projection to matrix params if it exists
if self.config.use_output_projection and self.output_projection is not None:
    matrix_params.extend(list(self.output_projection.parameters()))

embedding_params = list(self.transformer.wte.parameters())
```

**Update assertion for tied weights case:**

Current:
```python
if self.config.tie_weights:
    # ...
    assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params)
```

New (needs to account for output_projection):
```python
if self.config.tie_weights:
    # ...
    expected_params = len(matrix_params) + len(embedding_params)
    actual_params = len(list(self.parameters()))
    assert actual_params == expected_params, f"Param count mismatch: {actual_params} != {expected_params}"
```

**Note:** The assertion should already work since output_projection params are added to matrix_params.

---

## 2. Checkpoint Backward Compatibility (nanochat/checkpoint_manager.py)

### 2.1 build_model() (line ~70-78)
**Add fallback for use_output_projection:**

After the tie_weights fallback (line ~72-75):
```python
# Backward compatibility: if tie_weights not in old checkpoints, assume False (untied)
if "tie_weights" not in model_config_kwargs:
    model_config_kwargs["tie_weights"] = False
    log0("Warning: tie_weights not found in checkpoint metadata, assuming False (old untied model)")

# NEW: Backward compatibility for use_output_projection
if "use_output_projection" not in model_config_kwargs:
    model_config_kwargs["use_output_projection"] = False
    # Only log if tie_weights=True (otherwise it doesn't matter)
    if model_config_kwargs.get("tie_weights", False):
        log0("Warning: use_output_projection not found in checkpoint metadata, assuming False")
```

---

## 3. Training Script Changes (scripts/base_train.py)

### 3.1 User Settings (line ~38-42)
**Add new parameter:**

After tie_weights:
```python
# Model architecture
depth = 20
model_dim = depth * 64
max_seq_len = 2048
tie_weights = False
use_output_projection = False  # NEW: add projection layer before lm_head (only used if tie_weights=True)
```

### 3.2 Model Config (line ~129)
**Add to model_config_kwargs:**

Current:
```python
model_config_kwargs = dict(sequence_len=max_seq_len, vocab_size=vocab_size, n_layer=num_layers, n_head=num_heads, n_kv_head=num_kv_heads, n_embd=model_dim, tie_weights=tie_weights)
```

New:
```python
model_config_kwargs = dict(
    sequence_len=max_seq_len,
    vocab_size=vocab_size,
    n_layer=num_layers,
    n_head=num_heads,
    n_kv_head=num_kv_heads,
    n_embd=model_dim,
    tie_weights=tie_weights,
    use_output_projection=use_output_projection  # NEW
)
```

---

## 4. Configuration Wizard (scripts/configure.py)

### 4.1 Add Question (line ~111-118)
**After the tie_weights question:**

Current:
```python
tie_weights = get_bool_input("Tie embedding weights (wte and lm_head)? Reduces params by ~50%", default=True)

# Only ask for tied_weights_lr if tie_weights is enabled
if tie_weights:
    tied_weights_lr = get_float_input("Learning rate for tied weights (standard: 0.2)", default=0.2)
else:
    tied_weights_lr = 0.2
```

New:
```python
tie_weights = get_bool_input("Tie embedding weights (wte and lm_head)? Reduces params by ~50%", default=True)

# NEW: Only ask about output projection if weights are tied
if tie_weights:
    use_output_projection = get_bool_input(
        "Add output projection layer before lm_head? Helps with tied weights learning",
        default=False
    )
    tied_weights_lr = get_float_input("Learning rate for tied weights (standard: 0.2)", default=0.2)
else:
    use_output_projection = False  # Doesn't apply when untied
    tied_weights_lr = 0.2
```

### 4.2 Update Parameter Count Display (line ~131-162)
**Modify parameter calculation:**

Current:
```python
if tie_weights:
    wte_params = vocab_size * model_dim  # Shared with lm_head
    lm_head_params = 0  # Tied to wte
    total_params = wte_params + transformer_params
```

New:
```python
if tie_weights:
    wte_params = vocab_size * model_dim  # Shared with lm_head
    lm_head_params = 0  # Tied to wte

    # NEW: Account for output projection if enabled
    if use_output_projection:
        output_projection_params = model_dim * model_dim
        total_params = wte_params + transformer_params + output_projection_params
    else:
        output_projection_params = 0
        total_params = wte_params + transformer_params
```

**Update display section:**

Current:
```python
print("Parameter count:")
if tie_weights:
    print(f"  wte (tied with lm_head):     {wte_params:>12,} ({wte_params/1e6:>6.2f}M)")
else:
    # ...
print(f"  Transformer layers:          {transformer_params:>12,} ({transformer_params/1e6:>6.2f}M)")
```

New:
```python
print("Parameter count:")
if tie_weights:
    print(f"  wte (tied with lm_head):     {wte_params:>12,} ({wte_params/1e6:>6.2f}M)")
    if use_output_projection:
        print(f"  Output projection:           {output_projection_params:>12,} ({output_projection_params/1e6:>6.2f}M)")
else:
    print(f"  wte (embeddings):            {wte_params:>12,} ({wte_params/1e6:>6.2f}M)")
    print(f"  lm_head (unembedding):       {lm_head_params:>12,} ({lm_head_params/1e6:>6.2f}M)")
print(f"  Transformer layers:          {transformer_params:>12,} ({transformer_params/1e6:>6.2f}M)")
```

### 4.3 Update Config Template (line ~199-203)
**Add to generated config.py:**

After tie_weights:
```python
# Model architecture
depth = {depth}
model_dim = {model_dim}  # aspect ratio {model_dim / depth if depth != 0 else 0:.1f} (or custom)
max_seq_len = {max_seq_len}
tie_weights = {str(tie_weights)}  # tie wte and lm_head weights (reduces params by ~50%)
use_output_projection = {str(use_output_projection)}  # add projection before lm_head when tie_weights=True
```

---

## 5. Implementation Checklist

### Phase 1: Core Architecture
- [ ] Add `use_output_projection` to GPTConfig
- [ ] Modify `GPT.__init__()` to conditionally create output_projection
- [ ] Update `GPT.forward()` to route through projection when enabled
- [ ] Add output_projection params to Muon optimizer in setup_optimizers()
- [ ] Verify parameter assertions still pass

### Phase 2: Configuration
- [ ] Add `use_output_projection` to base_train.py user settings
- [ ] Add to model_config_kwargs dict
- [ ] Add question to configure.py
- [ ] Update parameter count calculation in configure.py
- [ ] Add to config.py template

### Phase 3: Backward Compatibility
- [ ] Add fallback in checkpoint_manager.py build_model()
- [ ] Test loading old checkpoints (should default to use_output_projection=False)
- [ ] Test loading new checkpoints with use_output_projection=True

### Phase 4: Testing
- [ ] Create new run with tie_weights=True, use_output_projection=False (baseline)
- [ ] Create new run with tie_weights=True, use_output_projection=True (experimental)
- [ ] Verify parameter counts match expectations
- [ ] Compare training curves
- [ ] Verify checkpoints save/load correctly

---

## 6. Expected Parameter Counts

**Example: depth=4, model_dim=512, vocab_size=24576**

| Configuration | wte | lm_head | output_proj | Transformer | Total |
|---------------|-----|---------|-------------|-------------|-------|
| Untied (original) | 12.6M | 12.6M | 0 | 6.3M | 31.5M |
| Tied, no projection | 12.6M | 0 (tied) | 0 | 6.3M | 18.9M |
| Tied, with projection | 12.6M | 0 (tied) | 0.26M | 6.3M | 19.2M |

**Projection overhead:** 0.26M params (1.4% increase over tied)

---

## 7. Testing Strategy

### Sanity Checks
1. **Untied model still works:** Set `tie_weights=False`, should behave identically to before
2. **Tied without projection still works:** Set `tie_weights=True, use_output_projection=False`
3. **Tied with projection trains:** Set `tie_weights=True, use_output_projection=True`

### Training Comparison
Run 3 experiments (same hyperparameters except architecture):
1. `tie_weights=False` (baseline untied)
2. `tie_weights=True, use_output_projection=False` (current slow learning)
3. `tie_weights=True, use_output_projection=True` (your hypothesis)

**Metrics to compare:**
- Training loss curves (first 1000 steps)
- Validation loss at checkpoints
- Learning speed (how fast loss decreases)
- Final model quality (base_eval scores)

---

## 8. Potential Issues & Debugging

### Issue: Output projection not being optimized
**Symptom:** Training loss doesn't improve, output_projection.weight stays near initialization
**Debug:** Print `model.output_projection.weight.abs().mean()` every 100 steps, verify it changes
**Fix:** Verify params are added to Muon optimizer correctly

### Issue: Gradient explosion
**Symptom:** Loss becomes NaN after a few steps
**Debug:** Add gradient norm logging for output_projection specifically
**Fix:** May need lower LR or gradient clipping (though existing grad_clip should handle it)

### Issue: Checkpoint loading fails
**Symptom:** Old checkpoints raise error about missing output_projection
**Debug:** Check that fallback logic in checkpoint_manager.py is triggered
**Fix:** Ensure `strict=True` is kept and fallback adds use_output_projection=False

---

## 9. Alternative Design: With Residual

If you change your mind about the residual, here's the modification:

**In forward() (line ~274-278):**
```python
if self.config.use_output_projection:
    # Apply projection WITH residual
    x_proj = self.output_projection(x)
    x = x + x_proj  # Residual connection
    logits = F.linear(x, self.transformer.wte.weight)
```

This would maintain gradient flow while still learning a transformation. But I understand you want pure transformation.

---

## Summary

**Files to modify:** 5
- nanochat/gpt.py (architecture)
- nanochat/checkpoint_manager.py (backward compat)
- scripts/base_train.py (training config)
- scripts/configure.py (wizard)
- CLAUDE.md (documentation, optional)

**Lines of code:** ~50-70 total across all files

**Risk level:** Low-Medium
- Low risk of breaking existing functionality (backward compatible)
- Medium risk that it doesn't solve the learning speed issue (hypothesis untested)

**Implementation time:** 30-60 minutes if careful

Good luck with the implementation! Let me know if you have questions or need clarification on any part.
