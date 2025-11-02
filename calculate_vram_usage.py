#!/usr/bin/env python3
"""
Calculate VRAM usage for nanochat training.

Usage:
    python calculate_vram_usage.py
"""

def calculate_vram(batch_size, seq_len=2048, vocab_size=24576, depth=4, model_dim=512, num_heads=8):
    """
    Calculate VRAM usage for training a transformer model.

    Based on:
    - Model parameters (stored in bf16)
    - Optimizer states (stored in fp32)
    - Gradients (stored in bf16)
    - Activations (stored in bf16, depends on batch size)
    """

    print("=" * 80)
    print(f"VRAM Calculation for Target Architecture")
    print("=" * 80)
    print(f"\nModel Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Depth (layers): {depth}")
    print(f"  Model dimension: {model_dim}")
    print(f"  Number of heads: {num_heads}")
    print(f"  Head dimension: {model_dim // num_heads}")

    # Calculate parameter counts
    wte_params = vocab_size * model_dim
    lm_head_params = model_dim * vocab_size

    # Transformer layer params
    attn_params_per_layer = 4 * model_dim * model_dim  # Q, K, V, output
    mlp_params_per_layer = 2 * 4 * model_dim * model_dim  # fc + proj (4x expansion)
    params_per_layer = attn_params_per_layer + mlp_params_per_layer
    transformer_params = depth * params_per_layer

    total_params = wte_params + lm_head_params + transformer_params

    print(f"\nParameter Breakdown:")
    print(f"  Token embedding (wte):  {wte_params:,} ({wte_params/1e6:.2f}M)")
    print(f"  Transformer layers:     {transformer_params:,} ({transformer_params/1e6:.2f}M)")
    print(f"  LM head (lm_head):      {lm_head_params:,} ({lm_head_params/1e6:.2f}M)")
    print(f"  TOTAL:                  {total_params:,} ({total_params/1e6:.2f}M)")

    # Memory calculations (in bytes)
    print(f"\n{'─' * 80}")
    print(f"VRAM Usage Breakdown:")
    print(f"{'─' * 80}")

    # 1. Model parameters (bf16 = 2 bytes)
    model_memory = total_params * 2
    print(f"\n1. Model Parameters (bf16):")
    print(f"   {total_params:,} params × 2 bytes = {model_memory / 1e6:.1f} MB")

    # 2. Optimizer states
    # AdamW for wte + lm_head (2 states: momentum + variance, fp32 = 4 bytes each)
    # Muon for transformer (1 state: momentum, fp32 = 4 bytes)
    adamw_params = wte_params + lm_head_params
    muon_params = transformer_params

    adamw_memory = adamw_params * 2 * 4  # 2 states × 4 bytes
    muon_memory = muon_params * 1 * 4    # 1 state × 4 bytes
    optimizer_memory = adamw_memory + muon_memory

    print(f"\n2. Optimizer States (fp32):")
    print(f"   AdamW (wte + lm_head): {adamw_params:,} params × 2 states × 4 bytes = {adamw_memory / 1e6:.1f} MB")
    print(f"   Muon (transformer):    {muon_params:,} params × 1 state × 4 bytes = {muon_memory / 1e6:.1f} MB")
    print(f"   Total: {optimizer_memory / 1e6:.1f} MB")

    # 3. Gradients (bf16 = 2 bytes)
    gradient_memory = total_params * 2
    print(f"\n3. Gradients (bf16):")
    print(f"   {total_params:,} params × 2 bytes = {gradient_memory / 1e6:.1f} MB")

    # 4. Activations (the big variable part)
    # This is an approximation based on transformer memory patterns
    # Main contributors:
    # - Input embeddings: B × T × D
    # - Attention QKV: 3 × B × T × D per layer
    # - Attention scores: B × H × T × T per layer (can be large!)
    # - Attention output: B × T × D per layer
    # - MLP intermediate: B × T × 4D per layer
    # - Residuals and layer norms

    B, T, D, H = batch_size, seq_len, model_dim, num_heads

    # Per layer activations (approximate)
    per_layer_activations = (
        3 * B * T * D +           # QKV projections
        B * H * T * T +           # Attention scores (the killer!)
        B * T * D +               # Attention output
        B * T * 4 * D +           # MLP intermediate (4x expansion)
        B * T * D * 2             # Residuals and norms (approx)
    )

    # Input embedding
    embedding_activation = B * T * D

    # Total activations across all layers
    activation_memory = (embedding_activation + depth * per_layer_activations) * 2  # bf16 = 2 bytes

    # Add some overhead for PyTorch internals, temporary buffers, etc. (20%)
    activation_memory *= 1.2

    print(f"\n4. Activations (bf16, batch_size={batch_size}):")
    print(f"   Input embedding:        {embedding_activation * 2 / 1e6:.1f} MB")
    print(f"   Per-layer activations:  {per_layer_activations * 2 / 1e6:.1f} MB × {depth} layers")
    print(f"   Attention scores alone: {B * H * T * T * 2 * depth / 1e6:.1f} MB (often the bottleneck!)")
    print(f"   Total (with 20% overhead): {activation_memory / 1e6:.1f} MB")

    # Total VRAM
    total_vram = model_memory + optimizer_memory + gradient_memory + activation_memory

    print(f"\n{'═' * 80}")
    print(f"TOTAL VRAM USAGE: {total_vram / 1e9:.2f} GB")
    print(f"{'═' * 80}")

    return total_vram


def main():
    # Target architecture from CHECKPOINT.md
    print("\nTarget Architecture: depth=4, vocab=24576, dim=512, heads=8")
    print("\nNote: This assumes training mode with gradient computation enabled.")
    print("      Inference mode uses much less memory (no optimizer, no gradients).\n")

    # Test different batch sizes
    batch_sizes = [1, 2, 4, 8, 16, 32]
    available_vram = 24 * 1e9  # 24 GB for RTX 3090

    print("\n" + "=" * 80)
    print("VRAM Usage vs Batch Size")
    print("=" * 80)

    results = []
    for bs in batch_sizes:
        vram = calculate_vram(bs, seq_len=2048, vocab_size=24576, depth=4, model_dim=512, num_heads=8)
        results.append((bs, vram))
        print("\n")

    print("=" * 80)
    print("Summary Table")
    print("=" * 80)
    print(f"\n{'Batch Size':<12} {'VRAM (GB)':<12} {'Fits in 24GB?':<15} {'Utilization':<15}")
    print("-" * 60)

    for bs, vram in results:
        vram_gb = vram / 1e9
        fits = "✓ YES" if vram < available_vram else "✗ NO"
        utilization = f"{100 * vram / available_vram:.1f}%"
        print(f"{bs:<12} {vram_gb:<12.2f} {fits:<15} {utilization:<15}")

    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    # Find the largest batch size that fits
    max_fitting_bs = None
    for bs, vram in results:
        if vram < available_vram * 0.9:  # Use 90% to leave some headroom
            max_fitting_bs = bs

    if max_fitting_bs:
        print(f"\nRecommended batch size: {max_fitting_bs}")
        print(f"  - Fits comfortably in 24GB VRAM")
        print(f"  - Leaves ~10% headroom for PyTorch overhead")
        print(f"\nYou can try higher batch sizes if needed, but monitor VRAM closely.")
    else:
        print("\nWARNING: Even batch_size=1 might be tight!")
        print("Consider:")
        print("  - Reducing sequence length (--max_seq_len)")
        print("  - Using gradient checkpointing (if implemented)")

    print(f"\nFor ittybitty.sh, use: --device_batch_size={max_fitting_bs}")
    print()


if __name__ == "__main__":
    main()
