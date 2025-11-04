#!/usr/bin/env python3
"""
Calculate the exact model size for the speedrun configuration (depth=20).
Uses the same architecture derivation as base_train.py
"""

def calculate_params(depth, vocab_size):
    """
    Calculate parameter breakdown for nanochat architecture.
    Uses the same formulas as base_train.py and gpt.py
    """

    # Architecture derivation from base_train.py
    num_layers = depth
    model_dim = depth * 64  # aspect ratio 64
    num_heads = max(1, (model_dim + 127) // 128)  # head dim 128 (ceil division)
    head_dim = 128

    # Embeddings (wte and lm_head)
    wte_params = vocab_size * model_dim
    lm_head_params = model_dim * vocab_size

    # Each transformer layer (no bias terms, from gpt.py)
    # Attention: Q, K, V, output projection (each is model_dim × model_dim)
    attn_params_per_layer = 4 * model_dim * model_dim

    # MLP: fc (model_dim → 4*model_dim) + proj (4*model_dim → model_dim)
    mlp_params_per_layer = model_dim * (4 * model_dim) + (4 * model_dim) * model_dim
    mlp_params_per_layer = 2 * 4 * model_dim * model_dim  # simplified

    params_per_layer = attn_params_per_layer + mlp_params_per_layer
    transformer_params = num_layers * params_per_layer

    # Total
    total_params = wte_params + lm_head_params + transformer_params

    # Calculate percentages
    wte_pct = 100 * wte_params / total_params
    lm_head_pct = 100 * lm_head_params / total_params
    transformer_pct = 100 * transformer_params / total_params

    return {
        'depth': depth,
        'model_dim': model_dim,
        'vocab_size': vocab_size,
        'num_heads': num_heads,
        'head_dim': head_dim,
        'wte_params': wte_params,
        'lm_head_params': lm_head_params,
        'transformer_params': transformer_params,
        'params_per_layer': params_per_layer,
        'total_params': total_params,
        'wte_pct': wte_pct,
        'lm_head_pct': lm_head_pct,
        'transformer_pct': transformer_pct,
    }


def print_spec(spec):
    """Pretty print a model specification."""
    print("\n" + "=" * 80)
    print("SPEEDRUN CONFIGURATION (depth=20)")
    print("=" * 80)
    print(f"\nArchitecture:")
    print(f"  Layers: {spec['depth']}")
    print(f"  Model dim: {spec['model_dim']}")
    print(f"  Num heads: {spec['num_heads']}")
    print(f"  Head dim: {spec['head_dim']}")
    print(f"  Vocab size: {spec['vocab_size']:,}")

    print(f"\nParameter Breakdown:")
    print(f"  Token embedding (wte):    {spec['wte_params']:>15,} params ({spec['wte_pct']:>5.2f}%)")
    print(f"  Transformer layers:       {spec['transformer_params']:>15,} params ({spec['transformer_pct']:>5.2f}%)")
    print(f"    └─ {spec['params_per_layer']:,} params/layer × {spec['depth']} layers")
    print(f"  LM head (lm_head):        {spec['lm_head_params']:>15,} params ({spec['lm_head_pct']:>5.2f}%)")
    print(f"  {'─' * 78}")
    print(f"  TOTAL:                    {spec['total_params']:>15,} params (100.00%)")
    print(f"  TOTAL (millions):         {spec['total_params']/1e6:>15.1f}M")
    print("=" * 80)


if __name__ == "__main__":
    # Speedrun configuration
    depth = 20
    vocab_size = 65536  # 2^16, default from tok_train.py

    spec = calculate_params(depth, vocab_size)
    print_spec(spec)
