#!/usr/bin/env python3
"""
Calculate nanochat architecture for 38M parameter target with balanced distribution.

Target:
- Total: 38M parameters
- wte (embeddings): ~33%
- lm_head (unembedding): ~33%
- Transformer layers: ~33%

Usage:
    python calculate_38M_architecture.py
"""

def calculate_params(depth, model_dim, vocab_size, head_dim=64):
    """
    Calculate parameter breakdown for nanochat architecture.

    Based on nanochat/gpt.py architecture:
    - Token embedding (wte): vocab_size × model_dim
    - LM head (lm_head): model_dim × vocab_size
    - Each transformer layer has:
        - Attention: 4 linear layers (Q, K, V, output) each model_dim × model_dim
        - MLP: 2 linear layers with 4x expansion
    - No bias terms (stated in gpt.py:10)

    Note: Using head_dim=64 for tiny models (vs. 128 in standard nanochat)
    """

    # Embeddings
    wte_params = vocab_size * model_dim
    lm_head_params = model_dim * vocab_size

    # Each transformer layer
    # Attention: Q, K, V, output projection (each is model_dim × model_dim, no bias)
    attn_params_per_layer = 4 * model_dim * model_dim

    # MLP: fc (model_dim → 4*model_dim) + proj (4*model_dim → model_dim), no bias
    mlp_params_per_layer = model_dim * (4 * model_dim) + (4 * model_dim) * model_dim
    mlp_params_per_layer = 2 * 4 * model_dim * model_dim  # simplified

    params_per_layer = attn_params_per_layer + mlp_params_per_layer
    transformer_params = depth * params_per_layer

    # Total
    total_params = wte_params + lm_head_params + transformer_params

    # Calculate percentages
    wte_pct = 100 * wte_params / total_params if total_params > 0 else 0
    lm_head_pct = 100 * lm_head_params / total_params if total_params > 0 else 0
    transformer_pct = 100 * transformer_params / total_params if total_params > 0 else 0

    # Calculate num_heads (from base_train.py:90)
    num_heads = max(1, (model_dim + head_dim - 1) // head_dim)

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


def find_architectures_for_vocab(vocab_size, target_total=38e6, target_pct=33.33):
    """
    For a given vocab size, find depth and model_dim that hit our targets.

    Strategy:
    - For wte and lm_head to each be ~33%, vocab_size × model_dim should be ~33% of total
    - So: vocab_size × model_dim ≈ 0.33 × 38M ≈ 12.67M
    - This gives us model_dim
    - Then find depth such that transformer layers ≈ 33% of total
    """

    # Calculate required model_dim for embeddings to be ~33%
    embedding_target = target_pct / 100 * target_total
    model_dim_float = embedding_target / vocab_size

    # Round to multiple of head_dim (64 for tiny models) for clean architecture
    head_dim = 64
    model_dim = round(model_dim_float / head_dim) * head_dim

    if model_dim == 0:
        return []

    # Now find depths that give us close to target
    results = []

    for depth in range(1, 30):
        spec = calculate_params(depth, model_dim, vocab_size, head_dim=head_dim)

        # Check if we're close to target
        total_diff_pct = abs(spec['total_params'] - target_total) / target_total * 100

        # Check if distribution is balanced
        pct_diff = max(
            abs(spec['wte_pct'] - target_pct),
            abs(spec['lm_head_pct'] - target_pct),
            abs(spec['transformer_pct'] - target_pct)
        )

        # Store if reasonably close
        if total_diff_pct < 20 and pct_diff < 10:
            results.append({
                **spec,
                'total_diff_pct': total_diff_pct,
                'pct_diff': pct_diff,
                'balance_score': pct_diff + total_diff_pct  # lower is better
            })

    return results


def print_spec(spec, label=""):
    """Pretty print a model specification."""
    print(f"\n{label}")
    print("=" * 80)
    print(f"Architecture:")
    print(f"  --depth={spec['depth']} --vocab_size={spec['vocab_size']}")
    print(f"  Layers: {spec['depth']}, Model dim: {spec['model_dim']}, Num heads: {spec['num_heads']}, Head dim: {spec['head_dim']}")
    print(f"\nParameter Breakdown:")
    print(f"  Token embedding (wte):    {spec['wte_params']:>12,} params ({spec['wte_pct']:>5.2f}%)")
    print(f"  Transformer layers:       {spec['transformer_params']:>12,} params ({spec['transformer_pct']:>5.2f}%)")
    print(f"    └─ {spec['params_per_layer']:,} params/layer × {spec['depth']} layers")
    print(f"  LM head (lm_head):        {spec['lm_head_params']:>12,} params ({spec['lm_head_pct']:>5.2f}%)")
    print(f"  {'─' * 76}")
    print(f"  TOTAL:                    {spec['total_params']:>12,} params (100.00%)")

    if 'total_diff_pct' in spec:
        print(f"\nTarget Accuracy:")
        print(f"  Total params vs 38M target: {spec['total_diff_pct']:>5.2f}% difference")
        print(f"  Max deviation from 33.33%:  {spec['pct_diff']:>5.2f}% points")
        print(f"  Balance score (lower=better): {spec['balance_score']:.2f}")


def main():
    print("=" * 80)
    print("nanochat 38M Parameter Architecture Calculator (64-dim heads)")
    print("=" * 80)
    print(f"\nTarget: 38M total parameters with 33.33% in each of wte/lm_head/transformer")
    print(f"\nNote: Using head_dim=64 (vs. 128 in standard nanochat) for tiny models")
    print(f"      This allows more flexibility in model_dim choices")
    print(f"\nReference: speedrun.sh uses vocab_size=65536")
    print(f"           But we may want smaller vocab for better parameter distribution")

    # Test different vocabulary sizes
    vocab_sizes = [
        4096,   # 2^12
        8192,   # 2^13
        16384,  # 2^14
        24576,  # 3 * 2^13
        32768,  # 2^15
        65536,  # 2^16 (nanochat default)
    ]

    all_results = []

    for vocab_size in vocab_sizes:
        results = find_architectures_for_vocab(vocab_size)
        if results:
            # Sort by balance score
            results.sort(key=lambda x: x['balance_score'])
            all_results.extend(results[:3])  # Keep top 3 for each vocab size

    # Sort all results by balance score
    all_results.sort(key=lambda x: x['balance_score'])

    print("\n" + "=" * 80)
    print("TOP 5 MOST BALANCED ARCHITECTURES")
    print("=" * 80)

    for i, spec in enumerate(all_results[:5], 1):
        print_spec(spec, f"#{i} - Best Match" if i == 1 else f"#{i}")

    print("\n" + "=" * 80)
    print("COMPARISON BY VOCABULARY SIZE")
    print("=" * 80)

    for vocab_size in vocab_sizes:
        matching = [r for r in all_results if r['vocab_size'] == vocab_size]
        if matching:
            best = matching[0]
            print(f"\nVocab {vocab_size:>6} (2^{vocab_size.bit_length()-1}): "
                  f"depth={best['depth']:>2}, dim={best['model_dim']:>4}, "
                  f"heads={best['num_heads']:>2} | "
                  f"{best['total_params']/1e6:.1f}M params | "
                  f"balance={best['balance_score']:.1f}")

    print("\n" + "=" * 80)
    print("RECOMMENDED CONFIGURATION")
    print("=" * 80)

    best = all_results[0]
    print_spec(best, "🎯 BEST BALANCED ARCHITECTURE")

    print(f"\nTo use this configuration in nanochat:")
    print(f"  1. Train tokenizer with vocab_size={best['vocab_size']}")
    print(f"     python -m scripts.tok_train --vocab_size={best['vocab_size']}")
    print(f"  2. Train model:")
    print(f"     torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \\")
    print(f"       --depth={best['depth']} --vocab_size={best['vocab_size']}")

    # Calculate training requirements
    chinchilla_tokens = 20 * best['total_params']
    chars_needed = chinchilla_tokens * 4.8  # assume 4.8 chars/token
    shards_needed = int(chars_needed / 250e6) + 1

    print(f"\nTraining data requirements (Chinchilla 20:1):")
    print(f"  Tokens: {chinchilla_tokens/1e6:.0f}M ({chinchilla_tokens/1e9:.2f}B)")
    print(f"  Characters: {chars_needed/1e9:.1f}B")
    print(f"  Data shards: ~{shards_needed} (@ 250M chars/shard)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
