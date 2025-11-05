"""
Model size and training time calculator.

Provides functions to:
1. Calculate model parameters based on GPTConfig
2. Estimate FLOPs per token
3. Estimate training time on RTX 3090

This module is used by scripts/estimate_model_size.py and scripts/configure.py
"""


def calculate_model_size(config):
    """
    Calculate model parameters and architecture details from GPTConfig.

    Args:
        config: GPTConfig object

    Returns:
        Dictionary with architecture details and parameter counts
    """
    depth = config.n_layer
    model_dim = config.n_embd
    vocab_size = config.vocab_size
    num_heads = config.n_head
    num_kv_heads = config.n_kv_head
    head_dim = model_dim // num_heads

    # Calculate parameter counts
    # Embeddings
    embedding_params = vocab_size * model_dim

    # Unembedding (if not tied)
    if config.tie_weights:
        unembedding_params = 0
    else:
        unembedding_params = vocab_size * model_dim

    # Output projection (if used)
    if config.use_output_projection:
        output_projection_params = model_dim * model_dim
    else:
        output_projection_params = 0

    # Transformer layers
    # Each layer has:
    # - Attention: c_q, c_k, c_v, c_proj
    # - MLP: c_fc, c_proj

    # Attention parameters
    attn_c_q_params = model_dim * model_dim
    attn_c_k_params = model_dim * (num_kv_heads * head_dim)
    attn_c_v_params = model_dim * (num_kv_heads * head_dim)
    attn_c_proj_params = model_dim * model_dim
    attn_params_per_layer = attn_c_q_params + attn_c_k_params + attn_c_v_params + attn_c_proj_params

    # MLP parameters
    mlp_c_fc_params = model_dim * (4 * model_dim)
    mlp_c_proj_params = (4 * model_dim) * model_dim
    mlp_params_per_layer = mlp_c_fc_params + mlp_c_proj_params

    params_per_layer = attn_params_per_layer + mlp_params_per_layer
    transformer_params = depth * params_per_layer

    # Total parameters
    total_params = embedding_params + unembedding_params + output_projection_params + transformer_params

    return {
        'depth': depth,
        'model_dim': model_dim,
        'num_heads': num_heads,
        'num_kv_heads': num_kv_heads,
        'head_dim': head_dim,
        'vocab_size': vocab_size,
        'max_seq_len': config.sequence_len,
        'tie_weights': config.tie_weights,
        'use_output_projection': config.use_output_projection,
        'embedding_params': embedding_params,
        'unembedding_params': unembedding_params,
        'output_projection_params': output_projection_params,
        'transformer_params': transformer_params,
        'params_per_layer': params_per_layer,
        'total_params': total_params,
    }


def estimate_flops_per_token(config, total_params):
    """
    Estimate FLOPs per token for the model.

    Formula from: https://arxiv.org/abs/2204.02311

    Args:
        config: GPTConfig object
        total_params: Total parameter count

    Returns:
        FLOPs per token (float)
    """
    nparams_embedding = config.vocab_size * config.n_embd
    l = config.n_layer
    h = config.n_head
    q = config.n_embd // config.n_head
    t = config.sequence_len

    num_flops_per_token = 6 * (total_params - nparams_embedding) + 12 * l * h * q * t
    return num_flops_per_token


def estimate_training_time(config, total_params, target_param_data_ratio, total_batch_size):
    """
    Estimate training time on single RTX 3090.

    Args:
        config: GPTConfig object
        total_params: Total parameter count
        target_param_data_ratio: Tokens per parameter (Chinchilla = 20)
        total_batch_size: Total batch size in tokens

    Returns:
        Dictionary with training time estimates and data requirements
    """
    # RTX 3090 specs (Ampere architecture) - hardcoded
    rtx3090_peak_flops = 71e12  # BF16 Tensor Core peak: ~71 TFLOPS
    mfu = 0.40  # Model FLOPs Utilization (40%)
    overhead_minutes = 5  # Fixed overhead per run

    # Calculate training parameters
    target_tokens = total_params * target_param_data_ratio
    num_iterations = int(target_tokens // total_batch_size)

    # Calculate FLOPs per token
    num_flops_per_token = estimate_flops_per_token(config, total_params)

    # Total FLOPs for training
    total_flops = num_flops_per_token * target_tokens
    total_petaflops = total_flops / 1e15

    # Calculate training time
    effective_flops = rtx3090_peak_flops * mfu
    training_time_seconds = total_flops / effective_flops
    training_time_minutes = training_time_seconds / 60 + overhead_minutes
    training_time_hours = training_time_minutes / 60

    # Calculate data requirements
    # Assuming 4.8 chars/token compression and 250M chars per shard
    chars_per_token = 4.8
    chars_needed = target_tokens * chars_per_token
    chars_per_shard = 250e6
    num_shards_needed = int(chars_needed / chars_per_shard) + 1

    return {
        'target_tokens': int(target_tokens),
        'num_iterations': num_iterations,
        'num_flops_per_token': num_flops_per_token,
        'total_flops': total_flops,
        'total_petaflops': total_petaflops,
        'training_time_seconds': training_time_seconds,
        'training_time_minutes': training_time_minutes,
        'training_time_hours': training_time_hours,
        'num_shards_needed': num_shards_needed,
        'mfu': mfu,
        'overhead_minutes': overhead_minutes,
        'rtx3090_peak_tflops': rtx3090_peak_flops / 1e12,
    }
