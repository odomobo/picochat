"""
GPT model configuration.
"""

from dataclasses import dataclass


@dataclass
class GPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (MQA)
    n_embd: int = 768
    tie_weights: bool = False # tie wte and lm_head weights (reduces params by ~50%)
    use_output_projection: bool = False # use an output projection just before lm_head
    activation_fn: str = "relu_squared" # activation function: relu_squared, relu, gelu
    head_dim: int = 128 # attention head dimension
    ffn_expansion_ratio: float = 4.0 # MLP expansion ratio (intermediate_dim = n_embd * ffn_expansion_ratio)
