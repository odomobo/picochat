"""
Model size and training time estimation wizard.

Interactive wizard to estimate model parameters and training time on RTX 3090.

Usage:
    python -m scripts.estimate_model_size
"""

from nanochat.gpt_config import GPTConfig
from nanochat.model_calculator import calculate_model_size, estimate_training_time


def get_int_input(prompt, default):
    """Get integer input with default value."""
    while True:
        response = input(f"{prompt} [default: {default}]: ").strip()
        if not response:
            return default
        try:
            return int(response)
        except ValueError:
            print(f"  ERROR: Please enter a valid integer")


def get_bool_input(prompt, default):
    """Get boolean input with default value (y/n)."""
    default_str = "y" if default else "n"
    while True:
        response = input(f"{prompt} (y/n) [default: {default_str}]: ").strip().lower()
        if not response:
            return default
        if response in ['y', 'yes']:
            return True
        if response in ['n', 'no']:
            return False
        print(f"  ERROR: Please enter 'y' or 'n'")


def main():
    """Run the interactive model size estimation wizard."""

    print()
    print("=" * 80)
    print("Model Size & Training Time Estimation Wizard")
    print("=" * 80)
    print()
    print("This wizard will estimate model parameters and training time on RTX 3090.")
    print("Press Enter to accept defaults shown in brackets.")
    print("=" * 80)
    print()

    # Get architecture parameters
    depth = get_int_input("Model depth (number of transformer layers)", default=4)
    model_dim = get_int_input("Model dim (embedding dimension)", default=512)
    vocab_size = get_int_input("Vocab size", default=24576)
    max_seq_len = get_int_input("Max sequence length (context window)", default=2048)
    tie_weights = get_bool_input("Tie embedding weights (wte and lm_head)?", default=True)
    use_output_projection = get_bool_input("Use output projection matrix?", default=True)

    print()
    # Get training parameters
    target_param_data_ratio = get_int_input("Target param:data ratio (Chinchilla=20)", default=20)
    total_batch_size = get_int_input("Total batch size (tokens)", default=524288)

    # Create config
    num_heads = max(1, (model_dim + 127) // 128)
    config = GPTConfig(
        sequence_len=max_seq_len,
        vocab_size=vocab_size,
        n_layer=depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        tie_weights=tie_weights,
        use_output_projection=use_output_projection
    )

    # Calculate model size
    model_info = calculate_model_size(config)

    # Estimate training time
    training_info = estimate_training_time(
        config=config,
        effective_params=model_info['effective_params'],
        transformer_params=model_info['transformer_params'],
        target_param_data_ratio=target_param_data_ratio,
        total_batch_size=total_batch_size
    )

    # Print results
    print()
    print("=" * 80)
    print("MODEL SIZE CALCULATION")
    print("=" * 80)
    print()
    print("Architecture:")
    print(f"  Depth:                       {model_info['depth']}")
    print(f"  Model dim:                   {model_info['model_dim']:,}")
    print(f"  Num heads:                   {model_info['num_heads']}")
    print(f"  Head dim:                    {model_info['head_dim']}")
    print(f"  Vocab size:                  {model_info['vocab_size']:,}")
    print(f"  Max seq len:                 {model_info['max_seq_len']:,}")
    print(f"  Weight tying:                {'enabled' if model_info['tie_weights'] else 'disabled'}")
    print(f"  Output projection:           {'enabled' if model_info['use_output_projection'] else 'disabled'}")
    print()
    print("Parameter count:")
    if model_info['tie_weights']:
        print(f"  wte (tied with lm_head):     {model_info['embedding_params']:>12,} ({model_info['embedding_params']/1e6:>6.2f}M)")
    else:
        print(f"  wte (embeddings):            {model_info['embedding_params']:>12,} ({model_info['embedding_params']/1e6:>6.2f}M)")
        print(f"  lm_head (unembedding):       {model_info['unembedding_params']:>12,} ({model_info['unembedding_params']/1e6:>6.2f}M)")
    if model_info['use_output_projection']:
        print(f"  Output projection:           {model_info['output_projection_params']:>12,} ({model_info['output_projection_params']/1e6:>6.2f}M)")
    print(f"  Transformer layers:          {model_info['transformer_params']:>12,} ({model_info['transformer_params']/1e6:>6.2f}M)")
    print(f"    (Per layer:                {model_info['params_per_layer']:>12,} ({model_info['params_per_layer']/1e6:>6.2f}M))")
    print(f"  {'─' * 50}")
    print(f"  Total:                       {model_info['total_params']:>12,} ({model_info['total_params']/1e6:>6.2f}M)")
    if model_info['tie_weights']:
        print(f"  Effective (tied wte x2):     {model_info['effective_params']:>12,} ({model_info['effective_params']/1e6:>6.2f}M)")
    print()

    print("=" * 80)
    print("TRAINING TIME ESTIMATION (RTX 3090)")
    print("=" * 80)
    print()
    print("Training configuration:")
    print(f"  Tokens:Params ratio:         {target_param_data_ratio}:1")
    print(f"  Target tokens:               {training_info['target_tokens']:>12,} ({training_info['target_tokens']/1e9:>6.2f}B)")
    print(f"  Num iterations:              {training_info['num_iterations']:>12,}")
    print(f"  Total batch size:            {total_batch_size:>12,} tokens")
    print()
    print("Compute requirements:")
    print(f"  FLOPs per token:             {training_info['num_flops_per_token']:>12.2e}")
    print(f"  Total FLOPs:                 {training_info['total_flops']:>12.2e}")
    print(f"  Total PetaFLOPs:             {training_info['total_petaflops']:>12.2f}")
    print()
    # Calculate hours and minutes for display
    total_minutes = int(training_info['training_time_seconds'] / 60)
    hours = total_minutes // 60
    minutes = total_minutes % 60

    print("Hardware & timing:")
    print(f"  Hardware:                    RTX 3090")
    print(f"  Peak performance:            {training_info['rtx3090_peak_tflops']:.1f} TFLOPS (BF16)")
    print(f"  Model FLOPs Utilization:     {training_info['mfu']*100:.1f}%")
    print(f"  Overhead:                    {training_info['overhead_minutes']} minutes")
    print(f"  Estimated training time:     {hours}:{minutes:02d}")
    print()
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
