"""
Configuration wizard for nanochat training runs.

Creates a config.py file in the current NANOCHAT_RUN_DIR with training parameters.
Asks interactive questions with sensible defaults and performs basic validation.

Usage:
    python -m scripts.configure
"""

import os
import sys

from nanochat.gpt_config import GPTConfig
from nanochat.model_calculator import calculate_model_size, estimate_training_time


def validate_batch_size(device_batch_size, total_batch_size, max_seq_len):
    """
    Validate that total_batch_size is divisible by (device_batch_size * max_seq_len).
    Returns True if valid, False otherwise.
    """
    tokens_per_fwdbwd = device_batch_size * max_seq_len
    return total_batch_size % tokens_per_fwdbwd == 0


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


def get_float_input(prompt, default):
    """Get float input with default value."""
    while True:
        response = input(f"{prompt} [default: {default}]: ").strip()
        if not response:
            return default
        try:
            return float(response)
        except ValueError:
            print(f"  ERROR: Please enter a valid number")


def get_string_input(prompt, default):
    """Get string input with default value."""
    response = input(f"{prompt} [default: {default}]: ").strip()
    if not response:
        return default
    return response


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
    """Run the interactive configuration wizard."""

    # Check that NANOCHAT_RUN_DIR is set
    run_dir = os.environ.get("NANOCHAT_RUN_DIR")
    if not run_dir:
        print("ERROR: NANOCHAT_RUN_DIR environment variable is not set")
        print("Please run: source scripts/run_init.sh <run_name>")
        sys.exit(1)

    # Check if config.py already exists
    config_path = os.path.join(run_dir, "config.py")
    if os.path.exists(config_path):
        print(f"ERROR: Configuration file already exists: {config_path}")
        print("This run has already been configured.")
        print("To reconfigure, delete the existing config file or choose a different run name.")
        sys.exit(1)

    print("=" * 80)
    print("nanochat Configuration Wizard")
    print("=" * 80)
    print(f"Creating configuration for: {run_dir}")
    print()
    print("This wizard will ask questions about your training run.")
    print("Press Enter to accept defaults shown in brackets.")
    print("=" * 80)
    print()

    # Get run name from environment or ask user
    wandb_run = os.environ.get("WANDB_RUN")
    if wandb_run:
        print(f"Using run name from environment: {wandb_run}")
        run_name = wandb_run
    else:
        run_name = get_string_input("Run name (for wandb logging)", default="dummy")
    print()

    # Corpus selection - scan base_data directory for available corpora
    data_dir = os.environ.get("NANOCHAT_DATA_DIR")
    if not data_dir:
        print("ERROR: NANOCHAT_DATA_DIR environment variable is not set")
        print("Please run: source scripts/run_init.sh <run_name>")
        sys.exit(1)

    base_data_dir = os.path.join(data_dir, "base_data")
    if not os.path.exists(base_data_dir):
        print(f"ERROR: Base data directory not found: {base_data_dir}")
        print("Please download or organize pretraining data first.")
        sys.exit(1)

    # Find all subdirectories containing .parquet files
    available_corpora = []
    for item in os.listdir(base_data_dir):
        corpus_path = os.path.join(base_data_dir, item)
        if os.path.isdir(corpus_path):
            # Check if this directory contains any .parquet files
            parquet_files = [f for f in os.listdir(corpus_path) if f.endswith('.parquet')]
            if parquet_files:
                available_corpora.append((item, len(parquet_files)))

    if not available_corpora:
        print(f"ERROR: No corpora found in {base_data_dir}")
        print("Expected structure: {base_data_dir}/{{corpus_name}}/*.parquet")
        print("Please organize your pretraining data into corpus subdirectories.")
        sys.exit(1)

    # Present corpus options
    print("Available pretraining corpora:")
    for idx, (corpus_name, num_shards) in enumerate(available_corpora, 1):
        print(f"  {idx}. {corpus_name} ({num_shards} shards)")
    print()

    # Select corpus
    if len(available_corpora) == 1:
        corpus_name = available_corpora[0][0]
        print(f"Auto-selected corpus: {corpus_name}")
    else:
        while True:
            response = input(f"Select corpus [1-{len(available_corpora)}]: ").strip()
            try:
                selection = int(response)
                if 1 <= selection <= len(available_corpora):
                    corpus_name = available_corpora[selection - 1][0]
                    print(f"Selected corpus: {corpus_name}")
                    break
                else:
                    print(f"  ERROR: Please enter a number between 1 and {len(available_corpora)}")
            except ValueError:
                print(f"  ERROR: Please enter a valid number")
    print()

    # Ask questions with defaults
    depth = get_int_input("Model depth (number of transformer layers)", default=4)
    default_model_dim = 512
    model_dim = get_int_input(f"Model dim (embedding dimension)", default=default_model_dim)
    tie_weights = get_bool_input("Tie embedding weights (wte and lm_head)? Reduces params by ~50%", default=True)

    # Architecture customization
    print()
    print("Advanced architecture options:")
    activation_fn = get_string_input("Activation function (relu_squared, relu, gelu)", default="relu_squared")
    if activation_fn not in ["relu_squared", "relu", "gelu"]:
        print(f"WARNING: Unknown activation function '{activation_fn}'. Proceeding anyway, but this may cause errors.")

    head_dim = get_int_input("Attention head dimension", default=128)
    ffn_expansion_ratio = get_float_input("FFN expansion ratio (intermediate_dim = model_dim * ratio)", default=4.0)

    # Warn if intermediate dimension isn't a multiple of 128
    intermediate_dim = int(model_dim * ffn_expansion_ratio)
    if intermediate_dim % 128 != 0:
        print(f"WARNING: FFN intermediate dimension ({intermediate_dim}) is not a multiple of 128.")
        print(f"         This may cause GPU inefficiencies. Consider adjusting model_dim or ffn_expansion_ratio.")
        print(f"         Suggested: model_dim={model_dim}, ffn_expansion_ratio={128 * round(intermediate_dim / 128) / model_dim:.2f}")
    print()

    # Only ask for tied_weights_lr if tie_weights is enabled
    if tie_weights:
        tied_weights_lr = get_float_input("Learning rate for tied weights (standard: 0.02)", default=0.02)
    else:
        tied_weights_lr = 0.02  # Default value, unused when tie_weights=False

    # only ask for use_output_projection if tie_weights is enabled; we can use this if it's not, but it's not that useful otherwise
    if tie_weights:
        use_output_projection = get_bool_input("Use output projection matrix? (before tied lm_head)? Probably really useful!", default=True)
    else:
        use_output_projection = False

    device_batch_size = get_int_input("Device batch size (sequences per GPU)", default=32)
    target_param_data_ratio = get_int_input("Target param:data ratio (Chinchilla=20, -1=explicit iterations)", default=20)
    total_batch_size = get_int_input("Total batch size (tokens)", default=524288)
    max_seq_len = get_int_input("Max sequence length (context window)", default=2048)

    print()
    print("=" * 80)
    print("Validating configuration...")
    print("=" * 80)

    # Validate batch size divisibility
    if not validate_batch_size(device_batch_size, total_batch_size, max_seq_len):
        tokens_per_fwdbwd = device_batch_size * max_seq_len
        print()
        print(f"ERROR: Invalid batch size configuration!")
        print(f"  total_batch_size ({total_batch_size}) must be divisible by")
        print(f"  (device_batch_size * max_seq_len) = ({device_batch_size} * {max_seq_len}) = {tokens_per_fwdbwd}")
        print()
        print(f"Suggestion: Adjust device_batch_size to one of these values:")
        # Find divisors
        divisors = []
        target = total_batch_size // max_seq_len
        for i in range(1, min(target + 1, 129)):  # Cap at 128 for sanity
            if target % i == 0:
                divisors.append(i)
        print(f"  Valid device_batch_size values: {divisors}")
        sys.exit(1)

    # Calculate derived architecture values and model size
    vocab_size = 24576  # 3 * 2^13, appropriate for tiny models
    num_heads = max(1, (model_dim + head_dim - 1) // head_dim)  # ceiling division

    # Create GPTConfig and calculate model size
    config = GPTConfig(
        sequence_len=max_seq_len,
        vocab_size=vocab_size,
        n_layer=depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        tie_weights=tie_weights,
        use_output_projection=use_output_projection,
        activation_fn=activation_fn,
        head_dim=head_dim,
        ffn_expansion_ratio=ffn_expansion_ratio
    )
    model_info = calculate_model_size(config)

    # Extract for easier access
    total_params = model_info['total_params']
    wte_params = model_info['embedding_params']
    lm_head_params = model_info['unembedding_params']
    output_projection_params = model_info['output_projection_params']
    transformer_params = model_info['transformer_params']
    

    print()
    print("Derived architecture:")
    print(f"  Depth: {depth}")
    print(f"  Model dim: {model_dim}")
    print(f"  Num heads: {num_heads}")
    print(f"  Head dim: {head_dim}")
    print(f"  Activation: {activation_fn}")
    print(f"  FFN expansion ratio: {ffn_expansion_ratio}x (intermediate dim: {intermediate_dim})")
    print(f"  Weight tying: {'enabled' if tie_weights else 'disabled'}")
    print(f"  Output projection: {'enabled' if use_output_projection else 'disabled'}")
    print()
    print("Parameter count:")
    if tie_weights:
        print(f"  wte (tied with lm_head):     {wte_params:>12,} ({wte_params/1e6:>6.2f}M)")
    else:
        print(f"  wte (embeddings):            {wte_params:>12,} ({wte_params/1e6:>6.2f}M)")
        print(f"  lm_head (unembedding):       {lm_head_params:>12,} ({lm_head_params/1e6:>6.2f}M)")
    if use_output_projection:
        print(f"  Output projection:           {output_projection_params:>12,} ({output_projection_params/1e3:>6.2f}K)")
    print(f"  Transformer layers:          {transformer_params:>12,} ({transformer_params/1e6:>6.2f}M)")
    print(f"  {'─' * 50}")
    print(f"  Total:                       {total_params:>12,} ({total_params/1e6:>6.2f}M)")
    if tie_weights:
        print(f"  Effective (tied wte x2):     {model_info['effective_params']:>12,} ({model_info['effective_params']/1e6:>6.2f}M)")
    print()

    # Display batch info
    tokens_per_fwdbwd = device_batch_size * max_seq_len
    grad_accum_steps = total_batch_size // tokens_per_fwdbwd
    print("Batch configuration:")
    print(f"  Tokens per forward/backward: {tokens_per_fwdbwd:,}")
    print(f"  Gradient accumulation steps: {grad_accum_steps}")
    print()

    # Training time estimate
    training_info = estimate_training_time(
        config=config,
        effective_params=model_info['effective_params'],
        target_param_data_ratio=target_param_data_ratio,
        total_batch_size=total_batch_size
    )
    total_minutes = int(training_info['training_time_seconds'] / 60)
    hours = total_minutes // 60
    minutes = total_minutes % 60
    print("Training time estimate (RTX 3090):")
    print(f"  Tokens:Effective Params ratio: {target_param_data_ratio}:1")
    print(f"  Target tokens:               {training_info['target_tokens']:>12,} ({training_info['target_tokens']/1e9:>6.2f}B)")
    print(f"  Num iterations:              {training_info['num_iterations']:>12,}")
    print(f"  Total PetaFLOPs:             {training_info['total_petaflops']:>12.2f}")
    print(f"  Estimated training time:     {hours}:{minutes:02d}")
    print()

    # Generate config file content
    config_content = f"""# nanochat training configuration
# Generated by configuration wizard
# Run directory: {run_dir}

# Run identification
run = "{run_name}"  # wandb run name ("dummy" disables wandb logging)

# Data
corpus_name = "{corpus_name}"  # pretraining corpus subdirectory in base_data/

# Model architecture
depth = {depth}
model_dim = {model_dim}  # aspect ratio {model_dim / depth if depth != 0 else 0:.1f} (or custom)
max_seq_len = {max_seq_len}
tie_weights = {str(tie_weights)}  # tie wte and lm_head weights (reduces params by ~50%)
activation_fn = "{activation_fn}"  # activation function: relu_squared, relu, gelu
head_dim = {head_dim}  # attention head dimension
ffn_expansion_ratio = {ffn_expansion_ratio}  # MLP expansion ratio

# Optimization
device_batch_size = {device_batch_size}
total_batch_size = {total_batch_size}
tied_weights_lr = {tied_weights_lr}  # learning rate for tied weights (when tie_weights=True)
use_output_projection = {use_output_projection}

# Training horizon
# Only one of (num_iterations, target_flops, target_param_data_ratio) will be used
# Priority: num_iterations > target_flops > target_param_data_ratio
num_iterations = -1  # explicit number of steps (-1 = use target_param_data_ratio)
target_flops = -1.0  # calculate iterations to reach target FLOPs (-1 = disabled)
target_param_data_ratio = {target_param_data_ratio}  # Chinchilla scaling (20 = 20 tokens per effective parameter)

# Derived values (calculated from config above):
# num_heads = max(1, (model_dim + head_dim - 1) // head_dim) = {num_heads}
# intermediate_dim = int(model_dim * ffn_expansion_ratio) = {intermediate_dim}
# tokens_per_fwdbwd = device_batch_size * max_seq_len = {tokens_per_fwdbwd:,}
# grad_accum_steps = total_batch_size // tokens_per_fwdbwd = {grad_accum_steps}
"""

    # Write config file
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        f.write(config_content)

    print("=" * 80)
    print(f"✓ Configuration saved to: {config_path}")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  1. Review the configuration file if needed")
    print("  2. Run: bash start_run.sh")
    print()


if __name__ == "__main__":
    main()
