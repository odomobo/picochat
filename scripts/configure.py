"""
Configuration wizard for nanochat training runs.

Creates a config.py file in the current NANOCHAT_RUN_DIR with training parameters.
Asks interactive questions with sensible defaults and performs basic validation.

Usage:
    python -m scripts.configure
"""

import os
import sys


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


def get_string_input(prompt, default):
    """Get string input with default value."""
    response = input(f"{prompt} [default: {default}]: ").strip()
    if not response:
        return default
    return response


def main():
    """Run the interactive configuration wizard."""

    # Check that NANOCHAT_RUN_DIR is set
    run_dir = os.environ.get("NANOCHAT_RUN_DIR")
    if not run_dir:
        print("ERROR: NANOCHAT_RUN_DIR environment variable is not set")
        print("Please run: source init_run.sh <run_name>")
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

    # Ask questions with defaults
    depth = get_int_input("Model depth (number of transformer layers)", default=4)
    default_model_dim = 256
    model_dim = get_int_input(f"Model dim (embedding dimension)", default=default_model_dim)
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

    # Calculate derived architecture values
    num_heads = max(1, (model_dim + 127) // 128)

    # Calculate parameter counts
    vocab_size = 24576  # 3 * 2^13, appropriate for tiny models
    wte_params = vocab_size * model_dim
    lm_head_params = vocab_size * model_dim
    # Per layer: attention (4 * model_dim^2) + MLP (8 * model_dim^2) = 12 * model_dim^2
    transformer_params = depth * 12 * model_dim * model_dim
    total_params = wte_params + transformer_params + lm_head_params

    print()
    print("Derived architecture:")
    print(f"  Depth: {depth}")
    print(f"  Model dim: {model_dim}")
    print(f"  Num heads: {num_heads}")
    print(f"  Head dim: 128")
    print()
    print("Parameter count:")
    print(f"  wte (embeddings):        {wte_params:>12,} ({wte_params/1e6:>6.2f}M)")
    print(f"  Transformer layers:      {transformer_params:>12,} ({transformer_params/1e6:>6.2f}M)")
    print(f"  lm_head (unembedding):   {lm_head_params:>12,} ({lm_head_params/1e6:>6.2f}M)")
    print(f"  {'─' * 50}")
    print(f"  Total:                   {total_params:>12,} ({total_params/1e6:>6.2f}M)")
    print()

    # Display batch info
    tokens_per_fwdbwd = device_batch_size * max_seq_len
    grad_accum_steps = total_batch_size // tokens_per_fwdbwd
    print("Batch configuration:")
    print(f"  Tokens per forward/backward: {tokens_per_fwdbwd:,}")
    print(f"  Gradient accumulation steps: {grad_accum_steps}")
    print()

    # Generate config file content
    config_content = f"""# nanochat training configuration
# Generated by configuration wizard
# Run directory: {run_dir}

# Run identification
run = "{run_name}"  # wandb run name ("dummy" disables wandb logging)

# Model architecture
depth = {depth}
model_dim = {model_dim}  # aspect ratio {model_dim / depth if depth != 0 else 0:.1f} (or custom)
max_seq_len = {max_seq_len}

# Optimization
device_batch_size = {device_batch_size}
total_batch_size = {total_batch_size}

# Training horizon
# Only one of (num_iterations, target_flops, target_param_data_ratio) will be used
# Priority: num_iterations > target_flops > target_param_data_ratio
num_iterations = -1  # explicit number of steps (-1 = use target_param_data_ratio)
target_flops = -1.0  # calculate iterations to reach target FLOPs (-1 = disabled)
target_param_data_ratio = {target_param_data_ratio}  # Chinchilla scaling (20 = 20 tokens per parameter)

# Derived values (calculated from config above):
# num_heads = max(1, (model_dim + 127) // 128) = {num_heads}
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
