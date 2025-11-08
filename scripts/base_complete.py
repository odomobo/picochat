"""
Interactive text completion script for base model.

Loads model once, then continuously accepts text input and generates completions.
Type text line-by-line, then type a delimiter on a new line to generate:
  - ### : Normal completion mode (streams text)
  - @@@ : Conviction display mode (shows token + conviction per line)

Usage:
  python -m scripts.base_complete
  python -m scripts.base_complete --device-type cpu
  python -m scripts.base_complete -t 0.5 -m 200
"""
import argparse
import sys
import time
import torch
from nanochat.common import compute_init, autodetect_device_type
from contextlib import nullcontext
from nanochat.engine import Engine
from nanochat.checkpoint_manager import load_model

parser = argparse.ArgumentParser(description='Interactive text completion using base model')
parser.add_argument('-g', '--model-tag', type=str, default=None, help='Model tag to load')
parser.add_argument('-s', '--step', type=int, default=None, help='Step to load')
parser.add_argument('-t', '--temperature', type=float, default=0.8, help='Temperature for generation')
parser.add_argument('-k', '--top-k', type=int, default=50, help='Top-k sampling parameter')
parser.add_argument('-m', '--max-tokens', type=int, default=256, help='Maximum tokens to generate')
parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps'], help='Device type: cuda|cpu|mps. empty => autodetect')
parser.add_argument('-d', '--dtype', type=str, default='bfloat16', choices=['float32', 'bfloat16'])
parser.add_argument('--delimiter', type=str, default='###', help='Delimiter to trigger completion (default: ###)')
args = parser.parse_args()

# ANSI color codes
GRAY = '\033[90m'
RESET = '\033[0m'

def format_token_for_conviction_display(token_text):
    """
    Replace whitespace chars with gray periods for conviction display.
    Pads to exactly 20 visible characters (accounting for ANSI codes).
    """
    visible_chars = 0
    result = ""

    for char in token_text:
        if char.isspace():
            result += f"{GRAY}.{RESET}"
            visible_chars += 1
        else:
            result += char
            visible_chars += 1

    # Pad with spaces to reach 20 visible characters
    if visible_chars < 20:
        result += " " * (20 - visible_chars)

    return result

# Init the model and tokenizer
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
ptdtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

print("Loading model...", flush=True)
model, tokenizer, meta = load_model("base", device, phase="eval", model_tag=args.model_tag, step=args.step)

# Create Engine for efficient generation
engine = Engine(model, tokenizer)

# Print instructions
print("\n" + "=" * 60)
print("Interactive Text Completion")
print("=" * 60)
print(f"Type your text line-by-line (use Enter for newlines)")
print(f"Type '###' on a new line for normal generation")
print(f"Type '@@@' on a new line for conviction display mode")
print(f"Press Ctrl+C to exit")
print("=" * 60)

# Interactive loop
while True:
    try:
        print()  # Blank line before new input
        lines = []
        delimiter = None

        # Collect lines until delimiter
        while True:
            line = input(f"{GRAY}> {RESET}")
            if line.strip() == "###" or line.strip() == "@@@":
                delimiter = line.strip()
                break
            lines.append(line)

        # Join lines (this naturally preserves intentional blank lines)
        input_text = '\n'.join(lines)

        # Skip empty input
        if not input_text.strip():
            continue

        # Tokenize input
        input_tokens = tokenizer.encode(input_text)

        # Warn if input is too long
        if len(input_tokens) > model.config.sequence_len:
            print(f"Warning: Input truncated to {model.config.sequence_len} tokens", file=sys.stderr)
            input_tokens = input_tokens[:model.config.sequence_len]

        # Generate and print completion
        generate_kwargs = {
            "num_samples": 1,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "seed": int(time.time() * 1000) % (2**31),  # Different seed each time
        }

        # Choose mode based on delimiter
        conviction_mode = (delimiter == "@@@")

        with autocast_ctx:
            if conviction_mode:
                # Conviction mode: stream token + conviction per line
                for token_column, token_masks, conviction_column in engine.generate(input_tokens, **generate_kwargs):
                    token = token_column[0]
                    conviction = conviction_column[0] if conviction_column else None
                    token_text = tokenizer.decode([token])
                    display_text = format_token_for_conviction_display(token_text)

                    if conviction is None:
                        conviction_str = "N/A"
                    else:
                        conviction_str = f"{conviction:08.4f}"

                    print(f"{display_text}| {conviction_str}")
            else:
                # Normal mode: print prompt in gray, then stream tokens
                print(f"\n{GRAY}{input_text}{RESET}", end='', flush=True)
                for token_column, token_masks, conviction_column in engine.generate(input_tokens, **generate_kwargs):
                    token = token_column[0]
                    token_text = tokenizer.decode([token])
                    print(token_text, end="", flush=True)

        # Print final newline after completion
        print("\n")

    except KeyboardInterrupt:
        print("\n\nGoodbye!")
        break
    except EOFError:
        print("\n\nGoodbye!")
        break
