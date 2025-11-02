"""
Simple text completion script for base model.

Reads text from stdin (until EOF), generates a completion, and prints:
- Input text in gray
- Completion in normal terminal color

Usage examples:
  echo "Once upon a time" | python -m scripts.base_complete
  cat myfile.txt | python -m scripts.base_complete -t 0.5 -m 200
  python -m scripts.base_complete < input.txt

  # Interactive (type text, then Ctrl+D to complete)
  python -m scripts.base_complete
"""
import argparse
import sys
import torch
from nanochat.common import compute_init, autodetect_device_type
from contextlib import nullcontext
from nanochat.engine import Engine
from nanochat.checkpoint_manager import load_model

parser = argparse.ArgumentParser(description='Complete text from stdin using base model')
parser.add_argument('-g', '--model-tag', type=str, default=None, help='Model tag to load')
parser.add_argument('-s', '--step', type=int, default=None, help='Step to load')
parser.add_argument('-t', '--temperature', type=float, default=0.8, help='Temperature for generation')
parser.add_argument('-k', '--top-k', type=int, default=50, help='Top-k sampling parameter')
parser.add_argument('-m', '--max-tokens', type=int, default=256, help='Maximum tokens to generate')
parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps'], help='Device type: cuda|cpu|mps. empty => autodetect')
parser.add_argument('-d', '--dtype', type=str, default='bfloat16', choices=['float32', 'bfloat16'])
args = parser.parse_args()

# ANSI color codes
GRAY = '\033[90m'
RESET = '\033[0m'

# Init the model and tokenizer
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
ptdtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

model, tokenizer, meta = load_model("base", device, phase="eval", model_tag=args.model_tag, step=args.step)

# Create Engine for efficient generation
engine = Engine(model, tokenizer)

# Read input from stdin
input_text = sys.stdin.read()

# Handle empty input
if not input_text.strip():
    print("Note: No input provided", file=sys.stderr)

# Tokenize input
input_tokens = tokenizer.encode(input_text)

# Warn if input is too long
if len(input_tokens) > model.config.sequence_len:
    print(f"Warning: Input truncated to {model.config.sequence_len} tokens", file=sys.stderr)
    input_tokens = input_tokens[:model.config.sequence_len]

# Print input in gray
print(f"{GRAY}{input_text}{RESET}", end='', flush=True)

# Generate and print completion in normal color
generate_kwargs = {
    "num_samples": 1,
    "max_tokens": args.max_tokens,
    "temperature": args.temperature,
    "top_k": args.top_k,
}

with autocast_ctx:
    for token_column, token_masks in engine.generate(input_tokens, **generate_kwargs):
        token = token_column[0]  # pop the batch dimension (num_samples=1)
        token_text = tokenizer.decode([token])
        print(token_text, end="", flush=True)

# Print final newline
print()
