"""
Utilities for saving and loading model/optim/state checkpoints.
"""
import os
import re
import glob
import json
import logging
import torch

from nanochat.common import get_run_dir
from nanochat.gpt import GPT
from nanochat.gpt_config import GPTConfig
from nanochat.tokenizer import get_tokenizer
from nanochat.common import setup_default_logging

# Set up logging
setup_default_logging()
logger = logging.getLogger(__name__)
def log0(message):
    if int(os.environ.get('RANK', 0)) == 0:
        logger.info(message)

def save_checkpoint(checkpoint_dir, step, model_data, optimizer_data, meta_data):
    assert int(os.environ.get('RANK', 0)) == 0 # prevent footguns for now
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Save the model state (parameters)
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    torch.save(model_data, model_path)
    log0(f"Saved model file to: {model_path}")
    # Save the optimizer state (useful for SFT or any other fine-tuning)
    if optimizer_data is not None:
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}.pt")
        torch.save(optimizer_data, optimizer_path)
        log0(f"Saved optimizer file to: {optimizer_path}")
    # Save the metadata dict as json
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, "w") as f:
        json.dump(meta_data, f, indent=2)
    log0(f"Saved metadata file to: {meta_path}")


def load_checkpoint(checkpoint_dir, step, device, load_optimizer=False):
    # Load the model state
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    model_data = torch.load(model_path, map_location=device)
    # Load the optimizer state if requested
    optimizer_data = None
    if load_optimizer:
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}.pt")
        optimizer_data = torch.load(optimizer_path, map_location=device)
    # Load the metadata
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, "r") as f:
        meta_data = json.load(f)
    return model_data, optimizer_data, meta_data

def _backward_compatibility(model_config_kwargs):
    """
    If certain settings are not present, default them to the the fallback option, to preserve backward compatibility.
    """

    if "tie_weights" not in model_config_kwargs:
        model_config_kwargs["tie_weights"] = False
        log0("Warning: tie_weights not found in checkpoint metadata, assuming False (old untied model)")

    if "use_output_projection" not in model_config_kwargs:
        model_config_kwargs["use_output_projection"] = False
        log0("Warning: use_output_projection not found in checkpoint metadata, assuming False (old untied model)")

    if "activation_fn" not in model_config_kwargs:
        model_config_kwargs["activation_fn"] = "relu_squared"
        log0("Warning: activation_fn not found in checkpoint metadata, assuming relu_squared (old behavior)")

    if "head_dim" not in model_config_kwargs:
        model_config_kwargs["head_dim"] = 128
        log0("Warning: head_dim not found in checkpoint metadata, assuming 128 (old behavior)")

    if "ffn_expansion_ratio" not in model_config_kwargs:
        model_config_kwargs["ffn_expansion_ratio"] = 4.0
        log0("Warning: ffn_expansion_ratio not found in checkpoint metadata, assuming 4.0 (old behavior)")

    # Remove n_head and n_kv_head if they exist (old checkpoints)
    # These are now derived from n_embd and head_dim
    if "n_head" in model_config_kwargs:
        del model_config_kwargs["n_head"]
        log0("Warning: n_head found in checkpoint metadata, removing (now derived from n_embd and head_dim)")

    if "n_kv_head" in model_config_kwargs:
        del model_config_kwargs["n_kv_head"]
        log0("Warning: n_kv_head found in checkpoint metadata, removing (now always equals n_head)")

    return model_config_kwargs

def build_model(checkpoint_dir, step, device, phase):
    """
    A bunch of repetitive code to build a model from a given checkpoint.
    Returns:
    - base model - uncompiled, not wrapped in DDP
    - tokenizer
    - meta data saved during base model training
    """
    assert phase in ["train", "eval"], f"Invalid phase: {phase}"
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, step, device, load_optimizer=False)
    # Hack: fix torch compile issue, which prepends all keys with _orig_mod.
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
    model_config_kwargs = _backward_compatibility(meta_data["model_config"])
    
    log0(f"Building model with config: {model_config_kwargs}")
    model_config = GPTConfig(**model_config_kwargs)
    with torch.device("meta"):
        model = GPT(model_config)
    # Load the model state
    model.to_empty(device=device)
    model.init_weights() # note: this is dumb, but we need to init the rotary embeddings. TODO: fix model re-init
    model.load_state_dict(model_data, strict=True, assign=True)
    # Put the model in the right training phase / mode
    if phase == "eval":
        model.eval()
    else:
        model.train()
    # Load the Tokenizer
    tokenizer = get_tokenizer()
    # Sanity check: compatibility between model and tokenizer
    assert tokenizer.get_vocab_size() == model_config_kwargs["vocab_size"]
    return model, tokenizer, meta_data


def find_largest_model(checkpoint_dir):
    # attempt to guess the model tag: take the biggest model available
    model_tags = [f for f in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, f))]
    if not model_tags:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    # 1) normally all model tags are of the form d<number>, try that first:
    candidates = []
    for model_tag in model_tags:
        match = re.match(r"d(\d+)", model_tag)
        if match:
            model_depth = int(match.group(1))
            candidates.append((model_depth, model_tag))
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    # 2) if that failed, take the most recently updated model:
    model_tags.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
    return model_tags[0]


def find_last_step(checkpoint_dir):
    # Look into checkpoint_dir and find model_<step>.pt with the highest step
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "model_*.pt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    last_step = int(max(os.path.basename(f).split("_")[-1].split(".")[0] for f in checkpoint_files))
    return last_step

# -----------------------------------------------------------------------------
# convenience functions that take into account nanochat's directory structure

def load_model_from_dir(checkpoints_dir, device, phase, model_tag=None, step=None):
    if model_tag is None:
        # guess the model tag by defaulting to the largest model
        model_tag = find_largest_model(checkpoints_dir)
        log0(f"No model tag provided, guessing model tag: {model_tag}")
    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
    if step is None:
        # guess the step by defaulting to the last step
        step = find_last_step(checkpoint_dir)
    assert step is not None, f"No checkpoints found in {checkpoint_dir}"
    # build the model
    log0(f"Loading model from {checkpoint_dir} with step {step}")
    model, tokenizer, meta_data = build_model(checkpoint_dir, step, device, phase)
    return model, tokenizer, meta_data

def load_model(source, *args, **kwargs):
    model_dir = {
        "base": "base_checkpoints",
        "mid": "mid_checkpoints",
        "sft": "chatsft_checkpoints",
        "rl": "chatrl_checkpoints",
    }[source]
    run_dir = get_run_dir()
    checkpoints_dir = os.path.join(run_dir, model_dir)
    return load_model_from_dir(checkpoints_dir, *args, **kwargs)
