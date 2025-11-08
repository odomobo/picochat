"""
Train model. Run as:

python base_train.py

or distributed as:

torchrun --nproc_per_node=8 base_train.py

If you are only on CPU/Macbook, you'll want to train a much much smaller LLM. Example:
python -m scripts.base_train --depth=4 --max_seq_len=512 --device_batch_size=1 --eval_tokens=512 --core_metric_every=-1 --total_batch_size=512 --num_iterations=20
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import time
from contextlib import nullcontext

import wandb
import torch

from nanochat.gpt import GPT
from nanochat.gpt_config import GPTConfig
from nanochat.dataloader import tokenizing_distributed_data_loader
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_data_dir, get_run_dir, autodetect_device_type
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine
from nanochat.model_calculator import calculate_model_size
from scripts.base_eval import evaluate_model
print_banner()

# -----------------------------------------------------------------------------
# User settings
run = "dummy" # wandb run name default ("dummy" is special - we won't log to wandb)
# Data
corpus_name = "" # name of corpus subdirectory in base_data/ (REQUIRED, set by config wizard)
# Runtime
device_type = "" # cuda|cpu|mps (empty => autodetect good device type default, in order: CUDA > MPS > CPU)
# Model architecture
depth = 20 # the depth of the Transformer model to train, rest of the kwargs are derived
model_dim = depth * 64 # embedding dimension
max_seq_len = 2048 # max context length
tie_weights = False # tie wte and lm_head weights (reduces params by ~50%, backward compatible default)
use_output_projection = False # output projection layer to reduce the limitations of tiny language models when working with tied weights
use_conviction_head = False # enable conviction head
conviction_function = "l2_distance" # conviction loss function: l2_distance, cosine_similarity
activation_fn = "relu_squared" # activation function: relu_squared, relu, gelu
head_dim = 128 # attention head dimension
num_heads = -1 # number of attention heads (-1 = derive from model_dim // head_dim)
num_kv_heads = -1 # number of key/value heads (-1 = same as num_heads, for MQA/GQA)
ffn_expansion_ratio = 4.0 # MLP expansion ratio (intermediate_dim = model_dim * ffn_expansion_ratio)
# Training horizon. Only one of these 3 will be used, in this order of precedence.
num_iterations = -1 # explicit number of steps of the optimization (-1 = disable)
target_flops = -1.0 # calculate num_iterations to reach target_flops. Useful for scaling laws experiments (-1 = disable)
target_param_data_ratio = 20 # calculate num_iterations to maintain fixed tokens:effective_params ratio (Chinchilla=20) (-1 = disable)
# Optimization
device_batch_size = 32 # per-device batch size (set to not OOM)
total_batch_size = 524288 # total desired batch size, in #tokens
embedding_lr = 0.2 # learning rate for the embedding parameters (Adam)
unembedding_lr = 0.004 # learning rate for the unembedding parameters (Adam)
tied_weights_lr = 0.2 # learning rate for tied weights when tie_weights=True (Adam)
weight_decay = 0.0 # weight decay for the embedding/unembedding parameters (Adam)
matrix_lr = 0.02 # learning rate for the matrix parameters (Muon)
grad_clip = 1.0 # gradient clipping value (0.0 = disabled)
conviction_loss_weight = 0.01 # weight for conviction loss term (only used if use_conviction_head=True)
warmup_ratio = 0.0 # ratio of iterations for LR warmup
warmdown_ratio = 0.2 # ratio of iterations for LR warmdown
final_lr_frac = 0.0 # final LR is this fraction of the initial LR
# Evaluation
eval_every = 100 # every how many steps to evaluate the model for val bpb
eval_tokens = 20*524288 # number of tokens to evaluate val loss on
core_metric_every = 2000 # every how many steps to evaluate the core metric (-1 = disable)
core_metric_max_per_task = 500 # examples per task in estimating the core metric
sample_every = 2000 # every how many steps to sample from the model
# Output
model_tag = "" # optionally override the model tag for the output checkpoint directory name

# Require configuration file to exist in NANOCHAT_RUN_DIR
run_dir = os.environ.get("NANOCHAT_RUN_DIR")
if not run_dir:
    raise RuntimeError(
        "NANOCHAT_RUN_DIR environment variable is not set.\n"
        "Please run: source scripts/run_init.sh <run_name>"
    )
config_path = os.path.join(run_dir, "config.py")
if not os.path.exists(config_path):
    raise RuntimeError(
        f"Configuration file not found: {config_path}\n"
        "This run has not been configured yet.\n"
        "The configuration should have been created by scripts/run_init.sh.\n"
        "If you skipped that step, run: python -m scripts.configure"
    )
# Prepend config file to sys.argv so configurator.py will load it
sys.argv.insert(1, config_path)

# now allow CLI to override the settings via the configurator lol
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('nanochat', 'configurator.py')).read()) # overrides from command line or config file
user_config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# Compute init
device_type = autodetect_device_type() if device_type == "" else device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

# wandb logging init
use_dummy_wandb = run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat", name=run, config=user_config)

# Configure wandb to use total_training_petaflops as primary x-axis for all metrics
if not use_dummy_wandb:
    wandb_run.define_metric("total_training_petaflops")
    wandb_run.define_metric("*", step_metric="total_training_petaflops")

# Tokenizer will be useful for evaluation, also we need the vocab size
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# Model kwargs are derived from the desired depth of the model
num_layers = depth

# Backward compatibility: calculate num_heads and num_kv_heads if not provided
if num_heads == -1:
    num_heads = model_dim // head_dim
if num_kv_heads == -1:
    num_kv_heads = num_heads

print0(f"num_layers: {num_layers}")
print0(f"model_dim: {model_dim}")
print0(f"head_dim: {head_dim}")
print0(f"num_heads: {num_heads}")
print0(f"num_kv_heads: {num_kv_heads}")

# Optimizer / data / training length related hyperparameters
# figure out the needed gradient accumulation to reach the desired total batch size
tokens_per_fwdbwd = device_batch_size * max_seq_len # tokens per iteration for a single rank
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size # total tokens per iteration for all ranks
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {device_batch_size} x {max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")
# -----------------------------------------------------------------------------
# Initialize the Model
model_config_kwargs = dict(sequence_len=max_seq_len, vocab_size=vocab_size, n_layer=num_layers, n_head=num_heads, n_kv_head=num_kv_heads, n_embd=model_dim, tie_weights=tie_weights, use_output_projection=use_output_projection, use_conviction_head=use_conviction_head, conviction_function=conviction_function, activation_fn=activation_fn, head_dim=head_dim, ffn_expansion_ratio=ffn_expansion_ratio)
with torch.device("meta"):
    model_config = GPTConfig(**model_config_kwargs)
    model = GPT(model_config)
model.to_empty(device=device)
model.init_weights()
orig_model = model # original, uncompiled model, for saving raw model state_dict
model = torch.compile(model, dynamic=False) # TODO: dynamic True/False think through
num_params = sum(p.numel() for p in model.parameters())
print0(f"Number of parameters: {num_params:,}")

# Calculate model parameters (for Chinchilla scaling)
model_size_info = calculate_model_size(model_config)
transformer_params = model_size_info['transformer_params']
effective_params = model_size_info['effective_params']
print0(f"Transformer parameters: {transformer_params:,}")
print0(f"Effective parameters (for Chinchilla scaling): {effective_params:,}")

num_flops_per_token = model.estimate_flops()
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# Calculate number of iterations. Either it is given, or from target flops, or from target data:param ratio (in that order)
assert num_iterations > 0 or target_param_data_ratio > 0 or target_flops > 0
if num_iterations > 0:
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif target_flops > 0:
    # calculate the number of iterations from the target flops
    num_iterations = round(target_flops / (num_flops_per_token * total_batch_size))
    print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
elif target_param_data_ratio > 0:
    # calculate the number of iterations from the target param data ratio (Chinchilla scaling based on effective params)
    target_tokens = target_param_data_ratio * effective_params
    num_iterations = target_tokens // total_batch_size
    print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")
else:
    raise ValueError("No training horizon specified")
total_tokens = total_batch_size * num_iterations
print0(f"Total number of training tokens: {total_tokens:,}")
print0(f"Tokens : Effective Params ratio: {total_batch_size * num_iterations / effective_params:.2f}") # Chinchilla is ~20
print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

# -----------------------------------------------------------------------------
# Initialize the Optimizer (Muon for Linear layers, AdamW for embedding and lm_head)
optimizers = model.setup_optimizers(unembedding_lr=unembedding_lr, embedding_lr=embedding_lr, tied_weights_lr=tied_weights_lr, matrix_lr=matrix_lr, weight_decay=weight_decay)
adamw_optimizer, muon_optimizer = optimizers

# Initialize the DataLoaders for train/val
if not corpus_name:
    raise ValueError("corpus_name is required. It should be set in the config.py file by the configuration wizard.")
train_loader = tokenizing_distributed_data_loader(device_batch_size, max_seq_len, split="train", corpus=corpus_name, device=device)
build_val_loader = lambda: tokenizing_distributed_data_loader(device_batch_size, max_seq_len, split="val", corpus=corpus_name, device=device)
x, y = next(train_loader) # kick off load of the very first batch of data

# -----------------------------------------------------------------------------
# Set up hyperparameter schedulers

# Learning rate scheduler
def get_lr_multiplier(it):
    warmup_iters = round(warmup_ratio * num_iterations)
    warmdown_iters = round(warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * final_lr_frac

# Momentum scheduler for Muon optimizer
def get_muon_momentum(it):
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum

# Gradient norm computation for logging
def compute_gradient_norms(model):
    """Compute L2 norms of gradients for different parameter groups (before clipping)."""

    # Compute L2 norm for each group
    def group_norm(params):
        grads = [p.grad for p in params if p.grad is not None]
        if not grads:
            return 0.0
        return torch.sqrt(sum(g.pow(2).sum() for g in grads)).item()
    
    ret = {}

    ret["grad_norm/total"] = group_norm(list(model.parameters()))

    # Separate parameters by group (matching optimizer setup)
    transformer_params = list(model.transformer.h.parameters())
    ret["grad_norm/transformer"] = group_norm(transformer_params)

    if model.config.use_output_projection:
        output_projection_params = list(model.output_projection.parameters())
        ret["grad_norm/output_projection"] = group_norm(output_projection_params)

    embedding_params = list(model.transformer.wte.parameters())
    ret["grad_norm/embedding"] = group_norm(embedding_params)

    if not model.config.tie_weights:
        lm_head_params = list(model.lm_head.parameters())
        ret["grad_norm/lm_head"] = group_norm(lm_head_params)

    if model.config.use_conviction_head:
        conviction_params = list(model.conviction_head.parameters())
        ret["grad_norm/conviction"] = group_norm(conviction_params)

    return ret

# -----------------------------------------------------------------------------
# Training loop
min_val_bpb = float("inf")
smooth_train_loss = 0 # EMA of training loss
ema_beta = 0.9 # EMA decay factor
total_training_time = 0 # total wall-clock time of training
# note that we run +1 steps only so that we can eval and save at the end
for step in range(num_iterations + 1):
    last_step = step == num_iterations
    flops_so_far = num_flops_per_token * total_batch_size * step

    # once in a while: evaluate the val bpb (all ranks participate)
    if last_step or step % eval_every == 0:
        model.eval()
        val_loader = build_val_loader()
        eval_steps = eval_tokens // (device_batch_size * max_seq_len * ddp_world_size)
        with autocast_ctx:
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.4f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({
            "total_training_petaflops": flops_so_far / 1e15,
            "val/bpb": val_bpb,
        })
        model.train()

    # once in a while: estimate the CORE metric (all ranks participate)
    # use the original uncompiled model because the inputs keep changing shape
    results = {}
    if core_metric_every > 0 and (last_step or (step > 0 and step % core_metric_every == 0)):
        model.eval()
        with autocast_ctx:
            results = evaluate_model(orig_model, tokenizer, device, max_per_task=core_metric_max_per_task)
        print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
        wandb_run.log({
            "total_training_petaflops": flops_so_far / 1e15,
            "benchmarks/core_metric": results["core_metric"],
            "benchmarks/centered_results": results["centered_results"],
        })
        model.train()

    # once in a while: sample from the model (only on master process)
    # use the original uncompiled model because the inputs keep changing shape
    if master_process and (last_step or (step > 0 and step % sample_every == 0)):
        model.eval()
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
            "The planets of the solar system are:",
            "My favorite color is",
            "If 5*x + 3 = 13, then x is",
        ]
        engine = Engine(orig_model, tokenizer) # use orig_model to avoid recompilation
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            with autocast_ctx:
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
            print0(tokenizer.decode(sample[0]))
        model.train()

    # save checkpoint at the end of the run (only on master process)
    if master_process and last_step:
        output_dirname = model_tag if model_tag else f"d{depth}" # e.g. d12
        checkpoint_dir = os.path.join(get_run_dir(), "base_checkpoints", output_dirname)
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(),
            [opt.state_dict() for opt in optimizers], # TODO: make sure saving across ranks is done correctly
            {
                "step": step,
                "val_bpb": val_bpb, # loss at last step
                "model_config": model_config_kwargs,
                "user_config": user_config, # inputs to the training script
                "device_batch_size": device_batch_size,
                "max_seq_len": max_seq_len,
            }
        )

    if last_step:
        break

    # -------------------------------------------------------------------------
    # single training step
    # evaluate the gradient
    synchronize()
    t0 = time.time()
    # Initialize accumulators for loss components
    total_ce_loss = 0.0
    total_conviction_loss = 0.0
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            output = model(x, y)
            loss, loss_components = orig_model.compute_training_loss(output, y, conviction_loss_weight)
            # Accumulate components
            total_ce_loss += loss_components["ce_loss"]
            if "conviction_loss" in loss_components:
                total_conviction_loss += loss_components["conviction_loss"]
        train_loss = loss.detach() # for logging
        loss = loss / grad_accum_steps # each .backward() is a grad sum => normalize loss here
        loss.backward()
        x, y = next(train_loader) # prefetch the next batch while the GPU is busy with forward/backward
    # Average the components over gradient accumulation steps
    avg_ce_loss = total_ce_loss / grad_accum_steps
    if use_conviction_head:
        avg_conviction_loss = total_conviction_loss / grad_accum_steps
    # log _every_ step for now
    log_this_step = (step % 1 == 0)
    # Compute gradient norms before clipping, only if we're going to log this step
    if log_this_step:
        grad_norms = compute_gradient_norms(orig_model)
    # gradient clipping (TODO possibly experiment with)
    if grad_clip > 0.0:
        torch.nn.utils.clip_grad_norm_(orig_model.parameters(), grad_clip)
    # step the optimizers
    lrm = get_lr_multiplier(step)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm
    muon_momentum = get_muon_momentum(step)
    for group in muon_optimizer.param_groups:
        group["momentum"] = muon_momentum
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # logging
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item() # EMA the training loss
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1)) # debias the EMA
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(total_batch_size / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    promised_flops_per_sec_3090 = 71e12 * ddp_world_size # bfloat16 RTX 3090
    mfu = 100 * flops_per_sec / promised_flops_per_sec_3090 # in %
    if step > 1:
        total_training_time += dt # only count the time after the first step
    print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f} | total time: {total_training_time/60:.2f}m")
    
    if log_this_step:
        log_dict = {
            "total_training_petaflops": flops_so_far / 1e15,
            "step": step,
            "total_tokens": step * total_batch_size,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/ce_loss": avg_ce_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
            **grad_norms,
        }
        # Only add conviction loss if enabled
        if use_conviction_head:
            log_dict["train/conviction_loss"] = avg_conviction_loss
        wandb_run.log(log_dict)

# print a few more stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
print0(f"Minimum validation bpb: {min_val_bpb:.4f}")

# Log to report
from nanochat.report import get_report
get_report().log(section="Base model training", data=[
    user_config, # CLI args
    { # stats about the training setup
        "Corpus": corpus_name,
        "Number of parameters": num_params,
        "Transformer parameters": transformer_params,
        "Effective parameters": effective_params,
        "Number of FLOPs per token": f"{num_flops_per_token:e}",
        "Calculated number of iterations": num_iterations,
        "Number of training tokens": total_tokens,
        "Tokens : Effective Params ratio": total_batch_size * num_iterations / effective_params,
        "DDP world size": ddp_world_size,
        "warmup_ratio": warmup_ratio,
        "warmdown_ratio": warmdown_ratio,
        "final_lr_frac": final_lr_frac,
    },
    { # stats about training outcomes
        "Minimum validation bpb": min_val_bpb,
        "Final validation bpb": val_bpb,
        "CORE metric estimate": results.get("core_metric", None),
        "MFU %": f"{mfu:.2f}%",
        "Total training flops": f"{flops_so_far:e}",
        "Total training time": f"{total_training_time/60:.2f}m",
        "Peak memory usage": f"{get_max_memory() / 1024 / 1024:.2f}MiB",
    }
])

# cleanup
wandb_run.finish() # wandb run finish
compute_cleanup()
