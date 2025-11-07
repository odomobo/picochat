"""
A number of functions that help with evaluating a base model.
"""
import math
import torch
import torch.nn.functional as F
import torch.distributed as dist

@torch.no_grad()
def evaluate_bpb(model, batches, steps, token_bytes):
    """
    Instead of the naive 'mean loss', this function returns the bits per byte (bpb),
    which is a tokenization vocab size-indepedent metric, meaning you are still comparing
    apples:apples if you change the vocab size. The way this works is that instead of just
    calculating the average loss as usual, you calculate the sum loss, and indepependently
    also the sum bytes (of all the target tokens), and divide. This normalizes the loss by
    the number of bytes that the target tokens represent.

    The added complexity is so that:
    1) All "normal" tokens are normalized by the length of the token in bytes
    2) No special tokens (e.g. <|bos|>) are included in the metric - they are masked out.
    3) No actively masked tokens (using ignore_index of e.g. -1) are included in the metric.

    In addition to evaluate_loss, we need the token_bytes tensor:
    It is a 1D tensor of shape (vocab_size,), indicating the number of bytes for
    each token id, or 0 if the token is to not be counted (e.g. special tokens).
    """
    # record the losses
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=model.get_device())
    total_bytes = torch.tensor(0, dtype=torch.int64, device=model.get_device())
    batch_iter = iter(batches)
    for _ in range(steps):
        x, y = next(batch_iter)
        output = model(x, y)
        logits = output["logits"]  # (B, T, vocab_size)
        # Compute per-token cross-entropy loss
        B, T = x.size()
        loss2d = compute_cross_entropy_loss(logits, y, reduction='none')  # (B*T,)
        y = y.view(-1) # flatten
        if (y.int() < 0).any(): # mps does not currently have kernel for < 0 for int64, only int32
            # slightly more complex code path if some target tokens are ignore_index (e.g. -1)
            # any target token < 0 is to be ignored: do NOT index token_bytes with negatives
            valid = y >= 0
            y_safe = torch.where(valid, y, torch.zeros_like(y))
            # map valid targets to their byte length; ignored targets contribute 0 bytes
            num_bytes2d = torch.where(
                valid,
                token_bytes[y_safe],
                torch.zeros_like(y, dtype=token_bytes.dtype)
            )
            total_nats += (loss2d * (num_bytes2d > 0)).sum()
            total_bytes += num_bytes2d.sum()
        else:
            # fast path: no ignored targets, safe to index directly
            num_bytes2d = token_bytes[y]
            total_nats += (loss2d * (num_bytes2d > 0)).sum()
            total_bytes += num_bytes2d.sum()
    # sum reduce across all ranks
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size > 1:
        dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)
    # move both to cpu, calculate bpb and return
    total_nats = total_nats.item()
    total_bytes = total_bytes.item()
    if total_bytes == 0:
        return float('inf')
    bpb = total_nats / (math.log(2) * total_bytes)
    return bpb


def compute_cross_entropy_loss(logits, targets, reduction='mean'):
    """
    Compute cross-entropy loss from logits and targets.

    Args:
        logits: (B, T, vocab_size) - model predictions
        targets: (B, T) - target token ids
        reduction: 'mean', 'sum', or 'none' - reduction method

    Returns:
        loss: Cross-entropy loss (scalar if reduction='mean'/'sum', (B*T,) if 'none')
    """
    logits = logits.float()  # use tf32/fp32 for logits
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=reduction)
    return loss


def compute_conviction_loss(conviction, last_hidden_state, targets, lm_head):
    """
    Compute conviction loss (MSE between predicted conviction and alignment target).

    Args:
        conviction: (B, T, 1) - predicted conviction scores
        last_hidden_state: (B, T, n_embd) - last hidden state before lm_head
        targets: (B, T) - target token ids
        lm_head: nn.Embedding - token embedding layer (e.g., model.transformer.wte)

    Returns:
        loss: Scalar conviction loss
    """
    # Get expected token embeddings
    expected_embeds = lm_head(targets)  # (B, T, n_embd)

    # Compute dot product as target (measure of alignment between expected and actual)
    conviction_target = (expected_embeds * last_hidden_state).sum(dim=-1, keepdim=True)  # (B, T, 1)

    # MSE loss between predicted conviction and target
    loss = F.mse_loss(conviction.squeeze(-1), conviction_target.squeeze(-1))

    return loss


def compute_training_loss(output, targets, model, conviction_loss_weight=0.01):
    """
    Compute the combined training loss for base model pretraining.

    Args:
        output: Dict from model.forward() containing:
            - "logits": (B, T, vocab_size)
            - "conviction": (B, T, 1) if conviction head enabled
            - "last_hidden_state": (B, T, n_embd) if conviction head enabled
        targets: (B, T) target token ids
        model: The GPT model (needed to access model.transformer.wte for conviction target)
        conviction_loss_weight: Weight for conviction loss term (default: 0.01)

    Returns:
        loss: Combined scalar loss (cross-entropy + conviction if enabled)
    """
    # Cross-entropy loss on logits
    logits = output["logits"]
    loss = compute_cross_entropy_loss(logits, targets, reduction='mean')

    # Add conviction loss if enabled
    if "conviction" in output:
        conviction = output["conviction"]  # (B, T, 1)
        last_hidden_state = output["last_hidden_state"]  # (B, T, n_embd)
        conviction_loss = compute_conviction_loss(conviction, last_hidden_state, targets, model.transformer.wte)
        loss = loss + conviction_loss_weight * conviction_loss

    return loss
