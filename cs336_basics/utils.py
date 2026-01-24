import torch
import math
from typing import Iterable

def softmax(x: torch.Tensor, dim: int=-1) -> torch.Tensor:
    in_dtype = x.dtype
    x = x.float()
    x = x - x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x)
    sum_exp = exp_x.sum(dim=dim, keepdim=True)
    out_features = exp_x / sum_exp
    return out_features.to(in_dtype)

def scaled_dot_product_attention(
        q: torch.Tensor, # (..., seq_len, d_k)
        k: torch.Tensor, # (..., seq_len, d_k)
        v: torch.Tensor, # (..., seq_len, d_v)
        mask: torch.Tensor | None=None # (seq_len, seq_len) boolean
) -> torch.Tensor:
    d_k = q.shape[-1]
    scores = (q @ k.transpose(-1,-2)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))  # (..., seq_len, seq_len)
    
    attn = softmax(scores, dim=-1)
    out_features = attn @ v # (..., seq_len, d_v)
    return out_features

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    orig_dtype = logits.dtype
    x = logits.float()

    log_denom = torch.logsumexp(x, dim=-1)
    target_logits = x.gather(
        dim=-1,
        index=targets.unsqueeze(-1)
    ).squeeze(-1)

    loss = log_denom - target_logits

    return loss.mean().to(orig_dtype)

def lr_cosine_schedule(
        t: int,
        alpha_max: float,
        alpha_min: float,
        T_w: int,
        T_c: int,
) -> float:
    if t < T_w:
        return (t / T_w) * alpha_max
    if t <= T_c:
        return alpha_min + 0.5 * (
            1.0 + math.cos((t - T_w) / (T_c - T_w) * math.pi)
        ) * (alpha_max - alpha_min)
    
    return alpha_min

def clip_grad_norm_(
        parameters: Iterable[torch.nn.Parameter],
        max_norm: float,
        eps: float = 1e-6,
) -> None:
    
    grads = [p.grad.detach() for p in parameters if p.grad is not None]
    
    if len(grads) == 0:
        return
    
    l2_norm = torch.sqrt(sum(g.pow(2).sum() for g in grads))

    if l2_norm > max_norm:
        scale = max_norm / (l2_norm + eps)
        for p in parameters:
            if p.grad is not None:
                p.grad.mul_(scale)
    