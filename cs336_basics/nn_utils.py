import torch
import math

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