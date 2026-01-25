from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from cs336_basics import tokenizer

@dataclass
class DecodeConfig:
    max_new_tokens: int = 128
    temperature: float = 1.0          
    top_p: float = 1.0                
    eos_token_id: Optional[int] = tokenizer.special_token_to_id["<|endoftext|>"]


def _sample_top_p(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=0)

    keep = cumulative <= top_p
    keep[0] = True  

    filtered_probs = sorted_probs * keep
    filtered_probs = filtered_probs / filtered_probs.sum()

    sampled_idx = torch.multinomial(filtered_probs, num_samples=1)
    return sorted_idx[sampled_idx].squeeze(0)


@torch.no_grad()
def decode(
    model,
    prompt_ids: torch.Tensor,
    cfg: DecodeConfig,
) -> torch.Tensor:
   
    model.eval()

    if prompt_ids.dim() == 1:
        prompt_ids = prompt_ids.unsqueeze(0)
    if prompt_ids.dim() != 2 or prompt_ids.size(0) != 1:
        raise ValueError("decode only supports a single prompt")

    device = next(model.parameters()).device
    x = prompt_ids.to(device=device, dtype=torch.long)

    max_seq_len = None
    for block in model.blocks:
        if hasattr(block.attn, "max_seq_len"):
            max_seq_len = block.attn.max_seq_len
            break

    for _ in range(cfg.max_new_tokens):
        x_in = x
        if max_seq_len is not None and x.size(1) > max_seq_len:
            x_in = x[:, -max_seq_len:]

        logits = model(x_in)          # (1, seq_len, vocab)
        logits = logits[:, -1, :]     # (1, vocab)

        if cfg.temperature is None or cfg.temperature <= 0.0:
            next_id = torch.argmax(logits, dim=-1)

        else:
            logits = logits / cfg.temperature
            probs = torch.softmax(logits, dim=-1).squeeze(0)

            if cfg.top_p is not None and cfg.top_p < 1.0:
                next_id = _sample_top_p(probs, cfg.top_p).unsqueeze(0)
            else:
                next_id = torch.multinomial(probs, num_samples=1)

        x = torch.cat([x, next_id.view(1, 1)], dim=1)

        if cfg.eos_token_id is not None:
            if int(next_id.item()) == int(cfg.eos_token_id):
                break

    return x
