import math
import torch
from torch import nn
from einops import rearrange

from cs336_basics.nn_utils import scaled_dot_product_attention


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.empty(in_features, out_features, device=device, dtype=dtype))

        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.W, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W
    
class Embedding(nn.Module):
    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            device=None,
            dtype=None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        self.W = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))

        nn.init.trunc_normal_(self.W, mean=0.0, std=1, a=-3, b=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W[x]

class RMSnorm(nn.Module):
    def __init__(
            self,
            d_model: int,
            eps: float = 1e-5,
            device = None,
            dtype = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.g = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        y = (x / rms) * self.g.float()
        return y.to(in_dtype)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model

        if d_ff is None:
            d_ff = int(8 * d_model / 3)
            d_ff =((d_ff + 64 - 1) // 64) * 64
        self.d_ff = d_ff

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.w1(x)
        b = self.w3(x)
        gated = (a * torch.sigmoid(a)) * b
        return self.w2(gated)

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None=None):
        super().__init__()
        assert d_k % 2 == 0, "RoPE requires d_k to be even."

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        idx = torch.arange(0, d_k, 2, device=device, dtype=torch.float32)
        inv_denom = self.theta ** (idx / d_k)
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        angles = positions[:,None] / inv_denom[None, :]

        cos = torch.cos(angles)
        sin = torch.sin(angles)

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.float()

        cos = self.cos[token_positions].to(torch.float32)
        sin = self.sin[token_positions].to(torch.float32)

        # Reshape into pairs: (..., seq_len, d_k/2, 2)
        x_pair = rearrange(x, "... s (d two) -> ... s d two", two=2)
        x1 = x_pair[..., 0]
        x2 = x_pair[..., 1]
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        y = torch.stack([y1, y2], dim=-1) # (..., seq_len, d_k/2, 2)
        out_features = rearrange(y, "... s d two -> ... s (d two)")

        return out_features.to(in_dtype)

class MultiHeadSelfAttention(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            max_seq_len: int,
            rope_theta: float,
            device=None,
            dtype=None,
    ):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = self.d_k
        self.max_seq_len = max_seq_len

         # Projections: (d_model -> h*d_k) etc. Here h*d_k == h*d_v == d_model.
        self.wq = Linear(d_model, d_model, device=device, dtype=dtype)
        self.wk = Linear(d_model, d_model, device=device, dtype=dtype)
        self.wv = Linear(d_model, d_model, device=device, dtype=dtype)
        self.wo = Linear(d_model, d_model, device=device, dtype=dtype)

        # RoPE
        if rope_theta is not None:
            self.rope = RotaryPositionalEmbedding(
                theta=rope_theta,
                d_k=self.d_k,
                max_seq_len=max_seq_len,
                device=device,
            )
        else:
            self.rope = None
        #Causal mask
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(
            self,
            x: torch.Tensor,
            token_positions: torch.Tensor,
    ) -> torch.Tensor:
        
        *lead, seq_len, _ = x.shape

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        q = rearrange(q, "... s (h d) -> ... h s d", h=self.num_heads)
        k = rearrange(k, "... s (h d) -> ... h s d", h=self.num_heads)
        v = rearrange(v, "... s (h d) -> ... h s d", h=self.num_heads)
        
        if self.rope is not None:
            tp = token_positions.unsqueeze(-2).expand(*lead, self.num_heads, seq_len)
            q = self.rope(q, tp)
            k = self.rope(k, tp)

        mask = self.causal_mask[:seq_len, :seq_len]
        out_features = scaled_dot_product_attention(q, k, v, mask)
        out_features = rearrange(out_features, "... h s d -> ... s (h d)")
        out_features = self.wo(out_features)

        return out_features

class TransformerBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_ff: int,
            max_seq_len: int,
            rope_theta: float,
            device=None,
            dtype=None,
    ):
        super().__init__()
        
        self.attn_norm = RMSnorm(d_model, device=device, dtype=dtype)
        self.ffn_norm = RMSnorm(d_model, device=device, dtype=dtype)

        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            device=device,
            dtype=dtype,
        )

        self.ffn = SwiGLU(
            d_model=d_model,
            d_ff=d_ff,
            device=device,
            dtype=dtype,
        )

    def forward(
            self,
            x: torch.Tensor,
            token_positions: torch.Tensor,
    ) -> torch.Tensor:
        
        x = x + self.attn(
            self.attn_norm(x),
            token_positions,
        )

        x = x + self.ffn(
            self.ffn_norm(x)
        )

        return x
    
class TransformerLM(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            d_model: int,
            num_layers: int,
            num_heads: int,
            d_ff: int,
            max_seq_len: int,
            rope_theta: float,
            device=None,
            dtype=None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers

        # Token embedding
        self.token_embedding = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype,
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=max_seq_len,
                rope_theta=rope_theta,
                device=device,
                dtype=dtype,
            ) for _ in range(num_layers)])
        
        # Final norm
        self.final_norm = RMSnorm(
            d_model,
            device=device,
            dtype=dtype,
        )

        # Output projection (logits)
        self.lm_head = Linear(
            d_model,
            vocab_size,
            device=device,
            dtype=dtype,
        )
    def forward(
            self,
            token_ids: torch.Tensor,
    ) -> torch.Tensor:
        
        batch, seq_len = token_ids.shape

        token_positions = torch.arange(
            seq_len,
            device = token_ids.device,
        ).unsqueeze(0).expand(batch, seq_len)

        x = self.token_embedding(token_ids) # (batch, seq_len, d_model)

        for block in self.blocks:
            x = block(x, token_positions)
        
        x = self.final_norm(x)
        logits = self.lm_head(x) # (batch, seq_len, vocab_size)

        return logits



