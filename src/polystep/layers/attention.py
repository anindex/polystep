"""Vmap-compatible multi-head attention using explicit matmul instead of SDPA (issue #151558)."""

import math
import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class VmapSafeMultiHeadAttention(nn.Module):
    """Multi-head attention via explicit matmul, vmap-safe.

    Returns only the output tensor, not ``(output, weights)``. No built-in causal
    mask; kdim/vdim must equal embed_dim.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        # Mirror nn.MultiheadAttention's signature so unsupported kwargs raise a clear error.
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        batch_first: bool = True,
    ):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        if kdim is not None and kdim != embed_dim:
            raise NotImplementedError(
                f"VmapSafeMultiHeadAttention only supports kdim == embed_dim, "
                f"got kdim={kdim}, embed_dim={embed_dim}. See LIMITATIONS.md."
            )
        if vdim is not None and vdim != embed_dim:
            raise NotImplementedError(
                f"VmapSafeMultiHeadAttention only supports vdim == embed_dim, "
                f"got vdim={vdim}, embed_dim={embed_dim}. See LIMITATIONS.md."
            )
        if add_bias_kv:
            raise NotImplementedError(
                "VmapSafeMultiHeadAttention does not support add_bias_kv=True. See LIMITATIONS.md."
            )
        if add_zero_attn:
            raise NotImplementedError(
                "VmapSafeMultiHeadAttention does not support add_zero_attn=True. See LIMITATIONS.md."
            )
        if not batch_first:
            raise NotImplementedError(
                "VmapSafeMultiHeadAttention only supports batch_first=True "
                "(input layout (batch, seq, embed_dim)). See LIMITATIONS.md."
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.dropout = dropout

        if dropout > 0:
            warnings.warn(
                "VmapSafeMultiHeadAttention with dropout > 0 requires eval mode under vmap. "
                "Call model.eval() before vmap evaluation.",
                stacklevel=2,
            )

        self.W_q = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.W_o = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """Compute multi-head attention.

        attn_mask is float (additive) or bool (True = masked), in 2D, 3D, or 4D,
        matching ``nn.MultiheadAttention``. Returns ``(batch, seq_q, embed_dim)``.
        """
        if need_weights:
            raise NotImplementedError(
                "VmapSafeMultiHeadAttention does not support need_weights=True. See LIMITATIONS.md."
            )
        if is_causal:
            raise NotImplementedError(
                "VmapSafeMultiHeadAttention does not support is_causal=True. "
                "Pass an explicit triangular attn_mask instead. See LIMITATIONS.md."
            )

        batch_size, seq_q, _ = query.shape
        seq_k = key.shape[1]

        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        Q = Q.view(batch_size, seq_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_k, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Bool masks fill with -inf; float masks add.
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)
            elif attn_mask.dim() != 4:
                raise ValueError(
                    "attn_mask must have 2, 3, or 4 dimensions, got "
                    f"{attn_mask.dim()}D (shape {tuple(attn_mask.shape)})"
                )
            if attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(attn_mask, float("-inf"))
            else:
                scores = scores + attn_mask

        if key_padding_mask is not None:
            padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(padding_mask, float("-inf"))

        # Softmax of an all -inf row is NaN, which spreads through the value mix.
        fully_masked = torch.isneginf(scores).all(dim=-1, keepdim=True)
        attn_weights = F.softmax(scores.masked_fill(fully_masked, 0.0), dim=-1)
        attn_weights = attn_weights.masked_fill(fully_masked, 0.0)
        attn_weights = self.attn_dropout(attn_weights)

        context = torch.matmul(attn_weights, V)

        # reshape, not .contiguous().view(): the latter copies once per candidate under vmap.
        context = context.transpose(1, 2).reshape(batch_size, seq_q, self.embed_dim)

        output = self.W_o(context)

        return output
