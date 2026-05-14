"""Vmap-compatible multi-head attention.

Historically a workaround for PyTorch Issue #151558 (mask validation under
``torch.vmap`` for ``F.scaled_dot_product_attention``). The upstream bug is
fixed as of PyTorch 2.12, so on a 2.12+ install ``nn.MultiheadAttention`` is
vmap-safe out of the box. This class is retained as a drop-in alternative
that works identically on older PyTorch (>=2.8) and is functionally
equivalent on 2.12+.

Example:
    >>> import torch
    >>> from polystep.layers import VmapSafeMultiHeadAttention
    >>> attn = VmapSafeMultiHeadAttention(embed_dim=64, num_heads=4)
    >>> x = torch.randn(2, 10, 64)  # (batch, seq, embed)
    >>> out = attn(x, x, x)  # self-attention
    >>> out.shape
    torch.Size([2, 10, 64])
"""

import math
import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class VmapSafeMultiHeadAttention(nn.Module):
    """Multi-head attention using explicit matmul operations for vmap compatibility.

    This implementation avoids F.scaled_dot_product_attention which has known
    vmap mask validation bugs (PyTorch Issue #151558). Instead, it uses explicit
    matrix multiplications that work correctly under torch.vmap.

    Limitations vs nn.MultiheadAttention:
        - No built-in causal masking (pass attn_mask manually)
        - Assumes batch-first layout: (batch, seq, embed_dim)
        - No add_bias_kv or add_zero_attn support
        - kdim/vdim must equal embed_dim (no cross-attention with different dims)

    Args:
        embed_dim: Total dimension of the model (must be divisible by num_heads).
        num_heads: Number of parallel attention heads.
        dropout: Dropout probability on attention weights. Default: 0.0.
        bias: Whether to add bias to projection layers. Default: True.

    Input shapes:
        - query: (batch, seq_q, embed_dim)
        - key: (batch, seq_k, embed_dim)
        - value: (batch, seq_k, embed_dim)
        - attn_mask: (seq_q, seq_k) or (batch, num_heads, seq_q, seq_k), additive mask
        - key_padding_mask: (batch, seq_k), boolean mask where True = padding

    Output shape:
        - (batch, seq_q, embed_dim)

    Example:
        >>> attn = VmapSafeMultiHeadAttention(embed_dim=64, num_heads=4)
        >>> x = torch.randn(2, 10, 64)
        >>> out = attn(x, x, x)  # self-attention
        >>> out.shape
        torch.Size([2, 10, 64])
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        # Mirror the upstream nn.MultiheadAttention signature so that
        # callers passing unsupported features get a clear
        # NotImplementedError instead of a generic Python
        # "got an unexpected keyword argument".
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        batch_first: bool = True,
    ):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

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
                "VmapSafeMultiHeadAttention does not support add_bias_kv=True. "
                "See LIMITATIONS.md."
            )
        if add_zero_attn:
            raise NotImplementedError(
                "VmapSafeMultiHeadAttention does not support add_zero_attn=True. "
                "See LIMITATIONS.md."
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

        # Separate projections for Q, K, V (not combined like GPT-2)
        # This makes vmap over parameters cleaner
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Output projection
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Dropout for attention weights
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

        Args:
            query: Query tensor of shape (batch, seq_q, embed_dim).
            key: Key tensor of shape (batch, seq_k, embed_dim).
            value: Value tensor of shape (batch, seq_k, embed_dim).
            attn_mask: Attention mask of shape (seq_q, seq_k) or
                (batch, num_heads, seq_q, seq_k). Float masks are added
                to attention scores before softmax (use -inf to mask).
                Bool masks are interpreted upstream-style: True positions
                are masked out (logits set to -inf).
            key_padding_mask: Boolean mask of shape (batch, seq_k) where True
                indicates padding positions that should not be attended to.
            need_weights: NOT supported. Returning attention weights
                from inside ``vmap`` requires extra reshapes that defeat
                the kernel fusion this layer is here to enable.
            is_causal: NOT supported. Pass an explicit triangular
                ``attn_mask`` instead.

        Returns:
            Output tensor of shape (batch, seq_q, embed_dim).
        """
        # Reject unsupported forward kwargs up front so the failure
        # mode is loud, not "wrong but plausible-looking output".
        if need_weights:
            raise NotImplementedError(
                "VmapSafeMultiHeadAttention does not support need_weights=True. "
                "See LIMITATIONS.md."
            )
        if is_causal:
            raise NotImplementedError(
                "VmapSafeMultiHeadAttention does not support is_causal=True. "
                "Pass an explicit triangular attn_mask instead. See LIMITATIONS.md."
            )

        batch_size, seq_q, _ = query.shape
        seq_k = key.shape[1]

        # Project Q, K, V: (batch, seq, embed_dim) -> (batch, seq, embed_dim)
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # Reshape to (batch, num_heads, seq, head_dim)
        Q = Q.view(batch_size, seq_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_k, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores: (batch, num_heads, seq_q, seq_k)
        # Q @ K.T = (batch, num_heads, seq_q, head_dim) @ (batch, num_heads, head_dim, seq_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Apply attention mask. Bool masks are mask-filled with -inf to
        # match upstream ``nn.MultiheadAttention`` semantics; an earlier
        # version silently added the bool tensor, which BOOSTED masked
        # logits by 1.0 instead of suppressing them. Float masks are
        # additive (callers use -inf for hard masks, finite values for
        # soft biases like ALiBi or relative-position embeddings).
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            if attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(attn_mask, float('-inf'))
            else:
                scores = scores + attn_mask

        # Apply key padding mask
        if key_padding_mask is not None:
            # (batch, seq_k) -> (batch, 1, 1, seq_k)
            padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            # True means padding, so we mask with -inf
            scores = scores.masked_fill(padding_mask, float('-inf'))

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention to values: (batch, num_heads, seq_q, head_dim)
        context = torch.matmul(attn_weights, V)

        # Reshape back: (batch, num_heads, seq_q, head_dim) -> (batch, seq_q, embed_dim)
        # Use reshape instead of .contiguous().view() to avoid a full tensor copy
        # per candidate under vmap (N candidates × this copy = ~1.3 GB overhead).
        context = context.transpose(1, 2).reshape(batch_size, seq_q, self.embed_dim)

        # Output projection
        output = self.W_o(context)

        return output
