"""Vmap-compatible PyTorch layers (attention, LSTM).

- VmapSafeMultiHeadAttention: avoids SDPA (PyTorch Issue #151558)
- VmapSafeLSTMCell / VmapSafeLSTM: explicit gate ops bypassing CuDNN

Trades some throughput for guaranteed vmap compatibility.
"""

from .attention import VmapSafeMultiHeadAttention
from .rnn import VmapSafeLSTMCell, VmapSafeLSTM

__all__ = [
    'VmapSafeMultiHeadAttention',
    'VmapSafeLSTMCell',
    'VmapSafeLSTM',
]
