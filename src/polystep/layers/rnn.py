"""Vmap-compatible LSTM using explicit per-step gate ops, bypassing CuDNN."""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class VmapSafeLSTMCell(nn.Module):
    """LSTM cell using explicit gate computations for vmap compatibility.

    Returns ``(h_new, (h_new, c_new))``, unlike ``nn.LSTMCell`` which returns
    ``(h, c)``.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_i = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.W_h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute one LSTM step."""
        h, c = state

        gates = self.W_i(x) + self.W_h(h)

        i, f, g, o = gates.chunk(4, dim=-1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c_new = f * c + i * g

        h_new = o * torch.tanh(c_new)

        return h_new, (h_new, c_new)


class VmapSafeLSTM(nn.Module):
    """Multi-layer LSTM using explicit gate computations for vmap compatibility.

    Lacks bidirectional, proj_size, and batch-second support; slower than CuDNN.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
        # Mirror nn.LSTM's signature so unsupported kwargs raise a clear error.
        bidirectional: bool = False,
        proj_size: int = 0,
        batch_first: bool = True,
    ):
        super().__init__()

        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if bidirectional:
            raise NotImplementedError("VmapSafeLSTM does not support bidirectional=True. See LIMITATIONS.md.")
        if proj_size != 0:
            raise NotImplementedError("VmapSafeLSTM does not support proj_size != 0. See LIMITATIONS.md.")
        if not batch_first:
            raise NotImplementedError(
                "VmapSafeLSTM only supports batch_first=True (input layout "
                "(batch, seq_len, input_size)). See LIMITATIONS.md."
            )

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.cells = nn.ModuleList()
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            self.cells.append(VmapSafeLSTMCell(layer_input_size, hidden_size, bias=bias))

        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Run the sequence through the stacked LSTM."""
        # PackedSequence would fail later with an opaque error, so reject it here.
        if isinstance(x, nn.utils.rnn.PackedSequence):
            raise NotImplementedError(
                "VmapSafeLSTM does not support PackedSequence input. Pad to a dense tensor first. See LIMITATIONS.md."
            )

        batch_size, seq_len, _ = x.shape
        device = x.device
        dtype = x.dtype

        # Use list-based states to avoid in-place updates under vmap.
        if state is None:
            h_list = [
                torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype) for _ in range(self.num_layers)
            ]
            c_list = [
                torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype) for _ in range(self.num_layers)
            ]
        else:
            h, c = state
            expected = (self.num_layers, batch_size, self.hidden_size)
            if tuple(h.shape) != expected or tuple(c.shape) != expected:
                raise ValueError(
                    f"initial h and c must both have shape {expected}, got {tuple(h.shape)} and {tuple(c.shape)}."
                )
            h_list = [h[i] for i in range(self.num_layers)]
            c_list = [c[i] for i in range(self.num_layers)]

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]

            new_h_list = []
            new_c_list = []
            for layer_idx, cell in enumerate(self.cells):
                h_layer = h_list[layer_idx]
                c_layer = c_list[layer_idx]

                h_new, (h_out, c_out) = cell(x_t, (h_layer, c_layer))

                new_h_list.append(h_out)
                new_c_list.append(c_out)

                if self.dropout_layer is not None and layer_idx < self.num_layers - 1:
                    x_t = self.dropout_layer(h_new)
                else:
                    x_t = h_new

            h_list = new_h_list
            c_list = new_c_list

            outputs.append(x_t)

        output = torch.stack(outputs, dim=1)

        h_n = torch.stack(h_list, dim=0)
        c_n = torch.stack(c_list, dim=0)

        return output, (h_n, c_n)
