"""Vmap-compatible LSTM bypassing CuDNN.

Standard ``nn.LSTM`` (via ``aten::lstm.input``) has no batching rule under
``torch.vmap`` and fails on both CPU and CUDA -- still the case on PyTorch
2.12 (verified). This implementation uses explicit per-step gate
computations, which are vmap-safe but ~2-5x slower than CuDNN.

Example:
    >>> import torch
    >>> from polystep.layers import VmapSafeLSTM
    >>> lstm = VmapSafeLSTM(input_size=32, hidden_size=64, num_layers=2)
    >>> x = torch.randn(4, 10, 32)  # (batch, seq, input)
    >>> out, (h_n, c_n) = lstm(x)
    >>> out.shape
    torch.Size([4, 10, 64])
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class VmapSafeLSTMCell(nn.Module):
    """LSTM cell using explicit gate computations for vmap compatibility.

    This implementation avoids CuDNN which has known issues under torch.vmap
    with ".data access" errors. Instead, explicit gate computations are used
    that work correctly under vectorization.

    Args:
        input_size: The number of expected features in the input.
        hidden_size: The number of features in the hidden state.
        bias: Whether to add bias to the linear transformations. Default: True.

    Input shapes:
        - x: (batch, input_size)
        - state: Tuple of (h, c) where both are (batch, hidden_size)

    Output shapes:
        - h_new: (batch, hidden_size) - new hidden state
        - (h_new, c_new): Tuple of new states for compatibility with nn.LSTMCell

    Example:
        >>> cell = VmapSafeLSTMCell(input_size=32, hidden_size=64)
        >>> x = torch.randn(4, 32)
        >>> h = torch.zeros(4, 64)
        >>> c = torch.zeros(4, 64)
        >>> h_new, (h_out, c_out) = cell(x, (h, c))
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

        # Combined input and hidden projections for all 4 gates
        # Gates: input (i), forget (f), cell input (g), output (o)
        self.W_i = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.W_h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute one LSTM step.

        Args:
            x: Input tensor of shape (batch, input_size).
            state: Tuple of (h, c) hidden and cell states,
                each of shape (batch, hidden_size).

        Returns:
            Tuple of:
                - h_new: New hidden state of shape (batch, hidden_size)
                - (h_new, c_new): Tuple of new states
        """
        h, c = state

        # Compute all gates at once: (batch, 4 * hidden_size)
        gates = self.W_i(x) + self.W_h(h)

        # Split into 4 gates: each (batch, hidden_size)
        i, f, g, o = gates.chunk(4, dim=-1)

        # Apply activations
        i = torch.sigmoid(i)  # Input gate
        f = torch.sigmoid(f)  # Forget gate
        g = torch.tanh(g)  # Cell input (candidate)
        o = torch.sigmoid(o)  # Output gate

        # Update cell state
        c_new = f * c + i * g

        # Compute new hidden state
        h_new = o * torch.tanh(c_new)

        return h_new, (h_new, c_new)


class VmapSafeLSTM(nn.Module):
    """Multi-layer LSTM using explicit gate computations for vmap compatibility.

    This implementation wraps multiple VmapSafeLSTMCell layers to provide
    a drop-in replacement for nn.LSTM that works under torch.vmap.

    Limitations vs nn.LSTM:
        - No bidirectional support
        - No proj_size support
        - Assumes batch-first layout: (batch, seq_len, input_size)
        - 2-5x slower than CuDNN (explicit gate computation)

    Note: Expected 2-5x slower than CuDNN but guaranteed vmap-safe.

    Args:
        input_size: The number of expected features in the input.
        hidden_size: The number of features in the hidden state.
        num_layers: Number of recurrent layers. Default: 1.
        bias: Whether to add bias to the linear transformations. Default: True.
        dropout: Dropout probability between layers (not after last layer).
            Default: 0.0.

    Input shapes:
        - x: (batch, seq_len, input_size)
        - state: Optional tuple of (h, c) where each is (num_layers, batch, hidden_size).
            If None, zero-initialized states are used.

    Output shapes:
        - output: (batch, seq_len, hidden_size) - output features from last layer
        - (h_n, c_n): Final states, each (num_layers, batch, hidden_size)

    Example:
        >>> lstm = VmapSafeLSTM(input_size=32, hidden_size=64, num_layers=2)
        >>> x = torch.randn(4, 10, 32)  # (batch, seq, input)
        >>> out, (h_n, c_n) = lstm(x)
        >>> out.shape
        torch.Size([4, 10, 64])
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
        # Mirror the upstream nn.LSTM signature so that callers passing
        # unsupported features get a clear NotImplementedError instead
        # of a generic Python "got an unexpected keyword argument".
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

        # Build cells for each layer
        self.cells = nn.ModuleList()
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            self.cells.append(VmapSafeLSTMCell(layer_input_size, hidden_size, bias=bias))

        # Dropout between layers (not applied after last layer)
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Process sequence through multi-layer LSTM.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size).
            state: Optional tuple of (h, c) initial states, each of shape
                (num_layers, batch, hidden_size). If None, zero states are used.

        Returns:
            Tuple of:
                - output: Tensor of shape (batch, seq_len, hidden_size)
                    containing output features from the last layer
                - (h_n, c_n): Tuple of final states, each of shape
                    (num_layers, batch, hidden_size)
        """
        # PackedSequence support is not implemented. nn.LSTM accepts
        # one, but our explicit-loop forward would silently treat it as
        # a tensor and crash later with an opaque AttributeError. Catch
        # it up front with a clear message.
        if isinstance(x, nn.utils.rnn.PackedSequence):
            raise NotImplementedError(
                "VmapSafeLSTM does not support PackedSequence input. Pad to a dense tensor first. See LIMITATIONS.md."
            )

        batch_size, seq_len, _ = x.shape
        device = x.device
        dtype = x.dtype

        # Initialize states if not provided
        # Use list-based state tracking to avoid in-place updates under vmap
        if state is None:
            h_list = [
                torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype) for _ in range(self.num_layers)
            ]
            c_list = [
                torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype) for _ in range(self.num_layers)
            ]
        else:
            h, c = state
            # Split stacked tensor into list of per-layer states
            h_list = [h[i] for i in range(self.num_layers)]
            c_list = [c[i] for i in range(self.num_layers)]

        # Process sequence timestep by timestep
        outputs = []
        for t in range(seq_len):
            # Get input for this timestep: (batch, input_size)
            x_t = x[:, t, :]

            # Process through layers, collecting new states
            new_h_list = []
            new_c_list = []
            for layer_idx, cell in enumerate(self.cells):
                h_layer = h_list[layer_idx]
                c_layer = c_list[layer_idx]

                # Cell forward pass
                h_new, (h_out, c_out) = cell(x_t, (h_layer, c_layer))

                # Collect new states (no in-place update)
                new_h_list.append(h_out)
                new_c_list.append(c_out)

                # Apply dropout between layers (not after last)
                if self.dropout_layer is not None and layer_idx < self.num_layers - 1:
                    x_t = self.dropout_layer(h_new)
                else:
                    x_t = h_new

            # Update state lists for next timestep
            h_list = new_h_list
            c_list = new_c_list

            # Collect output from last layer
            outputs.append(x_t)

        # Stack outputs: (batch, seq_len, hidden_size)
        output = torch.stack(outputs, dim=1)

        # Stack final states: (num_layers, batch, hidden_size)
        h_n = torch.stack(h_list, dim=0)
        c_n = torch.stack(c_list, dim=0)

        return output, (h_n, c_n)
