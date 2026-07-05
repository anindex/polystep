"""Parameter-particle transformation utilities.

Converts between nn.Module state_dict and flat particle arrays,
handling padding, shared parameter deduplication, and reconstruction
via ParamLayout metadata.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import Tuple


import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)


def get_device(model: nn.Module) -> torch.device:
    """Get device from model's first parameter, defaulting to CPU.

    Args:
        model: Any PyTorch module.

    Returns:
        Device of the first parameter, or ``torch.device('cpu')``
        if the module has no parameters or buffers.
    """
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def create_generator(seed: int, device: torch.device) -> torch.Generator:
    """Create a seeded ``torch.Generator`` on the given device.

    Args:
        seed: Integer seed for reproducibility.
        device: Device for the generator (must match tensors it will generate).

    Returns:
        Seeded ``torch.Generator``.
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return gen


@dataclass(frozen=True)
class ParamEntry:
    """Metadata for a single parameter/buffer in the layout.

    Attributes:
        key: state_dict key (e.g., "fc1.weight").
        shape: Original tensor shape.
        dtype: Original tensor dtype.
        offset: Element offset in the flat (deduplicated) array.
        numel: Number of elements.
        requires_grad: Whether the tensor requires gradient.
        module_path: Parent module path (e.g., "fc1" for "fc1.weight").
        shared_with: Keys sharing the same storage (empty for non-shared).
    """

    key: str
    shape: Tuple[int, ...]
    dtype: torch.dtype
    offset: int
    numel: int
    requires_grad: bool
    module_path: str
    shared_with: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ParamLayout:
    """Frozen layout describing how to flatten/unflatten nn.Module parameters.

    Created once via ``from_module()`` and reused for every flatten/unflatten
    call. Stores per-entry metadata, deduplication info for shared parameters,
    and padding/dtype information.

    Attributes:
        entries: Per-parameter metadata (only canonical/first-seen entries).
        total_params: Total element count (deduplicated).
        padded_size: Total after padding to ``particle_dim`` alignment.
        particle_dim: Number of elements per particle row (default 2).
        dominant_dtype: Most common dtype by element count.
        shared_groups: Tuples of keys sharing the same storage.
            The first key in each group is the canonical one stored in entries.
        _all_keys: All state_dict keys in original order (for unflatten).
    """

    entries: Tuple[ParamEntry, ...]
    total_params: int
    padded_size: int
    particle_dim: int
    dominant_dtype: torch.dtype
    shared_groups: Tuple[Tuple[str, ...], ...] = ()
    _all_keys: Tuple[str, ...] = ()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_module(
        cls,
        model: nn.Module,
        particle_dim: int = 2,
    ) -> ParamLayout:
        """Create a ``ParamLayout`` from any ``nn.Module``.

        Uses ``model.state_dict()`` to enumerate all parameters and buffers.
        Shared tensors (same ``data_ptr()``) are deduplicated: only the first
        occurrence is stored in *entries*; sharing relationships are recorded
        in *shared_groups*.

        Args:
            model: Any PyTorch module.
            particle_dim: Number of elements per particle row.

        Returns:
            Frozen ``ParamLayout`` ready for ``flatten()`` / ``unflatten()``.
        """
        sd = model.state_dict()

        if len(sd) == 0:
            return cls(
                entries=(),
                total_params=0,
                padded_size=0,
                particle_dim=particle_dim,
                dominant_dtype=torch.float32,
                shared_groups=(),
                _all_keys=(),
            )

        # --- Pass 1: detect shared storage via data_ptr ---
        seen_ptrs: dict[int, str] = {}  # data_ptr -> canonical key
        canonical_entries: list[ParamEntry] = []
        shared_map: dict[str, list[str]] = {}  # canonical_key -> [alias keys]
        offset = 0

        # Build a lookup for requires_grad from named_parameters.
        # Also collect data_ptrs of trainable params so shared aliases
        # (e.g., tied weights) can be detected even if they appear under
        # a different name in state_dict.
        param_grad = {}
        trainable_ptrs: set[int] = set()
        for name, param in model.named_parameters():
            param_grad[name] = param.requires_grad
            if param.requires_grad:
                trainable_ptrs.add(param.data_ptr())

        all_keys: list[str] = []

        for key, tensor in sd.items():
            # Skip non-trainable tensors (buffers). Only trainable
            # parameters (requires_grad=True) belong in the OT particle
            # array. Non-trainable float buffers (e.g., BatchNorm
            # running_mean/running_var) would otherwise drift randomly
            # since the evaluator overrides them with frozen values -
            # the optimizer gets no gradient signal for them.
            # Exception: shared/tied params may appear under a name not
            # in named_parameters() - detect them via data_ptr.
            requires_grad = param_grad.get(key, False)
            is_trainable_alias = tensor.data_ptr() in trainable_ptrs
            if not requires_grad and not is_trainable_alias:
                continue

            all_keys.append(key)
            ptr = tensor.data_ptr()

            if ptr in seen_ptrs and tensor.numel() > 0:
                # This tensor shares storage with an earlier one
                canonical_key = seen_ptrs[ptr]
                shared_map.setdefault(canonical_key, [canonical_key]).append(key)
                continue

            if tensor.numel() > 0:
                seen_ptrs[ptr] = key

            numel = tensor.numel()
            module_path = key.rsplit(".", 1)[0] if "." in key else ""

            canonical_entries.append(
                ParamEntry(
                    key=key,
                    shape=tuple(tensor.shape),
                    dtype=tensor.dtype,
                    offset=offset,
                    numel=numel,
                    requires_grad=requires_grad,
                    module_path=module_path,
                )
            )
            offset += numel

        total_params = offset

        # --- Build shared_groups tuples ---
        shared_groups: list[Tuple[str, ...]] = []
        # Also update entries with shared_with info
        updated_entries: list[ParamEntry] = []
        for entry in canonical_entries:
            if entry.key in shared_map:
                aliases = tuple(shared_map[entry.key])
                shared_groups.append(aliases)
                # Replace entry with shared_with populated
                # shared_with contains alias keys (excluding the canonical one)
                entry = ParamEntry(
                    key=entry.key,
                    shape=entry.shape,
                    dtype=entry.dtype,
                    offset=entry.offset,
                    numel=entry.numel,
                    requires_grad=entry.requires_grad,
                    module_path=entry.module_path,
                    shared_with=tuple(k for k in aliases if k != entry.key),
                )
            updated_entries.append(entry)

        # Surface tied-weight detection at INFO level so users see when
        # their model has shared parameters (e.g. transformer embedding
        # tied to lm_head). Otherwise the dedup happens silently and a
        # mysterious parameter-count gap is hard to track down.
        if shared_groups:
            tied_summary = ", ".join(f"{group[0]} <- {{{', '.join(group[1:])}}}" for group in shared_groups)
            logger.info(
                "ParamLayout deduplicated tied / shared weights: %s",
                tied_summary,
            )

        # --- Determine dominant dtype ---
        dtype_counts: dict[torch.dtype, int] = {}
        for entry in updated_entries:
            dtype_counts[entry.dtype] = dtype_counts.get(entry.dtype, 0) + entry.numel
        dominant_dtype = max(dtype_counts, key=dtype_counts.get) if dtype_counts else torch.float32

        # --- Padding ---
        padded_size = total_params + ((-total_params) % particle_dim) if total_params > 0 else 0

        return cls(
            entries=tuple(updated_entries),
            total_params=total_params,
            padded_size=padded_size,
            particle_dim=particle_dim,
            dominant_dtype=dominant_dtype,
            shared_groups=tuple(shared_groups),
            _all_keys=tuple(all_keys),
        )

    # ------------------------------------------------------------------
    # Flatten / Unflatten
    # ------------------------------------------------------------------

    def batch_unflatten(self, particles_batch: torch.Tensor) -> dict[str, torch.Tensor]:
        """Convert N particle vectors to stacked param dicts for vmap.

        Each key maps to a tensor with an extra leading batch dimension.
        Shared parameters produce aliased entries (same tensor object).

        Args:
            particles_batch: Tensor of shape ``(N, rows, particle_dim)`` or
                ``(N, flat_size)``. The last dimensions are flattened internally.

        Returns:
            Dict ``{key: tensor of shape (N, *original_shape)}`` suitable for
            ``torch.vmap`` over dimension 0.
        """
        if self.total_params == 0:
            return {}

        N = particles_batch.shape[0]
        flat = particles_batch.reshape(N, -1)
        stacked: dict[str, torch.Tensor] = {}

        for entry in self.entries:
            param = flat[:, entry.offset : entry.offset + entry.numel]
            param = param.reshape(N, *entry.shape)
            # Only cast if this entry's dtype differs from dominant_dtype
            # (avoids a no-op .to() kernel launch per entry).
            if entry.dtype != self.dominant_dtype:
                param = param.to(entry.dtype)
            stacked[entry.key] = param
            for alias in entry.shared_with:
                stacked[alias] = param

        return stacked

    # ------------------------------------------------------------------
    # Flatten / Unflatten
    # ------------------------------------------------------------------

    def flatten(self, model: nn.Module) -> torch.Tensor:
        """Flatten model state_dict to a 2D particle tensor.

        Each entry (deduplicated) is cast to ``dominant_dtype``, concatenated,
        padded, and reshaped to ``(N, particle_dim)``.

        Args:
            model: Module whose state_dict matches this layout.

        Returns:
            Tensor of shape ``(N, particle_dim)`` where
            ``N * particle_dim >= total_params``.
        """
        if self.total_params == 0:
            return torch.zeros(0, self.particle_dim, dtype=self.dominant_dtype)

        sd = model.state_dict()
        parts: list[torch.Tensor] = []
        for entry in self.entries:
            tensor = sd[entry.key]
            parts.append(tensor.detach().to(self.dominant_dtype).reshape(-1))

        raveled = torch.cat(parts)

        pad_size = self.padded_size - raveled.shape[0]
        if pad_size > 0:
            raveled = F.pad(raveled, (0, pad_size))

        return raveled.reshape(-1, self.particle_dim)

    def unflatten(self, particles: torch.Tensor) -> OrderedDict:
        """Reconstruct a state_dict from a particle tensor.

        Reverses ``flatten()``: slices the flat array by offset/numel,
        reshapes to original shape, casts back to original dtype, and
        handles shared parameters by assigning the canonical tensor to
        all alias keys.

        Args:
            particles: Tensor of shape ``(N, particle_dim)``.

        Returns:
            ``OrderedDict`` compatible with ``model.load_state_dict()``.
        """
        if self.total_params == 0:
            return OrderedDict()

        flat = particles.reshape(-1)
        reconstructed: dict[str, torch.Tensor] = {}

        for entry in self.entries:
            param = flat[entry.offset : entry.offset + entry.numel]
            param = param.reshape(entry.shape).to(entry.dtype)
            reconstructed[entry.key] = param

            # Shared parameters: assign the same tensor to alias keys
            for alias_key in entry.shared_with:
                reconstructed[alias_key] = param

        # Return in original key order
        result = OrderedDict()
        for key in self._all_keys:
            result[key] = reconstructed[key]

        return result
