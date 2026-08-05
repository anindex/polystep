"""Parameter-particle transformation utilities."""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import Tuple


import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)


def create_generator(seed: int, device: torch.device) -> torch.Generator:
    """Create a seeded ``torch.Generator`` on the given device."""
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return gen


def _element_span(tensor: torch.Tensor) -> Tuple[int, int]:
    """Half-open byte range this tensor can touch in its storage, from strides."""
    esize = tensor.element_size()
    start = tensor.storage_offset() * esize
    reach = sum((size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride())) * esize
    return start, start + reach + esize


@dataclass(frozen=True)
class ParamEntry:
    """Metadata for a single parameter/buffer in the layout."""

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
    """Frozen layout describing how to flatten/unflatten nn.Module parameters; created once via ``from_module()``."""

    entries: Tuple[ParamEntry, ...]
    total_params: int
    padded_size: int
    particle_dim: int
    dominant_dtype: torch.dtype
    shared_groups: Tuple[Tuple[str, ...], ...] = ()
    _all_keys: Tuple[str, ...] = ()

    @classmethod
    def from_module(
        cls,
        model: nn.Module,
        particle_dim: int = 2,
    ) -> ParamLayout:
        """Create a ``ParamLayout`` from any ``nn.Module``; shared tensors are deduplicated."""
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

        # Key shared storage on the storage, not data_ptr: views at different offsets would look independent and write the same bytes twice.
        seen_storage: dict[int, list[tuple[str, int, int, tuple]]] = {}
        canonical_entries: list[ParamEntry] = []
        shared_map: dict[str, list[str]] = {}  # canonical_key -> [alias keys]
        offset = 0

        # Collect trainable data_ptrs so tied weights are detected under a different name too.
        param_grad = {}
        trainable_ptrs: set[int] = set()
        for name, param in model.named_parameters():
            param_grad[name] = param.requires_grad
            if param.requires_grad:
                trainable_ptrs.add(param.data_ptr())

        all_keys: list[str] = []

        for key, tensor in sd.items():
            # Only trainable params belong here: buffers are frozen, and a running stat would drift with no signal.
            # Tied params may be absent from named_parameters(), so fall back to data_ptr.
            requires_grad = param_grad.get(key, False)
            is_trainable_alias = tensor.data_ptr() in trainable_ptrs
            if not requires_grad and not is_trainable_alias:
                continue

            all_keys.append(key)

            if tensor.numel() > 0:
                storage_id = tensor.untyped_storage().data_ptr()
                start, stop = _element_span(tensor)
                view = (tuple(tensor.shape), tensor.stride(), tensor.storage_offset())
                group = seen_storage.setdefault(storage_id, [])
                alias_of = None
                for canonical_key, c_start, c_stop, canonical_view in group:
                    if stop <= c_start or c_stop <= start:
                        continue  # disjoint slices of one buffer stay independent
                    # Only an identical view is a tie: a weight and its transpose match on pointer and shape, and merging would transpose one.
                    if view != canonical_view:
                        raise ValueError(
                            f"{key!r} and {canonical_key!r} share storage but are different views "
                            f"(shape/stride/offset {view} vs {canonical_view}). Overlapping views of "
                            "one buffer cannot be laid out as independent parameters."
                        )
                    alias_of = canonical_key
                    break
                if alias_of is not None:
                    shared_map.setdefault(alias_of, [alias_of]).append(key)
                    continue
                group.append((key, start, stop, view))

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

        shared_groups: list[Tuple[str, ...]] = []
        updated_entries: list[ParamEntry] = []
        for entry in canonical_entries:
            if entry.key in shared_map:
                aliases = tuple(shared_map[entry.key])
                shared_groups.append(aliases)
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

        # Log the dedup, or a shared embedding leaves an unexplained parameter-count gap.
        if shared_groups:
            tied_summary = ", ".join(f"{group[0]} <- {{{', '.join(group[1:])}}}" for group in shared_groups)
            logger.info(
                "ParamLayout deduplicated tied / shared weights: %s",
                tied_summary,
            )

        dtype_counts: dict[torch.dtype, int] = {}
        for entry in updated_entries:
            dtype_counts[entry.dtype] = dtype_counts.get(entry.dtype, 0) + entry.numel
        dominant_dtype = max(dtype_counts, key=dtype_counts.get) if dtype_counts else torch.float32

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

    def batch_unflatten(self, particles_batch: torch.Tensor) -> dict[str, torch.Tensor]:
        """Convert N particle vectors to stacked param dicts for vmap; tied weights appear once, under the canonical key."""
        if self.total_params == 0:
            return {}

        N = particles_batch.shape[0]
        flat = particles_batch.reshape(N, -1)
        stacked: dict[str, torch.Tensor] = {}

        for entry in self.entries:
            param = flat[:, entry.offset : entry.offset + entry.numel]
            param = param.reshape(N, *entry.shape)
            # Only cast when the dtype differs, to skip a no-op .to() kernel.
            if entry.dtype != self.dominant_dtype:
                param = param.to(entry.dtype)
            stacked[entry.key] = param

        return stacked

    def flatten(self, model: nn.Module) -> torch.Tensor:
        """Flatten a model state_dict to a 2D particle tensor."""
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
        """Reconstruct a state_dict from a particle tensor, assigning shared params to all alias keys."""
        if self.total_params == 0:
            return OrderedDict()

        flat = particles.reshape(-1)
        reconstructed: dict[str, torch.Tensor] = {}

        for entry in self.entries:
            param = flat[entry.offset : entry.offset + entry.numel]
            param = param.reshape(entry.shape).to(entry.dtype)
            reconstructed[entry.key] = param

            for alias_key in entry.shared_with:
                reconstructed[alias_key] = param

        result = OrderedDict()
        for key in self._all_keys:
            result[key] = reconstructed[key]

        return result
