"""Parameter-particle transformation utilities."""

from __future__ import annotations

import heapq
import itertools
import logging
import operator
from collections import OrderedDict
from dataclasses import dataclass, replace
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


def _byte_span(tensor: torch.Tensor) -> Tuple[int, int]:
    """Half-open absolute byte range this tensor can touch, from its address and strides."""
    esize = tensor.element_size()
    start, stride = tensor.data_ptr(), tensor.stride()
    reach = (sum(map(operator.mul, tensor.shape, stride)) - sum(stride)) * esize
    return start, start + reach + esize


def _shares_bytes(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Whether two tensors with overlapping spans touch a common byte, not only a common span."""
    (a_start, a_stop), (b_start, b_stop) = _byte_span(a), _byte_span(b)
    if a_start == b_start or (a if a_start < b_start else b).is_contiguous():
        return True
    stride, esize = a.stride(), a.element_size()
    if stride == b.stride() and esize == b.element_size() and max(stride, default=0) > 0:
        dim = stride.index(max(stride))
        period = stride[dim] * esize
        gap = (b_start - a_start) % period
        a_width = a_stop - a_start - (a.shape[dim] - 1) * period
        b_width = b_stop - b_start - (b.shape[dim] - 1) * period
        if a_width <= gap <= period - b_width:
            return False
    base = min(a_start, b_start)
    mask = torch.zeros(max(a_stop, b_stop) - base, dtype=torch.bool)

    def byte_view(t: torch.Tensor) -> torch.Tensor:
        esize = t.element_size()
        return mask.as_strided((*t.shape, esize), (*(s * esize for s in t.stride()), 1), t.data_ptr() - base)

    byte_view(a).fill_(True)
    return bool(byte_view(b).any())


def _dense_parts(tensor: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """Strided tensors holding the bytes of ``tensor``: itself, none if unmaterialized, or sparse or nested parts."""
    layout = tensor.layout
    if layout == torch.strided and not tensor.is_nested:
        return () if tensor.is_meta or nn.parameter.is_lazy(tensor) else (tensor,)
    if layout == torch.sparse_coo:
        return tensor._indices(), tensor._values()
    if layout in (torch.sparse_csr, torch.sparse_bsr):
        return tensor.crow_indices(), tensor.col_indices(), tensor.values()
    if layout in (torch.sparse_csc, torch.sparse_bsc):
        return tensor.ccol_indices(), tensor.row_indices(), tensor.values()
    if layout == torch.jagged:
        return tuple(part for part in (tensor.values(), tensor.offsets(), tensor.lengths()) if part is not None)
    return (tensor.values(),) if tensor.is_nested else ()


_validated_signature: list | None = None


def _trainable_owners(model: nn.Module) -> dict[str, int]:
    """Each name of a trainable parameter mapped to its id; raises if another registered tensor shares its bytes."""
    global _validated_signature
    params = list(model.named_parameters(remove_duplicate=False))
    owners = {name: id(param) for name, param in params if param.requires_grad}
    trainable = set(owners.values())
    tensors: dict[tuple[int, int], tuple[str, torch.Tensor]] = {}
    for is_buffer, named in enumerate((params, model.named_buffers(remove_duplicate=False))):
        for name, tensor in named:
            tensors.setdefault((id(tensor), is_buffer), (name, tensor))

    rows: dict[torch.device, list] = {}
    for (ident, is_buffer), (name, tensor) in tensors.items():
        for part in _dense_parts(tensor):
            if part.numel() > 0:
                trained = not is_buffer and ident in trainable
                rows.setdefault(part.device, []).append((*_byte_span(part), trained, name, part))

    def overlapping_pairs():
        for row in rows.values():
            row.sort(key=lambda span: span[0])
            others_open: list = []
            trained_open: list = []
            for i, (start, stop, trained, name, tensor) in enumerate(row):
                for heap in (others_open, trained_open):
                    while heap and heap[0][0] <= start:
                        heapq.heappop(heap)
                for _, _, other_name, other in trained_open:
                    yield name, other_name, tensor, other
                for _, _, other_name, other in others_open if trained else ():
                    yield other_name, name, other, tensor
                heapq.heappush(trained_open if trained else others_open, (stop, i, name, tensor))

    pairs = overlapping_pairs()
    first = next(pairs, None)
    if first is None:
        return owners
    signature = [(device, *span[:3], span[4].shape, span[4].stride()) for device, row in rows.items() for span in row]
    if signature == _validated_signature:
        return owners
    for name, trained_name, a, b in itertools.chain((first,), pairs):
        if _shares_bytes(a, b):
            raise ValueError(
                f"{name!r} and trainable parameter {trained_name!r} share storage but are different "
                f"tensors, so candidates would not reach {name!r}. Tie weights by assigning the "
                f"Parameter itself (module.b = module.a); otherwise register a clone of {name!r} "
                "or compute it in forward."
            )
    _validated_signature = signature
    return owners


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
        model = getattr(model, "_orig_mod", model)
        sd = {k: v for k, v in model.state_dict().items() if isinstance(v, torch.Tensor)}

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

        canonical_entries: list[ParamEntry] = []
        shared_map: dict[str, list[str]] = {}  # canonical_key -> [alias keys]
        offset = 0

        owners = _trainable_owners(model)
        canonical_keys: dict[int, str] = {}
        for name, ident in owners.items():
            canonical_keys.setdefault(ident, name)

        all_keys: list[str] = []

        for key, tensor in sd.items():
            # Only trainable params belong here: buffers are frozen, and a running stat would drift with no signal.
            if key not in owners:
                continue

            all_keys.append(key)

            alias_of = canonical_keys[owners[key]]
            if alias_of != key:
                shared_map.setdefault(alias_of, [alias_of]).append(key)
                continue

            numel = tensor.numel()
            module_path = key.rsplit(".", 1)[0] if "." in key else ""

            canonical_entries.append(
                ParamEntry(
                    key=key,
                    shape=tuple(tensor.shape),
                    dtype=tensor.dtype,
                    offset=offset,
                    numel=numel,
                    requires_grad=True,
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
                entry = replace(entry, shared_with=tuple(k for k in aliases if k != entry.key))
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
        model = getattr(model, "_orig_mod", model)
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
