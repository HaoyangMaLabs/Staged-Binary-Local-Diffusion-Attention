"""Exact sparse path-cover topologies for left-orbit v2.

Every route here gathers exact token positions.  There is no pooling, KV
compression, or dense ``L x L`` mask.  The topology controls only which exact
leftward positions are exposed at each layer.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .attention_common import _HeadRMSNorm, _apply_rope, _repeat_kv

from .topology import PATH_TOPOLOGIES, sbda_stage, topology_offsets


def build_binary_path_indices(
    length: int,
    *,
    layer: int,
    device: torch.device | None = None,
) -> Tensor:
    """Return exact ``[query, self/skip]`` source indices for one layer."""
    if length < 1:
        raise ValueError("length must be positive")
    if layer < 1:
        raise ValueError("layer must be positive")
    positions = torch.arange(length, device=device, dtype=torch.long)
    offsets = topology_offsets(layer, topology="binary")
    indices = [
        (positions - offset).clamp_min(0)
        for offset in offsets
    ]
    return torch.stack(indices, dim=-1)


def binary_path_reachability(length: int, layers: int) -> list[set[int]]:
    """Return structurally reachable source offsets for one query."""
    if length < 1 or layers < 1:
        raise ValueError("length and layers must be positive")
    reachable = [{0}]
    for layer in range(layers):
        stride = 2 ** layer
        previous = reachable[-1]
        reachable.append(previous | {offset + stride for offset in previous})
    return [set(offset for offset in group if offset < length) for group in reachable]


def path_reachability(
    length: int,
    layers: int,
    *,
    topology: str = "binary",
) -> list[set[int]]:
    """Return structural reachable offsets for a selected topology."""
    if length < 1 or layers < 1:
        raise ValueError("length and layers must be positive")
    reachable = [{0}]
    for layer in range(1, layers + 1):
        previous = reachable[-1]
        offsets = topology_offsets(
            layer, topology=topology, total_layers=layers,
        )
        expanded = {
            previous_offset + offset
            for previous_offset in previous
            for offset in offsets
        }
        reachable.append({offset for offset in expanded if offset < length})
    return reachable


def path_multiplicity(
    length: int,
    layers: int,
    *,
    topology: str = "binary",
) -> list[int]:
    """Count distinct route compositions reaching each causal offset."""
    if length < 1 or layers < 1:
        raise ValueError("length and layers must be positive")
    counts = [0] * length
    counts[0] = 1
    for layer in range(1, layers + 1):
        next_counts = [0] * length
        for previous_offset, count in enumerate(counts):
            if count == 0:
                continue
            for offset in topology_offsets(
                layer, topology=topology, total_layers=layers,
            ):
                target = previous_offset + offset
                if target < length:
                    next_counts[target] += count
        counts = next_counts
    return counts


class BinaryPathAttention(nn.Module):
    """Causal exact-token attention with a selectable sparse topology."""

    def __init__(
        self,
        dim: int,
        heads: int,
        layer: int,
        *,
        kv_heads: int | None = None,
        rope_theta: float = 1_000_000.0,
        topology: str = "binary",
        total_layers: int = 12,
        use_sdpa: bool = True,
        use_fused: bool = True,
        backward_mode: str = "token",
    ) -> None:
        super().__init__()
        if dim <= 0 or heads <= 0 or dim % heads:
            raise ValueError("dim must be positive, heads must be positive, and divide dim")
        if layer < 1:
            raise ValueError("layer must be positive")
        self.dim = int(dim)
        self.heads = int(heads)
        self.layer = int(layer)
        self.total_layers = int(total_layers)
        self.use_sdpa = bool(use_sdpa)
        self.use_fused = bool(use_fused)
        if backward_mode not in ("token", "block", "key"):
            raise ValueError("backward_mode must be token, block, or key")
        self.backward_mode = backward_mode
        if topology not in PATH_TOPOLOGIES:
            raise ValueError(f"unknown path topology: {topology}")
        self.topology = topology
        self.register_buffer(
            "_fused_offsets",
            torch.tensor(
                topology_offsets(
                    self.layer, topology=self.topology,
                    total_layers=self.total_layers,
                ),
                dtype=torch.int32,
            ),
            persistent=False,
        )
        self.kv_heads = self.heads if kv_heads is None else int(kv_heads)
        if self.kv_heads <= 0 or self.heads % self.kv_heads:
            raise ValueError("heads must be divisible by kv_heads")
        self.head_dim = self.dim // self.heads
        self.rope_theta = float(rope_theta)
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive")

        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, self.kv_heads * self.head_dim, bias=False)
        self.v = nn.Linear(dim, self.kv_heads * self.head_dim, bias=False)
        self.q_norm = _HeadRMSNorm(self.head_dim)
        self.k_norm = _HeadRMSNorm(self.head_dim)
        self.out = nn.Linear(dim, dim, bias=False)
        # The SBDA topology is fixed for a layer.  Keep the per-sequence
        # indices out of the hot path; chapter lengths are dynamic, so cache
        # by (device, length) instead of registering one static buffer.
        self._path_cache: dict[tuple[str, int | None, int], tuple[Tensor, Tensor]] = {}
        self._path_cache_limit = 4

    def _default_path_tensors(
        self, length: int, device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        key = (device.type, device.index, length)
        cached = self._path_cache.get(key)
        if cached is not None:
            return cached
        positions = torch.arange(length, device=device, dtype=torch.long)
        offsets = topology_offsets(
            self.layer, topology=self.topology,
            total_layers=self.total_layers,
        )
        offset_tensor = torch.tensor(offsets, device=device, dtype=torch.long)
        indices = (positions[:, None] - offset_tensor[None, :]).clamp_min(0)
        # A clamped index at the beginning is not a real source for a skipped
        # edge.  This is also the only static validity condition, so it can be
        # reused by every batch and every forward for this sequence length.
        source_valid = positions[:, None] >= offset_tensor[None, :]
        if len(self._path_cache) >= self._path_cache_limit:
            # Chapter lengths are dynamic.  Bound the cache so a long corpus
            # with one unique length per chapter cannot retain all index
            # tensors on the GPU.
            self._path_cache.pop(next(iter(self._path_cache)))
        self._path_cache[key] = (indices, source_valid)
        return indices, source_valid

    def forward(
        self,
        x: Tensor,
        valid_mask: Tensor | None = None,
        seam_reset: Tensor | None = None,
        path_indices: Tensor | None = None,
        path_valid: Tensor | None = None,
    ) -> Tensor:
        if x.ndim != 3:
            raise ValueError("x must have shape [batch, sequence, dim]")
        batch, length, dim = x.shape
        if dim != self.dim:
            raise ValueError(f"expected dim={self.dim}, got {dim}")
        if valid_mask is not None and (
            valid_mask.shape != (batch, length) or valid_mask.dtype != torch.bool
        ):
            raise ValueError("valid_mask must be bool with input shape")
        if seam_reset is not None and (
            seam_reset.shape != (batch, length) or seam_reset.dtype != torch.bool
        ):
            raise ValueError("seam_reset must be bool with input shape")
        if path_indices is not None:
            if path_indices.ndim != 3 or path_indices.shape[:2] != (batch, length):
                raise ValueError(
                    "path_indices must have shape [batch, sequence, source_count]"
                )
            if path_indices.dtype != torch.long:
                raise ValueError("path_indices must use torch.long")
            if path_indices.device != x.device:
                raise ValueError("path_indices must be on the same device as x")
            if path_valid is None:
                path_valid = torch.ones_like(path_indices, dtype=torch.bool)
            elif (
                path_valid.shape != path_indices.shape
                or path_valid.dtype != torch.bool
                or path_valid.device != x.device
            ):
                raise ValueError("path_valid must match path_indices as a bool tensor")
            if path_indices.numel() and (
                int(path_indices.min()) < 0 or int(path_indices.max()) >= length
            ):
                raise ValueError("path_indices contains an out-of-range source")

        q = self.q(x).view(batch, length, self.heads, self.head_dim)
        k = self.k(x).view(batch, length, self.kv_heads, self.head_dim)
        v = self.v(x).view(batch, length, self.kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = _apply_rope(q, k, theta=self.rope_theta)
        q = q.permute(0, 2, 1, 3)
        # Keep compressed K/V in the fused GQA route.  Expanding them before
        # dispatch defeats the memory benefit of grouped-query attention.
        k_comp = k.permute(0, 2, 1, 3)
        v_comp = v.permute(0, 2, 1, 3)

        # The ordinary SBDA route has no row padding, seam, or caller-supplied
        # path table.  In that case the exact source set is fixed per layer,
        # so fuse source addressing, score, softmax, and value reduction.  The
        # first few positions are handled inside the kernel by masking offsets
        # that are not yet in range.  All other cases retain the reference
        # gather path below.
        # The chapter trainer always supplies ``valid_mask``.  Fixed windows
        # are right-padded per row, and causal SBDA never reads those future
        # padding cells from an otherwise-valid query.  Therefore a prefix
        # mask is safe for the fused route; only masks with an internal hole
        # need the reference implementation.  This keeps short tail windows
        # from silently falling back to the much slower gather path.
        prefix_mask = (
            valid_mask is None
            or valid_mask.shape[1] < 2
            or bool(torch.all(~valid_mask[:, 1:] | valid_mask[:, :-1]))
        )
        if (
            self.use_fused and self.use_sdpa and x.is_cuda
            and prefix_mask and seam_reset is None
            and path_indices is None and path_valid is None
        ):
            try:
                if self.backward_mode == "key":
                    if self.kv_heads < self.heads:
                        from .sbda_triton import fused_sbda_gqa_key_backward
                        fused_attention = fused_sbda_gqa_key_backward
                    else:
                        from .sbda_triton import fused_sbda_key_backward
                        fused_attention = fused_sbda_key_backward
                elif self.backward_mode == "block":
                    from .sbda_triton import fused_sbda_block_full
                    fused_attention = fused_sbda_block_full
                else:
                    from .sbda_triton import fused_sbda
                    fused_attention = fused_sbda

                offsets = self._fused_offsets
                if offsets.numel() in (2, 4):
                    if self.backward_mode == "key" and self.kv_heads < self.heads:
                        attended = fused_attention(q, k_comp, v_comp, offsets)
                    else:
                        k_full = _repeat_kv(
                            k_comp.transpose(1, 2), self.heads // self.kv_heads,
                        ).permute(0, 2, 1, 3)
                        v_full = _repeat_kv(
                            v_comp.transpose(1, 2), self.heads // self.kv_heads,
                        ).permute(0, 2, 1, 3)
                        attended = fused_attention(q, k_full, v_full, offsets)
                        attended = attended.transpose(1, 2).reshape(batch, length, dim)
                    if self.backward_mode == "key" and self.kv_heads < self.heads:
                        attended = attended.transpose(1, 2).reshape(batch, length, dim)
                    out = self.out(attended)
                    # Keep the fused path's row semantics identical to the
                    # reference path.  Invalid right-padded query rows must
                    # not leave arbitrary projected values in the residual
                    # stream, even though causal source addressing prevents
                    # valid rows from reading them.
                    if valid_mask is not None:
                        out = out * valid_mask[:, :, None].to(dtype=out.dtype)
                    return out
            except (ImportError, RuntimeError, TypeError, ValueError):
                # Triton is optional and some GPUs/dtypes do not support its
                # atomics.  Fall through to the numerically identical eager
                # reference implementation in those environments.
                pass

        # Reference and non-key fused paths use the conventional expanded
        # layout.  This allocation is intentionally delayed until after the
        # native-GQA fast path has had a chance to return.
        k = _repeat_kv(
            k_comp.transpose(1, 2), self.heads // self.kv_heads,
        ).permute(0, 2, 1, 3)
        v = _repeat_kv(
            v_comp.transpose(1, 2), self.heads // self.kv_heads,
        ).permute(0, 2, 1, 3)

        if path_indices is not None:
            indices = path_indices
            source_count = indices.shape[-1]
            base_source_valid = torch.ones(
                (length, source_count), dtype=torch.bool, device=x.device,
            )
        else:
            indices, base_source_valid = self._default_path_tensors(
                length, x.device,
            )
            source_count = indices.shape[-1]
        # Exact source gathers; no pooling and no dense LxL mask.
        indices_batched = (
            indices.unsqueeze(0).expand(batch, -1, -1)
            if indices.ndim == 2 else indices
        )
        source_index = indices_batched[:, None, :, :, None].expand(
            batch, self.heads, length, source_count, self.head_dim,
        )
        source_k = k.unsqueeze(3).expand(
            batch, self.heads, length, source_count, self.head_dim,
        ).gather(2, source_index)
        source_v = v.unsqueeze(3).expand(
            batch, self.heads, length, source_count, self.head_dim,
        ).gather(2, source_index)

        source_valid = base_source_valid.unsqueeze(0).expand(
            batch, -1, -1,
        )
        if path_valid is not None:
            source_valid = source_valid & path_valid
        if path_indices is not None:
            positions = torch.arange(length, device=x.device)
            source_valid = source_valid & (indices_batched <= positions[None, :, None])
        if valid_mask is not None:
            source_valid = source_valid & valid_mask.unsqueeze(-1).expand(
                -1, -1, source_count
            ).gather(1, indices_batched)
            if seam_reset is not None:
                segment_ids = seam_reset.to(dtype=torch.int64).cumsum(dim=1)
                source_segments = segment_ids.unsqueeze(-1).expand(
                    -1, -1, source_count
                ).gather(1, indices_batched)
                source_valid = source_valid & (
                    source_segments == segment_ids[:, :, None]
                )
            # Padded query rows are discarded after attention; keep a finite
            # self key for those rows so softmax never sees an all-masked row.
            source_valid[:, :, 0] |= ~valid_mask
        elif seam_reset is not None:
            segment_ids = seam_reset.to(dtype=torch.int64).cumsum(dim=1)
            source_valid = source_valid & (
                segment_ids.unsqueeze(-1).expand(
                    -1, -1, source_count
                ).gather(1, indices_batched) == segment_ids[:, :, None]
            )

        if self.use_sdpa:
            # SBDA has only S=2 or S=4 sources.  A regular pair of batched
            # matmuls is faster than dispatching one tiny 5-D SDPA problem per
            # token, while retaining exactly the same scaled dot-product
            # computation and avoiding an LxL matrix.
            scores = torch.matmul(
                q.unsqueeze(-2), source_k.transpose(-1, -2),
            ).squeeze(-2)
            scores = scores * (self.head_dim ** -0.5)
            scores = scores.masked_fill(
                ~source_valid[:, None, :, :], torch.finfo(scores.dtype).min,
            )
            weights = scores.softmax(dim=-1)
            weights = weights.masked_fill(~source_valid[:, None, :, :], 0.0)
            attended = torch.matmul(
                weights.unsqueeze(-2), source_v,
            ).squeeze(-2)
        else:
            scores = (q.unsqueeze(3) * source_k).sum(dim=-1)
            scores = scores * (self.head_dim ** -0.5)
            scores = scores.masked_fill(
                ~source_valid[:, None, :, :], torch.finfo(scores.dtype).min,
            )
            weights = scores.softmax(dim=-1)
            weights = weights.masked_fill(~source_valid[:, None, :, :], 0.0)
            attended = (weights.unsqueeze(-1) * source_v).sum(dim=3)
        attended = attended.transpose(1, 2).reshape(batch, length, dim)
        out = self.out(attended)
        if valid_mask is not None:
            out = out * valid_mask[:, :, None].to(dtype=out.dtype)
        return out


class SBDAttention(BinaryPathAttention):
    """Staged Binary-Local Diffusion Attention baseline."""

    def __init__(
        self,
        dim: int,
        heads: int,
        layer: int,
        *,
        kv_heads: int | None = None,
        rope_theta: float = 1_000_000.0,
        total_layers: int = 12,
        use_sdpa: bool = True,
        use_fused: bool = True,
        backward_mode: str = "key",
    ) -> None:
        super().__init__(
            dim, heads, layer,
            kv_heads=kv_heads,
            rope_theta=rope_theta,
            topology="sbda",
            total_layers=total_layers,
            use_sdpa=use_sdpa,
            use_fused=use_fused,
            backward_mode=backward_mode,
        )


# Keep the correctly expanded public spelling while preserving the original
# short-name typo used by older checkpoints/scripts.
SBDAAttention = SBDAttention


__all__ = [
    "BinaryPathAttention",
    "SBDAAttention",
    "SBDAttention",
    "PATH_TOPOLOGIES",
    "binary_path_reachability",
    "build_binary_path_indices",
    "path_reachability",
    "path_multiplicity",
    "topology_offsets",
    "sbda_stage",
]
