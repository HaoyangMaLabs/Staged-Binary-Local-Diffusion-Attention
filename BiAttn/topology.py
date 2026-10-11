"""Pure-Python causal source schedules shared by models and graph scans."""

PATH_TOPOLOGIES = (
    "binary", "local_jump", "mesh", "torus", "butterfly", "high4",
    "radix4", "radix8", "mixed_local_radix4", "longformer_local",
    "sparse_local_stride", "dinat_like", "longnet_like", "power_attention",
    # Internal layer-expansion hypotheses used by BinaryDiffusionAttention.
    "h0", "h1", "h2", "h3", "h4_greedy", "h5", "h6", "h7", "h8",
    "h8a", "h8b", "h8c", "h8d", "h8e", "h8f", "h8g", "sbda",
)


def sbda_stage(layer: int) -> str:
    """Return the stage used by the staged binary-local route."""
    if layer < 1:
        raise ValueError("layer must be positive")
    return "binary" if layer <= 2 else "local_diffusion"


def _unique_offsets(offsets: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(dict.fromkeys(int(offset) for offset in offsets if offset >= 0))


def topology_offsets(
    layer: int,
    *,
    topology: str = "binary",
    total_layers: int = 12,
) -> tuple[int, ...]:
    """Return exact leftward offsets for a one-dimensional causal topology.

    ``0`` is the carry/self source.  The other offsets are exact token
    positions, not summaries.  ``mesh`` and ``torus`` are causal unrollings:
    a true future-wrapping torus is illegal for next-token prediction, so its
    wrap-like shortcuts are represented only by positive leftward offsets.
    """
    if layer < 1:
        raise ValueError("layer must be positive")
    if total_layers < 1:
        raise ValueError("total_layers must be positive")
    if topology not in PATH_TOPOLOGIES:
        raise ValueError(f"unknown path topology: {topology}")
    level = layer - 1
    stride = 2 ** level
    if topology == "h0":
        return (0, stride)
    if topology == "h1":
        return _unique_offsets((0, 1, 2, stride))
    if topology == "h2":
        return _unique_offsets((0, stride, 3 * stride))
    if topology == "h3":
        return _unique_offsets((0, *range(stride, 2 * stride)))
    if topology == "h4_greedy":
        # Fallback for callers that do not provide the length-dependent greedy
        # path table.  The WikiText benchmark supplies the exact greedy table
        # through path_indices; this deterministic radix-3 route keeps the
        # topology valid for ordinary direct use.
        return _unique_offsets((0, 3 ** level, 2 * (3 ** level)))
    if topology == "h5":
        return (0, 1, 2) if layer == 1 else (0, 2 ** layer)
    if topology == "h6":
        return _unique_offsets((0, stride, 2 * stride + 1))
    if topology == "h7":
        return (0, 1, 2) if layer <= 2 else (0, stride)
    if topology in ("h8", "sbda"):
        return (0, stride) if layer <= 2 else _unique_offsets((0, 1, 2, stride))
    if topology == "h8a":
        return (0, stride) if layer <= 2 else _unique_offsets((0, 1, stride))
    if topology == "h8b":
        return (0, stride) if layer <= 2 else _unique_offsets((0, 2, stride))
    if topology == "h8c":
        return (0, stride) if layer <= 2 else _unique_offsets((0, 1, 2, stride))
    if topology == "h8d":
        return (
            (0, 1, 2, stride)
            if layer >= 3 and layer % 2 == 1
            else (0, stride)
        )
    if topology == "h8e":
        return (
            _unique_offsets((0, 1, 2, stride))
            if layer > max(2, total_layers - 2)
            else (0, stride)
        )
    if topology == "h8f":
        return (
            (0, stride)
            if layer <= 2 else _unique_offsets((0, 1, 2, 3, stride))
        )
    if topology == "h8g":
        return (
            (0, stride)
            if layer <= 2 else _unique_offsets((0, 1, 2, 3, 4, stride))
        )
    if topology == "binary":
        return (0, stride)
    if topology == "local_jump":
        return _unique_offsets((0, 1, 2, stride))
    if topology == "mesh":
        # A coarse vertical jump plus its one-cell horizontal neighbors.
        return _unique_offsets((0, 1, stride, stride + 1))
    if topology == "torus":
        # Causal ring-like shortcuts, without illegal future reads.
        return _unique_offsets((0, 1, max(1, stride - 1), stride, stride + 1))
    if topology == "butterfly":
        # Four exact lanes on a dyadic scale; this is a broad, overlapping
        # butterfly rather than a radix-4 schedule, so it remains useful at
        # the same depth as the binary baseline.
        return _unique_offsets((0, stride, 2 * stride, 3 * stride))
    if topology == "high4":
        # High-connectivity local-to-global route.  It keeps a tiny local core
        # and adds four logarithmic shortcuts.
        return _unique_offsets((0, 1, 2, stride, 2 * stride, 3 * stride))
    if topology == "longformer_local":
        # Causal fixed sliding-window analogue.  Longformer also supports
        # optional dilation and global tokens; those are separate variants
        # below because future/global reads are not directly valid here.
        return (0, 1, 2, 3)
    if topology == "sparse_local_stride":
        # Fixed local-plus-strided analogue of Sparse Transformer.  The
        # strided factor is fixed across layers, unlike our dyadic route.
        fixed_stride = max(4, 2 ** max(1, total_layers // 2))
        return _unique_offsets((0, 1, 2, 3, fixed_stride))
    if topology == "dinat_like":
        # Causal 1-D analogue of local plus dilated neighborhood attention.
        dilation = 2 * stride
        return _unique_offsets((0, 1, 2, 3, dilation, 2 * dilation, 3 * dilation))
    if topology == "longnet_like":
        # Causal adaptation of LongNet's mixture of a local pattern and a
        # geometrically dilated pattern.  The published method uses segment
        # gather/scatter and dynamic mixing; this exact-token route preserves
        # the geometric local+dilation comparison while keeping the current
        # backbone and causal source semantics.
        dilation = stride
        return _unique_offsets((0, 1, 2, 3, dilation, 2 * dilation, 3 * dilation))
    if topology == "power_attention":
        # Token-level analogue of PowerAttention (Chen et al., 2025): retain
        # a local core and expose every power-of-two exact-token offset at
        # every layer.  The published implementation is block based and also
        # has sink tokens; those details are intentionally not hidden here.
        power_offsets = tuple(1 << exponent for exponent in range(total_layers))
        return _unique_offsets((0, 1, 2, 3, *power_offsets))
    if topology == "radix4":
        radix_stride = 4 ** level
        return tuple(q * radix_stride for q in range(4))
    if topology == "radix8":
        radix_stride = 8 ** level
        return tuple(q * radix_stride for q in range(8))
    split = max(1, total_layers // 2)
    if level < split:
        radix_stride = 2 ** level
        return (0, radix_stride)
    else:
        radix_stride = (2 ** split) * (4 ** (level - split))
    return tuple(q * radix_stride for q in range(4))


def greedy_offsets(
    reachable: set[int], used: set[int], length: int, budget: int = 2,
) -> tuple[int, ...]:
    """Choose unused offsets greedily for marginal reachable coverage."""
    chosen = [0]
    local_reachable = set(reachable)
    for _ in range(budget):
        candidates = (offset for offset in range(1, length) if offset not in used)
        candidates = tuple(candidates)
        if not candidates:
            break
        best = max(
            candidates,
            key=lambda offset: (
                len(local_reachable | {
                    item + offset for item in local_reachable
                    if item + offset < length
                }),
                -offset,
            ),
        )
        chosen.append(best)
        used.add(best)
        local_reachable |= {
            item + best for item in local_reachable if item + best < length
        }
    return tuple(chosen)


def greedy_internal_offsets(
    length: int, layers: int, *, budget: int = 2,
) -> list[tuple[int, ...]]:
    """Build the length-dependent H4 schedule with unused-offset tie breaking."""
    if length < 1 or layers < 1 or budget < 1:
        raise ValueError("length, layers and budget must be positive")
    reachable = {0}
    used: set[int] = set()
    schedule: list[tuple[int, ...]] = []
    for _ in range(layers):
        offsets = greedy_offsets(reachable, used, length, budget)
        schedule.append(offsets)
        reachable = {
            item + offset for item in reachable for offset in offsets
            if item + offset < length
        }
    return schedule
