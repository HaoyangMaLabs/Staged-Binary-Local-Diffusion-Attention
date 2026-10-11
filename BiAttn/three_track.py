"""MiniMind-compatible left attention and pooled bidirectional tracks.

At position ``t`` and dyadic width ``w``, the left track is the mean of the
trailing left half ``[t-w+1, t-w/2]``.  The optional right track mirrors that
layout, then shifts it one position to the right.  In triangle batches the
right track is masked by the row's valid prefix, so it can see the rest of
the input prefix but never the next-token target.

All operations are ordinary PyTorch operations; the two/three source
attention is flattened into batched BMM below.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .attention_common import _HeadRMSNorm, _apply_rope, _repeat_kv

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
except ImportError:  # Optional CUDA acceleration; SDPA remains the fallback.
    flash_attn_func = None
    flash_attn_varlen_func = None

def _right_padded_lengths(valid_mask: Tensor) -> Tensor | None:
    """Return prefix lengths when ``valid_mask`` is right-padded.

    Chapter batches are padded on the right.  Treating that layout specially
    lets SDPA use its fused causal path for each length group; constructing a
    dense padding mask would otherwise allocate a ``T x T`` temporary.
    """

    lengths = valid_mask.sum(dim=-1, dtype=torch.int64)
    positions = torch.arange(
        valid_mask.shape[-1], device=valid_mask.device,
    )
    expected = positions[None, :] < lengths[:, None]
    if torch.equal(valid_mask, expected):
        return lengths
    return None


def _right_padded_causal_sdpa(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    valid_mask: Tensor,
) -> Tensor:
    """Causal SDPA for right-padded rows without a padding attention mask.

    Right padding is always to the future of every valid query.  Causal SDPA
    therefore already excludes padded keys; only padded query outputs need to
    be cleared.  Keeping one batched call is important here: grouping every
    distinct chapter length would save little work but cause many tiny kernel
    launches.
    """

    lengths = _right_padded_lengths(valid_mask)
    if lengths is None:
        raise ValueError("valid_mask must be right-padded")
    attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return attended * valid_mask[:, None, :, None].to(dtype=attended.dtype)


def _right_padded_block_sdpa(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    valid_mask: Tensor,
) -> Tensor:
    """Block-causal SDPA for right-padded rows without a block mask."""

    lengths = _right_padded_lengths(valid_mask)
    if lengths is None:
        raise ValueError("valid_mask must be right-padded")
    attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return attended * valid_mask[:, None, :, None].to(dtype=attended.dtype)


def _two_stage_causal_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    valid_mask: Tensor | None,
    window_size: int,
) -> Tensor:
    """Full causal attention as local-softmax then window-softmax."""

    if window_size < 1:
        raise ValueError("two-stage attention window_size must be positive")
    batch, heads, length, head_dim = q.shape
    blocks = (length + window_size - 1) // window_size
    padded = blocks * window_size
    pad = padded - length
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1))
    scores = scores * (head_dim ** -0.5)
    if pad:
        scores = F.pad(scores, (0, pad), value=-float("inf"))

    key_positions = torch.arange(padded, device=q.device)
    query_positions = torch.arange(length, device=q.device)
    causal = key_positions[None, :] <= query_positions[:, None]
    causal = causal[None, None, :, :].expand(batch, 1, length, padded)
    if valid_mask is not None:
        if valid_mask.shape != (batch, length) or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be bool with input shape")
        key_valid = F.pad(valid_mask, (0, pad), value=False)
        causal = causal & key_valid[:, None, None, :]
        query_valid = valid_mask
    else:
        query_valid = None
    has_key = causal.any(dim=-1, keepdim=True)
    causal = causal | (~has_key & (key_positions == 0)[None, None, None, :])

    score_blocks = scores.reshape(batch, heads, length, blocks, window_size)
    mask_blocks = causal.reshape(batch, 1, length, blocks, window_size)
    safe_scores = score_blocks.masked_fill(~mask_blocks, -float("inf"))
    local_logz = torch.logsumexp(safe_scores, dim=-1)
    # A query has completely masked future blocks.  Softmax over an all-
    # -inf row is NaN; use a finite floor and zero those entries afterwards.
    finite_scores = safe_scores.masked_fill(
        ~mask_blocks, torch.finfo(safe_scores.dtype).min,
    )
    local_probs = torch.softmax(finite_scores, dim=-1)
    local_probs = local_probs * mask_blocks.to(local_probs.dtype)

    v_pad = F.pad(v.float(), (0, 0, 0, pad))
    v_blocks = v_pad.reshape(batch, heads, blocks, window_size, head_dim)
    local_context = torch.einsum(
        "bhlgw,bhgwd->bhlgd", local_probs, v_blocks,
    )
    block_mask = mask_blocks.any(dim=-1)
    second_scores = local_logz.masked_fill(~block_mask, -float("inf"))
    second_probs = torch.softmax(second_scores, dim=-1)
    attended = torch.einsum(
        "bhlg,bhlgd->bhld", second_probs, local_context,
    )
    if query_valid is not None:
        attended = attended * query_valid[:, None, :, None].to(attended.dtype)
    return attended.to(dtype=v.dtype)


class _StreamingCausalAttention(torch.autograd.Function):
    """Exact causal softmax attention without materializing the L by L scores.

    This is the same algebra as the two-stage factorization, but keeps only
    one key block and an online log-sum-exp accumulator.  The custom backward
    recomputes each block, so autograd does not retain all block score planes.
    """

    @staticmethod
    def forward(ctx, q, k, v, valid_mask, window_size):
        qf, kf, vf = q.float(), k.float(), v.float()
        batch, heads, length, head_dim = qf.shape
        scale = head_dim ** -0.5
        key_positions = torch.arange(length, device=q.device)
        query_positions = key_positions
        if valid_mask is None:
            key_valid = torch.ones(
                (batch, length), dtype=torch.bool, device=q.device,
            )
            query_valid = None
        else:
            key_valid = valid_mask.clone()
            key_valid[:, 0] |= ~key_valid.any(dim=1)
            query_valid = valid_mask

        m = torch.full(
            (batch, heads, length, 1), -torch.inf,
            dtype=torch.float32, device=q.device,
        )
        z = torch.zeros_like(m)
        acc = torch.zeros(
            (batch, heads, length, head_dim), dtype=torch.float32,
            device=q.device,
        )
        for start in range(0, length, window_size):
            end = min(start + window_size, length)
            kb = kf[:, :, start:end, :]
            vb = vf[:, :, start:end, :]
            scores = torch.matmul(qf, kb.transpose(-2, -1)) * scale
            causal = key_positions[start:end][None, :] <= query_positions[:, None]
            mask = causal[None, None, :, :] & key_valid[:, None, start:end, None].transpose(-1, -2)
            has = mask.any(dim=-1, keepdim=True)
            block_m = scores.masked_fill(~mask, -torch.inf).amax(dim=-1, keepdim=True)
            block_m_safe = torch.where(has, block_m, torch.zeros_like(block_m))
            weights = torch.where(
                mask, torch.exp(scores - block_m_safe), torch.zeros_like(scores),
            )
            block_z = weights.sum(dim=-1, keepdim=True)
            block_acc = torch.matmul(weights, vb)
            new_m = torch.where(has, torch.maximum(m, block_m), m)
            old_scale = torch.where(
                z > 0, torch.exp(m - new_m), torch.zeros_like(z),
            )
            block_scale = torch.where(
                has, torch.exp(block_m_safe - new_m), torch.zeros_like(z),
            )
            z = old_scale * z + block_scale * block_z
            acc = old_scale * acc + block_scale * block_acc
            m = new_m
        out = acc / z.clamp_min(torch.finfo(torch.float32).tiny)
        if query_valid is not None:
            out = out * query_valid[:, None, :, None].to(out.dtype)
        saved_query_valid = (
            torch.ones((batch, length), dtype=torch.bool, device=q.device)
            if query_valid is None else query_valid
        )
        ctx.save_for_backward(q, k, v, m, z, key_valid, saved_query_valid)
        ctx.window_size = window_size
        ctx.scale = scale
        return out.to(dtype=v.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, m, z, key_valid, query_valid = ctx.saved_tensors
        qf, kf, vf = q.float(), k.float(), v.float()
        go = grad_output.float() * query_valid[:, None, :, None].to(torch.float32)
        batch, heads, length, head_dim = qf.shape
        key_positions = torch.arange(length, device=q.device)
        query_positions = key_positions
        dq = torch.zeros_like(qf)
        dk = torch.zeros_like(kf)
        dv = torch.zeros_like(vf)
        for start in range(0, length, ctx.window_size):
            end = min(start + ctx.window_size, length)
            kb = kf[:, :, start:end, :]
            vb = vf[:, :, start:end, :]
            scores = torch.matmul(qf, kb.transpose(-2, -1)) * ctx.scale
            causal = key_positions[start:end][None, :] <= query_positions[:, None]
            mask = causal[None, None, :, :] & key_valid[:, None, start:end, None].transpose(-1, -2)
            probs = torch.where(
                mask,
                torch.exp(scores - m) / z.clamp_min(torch.finfo(torch.float32).tiny),
                torch.zeros_like(scores),
            )
            dprob = torch.matmul(go, vb.transpose(-2, -1))
            dscore = probs * (dprob - (dprob * probs).sum(dim=-1, keepdim=True))
            dq = dq + torch.matmul(dscore, kb) * ctx.scale
            dk[:, :, start:end, :] = dk[:, :, start:end, :] + torch.matmul(
                dscore.transpose(-2, -1), qf,
            ) * ctx.scale
            dv[:, :, start:end, :] = dv[:, :, start:end, :] + torch.matmul(
                probs.transpose(-2, -1), go,
            )
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None


def _streaming_causal_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    valid_mask: Tensor | None,
    window_size: int,
) -> Tensor:
    if window_size < 1:
        raise ValueError("streaming attention window_size must be positive")
    return _StreamingCausalAttention.apply(q, k, v, valid_mask, window_size)


class _StreamingBlockCausalAttention(torch.autograd.Function):
    """Online softmax over causal block summaries.

    Queries at position t can see only block ids strictly smaller than the
    block containing t.  Both forward and backward revisit block chunks, so
    the token-by-block score matrix is never retained.
    """

    @staticmethod
    def forward(ctx, q, block_k, block_v, query_blocks, block_valid, chunk_size):
        qf, kf, vf = q.float(), block_k.float(), block_v.float()
        batch, heads, length, head_dim = qf.shape
        block_count = kf.shape[-2]
        scale = head_dim ** -0.5
        query_blocks = query_blocks.to(device=q.device)
        block_valid = block_valid.to(device=q.device)
        m = torch.full(
            (batch, heads, length, 1), -torch.inf,
            dtype=torch.float32, device=q.device,
        )
        z = torch.zeros_like(m)
        acc = torch.zeros(
            (batch, heads, length, head_dim), dtype=torch.float32,
            device=q.device,
        )
        for start in range(0, block_count, chunk_size):
            end = min(start + chunk_size, block_count)
            kb = kf[:, :, start:end, :]
            vb = vf[:, :, start:end, :]
            block_positions = torch.arange(start, end, device=q.device)
            mask = (
                block_positions[None, :] < query_blocks[:, None]
            )[None, None, :, :] & block_valid[:, None, None, start:end]
            scores = torch.matmul(qf, kb.transpose(-2, -1)) * scale
            has = mask.any(dim=-1, keepdim=True)
            block_m = scores.masked_fill(~mask, -torch.inf).amax(
                dim=-1, keepdim=True,
            )
            block_m_safe = torch.where(has, block_m, torch.zeros_like(block_m))
            weights = torch.where(
                mask, torch.exp(scores - block_m_safe), torch.zeros_like(scores),
            )
            block_z = weights.sum(dim=-1, keepdim=True)
            block_acc = torch.matmul(weights, vb)
            new_m = torch.where(has, torch.maximum(m, block_m), m)
            old_scale = torch.where(
                z > 0, torch.exp(m - new_m), torch.zeros_like(z),
            )
            block_scale = torch.where(
                has, torch.exp(block_m_safe - new_m), torch.zeros_like(z),
            )
            z = old_scale * z + block_scale * block_z
            acc = old_scale * acc + block_scale * block_acc
            m = new_m
        out = acc / z.clamp_min(torch.finfo(torch.float32).tiny)
        ctx.save_for_backward(q, block_k, block_v, m, z, query_blocks, block_valid)
        ctx.chunk_size = chunk_size
        ctx.scale = scale
        return out.to(dtype=block_v.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        q, block_k, block_v, m, z, query_blocks, block_valid = ctx.saved_tensors
        qf, kf, vf = q.float(), block_k.float(), block_v.float()
        go = grad_output.float()
        batch, heads, length, head_dim = qf.shape
        block_count = kf.shape[-2]
        dq = torch.zeros_like(qf)
        dk = torch.zeros_like(kf)
        dv = torch.zeros_like(vf)
        for start in range(0, block_count, ctx.chunk_size):
            end = min(start + ctx.chunk_size, block_count)
            kb = kf[:, :, start:end, :]
            vb = vf[:, :, start:end, :]
            block_positions = torch.arange(start, end, device=q.device)
            mask = (
                block_positions[None, :] < query_blocks[:, None]
            )[None, None, :, :] & block_valid[:, None, None, start:end]
            scores = torch.matmul(qf, kb.transpose(-2, -1)) * ctx.scale
            probs = torch.where(
                mask,
                torch.exp(scores - m) / z.clamp_min(torch.finfo(torch.float32).tiny),
                torch.zeros_like(scores),
            )
            dprob = torch.matmul(go, vb.transpose(-2, -1))
            dscore = probs * (
                dprob - (dprob * probs).sum(dim=-1, keepdim=True)
            )
            dq = dq + torch.matmul(dscore, kb) * ctx.scale
            dk[:, :, start:end, :] = dk[:, :, start:end, :] + torch.matmul(
                dscore.transpose(-2, -1), qf,
            ) * ctx.scale
            dv[:, :, start:end, :] = dv[:, :, start:end, :] + torch.matmul(
                probs.transpose(-2, -1), go,
            )
        return (
            dq.to(q.dtype), dk.to(block_k.dtype), dv.to(block_v.dtype),
            None, None, None,
        )


def _streaming_block_causal_attention(
    q: Tensor,
    block_k: Tensor,
    block_v: Tensor,
    query_blocks: Tensor,
    block_valid: Tensor,
    chunk_size: int,
) -> Tensor:
    if chunk_size < 1:
        raise ValueError("streaming block chunk_size must be positive")
    return _StreamingBlockCausalAttention.apply(
        q, block_k, block_v, query_blocks, block_valid, chunk_size,
    )


def _chunked_block_causal_attention(
    q: Tensor,
    block_k: Tensor,
    block_v: Tensor,
    query_blocks: Tensor,
    block_valid: Tensor,
    query_chunk_size: int,
) -> Tensor:
    """Block-summary attention in query tiles, avoiding an L by G tensor."""

    batch, _, length, _ = q.shape
    block_count = block_k.shape[-2]
    if bool(block_valid.all()):
        # With complete blocks, use one regular no-mask SDPA per query block.
        # The key prefix is already causal at block granularity, so Flash/
        # Triton SDPA remains eligible and no [L, G] mask is constructed.
        outputs = []
        window_size = int((length + block_count - 1) // block_count)
        for block_index in range(block_count):
            start = block_index * window_size
            end = min(start + window_size, length)
            q_chunk = q[:, :, start:end, :]
            if block_index == 0:
                outputs.append(torch.zeros_like(q_chunk))
                continue

            def sdpa_prefix(q_value, k_value, v_value):
                return F.scaled_dot_product_attention(
                    q_value, k_value, v_value,
                    dropout_p=0.0, is_causal=False,
                )

            outputs.append(torch.utils.checkpoint.checkpoint(
                sdpa_prefix, q_chunk,
                block_k[:, :, :block_index, :],
                block_v[:, :, :block_index, :],
                use_reentrant=False,
            ))
        return torch.cat(outputs, dim=2)

    block_positions = torch.arange(block_count, device=q.device)
    outputs = []
    for start in range(0, length, query_chunk_size):
        end = min(start + query_chunk_size, length)
        query_ids = query_blocks[start:end]
        mask = (
            block_positions[None, :] < query_ids[:, None]
        )[None, None, :, :] & block_valid[:, None, None, :]
        q_chunk = q[:, :, start:end, :]

        def sdpa_chunk(q_value, k_value, v_value, mask_value):
            return F.scaled_dot_product_attention(
                q_value, k_value, v_value,
                attn_mask=mask_value,
                dropout_p=0.0,
                is_causal=False,
            )

        outputs.append(torch.utils.checkpoint.checkpoint(
            sdpa_chunk, q_chunk, block_k, block_v, mask,
            use_reentrant=False,
        ))
    return torch.cat(outputs, dim=2)


def _hierarchical_causal_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    valid_mask: Tensor | None,
    window_size: int,
) -> Tensor:
    """Approximate two-level causal attention with local and block paths.

    The local path attends causally inside the current block.  The global path
    attends only to mean K/V summaries of completed previous blocks.  It is
    deliberately not the exact softmax factorization: current-block detail is
    local, while long-range context is compressed to one vector per block.
    """

    if window_size < 1:
        raise ValueError("hierarchical attention window_size must be positive")
    batch, heads, length, head_dim = q.shape
    blocks = (length + window_size - 1) // window_size
    padded = blocks * window_size
    pad = padded - length
    if valid_mask is None:
        key_valid = torch.ones(
            (batch, padded), dtype=torch.bool, device=q.device,
        )
        query_valid = None
    else:
        key_valid = F.pad(valid_mask, (0, pad), value=False)
        key_valid[:, 0] |= ~key_valid.any(dim=1)
        query_valid = valid_mask

    q_pad = q if pad == 0 else F.pad(q, (0, 0, 0, pad))
    k_pad = k if pad == 0 else F.pad(k, (0, 0, 0, pad))
    v_pad = v if pad == 0 else F.pad(v, (0, 0, 0, pad))
    q_blocks = q_pad.reshape(batch, heads, blocks, window_size, head_dim)
    k_blocks = k_pad.reshape(batch, heads, blocks, window_size, head_dim)
    v_blocks = v_pad.reshape(batch, heads, blocks, window_size, head_dim)
    key_blocks = key_valid.reshape(batch, blocks, window_size)

    # First level: process blocks directly from the original sequence.  This
    # avoids the full-length B*blocks flattening copy; speed is intentionally
    # secondary here because the caller is optimizing peak memory.
    local_chunks = []
    local_causal = torch.ones(
        (window_size, window_size), dtype=torch.bool, device=q.device,
    ).tril()
    for block_index in range(blocks):
        start = block_index * window_size
        end = min(start + window_size, padded)
        q_local = q_pad[:, :, start:end, :]
        k_local = k_pad[:, :, start:end, :]
        v_local = v_pad[:, :, start:end, :]
        if valid_mask is None and pad == 0:
            def local_sdpa(q_value, k_value, v_value):
                return F.scaled_dot_product_attention(
                    q_value, k_value, v_value,
                    dropout_p=0.0, is_causal=True,
                )
            local = torch.utils.checkpoint.checkpoint(
                local_sdpa, q_local, k_local, v_local, use_reentrant=False,
            )
        else:
            local_mask = local_causal[None, None, :, :] & key_blocks[
                :, block_index:block_index + 1, None, :
            ]
            def local_sdpa_masked(q_value, k_value, v_value, mask_value):
                return F.scaled_dot_product_attention(
                    q_value, k_value, v_value,
                    attn_mask=mask_value.reshape(
                        batch, 1,
                        window_size, window_size,
                    ),
                    dropout_p=0.0, is_causal=False,
                )
            local = torch.utils.checkpoint.checkpoint(
                local_sdpa_masked, q_local, k_local, v_local, local_mask,
                use_reentrant=False,
            )
        local_chunks.append(local)
    local = torch.cat(local_chunks, dim=2)[:, :, :length]

    # Compress each completed block once.  The current block is excluded from
    # the global path, so a token never sees future tokens in its own block.
    block_mask = key_blocks[:, None, :, :, None].to(k_pad.dtype)
    counts = key_blocks.sum(dim=-1).clamp_min(1).to(k_pad.dtype)
    block_k = (k_blocks * block_mask).sum(dim=-2) / counts[:, None, :, None]
    block_v = (v_blocks * block_mask).sum(dim=-2) / counts[:, None, :, None]
    query_blocks = torch.arange(length, device=q.device) // window_size
    global_out = _chunked_block_causal_attention(
        q, block_k, block_v, query_blocks,
        key_blocks.any(dim=-1), max(256, window_size * 16),
    )

    attended = (local.float() + global_out) * 0.5
    if query_valid is not None:
        attended = attended * query_valid[:, None, :, None].to(attended.dtype)
    return attended.to(dtype=v.dtype)


def _resample_kv_grid(
    hidden: Tensor,
    valid_mask: Tensor | None,
    grid: int,
) -> tuple[Tensor, Tensor | None, Tensor]:
    """Map an L-token causal source stream onto an explicit KV grid.

    A smaller grid uses causal element-wise max-pooling over contiguous source
    intervals instead of mean-pooling.  A larger grid repeats source cells by
    nearest-left indexing; this increases attention resolution without
    inventing future information.  ``positions`` stores the latest source
    position represented by each KV cell and is used by the caller's causal
    mask.
    """

    batch, length, dim = hidden.shape
    if grid < 1:
        raise ValueError("grid must be positive")
    if valid_mask is not None and valid_mask.shape != (batch, length):
        raise ValueError("valid_mask must have shape [batch, sequence]")
    if grid == length:
        return hidden, valid_mask, torch.arange(length, device=hidden.device)
    if grid > length:
        indices = torch.div(
            torch.arange(grid, device=hidden.device) * length,
            grid,
            rounding_mode="floor",
        ).clamp_max(length - 1)
        sampled = hidden.index_select(1, indices)
        sampled_mask = (
            None if valid_mask is None else valid_mask.index_select(1, indices)
        )
        return sampled, sampled_mask, indices

    starts = torch.div(
        torch.arange(grid, device=hidden.device) * length,
        grid,
        rounding_mode="floor",
    )
    ends = torch.div(
        (torch.arange(grid, device=hidden.device) + 1) * length,
        grid,
        rounding_mode="floor",
    ) - 1
    ends = ends.clamp_min(starts)
    if valid_mask is None:
        pooled_mask = None
        source = hidden
    else:
        count_prefix = valid_mask.float().cumsum(dim=1)
        count_prefix = torch.cat(
            (torch.zeros_like(count_prefix[:, :1]), count_prefix), dim=1
        )
        valid_counts = (
            count_prefix.index_select(1, ends + 1)
            - count_prefix.index_select(1, starts)
        )
        pooled_mask = valid_counts > 0
        source = hidden.masked_fill(~valid_mask[:, :, None], float("-inf"))

    # Fast path for the usual exact partition (e.g. 128 -> 64): max_pool1d
    # uses non-overlapping windows and preserves the same [starts, ends].
    if length % grid == 0:
        width = length // grid
        pooled = F.max_pool1d(
            source.transpose(1, 2), kernel_size=width, stride=width
        ).transpose(1, 2)
    else:
        # Dynamic chapter lengths may not divide evenly.  Keep the exact
        # interval definition used by the causal positions in this fallback.
        pieces = [
            source[:, int(start):int(end) + 1].amax(dim=1)
            for start, end in zip(starts.tolist(), ends.tolist())
        ]
        pooled = torch.stack(pieces, dim=1)
    if valid_mask is not None:
        pooled = torch.where(pooled_mask[:, :, None], pooled, torch.zeros_like(pooled))
    return pooled.to(dtype=hidden.dtype), pooled_mask, ends


@dataclass(frozen=True)
class LeftTrackLayout:
    """Per-position left interval metadata.

    ``valid`` is false for the first half-window positions, where no complete
    left half exists.  ``starts``/``ends`` are clamped to legal indices even
    for invalid rows so they can be used directly with ``index_select``.
    """

    starts: Tensor
    ends: Tensor
    lengths: Tensor
    valid: Tensor
    width: int
    half: int


def build_right_track_layout(
    length: int,
    *,
    layer: int = 1,
    device: torch.device | None = None,
) -> LeftTrackLayout:
    """Build the mirrored, unshifted right-half interval metadata.

    The interval at position ``t`` is ``[t + w/2, t + w - 1]``.  The
    attention module shifts the resulting track one cell to the right, so
    output position ``t`` receives the right track that was built at
    ``t - 1``.  The shift is the alignment used by the bidirectional branch;
    valid-prefix masking is what removes the target at the right edge of a
    triangle row.
    """

    if length < 1:
        raise ValueError("length must be positive")
    if layer < 1:
        raise ValueError("layer must be positive")
    # Keep the nominal layer width independent of the current tile length.
    # Only interval indices are clamped below.  This is important for the
    # final short tile: eager and fixed-size compiled paths must describe the
    # same layer, rather than silently changing its half-window.
    width = 2 ** int(layer)
    half = max(1, width // 2)

    positions = torch.arange(length, device=device, dtype=torch.long)
    starts = (positions + half).clamp_max(length - 1)
    ends = (positions + width - 1).clamp_max(length - 1)
    valid = (positions + half < length) & (ends >= starts)
    lengths = (ends - starts + 1).to(dtype=torch.float32)
    return LeftTrackLayout(starts, ends, lengths, valid, width, half)


def build_left_track_layout(
    length: int,
    *,
    layer: int = 1,
    device: torch.device | None = None,
) -> LeftTrackLayout:
    """Build the causal left-half interval for every sequence position.

    ``layer=1`` uses width 2, ``layer=2`` width 4, and so on.  If the
    sequence is shorter, indices are clamped and incomplete tracks are
    marked invalid; the layer's nominal width is not rewritten.
    """

    if length < 1:
        raise ValueError("length must be positive")
    if layer < 1:
        raise ValueError("layer must be positive")
    # Keep the nominal layer width independent of the current tile length;
    # clamped indices and ``valid`` below provide the short-sequence safety.
    width = 2 ** int(layer)
    half = max(1, width // 2)

    positions = torch.arange(length, device=device, dtype=torch.long)
    starts = (positions - width + 1).clamp_min(0)
    ends = (positions - half).clamp_min(0)
    valid = (positions >= half) & (ends >= starts)
    lengths = (ends - starts + 1).to(dtype=torch.float32)
    return LeftTrackLayout(starts, ends, lengths, valid, width, half)


def left_track_pool(
    hidden: Tensor,
    layout: LeftTrackLayout,
) -> Tensor:
    """Return one mean-pooled left-track state per position.

    Args:
        hidden: ``[batch, sequence, dim]`` tensor.
        layout: output of :func:`build_left_track_layout` for that sequence.

    Returns:
        ``[batch, sequence, dim]``.  Invalid early positions are zero.  The
        operation is differentiable and uses only a prefix sum plus two
        indexed reads per position: ``O(batch * sequence * dim)``.
    """

    if hidden.ndim != 3:
        raise ValueError("hidden must have shape [batch, sequence, dim]")
    batch, length, _ = hidden.shape
    if layout.starts.shape != (length,):
        raise ValueError("layout length does not match hidden")
    device = hidden.device
    starts = layout.starts.to(device=device)
    ends = layout.ends.to(device=device)
    lengths = layout.lengths.to(device=device, dtype=hidden.dtype)
    valid = layout.valid.to(device=device)

    # Keep the prefix itself at length ``sequence``.  The former
    # ``cat(zero, cumsum)`` formulation allocated an additional full hidden
    # tensor on every layer.  ``starts - 1`` plus a zeroed gather is exactly
    # the same interval sum without that temporary.
    # The prefix sum is an accumulation over the whole chapter.  BF16 has
    # only seven mantissa bits, so doing this in hidden.dtype loses small
    # terms quickly on 4k--20k token chapters.  Keep the reduction in FP32
    # and cast only the gathered result back to the model dtype.
    prefix = hidden.float().cumsum(dim=1)
    start_index = (starts - 1).clamp_min(0)
    start_sum = prefix.index_select(1, start_index)
    start_sum = start_sum * (starts > 0).view(1, length, 1).to(prefix.dtype)
    sums = prefix.index_select(1, ends) - start_sum
    pooled = sums / lengths.float().clamp_min(1).view(1, length, 1)
    return (pooled * valid.view(1, length, 1)).to(dtype=hidden.dtype)


def _interval_pool(
    hidden: Tensor,
    starts: Tensor,
    ends: Tensor,
    valid: Tensor,
    valid_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Mean-pool intervals and return the per-position usable-cell mask."""

    batch, length, _ = hidden.shape
    if starts.shape != (length,) or ends.shape != (length,):
        raise ValueError("interval metadata does not match hidden")
    if valid_mask is not None and valid_mask.shape != (batch, length):
        raise ValueError("valid_mask must have shape [batch, sequence]")

    if valid_mask is None:
        masked = hidden
        counts = (ends - starts + 1).to(device=hidden.device, dtype=hidden.dtype)
        position_valid = valid.to(device=hidden.device)
    else:
        mask = valid_mask.to(device=hidden.device)
        masked = hidden * mask[:, :, None].to(dtype=hidden.dtype)
        # Keep interval cardinalities in FP32; BF16 cumsum is too coarse for
        # long chapter rows.
        count_prefix = mask.float().cumsum(dim=1)
        start_index = (starts - 1).clamp_min(0).to(device=hidden.device)
        end_index = ends.to(device=hidden.device)
        start_count = count_prefix.index_select(1, start_index)
        start_count = start_count * (starts > 0).view(1, length).to(hidden.dtype)
        counts = count_prefix.index_select(1, end_index) - start_count
        counts = counts.to(dtype=hidden.dtype)
        position_valid = valid.to(device=hidden.device).view(1, length)

    prefix = masked.cumsum(dim=1)
    start_index = (starts - 1).clamp_min(0).to(device=hidden.device)
    end_index = ends.to(device=hidden.device)
    start_sum = prefix.index_select(1, start_index)
    start_sum = start_sum * (starts > 0).view(1, length, 1).to(hidden.dtype)
    sums = prefix.index_select(1, end_index) - start_sum
    usable = counts > 0
    pooled = sums / counts.clamp_min(1).view(-1 if counts.ndim == 2 else 1, length, 1)
    if counts.ndim == 1:
        pooled = pooled * (position_valid.view(1, length, 1) & usable.view(1, length, 1)).to(hidden.dtype)
        pooled_valid = (position_valid & usable).view(1, length).expand(batch, length)
    else:
        pooled = pooled * (position_valid & usable).view(batch, length, 1).to(hidden.dtype)
        pooled_valid = position_valid & usable
    return pooled, pooled_valid


def right_track_pool(
    hidden: Tensor,
    layout: LeftTrackLayout,
    valid_mask: Tensor | None = None,
    *,
    causal: bool = False,
) -> tuple[Tensor, Tensor]:
    """Pool the mirrored right track and shift it one cell to the right.

    ``right[:, t]`` is the unshifted right interval computed at ``t - 1``;
    ``right[:, 0]`` is zero.  When ``valid_mask`` is supplied, invalid cells
    are removed from both the numerator and denominator.  This is the key
    leakage barrier for lower-triangle prefix rows: the target sits just to
    the right of the valid prefix and cannot enter this track.  With
    ``causal=True``, each shifted interval is additionally clipped at its own
    destination, which is required by dense full-sequence next-token CE.
    """

    if hidden.ndim != 3:
        raise ValueError("hidden must have shape [batch, sequence, dim]")
    batch, length, _ = hidden.shape
    if layout.starts.shape != (length,):
        raise ValueError("layout length does not match hidden")

    pool_layout = layout
    if causal:
        # A dense next-token pass has no row-specific prefix mask.  Clip the
        # shifted right interval at its destination so it cannot read future
        # targets from the full chapter.
        length = hidden.shape[1]
        positions = torch.arange(length, device=hidden.device)
        starts = torch.zeros(length, dtype=torch.long, device=hidden.device)
        ends = torch.zeros(length, dtype=torch.long, device=hidden.device)
        valid = torch.zeros(length, dtype=torch.bool, device=hidden.device)
        if length > 1:
            starts[1:] = layout.starts[:-1].to(device=hidden.device)
            ends[1:] = torch.minimum(
                layout.ends[:-1].to(device=hidden.device), positions[1:]
            )
            valid[1:] = layout.valid[:-1].to(device=hidden.device)
            valid[1:] &= ends[1:] >= starts[1:]
        lengths = (ends - starts + 1).to(dtype=torch.float32)
        pool_layout = LeftTrackLayout(
            starts, ends, lengths, valid, layout.width, layout.half,
        )
    pooled, raw_valid = _interval_pool(
        hidden, pool_layout.starts, pool_layout.ends, pool_layout.valid,
        valid_mask,
    )
    right = torch.zeros_like(pooled)
    right_valid = torch.zeros(
        (batch, length), dtype=torch.bool, device=hidden.device
    )
    if length > 1:
        right[:, 1:] = pooled[:, :-1]
        right_valid[:, 1:] = raw_valid[:, :-1]
    return right, right_valid


class BasicLeftTrack(nn.Module):
    """Reference left-track adapter with an optional learned track offset."""

    def __init__(self, dim: int, *, use_track_embedding: bool = True) -> None:
        super().__init__()
        self.dim = int(dim)
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        self.track_embedding = (
            nn.Parameter(torch.zeros(self.dim))
            if use_track_embedding else None
        )

    def forward(
        self,
        hidden: Tensor,
        *,
        layer: int = 1,
    ) -> tuple[Tensor, LeftTrackLayout]:
        if hidden.shape[-1] != self.dim:
            raise ValueError(f"expected hidden width {self.dim}")
        layout = build_left_track_layout(
            hidden.shape[1], layer=layer,
            device=hidden.device,
        )
        pooled = left_track_pool(hidden, layout)
        if self.track_embedding is not None:
            pooled = pooled + self.track_embedding.to(dtype=pooled.dtype)
            pooled = pooled * layout.valid.view(1, -1, 1).to(pooled.dtype)
        return pooled, layout


class LeftTrackAttention(nn.Module):
    """Two-source left attention, optionally with a shifted right track.

    With ``bidirectional=True`` the source grid becomes ``self + left +
    shifted-right``.  Lower-triangle rows use ``valid_mask`` to cut off the
    target; dense full-chapter next-CE uses ``causal_right=True`` to apply the
    equivalent cut independently for every destination position.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        layer: int,
        bidirectional: bool = False,
        full_causal: bool = False,
        causal_right: bool = False,
        kv_heads: int | None = None,
        rope_theta: float = 1_000_000.0,
        attention_dim: int | None = None,
        attention_stride: int = 1,
        attention_grid: int | None = None,
        flash_attention: bool = False,
        attention_activation: str = "softmax",
    ) -> None:
        super().__init__()
        if dim <= 0 or heads <= 0 or dim % heads:
            raise ValueError("dim must be positive and divisible by heads")
        if layer < 1:
            raise ValueError("layer must be positive")
        self.dim = int(dim)
        self.heads = int(heads)
        self.attention_dim = self.dim if attention_dim is None else int(attention_dim)
        if self.attention_dim <= 0 or self.attention_dim % self.heads:
            raise ValueError("attention_dim must be positive and divisible by heads")
        self.head_dim = self.attention_dim // self.heads
        self.attention_stride = int(attention_stride)
        if self.attention_stride <= 0:
            raise ValueError("attention_stride must be positive")
        self.attention_grid = (
            None if attention_grid is None else int(attention_grid)
        )
        if self.attention_grid is not None and self.attention_grid <= 0:
            raise ValueError("attention_grid must be positive")
        self.flash_attention = bool(flash_attention)
        if attention_activation not in {"softmax", "sigmoid"}:
            raise ValueError("attention_activation must be softmax or sigmoid")
        self.attention_activation = attention_activation
        self.layer = int(layer)
        self.bidirectional = bool(bidirectional)
        self.full_causal = bool(full_causal)
        if self.bidirectional and self.full_causal:
            raise ValueError("bidirectional and full_causal are mutually exclusive")
        self.causal_right = bool(causal_right)
        self.kv_heads = self.heads if kv_heads is None else int(kv_heads)
        if self.kv_heads <= 0 or self.heads % self.kv_heads:
            raise ValueError("heads must be divisible by kv_heads")
        self.rope_theta = float(rope_theta)
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive")

        # The left-only branch is a drop-in MiniMind attention replacement:
        # separate bias-free Q/K/V projections, per-head Q/K RMSNorm, RoPE,
        # GQA, and aligned block-causal SDPA.  The pooled-track projections
        # are only constructed for the bidirectional experimental branch.
        self.left_q = nn.Linear(dim, self.attention_dim, bias=False)
        self.left_k = nn.Linear(dim, self.kv_heads * self.head_dim, bias=False)
        self.left_v = nn.Linear(dim, self.kv_heads * self.head_dim, bias=False)
        self.left_out = nn.Linear(self.attention_dim, dim, bias=False)
        self.left_q_norm = _HeadRMSNorm(self.head_dim)
        self.left_k_norm = _HeadRMSNorm(self.head_dim)
        if self.bidirectional:
            # Q only sees the destination token.  K/V see each source cell;
            # the projection is applied independently to each source.
            self.q = nn.Linear(dim, dim)
            self.kv = nn.Linear(dim, 2 * dim)
            self.out = nn.Linear(dim, dim)
        else:
            # Do not register the old pooled-track route in left/full models:
            # it was never reached by forward(), but still inflated the
            # optimizer and checkpoint parameter count.
            self.q = None
            self.kv = None
            self.out = None
        self._layout_cache: dict[tuple[str, int | None, int], LeftTrackLayout] = {}

    def _left_block_attention(
        self,
        x: Tensor,
        valid_mask: Tensor | None,
        seam_reset: Tensor | None = None,
    ) -> Tensor:
        """MiniMind-compatible aligned dyadic causal attention.

        The sequence is partitioned into aligned blocks of width
        ``2 ** layer``.  Each destination attends to the causal prefix of its
        own block.  This is the same left-track mechanism used by the former
        MiniMind drop-in implementation, but batched into one SDPA call.
        """

        batch, length, _ = x.shape
        q = self.left_q(x).view(batch, length, self.heads, self.head_dim)
        k = self.left_k(x).view(batch, length, self.kv_heads, self.head_dim)
        v = self.left_v(x).view(batch, length, self.kv_heads, self.head_dim)
        q = self.left_q_norm(q)
        k = self.left_k_norm(k)
        q, k = _apply_rope(q, k, theta=self.rope_theta)
        k = _repeat_kv(k, self.heads // self.kv_heads)
        v = _repeat_kv(v, self.heads // self.kv_heads)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        width = min(2 ** self.layer, length)
        blocks = (length + width - 1) // width
        padded = blocks * width
        pad = padded - length
        if pad:
            q = torch.cat((q, q.new_zeros((batch, self.heads, pad, self.head_dim))), dim=2)
            k = torch.cat((k, k.new_zeros((batch, self.heads, pad, self.head_dim))), dim=2)
            v = torch.cat((v, v.new_zeros((batch, self.heads, pad, self.head_dim))), dim=2)

        q_blocks = q.reshape(batch, self.heads, blocks, width, self.head_dim)
        k_blocks = k.reshape(batch, self.heads, blocks, width, self.head_dim)
        v_blocks = v.reshape(batch, self.heads, blocks, width, self.head_dim)
        q_blocks = q_blocks.permute(0, 2, 1, 3, 4).reshape(
            batch * blocks, self.heads, width, self.head_dim
        )
        k_blocks = k_blocks.permute(0, 2, 1, 3, 4).reshape(
            batch * blocks, self.heads, width, self.head_dim
        )
        v_blocks = v_blocks.permute(0, 2, 1, 3, 4).reshape(
            batch * blocks, self.heads, width, self.head_dim
        )

        if seam_reset is not None:
            if seam_reset.shape != (batch, length) or seam_reset.dtype != torch.bool:
                raise ValueError("seam_reset must be bool with input shape")
            # The separator belongs to the new chapter.  Its state and all
            # following states may attend to it, but never to keys before it.
            segment_ids = seam_reset.to(dtype=torch.int64).cumsum(dim=1)
        else:
            segment_ids = None

        if valid_mask is None and segment_ids is None:
            attended = F.scaled_dot_product_attention(
                q_blocks, k_blocks, v_blocks, is_causal=True
            )
            query_valid = None
        else:
            if valid_mask is not None and (
                valid_mask.shape != (batch, length) or valid_mask.dtype != torch.bool
            ):
                raise ValueError("valid_mask must be bool with input shape")
            key_valid = (
                torch.ones((batch, length), dtype=torch.bool, device=x.device)
                if valid_mask is None else valid_mask
            )
            if pad:
                key_valid = torch.cat(
                    (key_valid, key_valid.new_zeros((batch, pad))), dim=1
                )
            key_valid = key_valid.reshape(batch, blocks, width)
            # SDPA does not define an all-masked row.  Add one zero dummy key
            # to blocks containing no real token; those query rows are zeroed
            # after attention and cannot affect the next layer's valid prefix.
            has_key = key_valid.any(dim=-1)
            key_valid = key_valid.clone()
            key_valid[:, :, 0] |= ~has_key
            if (
                segment_ids is None
                and valid_mask is not None
                and _right_padded_lengths(valid_mask) is not None
            ):
                # Every block is itself a prefix.  Slice each length group
                # before SDPA instead of materializing [B*blocks, W, W].
                attended = _right_padded_block_sdpa(
                    q_blocks, k_blocks, v_blocks,
                    key_valid.reshape(batch * blocks, width),
                )
            else:
                causal = torch.ones(
                    (width, width), dtype=torch.bool, device=x.device
                ).tril()
                mask = causal[None, None, :, :] & key_valid[:, :, None, None, :]
                if segment_ids is not None:
                    if pad:
                        segment_ids = torch.cat(
                            (segment_ids, segment_ids.new_full((batch, pad), -1)),
                            dim=1,
                        )
                    block_segments = segment_ids.reshape(batch, blocks, width)
                    same_segment = (
                        block_segments[:, :, :, None] == block_segments[:, :, None, :]
                    )
                    mask = mask & same_segment[:, :, None, :, :]
                    query_valid = (
                        torch.ones((batch, length), dtype=torch.bool, device=x.device)
                        if valid_mask is None else valid_mask
                    )
                    if pad:
                        query_valid_padded = torch.cat(
                            (query_valid, query_valid.new_zeros((batch, pad))), dim=1
                        )
                    else:
                        query_valid_padded = query_valid
                    # Padded query rows are discarded below, but SDPA still
                    # needs one finite key for them.
                    mask = mask | ~query_valid_padded.reshape(
                        batch, blocks, width
                    )[:, :, None, :, None]
                attended = F.scaled_dot_product_attention(
                    q_blocks, k_blocks, v_blocks,
                    attn_mask=mask.reshape(batch * blocks, 1, width, width),
                    is_causal=False,
                )
            query_valid = valid_mask

        attended = attended.reshape(batch, blocks, self.heads, width, self.head_dim)
        attended = attended.permute(0, 2, 1, 3, 4).reshape(
            batch, self.heads, padded, self.head_dim
        )[:, :, :length]
        attended = attended.transpose(1, 2).reshape(
            batch, length, self.attention_dim
        )
        if query_valid is not None:
            attended = attended * query_valid[:, :, None].to(dtype=attended.dtype)
        return self.left_out(attended)

    def _full_causal_attention(
        self,
        x: Tensor,
        valid_mask: Tensor | None,
    ) -> Tensor:
        """Full causal attention, optionally with a compressed KV grid.

        ``attention_stride=2`` keeps the residual stream at length ``L`` but
        pools adjacent source cells into ``ceil(L/2)`` KV cells.  A pooled cell
        ending at position ``2*j+1`` is visible only from destination
        positions at or after that endpoint; this is the causal boundary that
        makes the L/2 grid safe for dense next-token CE.
        """

        batch, length, _ = x.shape
        q = self.left_q(x).view(batch, length, self.heads, self.head_dim)
        target_grid = self.attention_grid
        if target_grid is None:
            target_grid = (length + self.attention_stride - 1) // self.attention_stride
        k_source, source_mask, source_positions = _resample_kv_grid(
            x, valid_mask, target_grid
        )
        v_source = k_source
        source_length = k_source.shape[1]
        k = self.left_k(k_source).view(
            batch, source_length, self.kv_heads, self.head_dim
        )
        v = self.left_v(v_source).view(
            batch, source_length, self.kv_heads, self.head_dim
        )
        q = self.left_q_norm(q)
        k = self.left_k_norm(k)
        if source_length != length:
            q, k = _apply_rope(
                q, k, theta=self.rope_theta, k_positions=source_positions
            )
        else:
            q, k = _apply_rope(q, k, theta=self.rope_theta)
        k = _repeat_kv(k, self.heads // self.kv_heads)
        v = _repeat_kv(v, self.heads // self.kv_heads)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        can_flash = (
            self.attention_activation == "softmax"
            and
            self.flash_attention
            and source_length == length
            and x.is_cuda
            and x.dtype in (torch.float16, torch.bfloat16)
            and flash_attn_func is not None
        )
        if can_flash:
            q_flash = q.transpose(1, 2)
            k_flash = k.transpose(1, 2)
            v_flash = v.transpose(1, 2)
            if valid_mask is None or bool(valid_mask.all()):
                attended = flash_attn_func(
                    q_flash, k_flash, v_flash, causal=True,
                ).transpose(1, 2)
            elif flash_attn_varlen_func is not None:
                lengths = valid_mask.sum(dim=1).to(dtype=torch.int32)
                cu = torch.cat(
                    (torch.zeros(1, dtype=torch.int32, device=x.device),
                     lengths.cumsum(dim=0).to(dtype=torch.int32)),
                )
                q_flat = q_flash[valid_mask]
                k_flat = k_flash[valid_mask]
                v_flat = v_flash[valid_mask]
                attended_flat = flash_attn_varlen_func(
                    q_flat, k_flat, v_flat,
                    cu, cu, int(lengths.max()), int(lengths.max()),
                    causal=True,
                )
                attended_flash = torch.zeros_like(q_flash)
                attended_flash[valid_mask] = attended_flat
                attended = attended_flash.transpose(1, 2)
            else:
                can_flash = False
        if can_flash:
            query_valid = valid_mask
        elif (
            self.attention_activation == "softmax"
            and source_length == length and valid_mask is None
        ):
            attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            query_valid = None
        elif self.attention_activation == "sigmoid":
            # Sigmoid attention is an unnormalized positive kernel.  Keep the
            # total context mass instead of dividing by sum(weights), which
            # would turn this into a softmax-like weighted average.  A
            # sqrt(length) correction prevents long prefixes from making the
            # residual grow linearly while retaining multi-value accumulation.
            # This path deliberately materializes only the local training
            # window's score plane; it is not used by the flash/SDPA path.
            scores = torch.matmul(
                q.float(), k.float().transpose(-2, -1),
            ) * (self.head_dim ** -0.5)
            if source_length == length:
                causal = torch.ones(
                    (length, length), dtype=torch.bool, device=x.device,
                ).tril()
                if valid_mask is not None:
                    key_valid = valid_mask.clone()
                    key_valid[:, 0] |= ~key_valid.any(dim=1)
                    causal = causal[None, None, :, :] & key_valid[:, None, None, :]
                else:
                    causal = causal[None, None, :, :]
            else:
                dummy = torch.ones(
                    (batch, length, 1), dtype=torch.bool, device=x.device,
                )
                causal = source_positions[None, :] <= torch.arange(
                    length, device=x.device,
                )[:, None]
                if valid_mask is not None:
                    causal = causal[None, :, :] & source_mask[:, None, :]
                else:
                    causal = causal[None, :, :].expand(batch, -1, -1)
                causal = torch.cat((causal, dummy), dim=-1)[:, None, :, :]
            weights = torch.sigmoid(scores.masked_fill(~causal, -torch.inf))
            weights = weights * causal.to(weights.dtype)
            context_size = causal.sum(dim=-1, keepdim=True).clamp_min(1)
            context_scale = context_size.to(dtype=weights.dtype).sqrt()
            attended = torch.matmul(weights, v.float()) / context_scale
            # Keep the accumulated sigmoid context in FP32.  The caller may
            # be under FP16 autocast; converting here causes residual drift
            # and can make the distillation KL explode on the 151k-vocab head.
            query_valid = valid_mask
        else:
            if valid_mask is not None and (
                valid_mask.shape != (batch, length)
                or valid_mask.dtype != torch.bool
            ):
                raise ValueError("valid_mask must be bool with input shape")
            if source_length == length:
                key_valid = valid_mask.clone() if valid_mask is not None else None
                if key_valid is not None:
                    # Keep SDPA finite for a completely padded row.  Its
                    # output is discarded by query_valid below.
                    key_valid[:, 0] |= ~key_valid.any(dim=1)
                if (
                    key_valid is not None
                    and _right_padded_lengths(valid_mask) is not None
                ):
                    attended = _right_padded_causal_sdpa(q, k, v, valid_mask)
                    query_valid = valid_mask
                    mask = None
                else:
                    causal = torch.ones(
                        (length, length), dtype=torch.bool, device=x.device
                    ).tril()
                    mask = causal[None, None, :, :]
                    if key_valid is not None:
                        mask = mask & key_valid[:, None, None, :]
            else:
                # Append one zero-valued dummy KV cell for the first query,
                # which has no complete causal pair yet.
                dummy_k = torch.zeros(
                    (batch, self.heads, 1, self.head_dim),
                    dtype=k.dtype, device=x.device,
                )
                dummy_v = torch.zeros_like(dummy_k)
                k = torch.cat((k, dummy_k), dim=2)
                v = torch.cat((v, dummy_v), dim=2)
                query_positions = torch.arange(length, device=x.device)
                causal = source_positions[None, :] <= query_positions[:, None]
                if valid_mask is not None:
                    causal = causal[None, :, :] & source_mask[:, None, :]
                else:
                    causal = causal[None, :, :].expand(batch, -1, -1)
                dummy = torch.ones(
                    (batch, length, 1), dtype=torch.bool, device=x.device
                )
                mask = torch.cat((causal, dummy), dim=2)[:, None, :, :]
            if mask is not None:
                attended = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=mask, is_causal=False
                )
                query_valid = valid_mask

        attended = attended.transpose(1, 2).reshape(
            batch, length, self.attention_dim
        )
        if query_valid is not None:
            attended = attended * query_valid[:, :, None].to(dtype=attended.dtype)
        return self.left_out(attended)

    def _layout(self, length: int, device: torch.device) -> LeftTrackLayout:
        key = (device.type, device.index, length)
        layout = self._layout_cache.get(key)
        if layout is None:
            layout = build_left_track_layout(
                length, layer=self.layer, device=device,
            )
            self._layout_cache[key] = layout
        return layout

    def forward(
        self,
        x: Tensor,
        valid_mask: Tensor | None = None,
        seam_reset: Tensor | None = None,
    ) -> Tensor:
        if x.ndim != 3:
            raise ValueError("x must have shape [batch, sequence, dim]")
        batch, length, dim = x.shape
        if dim != self.dim:
            raise ValueError(f"expected dim={self.dim}, got {dim}")
        if valid_mask is not None and valid_mask.shape != (batch, length):
            raise ValueError("valid_mask must have shape [batch, sequence]")
        if seam_reset is not None and (
            seam_reset.shape != (batch, length) or seam_reset.dtype != torch.bool
        ):
            raise ValueError("seam_reset must be bool with input shape")

        if self.full_causal:
            if seam_reset is not None:
                raise ValueError("seam_reset is only supported by the left backend")
            return self._full_causal_attention(x, valid_mask)
        if not self.bidirectional:
            return self._left_block_attention(x, valid_mask, seam_reset)
        if seam_reset is not None:
            raise ValueError("seam_reset is only supported by the left backend")

        layout = self._layout(length, x.device)
        left = left_track_pool(x, layout)
        right_valid: Tensor | None = None
        if self.bidirectional:
            right_layout = build_right_track_layout(
                length, layer=self.layer, device=x.device,
            )
            right, right_valid = right_track_pool(
                x, right_layout, valid_mask, causal=self.causal_right,
            )

        # The left-only CPU path has exactly two sources.  Flattening those
        # two source grids into one projection and reducing the two-way
        # attention analytically avoids the per-position BMM/softmax launch
        # pattern used by the general (and GPU-tuned) path below.  For two
        # scores, the left weight is sigmoid(score_left - score_self), so the
        # result is ``v_self + weight_left * (v_left - v_self)``.  Keep this
        # branch CPU-only: the CUDA path is intentionally unchanged.
        if not self.bidirectional and x.device.type == "cpu":
            q = self.q(x).view(batch, length, self.heads, self.head_dim)
            source_values = torch.cat((x, left), dim=1)
            kv = self.kv(source_values).view(
                batch, 2, length, 2, self.heads, self.head_dim,
            )
            k = kv[:, :, :, 0]
            v = kv[:, :, :, 1]
            q = q * (self.head_dim ** -0.5)
            self_scores = (q * k[:, 0]).sum(dim=-1)
            left_scores = (q * k[:, 1]).sum(dim=-1)
            left_weight = torch.sigmoid(left_scores - self_scores)
            left_weight = left_weight * layout.valid.to(
                device=x.device, dtype=left_weight.dtype
            ).view(1, length, 1)
            attended = v[:, 0] + left_weight.unsqueeze(-1) * (v[:, 1] - v[:, 0])
            attended = attended.reshape(batch, length, dim)
            if valid_mask is not None:
                attended = attended * valid_mask[:, :, None].to(x.dtype)
            return self.out(attended)

        q = self.q(x).view(batch, length, self.heads, self.head_dim)
        sources = (torch.stack((x, left, right), dim=2)
                   if self.bidirectional else torch.stack((x, left), dim=2))
        source_count = sources.shape[2]
        kv = self.kv(sources).view(
            batch, length, source_count, 2, self.heads, self.head_dim,
        )
        k, v = kv.unbind(dim=3)
        # Flatten (batch, head, position) into one BMM batch.  The reference
        # expression used elementwise multiply/sum and weighted add for every
        # position; BMM lets CUDA handle the same two-source attention in
        # larger regular kernels.
        q = q.transpose(1, 2)
        k = k.permute(0, 3, 1, 2, 4)
        v = v.permute(0, 3, 1, 2, 4)
        flat_q = q.reshape(batch * self.heads * length, 1, self.head_dim)
        flat_k = k.reshape(batch * self.heads * length, source_count, self.head_dim)
        flat_v = v.reshape(batch * self.heads * length, source_count, self.head_dim)
        scores = torch.bmm(flat_q, flat_k.transpose(1, 2))
        scores = scores * (self.head_dim ** -0.5)
        source_valid = torch.ones(
            (batch, length, source_count), dtype=torch.bool, device=x.device
        )
        source_valid[:, :, 1] = layout.valid.to(device=x.device)
        if self.bidirectional:
            assert right_valid is not None
            source_valid[:, :, 2] = right_valid
        if valid_mask is not None:
            source_valid[:, :, 0] = valid_mask
        invalid = (~source_valid).unsqueeze(1).expand(
            batch, self.heads, length, source_count,
        ).reshape(batch * self.heads * length, source_count)
        scores = scores.masked_fill(
            invalid[:, None, :], torch.finfo(scores.dtype).min,
        )
        weights = scores.softmax(dim=-1)
        attended = torch.bmm(weights, flat_v)
        attended = attended.view(batch, self.heads, length, self.head_dim)
        attended = attended.transpose(1, 2).contiguous().view(batch, length, dim)
        out = self.out(attended)
        if valid_mask is not None:
            out = out * valid_mask[:, :, None].to(x.dtype)
        return out


__all__ = [
    "BasicLeftTrack",
    "LeftTrackAttention",
    "LeftTrackLayout",
    "build_left_track_layout",
    "build_right_track_layout",
    "left_track_pool",
    "right_track_pool",
]
