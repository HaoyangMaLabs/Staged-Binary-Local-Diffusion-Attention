"""Fused exact-source SBDA attention kernels.

The regular SBDA path has only two or four exact causal sources per query.
Calling general gather/matmul/softmax operators for that shape launches a
large number of tiny kernels.  These kernels fuse the source loads, score,
softmax, value reduction, and their backward pass.  They intentionally cover
only the all-valid default route; masked/seamed/path-table inputs use the
reference implementation in ``left_v2``.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - CPU/test environments
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _sbda_forward_kernel(
        q_ptr, k_ptr, v_ptr, out_ptr, offsets_ptr,
        batch, heads, length, head_dim, scale,
        sqb, sqh, sql, sqd,
        skb, skh, skl, skd,
        svb, svh, svl, svd,
        sob, soh, sol, sod,
        S: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        position = pid % length
        head = (pid // length) % heads
        batch_index = pid // (length * heads)
        d = tl.arange(0, BLOCK_D)
        d_mask = d < head_dim
        q_base = (
            q_ptr + batch_index * sqb + head * sqh + position * sql
        )
        qv = tl.load(q_base + d * sqd, mask=d_mask, other=0.0).to(tl.float32)
        offsets = tl.load(offsets_ptr + tl.arange(0, S))
        sources = position - offsets
        valid = sources >= 0
        sources = tl.maximum(sources, 0)
        k_base = k_ptr + batch_index * skb + head * skh
        v_base = v_ptr + batch_index * svb + head * svh
        k_ptrs = k_base + sources[:, None] * skl + d[None, :] * skd
        v_ptrs = v_base + sources[:, None] * svl + d[None, :] * svd
        kv_mask = valid[:, None] & d_mask[None, :]
        keys = tl.load(k_ptrs, mask=kv_mask, other=0.0).to(tl.float32)
        values = tl.load(v_ptrs, mask=kv_mask, other=0.0).to(tl.float32)
        scores = tl.sum(keys * qv[None, :], axis=1) * scale
        scores = tl.where(valid, scores, float("-inf"))
        maximum = tl.max(scores, axis=0)
        weights = tl.exp(scores - maximum)
        weights = weights / tl.sum(weights, axis=0)
        result = tl.sum(values * weights[:, None], axis=0)
        out_base = (
            out_ptr + batch_index * sob + head * soh + position * sol
        )
        tl.store(out_base + d * sod, result, mask=d_mask)


    @triton.jit
    def _sbda_backward_kernel(
        q_ptr, k_ptr, v_ptr, grad_out_ptr,
        grad_q_ptr, grad_k_ptr, grad_v_ptr, offsets_ptr,
        batch, heads, length, head_dim, scale,
        sqb, sqh, sql, sqd,
        skb, skh, skl, skd,
        svb, svh, svl, svd,
        gob, goh, gol, god,
        gqb, gqh, gql, gqd,
        gkb, gkh, gkl, gkd,
        gvb, gvh, gvl, gvd,
        S: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        position = pid % length
        head = (pid // length) % heads
        batch_index = pid // (length * heads)
        d = tl.arange(0, BLOCK_D)
        d_mask = d < head_dim
        q_base = q_ptr + batch_index * sqb + head * sqh + position * sql
        qv = tl.load(q_base + d * sqd, mask=d_mask, other=0.0).to(tl.float32)
        go_base = grad_out_ptr + batch_index * gob + head * goh + position * gol
        grad_out = tl.load(go_base + d * god, mask=d_mask, other=0.0).to(tl.float32)
        offsets = tl.load(offsets_ptr + tl.arange(0, S))
        sources = position - offsets
        valid = sources >= 0
        sources = tl.maximum(sources, 0)
        k_base = k_ptr + batch_index * skb + head * skh
        v_base = v_ptr + batch_index * svb + head * svh
        k_ptrs = k_base + sources[:, None] * skl + d[None, :] * skd
        v_ptrs = v_base + sources[:, None] * svl + d[None, :] * svd
        kv_mask = valid[:, None] & d_mask[None, :]
        keys = tl.load(k_ptrs, mask=kv_mask, other=0.0).to(tl.float32)
        values = tl.load(v_ptrs, mask=kv_mask, other=0.0).to(tl.float32)
        scores = tl.sum(keys * qv[None, :], axis=1) * scale
        scores = tl.where(valid, scores, float("-inf"))
        maximum = tl.max(scores, axis=0)
        weights = tl.exp(scores - maximum)
        weights = weights / tl.sum(weights, axis=0)
        d_weights = tl.sum(values * grad_out[None, :], axis=1)
        centered = d_weights - tl.sum(weights * d_weights, axis=0)
        d_scores = weights * centered
        grad_q = tl.sum(d_scores[:, None] * keys, axis=0) * scale
        grad_q_base = grad_q_ptr + batch_index * gqb + head * gqh + position * gql
        tl.store(grad_q_base + d * gqd, grad_q, mask=d_mask)

        grad_k_base = grad_k_ptr + batch_index * gkb + head * gkh
        grad_v_base = grad_v_ptr + batch_index * gvb + head * gvh
        grad_k_ptrs = grad_k_base + sources[:, None] * gkl + d[None, :] * gkd
        grad_v_ptrs = grad_v_base + sources[:, None] * gvl + d[None, :] * gvd
        # Invalid early sources are clamped to position zero only for safe
        # loads; their gradient contribution is masked out here.
        atomic_mask = kv_mask
        tl.atomic_add(
            grad_k_ptrs, d_scores[:, None] * scale * qv[None, :],
            mask=atomic_mask,
        )
        tl.atomic_add(
            grad_v_ptrs, weights[:, None] * grad_out[None, :],
            mask=atomic_mask,
        )


    @triton.jit
    def _sbda_block_forward_kernel(
        q_ptr, k_ptr, v_ptr, out_ptr, offsets_ptr,
        batch, heads, length, head_dim, scale,
        sqb, sqh, sql, sqd,
        skb, skh, skl, skd,
        svb, svh, svl, svd,
        sob, soh, sol, sod,
        S: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Process a block of query rows in one program.

        SBDA has a compile-time number of exact sources.  The online softmax
        recurrence lets the kernel keep only a ``[BLOCK_M, BLOCK_D]`` value
        accumulator in registers; it never materializes a gather tensor or a
        score matrix.
        """
        pid = tl.program_id(0)
        blocks = tl.cdiv(length, BLOCK_M)
        block = pid % blocks
        head = (pid // blocks) % heads
        batch_index = pid // (blocks * heads)
        rows = block * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.arange(0, BLOCK_D)
        row_valid = rows < length
        d_valid = cols < head_dim

        q_base = q_ptr + batch_index * sqb + head * sqh
        q = tl.load(
            q_base + rows[:, None] * sql + cols[None, :] * sqd,
            mask=row_valid[:, None] & d_valid[None, :], other=0.0,
        ).to(tl.float32)
        k_base = k_ptr + batch_index * skb + head * skh
        v_base = v_ptr + batch_index * svb + head * svh
        out_base = out_ptr + batch_index * sob + head * soh

        running_max = tl.full((BLOCK_M,), float("-inf"), tl.float32)
        running_sum = tl.zeros((BLOCK_M,), tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
        for source in range(S):
            offset = tl.load(offsets_ptr + source)
            source_rows = rows - offset
            source_valid = (source_rows >= 0) & row_valid
            source_rows = tl.maximum(source_rows, 0)
            k = tl.load(
                k_base + source_rows[:, None] * skl + cols[None, :] * skd,
                mask=source_valid[:, None] & d_valid[None, :], other=0.0,
            ).to(tl.float32)
            v = tl.load(
                v_base + source_rows[:, None] * svl + cols[None, :] * svd,
                mask=source_valid[:, None] & d_valid[None, :], other=0.0,
            ).to(tl.float32)
            score = tl.sum(q * k, axis=1) * scale
            score = tl.where(source_valid, score, float("-inf"))
            new_max = tl.maximum(running_max, score)
            old_factor = tl.exp(running_max - new_max)
            new_factor = tl.exp(score - new_max)
            running_sum = running_sum * old_factor + new_factor
            acc = acc * old_factor[:, None] + new_factor[:, None] * v
            running_max = new_max

        out = acc / running_sum[:, None]
        tl.store(
            out_base + rows[:, None] * sol + cols[None, :] * sod,
            out, mask=row_valid[:, None] & d_valid[None, :],
        )


    @triton.jit
    def _sbda_block_backward_kernel(
        q_ptr, k_ptr, v_ptr, grad_out_ptr,
        grad_q_ptr, grad_k_ptr, grad_v_ptr, offsets_ptr,
        batch, heads, length, head_dim, scale,
        sqb, sqh, sql, sqd,
        skb, skh, skl, skd,
        svb, svh, svl, svd,
        gob, goh, gol, god,
        gqb, gqh, gql, gqd,
        gkb, gkh, gkl, gkd,
        gvb, gvh, gvl, gvd,
        S: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Block-query backward for exact-source SBDA.

        ``dQ`` is written once per query block.  ``dK`` and ``dV`` receive a
        bounded number (2 or 4) of contributions and are accumulated in FP32
        with atomics.  This removes the one-program-per-token launch pattern;
        a later key-oriented variant can remove these atomics if profiling
        shows they dominate on a particular GPU.
        """
        pid = tl.program_id(0)
        blocks = tl.cdiv(length, BLOCK_M)
        block = pid % blocks
        head = (pid // blocks) % heads
        batch_index = pid // (blocks * heads)
        rows = block * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.arange(0, BLOCK_D)
        row_valid = rows < length
        d_valid = cols < head_dim

        q_base = q_ptr + batch_index * sqb + head * sqh
        k_base = k_ptr + batch_index * skb + head * skh
        v_base = v_ptr + batch_index * svb + head * svh
        go_base = grad_out_ptr + batch_index * gob + head * goh
        q = tl.load(
            q_base + rows[:, None] * sql + cols[None, :] * sqd,
            mask=row_valid[:, None] & d_valid[None, :], other=0.0,
        ).to(tl.float32)
        grad_out = tl.load(
            go_base + rows[:, None] * gol + cols[None, :] * god,
            mask=row_valid[:, None] & d_valid[None, :], other=0.0,
        ).to(tl.float32)

        # First pass: exact online softmax statistics.
        running_max = tl.full((BLOCK_M,), float("-inf"), tl.float32)
        running_sum = tl.zeros((BLOCK_M,), tl.float32)
        for source in range(S):
            offset = tl.load(offsets_ptr + source)
            source_rows = rows - offset
            source_valid = (source_rows >= 0) & row_valid
            source_rows = tl.maximum(source_rows, 0)
            k = tl.load(
                k_base + source_rows[:, None] * skl + cols[None, :] * skd,
                mask=source_valid[:, None] & d_valid[None, :], other=0.0,
            ).to(tl.float32)
            score = tl.sum(q * k, axis=1) * scale
            score = tl.where(source_valid, score, float("-inf"))
            new_max = tl.maximum(running_max, score)
            old_factor = tl.exp(running_max - new_max)
            new_factor = tl.exp(score - new_max)
            running_sum = running_sum * old_factor + new_factor
            running_max = new_max

        grad_q = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
        for source in range(S):
            offset = tl.load(offsets_ptr + source)
            source_rows = rows - offset
            source_valid = (source_rows >= 0) & row_valid
            source_rows = tl.maximum(source_rows, 0)
            k = tl.load(
                k_base + source_rows[:, None] * skl + cols[None, :] * skd,
                mask=source_valid[:, None] & d_valid[None, :], other=0.0,
            ).to(tl.float32)
            v = tl.load(
                v_base + source_rows[:, None] * svl + cols[None, :] * svd,
                mask=source_valid[:, None] & d_valid[None, :], other=0.0,
            ).to(tl.float32)
            score = tl.sum(q * k, axis=1) * scale
            score = tl.where(source_valid, score, float("-inf"))
            weights = tl.exp(score - running_max) / running_sum
            d_weights = tl.sum(v * grad_out, axis=1)
            # The softmax Jacobian is applied after all source probabilities
            # are known.  The correction term is accumulated in a second
            # source loop below, so this first loop only supplies d_weights.
            # ``d_prob_sum`` is recomputed explicitly to keep register use
            # predictable for S=2/4.
            d_prob_sum = tl.zeros((BLOCK_M,), tl.float32)
            for other in range(S):
                other_offset = tl.load(offsets_ptr + other)
                other_rows = rows - other_offset
                other_valid = (other_rows >= 0) & row_valid
                other_rows = tl.maximum(other_rows, 0)
                other_k = tl.load(
                    k_base + other_rows[:, None] * skl + cols[None, :] * skd,
                    mask=other_valid[:, None] & d_valid[None, :], other=0.0,
                ).to(tl.float32)
                other_v = tl.load(
                    v_base + other_rows[:, None] * svl + cols[None, :] * svd,
                    mask=other_valid[:, None] & d_valid[None, :], other=0.0,
                ).to(tl.float32)
                other_score = tl.sum(q * other_k, axis=1) * scale
                other_score = tl.where(other_valid, other_score, float("-inf"))
                other_weight = tl.exp(other_score - running_max) / running_sum
                d_prob_sum += other_weight * tl.sum(other_v * grad_out, axis=1)
            d_score = weights * (d_weights - d_prob_sum)
            grad_q += d_score[:, None] * k * scale

            grad_k_base = grad_k_ptr + batch_index * gkb + head * gkh
            grad_v_base = grad_v_ptr + batch_index * gvb + head * gvh
            grad_k_ptrs = grad_k_base + source_rows[:, None] * gkl + cols[None, :] * gkd
            grad_v_ptrs = grad_v_base + source_rows[:, None] * gvl + cols[None, :] * gvd
            atomic_mask = source_valid[:, None] & d_valid[None, :]
            tl.atomic_add(
                grad_k_ptrs, d_score[:, None] * scale * q,
                mask=atomic_mask,
            )
            tl.atomic_add(
                grad_v_ptrs, weights[:, None] * grad_out,
                mask=atomic_mask,
            )

        grad_q_base = grad_q_ptr + batch_index * gqb + head * gqh
        tl.store(
            grad_q_base + rows[:, None] * gql + cols[None, :] * gqd,
            grad_q, mask=row_valid[:, None] & d_valid[None, :],
        )


    @triton.jit
    def _sbda_block_backward_vector_kernel(
        q_ptr, k_ptr, v_ptr, grad_out_ptr,
        grad_q_ptr, grad_k_ptr, grad_v_ptr, offsets_ptr,
        batch, heads, length, head_dim, scale,
        sqb, sqh, sql, sqd,
        skb, skh, skl, skd,
        svb, svh, svl, svd,
        gob, goh, gol, god,
        gqb, gqh, gql, gqd,
        gkb, gkh, gkl, gkd,
        gvb, gvh, gvl, gvd,
        S: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Vectorized block backward with one source dimension.

        ``S`` is only 2 or 4, so keeping the source axis explicit is cheaper
        than the previous nested source recomputation.  The kernel computes
        all source probabilities and their softmax Jacobian together, then
        scatters the bounded K/V gradients with FP32 atomics.
        """
        pid = tl.program_id(0)
        blocks = tl.cdiv(length, BLOCK_M)
        block = pid % blocks
        head = (pid // blocks) % heads
        batch_index = pid // (blocks * heads)
        rows = block * BLOCK_M + tl.arange(0, BLOCK_M)
        sources = tl.arange(0, S)
        cols = tl.arange(0, BLOCK_D)
        row_valid = rows < length
        d_valid = cols < head_dim

        q_base = q_ptr + batch_index * sqb + head * sqh
        k_base = k_ptr + batch_index * skb + head * skh
        v_base = v_ptr + batch_index * svb + head * svh
        go_base = grad_out_ptr + batch_index * gob + head * goh
        q = tl.load(
            q_base + rows[:, None] * sql + cols[None, :] * sqd,
            mask=row_valid[:, None] & d_valid[None, :], other=0.0,
        ).to(tl.float32)
        grad_out = tl.load(
            go_base + rows[:, None] * gol + cols[None, :] * god,
            mask=row_valid[:, None] & d_valid[None, :], other=0.0,
        ).to(tl.float32)

        offsets = tl.load(offsets_ptr + sources)
        source_rows = rows[:, None] - offsets[None, :]
        source_valid = (source_rows >= 0) & row_valid[:, None]
        source_rows = tl.maximum(source_rows, 0)
        ptrs = (
            source_rows[:, :, None] * skl
            + cols[None, None, :] * skd
        )
        k = tl.load(
            k_base + ptrs,
            mask=source_valid[:, :, None] & d_valid[None, None, :],
            other=0.0,
        ).to(tl.float32)
        v = tl.load(
            v_base + source_rows[:, :, None] * svl
            + cols[None, None, :] * svd,
            mask=source_valid[:, :, None] & d_valid[None, None, :],
            other=0.0,
        ).to(tl.float32)

        scores = tl.sum(q[:, None, :] * k, axis=2) * scale
        scores = tl.where(source_valid, scores, float("-inf"))
        maximum = tl.max(scores, axis=1)
        unnormalized = tl.exp(scores - maximum[:, None])
        weights = unnormalized / tl.sum(unnormalized, axis=1)[:, None]
        d_weights = tl.sum(v * grad_out[:, None, :], axis=2)
        correction = tl.sum(weights * d_weights, axis=1)
        d_scores = weights * (d_weights - correction[:, None])
        grad_q = tl.sum(d_scores[:, :, None] * k, axis=1) * scale

        grad_k_base = grad_k_ptr + batch_index * gkb + head * gkh
        grad_v_base = grad_v_ptr + batch_index * gvb + head * gvh
        grad_k_ptrs = (
            grad_k_base + source_rows[:, :, None] * gkl
            + cols[None, None, :] * gkd
        )
        grad_v_ptrs = (
            grad_v_base + source_rows[:, :, None] * gvl
            + cols[None, None, :] * gvd
        )
        atomic_mask = source_valid[:, :, None] & d_valid[None, None, :]
        tl.atomic_add(
            grad_k_ptrs, d_scores[:, :, None] * scale * q[:, None, :],
            mask=atomic_mask,
        )
        tl.atomic_add(
            grad_v_ptrs, weights[:, :, None] * grad_out[:, None, :],
            mask=atomic_mask,
        )

        grad_q_base = grad_q_ptr + batch_index * gqb + head * gqh
        tl.store(
            grad_q_base + rows[:, None] * gql + cols[None, :] * gqd,
            grad_q, mask=row_valid[:, None] & d_valid[None, :],
        )


    @triton.jit
    def _sbda_weights_kernel(
        q_ptr, k_ptr, weights_ptr, offsets_ptr,
        batch, heads, length, head_dim, scale,
        sqb, sqh, sql, sqd,
        skb, skh, skl, skd,
        swb, swh, swl, sws,
        S: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Save only the small per-query SBDA softmax table."""
        pid = tl.program_id(0)
        blocks = tl.cdiv(length, BLOCK_M)
        block = pid % blocks
        head = (pid // blocks) % heads
        batch_index = pid // (blocks * heads)
        rows = block * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.arange(0, BLOCK_D)
        row_valid = rows < length
        d_valid = cols < head_dim
        q_base = q_ptr + batch_index * sqb + head * sqh
        k_base = k_ptr + batch_index * skb + head * skh
        q = tl.load(
            q_base + rows[:, None] * sql + cols[None, :] * sqd,
            mask=row_valid[:, None] & d_valid[None, :], other=0.0,
        ).to(tl.float32)

        running_max = tl.full((BLOCK_M,), float("-inf"), tl.float32)
        running_sum = tl.zeros((BLOCK_M,), tl.float32)
        for source in range(S):
            offset = tl.load(offsets_ptr + source)
            source_rows = rows - offset
            source_valid = (source_rows >= 0) & row_valid
            source_rows = tl.maximum(source_rows, 0)
            k = tl.load(
                k_base + source_rows[:, None] * skl + cols[None, :] * skd,
                mask=source_valid[:, None] & d_valid[None, :], other=0.0,
            ).to(tl.float32)
            score = tl.sum(q * k, axis=1) * scale
            score = tl.where(source_valid, score, float("-inf"))
            new_max = tl.maximum(running_max, score)
            old_factor = tl.exp(running_max - new_max)
            new_factor = tl.exp(score - new_max)
            running_sum = running_sum * old_factor + new_factor
            running_max = new_max

        weights_base = weights_ptr + batch_index * swb + head * swh
        for source in range(S):
            offset = tl.load(offsets_ptr + source)
            source_rows = rows - offset
            source_valid = (source_rows >= 0) & row_valid
            source_rows = tl.maximum(source_rows, 0)
            k = tl.load(
                k_base + source_rows[:, None] * skl + cols[None, :] * skd,
                mask=source_valid[:, None] & d_valid[None, :], other=0.0,
            ).to(tl.float32)
            score = tl.sum(q * k, axis=1) * scale
            score = tl.where(source_valid, score, float("-inf"))
            weight = tl.exp(score - running_max) / running_sum
            tl.store(
                weights_base + rows * swl + source * sws,
                weight, mask=row_valid,
            )


    @triton.jit
    def _sbda_dq_from_weights_kernel(
        q_ptr, k_ptr, v_ptr, grad_out_ptr, weights_ptr, offsets_ptr,
        grad_q_ptr,
        batch, heads, length, head_dim, scale,
        sqb, sqh, sql, sqd,
        skb, skh, skl, skd,
        svb, svh, svl, svd,
        gob, goh, gol, god,
        swb, swh, swl, sws,
        gqb, gqh, gql, gqd,
        S: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Compute dQ from saved small source weights."""
        pid = tl.program_id(0)
        blocks = tl.cdiv(length, BLOCK_M)
        block = pid % blocks
        head = (pid // blocks) % heads
        batch_index = pid // (blocks * heads)
        rows = block * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.arange(0, BLOCK_D)
        row_valid = rows < length
        d_valid = cols < head_dim
        q_base = q_ptr + batch_index * sqb + head * sqh
        k_base = k_ptr + batch_index * skb + head * skh
        v_base = v_ptr + batch_index * svb + head * svh
        go_base = grad_out_ptr + batch_index * gob + head * goh
        weights_base = weights_ptr + batch_index * swb + head * swh
        q = tl.load(
            q_base + rows[:, None] * sql + cols[None, :] * sqd,
            mask=row_valid[:, None] & d_valid[None, :], other=0.0,
        ).to(tl.float32)
        grad_out = tl.load(
            go_base + rows[:, None] * gol + cols[None, :] * god,
            mask=row_valid[:, None] & d_valid[None, :], other=0.0,
        ).to(tl.float32)

        correction = tl.zeros((BLOCK_M,), tl.float32)
        for source in range(S):
            offset = tl.load(offsets_ptr + source)
            source_rows = rows - offset
            source_valid = (source_rows >= 0) & row_valid
            source_rows = tl.maximum(source_rows, 0)
            v = tl.load(
                v_base + source_rows[:, None] * svl + cols[None, :] * svd,
                mask=source_valid[:, None] & d_valid[None, :], other=0.0,
            ).to(tl.float32)
            weight = tl.load(
                weights_base + rows * swl + source * sws,
                mask=row_valid, other=0.0,
            ).to(tl.float32)
            correction += weight * tl.sum(v * grad_out, axis=1)

        grad_q = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
        for source in range(S):
            offset = tl.load(offsets_ptr + source)
            source_rows = rows - offset
            source_valid = (source_rows >= 0) & row_valid
            source_rows = tl.maximum(source_rows, 0)
            k = tl.load(
                k_base + source_rows[:, None] * skl + cols[None, :] * skd,
                mask=source_valid[:, None] & d_valid[None, :], other=0.0,
            ).to(tl.float32)
            v = tl.load(
                v_base + source_rows[:, None] * svl + cols[None, :] * svd,
                mask=source_valid[:, None] & d_valid[None, :], other=0.0,
            ).to(tl.float32)
            weight = tl.load(
                weights_base + rows * swl + source * sws,
                mask=row_valid, other=0.0,
            ).to(tl.float32)
            d_weight = tl.sum(v * grad_out, axis=1)
            d_score = weight * (d_weight - correction)
            grad_q += d_score[:, None] * k * scale

        grad_q_base = grad_q_ptr + batch_index * gqb + head * gqh
        tl.store(
            grad_q_base + rows[:, None] * gql + cols[None, :] * gqd,
            grad_q, mask=row_valid[:, None] & d_valid[None, :],
        )


    @triton.jit
    def _sbda_dkv_key_kernel(
        q_ptr, k_ptr, v_ptr, grad_out_ptr, weights_ptr, offsets_ptr,
        grad_k_ptr, grad_v_ptr,
        batch, heads, length, head_dim, scale,
        sqb, sqh, sql, sqd,
        skb, skh, skl, skd,
        svb, svh, svl, svd,
        gob, goh, gol, god,
        swb, swh, swl, sws,
        gkb, gkh, gkl, gkd,
        gvb, gvh, gvl, gvd,
        S: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Key-centric dK/dV: each key block is written exactly once."""
        pid = tl.program_id(0)
        blocks = tl.cdiv(length, BLOCK_M)
        block = pid % blocks
        head = (pid // blocks) % heads
        batch_index = pid // (blocks * heads)
        keys = block * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.arange(0, BLOCK_D)
        key_valid = keys < length
        d_valid = cols < head_dim
        q_base = q_ptr + batch_index * sqb + head * sqh
        v_base = v_ptr + batch_index * svb + head * svh
        go_base = grad_out_ptr + batch_index * gob + head * goh
        weights_base = weights_ptr + batch_index * swb + head * swh
        grad_k = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
        grad_v = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

        for target in range(S):
            target_offset = tl.load(offsets_ptr + target)
            queries = keys + target_offset
            query_valid = (queries < length) & key_valid
            q = tl.load(
                q_base + queries[:, None] * sql + cols[None, :] * sqd,
                mask=query_valid[:, None] & d_valid[None, :], other=0.0,
            ).to(tl.float32)
            grad_out = tl.load(
                go_base + queries[:, None] * gol + cols[None, :] * god,
                mask=query_valid[:, None] & d_valid[None, :], other=0.0,
            ).to(tl.float32)
            correction = tl.zeros((BLOCK_M,), tl.float32)
            target_weight = tl.zeros((BLOCK_M,), tl.float32)
            target_d_weight = tl.zeros((BLOCK_M,), tl.float32)

            for source in range(S):
                source_offset = tl.load(offsets_ptr + source)
                source_rows = queries - source_offset
                source_valid = (source_rows >= 0) & query_valid
                source_rows = tl.maximum(source_rows, 0)
                source_v = tl.load(
                    v_base + source_rows[:, None] * svl + cols[None, :] * svd,
                    mask=source_valid[:, None] & d_valid[None, :], other=0.0,
                ).to(tl.float32)
                weight = tl.load(
                    weights_base + queries * swl + source * sws,
                    mask=query_valid, other=0.0,
                ).to(tl.float32)
                d_weight = tl.sum(source_v * grad_out, axis=1)
                correction += weight * d_weight
                if source == target:
                    target_weight = weight
                    target_d_weight = d_weight

            d_score = target_weight * (target_d_weight - correction)
            grad_k += d_score[:, None] * q * scale
            grad_v += target_weight[:, None] * grad_out

        grad_k_base = grad_k_ptr + batch_index * gkb + head * gkh
        grad_v_base = grad_v_ptr + batch_index * gvb + head * gvh
        tl.store(
            grad_k_base + keys[:, None] * gkl + cols[None, :] * gkd,
            grad_k, mask=key_valid[:, None] & d_valid[None, :],
        )
        tl.store(
            grad_v_base + keys[:, None] * gvl + cols[None, :] * gvd,
            grad_v, mask=key_valid[:, None] & d_valid[None, :],
        )


    @triton.jit
    def _sbda_block_forward_gqa_kernel(
        q_ptr, k_ptr, v_ptr, out_ptr, offsets_ptr,
        batch, heads, kv_heads, length, head_dim, scale,
        sqb, sqh, sql, sqd,
        skb, skh, skl, skd,
        svb, svh, svl, svd,
        sob, soh, sol, sod,
        S: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """GQA forward without expanding K/V over query heads."""
        pid = tl.program_id(0)
        blocks = tl.cdiv(length, BLOCK_M)
        block = pid % blocks
        head = (pid // blocks) % heads
        batch_index = pid // (blocks * heads)
        kv_head = head // GROUP_SIZE
        rows = block * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.arange(0, BLOCK_D)
        row_valid = rows < length
        d_valid = cols < head_dim
        q_base = q_ptr + batch_index * sqb + head * sqh
        k_base = k_ptr + batch_index * skb + kv_head * skh
        v_base = v_ptr + batch_index * svb + kv_head * svh
        out_base = out_ptr + batch_index * sob + head * soh
        running_max = tl.full((BLOCK_M,), float("-inf"), tl.float32)
        running_sum = tl.zeros((BLOCK_M,), tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
        q = tl.load(
            q_base + rows[:, None] * sql + cols[None, :] * sqd,
            mask=row_valid[:, None] & d_valid[None, :], other=0.0,
        ).to(tl.float32)
        for source in range(S):
            offset = tl.load(offsets_ptr + source)
            source_rows = rows - offset
            source_valid = (source_rows >= 0) & row_valid
            source_rows = tl.maximum(source_rows, 0)
            k = tl.load(
                k_base + source_rows[:, None] * skl + cols[None, :] * skd,
                mask=source_valid[:, None] & d_valid[None, :], other=0.0,
            ).to(tl.float32)
            v = tl.load(
                v_base + source_rows[:, None] * svl + cols[None, :] * svd,
                mask=source_valid[:, None] & d_valid[None, :], other=0.0,
            ).to(tl.float32)
            score = tl.sum(q * k, axis=1) * scale
            score = tl.where(source_valid, score, float("-inf"))
            new_max = tl.maximum(running_max, score)
            old_factor = tl.exp(running_max - new_max)
            new_factor = tl.exp(score - new_max)
            running_sum = running_sum * old_factor + new_factor
            acc = acc * old_factor[:, None] + new_factor[:, None] * v
            running_max = new_max
        tl.store(
            out_base + rows[:, None] * sol + cols[None, :] * sod,
            acc / running_sum[:, None],
            mask=row_valid[:, None] & d_valid[None, :],
        )


    @triton.jit
    def _sbda_weights_gqa_kernel(
        q_ptr, k_ptr, weights_ptr, offsets_ptr,
        batch, heads, kv_heads, length, head_dim, scale,
        sqb, sqh, sql, sqd,
        skb, skh, skl, skd,
        swb, swh, swl, sws,
        S: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Save exact source probabilities while keeping compressed K/V."""
        pid = tl.program_id(0)
        blocks = tl.cdiv(length, BLOCK_M)
        block = pid % blocks
        head = (pid // blocks) % heads
        batch_index = pid // (blocks * heads)
        kv_head = head // GROUP_SIZE
        rows = block * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.arange(0, BLOCK_D)
        row_valid = rows < length
        d_valid = cols < head_dim
        q_base = q_ptr + batch_index * sqb + head * sqh
        k_base = k_ptr + batch_index * skb + kv_head * skh
        q = tl.load(
            q_base + rows[:, None] * sql + cols[None, :] * sqd,
            mask=row_valid[:, None] & d_valid[None, :], other=0.0,
        ).to(tl.float32)
        running_max = tl.full((BLOCK_M,), float("-inf"), tl.float32)
        running_sum = tl.zeros((BLOCK_M,), tl.float32)
        for source in range(S):
            offset = tl.load(offsets_ptr + source)
            source_rows = rows - offset
            source_valid = (source_rows >= 0) & row_valid
            source_rows = tl.maximum(source_rows, 0)
            k = tl.load(
                k_base + source_rows[:, None] * skl + cols[None, :] * skd,
                mask=source_valid[:, None] & d_valid[None, :], other=0.0,
            ).to(tl.float32)
            score = tl.sum(q * k, axis=1) * scale
            score = tl.where(source_valid, score, float("-inf"))
            new_max = tl.maximum(running_max, score)
            old_factor = tl.exp(running_max - new_max)
            new_factor = tl.exp(score - new_max)
            running_sum = running_sum * old_factor + new_factor
            running_max = new_max
        weights_base = weights_ptr + batch_index * swb + head * swh
        for source in range(S):
            offset = tl.load(offsets_ptr + source)
            source_rows = rows - offset
            source_valid = (source_rows >= 0) & row_valid
            source_rows = tl.maximum(source_rows, 0)
            k = tl.load(
                k_base + source_rows[:, None] * skl + cols[None, :] * skd,
                mask=source_valid[:, None] & d_valid[None, :], other=0.0,
            ).to(tl.float32)
            score = tl.sum(q * k, axis=1) * scale
            score = tl.where(source_valid, score, float("-inf"))
            tl.store(
                weights_base + rows * swl + source * sws,
                tl.exp(score - running_max) / running_sum,
                mask=row_valid,
            )


    @triton.jit
    def _sbda_forward_weights_gqa_kernel(
        q_ptr, k_ptr, v_ptr, out_ptr, weights_ptr, offsets_ptr,
        batch, heads, kv_heads, length, head_dim, scale,
        sqb, sqh, sql, sqd,
        skb, skh, skl, skd,
        svb, svh, svl, svd,
        sob, soh, sol, sod,
        swb, swh, swl, sws,
        S: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """GQA forward that writes output and backward probabilities together.

        The key-backward path needs the exact source probabilities.  The old
        implementation ran one complete forward kernel and then a second
        kernel that recomputed the same QK scores.  This kernel keeps the
        online-softmax statistics in registers, then makes one probability
        pass that writes both the compact weights and the value reduction.
        """
        pid = tl.program_id(0)
        blocks = tl.cdiv(length, BLOCK_M)
        block = pid % blocks
        head = (pid // blocks) % heads
        batch_index = pid // (blocks * heads)
        kv_head = head // GROUP_SIZE
        rows = block * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.arange(0, BLOCK_D)
        row_valid = rows < length
        d_valid = cols < head_dim
        q_base = q_ptr + batch_index * sqb + head * sqh
        k_base = k_ptr + batch_index * skb + kv_head * skh
        v_base = v_ptr + batch_index * svb + kv_head * svh
        out_base = out_ptr + batch_index * sob + head * soh
        weights_base = weights_ptr + batch_index * swb + head * swh
        q = tl.load(
            q_base + rows[:, None] * sql + cols[None, :] * sqd,
            mask=row_valid[:, None] & d_valid[None, :], other=0.0,
        ).to(tl.float32)
        running_max = tl.full((BLOCK_M,), float("-inf"), tl.float32)
        running_sum = tl.zeros((BLOCK_M,), tl.float32)
        for source in range(S):
            offset = tl.load(offsets_ptr + source)
            source_rows = rows - offset
            source_valid = (source_rows >= 0) & row_valid
            source_rows = tl.maximum(source_rows, 0)
            k = tl.load(
                k_base + source_rows[:, None] * skl + cols[None, :] * skd,
                mask=source_valid[:, None] & d_valid[None, :], other=0.0,
            ).to(tl.float32)
            score = tl.sum(q * k, axis=1) * scale
            score = tl.where(source_valid, score, float("-inf"))
            new_max = tl.maximum(running_max, score)
            old_factor = tl.exp(running_max - new_max)
            new_factor = tl.exp(score - new_max)
            running_sum = running_sum * old_factor + new_factor
            running_max = new_max
            # Reuse the compact probability workspace for raw scores.  The
            # second pass only needs V plus these scores; recomputing QK here
            # would duplicate the most expensive part of this kernel.
            tl.store(
                weights_base + rows * swl + source * sws,
                score,
                mask=row_valid,
            )

        acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
        for source in range(S):
            offset = tl.load(offsets_ptr + source)
            source_rows = rows - offset
            source_valid = (source_rows >= 0) & row_valid
            source_rows = tl.maximum(source_rows, 0)
            v = tl.load(
                v_base + source_rows[:, None] * svl + cols[None, :] * svd,
                mask=source_valid[:, None] & d_valid[None, :], other=0.0,
            ).to(tl.float32)
            score = tl.load(
                weights_base + rows * swl + source * sws,
                mask=row_valid, other=float("-inf"),
            ).to(tl.float32)
            weight = tl.exp(score - running_max) / running_sum
            tl.store(
                weights_base + rows * swl + source * sws,
                weight,
                mask=row_valid,
            )
            acc += weight[:, None] * v

        tl.store(
            out_base + rows[:, None] * sol + cols[None, :] * sod,
            acc,
            mask=row_valid[:, None] & d_valid[None, :],
        )


    @triton.jit
    def _sbda_dq_from_weights_gqa_kernel(
        q_ptr, k_ptr, v_ptr, grad_out_ptr, weights_ptr, offsets_ptr,
        grad_q_ptr,
        batch, heads, kv_heads, length, head_dim, scale,
        sqb, sqh, sql, sqd,
        skb, skh, skl, skd,
        svb, svh, svl, svd,
        gob, goh, gol, god,
        swb, swh, swl, sws,
        gqb, gqh, gql, gqd,
        S: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """GQA dQ; each Q head reads its shared compressed K/V head."""
        pid = tl.program_id(0)
        blocks = tl.cdiv(length, BLOCK_M)
        block = pid % blocks
        head = (pid // blocks) % heads
        batch_index = pid // (blocks * heads)
        kv_head = head // GROUP_SIZE
        rows = block * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.arange(0, BLOCK_D)
        row_valid = rows < length
        d_valid = cols < head_dim
        q_base = q_ptr + batch_index * sqb + head * sqh
        k_base = k_ptr + batch_index * skb + kv_head * skh
        v_base = v_ptr + batch_index * svb + kv_head * svh
        go_base = grad_out_ptr + batch_index * gob + head * goh
        weights_base = weights_ptr + batch_index * swb + head * swh
        q = tl.load(
            q_base + rows[:, None] * sql + cols[None, :] * sqd,
            mask=row_valid[:, None] & d_valid[None, :], other=0.0,
        ).to(tl.float32)
        grad_out = tl.load(
            go_base + rows[:, None] * gol + cols[None, :] * god,
            mask=row_valid[:, None] & d_valid[None, :], other=0.0,
        ).to(tl.float32)
        correction = tl.zeros((BLOCK_M,), tl.float32)
        for source in range(S):
            offset = tl.load(offsets_ptr + source)
            source_rows = rows - offset
            source_valid = (source_rows >= 0) & row_valid
            source_rows = tl.maximum(source_rows, 0)
            v = tl.load(
                v_base + source_rows[:, None] * svl + cols[None, :] * svd,
                mask=source_valid[:, None] & d_valid[None, :], other=0.0,
            ).to(tl.float32)
            weight = tl.load(
                weights_base + rows * swl + source * sws,
                mask=row_valid, other=0.0,
            ).to(tl.float32)
            correction += weight * tl.sum(v * grad_out, axis=1)
        grad_q = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
        for source in range(S):
            offset = tl.load(offsets_ptr + source)
            source_rows = rows - offset
            source_valid = (source_rows >= 0) & row_valid
            source_rows = tl.maximum(source_rows, 0)
            k = tl.load(
                k_base + source_rows[:, None] * skl + cols[None, :] * skd,
                mask=source_valid[:, None] & d_valid[None, :], other=0.0,
            ).to(tl.float32)
            v = tl.load(
                v_base + source_rows[:, None] * svl + cols[None, :] * svd,
                mask=source_valid[:, None] & d_valid[None, :], other=0.0,
            ).to(tl.float32)
            weight = tl.load(
                weights_base + rows * swl + source * sws,
                mask=row_valid, other=0.0,
            ).to(tl.float32)
            d_score = weight * (tl.sum(v * grad_out, axis=1) - correction)
            grad_q += d_score[:, None] * k * scale
        grad_q_base = grad_q_ptr + batch_index * gqb + head * gqh
        tl.store(
            grad_q_base + rows[:, None] * gql + cols[None, :] * gqd,
            grad_q, mask=row_valid[:, None] & d_valid[None, :],
        )


    @triton.jit
    def _sbda_dkv_key_gqa_kernel(
        q_ptr, k_ptr, v_ptr, grad_out_ptr, weights_ptr, offsets_ptr,
        grad_k_ptr, grad_v_ptr,
        batch, heads, kv_heads, length, head_dim, scale,
        sqb, sqh, sql, sqd,
        skb, skh, skl, skd,
        svb, svh, svl, svd,
        gob, goh, gol, god,
        swb, swh, swl, sws,
        gkb, gkh, gkl, gkd,
        gvb, gvh, gvl, gvd,
        S: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """GQA dK/dV; one program owns a KV-head/key block."""
        pid = tl.program_id(0)
        blocks = tl.cdiv(length, BLOCK_M)
        block = pid % blocks
        kv_head = (pid // blocks) % kv_heads
        batch_index = pid // (blocks * kv_heads)
        keys = block * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.arange(0, BLOCK_D)
        key_valid = keys < length
        d_valid = cols < head_dim
        k_grad = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
        v_grad = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
        k_base = k_ptr + batch_index * skb + kv_head * skh
        v_base = v_ptr + batch_index * svb + kv_head * svh

        for group in range(GROUP_SIZE):
            head = kv_head * GROUP_SIZE + group
            q_base = q_ptr + batch_index * sqb + head * sqh
            go_base = grad_out_ptr + batch_index * gob + head * goh
            weights_base = weights_ptr + batch_index * swb + head * swh
            for target in range(S):
                target_offset = tl.load(offsets_ptr + target)
                queries = keys + target_offset
                query_valid = (queries < length) & key_valid
                q = tl.load(
                    q_base + queries[:, None] * sql + cols[None, :] * sqd,
                    mask=query_valid[:, None] & d_valid[None, :], other=0.0,
                ).to(tl.float32)
                grad_out = tl.load(
                    go_base + queries[:, None] * gol + cols[None, :] * god,
                    mask=query_valid[:, None] & d_valid[None, :], other=0.0,
                ).to(tl.float32)
                correction = tl.zeros((BLOCK_M,), tl.float32)
                target_weight = tl.zeros((BLOCK_M,), tl.float32)
                target_d_weight = tl.zeros((BLOCK_M,), tl.float32)
                for source in range(S):
                    source_offset = tl.load(offsets_ptr + source)
                    source_rows = queries - source_offset
                    source_valid = (source_rows >= 0) & query_valid
                    source_rows = tl.maximum(source_rows, 0)
                    source_v = tl.load(
                        v_base + source_rows[:, None] * svl + cols[None, :] * svd,
                        mask=source_valid[:, None] & d_valid[None, :], other=0.0,
                    ).to(tl.float32)
                    weight = tl.load(
                        weights_base + queries * swl + source * sws,
                        mask=query_valid, other=0.0,
                    ).to(tl.float32)
                    d_weight = tl.sum(source_v * grad_out, axis=1)
                    correction += weight * d_weight
                    if source == target:
                        target_weight = weight
                        target_d_weight = d_weight
                d_score = target_weight * (target_d_weight - correction)
                k_grad += d_score[:, None] * q * scale
                v_grad += target_weight[:, None] * grad_out

        grad_k_base = grad_k_ptr + batch_index * gkb + kv_head * gkh
        grad_v_base = grad_v_ptr + batch_index * gvb + kv_head * gvh
        tl.store(
            grad_k_base + keys[:, None] * gkl + cols[None, :] * gkd,
            k_grad, mask=key_valid[:, None] & d_valid[None, :],
        )
        tl.store(
            grad_v_base + keys[:, None] * gvl + cols[None, :] * gvd,
            v_grad, mask=key_valid[:, None] & d_valid[None, :],
        )


def _kernel_block(head_dim: int) -> int:
    if head_dim <= 32:
        return 32
    if head_dim <= 64:
        return 64
    if head_dim <= 128:
        return 128
    raise ValueError("SBDA fused kernel supports head_dim <= 128")


class _FusedSBDA(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        offsets: torch.Tensor,
    ) -> torch.Tensor:
        if triton is None or not q.is_cuda:
            raise RuntimeError("fused SBDA requires Triton and CUDA")
        if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
            raise ValueError("q, k, and v must have matching [B,H,L,D] shapes")
        batch, heads, length, head_dim = q.shape
        source_count = int(offsets.numel())
        if source_count not in (2, 4):
            raise ValueError("fused SBDA supports two or four sources")
        if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError("unsupported SBDA dtype")
        offsets = offsets.to(device=q.device, dtype=torch.int32).contiguous()
        out = torch.empty_like(q)
        block = _kernel_block(head_dim)
        grid = (batch * heads * length,)
        _sbda_forward_kernel[grid](
            q, k, v, out, offsets,
            batch, heads, length, head_dim, head_dim ** -0.5,
            *q.stride(), *k.stride(), *v.stride(), *out.stride(),
            S=source_count, BLOCK_D=block, num_warps=4,
        )
        ctx.save_for_backward(q, k, v, offsets)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        q, k, v, offsets = ctx.saved_tensors
        batch, heads, length, head_dim = q.shape
        block = _kernel_block(head_dim)
        # Keep the reduction in FP32: K/V receive atomic contributions from
        # many query positions, and BF16 atomics save workspace but cost more
        # time and noticeably perturb the gradient.  The buffers are scoped to
        # this backward call and are released as soon as the layer completes.
        grad_q = torch.empty_like(q, dtype=torch.float32)
        grad_k = torch.zeros_like(k, dtype=torch.float32)
        grad_v = torch.zeros_like(v, dtype=torch.float32)
        grad_output = grad_output.contiguous()
        grid = (batch * heads * length,)
        _sbda_backward_kernel[grid](
            q, k, v, grad_output,
            grad_q, grad_k, grad_v, offsets,
            batch, heads, length, head_dim, head_dim ** -0.5,
            *q.stride(), *k.stride(), *v.stride(), *grad_output.stride(),
            *grad_q.stride(), *grad_k.stride(), *grad_v.stride(),
            S=int(offsets.numel()), BLOCK_D=block, num_warps=4,
        )
        return (
            grad_q.to(dtype=q.dtype),
            grad_k.to(dtype=k.dtype),
            grad_v.to(dtype=v.dtype),
            None,
        )


def _block_m(length: int) -> int:
    # Keep short-sequence tiles deliberately small. A 64x64 FP32 accumulator
    # in the GQA/key path is needlessly register-heavy for 128-token windows
    # and can make first-time Triton compilation look hung.
    if length <= 128:
        return 32
    if length <= 2048:
        return 64
    return 128


def _backward_block_m(length: int) -> int:
    # The backward keeps a source axis in registers.  A smaller query tile
    # avoids register pressure while still amortizing the launch overhead.
    if length <= 128:
        return 16
    if length <= 256:
        return 32
    return 64


class _FusedSBDABlock(torch.autograd.Function):
    """Block-tiled exact-source SBDA with a stable fused backward.

    The forward uses the block kernel.  The production backward intentionally
    keeps the smaller one-token kernel: on Ampere GPUs the vectorized
    ``[BLOCK_M, S, D]`` backward has enough register pressure to lose against
    the compact atomic kernel, even though its arithmetic is more regular.
    """

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        offsets: torch.Tensor,
    ) -> torch.Tensor:
        if triton is None or not q.is_cuda:
            raise RuntimeError("block fused SBDA requires Triton and CUDA")
        if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
            raise ValueError("q, k, and v must have matching [B,H,L,D] shapes")
        batch, heads, length, head_dim = q.shape
        source_count = int(offsets.numel())
        if source_count not in (2, 4):
            raise ValueError("block fused SBDA supports two or four sources")
        if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError("unsupported SBDA dtype")
        offsets = offsets.to(device=q.device, dtype=torch.int32).contiguous()
        out = torch.empty_like(q)
        block_m = _block_m(length)
        block_d = _kernel_block(head_dim)
        grid = (batch * heads * triton.cdiv(length, block_m),)
        _sbda_block_forward_kernel[grid](
            q, k, v, out, offsets,
            batch, heads, length, head_dim, head_dim ** -0.5,
            *q.stride(), *k.stride(), *v.stride(), *out.stride(),
            S=source_count, BLOCK_M=block_m, BLOCK_D=block_d,
            num_warps=4,
        )
        ctx.save_for_backward(q, k, v, offsets)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        q, k, v, offsets = ctx.saved_tensors
        batch, heads, length, head_dim = q.shape
        block_d = _kernel_block(head_dim)
        grad_output = grad_output.contiguous()
        # Keep the compact proven backward for now.  It also accumulates K/V
        # in FP32, avoiding BF16 atomic accumulation noise.  The block forward
        # still removes most of the forward launch overhead; a key-oriented
        # backward can be added later if profiling justifies the complexity.
        grad_q = torch.empty_like(q, dtype=torch.float32)
        grad_k = torch.zeros_like(k, dtype=torch.float32)
        grad_v = torch.zeros_like(v, dtype=torch.float32)
        grid = (batch * heads * length,)
        _sbda_backward_kernel[grid](
            q, k, v, grad_output,
            grad_q, grad_k, grad_v, offsets,
            batch, heads, length, head_dim, head_dim ** -0.5,
            *q.stride(), *k.stride(), *v.stride(), *grad_output.stride(),
            *grad_q.stride(), *grad_k.stride(), *grad_v.stride(),
            S=int(offsets.numel()), BLOCK_D=block_d,
            num_warps=4,
        )
        return (
            grad_q.to(dtype=q.dtype),
            grad_k.to(dtype=k.dtype),
            grad_v.to(dtype=v.dtype),
            None,
        )


class _FusedSBDABlockFull(torch.autograd.Function):
    """Experimental all-block path using the vectorized block backward."""

    @staticmethod
    def forward(ctx, q, k, v, offsets):
        if triton is None or not q.is_cuda:
            raise RuntimeError("block fused SBDA requires Triton and CUDA")
        if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
            raise ValueError("q, k, and v must have matching [B,H,L,D] shapes")
        batch, heads, length, head_dim = q.shape
        source_count = int(offsets.numel())
        if source_count not in (2, 4):
            raise ValueError("block fused SBDA supports two or four sources")
        offsets = offsets.to(device=q.device, dtype=torch.int32).contiguous()
        out = torch.empty_like(q)
        block_m = _block_m(length)
        block_d = _kernel_block(head_dim)
        grid = (batch * heads * triton.cdiv(length, block_m),)
        _sbda_block_forward_kernel[grid](
            q, k, v, out, offsets,
            batch, heads, length, head_dim, head_dim ** -0.5,
            *q.stride(), *k.stride(), *v.stride(), *out.stride(),
            S=source_count, BLOCK_M=block_m, BLOCK_D=block_d,
            num_warps=4,
        )
        ctx.save_for_backward(q, k, v, offsets)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, offsets = ctx.saved_tensors
        batch, heads, length, head_dim = q.shape
        block_m = _backward_block_m(length)
        block_d = _kernel_block(head_dim)
        grad_output = grad_output.contiguous()
        grad_q = torch.empty_like(q, dtype=torch.float32)
        grad_k = torch.zeros_like(k, dtype=torch.float32)
        grad_v = torch.zeros_like(v, dtype=torch.float32)
        grid = (batch * heads * triton.cdiv(length, block_m),)
        _sbda_block_backward_vector_kernel[grid](
            q, k, v, grad_output,
            grad_q, grad_k, grad_v, offsets,
            batch, heads, length, head_dim, head_dim ** -0.5,
            *q.stride(), *k.stride(), *v.stride(), *grad_output.stride(),
            *grad_q.stride(), *grad_k.stride(), *grad_v.stride(),
            S=int(offsets.numel()), BLOCK_M=block_m, BLOCK_D=block_d,
            num_warps=4,
        )
        return (
            grad_q.to(dtype=q.dtype),
            grad_k.to(dtype=k.dtype),
            grad_v.to(dtype=v.dtype),
            None,
        )


class _FusedSBDAKeyBackward(torch.autograd.Function):
    """SBDA with saved small weights and key-centric K/V backward."""

    @staticmethod
    def forward(ctx, q, k, v, offsets):
        if triton is None or not q.is_cuda:
            raise RuntimeError("key-backward SBDA requires Triton and CUDA")
        if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
            raise ValueError("q, k, and v must have matching [B,H,L,D] shapes")
        batch, heads, length, head_dim = q.shape
        source_count = int(offsets.numel())
        if source_count not in (2, 4):
            raise ValueError("key-backward SBDA supports two or four sources")
        offsets = offsets.to(device=q.device, dtype=torch.int32).contiguous()
        out = torch.empty_like(q)
        weights = torch.empty(
            (batch, heads, length, source_count),
            device=q.device, dtype=q.dtype,
        )
        block_m = _block_m(length)
        block_d = _kernel_block(head_dim)
        grid = (batch * heads * triton.cdiv(length, block_m),)
        _sbda_block_forward_kernel[grid](
            q, k, v, out, offsets,
            batch, heads, length, head_dim, head_dim ** -0.5,
            *q.stride(), *k.stride(), *v.stride(), *out.stride(),
            S=source_count, BLOCK_M=block_m, BLOCK_D=block_d,
            num_warps=4,
        )
        _sbda_weights_kernel[grid](
            q, k, weights, offsets,
            batch, heads, length, head_dim, head_dim ** -0.5,
            *q.stride(), *k.stride(), *weights.stride(),
            S=source_count, BLOCK_M=block_m, BLOCK_D=block_d,
            num_warps=4,
        )
        ctx.save_for_backward(q, k, v, weights, offsets)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, weights, offsets = ctx.saved_tensors
        batch, heads, length, head_dim = q.shape
        block_m = _backward_block_m(length)
        block_d = _kernel_block(head_dim)
        grad_output = grad_output.contiguous()
        # The kernels write each output gradient exactly once.  There is no
        # FP32 full-sequence accumulation buffer and no K/V atomic_add.
        grad_q = torch.empty_like(q)
        grad_k = torch.empty_like(k)
        grad_v = torch.empty_like(v)
        grid = (batch * heads * triton.cdiv(length, block_m),)
        _sbda_dq_from_weights_kernel[grid](
            q, k, v, grad_output, weights, offsets,
            grad_q,
            batch, heads, length, head_dim, head_dim ** -0.5,
            *q.stride(), *k.stride(), *v.stride(), *grad_output.stride(),
            *weights.stride(), *grad_q.stride(),
            S=int(offsets.numel()), BLOCK_M=block_m, BLOCK_D=block_d,
            num_warps=4,
        )
        _sbda_dkv_key_kernel[grid](
            q, k, v, grad_output, weights, offsets,
            grad_k, grad_v,
            batch, heads, length, head_dim, head_dim ** -0.5,
            *q.stride(), *k.stride(), *v.stride(), *grad_output.stride(),
            *weights.stride(), *grad_k.stride(), *grad_v.stride(),
            S=int(offsets.numel()), BLOCK_M=block_m, BLOCK_D=block_d,
            num_warps=4,
        )
        return grad_q, grad_k, grad_v, None


class _FusedSBDAGQAKeyBackward(torch.autograd.Function):
    """Native-GQA SBDA: keep K/V at ``kv_heads`` through forward/backward."""

    @staticmethod
    def forward(ctx, q, k, v, offsets):
        if triton is None or not q.is_cuda:
            raise RuntimeError("native-GQA SBDA requires Triton and CUDA")
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError("q, k, and v must have [B,H,L,D] shapes")
        if k.shape != v.shape or q.shape[0] != k.shape[0] or q.shape[2:] != k.shape[2:]:
            raise ValueError("q/k/v sequence and head dimensions must match")
        batch, heads, length, head_dim = q.shape
        kv_heads = k.shape[1]
        if kv_heads <= 0 or heads % kv_heads:
            raise ValueError("query heads must be divisible by kv heads")
        source_count = int(offsets.numel())
        if source_count not in (2, 4):
            raise ValueError("native-GQA SBDA supports two or four sources")
        offsets = offsets.to(device=q.device, dtype=torch.int32).contiguous()
        out = torch.empty_like(q)
        weights = torch.empty(
            # This workspace also holds raw scores. Keep them at the same
            # precision as the online softmax denominator, and retain FP32
            # probabilities for backward.
            (batch, heads, length, source_count), device=q.device, dtype=torch.float32,
        )
        block_m = _block_m(length)
        block_d = _kernel_block(head_dim)
        group_size = heads // kv_heads
        grid = (batch * heads * triton.cdiv(length, block_m),)
        _sbda_forward_weights_gqa_kernel[grid](
            q, k, v, out, weights, offsets,
            batch, heads, kv_heads, length, head_dim, head_dim ** -0.5,
            *q.stride(), *k.stride(), *v.stride(), *out.stride(),
            *weights.stride(),
            S=source_count, GROUP_SIZE=group_size,
            BLOCK_M=block_m, BLOCK_D=block_d, num_warps=4,
        )
        ctx.save_for_backward(q, k, v, weights, offsets)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, weights, offsets = ctx.saved_tensors
        batch, heads, length, head_dim = q.shape
        kv_heads = k.shape[1]
        group_size = heads // kv_heads
        block_m = _backward_block_m(length)
        block_d = _kernel_block(head_dim)
        grad_output = grad_output.contiguous()
        grad_q = torch.empty_like(q)
        grad_k = torch.empty_like(k)
        grad_v = torch.empty_like(v)
        q_grid = (batch * heads * triton.cdiv(length, block_m),)
        _sbda_dq_from_weights_gqa_kernel[q_grid](
            q, k, v, grad_output, weights, offsets, grad_q,
            batch, heads, kv_heads, length, head_dim, head_dim ** -0.5,
            *q.stride(), *k.stride(), *v.stride(), *grad_output.stride(),
            *weights.stride(), *grad_q.stride(),
            S=int(offsets.numel()), GROUP_SIZE=group_size,
            BLOCK_M=block_m, BLOCK_D=block_d, num_warps=4,
        )
        kv_grid = (batch * kv_heads * triton.cdiv(length, block_m),)
        _sbda_dkv_key_gqa_kernel[kv_grid](
            q, k, v, grad_output, weights, offsets, grad_k, grad_v,
            batch, heads, kv_heads, length, head_dim, head_dim ** -0.5,
            *q.stride(), *k.stride(), *v.stride(), *grad_output.stride(),
            *weights.stride(), *grad_k.stride(), *grad_v.stride(),
            S=int(offsets.numel()), GROUP_SIZE=group_size,
            BLOCK_M=block_m, BLOCK_D=block_d, num_warps=4,
        )
        return grad_q, grad_k, grad_v, None


def fused_sbda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """Run fused all-valid SBDA for q/k/v in ``[B,H,L,D]`` layout."""
    return _FusedSBDA.apply(q, k, v, offsets)


def fused_sbda_block(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """Run block-tiled fused SBDA with a custom forward/backward."""
    return _FusedSBDABlock.apply(q, k, v, offsets)


def fused_sbda_block_full(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """Experimental block forward plus vectorized block backward."""
    return _FusedSBDABlockFull.apply(q, k, v, offsets)


def fused_sbda_key_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """Run SBDA with saved small weights and key-centric backward."""
    return _FusedSBDAKeyBackward.apply(q, k, v, offsets)


def fused_sbda_gqa_key_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """Run SBDA with native GQA K/V heads and key-centric backward."""
    # The 640-dim/16-head production model has head_dim=40, which is padded
    # to a BLOCK_D=64 Triton tile. Combined with the GQA key-centric backward
    # this creates a high-register-pressure CUDA 13 specialization. Let the
    # caller's existing RuntimeError fallback use reference SBDA for such
    # non-canonical widths; 32/64/128 are the stable fused widths.
    if q.shape[-1] not in (32, 64, 128):
        raise RuntimeError(
            "GQA key-backward fused SBDA disabled for non-canonical head_dim"
        )
    return _FusedSBDAGQAKeyBackward.apply(q, k, v, offsets)


__all__ = [
    "fused_sbda", "fused_sbda_block", "fused_sbda_block_full",
    "fused_sbda_key_backward",
    "fused_sbda_gqa_key_backward",
]
