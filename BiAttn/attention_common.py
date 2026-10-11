"""Shared RoPE, per-head normalization and grouped-query helpers."""

from __future__ import annotations

import torch
from torch import Tensor, nn

def _rotate_half(x: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _rope_cos_sin(
    positions: Tensor, head_dim: int, theta: float, dtype: torch.dtype
) -> tuple[Tensor, Tensor]:
    indices = torch.arange(0, head_dim, 2, device=positions.device, dtype=torch.float32)
    inverse = theta ** (-indices / head_dim)
    angles = torch.outer(positions.to(torch.float32), inverse)
    return (
        torch.cat((angles.cos(), angles.cos()), dim=-1)[None, :, None, :].to(dtype),
        torch.cat((angles.sin(), angles.sin()), dim=-1)[None, :, None, :].to(dtype),
    )


def _apply_rope(
    q: Tensor,
    k: Tensor,
    *,
    theta: float,
    q_positions: Tensor | None = None,
    k_positions: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Apply half-split RoPE to Q/K grids that may have different lengths."""

    head_dim = q.shape[-1]
    if head_dim % 2:
        # A one-dimensional attention head has no valid two-coordinate
        # rotation.  Keeping the scalar unchanged is the only information-
        # preserving fallback and still leaves the causal grid/mask intact.
        return q, k
    if q_positions is None:
        q_positions = torch.arange(q.shape[1], device=q.device)
    if k_positions is None:
        k_positions = torch.arange(k.shape[1], device=k.device)
    q_cos, q_sin = _rope_cos_sin(q_positions, head_dim, theta, q.dtype)
    k_cos, k_sin = _rope_cos_sin(k_positions, head_dim, theta, k.dtype)
    return (
        q * q_cos + _rotate_half(q) * q_sin,
        k * k_cos + _rotate_half(k) * k_sin,
    )


def _repeat_kv(x: Tensor, repeats: int) -> Tensor:
    if repeats == 1:
        return x
    batch, length, heads, head_dim = x.shape
    return (
        x[:, :, :, None, :]
        .expand(batch, length, heads, repeats, head_dim)
        .reshape(batch, length, heads * repeats, head_dim)
    )


class _HeadRMSNorm(nn.Module):
    """MiniMind-compatible RMSNorm for the per-head Q/K tensors."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        normalized = x.float() * torch.rsqrt(
            x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        return (normalized * self.weight).to(dtype=x.dtype)
