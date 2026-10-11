"""Dyadic left/right-track attention language model."""

from __future__ import annotations

from dataclasses import dataclass
import math
import warnings
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .left_v2 import PATH_TOPOLOGIES, BinaryPathAttention, SBDAttention
from .three_track import LeftTrackAttention


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        if hasattr(F, "rms_norm"):
            return F.rms_norm(x, (self.dim,), self.weight, self.eps)
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps).to(dtype=x.dtype) * self.weight


class SwiGLU(nn.Module):
    def __init__(
        self,
        dim: int,
        ff_mult: float,
        dropout: float = 0.0,
        *,
        hidden_size: int | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        # MiniMind's SwiGLU width is rounded to a multiple of 64.  The old
        # 2*dim*ff_mult/3 formula made the default 128d block only 171 wide,
        # which was the main capacity regression found in the ablation.
        hidden = (
            max(64, math.ceil(dim * math.pi / 64) * 64)
            if hidden_size is None else hidden_size
        )
        if hidden < 1:
            raise ValueError("SwiGLU hidden_size must be positive")
        self.hidden_size = hidden
        self.gate = nn.Linear(dim, hidden, bias=bias)
        self.up = nn.Linear(dim, hidden, bias=bias)
        self.down = nn.Linear(hidden, dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class DeepSeekMoE(nn.Module):
    """Small DeepSeek-style MoE: one shared expert plus routed top-k experts.

    The shared expert is always active.  The routed experts are selected per
    token, so increasing their count increases capacity without making every
    token execute every expert.
    """

    def __init__(
        self,
        dim: int,
        hidden_size: int,
        experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if experts <= 0 or top_k <= 0 or top_k > experts:
            raise ValueError("experts must be positive and 0 < top_k <= experts")
        self.experts = int(experts)
        self.top_k = int(top_k)
        self.router = nn.Linear(dim, experts, bias=False)
        self.shared = SwiGLU(
            dim, 1.0, dropout, hidden_size=hidden_size, bias=bias,
        )
        self.routed = nn.ModuleList(
            SwiGLU(dim, 1.0, dropout, hidden_size=hidden_size, bias=bias)
            for _ in range(experts)
        )

    def forward(self, x: Tensor) -> Tensor:
        # Routing is done in fp32 for stable top-k decisions under autocast.
        route_logits = self.router(x.float())
        route_values, route_indices = torch.topk(
            route_logits, self.top_k, dim=-1,
        )
        route_weights = F.softmax(route_values, dim=-1).to(dtype=x.dtype)
        flat_x = x.reshape(-1, x.shape[-1])
        flat_indices = route_indices.reshape(-1)
        flat_weights = route_weights.reshape(-1)
        routed = torch.zeros_like(flat_x)
        # Fixed-capacity dispatch keeps the compiled graph shape-stable while
        # still evaluating only a small token bucket per routed expert.
        capacity = max(
            1, math.ceil(flat_x.shape[0] * self.top_k / self.experts * 1.5),
        )
        for expert_index, expert in enumerate(self.routed):
            selected = (flat_indices == expert_index).to(dtype=x.dtype)
            selected_scores, selected_slots = torch.topk(
                selected, capacity, dim=0, largest=True, sorted=False,
            )
            token_positions = torch.div(
                selected_slots, self.top_k, rounding_mode="floor",
            )
            expert_out = expert(flat_x.index_select(0, token_positions))
            weights = flat_weights.index_select(0, selected_slots)
            weights = weights * selected_scores.to(dtype=weights.dtype)
            expert_out = expert_out * weights[:, None]
            routed.index_add_(0, token_positions, expert_out)
        return self.shared(x) + routed.reshape_as(x)


class FactorizedTokenIO(nn.Module):
    """1024-wide token features with a low-rank tied vocabulary table."""

    def __init__(
        self, vocab_size: int, model_dim: int, token_dim: int, rank: int,
    ) -> None:
        super().__init__()
        self.token_table = nn.Embedding(vocab_size, rank)
        self.up = nn.Linear(rank, token_dim, bias=False)
        self.input_proj = nn.Linear(token_dim, model_dim, bias=False)
        self.output_proj = nn.Linear(model_dim, rank, bias=False)

    def embed(self, input_ids: Tensor) -> Tensor:
        return self.input_proj(self.up(self.token_table(input_ids)))

    def forward(self, hidden: Tensor) -> Tensor:
        # Keep the large-vocabulary projection in FP32 under FP16 autocast.
        # The sigmoid distillation loss is sensitive to small probability
        # differences across 151k classes; casting the logits to FP16 before
        # log-softmax can make the KL term explode even when the loss itself
        # is later converted back to FP32.
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            projected = self.output_proj(hidden.float())
            return F.linear(projected, self.token_table.weight.float())


@dataclass
class BiAttnConfig:
    vocab_size: int
    dim: int = 128
    layers: int = 12
    heads: int = 4
    kv_heads: int | None = None
    rope_theta: float = 1_000_000.0
    ff_mult: float = 2.0
    ff_hidden: int | None = None
    ff_bias: bool = False
    # The paper uses dense SwiGLU feed-forward blocks.
    ff_mode: str = "dense"
    dropout: float = 0.0
    tie_embeddings: bool = True
    # One implementation, with either two sources (left) or three sources
    # (self + left + shifted-right).  The latter is safe in triangle rows
    # because valid_mask cuts the right track off before the target.
    attention_backend: str = "left"
    path_topology: str = "binary"
    path_sdpa: bool = True
    sbda_fused: bool = True
    sbda_backward: str = "key"
    # Dense full-chapter CE needs a per-destination clip on the right track;
    # otherwise the parallel pass would read future targets.
    causal_right: bool = False
    # Optional equal-grid attention projection.  The residual stream remains
    # ``dim`` while attention may use a wider channel and a strided KV grid.
    attention_dim: int | None = None
    attention_stride: int = 1
    attention_grid: int | None = None
    flash_attention: bool = False
    attention_activation: str = "softmax"
    # Optional factorized token IO.  ``token_emb_dim`` is the effective token
    # feature width; ``embedding_rank`` keeps the large Qwen vocabulary small.
    token_emb_dim: int | None = None
    embedding_rank: int = 32
    moe_experts: int = 8
    moe_top_k: int = 2

    def __post_init__(self) -> None:
        # Fail safe for the experimental right-track backend.  The training
        # entry point already sets this explicitly, but direct callers often
        # construct ``BiAttnConfig(attention_backend="bidirectional")`` and
        # omit the flag.  Do not let that silently expose future positions.
        if self.attention_backend == "bidirectional" and not self.causal_right:
            warnings.warn(
                "bidirectional attention: enabling causal_right automatically",
                RuntimeWarning,
                stacklevel=2,
            )
            self.causal_right = True

    def window_schedule(self) -> tuple[int, ...]:
        return tuple(2 ** (index + 1) for index in range(self.layers))

    def validate(self) -> None:
        if self.vocab_size <= 0 or self.dim <= 0 or self.layers <= 0:
            raise ValueError("vocab_size, dim, and layers must be positive")
        if self.attention_backend not in (
            "left", "left_v2", "sbda", "full", "bidirectional",
        ):
            raise ValueError(
                "attention_backend must be left, left_v2, sbda, full, or bidirectional"
            )
        if self.attention_activation not in ("softmax", "sigmoid"):
            raise ValueError("attention_activation must be softmax or sigmoid")
        if self.path_topology not in PATH_TOPOLOGIES:
            raise ValueError(f"unknown path_topology: {self.path_topology}")
        if self.heads <= 0 or self.dim % self.heads:
            raise ValueError("dim must be divisible by heads")
        kv_heads = self.heads if self.kv_heads is None else self.kv_heads
        if kv_heads <= 0 or self.heads % kv_heads:
            raise ValueError("heads must be divisible by kv_heads")
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive")
        if self.ff_mult <= 0 or not 0 <= self.dropout < 1:
            raise ValueError("ff_mult must be positive and dropout in [0, 1)")
        if self.ff_mode not in ("dense", "deepseek_moe"):
            raise ValueError("ff_mode must be dense or deepseek_moe")
        if self.sbda_backward not in ("token", "block", "key"):
            raise ValueError("sbda_backward must be token, block, or key")
        if self.ff_hidden is not None and self.ff_hidden <= 0:
            raise ValueError("ff_hidden must be positive when provided")
        if self.token_emb_dim is not None and self.token_emb_dim <= 0:
            raise ValueError("token_emb_dim must be positive when provided")
        if self.embedding_rank <= 0:
            raise ValueError("embedding_rank must be positive")
        if self.moe_experts <= 0 or self.moe_top_k <= 0 or self.moe_top_k > self.moe_experts:
            raise ValueError("invalid MoE expert/top-k configuration")
        attention_dim = self.dim if self.attention_dim is None else self.attention_dim
        if attention_dim <= 0 or attention_dim % self.heads:
            raise ValueError("attention_dim must be positive and divisible by heads")
        if self.attention_stride <= 0:
            raise ValueError("attention_stride must be positive")
        if self.attention_grid is not None and self.attention_grid <= 0:
            raise ValueError("attention_grid must be positive when provided")
class BiAttnBlock(nn.Module):
    def __init__(self, config: BiAttnConfig, layer_index: int) -> None:
        super().__init__()
        self.layer_index = int(layer_index)
        self.norm1 = RMSNorm(config.dim)
        if config.attention_backend == "sbda":
            self.attn = SBDAttention(
                config.dim, config.heads, layer=layer_index + 1,
                kv_heads=config.kv_heads, rope_theta=config.rope_theta,
                total_layers=config.layers,
                use_sdpa=config.path_sdpa,
                use_fused=config.sbda_fused,
                backward_mode=config.sbda_backward,
            )
        elif config.attention_backend == "left_v2":
            self.attn = BinaryPathAttention(
                config.dim, config.heads, layer=layer_index + 1,
                kv_heads=config.kv_heads, rope_theta=config.rope_theta,
                topology=config.path_topology,
                total_layers=config.layers,
                use_sdpa=config.path_sdpa,
            )
        else:
            self.attn = LeftTrackAttention(
                config.dim, config.heads, layer=layer_index + 1,
                bidirectional=config.attention_backend == "bidirectional",
                full_causal=config.attention_backend == "full",
                causal_right=config.causal_right,
                kv_heads=config.kv_heads,
                rope_theta=config.rope_theta,
                attention_dim=config.attention_dim,
                attention_stride=config.attention_stride,
                attention_grid=config.attention_grid,
                flash_attention=config.flash_attention,
                attention_activation=config.attention_activation,
            )
        self.norm2 = RMSNorm(config.dim)
        if config.ff_mode == "deepseek_moe":
            hidden = config.ff_hidden
            if hidden is None:
                hidden = max(64, math.ceil(config.dim * math.pi / 64) * 64)
            self.ff = DeepSeekMoE(
                config.dim, hidden, config.moe_experts, config.moe_top_k,
                config.dropout, config.ff_bias,
            )
        else:
            self.ff = SwiGLU(
                config.dim, config.ff_mult, config.dropout,
                hidden_size=config.ff_hidden, bias=config.ff_bias,
            )

    def forward(
        self, x: Tensor, valid_mask: Optional[Tensor],
        seam_reset: Optional[Tensor] = None,
        path_indices: Optional[Tensor] = None,
        path_valid: Optional[Tensor] = None,
    ) -> Tensor:
        sigmoid_fp32 = (
            isinstance(self.attn, LeftTrackAttention)
            and self.attn.attention_activation == "sigmoid"
        )
        if sigmoid_fp32:
            # Sigmoid attention keeps a multi-value context mass rather than
            # a softmax-normalized average.  Keep its score, accumulation,
            # and residual stream in FP32 so FP16 does not amplify small
            # rounding errors through successive residual blocks.
            with torch.autocast(device_type=x.device.type, enabled=False):
                normalized = self.norm1(x.float())
                attended = self.attn(
                    normalized, valid_mask, seam_reset,
                )
            x = x.float() + attended.float()
        else:
            normalized = self.norm1(x)
            if isinstance(self.attn, BinaryPathAttention):
                layer_paths = None if path_indices is None else path_indices[:, self.layer_index]
                layer_valid = None if path_valid is None else path_valid[:, self.layer_index]
                attended = self.attn(
                    normalized, valid_mask, seam_reset, layer_paths, layer_valid,
                )
            else:
                attended = self.attn(normalized, valid_mask, seam_reset)
            x = x + attended
        ff_input = self.norm2(x)
        ff = self.ff(ff_input)
        if valid_mask is not None:
            ff = ff * valid_mask[:, :, None].to(x.dtype)
        return x + ff


class BiAttnLM(nn.Module):
    """Character LM backbone used with lower-triangular prefix batches."""

    def __init__(self, config: BiAttnConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.factorized_io = config.token_emb_dim is not None
        if self.factorized_io:
            self.token_io = FactorizedTokenIO(
                config.vocab_size, config.dim, config.token_emb_dim,
                config.embedding_rank,
            )
            self.embedding = None
        else:
            self.token_io = None
            self.embedding = nn.Embedding(config.vocab_size, config.dim)
        self.drop = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            BiAttnBlock(config, layer_index)
            for layer_index in range(config.layers)
        )
        self.norm = RMSNorm(config.dim)
        if self.factorized_io:
            self.lm_head = self.token_io
        else:
            self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
            if config.tie_embeddings:
                self.lm_head.weight = self.embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def forward_hidden(
        self, input_ids: Tensor, valid_mask: Optional[Tensor] = None,
        seam_reset: Optional[Tensor] = None,
        path_indices: Optional[Tensor] = None,
        path_valid: Optional[Tensor] = None,
        path_active: Optional[Tensor] = None,
    ) -> Tensor:
        """Return the final hidden states before the vocabulary projection."""
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if valid_mask is not None and (
            valid_mask.shape != input_ids.shape or valid_mask.dtype != torch.bool
        ):
            raise ValueError("valid_mask must be bool with input_ids shape")
        if seam_reset is not None and (
            seam_reset.shape != input_ids.shape or seam_reset.dtype != torch.bool
        ):
            raise ValueError("seam_reset must be bool with input_ids shape")
        if path_indices is not None and (
            path_indices.ndim != 4
            or path_indices.shape[0] != input_ids.shape[0]
            or path_indices.shape[2] != input_ids.shape[1]
            or path_indices.dtype != torch.long
        ):
            raise ValueError(
                "path_indices must have shape [batch, layers, sequence, source_count]"
            )
        if path_valid is not None and (
            path_indices is None
            or path_valid.shape != path_indices.shape
            or path_valid.dtype != torch.bool
        ):
            raise ValueError("path_valid must match path_indices as a bool tensor")
        if path_active is not None and (
            path_indices is None
            or path_active.ndim != 2
            or path_active.shape != path_indices.shape[:2]
            or path_active.dtype != torch.bool
            or path_active.device != input_ids.device
        ):
            raise ValueError(
                "path_active must have shape [batch, layers] and match path_indices"
            )
        if path_indices is not None and path_indices.shape[1] != len(self.layers):
            raise ValueError("path_indices layer count must match the model")
        if path_active is not None and path_active.shape[1] != len(self.layers):
            raise ValueError("path_active layer count must match the model")
        # A full chapter/window has a mask only for API uniformity.  Removing
        # an all-true mask once here avoids rebuilding and propagating the
        # same validity tensor through every sparse-attention layer.  The
        # reduction is one synchronization per forward, instead of repeated
        # mask broadcasts and gathers in every layer.
        # The eager all-true fast path avoids repeatedly carrying a mask
        # through every layer.  During torch.compile, however, converting the
        # reduction to a Python bool forces a graph break; keep the static
        # mask in the compiled graph instead.
        compiling = (
            hasattr(torch, "compiler") and torch.compiler.is_compiling()
        )
        if valid_mask is not None and not compiling and bool(valid_mask.all()):
            valid_mask = None
        if self.factorized_io:
            x = self.drop(self.token_io.embed(input_ids))
        else:
            x = self.drop(self.embedding(input_ids))
        for layer_index, layer in enumerate(self.layers):
            if path_active is None:
                updated = layer(
                    x, valid_mask, seam_reset, path_indices, path_valid,
                )
                x = updated
                continue
            active = path_active[:, layer_index]
            if not bool(active.any()):
                continue
            if bool(active.all()):
                updated = layer(
                    x, valid_mask, seam_reset, path_indices, path_valid,
                )
                x = updated
                continue
            rows = active.nonzero(as_tuple=False).flatten()
            updated = layer(
                x.index_select(0, rows),
                None if valid_mask is None else valid_mask.index_select(0, rows),
                None if seam_reset is None else seam_reset.index_select(0, rows),
                None if path_indices is None else path_indices.index_select(0, rows),
                None if path_valid is None else path_valid.index_select(0, rows),
            )
            x = x.clone()
            x.index_copy_(0, rows, updated)
        return self.norm(x)

    def forward(
        self, input_ids: Tensor, valid_mask: Optional[Tensor] = None,
        seam_reset: Optional[Tensor] = None,
        path_indices: Optional[Tensor] = None,
        path_valid: Optional[Tensor] = None,
        path_active: Optional[Tensor] = None,
    ) -> Tensor:
        return self.lm_head(self.forward_hidden(
            input_ids, valid_mask, seam_reset, path_indices, path_valid,
            path_active,
        ))

    def next_logits(self, input_ids: Tensor, valid_mask: Tensor) -> Tensor:
        """Return one logit vector per triangle row, at its last valid token."""

        hidden = self.forward_hidden(input_ids, valid_mask)
        lengths = valid_mask.sum(dim=1) - 1
        rows = torch.arange(input_ids.shape[0], device=input_ids.device)
        return self.lm_head(hidden[rows, lengths])

    def next_predictions(self, input_ids: Tensor, valid_mask: Tensor) -> Tensor:
        """Decode the next token at each triangle row."""
        return self.next_logits(input_ids, valid_mask).argmax(dim=-1)

    def triangle_loss(
        self, input_ids: Tensor, valid_mask: Tensor, targets: Tensor
    ) -> Tensor:
        if targets.ndim != 1 or targets.shape[0] != input_ids.shape[0]:
            raise ValueError("targets must contain one next token per triangle row")
        return F.cross_entropy(self.next_logits(input_ids, valid_mask), targets)
