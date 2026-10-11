"""Matched model configuration for the long-context quality and speed studies."""

from BiAttn import BiAttnConfig


def comparison_config(
    mode: str, *, vocab_size: int, dim: int, layers: int, heads: int, ff_hidden: int,
) -> BiAttnConfig:
    if mode not in {"sbda", "causal"}:
        raise ValueError(f"unknown comparison mode: {mode}")
    return BiAttnConfig(
        vocab_size=vocab_size, dim=dim, layers=layers, heads=heads,
        ff_hidden=ff_hidden, tie_embeddings=True,
        attention_backend="sbda" if mode == "sbda" else "full",
        path_sdpa=True, sbda_fused=True, sbda_backward="key", dropout=0.0,
    )
