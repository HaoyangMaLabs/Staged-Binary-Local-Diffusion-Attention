"""SBDA models, with lightweight topology imports for structural scans."""

from .topology import sbda_stage

__all__ = ["BiAttnConfig", "BiAttnLM", "SBDAAttention", "sbda_stage"]


def __getattr__(name: str):
    if name in {"BiAttnConfig", "BiAttnLM"}:
        from . import model
        value = getattr(model, name)
    elif name == "SBDAAttention":
        from .left_v2 import SBDAAttention
        value = SBDAAttention
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value
