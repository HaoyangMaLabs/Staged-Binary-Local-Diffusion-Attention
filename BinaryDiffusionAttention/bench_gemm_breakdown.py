"""Profile unchanged dense/SBDA models; GPU-kernel shares are NOT wall-time shares.

Forward: original 96d/8-layer/256-FFN/6400-vocab hidden-state benchmark.
Train: original 64d/16-layer/128-FFN/1014-vocab full CE + AdamW step.
Random device-resident IDs isolate computation; no dataset IO or quality claims.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from contextlib import nullcontext
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch
from torch import nn
from torch.nn import functional as F
from torch.profiler import profile, ProfilerActivity, record_function

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from BiAttn.model import BiAttnConfig, BiAttnLM


def instrument(model):
    """Local per-instance wrappers: never edit the backbone or its mathematics."""
    originals = []
    for name, module in model.named_modules():
        tag = None
        if isinstance(module, nn.Linear):
            if name == "lm_head":
                tag = "vocab"
            elif ".ff." in name:
                tag = "ffn"
            elif name.endswith((".out", ".left_out")):
                tag = "out"
            elif ".attn." in name:
                tag = "qkv"
        elif name.endswith(".attn"):
            tag = "attention"
        if tag:
            fn = module.forward
            originals.append((module, fn))
            def wrapped(*args, _fn=fn, _tag=tag, **kwargs):
                with record_function("component:" + _tag):
                    return _fn(*args, **kwargs)
            module.forward = wrapped
    return originals


def ancestors(event):
    while event is not None:
        yield event
        event = event.cpu_parent


def summarize(prof):
    events = prof.events()
    forward_sequences = {}
    for event in events:
        chain = list(ancestors(event))
        is_forward = any(e.name == "phase:forward" for e in chain)
        component = next((e.name.split(":", 1)[1] for e in chain
                          if e.name.startswith("component:")), None)
        if is_forward and component and event.sequence_nr >= 0:
            forward_sequences[event.sequence_nr] = component
    totals, phases, operators = defaultdict(float), defaultdict(float), defaultdict(float)
    unmapped_backward_gemm = 0.
    for event in events:
        # Self device time on CPU launch/operator records avoids counting
        # inclusive parent regions or the separate CUDA kernel records twice.
        if event.device_type != torch.autograd.DeviceType.CPU:
            continue
        us = event.self_device_time_total
        if us <= 0:
            continue
        chain = list(ancestors(event))
        phase = next((e.name.split(":", 1)[1] for e in chain if e.name.startswith("phase:")), "other")
        # Autograd launches on worker threads; it need not inherit the
        # record_function region on the Python thread that calls backward().
        if phase == "other" and any("Backward" in e.name or "autograd::engine::evaluate_function" in e.name for e in chain):
            phase = "backward"
        component = next((e.name.split(":", 1)[1] for e in chain
                          if e.name.startswith("component:")), None)
        if phase == "backward" and component is None:
            component = next((forward_sequences[e.sequence_nr] for e in chain
                              if e.sequence_nr in forward_sequences), None)
        is_gemm = event.name in {"aten::mm", "aten::addmm", "aten::bmm", "aten::baddbmm"}
        if is_gemm and component in {"qkv", "out", "ffn", "vocab"}:
            category = component + "_gemm"
        elif component == "attention" and ("scaled_dot_product" in event.name or
                "flash" in event.name.lower() or "efficient_attention" in event.name or
                "SBDA" in event.name or "triton" in event.name.lower()):
            category = "attention_kernel"
        elif phase == "optimizer":
            category = "optimizer_gpu"
        else:
            category = "other_gpu"
            if phase == "backward" and is_gemm and component is None:
                unmapped_backward_gemm += us / 1000
        totals[category] += us / 1000
        phases[phase] += us / 1000
        operators[event.name] += us / 1000
    total = sum(totals.values())
    assert total > 0, "CUPTI returned no GPU kernel times"
    return {
        "gpu_kernel_ms": total,
        "category_ms": dict(totals),
        "category_pct_gpu_kernel_time": {k: v/total*100 for k, v in totals.items()},
        "phase_gpu_kernel_ms": dict(phases),
        "unmapped_backward_gemm_ms": unmapped_backward_gemm,
        "top_operators_gpu_ms": sorted(operators.items(), key=lambda x: -x[1])[:25],
    }


def run(args):
    torch.manual_seed(1)
    dim, layers, ff, vocab = (96, 8, 256, 6400) if args.kind == "forward" else (64, 16, 128, 1014)
    if args.layers is not None:
        layers = args.layers
    model = BiAttnLM(BiAttnConfig(vocab_size=vocab, dim=dim, layers=layers, heads=4,
        ff_hidden=ff, tie_embeddings=True, attention_backend="sbda" if args.mode == "sbda" else "full",
        path_sdpa=True, sbda_fused=True, sbda_backward="key", dropout=0.)).cuda()
    training = args.kind == "train"
    model.train(training)
    ids = torch.randint(vocab, (1, args.length+1), device="cuda")
    x, targets = ids[:, :-1].contiguous(), ids[:, 1:].contiguous()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4) if training else None

    def step(tagged=False):
        scope = record_function if tagged else lambda _: nullcontext()
        if training:
            optimizer.zero_grad(set_to_none=True)
        with scope("phase:forward"), torch.autocast("cuda", dtype=torch.bfloat16):
            if training:
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, vocab), targets.reshape(-1))
            else:
                with torch.inference_mode():
                    result = model.forward_hidden(x)
        if training:
            with scope("phase:backward"):
                loss.backward()
            with scope("phase:optimizer"):
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    wall_ms, stream_ms = [], []
    for _ in range(args.iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        wall_start = time.perf_counter()
        start.record()
        step()
        end.record()
        torch.cuda.synchronize()
        wall_ms.append((time.perf_counter()-wall_start)*1000)
        stream_ms.append(start.elapsed_time(end))
    peak = torch.cuda.max_memory_allocated()/2**30
    originals = instrument(model)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(args.profile_iters):
            step(tagged=True)
        torch.cuda.synchronize()
    for module, fn in originals:
        module.forward = fn
    breakdown = summarize(prof)
    for key in ["gpu_kernel_ms", "unmapped_backward_gemm_ms"]:
        breakdown[key] /= args.profile_iters
    for key in ["category_ms", "phase_gpu_kernel_ms"]:
        breakdown[key] = {k: v/args.profile_iters for k, v in breakdown[key].items()}
    breakdown["top_operators_gpu_ms"] = [(k, v/args.profile_iters) for k, v in breakdown["top_operators_gpu_ms"]]
    fixed = sum(v for k, v in breakdown["category_ms"].items() if k.endswith("_gemm"))
    edges = sum(sum(max(args.length-o, 0) for o in
                ([0, 2**l] if l < 2 else [0, 1, 2, 2**l])) for l in range(layers))
    fixed_flops = 2*args.length*(layers*(4*dim*dim+3*dim*ff)+(dim*vocab if training else 0))
    dense_flops = 2*layers*dim*args.length*(args.length+1)
    sparse_flops = 4*dim*edges
    result = {
        "status": "ok", "created_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], text=True).strip(),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "mode": args.mode, "kind": args.kind, "length": args.length, "dtype": "bf16", "batch": 1,
        "config": {"dim": dim, "layers": layers, "ff_hidden": ff, "vocab": vocab, "heads": 4},
        "unprofiled_wall_ms_median": statistics.median(wall_ms), "wall_ms_samples": wall_ms,
        "cuda_stream_ms_median": statistics.median(stream_ms), "peak_allocated_gib": peak,
        "warmup": args.warmup, "iters": args.iters, "profile_iters": args.profile_iters,
        **breakdown,
        "fixed_gemm_pct_gpu_kernel_time": fixed / breakdown["gpu_kernel_ms"] * 100,
        "gpu_kernel_only_ceiling_if_only_gemm_remains": breakdown["gpu_kernel_ms"] / fixed if fixed else None,
        "forward_flop_model": {"fixed_gemm": fixed_flops, "dense_interactions": dense_flops,
            "sparse_interactions": sparse_flops, "dense_fixed_gemm_pct": 100*fixed_flops/(fixed_flops+dense_flops)},
        "notes": ["GPU shares sum exclusive CPU-operator device times, not end-to-end wall-time shares.",
                  "Backward GEMMs mapped using forward/backward autograd sequence numbers.",
                  "GEMM means mm/addmm/bmm/baddbmm kernels; casts, activations and launch gaps excluded.",
                  "Kernel-only ceiling is a decomposition thought experiment, not a wall-time bound.",
                  "Forward FLOP model excludes softmax, norms, activations and causal-tile overcompute."]}
    if args.trace:
        Path(args.trace).parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(args.trace)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["causal", "sbda"], required=True)
    p.add_argument("--kind", choices=["forward", "train"], required=True)
    p.add_argument("--length", type=int, required=True)
    p.add_argument("--layers", type=int, default=None)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--profile-iters", type=int, default=2)
    p.add_argument("--output", required=True)
    p.add_argument("--trace", default="")
    args = p.parse_args()
    try:
        result = run(args)
    except torch.cuda.OutOfMemoryError as exc:
        result = {"status": "oom", "mode": args.mode, "kind": args.kind, "length": args.length, "error": str(exc)}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
