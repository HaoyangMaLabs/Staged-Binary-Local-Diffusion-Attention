"""Forward-only SBDA versus standard causal-attention benchmark.

This benchmark intentionally measures the complete model forward_hidden path,
not backward, optimizer work, or vocabulary projection.  Each length/mode is
run in a child process so an OOM or a timeout in dense causal attention does
not contaminate the remaining measurements.

Example:

    python BinaryDiffusionAttention/bench_sbda_vs_causal_forward.py \
        --grid --device cuda --dtype bf16 \
        --output BinaryDiffusionAttention/results/sbda_vs_causal_4090.json

The default model matches the small WikiText throughput experiments:
dim=96, layers=8, heads=4, FFN hidden=256, tied 6400-token vocabulary.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def gpu_info() -> str:
    try:
        value = subprocess.check_output(
            [
                "nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ], text=True, stderr=subprocess.STDOUT, timeout=10,
        ).strip()
        return value
    except Exception as exc:  # pragma: no cover - CPU/local smoke only
        return f"unavailable: {exc}"


def parse_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError(f"unsupported dtype: {name}")


def build_model(mode: str, args: argparse.Namespace) -> torch.nn.Module:
    from BiAttn import BiAttnLM
    from BinaryDiffusionAttention.comparison import comparison_config

    config = comparison_config(
        mode, vocab_size=args.vocab_size, dim=args.dim, layers=args.layers,
        heads=args.heads, ff_hidden=args.ff_hidden,
    )
    return BiAttnLM(config)


def run_single(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        return {"status": "no_cuda", "mode": args.mode, "length": args.length}
    device = torch.device(args.device)
    dtype = parse_dtype(args.dtype)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats(device)

    try:
        model = build_model(args.mode, args).to(device).eval()
        input_ids = torch.randint(
            0, args.vocab_size, (1, args.length),
            device=device, dtype=torch.long,
        )
        valid_mask = None
        # The all-valid case is deliberately represented as None.  This keeps
        # the dense path on native causal SDPA and the SBDA path on its fused
        # exact-source kernel.
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=dtype):
                for _ in range(args.warmup):
                    model.forward_hidden(input_ids, valid_mask)
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                started = time.perf_counter()
                for _ in range(args.iters):
                    output = model.forward_hidden(input_ids, valid_mask)
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - started
        peak = torch.cuda.max_memory_allocated(device) / (1024**3)
        return {
            "status": "ok",
            "mode": args.mode,
            "length": args.length,
            "iters": args.iters,
            "warmup": args.warmup,
            "seconds_per_forward": elapsed / args.iters,
            "tokens_per_second": args.length * args.iters / elapsed,
            "peak_allocated_gib": peak,
            "output_shape": list(output.shape),
        }
    except torch.cuda.OutOfMemoryError as exc:
        return {
            "status": "oom", "mode": args.mode, "length": args.length,
            "error": str(exc).splitlines()[0][:300],
        }
    except RuntimeError as exc:
        return {
            "status": "runtime_error", "mode": args.mode, "length": args.length,
            "error": str(exc).splitlines()[0][:500],
        }


def child_command(args: argparse.Namespace, mode: str, length: int) -> list[str]:
    return [
        sys.executable, str(Path(__file__).resolve()), "--single",
        "--mode", mode, "--length", str(length),
        "--device", args.device, "--dtype", args.dtype,
        "--dim", str(args.dim), "--layers", str(args.layers),
        "--heads", str(args.heads), "--ff-hidden", str(args.ff_hidden),
        "--vocab-size", str(args.vocab_size), "--iters", str(args.iters),
        "--warmup", str(args.warmup), "--seed", str(args.seed),
    ]


def run_grid(args: argparse.Namespace) -> dict[str, object]:
    # Both modes use the same lengths.  The 1M point is intentionally included
    # because FlashAttention-style causal SDPA may still fit in memory even
    # though its quadratic work makes it much slower.
    common_lengths = [
        4096, 8192, 16384, 32768, 65536, 131072,
        262144, 524288, 1048576,
    ]
    results: list[dict[str, object]] = []
    for mode, lengths in (("causal", common_lengths), ("sbda", common_lengths)):
        for length in lengths:
            print(f"running mode={mode} length={length}", flush=True)
            started = time.perf_counter()
            try:
                completed = subprocess.run(
                    child_command(args, mode, length),
                    cwd=REPO_ROOT,
                    capture_output=True,
                    text=True,
                    timeout=args.timeout,
                )
                lines = [line for line in completed.stdout.splitlines() if line.strip()]
                result = json.loads(lines[-1]) if lines else {
                    "status": "no_result", "stdout": completed.stdout[-500:],
                }
                if completed.returncode != 0 and result.get("status") == "ok":
                    result["status"] = "child_exit"
                if completed.stderr.strip():
                    result["stderr_tail"] = completed.stderr[-1000:]
            except subprocess.TimeoutExpired:
                result = {"status": "timeout", "mode": mode, "length": length,
                          "timeout_seconds": args.timeout}
            result["wall_seconds_including_startup"] = time.perf_counter() - started
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
            if mode == "causal" and result.get("status") in {"oom", "timeout"}:
                # A longer dense causal run cannot become cheaper; stop the
                # dense tail and proceed to the sparse maximum-length test.
                break
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": gpu_info(),
        "benchmark": "forward_hidden only; no backward, optimizer, or vocab head",
        "dtype": args.dtype,
        "batch_size": 1,
        "model": {
            "dim": args.dim, "layers": args.layers, "heads": args.heads,
            "ff_hidden": args.ff_hidden, "vocab_size": args.vocab_size,
        },
        "results": results,
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", action="store_true")
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--mode", choices=["sbda", "causal"], default="sbda")
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--dim", type=int, default=96)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ff-hidden", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=6400)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    if not args.grid and not args.single:
        parser.error("choose --grid or --single")
    return args


if __name__ == "__main__":
    cli = parse_args()
    if cli.single:
        print(json.dumps(run_single(cli), ensure_ascii=False), flush=True)
    else:
        run_grid(cli)
