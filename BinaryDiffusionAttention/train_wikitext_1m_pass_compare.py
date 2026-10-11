"""Compare matched dense-causal and SBDA training on WikiText-2.

Sequence length, model depth and training passes are configurable. Complete,
non-overlapping windows are repeated in the same order for both modes;
training and held-out validation cross-entropy are recorded per pass.
The filename retains the original million-position pilot entry point.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from BiAttn import BiAttnLM
from BinaryDiffusionAttention.comparison import comparison_config
from BinaryDiffusionAttention.datasets import (
    complete_windows, download_wikitext, encode, make_vocab, read_split,
)


def gpu_info() -> str:
    try:
        return torch.cuda.get_device_name(0)
    except Exception:
        return "unavailable"


def one_mode(args: argparse.Namespace) -> dict[str, object]:
    if args.seq_len < 1 or args.passes < 1:
        raise ValueError("seq_len and passes must be positive")
    device = torch.device(args.device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    use_cuda = device.type == "cuda"
    if use_cuda and not torch.cuda.is_available():
        return {"status": "no_cuda", "mode": args.mode}
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)

    paths = download_wikitext(Path(args.data_root))
    train_text = read_split(paths["train"])
    vocab = make_vocab(train_text)
    train_ids = encode(train_text, vocab)
    windows, dropped_train_tokens = complete_windows(train_ids, args.seq_len)
    if not windows:
        raise RuntimeError(f"WikiText-2 train split has no complete window for seq_len={args.seq_len}")
    valid_text = read_split(paths["valid"])
    valid_ids = encode(valid_text, vocab)
    dev_windows, dropped_valid_tokens = complete_windows(valid_ids, args.seq_len)
    if not dev_windows:
        raise RuntimeError("WikiText-2 validation split has no complete window")

    config = comparison_config(
        args.mode, vocab_size=len(vocab), dim=args.dim, layers=args.layers,
        heads=args.heads, ff_hidden=args.ff_hidden,
    )
    model = BiAttnLM(config).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    pass_records: list[dict[str, object]] = []
    total_tokens = 0
    total_seconds = 0.0

    for pass_index in range(args.passes):
        pass_losses: list[float] = []
        pass_started = time.perf_counter()
        for window_index, sequence in enumerate(windows):
            inputs = sequence[:-1].unsqueeze(0).to(device=device)
            targets = sequence[1:].unsqueeze(0).to(device=device)
            optimizer.zero_grad(set_to_none=True)
            with (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_cuda else nullcontext()
            ):
                logits = model(inputs)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if use_cuda:
                torch.cuda.synchronize(device)
            value = float(loss.detach())
            pass_losses.append(value)
            total_tokens += targets.numel()
            print(
                f"mode={args.mode} pass={pass_index + 1}/{args.passes} "
                f"window={window_index + 1}/{len(windows)} loss={value:.6f}",
                flush=True,
            )
        pass_seconds = time.perf_counter() - pass_started
        total_seconds += pass_seconds
        pass_records.append({
            "pass": pass_index + 1,
            "mean_loss": sum(pass_losses) / len(pass_losses),
            "windows": len(windows),
            "tokens": len(windows) * args.seq_len,
            "seconds": pass_seconds,
            "tokens_per_second": len(windows) * args.seq_len / pass_seconds,
        })
        model.eval()
        dev_losses: list[float] = []
        with torch.no_grad():
            for sequence in dev_windows:
                inputs = sequence[:-1].unsqueeze(0).to(device=device)
                targets = sequence[1:].unsqueeze(0).to(device=device)
                with (
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if use_cuda else nullcontext()
                ):
                    logits = model(inputs)
                    dev_loss = F.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                    )
                dev_losses.append(float(dev_loss))
        if use_cuda:
            torch.cuda.synchronize(device)
        pass_records[-1]["dev_loss"] = sum(dev_losses) / len(dev_losses)
        pass_records[-1]["dev_windows"] = len(dev_windows)
        model.train()

    result = {
        "status": "ok",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "hardware": gpu_info() if use_cuda else "cpu",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "dtype": "bf16" if use_cuda else "fp32",
        "batch_size": 1,
        "seq_len": args.seq_len,
        "passes": args.passes,
        "complete_windows": len(windows),
        "dropped_train_tokens": dropped_train_tokens,
        "dev_source": "wiki.valid.raw",
        "dev_complete_windows": len(dev_windows),
        "dropped_valid_tokens": dropped_valid_tokens,
        "vocab_size": len(vocab),
        "model": {
            "dim": args.dim, "layers": args.layers, "heads": args.heads,
            "ff_hidden": args.ff_hidden,
        },
        "optimizer": {"name": "AdamW", "lr": args.lr},
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30 if use_cuda else 0.0,
        "total_seconds": total_seconds,
        "tokens_per_second_all_passes": total_tokens / total_seconds,
        "pass_records": pass_records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["causal", "sbda"], required=True)
    p.add_argument("--data-root", default=str(Path(__file__).resolve().parent / "data"))
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seq-len", type=int, default=1_048_576)
    p.add_argument("--passes", type=int, default=3)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--ff-hidden", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


if __name__ == "__main__":
    one_mode(parse_args())
