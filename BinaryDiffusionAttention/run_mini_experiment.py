"""Mini matched comparison for dense and fixed sparse causal attention.

The script deliberately uses only the Python standard library plus PyTorch and
the repository's BiAttn implementation. It supports WikiText-2, Tiny
Shakespeare, and Penn Treebank, uses a character vocabulary, and compares
exact-token path topologies under the same optimizer and batches.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F

# Allow the script to be launched directly as
# ``python BinaryDiffusionAttention/run_mini_experiment.py`` from any cwd.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from BiAttn.left_v2 import path_reachability, topology_offsets
from BiAttn.model import BiAttnConfig, BiAttnLM
from BiAttn.topology import greedy_internal_offsets
from BinaryDiffusionAttention.datasets import download_dataset, encode, make_vocab, read_split


MODE_SPECS = {
    "full": ("full", "binary"),
    "left_local": ("left", "binary"),
    "binary_diffusion": ("left_v2", "binary"),
    "local_jump": ("left_v2", "local_jump"),
    "butterfly": ("left_v2", "butterfly"),
    "radix4": ("left_v2", "radix4"),
    "high4": ("left_v2", "high4"),
    "longformer_local": ("left_v2", "longformer_local"),
    "sparse_local_stride": ("left_v2", "sparse_local_stride"),
    "dinat_like": ("left_v2", "dinat_like"),
    "longnet_like": ("left_v2", "longnet_like"),
    "power_attention": ("left_v2", "power_attention"),
    "h0": ("left_v2", "h0"),
    "h1": ("left_v2", "h1"),
    "h2": ("left_v2", "h2"),
    "h3": ("left_v2", "h3"),
    "h4": ("left_v2", "h4_greedy"),
    "h5": ("left_v2", "h5"),
    "h6": ("left_v2", "h6"),
    "h7": ("left_v2", "h7"),
    "h8": ("left_v2", "h8"),
    "h8a": ("left_v2", "h8a"),
    "h8b": ("left_v2", "h8b"),
    "h8c": ("left_v2", "h8c"),
    "h8d": ("left_v2", "h8d"),
    "h8e": ("left_v2", "h8e"),
    "h8f": ("left_v2", "h8f"),
    "h8g": ("left_v2", "h8g"),
}


def sample_batch(
    data: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
    rng: random.Random,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if data.numel() <= seq_len + 1:
        raise ValueError("dataset split is shorter than seq_len")
    starts = [rng.randrange(0, data.numel() - seq_len - 1) for _ in range(batch_size)]
    inputs = torch.stack([data[start : start + seq_len] for start in starts])
    targets = torch.stack([data[start + 1 : start + seq_len + 1] for start in starts])
    return inputs.to(device=device), targets.to(device=device)


def eval_loss(
    model: BiAttnLM,
    data: torch.Tensor,
    *,
    seq_len: int,
    batch_size: int,
    batches: int,
    device: torch.device,
    path_indices: torch.Tensor | None = None,
    path_valid: torch.Tensor | None = None,
) -> float:
    model.eval()
    total = 0.0
    tokens = 0
    with torch.no_grad():
        for batch_index in range(batches):
            start = (batch_index * seq_len * batch_size) % (data.numel() - seq_len - 1)
            starts = [
                (start + row * seq_len) % (data.numel() - seq_len - 1)
                for row in range(batch_size)
            ]
            inputs = torch.stack([data[pos : pos + seq_len] for pos in starts]).to(device)
            targets = torch.stack(
                [data[pos + 1 : pos + seq_len + 1] for pos in starts]
            ).to(device)
            valid = torch.ones_like(inputs, dtype=torch.bool)
            logits = model(
                inputs, valid, path_indices=path_indices, path_valid=path_valid,
            )
            total += float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="sum"))
            tokens += targets.numel()
    model.train()
    return total / max(1, tokens)


def h4_path_tensors(
    *, batch_size: int, length: int, layers: int, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return padded exact source paths and masks for H4."""
    schedule = greedy_internal_offsets(length, layers)
    source_count = max(len(offsets) for offsets in schedule)
    positions = torch.arange(length, device=device, dtype=torch.long)
    indices = torch.zeros(
        (batch_size, layers, length, source_count),
        dtype=torch.long, device=device,
    )
    valid = torch.zeros_like(indices, dtype=torch.bool)
    for layer_index, offsets in enumerate(schedule):
        for source_index, offset in enumerate(offsets):
            indices[:, layer_index, :, source_index] = (positions - offset).clamp_min(0)
            valid[:, layer_index, :, source_index] = positions >= offset
    return indices, valid


def internal_route_stats(
    route: str, *, seq_len: int, layers: int,
) -> tuple[float, int]:
    if route == "h4":
        schedule = greedy_internal_offsets(seq_len, layers)
        route_offsets = schedule
    else:
        route_offsets = [
            topology_offsets(layer, topology=route, total_layers=layers)
            for layer in range(1, layers + 1)
        ]
    source_total = sum(
        max(0, seq_len - offset)
        for offsets in route_offsets for offset in offsets
    )
    reachable = {0}
    for offsets in route_offsets:
        reachable = {
            previous + offset
            for previous in reachable
            for offset in offsets
            if previous + offset < seq_len
        }
    return source_total / (layers * seq_len), len(reachable)


def source_edges_per_query(
    mode: str, *, seq_len: int, layers: int,
) -> float:
    backend, topology = MODE_SPECS[mode]
    if backend == "full":
        return (seq_len + 1) / 2.0
    if backend == "left":
        # Aligned dyadic block-causal attention.  This is the average causal
        # prefix width over the layers and positions.
        total = 0
        for layer in range(1, layers + 1):
            width = min(2**layer, seq_len)
            total += sum(min(index % width + 1, width) for index in range(seq_len))
        return total / (layers * seq_len)
    if mode.startswith("h"):
        return internal_route_stats(mode, seq_len=seq_len, layers=layers)[0]
    return float(sum(len(topology_offsets(layer, topology=topology, total_layers=layers)) for layer in range(1, layers + 1)) / layers)


def run_one(
    mode: str,
    *,
    train: torch.Tensor,
    valid_data: torch.Tensor,
    vocab_size: int,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, object]:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(args.device)
    backend, topology = MODE_SPECS[mode]
    config = BiAttnConfig(
        vocab_size=vocab_size,
        dim=args.dim,
        layers=args.layers,
        heads=args.heads,
        ff_hidden=args.ff_hidden,
        tie_embeddings=True,
        attention_backend=backend,
        path_topology=topology,
        path_sdpa=True,
    )
    model = BiAttnLM(config).to(device)
    path_indices = path_valid = None
    if mode == "h4":
        path_indices, path_valid = h4_path_tensors(
            batch_size=args.batch_size, length=args.seq_len,
            layers=args.layers, device=device,
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    rng = random.Random(seed + 17)
    warmup = min(args.warmup, args.steps)
    losses: list[float] = []
    train_loss_trace: dict[str, float] = {}
    started = time.perf_counter()
    measured_started: float | None = None
    measured_tokens = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    for step in range(args.steps):
        inputs, targets = sample_batch(
            train, batch_size=args.batch_size, seq_len=args.seq_len,
            rng=rng, device=device,
        )
        valid = torch.ones_like(inputs, dtype=torch.bool)
        optimizer.zero_grad(set_to_none=True)
        logits = model(
            inputs, valid, path_indices=path_indices, path_valid=path_valid,
        )
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        if step == 0 or (step + 1) % args.trace_every == 0 or step + 1 == args.steps:
            train_loss_trace[str(step + 1)] = float(loss.detach())
        if step == warmup:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            measured_started = time.perf_counter()
        if step >= warmup:
            measured_tokens += targets.numel()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    measured_elapsed = (
        elapsed if measured_started is None
        else max(1e-9, time.perf_counter() - measured_started)
    )
    val = eval_loss(
        model, valid_data, seq_len=args.seq_len, batch_size=args.batch_size,
        batches=args.eval_batches, device=device,
        path_indices=path_indices, path_valid=path_valid,
    )
    reachable = (
        internal_route_stats(mode, seq_len=args.seq_len, layers=args.layers)[1]
        if mode.startswith("h") else
        len(path_reachability(args.seq_len, args.layers, topology=topology)[-1])
        if backend == "left_v2" else
        (args.seq_len if backend == "full" else min(2**args.layers, args.seq_len))
    )
    peak_memory = 0
    if device.type == "cuda":
        peak_memory = int(torch.cuda.max_memory_allocated(device) / (1024**2))
    return {
        "mode": mode,
        "backend": backend,
        "topology": topology,
        "seed": seed,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "vocab_size": vocab_size,
        "dim": args.dim,
        "layers": args.layers,
        "heads": args.heads,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "final_train_loss": losses[-1],
        "mean_last_10_loss": sum(losses[-10:]) / min(10, len(losses)),
        "train_loss_trace": train_loss_trace,
        "val_loss": val,
        "val_ppl": float(torch.exp(torch.tensor(min(val, 20.0)))),
        "tokens_per_second_after_warmup": measured_tokens / measured_elapsed,
        "elapsed_seconds": elapsed,
        "peak_memory_mb": peak_memory,
        "source_edges_per_query": source_edges_per_query(mode, seq_len=args.seq_len, layers=args.layers),
        "reachable_sources": reachable,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument(
        "--dataset", choices=("wikitext2", "tinyshakespeare", "ptb"),
        default="wikitext2",
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--trace-every", type=int, default=50,
        help="record one train-loss point every N optimizer steps",
    )
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dim", type=int, default=96)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ff-hidden", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--modes", default=",".join(MODE_SPECS))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps < 1 or args.seq_len < 2 or args.batch_size < 1 or args.trace_every < 1:
        raise ValueError("steps, seq-len, batch-size, and trace-every must be positive")
    paths = download_dataset(args.data_root, args.dataset)
    train_text = read_split(paths["train"])
    vocab = make_vocab(train_text)
    train = encode(train_text, vocab)
    valid_data = encode(read_split(paths["valid"]), vocab)
    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    unknown_modes = sorted(set(modes) - set(MODE_SPECS))
    if unknown_modes:
        raise ValueError(f"unknown modes: {unknown_modes}")
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    results: list[dict[str, object]] = []
    print(
        f"dataset={args.dataset} train_chars:{train.numel()} "
        f"valid_chars:{valid_data.numel()} vocab:{len(vocab)}",
        flush=True,
    )
    print(f"modes={modes} seeds={seeds} device={args.device}", flush=True)
    for seed in seeds:
        for mode in modes:
            result = run_one(
                mode, train=train, valid_data=valid_data, vocab_size=len(vocab),
                args=args, seed=seed,
            )
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    json_path = args.output_dir / f"{args.dataset}_binary_diffusion_{stamp}.json"
    csv_path = args.output_dir / f"{args.dataset}_binary_diffusion_{stamp}.csv"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    fields = list(results[0]) if results else []
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    print(f"saved_json={json_path}", flush=True)
    print(f"saved_csv={csv_path}", flush=True)


if __name__ == "__main__":
    main()
