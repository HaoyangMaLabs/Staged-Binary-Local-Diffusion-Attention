"""Shared deterministic character datasets for paper experiments."""

from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
import urllib.request

import torch

MIRROR_BASE = "https://cosmo.zip/pub/datasets/wikitext-2-raw"
SPLIT_FILES = {
    "train": "wiki.train.raw",
    "valid": "wiki.valid.raw",
    "test": "wiki.test.raw",
}
PTB_BASE_URL = "https://raw.githubusercontent.com/wojzaremba/lstm/master/data"
TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)


def download_wikitext(root: Path) -> dict[str, str]:
    """Return cached WikiText-2 raw splits, downloading missing files."""
    paths: dict[str, str] = {}
    for split, filename in SPLIT_FILES.items():
        target = root / "wikitext-2-raw" / filename
        download_file(f"{MIRROR_BASE}/{filename}", target, minimum_bytes=1024)
        paths[split] = str(target)
    return paths


def download_file(url: str, target: Path, *, minimum_bytes: int = 128) -> None:
    """Download atomically so interrupted transfers cannot poison the cache."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size >= minimum_bytes:
        return
    print(f"downloading {url}", flush=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
            temporary = Path(stream.name)
            with urllib.request.urlopen(url, timeout=60) as response:
                shutil.copyfileobj(response, stream)
        if temporary.stat().st_size < minimum_bytes:
            raise ValueError(f"download contains fewer than {minimum_bytes} bytes")
        temporary.replace(target)
    except Exception as error:
        raise RuntimeError(
            f"public dataset download failed for {url}; retry or place {target} manually"
        ) from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def download_tiny_shakespeare(root: Path) -> dict[str, str]:
    """Download Tiny Shakespeare and make deterministic contiguous splits."""
    dataset_root = root / "tinyshakespeare"
    raw_path = dataset_root / "input.txt"
    download_file(TINY_SHAKESPEARE_URL, raw_path, minimum_bytes=1024)
    raw = raw_path.read_text(encoding="utf-8")
    split_points = (int(len(raw) * 0.90), int(len(raw) * 0.95))
    chunks = {
        "train": raw[:split_points[0]],
        "valid": raw[split_points[0]:split_points[1]],
        "test": raw[split_points[1]:],
    }
    paths: dict[str, str] = {}
    for split, text in chunks.items():
        target = dataset_root / f"{split}.txt"
        if not target.exists() or target.stat().st_size < 128:
            target.write_text(text, encoding="utf-8")
        paths[split] = str(target)
    return paths


def download_ptb(root: Path) -> dict[str, str]:
    dataset_root = root / "ptb"
    paths: dict[str, str] = {}
    for split in ("train", "valid", "test"):
        filename = f"ptb.{split}.txt"
        target = dataset_root / filename
        download_file(f"{PTB_BASE_URL}/{filename}", target, minimum_bytes=1024)
        paths[split] = str(target)
    return paths


def download_dataset(root: Path, dataset: str) -> dict[str, str]:
    if dataset == "wikitext2":
        return download_wikitext(root)
    if dataset == "tinyshakespeare":
        return download_tiny_shakespeare(root)
    if dataset == "ptb":
        return download_ptb(root)
    raise ValueError(f"unknown dataset: {dataset}")


def read_split(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def make_vocab(text: str) -> dict[str, int]:
    # Keep a deterministic unknown token at id zero.  All train characters are
    # retained, including whitespace and punctuation.
    chars = sorted(set(text))
    return {"<unk>": 0, **{char: index + 1 for index, char in enumerate(chars)}}


def encode(text: str, vocab: dict[str, int]) -> torch.Tensor:
    unknown = vocab["<unk>"]
    return torch.tensor([vocab.get(char, unknown) for char in text], dtype=torch.long)


def complete_windows(ids: torch.Tensor, seq_len: int) -> tuple[tuple[torch.Tensor, ...], int]:
    if seq_len < 1:
        raise ValueError("seq_len must be positive")
    if ids.numel() == 0:
        return (), 0
    full_windows = (ids.numel() - 1) // seq_len
    windows = tuple(
        ids[start : start + seq_len + 1]
        for start in range(0, full_windows * seq_len, seq_len)
    )
    dropped = int(ids.numel() - 1 - len(windows) * seq_len)
    return windows, dropped


