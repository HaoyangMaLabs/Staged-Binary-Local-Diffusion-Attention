"""Book-disjoint PG19-mini; no cross-book windows, character-level next-token CE."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASE = "https://storage.googleapis.com/deepmind-gutenberg"


def get(url):
    for attempt in range(3):
        try:
            return urllib.request.urlopen(url, timeout=45).read()
        except Exception:
            if attempt == 2:
                raise


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    manifest = {}
    ns = {"s": "http://doc.s3.amazonaws.com/2006-03-01"}
    for split, count in [("train", 30), ("validation", 5), ("test", 5)]:
        keys, marker = [], ""
        while True:
            params = urllib.parse.urlencode({"prefix": split + "/", "max-keys": 1000, "marker": marker})
            tree = ET.fromstring(get(BASE + "?" + params))
            keys.extend(e.text for e in tree.findall("s:Contents/s:Key", ns) if e.text.endswith(".txt"))
            if tree.findtext("s:IsTruncated", namespaces=ns) != "true":
                break
            marker = tree.findtext("s:NextMarker", namespaces=ns) or keys[-1]
        selected = sorted(keys, key=lambda k: hashlib.sha256(("pg19-mini-v1:" + k).encode()).hexdigest())[:count]
        def fetch(key):
            path = root / key
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_bytes(get(BASE + "/" + key))
            raw = path.read_bytes()
            text = raw.decode("utf-8")
            return {"key": key, "sha256": hashlib.sha256(raw).hexdigest(),
                    "bytes": len(raw), "characters": len(text), "words": len(text.split())}
        with ThreadPoolExecutor(max_workers=12) as pool:
            manifest[split] = list(pool.map(fetch, selected))
        print(split, len(keys), "selected", len(selected), flush=True)
    (root / "manifest.json").write_text(json.dumps({"source": BASE, "selection": "sha256-ranked pg19-mini-v1 keys; official splits", "splits": manifest}, indent=2))


def train(args):
    import torch
    from torch.nn import functional as F
    from BiAttn.model import BiAttnConfig, BiAttnLM
    torch.manual_seed(args.seed)
    manifest = json.loads((args.data / "manifest.json").read_text())
    texts = {split: [(args.data / book["key"]).read_text(encoding="utf-8") for book in books]
             for split, books in manifest["splits"].items()}
    alphabet = sorted(set("".join(texts["train"])))
    vocab = {char: i+1 for i, char in enumerate(alphabet)}
    windows = {}
    for split, books in texts.items():
        sequences = []
        for text in books:
            ids = torch.tensor([vocab.get(char, 0) for char in text], dtype=torch.long)
            sequences.extend(ids[start:min(start + args.window + 1, len(ids))]
                             for start in range(0, len(ids)-1, args.window))
        windows[split] = sequences
    cfg = BiAttnConfig(vocab_size=len(vocab)+1, dim=64, layers=15, heads=4, ff_hidden=128,
                      tie_embeddings=True, attention_backend="full" if args.mode == "causal" else "sbda",
                      path_sdpa=True, sbda_fused=True, sbda_backward="key", dropout=0.)
    model = BiAttnLM(cfg).cuda()
    # Same conceptual parameter gets identical initialization independent of backend names.
    for name, param in model.named_parameters():
        canonical = name.replace("left_q", "q").replace("left_k", "k").replace("left_v", "v").replace("left_out", "out")
        generator = torch.Generator(device="cuda")
        generator.manual_seed(int(hashlib.sha256(f"{args.seed}:{canonical}".encode()).hexdigest()[:15], 16))
        with torch.no_grad():
            if param.ndim >= 2:
                param.normal_(0, .02, generator=generator)
            elif name.endswith("bias"):
                param.zero_()
            else:
                param.fill_(1.)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    args.output.mkdir(parents=True, exist_ok=True)
    history, best, step = [], math.inf, 0
    def transfer(seq):
        x, y = seq[:-1].unsqueeze(0).cuda(), seq[1:].cuda()
        if args.pad_buckets:
            # Right padding cannot enter any valid causal prefix. Ignore padded targets.
            bucket = min(args.window, max(128, 1 << (y.numel()-1).bit_length()))
            x = F.pad(x, (0, bucket-y.numel()), value=0)
            y = F.pad(y, (0, bucket-y.numel()), value=-100)
        return x, y
    def evaluate(split):
        model.eval()
        nll, correct, count = 0., 0, 0
        with torch.no_grad():
            for seq in windows[split]:
                x, y = transfer(seq)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(x)[0]
                    loss = F.cross_entropy(logits, y, reduction="sum")
                nll += loss.item()
                correct += (logits.argmax(-1) == y).sum().item()
                count += seq.numel()-1
        return {"ce": nll/count, "character_perplexity": math.exp(nll/count), "accuracy": correct/count, "tokens": count}
    (args.output / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False))
    result = {"mode": args.mode, "seed": args.seed, "unit": "Unicode characters; train-only vocabulary with UNK=0",
              "unknown_fraction": {s: sum(c not in vocab for t in ts for c in t)/sum(map(len, ts)) for s, ts in texts.items()},
              "window": args.window, "config": {"dim": 64, "layers": 15, "ff_hidden": 128, "vocab": len(vocab)+1},
              "padding": "right power-of-two buckets, ignored CE targets" if args.pad_buckets else "none",
              "parameters": sum(p.numel() for p in model.parameters()),
              "hardware": subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], text=True).strip(),
              "windows": {k: len(v) for k, v in windows.items()}, "records": history}
    # Prefix invariance checks both causal paths before committing a long run.
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        probe = torch.arange(64, device="cuda").remainder(len(vocab)+1).unsqueeze(0)
        changed = probe.clone()
        changed[:, 32:] = (changed[:, 32:]+1).remainder(len(vocab)+1)
        a, b = model(probe)[:, :32], model(changed)[:, :32]
        assert torch.equal(a, b), "Future-token leakage in attention"
    print(json.dumps({"causality": "passed", "parameters": result["parameters"], "windows": result["windows"],
                      "unknown_fraction": result["unknown_fraction"]}), flush=True)
    for epoch in range(args.passes):
        model.train()
        order = list(range(len(windows["train"])))
        random.Random(args.seed * 10000 + epoch).shuffle(order)
        nll, correct, count, start = 0., 0, 0, time.perf_counter()
        for index in order:
            seq = windows["train"][index]
            x, y = transfer(seq)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)[0]
                loss = F.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            nll += loss.item() * (seq.numel()-1)
            correct += (logits.detach().argmax(-1) == y).sum().item()
            count += seq.numel()-1
            step += 1
            if step % 10 == 0 or step == 1:
                print(json.dumps({"step": step, "pass": epoch+1, "train_ce": nll/count, "accuracy": correct/count}), flush=True)
            if step % 500 == 0:
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step}, args.output / "rolling.pt")
            if args.smoke_steps and step >= args.smoke_steps:
                print("smoke complete", flush=True)
                return
        record = {"pass": epoch+1, "train_ce": nll/count, "train_accuracy": correct/count,
                  "train_tokens": count, "seconds": time.perf_counter()-start, "dev": evaluate("validation")}
        history.append(record)
        if record["dev"]["ce"] < best:
            best = record["dev"]["ce"]
            result["best_pass"] = epoch+1
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step}, args.output / "best.pt")
        (args.output / "result.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(record), flush=True)
    state = torch.load(args.output / "best.pt", map_location="cuda", weights_only=False)
    model.load_state_dict(state["model"])
    result["test_at_best_dev"] = evaluate("test")
    result["train_at_best_dev"] = evaluate("train")
    result["status"] = "complete"
    (args.output / "result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--data", type=Path, default=ROOT / "BinaryDiffusionAttention/data/pg19_mini_v1")
    p.add_argument("--mode", choices=["causal", "sbda"], default="causal")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--window", type=int, default=32768)
    p.add_argument("--passes", type=int, default=20)
    p.add_argument("--smoke-steps", type=int, default=0)
    p.add_argument("--pad-buckets", action="store_true")
    p.add_argument("--output", type=Path, default=ROOT / "BinaryDiffusionAttention/results/pg19_mini_v1")
    args = p.parse_args()
    prepare(args.data) if args.prepare else train(args)
