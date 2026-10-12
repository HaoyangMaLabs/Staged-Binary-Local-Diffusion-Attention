"""Repository-disjoint Python subset of The Stack Smol; reuse matched trainer."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "BinaryDiffusionAttention"))


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prepare(source, destination):
    # The upstream file is JSON lines; handle JSON arrays too.
    with source.open(encoding="utf-8") as stream:
        first = stream.read(1)
        stream.seek(0)
        rows = json.load(stream) if first == "[" else [json.loads(line) for line in stream if line.strip()]
    groups, seen = defaultdict(list), set()
    dropped = defaultdict(int)
    for row in sorted(rows, key=lambda r: (str(r.get("repository_name", "")), str(r.get("path", "")))):
        repo = row.get("repository_name")
        text = row.get("content", "").replace("\r\n", "\n").replace("\r", "\n")
        if not repo or len(text) < 2:
            dropped["missing_repository_or_too_short"] += 1
            continue
        sha = digest(text)
        if sha in seen:
            dropped["duplicate_content"] += 1
            continue
        seen.add(sha)
        bucket = int(digest("stack-smol-split-v1:" + repo)[:8], 16) % 10
        split = "train" if bucket < 8 else "validation" if bucket == 8 else "test"
        groups[(split, repo)].append({"text": text, "sha256": sha, "repository": repo,
                                     "upstream_path": row.get("path"), "licenses": row.get("licenses")})
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {"source": "bigcode/the-stack-smol/data/python/data.json",
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "split": "repository hash modulo 10: train 0..7, validation 8, test 9; content dedup before split",
                "selection": "hash-ranked whole repositories up to target character budgets; no content-based selection",
                "dropped": dict(dropped), "upstream_rows": len(rows), "splits": {}, "statistics": {}}
    for split, budget in [("train", 10877012), ("validation", 1105235), ("test", 1957650)]:
        repos = sorted([repo for s, repo in groups if s == split], key=lambda r: digest("stack-smol-order-v1:" + r))
        selected, total, used_repos = [], 0, []
        for repo in repos:
            used_repos.append(repo)
            for record in groups[(split, repo)]:
                key = f"{split}/{record['sha256']}.txt"
                (destination / split).mkdir(exist_ok=True)
                (destination / key).write_text(record["text"], encoding="utf-8")
                selected.append({k: v for k, v in record.items() if k != "text"} |
                                {"key": key, "characters": len(record["text"])})
                total += len(record["text"])
            if total >= budget:
                break
        lengths = sorted(r["characters"] for r in selected)
        assert lengths, f"Empty {split}"
        manifest["splits"][split] = selected
        manifest["statistics"][split] = {"files": len(lengths), "repositories": len(used_repos),
            "characters": total, "target_budget": budget, "min": lengths[0], "median": statistics.median(lengths),
            "p90": lengths[int((len(lengths)-1)*.9)], "max": lengths[-1],
            "files_over_32k": sum(n > 32768 for n in lengths),
            "windows_32k": sum((n-2)//32768+1 for n in lengths)}
    repo_sets = [{r["repository"] for r in records} for records in manifest["splits"].values()]
    assert all(not a & b for i, a in enumerate(repo_sets) for b in repo_sets[i+1:])
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(json.dumps(manifest["statistics"], indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prepare", type=Path)
    p.add_argument("--data", type=Path, default=ROOT / "BinaryDiffusionAttention/data/stack_smol_python_v1")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.prepare:
        prepare(args.prepare, args.data)
    else:
        output = ROOT / "BinaryDiffusionAttention/results/stack_smol_python_32k_v1"
        output.mkdir(parents=True, exist_ok=True)
        for seed in ([1] if args.smoke else [1, 2, 3]):
            for mode in ["causal", "sbda"]:
                name = f"{mode}_seed{seed}" + ("_smoke" if args.smoke else "")
                cmd = [sys.executable, str(ROOT / "BinaryDiffusionAttention/pg19_mini_compare.py"),
                       "--data", str(args.data), "--mode", mode, "--seed", str(seed),
                       "--passes", "20", "--pad-buckets", "--output", str(output / name)]
                if args.smoke:
                    cmd += ["--smoke-steps", "1"]
                print("starting", name, flush=True)
                with (output / f"{name}.log").open("w") as log:
                    subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                print("finished", name, flush=True)
