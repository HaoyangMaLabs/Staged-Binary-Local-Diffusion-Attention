"""Sequential single-GPU, paired-seed accuracy comparison."""
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
output = root / "BinaryDiffusionAttention/results/pg19_mini_32k_v1"
output.mkdir(parents=True, exist_ok=True)
for seed in [1, 2, 3]:
    for mode in ["causal", "sbda"]:
        run = output / f"{mode}_seed{seed}"
        with (output / f"{mode}_seed{seed}.log").open("w") as log:
            cmd = [sys.executable, str(root / "BinaryDiffusionAttention/pg19_mini_compare.py"),
                   "--mode", mode, "--seed", str(seed), "--passes", "20", "--output", str(run)]
            print("starting", mode, seed, flush=True)
            subprocess.run(cmd, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
            print("finished", mode, seed, flush=True)
