"""Run profiler cases in isolated processes; preserve each completed result."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default=str(HERE / "results" / "gemm_breakdown_4090_v1"))
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cases = [(kind, mode, n) for kind, ns in [("forward", [4096, 16384, 65536, 1048576]),
                                             ("train", [4096, 16384, 65536])]
             for n in ns for mode in ["causal", "sbda"]]
    records = []
    for kind, mode, n in cases:
        name = f"{kind}_{mode}_{n}"
        target = out / f"{name}.json"
        cmd = [sys.executable, str(HERE / "bench_gemm_breakdown.py"), "--kind", kind,
               "--mode", mode, "--length", str(n), "--output", str(target)]
        print("START", name, flush=True)
        with (out / f"{name}.log").open("w") as log:
            try:
                done = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=300)
                result = json.loads(target.read_text()) if target.exists() else {
                    "status": "failed", "kind": kind, "mode": mode, "length": n, "returncode": done.returncode}
            except subprocess.TimeoutExpired:
                result = {"status": "timeout", "kind": kind, "mode": mode, "length": n}
        records.append(result)
        (out / "summary.json").write_text(json.dumps({
            "created_utc": datetime.now(timezone.utc).isoformat(), "results": records}, indent=2)+"\n")
        print("END", name, result.get("status"),
              "wall_ms", result.get("unprofiled_wall_ms_median"),
              "gemm_pct", result.get("fixed_gemm_pct_gpu_kernel_time"), flush=True)
