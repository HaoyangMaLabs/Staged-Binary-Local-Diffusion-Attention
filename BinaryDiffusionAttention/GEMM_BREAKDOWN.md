# Fixed GEMM and attention cost audit

Measured on 2026-10-11. This is a computation-only audit of the manuscript's
configurations, not a new WikiText quality experiment. Random, device-resident
token IDs are used; model equations and backends are unchanged.

## Main finding

The interaction-count reduction is not a whole-model speedup expectation.
Fixed QKV/output/FFN GEMMs remain, but **in these narrow models they are not
the largest remaining cost**. Elementwise operations, normalization, RoPE,
casting/copies, and training output/loss work also remain.

At 1M positions, dense hidden-state forward uses 16,658.909 ms of profiled GPU
kernel execution per forward:

| GPU component | Dense ms | Staged binary-local ms |
|---|---:|---:|
| QKV projection GEMMs | 10.615 | 10.571 |
| Attention output GEMMs | 3.236 | 3.210 |
| FFN GEMMs | 19.759 | 19.705 |
| Attention interaction kernels | 16,395.031 | 31.875 |
| Other GPU kernels | 230.268 | 229.161 |
| Total GPU kernel execution | 16,658.909 | 294.521 |

Dense GEMMs total 33.610 ms, **0.202%** of GPU kernel time. All non-interaction
kernels total 263.878 ms, **1.584%**. If the interaction kernels were free and
all other kernel durations stayed unchanged, the GPU-kernel-only ratio would
be **63.13×**. Measured unprofiled wall-time speedup is **56.31×**.

These are different timing domains: 63.13× is a kernel-time decomposition
thought experiment, not a strict end-to-end wall-time ceiling or a hardware
utilization score. Nonetheless, the nearly identical non-interaction kernel
durations across paths show why a connection-count ratio around 140,000× is
not a realistic total-runtime target. The interaction kernels themselves
accelerate about 514× in this configuration.

The previous claim that kernel launch overhead consumes most of the missing
theoretical gain is not supported for the 1M case. First account for all work
that graph sparsification does not remove. Launch/host overhead remains
important at short context and in these small training models, but requires
a separate timeline analysis for precise attribution.

## Hardware and protocol

- GPU reports NVIDIA GeForce RTX 4090, **24,564 MiB** (about 24 GiB), driver
  610.43.03, power limit 450 W. This is **not** the earlier 49,140 MiB server
  configuration; do not silently replace the old experiment's hardware record.
- CPU reports AMD EPYC 7502; online logical CPUs 0–13.
- Existing environment: `/root/ndsm/.venv-cu130/bin/python`; PyTorch
  2.14.0+cu130, CUDA 13.0. No replacement environment or Torch download.
- BF16 autocast, batch one, dropout zero, no `torch.compile`.
- Forward configuration: width 96, eight layers, four heads, FFN 256,
  vocabulary 6,400; `forward_hidden` excludes vocabulary output.
- Training configuration: width 64, sixteen layers, four heads, FFN 128,
  vocabulary 1,014; full-vocabulary CE, backward, gradient clipping and AdamW.
  Forward/backward GEMMs are both included in the training categories.
- Two warmup iterations, three unprofiled synchronized iterations (median
  reported), and two separate profiler iterations. Compilation is warmed up.
- Every configuration runs in a fresh child process. Fourteen cases completed
  successfully. No checkpoints or datasets are needed for this audit.
- Dense profiler dispatch is `aten::_flash_attention_forward` and, in training,
  its backward; sparse dispatch is the fused SBDA Triton autograd path.
- Summed **exclusive CPU-operator device times** avoid double-counting parent
  regions and duplicate CUDA kernel records. GEMM categories contain only
  `mm`, `addmm`, `bmm`, and `baddbmm`, not casting or nonlinear work. Attention
  QK/AV multiplication inside fused attention remains in the interaction category.
- Autograd sequence numbers map backward operators to their forward projection
  scopes. All cases have zero unmapped backward GEMM time.
- Optimizer GPU time includes clipping and optimizer kernels. Other GPU time
  includes CE, casts/copies, embedding, normalization, RoPE, activations and
  residual operations. This category is not itself a kernel-launch measurement.
- GPU-kernel shares below do **not** measure the fraction of wall time. Profiled
  kernel totals and unprofiled wall times are separate runs; their difference
  must not be treated as an exact launch-overhead measurement.

## GPU-kernel time shares

| Mode | Context | Path | Fixed GEMM | Interaction | Other GPU | Optimizer GPU |
|---|---:|---|---:|---:|---:|---:|
| Forward | 4,096 | Dense | 11.11% | 33.75% | 55.14% | — |
| Forward | 4,096 | Staged | 14.04% | 16.14% | 69.82% | — |
| Forward | 16,384 | Dense | 5.23% | 70.53% | 24.25% | — |
| Forward | 16,384 | Staged | 14.68% | 18.29% | 67.03% | — |
| Forward | 65,536 | Dense | 1.76% | 86.36% | 11.88% | — |
| Forward | 65,536 | Staged | 11.13% | 17.99% | 70.88% | — |
| Forward | 1,048,576 | Dense | 0.20% | 98.42% | 1.38% | — |
| Forward | 1,048,576 | Staged | 11.37% | 10.82% | 77.81% | — |
| Training | 4,096 | Dense | 11.82% | 38.19% | 48.86% | 1.14% |
| Training | 4,096 | Staged | 18.17% | 4.34% | 75.72% | 1.77% |
| Training | 16,384 | Dense | 4.54% | 72.34% | 22.85% | 0.27% |
| Training | 16,384 | Staged | 15.55% | 4.87% | 78.64% | 0.94% |
| Training | 65,536 | Dense | 1.26% | 91.35% | 7.37% | 0.02% |
| Training | 65,536 | Staged | 13.85% | 5.43% | 80.44% | 0.27% |

Fixed GEMM means fixed *per-token computation paths*, not a fixed percentage.
The dense interaction cost grows quadratically; the fixed GEMM cost grows
linearly at fixed depth/width. Ratios change with context and model shape.
Do not generalize these percentages to wide production LLMs.

## Unprofiled synchronized wall time

| Mode | Context | Dense ms | Staged ms | Measured speedup | Kernel-only zero-interaction ratio¹ |
|---|---:|---:|---:|---:|---:|
| Forward | 4,096 | 12.237 | 10.645 | 1.15× | 1.51× |
| Forward | 16,384 | 11.252 | 10.539 | 1.07× | 3.39× |
| Forward | 65,536 | 74.965 | 12.874 | 5.82× | 7.33× |
| Forward | 1,048,576 | 16,593.358 | 294.666 | 56.31× | 63.13× |
| Training | 4,096 | 63.324 | 80.655 | 0.79× | 1.62× |
| Training | 16,384 | 62.643 | 65.199 | 0.96× | 3.61× |
| Training | 65,536 | 470.386 | 65.106 | 7.22× | 11.56× |

¹ Dense total kernel time divided by dense non-interaction kernel time.
It assumes only interaction kernels vanish, not norms, FFNs, CE or optimizer
work; it excludes host gaps and is not a wall-time guarantee. The new training
speedup is a warmed-up microbenchmark, not a revision of the historical 20-pass
5.47× result (which includes different timing conditions and first compilation).

The eight-layer forward model still does not provide 1M structural coverage.
The sixteen-layer 65K training model does. This audit does not evaluate either
model's semantic capability or compare newly trained losses.

## Training at 65K: fixed GEMM composition

| GEMM category, forward + backward | Dense ms | Staged ms |
|---|---:|---:|
| QKV | 2.278 | 2.302 |
| Attention output | 0.643 | 0.647 |
| FFN | 2.512 | 2.485 |
| Vocabulary output | 0.448 | 0.451 |
| Total | 5.881 | 5.885 |

GPU interaction kernels take 427.637 ms in dense training and 2.308 ms in
staged training. Other kernels take 34.490 / 34.179 ms respectively. This
establishes both the retained GEMM floor and the larger non-GEMM retained cost.

## Analytical forward arithmetic (not timing)

For batch one, equal Q/K/V width d, SwiGLU width f, L layers, vocabulary V:

```text
Fixed GEMM FLOPs = 2 N [L (4 d² + 3 d f) + d V if output head included]
Dense interaction FLOPs = 2 L d N (N + 1)
Sparse interaction FLOPs = 4 d × total valid sparse edges
```

These count multiply-add as two FLOPs and omit softmax, activations,
normalization, copies, causal-tile overcompute, and backward/optimizer work.
They therefore must not be substituted for measured GPU-time fractions.
The formula/results are retained in each JSON under `forward_flop_model`.

## Artifacts and reproduction

- `bench_gemm_breakdown.py`: single-case measurement and attribution.
- `run_gemm_breakdown_grid.py`: isolated case grid, 300-second case timeout,
  incremental summary and logs.
- `results/gemm_breakdown_4090_v1/summary.json`: all fourteen measurements.
- Same directory: individual JSONs, per-case logs and `grid.log`.
- No continuing GPU workload remains after this test.

```bash
python BinaryDiffusionAttention/run_gemm_breakdown_grid.py
python BinaryDiffusionAttention/bench_gemm_breakdown.py \
  --kind train --mode causal --length 65536 \
  --output BinaryDiffusionAttention/results/new_train_causal_65536.json
```

Remote implementation SHA256 values (the measured checkout):

```text
BiAttn/model.py       3e4736dc495b8ad5622b5e500ced74f5b84999daba0da402c94856af3bad4625
BiAttn/left_v2.py     d2c07d5b26c8b03f6fa6b64ac22b7b1bd63a64080395b03dbb1a464c207d704e
BiAttn/sbda_triton.py db43154232993cb7ce154d220eaf9b0c38190608d59eedf06055b5048e09143e
BiAttn/three_track.py 1a7f9c28552cef4c3955e2f402b9117fad577fafe7d41521c2a34a175b65ba4d
```

The remote backbone has additional inactive two-stage attention options
relative to the local checkout. The selected standard-softmax branch and
fused sparse kernel implementation match the tested design; the remote
backbone was preserved, not overwritten during this audit.
