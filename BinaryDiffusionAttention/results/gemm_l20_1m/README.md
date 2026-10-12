# Matched twenty-layer million-position forward comparison

Measured 2026-10-11 on the remote NVIDIA GeForce RTX 4090 reporting 24,564 MiB. BF16, batch 1, length 1,048,576, width 96, twenty layers, four heads, FFN hidden 256, vocabulary 6,400. Dense causal FlashAttention versus fused Triton Staged Binary-Local Diffusion Attention. Random GPU-resident token IDs; no quality measurement. Both paths have sufficient structural depth for this window.

One warmup, three synchronized unprofiled forward timings (median reported), and one separate profiler iteration. Vocabulary output, backward, and optimizer excluded. Attention times are summed GPU interaction-kernel times across all twenty layers, not an independent standalone-attention wall-clock benchmark. They exclude QKV/output projections, FFNs, normalization, and other backbone work.

| Measurement | Dense | Sparse | Speedup |
|---|---:|---:|---:|
| Attention interaction GPU kernels | 40,902.827 ms | 85.826 ms | 476.580x |
| Backbone wall time | 41,491.949 ms | 740.775 ms | 56.012x |
| Peak allocated memory | 3.213815 GiB | 3.213824 GiB | — |

At this length and depth the exact graph-edge ratio is approximately 139,810x. It is not a whole-model FLOP or speed ratio. Raw JSON and run logs are alongside this file. Historical training experiments retain their original layer counts and hardware; these new results do not replace their loss measurements.

Reproduce each mode with the existing CUDA environment:

```bash
python BinaryDiffusionAttention/bench_gemm_breakdown.py --kind forward --mode causal --length 1048576 --layers 20 --warmup 1 --iters 3 --profile-iters 1 --output BinaryDiffusionAttention/results/gemm_l20_1m/causal.json
python BinaryDiffusionAttention/bench_gemm_breakdown.py --kind forward --mode sbda --length 1048576 --layers 20 --warmup 1 --iters 3 --profile-iters 1 --output BinaryDiffusionAttention/results/gemm_l20_1m/sbda.json
```
