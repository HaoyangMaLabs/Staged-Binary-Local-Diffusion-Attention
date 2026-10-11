# Binary Diffusion Attention

This directory contains the paper, saved experiment records, and reproduction
drivers for staged binary-local diffusion attention implemented in `BiAttn`.

The central hypothesis is:

> A causal sequence can spread information from local positions to the whole
> prefix through a dyadic message-passing schedule. At layer `l`, a token
> communicates with itself and the token `2**(l-1)` positions to its left.

The dyadic scaffold has two exact sources per query and layer. SBDA adds
the two nearest left neighbors from layer three onward, using at most four sources. It does not pool tokens,
compress K/V, or materialize an `N x N` attention matrix. Composing the binary
offsets gives every causal offset below `2**L` a path after `L` layers.

## Directory contents

Shared implementation modules:

- `datasets.py`: cached dataset downloads, character vocabulary, encoding and complete windows.
- `comparison.py`: identical backbone configuration for long-context training and forward benchmarks.

Experiment drivers keep their original filenames and command-line options.
The long-context training driver uses BF16 on CUDA and FP32 for CPU smoke checks;
GPU benchmark measurements remain CUDA-only. Dataset downloads replace cached
files atomically after a complete transfer. Training character ordering and
split boundaries are unchanged.

- [`paper_en.md`](paper_en.md) and [`paper_en.pdf`](../paper_en.pdf): English manuscript and exported PDF.
- [`experiment.md`](experiment.md): experiment protocol and interpretation rules.
- [`NOVELTY_AUDIT.md`](NOVELTY_AUDIT.md): conservative prior-art comparison and
  safe novelty claims.
- [`run_mini_experiment.py`](run_mini_experiment.py): downloads WikiText-2 raw,
  trains matched small models, and writes JSON/CSV results.
- [`train_wikitext_1m_pass_compare.py`](train_wikitext_1m_pass_compare.py): configurable full-window training and held-out validation for dense/SBDA comparisons.
- [`run_graph_coverage.py`](run_graph_coverage.py): measures layer-wise
  reachability, exact source cost, and structural cell reuse without training.
- [`bench_sbda_vs_causal_forward.py`](bench_sbda_vs_causal_forward.py): runs a
  BF16 batch-1 long-context forward throughput comparison on CUDA.

## Quick start

From the repository root:

```bash
python BinaryDiffusionAttention/run_mini_experiment.py \
  --data-root BinaryDiffusionAttention/data \
  --steps 20 \
  --seq-len 256 \
  --batch-size 4
```

The first run downloads the public WikiText-2 raw split files. Results are
written under `BinaryDiffusionAttention/results/`; downloaded data and generated results are retained in this repository.

Structural scan without training:

```bash
python BinaryDiffusionAttention/run_graph_coverage.py \
  --lengths 256,512,1024,2048 \
  --layers 4,8,16
```

For a useful local comparison, use a small model first:

```bash
python BinaryDiffusionAttention/run_mini_experiment.py \
  --data-root BinaryDiffusionAttention/data \
  --steps 500 \
  --seq-len 256 \
  --batch-size 8 \
  --dim 96 --layers 8 --heads 4 --ff-hidden 256 \
  --seeds 1,2,3
```

The script compares:

| Name | Existing implementation | Purpose |
|---|---|---|
| `full` | dense causal attention | matched dense reference and quadratic cost |
| `left_local` | aligned dyadic local attention | local-only sparse reference |
| `binary_diffusion` | `left_v2`, binary | proposed two-source diffusion route |
| `local_jump` | `left_v2`, local + dyadic jump | local shortcut ablation |
| `butterfly` | `left_v2`, four dyadic lanes | wider multi-lane propagation |
| `radix4` | `left_v2`, radix-4 | fewer, wider propagation stages |
| `high4` | `left_v2`, local + four jumps | higher edge-budget sparse reference |
| `longformer_local` | fixed four-cell causal window | Longformer local-window analogue |
| `sparse_local_stride` | fixed local window + fixed stride | Sparse Transformer analogue |
| `dinat_like` | local window + dilated taps | DiNAT-style 1-D analogue |
| `longnet_like` | local window + geometric dilated taps | LongNet-inspired causal adaptation |
| `power_attention` | local core + all power-of-two taps | PowerAttention token-level analogue |

The current script is intentionally a mini benchmark, not a claim of state of
the art. A positive result requires lower validation NLL at matched parameter
count and token budget, plus a useful tokens/s and memory advantage.

The same comparison can be run on the two additional small public corpora used
for cross-dataset checking:

```bash
python BinaryDiffusionAttention/run_mini_experiment.py \
  --dataset tinyshakespeare --modes full,h0,h1,h8
python BinaryDiffusionAttention/run_mini_experiment.py \
  --dataset ptb --modes full,h0,h1,h8
```

The loader uses the public Tiny Shakespeare raw text and the public Penn
Treebank train/valid/test text splits, with a character-level vocabulary.
