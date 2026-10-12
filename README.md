# Staged Binary-Local Diffusion Attention

Code, data, and experiment records for **Staged Binary-Local Diffusion Attention: Effective Few-Step Diffusion Along the Time Axis**.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23292158.svg)](https://doi.org/10.5281/zenodo.23292158)

**Author:** HaoyangMa · [haoyang@haoyangma.com](mailto:haoyang@haoyangma.com)

[English paper (PDF)](paper_en.pdf) · [Manuscript source](BinaryDiffusionAttention/paper_en.md) · [Experiment protocol](BinaryDiffusionAttention/experiment.md) · [Experiment guide](BinaryDiffusionAttention/README.md)

## How it works

SBDA forms contextual representations through learned causal propagation across layers. Layer `l` reads the current position and a position `2**(l-1)` steps to its left. From layer three onward, it also reads the two nearest left neighbors. Each position therefore attends to at most four sources per layer.

Composing the dyadic paths provides complete prefix reachability in logarithmic depth. This is a graph coverage property; learning quality is evaluated separately through next-character cross-entropy. The implementation uses ordinary content-dependent attention over these sources, without token pooling or explicit key/value compression.

## Recorded results

The current manuscript reports paired three-seed accuracy comparisons at a maximum of 32,768 characters, fifteen layers and twenty training passes. Test metrics use the best validation checkpoint.

| Dataset / measurement | SBDA | Dense causal attention |
|---|---:|---:|
| PG-19 subset, train / test cross-entropy | 1.3076 / 1.4330 | 1.4395 / 1.5558 |
| PG-19 subset, test accuracy | 57.68% | 53.76% |
| Python subset, train / test cross-entropy | 1.2262 / 1.4252 | 0.7662 / 0.8394 |
| Matched 20-layer, 1M attention-interaction kernel speed | 476.58× | 1× |
| Matched 20-layer, 1M backbone wall-time speed | 56.01× | 1× |

SBDA improves the tested book results but has substantially worse code accuracy. These fixed-budget runs do not establish convergence. The separate forward benchmark includes projections, FFNs and normalization, and excludes the vocabulary head, backward pass and optimizer. Peak allocated memory is nearly unchanged.

Original records and protocols:

- Book accuracy: [protocol](BinaryDiffusionAttention/PG19_MINI_EXPERIMENT.md) and [three-seed results](BinaryDiffusionAttention/results/pg19_mini_32k_v1/SUMMARY.md).
- Python accuracy: [protocol](BinaryDiffusionAttention/STACK_SMOL_EXPERIMENT.md) and [six saved runs](BinaryDiffusionAttention/results/stack_smol_python_32k_v1/).
- Matched million-position forward benchmark: [records and scope](BinaryDiffusionAttention/results/gemm_l20_1m/README.md).
- Kernel cost breakdown: [analysis](BinaryDiffusionAttention/GEMM_BREAKDOWN.md) and [JSON/log records](BinaryDiffusionAttention/results/gemm_breakdown_4090_v1/).
- Historical WikiText experiments remain in [results/](BinaryDiffusionAttention/results/); the manuscript distinguishes them from the primary accuracy comparisons.

## Repository layout

```text
BiAttn/                         Model implementation and sparse attention kernels
  model.py                      Shared language-model backbone
  topology.py                   Pure-Python source schedules and topology rules
  left_v2.py                    Exact sparse attention implementation
  attention_common.py           Shared RoPE and head normalization
  sbda_triton.py                Fused CUDA kernels
BinaryDiffusionAttention/       Experiment drivers and manuscript
  datasets.py                   Dataset loading, character encoding and windows
  comparison.py                 Matched dense/SBDA model configurations
  data/                         WikiText-2, PTB and Tiny Shakespeare splits
  results/                      Recorded JSON, CSV and experiment logs
  figures/                      Paper figures and PDF previews
  paper_en.md                   English manuscript source
paper_en.pdf                    Exported English paper
tests/                          Model, causality, topology and data tests
IMPORT_MANIFEST.json             File inventory and checksums
```

See the [model guide](BiAttn/README.md) for implementation details and the [experiment guide](BinaryDiffusionAttention/README.md) for route names and additional datasets.

## Environment

The locked environment uses **Python 3.14** and **PyTorch 2.14.0**, with CUDA 13.0 wheels. Install it with `uv` from the repository root:

```bash
uv sync --locked
```

CPU execution supports small training checks. The paper's CUDA training path uses BF16, and fused BF16 kernel tests require an Ampere-or-newer GPU. The forward benchmark requires CUDA. Structural graph scans use only the Python standard library and can run without installing PyTorch.

All commands below run from the repository root. Dataset paths default to the experiment directory; explicit relative output paths are resolved from the current working directory.

## Quick start

Scan graph coverage without training:

```bash
python3 BinaryDiffusionAttention/run_graph_coverage.py \
  --lengths 256 --layers 8 --routes dense,H0,H8
```

Run a small CPU comparison between dense attention and the staged route:

```bash
uv run python BinaryDiffusionAttention/run_mini_experiment.py \
  --device cpu --steps 2 --seq-len 32 --batch-size 1 \
  --dim 16 --layers 3 --heads 2 --ff-hidden 32 \
  --modes full,h8 --output-dir /tmp/sbda-smoke
```

This checks execution on a small model; it does not reproduce the paper's reported training results. Dataset files are included, and missing files are downloaded by the loader.

Run the regression tests:

```bash
uv run python -m unittest discover -s tests
```

The recorded verification ran 32 tests: 31 passed and one fused BF16 GPU test was skipped. A 108-row topology scan also matched the original scanner exactly. The paper's large CUDA training runs have not been repeated during repository cleanup.

## Reproduce the paper experiments

[Appendix B](BinaryDiffusionAttention/paper_en.md#appendix-b-reproduction-commands) contains the recorded reproduction commands. Prepare and run the primary book comparison with `pg19_mini_compare.py --prepare` and `run_pg19_mini_grid.py`. For the Python subset, use `stack_smol_compare.py --prepare /path/to/data/python/data.json`, then its `--smoke` check and training queue. These new dataset subsets must be prepared before training. See the linked protocols for selection and provenance.

For the historical 65K comparison, run each mode with the same settings and save new outputs separately:

```bash
for mode in causal sbda; do
  uv run python BinaryDiffusionAttention/train_wikitext_1m_pass_compare.py \
    --mode "$mode" --device cuda --seq-len 65536 \
    --layers 16 --dim 64 --heads 4 --ff-hidden 128 \
    --passes 20 --seed 1 --lr 0.0003 \
    --output "BinaryDiffusionAttention/results/reproduced_65k_${mode}.json"
done
```

Run the separate forward throughput grid:

```bash
uv run python BinaryDiffusionAttention/bench_sbda_vs_causal_forward.py \
  --grid --device cuda --dtype bf16 \
  --output BinaryDiffusionAttention/results/reproduced_forward.json
```

The long-context driver's filename retains `1m` for compatibility, but sequence length, depth and pass count are configurable. It also accepts `--device cpu` for small FP32 checks. Refer to the paper for hardware, model settings, token budgets and interpretation of each experiment.

## Rebuild figures and the English PDF

The export tools install their extra dependencies on demand:

```bash
uv run --with matplotlib python BinaryDiffusionAttention/plot_paper_figures.py
uv run --with pypandoc-binary --with typst --with pymupdf \
  python BinaryDiffusionAttention/export_paper_pdf.py
```

Figure generation reads the saved results and updates `figures/`. PDF export reads `paper_en.md`, writes `paper_en.pdf` in the repository root, and refreshes selected page previews. The PDF includes the author name and email shown above.

## Citation and archival

[CITATION.cff](CITATION.cff) supplies GitHub citation metadata; [.zenodo.json](.zenodo.json) supplies the author, title and license for Zenodo archiving. The current release is `v0.1.3`; see its [release notes](docs/releases/v0.1.3.md). The DOI below identifies the archived `v0.1.1` release; the new version DOI will be assigned by Zenodo after publication. The first archival release was `v0.1.0`. The `v0.1.1` release is archived on [Zenodo](https://zenodo.org/records/23292158).

- Version DOI (use to cite these exact artifacts): [10.5281/zenodo.23292158](https://doi.org/10.5281/zenodo.23292158).
- All-versions DOI: [10.5281/zenodo.23292157](https://doi.org/10.5281/zenodo.23292157).

Suggested citation: HaoyangMa. (2026). *Staged Binary-Local Diffusion Attention: Effective Few-Step Diffusion Along the Time Axis* (Version 0.1.1) [Software]. Zenodo. https://doi.org/10.5281/zenodo.23292158

This DOI identifies the software release, which includes the English manuscript; it is not a separate publication DOI for the paper.

## Artifacts and license

Paper data, result tables and the recorded experiment logs are tracked in Git. Python caches, virtual environments, build artifacts, local editor files and temporary logs outside the experiment results directory are ignored. [IMPORT_MANIFEST.json](IMPORT_MANIFEST.json) records file sizes and SHA-256 checksums, including original checksums for modified imported files.

Repository code is provided under the [MIT License](LICENSE). Included datasets retain their respective upstream terms.
