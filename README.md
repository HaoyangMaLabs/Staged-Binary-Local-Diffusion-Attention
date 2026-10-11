# Staged Binary-Local Diffusion Attention

Code, data, and experiment records for **Staged Binary-Local Diffusion Attention: Effective Few-Step Diffusion Along the Time Axis**.

**Author:** HaoyangMa · [haoyang@haoyangma.com](mailto:haoyang@haoyangma.com)

[English paper (PDF)](paper_en.pdf) · [Manuscript source](BinaryDiffusionAttention/paper_en.md) · [Experiment protocol](BinaryDiffusionAttention/experiment.md) · [Experiment guide](BinaryDiffusionAttention/README.md)

## How it works

SBDA forms contextual representations through learned causal propagation across layers. Layer `l` reads the current position and a position `2**(l-1)` steps to its left. From layer three onward, it also reads the two nearest left neighbors. Each position therefore attends to at most four sources per layer.

Composing the dyadic paths provides complete prefix reachability in logarithmic depth. This is a graph coverage property; learning quality is evaluated separately through next-character cross-entropy. The implementation uses ordinary content-dependent attention over these sources, without token pooling or explicit key/value compression.

## Recorded results

The paper reports parameter-matched comparisons on WikiText-2:

| Measurement | SBDA | Dense causal attention |
|---|---:|---:|
| Final-pass training cross-entropy, 65,536 positions | 1.3256 | 2.3188 |
| Held-out validation cross-entropy, 65,536 positions | 1.3739 | 2.2956 |
| Training speed at 65K, relative to dense | 5.47× | 1× |
| Hidden-state forward speed at 1M, relative to dense | 55.57× | 1× |

The 65K comparison uses sixteen layers and twenty training passes. The 1M throughput result comes from a separate shallow forward benchmark and excludes vocabulary projection and training. Allocated memory is nearly unchanged in the reported comparisons. These results describe the tested configurations; structural reachability alone does not establish model quality.

Original records are available in [results/](BinaryDiffusionAttention/results/), including:

- 65K training and validation: [SBDA JSON](BinaryDiffusionAttention/results/wikitext_65536_l16_sbda_20pass_dev.json), [SBDA log](BinaryDiffusionAttention/results/wikitext_65536_l16_sbda_20pass_dev.log), [dense JSON](BinaryDiffusionAttention/results/wikitext_65536_l16_causal_20pass_dev.json), [dense log](BinaryDiffusionAttention/results/wikitext_65536_l16_causal_20pass_dev.log).
- Forward benchmark: [execution log](BinaryDiffusionAttention/results/sbda_vs_causal_4090_forward.log).
- Earlier 1M pilot: [SBDA log](BinaryDiffusionAttention/results/wikitext_1m_sbda_3pass.log), [dense log](BinaryDiffusionAttention/results/wikitext_1m_causal_3pass.log). This pilot predates held-out validation in the current driver; its saved records are the authoritative account of that run.

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

[Appendix B](BinaryDiffusionAttention/paper_en.md#appendix-b-reproduction-commands) contains the recorded reproduction commands. For the 65K comparison, run each mode with the same settings and save new outputs separately:

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

[CITATION.cff](CITATION.cff) supplies GitHub citation metadata; [.zenodo.json](.zenodo.json) supplies the author, title and license for Zenodo archiving. The current release is `v0.1.1`; see its [release notes](docs/releases/v0.1.1.md). The first archival release was `v0.1.0`. The linked Zenodo integration assigns a DOI after it processes the published GitHub release. No DOI is asserted before that record exists.

## Artifacts and license

Paper data, result tables and the five recorded experiment logs are tracked in Git. Python caches, virtual environments, build artifacts, local editor files and temporary logs outside the experiment results directory are ignored. [IMPORT_MANIFEST.json](IMPORT_MANIFEST.json) records file sizes and SHA-256 checksums, including original checksums for modified imported files.

Repository code is provided under the [MIT License](LICENSE). Included datasets retain their respective upstream terms.
