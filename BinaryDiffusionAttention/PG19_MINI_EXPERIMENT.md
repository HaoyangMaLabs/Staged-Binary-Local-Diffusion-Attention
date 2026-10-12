# PG-19-mini accuracy comparison

This is a custom, fixed subset of [official PG-19](https://github.com/google-deepmind/pg19), not an official full-dataset benchmark. Select 30 training books, 5 validation books, and 5 test books by SHA256 ranking of `pg19-mini-v1:` plus object key within each official split. No selection based on model results. IDs and original byte hashes are saved in `data/pg19_mini_v1/manifest.json`.

| Split | Books | Unicode characters | Shortest book | Longest book |
|---|---:|---:|---:|---:|
| Train | 30 | 10,877,012 | 1,974 | 787,741 |
| Validation | 5 | 1,105,235 | 44,891 | 502,104 |
| Test | 5 | 1,957,650 | 148,769 | 844,569 |

Keep book boundaries. Maximum input length 32,768 **characters**, with 32,769 characters used to form shifted inputs/targets. Consecutive windows share only their boundary character, so each next-character transition within a book is scored once per pass. Retain short books and partial tail windows. Batch one requires no padding. No attention crosses book boundaries or window boundaries. Train-only character vocabulary plus UNK; report held-out unknown fractions explicitly. Perplexity is character-level, not the official word-level PG-19 score.

Both models: width 64, 15 layers, 4 heads, SwiGLU hidden 128, tied embedding/head, no dropout, BF16, AdamW learning rate 0.0003, weight decay 0.01, clip norm 1. Dense causal FlashAttention versus fused Triton Staged Binary-Local Diffusion Attention. Fifteen sparse layers provide structural reachability across 32K positions. All non-attention architecture choices are matched. Conceptually corresponding parameters are initialized identically by canonical name and paired seed; identical per-pass sample permutations.

Schedule: 20 passes each, seeds 1/2/3. Character-weighted online train CE and accuracy each pass; separate validation after every pass. Save best validation checkpoint and evaluate test **only at that checkpoint**; also evaluate train at that checkpoint. Rolling model+optimizer checkpoint every 500 steps. Three-seed means and dispersion are required before an overall comparison; first-run loss is not a final result. If either model is still learning substantially at pass 20, extend both with the same protocol before claiming convergence.

Implementation: `pg19_mini_compare.py`; sequential single-GPU queue: `run_pg19_mini_grid.py`. Before training, both paths must pass future-token prefix-invariance and a finite forward/backward smoke test. The server's actual GPU/driver is saved in each result. No torch.compile, no teacher, no cross-book concatenation.

```bash
python BinaryDiffusionAttention/pg19_mini_compare.py --prepare
python BinaryDiffusionAttention/run_pg19_mini_grid.py
```
