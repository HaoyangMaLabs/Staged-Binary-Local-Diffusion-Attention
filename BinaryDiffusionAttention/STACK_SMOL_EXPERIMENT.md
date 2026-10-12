# The Stack Smol Python accuracy comparison

Source: [bigcode/the-stack-smol](https://huggingface.co/datasets/bigcode/the-stack-smol), `data/python/data.json`, revision `4a6938ce94446f324c6629e7de00ac591710044b`. Downloaded with the user's authorized HF account. Original licenses/provenance retained in the manifest. This is a fixed custom Python subset, not the full 30-language dataset or an official benchmark split.

Normalize CRLF/CR to LF, globally remove exact duplicate content, discard missing-repository records and files shorter than two characters. Hash repository identities into 80/10/10 train/validation/test partitions, then select hash-ranked whole repositories until approximately the PG-19-mini character budgets are reached. No repository or exact normalized file overlaps splits. Near-duplicate forks are not guaranteed absent. Do not infer full project context: sampled files remain independent.

| Split | Files | Repositories | Characters | Median file length | Maximum | Files >32K |
|---|---:|---:|---:|---:|---:|---:|
| Train | 1,261 | 1,165 | 10,888,182 | 2,769 | 460,275 | 56 |
| Validation | 138 | 134 | 1,110,223 | 2,880.5 | 149,012 | 5 |
| Test | 248 | 234 | 1,963,664 | 2,663.5 | 171,356 | 13 |

Unlike PG-19 books, most files are short. Report this as a code-domain accuracy experiment, not proof that every sample requires long context.

Use the same paired trainer/configuration as PG-19: character-level next-token prediction, width 64, 15 layers, 4 heads, SwiGLU hidden 128, tied embedding/head, BF16, AdamW 3e-4, weight decay .01, gradient clipping 1, no dropout. Train-only vocabulary plus UNK, with held-out unknown rates recorded. Dense causal FlashAttention versus fused Triton Staged Binary-Local Diffusion Attention. Maximum 32,768 input characters, retain all tails, no cross-file concatenation. Batch one.

To bound Triton specialization for thousands of different short file lengths, right-pad each input to a power-of-two bucket (minimum 128, maximum 32K). Future right-padding cannot influence valid causal positions. Padded targets are -100 and excluded from CE and accuracy; both paths use identical buckets. Only real target counts weight reported metrics. No torch.compile.

Schedule: both architectures, 20 passes, seeds 1/2/3 sequentially. Check future-token prefix invariance and finite forward/backward smoke first. Best validation checkpoint selects final test and train evaluation. Record online train CE/accuracy each pass; save model+optimizer rolling checkpoint every 500 steps. Do not claim convergence from a fixed 20-pass budget.

Files: `stack_smol_compare.py` prepares the data and orchestrates six runs, reusing `pg19_mini_compare.py`. Manifest at `data/stack_smol_python_v1/manifest.json`; outputs at `results/stack_smol_python_32k_v1/`.

```bash
python BinaryDiffusionAttention/stack_smol_compare.py --prepare /path/to/data/python/data.json
python BinaryDiffusionAttention/stack_smol_compare.py --smoke
python BinaryDiffusionAttention/stack_smol_compare.py
```
