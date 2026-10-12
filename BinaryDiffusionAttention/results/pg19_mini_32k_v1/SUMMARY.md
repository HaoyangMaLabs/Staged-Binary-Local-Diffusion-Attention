# PG-19-mini: 20-pass paired-seed accuracy results

All six runs completed. Settings and fixed book selection: `../../PG19_MINI_EXPERIMENT.md`. Train/validation/test: 30/5/5 official-split books; maximum 32,768 Unicode-character inputs, no cross-book concatenation; 15 layers, width 64, FFN 128, 628,832 parameters each. Three paired seeds; each run selects its best validation checkpoint. Values below are arithmetic means across seeds; CE dispersion is sample standard deviation. Train is evaluated at the selected checkpoint, not the online training-pass mean. All metrics are weighted by valid character transitions.

| Model | Train CE | Test CE | Train accuracy | Test accuracy | Test character PPL |
|---|---:|---:|---:|---:|---:|
| Dense causal | 1.4395 ± 0.0314 | 1.5558 ± 0.0234 | 56.08% | 53.76% | 4.740 |
| Staged Binary-Local Diffusion Attention | 1.3076 ± 0.0038 | 1.4330 ± 0.0043 | 60.11% | 57.68% | 4.191 |

| Seed | Dense Test CE | Sparse Test CE | Dense best pass | Sparse best pass |
|---|---:|---:|---:|---:|
| 1 | 1.58266 | 1.43000 | 20 | 20 |
| 2 | 1.54015 | 1.43112 | 20 | 20 |
| 3 | 1.54464 | 1.43795 | 20 | 19 |

Under the same 20-pass data budget, sparse wins on all three paired seeds: mean Test CE lower by 0.12279; Test accuracy higher by 3.923 percentage points. This is evidence for the tested small-model, fixed-subset setting, not universal superiority or matched converged performance. Dense selects pass 20 in all seeds and continues improving; convergence has not been established. No additional passes launched automatically.

Actual server: NVIDIA GeForce RTX 4090, 24,564 MiB, driver 610.43.03. Logs, vocabularies and result JSON files downloaded locally; checkpoints remain on the server. Train-only vocabulary has 187 entries including UNK. Validation unknown fraction 0.00046098%; test 0.00067641%. Raw download character counts include CRLF; Python text reads normalize newlines, yielding 10,663,144 training targets per pass and 1,921,910 test targets. Tail windows retained, no padding used at batch one. Perplexity is character-level, not official PG-19 word perplexity.
