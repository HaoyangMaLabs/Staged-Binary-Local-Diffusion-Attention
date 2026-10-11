# What actually controls train loss?

The current WikiText result rejects the naive claim

> faster structural coverage + fewer repeated cells = lower train loss.

The next experiment treats coverage and repetition as explanatory factors and
uses train loss at a fixed seen-token budget as the primary dependent variable.
Validation loss, throughput, and memory remain secondary checks.

## Factors to isolate

1. **Coverage schedule.** How quickly the reachable prefix grows with depth.
   Record `R_l`, coverage AUC, first layer reaching the full context, and mean
   shortest path.
2. **Cross-layer cell repetition.** Fraction of `(query,key)` cells reused by
   later layers. Report both total repeat rate and repeat rate for local cells
   (offsets 1–2), because these have a different semantic role from repeated
   long jumps.
3. **Local anchors.** Number and placement of self/nearby sources. Compare no
   repeated local core, local core at every layer, local core early, and local
   core late.
4. **Path multiplicity.** Number of distinct layer compositions reaching each
   source offset. Coverage alone ignores whether a source has one fragile path
   or several redundant paths.
5. **Source budget.** Average exact sources per query and total gathered cells.
   This is a control variable, not the objective.

## Controlled route groups

Use the same vocabulary, model width/depth, optimizer, batch, context, and
seen-token budget for every route. Run three seeds and save train loss at steps
0/50/100/.../500.

| comparison | routes | question |
|---|---|---|
| dense upper-coverage control | dense vs H0 | Does full one-layer coverage reduce train loss? |
| expansion speed | H0 vs H2 vs H4 | Does faster reachability help when source cost is similar? |
| local repetition | H0 vs H8 | Does adding repeated local support improve optimization? |
| local timing | H1 vs H8 | Is local support better every layer or only after expansion begins? |
| unique expansion | H2 vs H6 | Does the exact long-jump schedule matter after source cost is matched? |
| late expansion | H0 vs H7 | Does delaying long jumps improve early optimization? |
| shell upper bound | H3 vs H4 | Is low repetition worth the shell's extra source cost? |

## Primary readout

For each route, report:

- train loss at every trace step, mean and standard deviation over seeds;
- final train loss and last-10-step mean;
- initial-to-final loss drop and area under the train-loss curve;
- coverage AUC, first-full-coverage layer, repeat rate, local-repeat rate,
  path multiplicity, and source cost.

The decisive test is not a raw correlation over nine routes. It is the paired
comparison within each group above. A factor is considered supported only if it
improves the train-loss curve in both the matched route pair and the three-seed
aggregate without an unreported source-budget increase.

## Current preliminary conclusion

The existing 3-seed screening already gives a useful warning:

- H4 reaches all 2048 structural positions by layer 8 but has worse train loss
  than H0/H1/H8.
- H3 has the lowest repeat rate but is not the best-trained route.
- H8 has high local repetition and modest 8-layer coverage, yet has the lowest
  train loss.

Therefore the leading hypothesis is now:

> Train loss is controlled by the interaction of useful local anchors,
> expansion timing, and path multiplicity; raw coverage speed and raw repeat
> rate are insufficient statistics.

The trace-enabled runner is
`run_mini_experiment.py --trace-every 50`. Results should be kept in a
separate output directory from the earlier screening run.

## First controlled result

The first factorial-style run used WikiText-2, 8 layers, 256 context, width
96, batch 8, 500 steps, and seeds 1/2/3. Mean train loss over seeds:

| route | step 100 | step 200 | step 300 | step 400 | step 500 | train-curve area |
|---|---:|---:|---:|---:|---:|---:|
| dense | 3.130 | 2.599 | 2.308 | 2.154 | 2.051 | 2.859 |
| H0 | 3.061 | 2.448 | 2.136 | 1.992 | 1.922 | 2.733 |
| H1 | 3.094 | 2.460 | 2.134 | 1.970 | 1.888 | 2.731 |
| H2 | 3.089 | 2.480 | 2.174 | 2.034 | 1.950 | 2.765 |
| H3 | 3.074 | 2.453 | 2.161 | 2.021 | 1.944 | 2.749 |
| H4 | 3.089 | 2.508 | 2.208 | 2.066 | 1.995 | 2.787 |
| H5 | 3.085 | 2.511 | 2.235 | 2.106 | 2.025 | 2.806 |
| H6 | 3.084 | 2.484 | 2.182 | 2.034 | 1.955 | 2.767 |
| H7 | 3.083 | 2.464 | 2.155 | 2.006 | 1.927 | 2.747 |
| H8 | **3.056** | **2.423** | **2.105** | **1.937** | **1.864** | **2.703** |

The first result is decisive against coverage-only explanations. H4 reaches
the entire structural context by layer 8, but its step-500 train loss is
higher than H0, H1, and H8. H3 has the lowest repeated-cell rate, but is not
the best train-loss route. H8 has only 256 reachable offsets at layer 8 and a
high repeated-cell rate, yet it has the lowest train-loss curve after the
early warm-up period.

The working explanation is now narrower: repeated local anchors provide a
stable short-range learning path, while the expanding jump supplies long-range
mixing. If the route expands too aggressively (H4) or spends too much budget
on a broad shell (H3), structural reachability improves without producing a
better optimization path. This is still a small-model result; the next test
should vary depth and context while preserving the same loss trace protocol.

Trace-bearing raw output:
`results/wikitext_factorial_3seed/wikitext2_binary_diffusion_20260928_053606.csv`.

## H8 local-path timing ablation

All five variants have the same dyadic expansion and therefore the same
coverage: 16/256/2048 reachable offsets at layers 4/8/16. Only the local
source schedule changes.

| variant | local schedule | last-10 train loss | step 200 | step 300 | step 500 | src/q |
|---|---|---:|---:|---:|---:|---:|
| H8-a | `{1}` from layer 3 | 1.9010 ± .0067 | 2.4250 | 2.1173 | 1.8733 | 2.62 |
| H8-b | `{2}` from layer 3 | 1.9261 ± .0101 | 2.4450 | 2.1249 | 1.9087 | 2.62 |
| H8-c | `{1,2}` from layer 3 | **1.8885 ± .0061** | 2.4227 | **2.1047** | **1.8637** | 3.37 |
| H8-d | `{1,2}` on odd layers | 1.9041 ± .0080 | **2.4136** | 2.1040 | 1.8784 | 2.62 |
| H8-e | `{1,2}` only in last 2 layers | 1.9340 ± .0077 | 2.4432 | 2.1345 | 1.9038 | 2.37 |

This isolates the local effect more cleanly than the original H0–H8 sweep:

1. Offset `1` is more useful than offset `2`; H8-a beats H8-b by about 0.025
   train loss at step 500.
2. Adding offset `2` on top of offset `1` improves the final loss by another
   approximately 0.010–0.013, so the two local scales are complementary.
3. Continuous local support (H8-c) is best overall. Alternating local support
   (H8-d) is close while using fewer cells, so it is the best efficiency
   compromise in this test.
4. Delaying local support until only the final two layers (H8-e) is clearly
   worse. Local anchors must participate during the middle of optimization,
   not merely repair the final representation.

Thus the current causal explanation is stronger than “repetition helps”:
**repeated nearest-neighbor support during the expansion phase helps train
loss, with offset 1 carrying the main effect and offset 2 providing a smaller
complementary effect.**

Raw output:
`results/wikitext_h8_local_ablation_3seed/wikitext2_binary_diffusion_20260928_060933.csv`.

### Wider local neighborhoods

The same test was extended with local sets `{1,2,3}` and `{1,2,3,4}` from
layer 3 onward. The dyadic jump schedule and all training settings stayed
unchanged.

| variant | local set | last-10 train loss | step 300 | step 500 | src/q | tokens/s |
|---|---|---:|---:|---:|---:|---:|
| H8-c | `{1,2}` | 1.8885 ± .0061 | 2.1047 | 1.8637 | 3.37 | 57.3k |
| H8-f | `{1,2,3}` | **1.8860 ± .0096** | **2.1036** | **1.8538** | 4.11 | 54.6k |
| H8-g | `{1,2,3,4}` | 1.8892 ± .0091 | 2.1070 | 1.8622 | 4.72 | 54.1k |

Offset 3 gives a very small improvement over `{1,2}`; the difference is within
the three-seed variation, so it is not yet a strong claim. Offset 4 gives no
further train-loss benefit while increasing the source budget by about 40%
relative to H8-c and reducing throughput. The current practical sweet spot is
therefore a local set of `{1,2}` or `{1,2,3}`, with `{1,2,3}` worth retaining as
the slightly wider candidate for a longer run.

Raw output:
`results/wikitext_h8_local_width_3seed/wikitext2_binary_diffusion_20260928_062051.csv`.

## Cross-dataset confirmation

To check that the H8 result was not specific to WikiText-2, the same four-way
comparison was run for 500 steps and three seeds on Tiny Shakespeare and Penn
Treebank. The model, context, batch, optimizer, and token budget were kept the
same; only the character vocabulary and corpus changed.

| dataset | route | last-10 train loss | validation loss |
|---|---|---:|---:|
| Tiny Shakespeare | dense | 1.9469 ± .0126 | 1.9864 ± .0175 |
| Tiny Shakespeare | H0 | 1.7622 ± .0114 | 1.8356 ± .0124 |
| Tiny Shakespeare | H1 | 1.7208 ± .0125 | 1.7974 ± .0089 |
| Tiny Shakespeare | H8 | **1.6987 ± .0133** | **1.7817 ± .0173** |
| Penn Treebank | dense | 1.7171 ± .0132 | 1.7497 ± .0035 |
| Penn Treebank | H0 | 1.5369 ± .0216 | 1.5766 ± .0059 |
| Penn Treebank | H1 | 1.4871 ± .0246 | 1.5247 ± .0070 |
| Penn Treebank | H8 | **1.4674 ± .0184** | **1.5111 ± .0028** |

The ranking is consistent across all three corpora:

```text
H8 < H1 < H0 < dense
```

for train loss, where lower is better. This is stronger evidence that the
benefit comes from the route structure—dyadic expansion plus persistent local
support—rather than a WikiText-specific token or phrase distribution.

Raw outputs:

- `results/tinyshakespeare_h0_h1_h8_dense_3seed/`
- `results/ptb_h0_h1_h8_dense_3seed/`
