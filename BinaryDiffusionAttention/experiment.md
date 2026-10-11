# Mini experiment protocol

## Dataset

Use **WikiText-2 raw** from Merity et al. The archive is only a few MB and
contains the standard train, validation, and test splits. We keep original
case, punctuation, and line structure, then use a character vocabulary built
from train text. This makes the first experiment directly test the attention
path rather than introduce tokenizer differences.

The script downloads the three public raw split files on demand. No book corpus
or cloud-only data is required.

## Fair comparison

Every model uses the same:

- character vocabulary and batches;
- parameter width, layer count, heads, FFN width, tied embeddings, and seed;
- context length and optimizer;
- number of optimizer steps and, for the main comparison, the same number of
  seen training tokens.

The dense model is the reference. `left_local` is the existing aligned local
track. The remaining models use exact source gathers from `BiAttn.left_v2`.
There is no pooling or hidden-state summary in the proposed binary route.

The published methods are not identical to these causal one-dimensional
variants. Longformer and BigBird include global-token/task-specific patterns;
BigBird is primarily an encoder-style bidirectional sparse graph. Dynamic
content-routing methods such as Routing Transformer and Reformer also require
different routing/training code. The fixed-path analogues below are the fair
same-backbone comparisons used in this experiment.

`longnet_like` is a separate causal adaptation of LongNet's geometric mixture
of local and dilated patterns. It preserves the current exact-token backbone;
it does not reproduce LongNet's segment gather/scatter and dynamic mixture
weights, so its result is labeled an analogue rather than an official LongNet
implementation.

`power_attention` is a separate token-level analogue of Chen et al.'s 2025
PowerAttention. It uses a local core plus every power-of-two exact-token offset
at each layer. The published method is block based and includes sink tokens;
those details are kept explicit rather than silently treated as identical.

## Main experimental focus

The main baseline is standard dense causal attention. Literature-aligned
methods are secondary reference points only. The core experiment compares
design choices inside our principle:

> use a small exact-source/cell budget to obtain broad information coverage as
> early as possible, while minimizing repeated structural `q,k` cells.

This is a three-way trade-off between coverage speed, model quality, and actual
systems cost. It is not a minimum-edge contest.

## Why shallow models are insufficient

For a length-(N) sequence, a dyadic path needs roughly

\[
\lceil\log_2 N\rceil
\]

layers before its graph can cover the full interval. A 4- or 8-layer model at
context 256 mostly measures local optimization and projection/FFN cost. The
main depth sweep must include 8, 16, 24, and 32 layers, with context lengths
256, 512, 1024, and 2048.

Within each comparison, width, FFN ratio, optimizer, seen tokens, batch shape,
tokenizer, and initialization remain matched. If changing depth changes the
parameter count, report both matched-depth and parameter-matched controls.

## Proposed-design ablations

| family | controlled question |
|---|---|
| dense attention | reference quality ceiling and quadratic cost |
| one dyadic source/layer | does scale-selected propagation work at small source cost? |
| same source budget, permuted scales | is ascending (1,2,4,\ldots) important? |
| same source budget, random fixed scales | is deterministic coverage better than equal-budget random routing? |
| one scale repeated at every layer | does depth-wise scale growth matter? |
| dyadic source + repeated local core | does local coverage justify repeated cells? |
| dyadic source + non-repeating local cells | can local coverage be retained without duplication? |
| 1, 2, 4, and 8 sources/layer | quality/coverage/compute frontier |
| greedy new-coverage route | does marginal coverage selection beat fixed dyadic order? |

The external methods remain in a small appendix table and do not receive the
same sweep budget as these internal design variants.

## Internal layer-expansion hypotheses

All hypotheses below use the same residual exact-token propagation rule. For a
query at the right boundary, let `R_0={0}` and let `S_l` be the source
offsets exposed at layer `l`:

\[
R_l=\{r+s:r\in R_{l-1},s\in S_l\}\cap[0,N).
\]

The experiments vary only `S_l`, so any difference is attributable to the
layer-expansion geometry.

| ID | hypothesis | layer source set `S_l` | purpose |
|---|---|---|---|
| H0 | strict binary | `\{0, 2**(l-1)\}` | minimum simple route; baseline internal hypothesis |
| H1 | local jump | `\{0, 1, 2, 2**(l-1)\}` | tests whether repeated local cells improve early coverage |
| H2 | disjoint double jump | `\{0, 2**(l-1), 3*2**(l-1)\}` | two expanding lanes with no repeated nonzero offsets |
| H3 | dyadic shell | `\{0\} ∪ [2**(l-1), 2**l-1]` | near-maximal new coverage per layer; deliberately expensive upper bound |
| H4 | unique greedy | choose unused offsets maximizing new `R_l` per source | direct coverage-per-cell optimization |
| H5 | unique local-plus-jump | local offsets are assigned once, then only expanding jumps | tests whether local help is needed repeatedly |
| H6 | staggered radix | fixed small number of expanding offsets with layer-specific phase | tests wider propagation without shell-level cost |
| H7 | late expansion | hold a short local schedule first, then begin dyadic expansion | tests whether local stabilization before global spread helps |
| H8 | early expansion | start dyadic expansion immediately, add local only after coverage begins | tests the opposite order |

H2/H3/H4 are especially important for the “single-use” claim. H1 is a useful
negative control because its repeated local core should increase coverage speed
but lower the unique-cell rate. H4 is not intended to be the final deployable
route; it estimates the best attainable structural frontier under a source
budget.

The primary selection criterion is train loss at a fixed seen-token budget;
structural metrics are explanatory variables rather than the objective:

\[
\text{train loss}\quad\leftarrow\quad
\big(\text{coverage speed},\;\text{repeat-cell rate},\;
\text{source cost},\;\text{route order}\big).
\]

A route wins only if it lowers train loss at matched token budget. Coverage,
repeat rate, source cost, and wall-clock time explain why it may do so; they do
not replace the loss measurement.

## Required metrics

For each query and depth, record reachable positions `R_l`, coverage AUC over
depth, and mean/max shortest path. Let (E_l) be selected structural cells and
(U_l) their unique set:

\[
\text{source cost}=|E_l|,\qquad
\text{unique-cell rate}=|U_l|/|E_l|,
\]

\[
\text{coverage efficiency}=\text{coverage AUC}/|E_l|.
\]

The unique-cell metric is structural: the same `(query,key)` coordinates at a
different layer still use different hidden states, but count as repeated grid
coverage for this hypothesis.

At a fixed seen-token budget also report train loss, validation NLL/PPL,
distance-controlled copy/passkey accuracy, forward and train tokens/s, peak
memory, selected FLOPs, and compile/kernel-launch overhead. The main plots are
quality versus source cost, quality versus unique-cell rate, coverage AUC versus
source cost, and tokens/s versus context length.

## Main run matrix

1. **Graph-only:** route variants at depths 4, 8, 16, 24, 32 and contexts 256,
   512, 1024, 2048.
2. **Smoke training:** dense plus internal variants, depth 16, context 512,
   100 steps, one seed.
3. **Matched quality:** dense plus the best 4–6 internal variants, depths 8,
   16, 24, contexts 512/2048, three seeds, fixed seen-token budget.
4. **Long-range probe:** copy/passkey positions at distances 32, 128, 512,
   and 1024.
5. **Systems run:** contexts 256/512/1024/2048/4096, forward and backward,
   plus a separate batch-1 decode-style test.

The external comparison table is populated only for the final selected route.

The primary controls are therefore dense attention and the internal
layer-schedule variants. PowerAttention, BF1, LogSparse, and LongNet are
related-work references rather than interchangeable controls: most use a
multi-layer graph whose mask is repeated or mixed across layers, whereas this
project specifically tests whether assigning one expanding exact-token scale
to each layer improves coverage-per-cell. TRSP is the closest conceptual
reference because it explicitly expands a side-path bandwidth with depth, but
its operator is not the same attention-cell computation.

## Initial graph-only result: context 2048

The first structural scan used the final query position in a length-2048
sequence. `coverage` is the number of distinct source positions reachable after
all layers; `AUC` is the sum of reachable positions after each layer; `src/q`
is the average exact source count per query per layer; and `unique-cell rate`
counts repeated structural `(query,key)` coordinates across layers.

| route | layers | coverage | AUC | src/q | unique-cell rate |
|---|---:|---:|---:|---:|---:|
| binary | 4 | 16 | 30 | 2.00 | 0.625 |
| binary | 8 | 256 | 510 | 1.98 | 0.559 |
| binary | 16 | **2048** | 14,334 | **1.63** | 0.423 |
| local-jump | 4 | 17 | 34 | 3.50 | 0.357 |
| local-jump | 8 | 257 | 518 | 3.73 | 0.297 |
| local-jump | 16 | **2048** | 14,344 | 3.50 | 0.197 |
| butterfly | 4 | 46 | 82 | 3.99 | 0.624 |
| butterfly | 8 | 766 | 1,514 | 3.91 | 0.556 |
| butterfly | 16 | **2048** | 17,384 | 2.72 | 0.448 |
| high-4 | 4 | 46 | 82 | 5.24 | 0.475 |
| high-4 | 8 | 766 | 1,514 | 5.53 | 0.393 |
| high-4 | 16 | **2048** | 17,384 | 4.53 | 0.269 |
| PowerAttention analogue | 4 | 30 | 72 | 5.99 | 0.250 |
| PowerAttention analogue | 8 | 736 | 2,575 | 9.87 | 0.125 |
| PowerAttention analogue | 16 | **2048** | 24,063 | 12.00 | **0.062** |

Dense attention reaches the entire prefix in one layer, but its causal source
cost is approximately (N/2) per query per layer and its structural-cell reuse
rate is approximately (1/L). The scan confirms that 4 layers are insufficient
for the intended long-range claim, 8 layers are transitional, and 16 layers
is the first useful regime for the 2048 context. The next quality experiment
should therefore use 16 layers as the primary setting, with 8 layers retained
as a shallow failure/control point.

## Internal layer-expansion hypotheses

The next scan tests only our own layer-indexed expansion family. These are
controlled structural hypotheses for the same principle: use a small number of
exact source cells, expand the reachable prefix with depth, and avoid
repeatedly lighting the same structural `(query,key)` coordinates.

| route | construction |
|---|---|
| H0 | strict binary: self plus one dyadic jump |
| H1 | local core `(1,2)` plus the dyadic jump |
| H2 | two disjoint dyadic jumps: `2^(l-1)` and `3·2^(l-1)` |
| H3 | dyadic shell: all offsets in `[2^(l-1), 2^l)` |
| H4 | stateful greedy: two unused offsets selected for maximum marginal reach |
| H5 | local core only at layer 1, then one new dyadic jump per layer |
| H6 | staggered radix: dyadic jump plus `2·2^(l-1)+1` |
| H7 | late expansion: local core for two layers, then binary expansion |
| H8 | early expansion: binary for two layers, then local core plus binary |

The graph-only sweep used lengths 256, 512, 1024, and 2048 and depths 4, 8,
and 16. The complete machine-readable result is
`results/internal_hypotheses_all.csv`. The 2048-token frontier is:

| route | L | coverage | AUC | src/q | unique-cell rate |
|---|---:|---:|---:|---:|---:|
| H0 | 4 / 8 / 16 | 16 / 256 / 2048 | 30 / 510 / 14334 | 2.00 / 1.98 / 1.63 | .625 / .559 / .423 |
| H1 | 4 / 8 / 16 | 17 / 257 / 2048 | 34 / 518 / 14344 | 3.50 / 3.73 / 3.50 | .357 / .297 / .197 |
| H2 | 4 / 8 / 16 | 42 / 758 / 2048 | 72 / 1478 / 17338 | 2.99 / 2.94 / 2.16 | .749 / .702 / .565 |
| H3 | 4 / 8 / 16 | 27 / 503 / 2048 | 46 / 976 / 16315 | 4.74 / 30.88 / 64.97 | .842 / .972 / .986 |
| H4 | 4 / 8 / 16 | 81 / 2048 / 2048 | 120 / 4598 / 20982 | 2.99 / 2.82 / 2.90 | .749 / .689 / .677 |
| H5 | 4 / 8 / 16 | 24 / 384 / 1536 | 45 / 765 / 12285 | 2.25 / 2.09 / 1.63 | .666 / .582 / .423 |
| H6 | 4 / 8 / 16 | 33 / 517 / 2048 | 59 / 1033 / 16399 | 2.99 / 2.95 / 2.19 | .750 / .704 / .571 |
| H7 | 4 / 8 / 16 | 17 / 257 / 2048 | 34 / 518 / 14344 | 2.50 / 2.23 / 1.75 | .500 / .497 / .393 |
| H8 | 4 / 8 / 16 | 16 / 256 / 2048 | 30 / 510 / 14334 | 3.00 / 3.48 / 3.37 | .416 / .318 / .204 |

The immediate read is that H4 is the strongest structural candidate: it reaches
the full 2048-prefix by eight layers while staying near three sources per query
per layer and retaining a high unique-cell rate. H2 and H6 are the lower-cell,
non-greedy alternatives. H3 has an even higher unique-cell rate, but its shell
width grows with the layer and makes the source budget explode, so it is not an
efficiency candidate at depth 8 or 16. H5 is too conservative at this depth.
H7/H8 are useful timing and optimization controls for expansion order, not
expected winners.

This is still a graph result, not a language-model result. The next training
matrix should use H0, H2, H4, H6, H7, and H8 at 16 layers, with H1/H3/H5 kept
as diagnostic controls only if the source budget is explicitly matched.

## Recommended runs

1. **Smoke correctness:** 20 steps, sequence length 256, all modes, one seed.
   Confirm that all modes produce finite loss and that the output shapes agree.
2. **Matched mini training:** 500 steps, sequence length 256, three seeds.
   Report mean and standard deviation of validation NLL/PPL.
3. **Context scaling:** sequence lengths 128, 256, 512 with the same model and
   token budget. Report validation NLL, tokens/s, peak memory, and the number of
   exact source edges.
4. **Depth ablation:** use the expanded 8/16/24/32-layer sweep above; 4-layer
   results are graph diagnostics only, not a quality conclusion.

## Initial smoke result

The first local run used one seed, 20 steps, a 256-character context, batch
size 4, width 96, 8 layers, 4 heads, and FFN width 256. All models had
984,096 parameters. These numbers are only a pipeline check; they are far too
short to support an accuracy claim.

| mode | validation loss | validation PPL | tokens/s after warmup | exact sources/query |
|---|---:|---:|---:|---:|
| full | 5.6163 | 274.87 | 17,598 | 128.5 |
| left-local | 5.5727 | 263.15 | 18,328 | 32.375 |
| binary diffusion | 5.5556 | 258.68 | 16,878 | 2 |
| local-jump | 5.5887 | 267.39 | 15,452 | 3.75 |
| butterfly | 5.5871 | 266.96 | 14,340 | 4 |
| radix-4 | 5.5683 | 261.99 | 14,966 | 4 |
| high-4 | 5.5945 | 268.93 | 13,668 | 5.625 |

The raw output is generated under `results/` and is intentionally ignored by
Git. The only conclusion from this smoke is that all paths execute and the
binary route has the smallest exact-source budget; longer matched runs are
required to evaluate quality and real end-to-end speed.

## Completed matched results

The completed comparison used the GTX 1080 Ti, 500 optimizer steps, 3 seeds,
sequence length 256, batch size 8, width 96, 8 layers, 4 heads, and FFN width
256. All variants had 984,096 parameters.

| mode | validation loss | PPL | tokens/s | peak MB | sources/query |
|---|---:|---:|---:|---:|---:|
| full | 2.0850 ± 0.0012 | 8.044 ± 0.009 | 48,865 ± 14,625 | 204 | 128.5 |
| left-local | 2.0684 ± 0.0072 | 7.912 ± 0.057 | 49,986 ± 5,244 | 205 | 32.375 |
| binary diffusion | **1.9516 ± 0.0038** | **7.040 ± 0.027** | 48,942 ± 24,816 | 226 | **2** |
| local-jump | **1.9178 ± 0.0150** | **6.807 ± 0.102** | 60,819 ± 333 | 253 | 3.75 |
| butterfly | 2.0003 ± 0.0146 | 7.392 ± 0.108 | 60,755 ± 296 | 257 | 4 |
| radix-4 | 2.0507 ± 0.0082 | 7.774 ± 0.064 | 60,705 ± 268 | 257 | 4 |
| high-4 | 1.9301 ± 0.0104 | 6.890 ± 0.072 | 57,574 ± 379 | 272 | 5.625 |

The sparse routes beat dense validation loss under this short fixed training
budget, but this is not an asymptotic accuracy claim. The local-jump and high-4
routes also show that a small local core plus a long jump can be a stronger
practical point than the strict two-source binary route.

The literature-aligned causal analogues were then run with the same 500-step,
three-seed configuration:

| analogue | validation loss | PPL | tokens/s | peak MB | sources/query |
|---|---:|---:|---:|---:|---:|
| Longformer-local | 1.9214 ± 0.0030 | 6.831 ± 0.021 | 56,773 ± 1,717 | 257 | 4 |
| Sparse local-stride | 1.9323 ± 0.0020 | 6.905 ± 0.014 | 55,719 ± 1,260 | 264 | 5 |
| DiNAT-like | 1.9425 ± 0.0103 | 6.976 ± 0.072 | 46,567 ± 6,712 | 285 | 6.875 |
| LongNet-like | 1.9391 ± 0.0078 | 6.953 ± 0.055 | 50,423 ± 1,991 | 282 | 6.5 |
| PowerAttention analogue | 1.9613 ± 0.0090 | 7.109 ± 0.064 | 45,334 ± 227 | 329 | 10 |

The temporary `dilated_local` name was removed from the method list because its
offset set was exactly the same as `butterfly`; it was not counted as an
independent method.

The LongNet-like analogue uses the causal exact-token pattern
`{0, 1, 2, 3, d, 2d, 3d}` with a geometric layer-dependent dilation `d`.
It is deliberately labeled an analogue: the published LongNet design uses
geometric segment sizes/rates with segment gather/scatter and dynamic mixture
weights, which are not present in this fixed-path control.

### WikiText-2 internal-hypothesis test

All internal routes were tested with one seed for 500 steps on the same
WikiText-2 character task: sequence length 256, batch size 8, width 96,
8 layers, 4 heads, FFN width 256. Every model has 984,096 parameters.

| route | val loss | PPL | tokens/s | peak MB | src/q |
|---|---:|---:|---:|---:|---:|
| dense | 2.1127 | 8.271 | 58,676 | 204 | 128.50 |
| H0 | 1.9780 | 7.228 | 61,672 | 226 | 1.88 |
| H1 | 1.9448 | 6.992 | 60,235 | 253 | 3.62 |
| H2 | 2.0085 | 7.452 | 60,271 | 239 | 2.56 |
| H3 | 1.9949 | 7.352 | 29,489 | 674 | 16.94 |
| H4 | 2.0363 | 7.662 | 52,939 | 240 | 2.80 |
| H5 | 2.0865 | 8.056 | 62,276 | 226 | 1.88 |
| H6 | 2.0094 | 7.458 | 60,946 | 239 | 2.62 |
| H7 | 1.9913 | 7.325 | 62,167 | 227 | 2.12 |
| H8 | **1.9236** | **6.846** | 60,360 | 251 | 3.37 |

This is a one-seed screening result, not a final quality claim. H8 is the best
language-model result in this run, while H4's graph-only coverage advantage does
not transfer directly to loss. H3 confirms the cost problem experimentally:
its larger shell source set is about half the throughput and uses substantially
more memory. The result supports separating structural coverage from trainable
information routing; the next confirmation should repeat dense, H0, H1, H4,
H7, and H8 across three seeds.

Raw output: `results/wikitext_internal/`.

### Three-seed confirmation

The same WikiText-2 setup was repeated with seeds 1, 2, and 3. The structural
coverage and repeat rate are deterministic for a fixed route, so the three
seeds test whether the language-model result is stable rather than changing
the graph itself.

| route | val loss | PPL | tokens/s | src/q | 8-layer coverage | 8-layer repeat |
|---|---:|---:|---:|---:|---:|---:|
| dense | 2.1092 ± .0036 | 8.242 | 53.2k | 128.50 | 2048 | .875 |
| H0 | 1.9766 ± .0085 | 7.218 | 59.0k | 1.88 | 256 | .441 |
| H1 | 1.9425 ± .0048 | 6.976 | 57.7k | 3.62 | 257 | .703 |
| H2 | 2.0094 ± .0006 | 7.459 | 59.0k | 2.56 | 758 | .298 |
| H3 | 1.9933 ± .0013 | 7.340 | 28.7k | 16.94 | 503 | .028 |
| H4 | 2.0366 ± .0020 | 7.664 | 51.2k | 2.80 | **2048** | .311 |
| H5 | 2.0848 ± .0048 | 8.043 | 58.0k | 1.88 | 384 | .418 |
| H6 | 2.0116 ± .0016 | 7.475 | 58.2k | 2.62 | 517 | .296 |
| H7 | 1.9809 ± .0074 | 7.250 | 58.3k | 2.12 | 257 | .503 |
| H8 | **1.9215 ± .0066** | **6.832** | 57.9k | 3.37 | 256 | .682 |

Train loss is the primary result for this test. The final train loss and the
mean of the last ten train-loss samples over the three seeds are:

| route | final train loss | last-10 train loss |
|---|---:|---:|
| dense | 2.0508 ± .0367 | 2.0802 ± .0135 |
| H0 | 1.9215 ± .0400 | 1.9500 ± .0092 |
| H1 | 1.8884 ± .0363 | 1.9162 ± .0041 |
| H2 | 1.9498 ± .0329 | 1.9817 ± .0086 |
| H3 | 1.9437 ± .0350 | 1.9687 ± .0112 |
| H4 | 1.9949 ± .0368 | 2.0105 ± .0117 |
| H5 | 2.0254 ± .0427 | 2.0566 ± .0151 |
| H6 | 1.9550 ± .0378 | 1.9811 ± .0072 |
| H7 | 1.9265 ± .0452 | 1.9586 ± .0063 |
| H8 | **1.8637 ± .0442** | **1.8885 ± .0061** |

This does not support the simple rule “faster coverage plus lower repeat rate
must lower train loss”. H4 has the fastest graph coverage, but its train loss
is worse than H0/H1/H8. H3 has the lowest repeat rate, but is also not the
best-trained route. H8 has high repeat rate and modest 8-layer coverage, yet
has the lowest train loss. Across H0–H8, the correlation of mean last-10 train
loss with 8-layer coverage is +0.43, while correlation with repeat rate is
-0.55; both are opposite to the naive expected direction in this small,
confounded route family. Source count alone is nearly uninformative here
(correlation -0.03).

The current evidence therefore favors a refined hypothesis: coverage is useful
only when its path order is trainable, and some repeated local cells act as a
conditioning/short-range support rather than pure waste. Raw three-seed output:
`results/wikitext_internal_3seed/`.

### Context scaling

These are one-seed, 300-step runs with the same model and batch size. The
results show the intended regime clearly: at context 128, kernel overhead makes
binary slightly slower than dense; at context 512, binary is faster. Its
current gathered-source implementation does not yet reduce peak memory versus
fused dense SDPA.

| context | mode | val loss | tokens/s | peak MB | reachable sources |
|---:|---|---:|---:|---:|---:|
| 128 | full | 2.4129 | 43,762 | 116 | 128 |
| 128 | left-local | 2.3743 | 40,097 | 116 | 128 |
| 128 | binary | 2.2955 | 39,962 | 125 | 128 |
| 128 | local-jump | 2.3026 | 38,928 | 137 | 128 |
| 128 | high-4 | 2.3159 | 37,989 | 150 | 128 |
| 512 | full | 2.3267 | 66,564 | 391 | 512 |
| 512 | left-local | 2.2181 | 67,591 | 391 | 256 |
| 512 | binary | **2.1072** | **81,040** | 428 | 256 |
| 512 | local-jump | **2.1012** | 76,372 | 474 | 257 |
| 512 | high-4 | 2.1279 | 71,743 | 522 | 512 |

### Depth scaling

With context 512, binary reaches 16, 64, 256, and 512 structural offsets at
depths 4, 6, 8, and 10 respectively. Validation improves through depth 8 and
then plateaus, while the model grows and throughput falls.

| layers | parameters | val loss | tokens/s | peak MB | reachable sources |
|---:|---:|---:|---:|---:|---:|
| 4 | 540,768 | 2.1682 | 148,707 | 257 | 16 |
| 6 | 762,432 | 2.1293 | 103,943 | 344 | 64 |
| 8 | 984,096 | **2.1072** | 80,906 | 429 | 256 |
| 10 | 1,205,760 | 2.1124 | 64,949 | 512 | 512 |

### 4090 long-context forward benchmark

To separate attention-path throughput from training effects, we also ran a
forward-only benchmark on an NVIDIA GeForce RTX 4090 with 49,140 MiB of
reported memory, driver 610.43.03, CUDA 13.0, and PyTorch 2.14.0+cu130. The
test used batch size 1, BF16, an 8-layer model with width 96, four heads, FFN
hidden size 256, and a 6400-token tied vocabulary. It called
`forward_hidden`, so it excluded backward, optimizer updates, and the final
vocabulary projection. Each point used two warmups and three timed forwards.

This was a matched implementation comparison. The standard causal reference
uses `attention_backend="full"` and calls PyTorch
`scaled_dot_product_attention(..., is_causal=True)`. Although the optional
external FlashAttention flag is off, PyTorch selected its fused Flash
Attention kernel automatically; the profiler showed
`aten::_scaled_dot_product_flash_attention`. SBDA used its fused Triton
exact-source kernel. Neither path used `torch.compile`.

| context | standard causal tok/s | SBDA tok/s | SBDA speedup | peak GiB, standard / SBDA |
|---:|---:|---:|---:|---:|
| 4,096 | 426,022 | 404,410 | 0.95x | 0.028 / 0.028 |
| 8,192 | 849,205 | 777,500 | 0.92x | 0.043 / 0.043 |
| 16,384 | 1,577,368 | 1,585,836 | 1.01x | 0.069 / 0.069 |
| 32,768 | 1,348,206 | 3,009,886 | 2.23x | 0.125 / 0.125 |
| 65,536 | 924,043 | 4,780,205 | 5.17x | 0.237 / 0.237 |
| 131,072 | 515,188 | 4,492,494 | 8.72x | 0.460 / 0.460 |
| 262,144 | 266,107 | 3,750,987 | 14.10x | 0.906 / 0.906 |
| 524,288 | 130,507 | 3,641,245 | 27.90x | 1.799 / 1.799 |
| 1,048,576 | 64,231 | 3,569,581 | 55.57x | 3.584 / 3.584 |

Both models ran at 1M context without out-of-memory. Standard causal attention
was still memory-feasible because fused SDPA did not materialize a full
`N x N` score matrix, but its quadratic work reduced throughput to about 64K
tokens/s. SBDA remained near 3.57M tokens/s. At 4K and 8K, SBDA was slightly
slower because fixed-source kernel overhead dominates; the crossover appeared
near 16K.

This is a forward systems result for the small matched model, not a claim that
training a large language model at 1M context will obtain the same multiplier.
The reproducible script is `bench_sbda_vs_causal_forward.py`, and the raw
machine-readable output is `results/sbda_vs_causal_4090_forward.json`.

### WikiText-2 1M-context training comparison

We then trained the same small model with standard causal attention and SBDA
for three passes over WikiText-2. The training file was split into ten
non-overlapping complete windows of 1,048,576 characters; the remaining
433,131 characters were excluded rather than padded. Each pass therefore had
ten updates, for thirty 1M-token updates per mode. The run used batch size 1,
BF16, width 64, four layers, four heads, FFN width 128, a tied character
vocabulary of 1,014 entries, AdamW with learning rate 3e-4, and no
`torch.compile`. Both modes used the same data order and initialization seed.

The standard reference used PyTorch fused causal SDPA/FlashAttention. SBDA
used the fused Triton kernel, including its first-pass compilation time.

| mode | pass 1 mean CE | pass 2 mean CE | pass 3 mean CE | total time | overall tok/s | peak GiB |
|---|---:|---:|---:|---:|---:|---:|
| standard causal | 6.5671 | 6.1880 | 5.8487 | 813.4 s | 38.7K | 29.515 |
| SBDA | 6.5823 | 6.1715 | 5.8294 | 12.3 s | 2.55M | 29.546 |

After the first-pass kernel compilation, SBDA reached 3.52M and 3.51M tok/s
on passes 2 and 3, versus 38.7K tok/s for the standard path: about 91x in
steady-state throughput and 66x including compilation. The final train CE was
close in this short run; SBDA was lower by 0.0193, which is not enough to
claim a quality advantage from one seed and only three passes. Peak training
memory was effectively identical because the 1M-position vocabulary logits,
not the attention score matrix, dominate this small experiment.

This is a systems smoke test, not a language-quality benchmark: there is no
held-out evaluation here, and the model is deliberately small so the two
1M-context training runs finish quickly. The reproducible script is
`train_wikitext_1m_pass_compare.py`; raw outputs are
`results/wikitext_1m_causal_3pass.json` and
`results/wikitext_1m_sbda_3pass.json`.

### WikiText-2 65K-context, 16-layer training comparison with dev

To test whether the binary route still covers a practically larger receptive
field, the same comparison was repeated at `seq_len=65,536` with 16 layers
and 20 passes. WikiText-2 `wiki.train.raw` was split into 166 complete
non-overlapping windows; 39,915 trailing training characters were dropped.
The held-out source was `wiki.valid.raw`, giving 17 complete dev windows after
dropping 30,635 trailing characters. Batch size was 1, with BF16, width 64,
four heads, FFN width 128, a tied 1,014-character vocabulary, AdamW at
3e-4, and no `torch.compile`. The causal baseline used fused PyTorch causal
SDPA/FlashAttention; SBDA used its fused Triton path.

Hardware was the remote `temp-server.yaml` machine: NVIDIA GeForce RTX 4090
with 49,140 MiB reported VRAM (about 48 GiB), driver 610.43.03, CUDA 13.0,
and PyTorch 2.14.0+cu130. The measured peak allocated memory was about 5.1
GiB for both runs.

| mode | pass 1 train CE | pass 20 train CE | pass 20 dev CE | total time | overall tok/s | peak GiB |
|---|---:|---:|---:|---:|---:|---:|
| standard causal | 4.4062 | 2.3188 | 2.2956 | 1548.7 s | 140.5K | 5.099 |
| SBDA | 4.3815 | 1.3256 | 1.3739 | 283.3 s | 767.9K | 5.113 |

At this length and depth, SBDA was about 5.47x faster end to end, including
kernel overhead, while using essentially the same allocated memory. Its final
train CE was 0.993 lower and its dev CE was 0.922 lower in this run. This is
an optimization/throughput comparison on a small character model, not yet a
matched-quality claim: the two modes do not converge to the same loss, and
the result is from one seed.

The raw records are `results/wikitext_65536_l16_causal_20pass_dev.json` and
`results/wikitext_65536_l16_sbda_20pass_dev.json`; complete console logs are
`results/wikitext_65536_l16_causal_20pass_dev.log` and
`results/wikitext_65536_l16_sbda_20pass_dev.log`.

## Metrics

- `val_loss`: mean character next-token cross entropy;
- `val_ppl`: `exp(val_loss)`; useful but less comparable across tokenizations;
- `tokens_per_second`: measured after warmup;
- `peak_memory_mb`: CUDA allocated peak when running on CUDA;
- `source_edges_per_query`: average exact attention sources per destination;
- `reachable_sources`: structural number of reachable causal offsets after all
  layers;
- parameter count and wall-clock time.

The primary learning metric should be train loss at a fixed seen-token budget;
validation NLL/PPL is an auxiliary generalization check. The primary systems
metrics are tokens/s, peak memory, coverage speed, and repeated-cell rate at a
fixed context length.

## Interpretation guardrails

The binary graph has global *path reachability* after enough layers, but this
does not mean it computes the same function as dense attention in one layer.
Information passes through intermediate residual states and can attenuate,
interfere, or become difficult to optimize. Therefore the paper must report:

- route depth and structural reachability;
- optimization stability and seed variance;
- long-context behavior;
- wall-clock measurements, not only asymptotic edge counts.

The existing implementation uses tiny exact-source SDPA calls. The theoretical
attention interaction count is linear in sequence length for fixed layers and
source count, but Q/K/V projections and the FFN remain and can dominate on
short sequences. This is why the experiment includes both source counts and
end-to-end throughput.
