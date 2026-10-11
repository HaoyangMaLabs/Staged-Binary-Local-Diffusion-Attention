# Novelty audit: local-jump and binary diffusion

This is a conservative prior-art audit, not a novelty or patent opinion. It is
intended to keep the paper's claims aligned with the published definitions.

## Decision record (2026-09-28)

The high-level combination of layer-wise exponential expansion, binary/dyadic
long-range routing, and multi-layer coverage propagation is not treated as
new. PowerAttention, BF1, TRSP, LogSparse, and related hierarchical/binary
methods already cover this design space at the principle level.

The remaining research claim is intentionally narrower:

> A causal exact-token dyadic path cover that reaches broad information
> coverage quickly with a small source/cell budget, while avoiding repeated
> structural `q,k` cells where possible. The layer `l` route exposes the
> scale-appropriate offset `2**(l-1)` and residual composition supplies
> continuous coverage.

This claim is still a hypothesis until coverage-per-compute, unique-cell rate,
and propagation latency are formally measured against PowerAttention and BF1
under matched budgets. The project must not describe itself as the first
exponential, dyadic, binary, or logarithmic sparse-attention method.

The optimization target is therefore not minimum edges in isolation. It is a
path-efficiency trade-off:

\[
\text{high coverage speed}\;/\;\text{low exact-source cost}
\]

with a secondary preference for computing each structural `q,k` grid cell only
once. A route with slightly more sources can be preferable if it reaches useful
positions much earlier or avoids repeated local cells.

## What is already established

### Layer-indexed expansion versus ordinary multi-layer reachability

These are not the same design:

| work | does the layer index choose the expansion scale? | relation to our route |
|---|---|---|
| PowerAttention | generally no; each sparse layer exposes the power-of-two mask, plus local/sink positions | multi-layer binary reachability, but not one scale selected by layer |
| BF1 | no; each converted block layer uses the same local/global-first/dyadic block rule | binary-decomposition reachability, but not a layer-indexed token schedule |
| LogSparse | no; logarithmic predecessors are selected from position/index geometry | log-spaced positions, not layer-controlled expansion |
| LongNet | mainly a mixture of geometric segment/rate patterns inside the attention design | geometric dilation family, not our one-scale-per-layer exact-token rule |
| TRSP | yes; its side-path bandwidth is explicitly (b_l=2^l) and dyadic scales are distributed across layers | closest principle-level precedent, but not exact-token attention |
| our route | yes; layer (l) selects exact offset (2^{l-1}) | layer-indexed exact-token coverage diffusion |

The main claim under investigation is the last row. The other sparse methods
remain useful related work, but they are not all valid primary controls for this
specific hypothesis.

| Idea | Closest prior work | What the prior work actually fixes |
|---|---|---|
| local + strided sparse attention | Sparse Transformer | block-local and strided patterns, commonly assigned to different heads; stride is tied to a block factor rather than a single dyadic edge per layer |
| local sliding window | Longformer | fixed local window; stacked layers grow the receptive field |
| local + dilated window | Longformer | optional dilated sliding windows; dilation/window can vary by head/layer; global tokens are a separate mechanism |
| local and dilated neighborhood attention | DiNAT | vision-oriented NA/DiNA blocks with a dilation schedule; not a causal 1-D exact-token residual graph |
| exponential long-range dilation | LongNet | geometric segment sizes and dilation rates, multiple attention patterns, linear attention interaction complexity and logarithmic dependency path |
| power-of-two receptive-field expansion | PowerAttention | exact power-of-two distance mask, local/sink additions, exponential and complete receptive-field analysis |
| logarithmic historical predecessors | LogSparse Transformer | local attention plus logarithmically spaced predecessor positions |
| binary interval hierarchy | BP-Transformer | fine-to-coarse attention over binary-partitioned multi-scale spans and summaries |
| causal dyadic block graph | BF1 | local neighborhood, global first block, and dyadically spaced historical blocks with binary-decomposition paths |
| hierarchical matrix attention | H-Transformer-1D | multiscale/H-matrix attention computation rather than exact-token residual paths |
| layer-wise exponential bandwidth and dyadic propagation | TRSP | side-path bandwidth (b_l=2^l) with dyadic scales distributed across layers; not exact-token softmax routing |
| power-law similarity / linear state | Symmetric Power Transformers | even-power kernel/similarity with a recurrent linear-attention state; not a sparse token graph |
| tunable subquadratic attention order | Power-based Partial Attention | varies interaction order from linear-window to full attention; not a fixed path topology |
| kernelized linear attention | Performer / Linear Transformer | replaces pairwise softmax with feature maps and associative prefix state; no explicit sparse token path |
| hash/content routing | Reformer / Routing Transformer | selects buckets or clusters from token content; dynamic and not fixed geometric propagation |
| local/random/global graph | BigBird | combines local windows, random links, and global tokens; graph connectivity is not dyadic residual routing |
| dynamic inference-time patterns | MInference | chooses A-shape, vertical-slash, and block-sparse patterns per head for long-context prefilling |
| hierarchical trainable sparse attention | NSA | combines coarse token compression with fine token selection and hardware-aligned kernels |

Therefore the following claims are **not novel by themselves**:

- “local attention plus a long jump”;
- “dilated attention expands the receptive field”;
- “fixed sparse attention has linear interaction cost”;
- “stacked sparse layers can connect distant positions.”

## What may be distinctive in this project

The current binary route has a more specific definition than the general
power-of-two family, but it is not the first power-of-two sparse attention
pattern:

\[
S_l(i)=\{i, i-2^{l-1}\}.
\]

It uses exactly two exact token states per query and layer. Under residual
composition, every offset below `2**L` has a binary-sum route. It does not use
pooling, segment summaries, block-wide strided attention, global tokens, or a
learned router.

The `local_jump` variant is:

\[
S_l(i)=\{i, i-1, i-2, i-2^{l-1}\}.
\]

PowerAttention (Chen et al., 2025) directly connects power-of-two offsets and
adds a local/sink mask in a block-based implementation. LogSparse already uses
logarithmically spaced predecessor positions. BP-Transformer uses binary
partitioned multiscale spans, and BF1 (2026) explicitly defines a causal
dyadic block graph whose reachability proof uses binary decomposition. These
works substantially reduce the novelty space for our binary route. TRSP also
covers the layer-wise exponential-scale and cross-layer propagation principle,
although its operator is a triangular-box side path rather than attention over
selected exact tokens. The only remaining distinction we have identified is
the stricter token-level schedule:
one non-self dyadic edge selected by layer, with no block summary, sink block,
or all-powers-per-layer mask. This is a candidate graph simplification and
implementation point, not a new sparse-attention paradigm.

## Comparison with the closest methods

### Sparse Transformer

Sparse Transformer uses a block-local pattern and a strided pattern. The
strided pattern reaches many positions sharing a residue/block coordinate; it
is not one exact offset `2**l` and it is not generated by binary composition
across residual layers. Our `sparse_local_stride` experiment is a controlled
causal analogue, not a reimplementation of the full head split.

### Longformer

Longformer uses a fixed sliding window, optional dilation, and task-specific
global tokens. Its language-model setup uses increasing window sizes and only
some higher-layer heads with dilation. This is close in motivation but not the
same as giving every layer one binary jump plus a local core.

### DiNAT

DiNAT is a hierarchical vision Transformer built from neighborhood and dilated
neighborhood attention. It is the closest “local+dilated” architectural family,
but its spatial 2-D setting, layer schedule, and neighborhood attention differ
from this causal 1-D exact-token graph.

### LongNet

LongNet is the closest prior work for exponential receptive-field growth. It
uses multiple segment sizes and dilation rates in geometric progression. Our
route is more restrictive: two exact sources per layer and binary path
composition. A direct LongNet implementation should be added before making a
stronger novelty or efficiency claim.

### PowerAttention

PowerAttention is the closest direct prior art to the current binary idea. Its
paper explicitly models sparse attention as a DAG, connects exact positions at
power-of-two distances, and analyzes logarithmic path length with complete
coverage inside the reachable interval. Its pseudocode adds a local window and
sink block to the power mask. Our proposed `power_attention` control mirrors
that logic at token level; the `binary` route remains a stricter
one-edge-per-layer variant. Any paper introduction must cite PowerAttention
and remove claims that exponential receptive-field growth or power-of-two
routing is new.

### LogSparse, BP-Transformer, and BF1

LogSparse is an early direct predecessor for local plus logarithmically spaced
historical positions. BP-Transformer builds a binary partition tree and uses
fine-to-coarse span representations, so it is not exact-token equivalent but
shares the binary multiscale idea. BF1 is the closest recent systems-level
comparison: it defines a causal block route with local predecessors, a global
first block, and dyadic historical blocks, then proves reachability by binary
decomposition. Our implementation differs by operating on exact tokens and by
using one dyadic edge per layer, but the graph-level motivation is not new.

TRSP gives a separate collision on the layer schedule: its bandwidth grows as
(2^l) and its dyadic side-path scales are distributed across depth. It is not
a token-selection attention baseline, so it does not erase the exact operator
distinction, but it does invalidate any broad claim that layer-wise exponential
expansion plus multiscale propagation is new.

The term “power attention” is overloaded. Symmetric Power Transformers raise a
query-key similarity to an even power and exploit a linear-attention state; the
power is a feature-map/state-size choice, not a power-of-two token mask.
Power-based Partial Attention uses “power” to denote a tunable sequence-length
complexity exponent. These belong in the related-work map but are not direct
architectural competitors to our fixed exact-token path.

The added `longnet_like` control uses a causal local core plus geometric exact
token taps. On the matched WikiText-2 run it reached validation loss
`1.9391 ± 0.0078`, compared with `1.9178 ± 0.0150` for `local_jump` and
`1.9516 ± 0.0038` for strict `binary diffusion`. This is evidence about the
chosen fixed-path analogue only; it is not a result for the full LongNet
architecture.

## Safe paper claim

The defensible claim is:

> We study a causal exact-token dyadic path-cover attention mechanism and a
> local-augmented variant. The mechanism combines known local/dilated sparse
> attention ideas with a particularly restrictive two-source binary graph;
> the contribution is the formulation and controlled comparison of this path
> geometry, not the invention of local or dilated attention in general.

## Claims to avoid

- “We are the first local+dilated attention method.”
- “We are the first logarithmic-reachability sparse Transformer.”
- “We are the first power-of-two sparse attention method.”
- “We are the first dyadic/log-sparse causal attention route.”
- “We are the first method to expand a receptive field exponentially across
  layers.”
- “The binary path is equivalent to dense attention.”
- “The current implementation is always faster or uses less memory.”
