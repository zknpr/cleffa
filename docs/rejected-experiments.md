# Rejected experiments

Changes that were built, checked for exact output where that applied, and timed, but not
retained. Each entry records the hypothesis, the frozen pass criterion, the measurement and
why it failed, so the idea is not retried without new evidence. Kernel checks alone never
qualified a change: a candidate that changed arithmetic also needed whole-model probability
error against the FP32 oracle, and a candidate that preserved bits still needed a paired
complete-request gain.

Measurement conventions are those of the [performance history](performance-history.md).
Evidence directories are local and git-ignored under `golden/`.

## Attention

### Larger softmax key blocks

Computing 64 or 128 keys per block instead of 32 lowered isolated attention time by 9–18%
but changed the online softmax grouping: Flash mean probability error against FP32 rose from
0.00003576 to 0.00005265 (64 keys) and 0.00004813 (128 keys), 27B from 0.00007982 to
0.00010909 and 0.00009872, with decisions unchanged. Rejected on the precision regression.
The retained alternative prefetches 64 score columns while keeping every 32-key softmax
update; see [context scaling](attention.md#context-scaling-and-64-key-prefetch).

### More SIMDgroups per query tile

An isolated follow-up keeps the 32-query, 64-key tile and all FP32 arithmetic, but assigns
its work to eight or sixteen SIMDgroups instead of four. Query/key score work is divided
across key columns. Only four groups perform the original softmax updates; all groups own
disjoint output columns. This reduces output accumulator storage per thread from 64 floats
to 32 or 16, without changing each output's reduction order.

Both variants preserve every tested output bit on the original grid and on full-FP32
mantissa fixtures. The four runs pass 32 small packed checks, 64 long packed checks and 96
single-shape/head checks. These include FP16/BF16 output, mixed overflow flags, poisoned
output/tail guards, record invariance and sampled float64 checks.

Neither improves the relevant timings. Across the tested lengths from 1,024 through 16,347
tokens, eight groups are 1.9–11.2% slower than current four-group prefetch, and sixteen
groups are 19.4–33.1% slower. These are isolated screening means from six measured calls per
arm after two warm-ups, with alternating dispatch order. They are not whole-request
measurements.

At 16,347 tokens on the full-FP32 fixture:

| Groups | Query heads | Four-group baseline | Candidate | Added kernel time |
|---:|---:|---:|---:|---:|
| 8 | 16 | 263.934 ms | 290.218 ms | 10.0% |
| 8 | 24 | 387.235 ms | 417.652 ms | 7.9% |
| 16 | 16 | 256.052 ms | 312.036 ms | 21.9% |
| 16 | 24 | 394.293 ms | 491.258 ms | 24.6% |

Lower per-thread accumulator storage does not produce a speedup in this experiment. No
counter trace isolates why. The candidates are rejected at the kernel screen; their prepared
full-engine builds and qualification runner are not run. That experiment leaves production
inference sources and binaries unchanged.

Artifacts, source hashes, raw logs and the parsed timing table are under
`golden/attention-groups-20261004/`, particularly `manifest.json`, `results.json` and
`analysis.json`. The retained change is the broader precision fixture described in [the
precision report](attention.md#compensated-attention-precision-screen).

### Full-precision MPP value products and head grouping

A full-FP32 MPP prototype with a single 256-column value product failed the one-token
oracle. A version using 64-column products passed, but was slower. A second version combines
the production kernel's shared 32-row scores with MPP value products and cooperative
accumulators. It preserves all output bits in the packed, overflow, NaN-guard and sampled
float64 checks. Alternating two warm-ups and six measured calls per kernel gave:

| Tokens / query heads | Production reuse | MPP value products |
|---|---:|---:|
| 4,510 / 24 | 26.511 ms | 30.220 ms |
| 8,072 / 24 | 83.618 ms | 95.400 ms |
| 16,347 / 24 | 368.463 ms | 417.941 ms |

Grouping all query heads that share a KV head into one threadgroup was also slower. Both
16-query-row and 32-query-row variants passed the same exact-output checks, but lost to the
production kernel at 4,510, 8,072 and 16,347 tokens for both head counts. These variants
were rejected without changing the production shader.

### Grouped query tiles, multi-tile SIMDgroups and a split down-projection reduction

Other attention experiments were less useful. Merely grouping four existing query tiles
saved about 2–3% on long 27B requests but did not establish a packed-workload gain. Holding
multiple complete output tiles in one SIMDgroup increased register pressure and regressed.
Splitting the down-projection reduction changed rounding and produced no useful long-request
gain; it was not retained or accepted as a quality-preserving change.

### Compensated-attention variants

Ten variants of the compensated tensor-unit kernel (128- or 32-key blocks, rounded or
unrounded denominators, probability scales of 1,024 or 32,768, scaled residual planes, and a
fourth residual-times-residual product) all kept 46/46 decisions and all increased mean
probability error against FP32 on both models; none improved the mean on either. The
original `c` variant was later adopted on labeled-task evidence; the others remain
rejected. The table is in the
[precision screen](attention.md#compensated-attention-precision-screen).

### Exact rescaling shortcut

Status: all four isolated numerical screens pass; the change is not adopted because the
timing improvement is inconsistent. No whole-model speedup is established. Artifacts are in
`golden/attention-alpha-20261005/`.

The ordinary FP32 attention loop rescales the accumulator with `exp(m - mn)`. When the old
and new running maxima are the same finite float, that factor is exactly one. The scalar
variant tests this per row. The uniform variant uses `simd_all` before the divergent
finite-maximum branch, then skips the exponential only if every participating lane has the
same finite maximum. Infinite and NaN cases retain the original expression. The 32-key
update order, operand precision and complete context remain unchanged.

The screen compares the original and candidate **prefetch-64** kernels on the same arrays,
under Metal safe math. Each shape has two warm pairs followed by six measured pairs in three
alternating quartets. It covers 1,024, 2,235, 8,072 and 16,347 tokens with 16 or 24 heads,
both grid-valued and unrestricted FP32 input patterns, and the existing
packed/poison/overflow and float64 checks. Timings are isolated GPU dispatches, not request
latency.

The scalar variant passes both numerical screens but has inconsistent timings, including
slower 16K/24-head medians in both input distributions. It has not been advanced to
whole-model qualification.

The uniform grid screen passes exact-bit comparisons. Its long 24-head shapes provide a
narrower lead:

| Tokens | Heads | Median kernel-time reduction | Reductions in the three paired quartets |
|---:|---:|---:|---|
| 8,072 | 16 | -0.31% | +1.55%, +0.22%, -2.32% |
| 8,072 | 24 | +2.06% | +1.19%, +2.11%, +4.73% |
| 16,347 | 16 | -2.90% | -0.72%, -1.37%, -5.02% |
| 16,347 | 24 | +3.91% | +6.76%, +3.70%, +3.31% |

The second uniform screen also passes every numerical check. At 24 heads, its 8K median is
1.18% slower, with quartet reductions of -1.64%, +2.47% and -2.00%. Its 16K median improves
by only 0.27%, versus 3.91% in the grid screen. This does not establish a useful
request-level gain. `decision.json` records non-adoption; no whole-model evaluation was
launched.

While the second screen was queued, `build_engine.py` prepared an isolated prototype from
the frozen, fully qualified `integrated32` cache engine. It adds an independent
`attention_prefetch_alpha64` pipeline and selects it only for single 24-head requests with
at least 4,096 tokens. The original ordinary and cache attention bodies remain verbatim.
This experiment deliberately duplicates the kernel body to make that separation inspectable;
it is not a proposed final source layout.

The host prototype builds without warnings and contains the intended shader. Its shader JIT,
complete-model numerical checks and request-level timing have **not** run. It remains unused
after the second screen failed to support the timing lead. Cache adoption does not depend on
this experiment.

### Wider compensated products and larger query tiles

Combining each compensated attention product's three matrix multiplications into one wider
multiplication failed the speed screen. No engine code changed. The experiment preserves all
three terms, their order and operand precision; it does not shorten the context or change
the weights.

The frozen criterion required at least a 5% median reduction at both 512 and 2,048
threadgroups, with all four ABBA quartets faster. Neither QK nor PV passed.

| Product | Groups | Three products, ms | Wider product, ms | Latency reduction |
|---|---:|---:|---:|---:|
| QK | 128 | 0.1208 | 0.1234 | −2.2% |
| QK | 512 | 0.3789 | 0.6273 | −65.6% |
| QK | 2,048 | 0.9035 | 1.2613 | −39.6% |
| PV | 128 | 0.0483 | 0.0469 | 3.0% |
| PV | 512 | 0.3731 | 0.4977 | −33.4% |
| PV | 2,048 | 0.9629 | 1.0395 | −8.0% |

These are isolated GPU product timings on the M5 Max, not whole-attention or model timings.
Packing was performed on the CPU before timing, so the wider variant did not pay its
additional layout cost. The small cases are noisy; at 2,048 groups both products were slower
in every quartet. There is no evidence here to justify a full-attention prototype or another
task-evaluation run.

All 33,030,144 output elements matched the baseline bits. Finite-output and NaN-guard checks
passed, as did sampled float64 error checks. This is numerical evidence for these inputs,
not a task-quality result.

Source, binary and log hashes are recorded in
`golden/attention-contraction-20261005/manifest.json`, `result.json` and `decision.json`;
the criterion was recorded before the run in `screen-criteria.json`.

#### Larger query tiles

A separate complete-attention screen doubled the query tile from 32 to 64 rows, preserving
the 128-key blocks and all six compensated products. Scores and probabilities occupied 32
KiB of threadgroup memory; row scales and rescale votes used device scratch with explicit
barriers. This includes softmax, rescaling and scratch traffic, unlike the product-only
contraction experiment above.

All 307,353,600 output elements matched the 32-row kernel across 24 cases, including ragged
and short boundaries and both model head counts. Finite-output, float64 bounds and
output/scratch guards passed. The ten timed shapes from 1,024 to 16,347 tokens were slower
in every one of their four ABBA quartets. At 8K and 16K, latency increased by approximately
13%; shorter timed cases regressed by 16–21%.

This implementation fails the frozen 10% long-attention improvement criterion and is
rejected. It was not integrated or sent to task evaluation. Sources, binary, raw log,
criteria and decision are preserved in `golden/attention-query64-20261005/`. This result
concerns the tested scratch and execution layout; it does not establish a general limit for
larger tiles.

#### Shared-memory variant

A mapping probe captured every cooperative output coordinate for the 32-row and 64-row
products. Each SIMD group holds rows whose softmax state is produced by other groups, so the
existing ownership cannot exchange scales using only within-group shuffles. This is an
observation on the tested device and types, not a portable assumption about Metal's layout.

The next 64-row variant removed device scratch entirely. It placed row scales and votes in
the score buffer after all score reads, then overwrote them with probabilities after all
rescaling reads. Two extra shared-memory barriers protect those transitions. Products,
operands, softmax reduction order and input coverage were unchanged.

All 307,353,600 output elements again matched production bits across the 24 fixtures.
Float64 checks and guards passed. Every quartet of every timed shape was slower. At 8,072
and 16,347 tokens the regressions ranged from 6.9% to 11.5%; the shorter measured lengths
regressed by 11.6% to 15.8%. The frozen promotion criterion was a 10% improvement at both
long lengths with all quartets positive. This variant is rejected without model evaluation
or production changes.

The mapping is preserved in `golden/attention-rowmap-20261005/`; the shared-memory sources,
pinned criterion, samples and rejection are in `golden/attention-query64-shared-20261005/`.

#### Row-local rescaling

A separate variant retained production's 32-row geometry and removed the group-wide rescale
vote. Each thread skipped an accumulator multiplication when that row's scale equaled one,
preserving the nonidentity arithmetic and the barrier publishing probability planes and row
scales.

It also matched all 307,353,600 output elements and passed the float64 and guard checks. All
ten timing medians regressed. At 8,072 tokens it was 15.9% to 16.0% slower; at 16,347 tokens
it was 8.0% to 8.4% slower. Every long-input quartet regressed for both head counts. One
short quartet was faster, but its overall median regressed. This fails the predeclared 5%
improvement requirement at both long lengths and is rejected. The result supports retaining
the current uniform skip for this implementation; it does not separately measure branch,
register or memory costs.

`golden/attention-local-scale-20261005/` preserves the complete screen.

### Dispatch ordering

All three candidates are rejected: every one of the 36 timed configurations and all 216
paired quartets are slower than the unchanged kernel. All 1,348,331,520 elements in the
initial correctness comparisons match bit for bit. No production change or model evaluation
follows this screen.

#### Method

Only the mapping from dispatched threadgroups to query blocks and heads changes. The three
variants interleave all query heads, interleave heads sharing keys and values, or visit four
adjacent query blocks before switching between those heads. CPU enumeration verifies
one-to-one coverage, including ragged grids. The original attention arithmetic is copied
verbatim.

Both 16- and 24-head configurations cover 13 lengths from 1 to 16,347 tokens. Six lengths
from 1,024 upward are timed. Each configuration uses four warmups and six ABBA quartets,
with 12 measured samples per arm. Every timed output is also compared bit for bit outside
the GPU timestamp interval. Sampled float64, finite-output and output-poison checks pass.

The frozen advancement gate requires at least 5% median gain at 8,072 and 16,347 tokens for
both head counts, all paired quartets faster there, and no timed median regression above 2%.
No candidate passes.

| Heads | Tokens | Mapping | Baseline ms | Candidate ms | Slower by |
|---:|---:|---|---:|---:|---:|
| 16 | 8,072 | kv | 33.708 | 37.620 | 11.61% |
| 16 | 8,072 | tile4 | 33.641 | 38.565 | 14.64% |
| 16 | 8,072 | all | 33.682 | 38.936 | 15.60% |
| 16 | 16,347 | all | 172.597 | 186.282 | 7.93% |
| 16 | 16,347 | kv | 171.327 | 188.788 | 10.19% |
| 16 | 16,347 | tile4 | 172.630 | 186.826 | 8.22% |
| 24 | 8,072 | kv | 59.288 | 64.670 | 9.08% |
| 24 | 8,072 | tile4 | 59.372 | 65.273 | 9.94% |
| 24 | 8,072 | all | 63.144 | 69.852 | 10.62% |
| 24 | 16,347 | all | 303.797 | 326.853 | 7.59% |
| 24 | 16,347 | kv | 304.913 | 326.050 | 6.93% |
| 24 | 16,347 | tile4 | 299.421 | 322.509 | 7.71% |

These are isolated attention timings. They do not establish complete-request latency or a
hardware ceiling. Kernel arithmetic is unchanged, but the mapping adds coordinate arithmetic
as well as changing scheduling; the screen does not attribute the regression to either cause
independently.

#### Evidence

`golden/attention-dispatch-order-v2-20261005/` contains frozen criteria and hashes, original
and candidate shaders, the probe, raw samples, result and independent CPU verification. The
first attempt in `golden/attention-dispatch-order-20261005/` failed shader compilation
because two grid builtins had different vector widths. That attempt performed no benchmark;
its source and error remain preserved.

### Register softmax

The candidate preserves all 449,443,840 compared output elements but does not deliver a
qualifying speed improvement. It is rejected without model evaluation or production changes.

#### Hypothesis and implementation

The current [full-request
profile](long-request-timing.md#shader-timeline-and-category-profile) assigns about 18% of
long Flash GPU category time to attention. The existing attention kernel writes the
cooperative score tensor to threadgroup memory, reads those scores into per-thread arrays,
and reuses the shared allocation for high and residual probability planes.

A small GPU probe enumerates all 4,096 score coordinates and verifies unique, complete
ownership. On this M5 Max, each thread owns eight columns in each of four rows; a row spans
all four SIMD groups. Direct register access therefore requires a coordinated maximum
reduction. The first probe failed shader compilation because the MPP call requires named
lvalue operands. Its corrected version passes; both sources and logs remain saved. No model
inference was part of this layout probe.

The candidate masks and scales scores in cooperative registers. SIMD shuffles and a 512-byte
shared array combine per-row maxima. It writes the probability planes directly, then the
original row owners read those planes and sum them in exactly the original order. Matrix
products, block size, probability rounding, online normalization and output gating remain
unchanged.

This removes the shared score write, but introduces a maximum reduction across SIMD groups
and reads rounded probabilities for the denominator. It does not remove all shared-memory
traffic. The candidate assumes the measured cooperative layout. Production would need a
supported layout contract or a validated runtime guard and fallback; neither was implemented
for this screen.

The same device query reports dispatch-boundary counter sampling unsupported and
stage-boundary sampling supported. That limits direct timestamp insertion inside the
existing single compute encoder on this device.

#### Correctness and timing

The unchanged baseline and candidate use the same FP32-mantissa synthetic inputs and Safe
math compilation. Twenty-six shape/head checks span 1, 31, 32, 33, 63, 64, 65, 1,024, 2,048,
2,235, 8,072, 13,876 and 16,347 tokens with 16 or 24 query heads. Every compared output bit
matches. Existing sampled float64, finite-output and poisoned-output guard checks pass.

Each timed shape has four warmups followed by four ABBA quartets, giving eight measured
samples per arm. The gate was frozen before execution: at least 5% lower median
attention-kernel time at all three long lengths for both head counts, every long paired
quartet faster, and no more than 2% median regression at the shorter timed lengths.

| Query heads | Tokens | Baseline median | Candidate median | Reduction | Quartets faster |
|---|---:|---:|---:|---:|---:|
| 16 | 8,072 | 33.621 ms | 33.555 ms | 0.20% | 2/4 |
| 16 | 13,876 | 99.500 ms | 98.666 ms | 0.84% | 4/4 |
| 16 | 16,347 | 145.927 ms | 143.127 ms | 1.92% | 3/4 |
| 24 | 8,072 | 53.579 ms | 52.325 ms | 2.34% | 4/4 |
| 24 | 13,876 | 172.263 ms | 170.645 ms | 0.94% | 3/4 |
| 24 | 16,347 | 254.971 ms | 247.147 ms | 3.07% | 2/4 |

No long median reaches the 5% criterion. The 24-head, 2,048-token case regresses 4.10%, also
failing the gate. The largest short median gain is 7.51% at 1,024 tokens with 16 heads; it
does not establish a useful complete-request gain.

These timings cover isolated attention, including its internal softmax. They exclude input
preparation, projections and the CPU head. This screen does not qualify packed records,
prefix reuse, overflow handling or portability, and does not claim a whole-model speedup.
Those further checks were not started because the performance gate failed.

#### Evidence

`golden/attention-register-softmax-20261005/` contains the layout probe, source generator,
unchanged baseline, candidate, frozen criteria, raw checks/timings, hash manifest and
`decision.json`. The CPU audit independently reconstructs medians from saved samples and
verifies source, binary and log hashes. No model weights, requests, goldens or production
files changed.

## GEMM

### MLX and MPSGraph

MLX 0.32.3 was installed into an isolated, ignored directory. The comparison uses identical
FP16 activations and BF16 weights, with FP32 output. It runs Cleffa's MPP GEMM algorithm
through an MLX custom Metal kernel, so both alternatives include Python dispatch, evaluation
and synchronization. There are four warm-ups and eight timed calls per method, with order
reversed each round. These are matrix-operation timings, not complete-request timings.

`MLX mixed` receives the original 16-bit arrays. `MLX precast` receives their exact FP32
expansions, prepared before timing. The latter deliberately excludes conversion cost.

| Shape: tokens × input width × output width | Cleffa MPP | MLX mixed | MLX precast |
|---|---:|---:|---:|
| 346 × 4,096 × 24,576 | 1.44 ms | 3.52 ms | 2.36 ms |
| 4,510 × 5,120 × 34,816 | 25.92 ms | 42.71 ms | 40.04 ms |

All output bits match across methods in these two cases. Sixty-four sampled outputs also
pass the float64 dot-product check; maximum error divided by sum of absolute products is
7.19e-8 and 4.10e-8 respectively. The initial eight-shape sweep also covered down
projections and found no MLX win. Some down-projection reductions differed in bits, while
passing the sampled oracle.

MLX's default TF32 mode was enabled. Its reduced operand precision does not discard bits
from these already-rounded FP16/BF16 inputs. This conclusion does **not** extend to
arbitrary FP32 activations, including the head or attention. See MLX's [precision
documentation](https://ml-explore.github.io/mlx/build/html/usage/precision.html).

An unprofiled Objective-C MPSGraph comparison at 4,510 × 5,120 × 34,816 measured 42.33 ms,
against 25.00 ms for MPP in that process. Graph operands were expanded to FP32 and the FP19
operand mode was allowed; outputs matched bit-for-bit and passed the same sampled oracle.
Command-buffer completion wall time was used because MPSGraph can split its work across
command buffers. A separate Instruments run was slower and is not used for this comparison.
That trace did not capture neural-accelerator utilization, so it is not evidence of peak
hardware use.

Reproduce the MLX comparison with the tracked benchmark:

```sh
uv pip install --python .venv/bin/python --target golden/mlx-env mlx==0.32.3
PYTHONPATH=golden/mlx-env .venv/bin/python -B bench/compare_mlx.py \
    4510 5120 34816 --output golden/mlx-comparison.json
```

NumPy is also required; it is already part of the reference environment. The benchmark
records the device, MLX version, TF32 setting, its own source hash, individual samples,
output dtype, bit differences and sampled numerical errors. The tracked version was run on
both table shapes and the ragged shape 33 × 63 × 65.

### Persistent scheduling and fixed dimensions

Further probes live under `golden/perf-persistent-20261004/`. A persistent GEMM assigns a
fixed pool of 80, 160, 320 or 640 threadgroups to multiple output tiles. Row-major,
column-major and four-row-block traversals were tested with 64x128 and 32x256 tiles. At
4,510 x 5,120 x 34,816, the best persistent variant was about 25.54 ms versus 24.75 ms for
the existing dispatch. Some down-projection variants improved by only a few percent. All
tested outputs matched bitwise and passed sampled float64 and output-guard checks.

Specializing the contraction dimension, tensor extents and slices at compile time did not
materially improve the expansion either. These probes covered the 27B expansion and both
models' down projections. The simpler dynamic 32x256 tile was competitive with the
specialized down-projection variants, so it was tested in complete inference.

### Larger threadgroups

`golden/perf-groups-20261004/` tests eleven tile/threadgroup combinations, including 64x256
and 128x128 tiles with eight SIMDgroups, and 128x256, 256x128 and 64x512 tiles with sixteen.
Each variant alternates with the existing 64x128/four-group baseline. Two warm-ups precede
six measured calls per path. All outputs match bitwise, remain finite after poisoning, pass
64 sampled float64 checks, and preserve output guards on ragged 33x63x65 and both large
shapes.

At 4,510x5,120x34,816, the larger candidates offer no useful gain: the 64x256/eight-group
variant takes 25.856 ms versus its paired baseline's 25.987 ms, while the remaining larger
variants are slower. For 4,510x17,408x5,120, the already-retained 32x256/four-group tile
remains competitive with or faster than the new candidates. No dispatch changes were made.

The model dimensions independently give about 220 trillion operations for the 27B backbone's
linear layers at 4,510 tokens. Executing that work in 750 ms would require about 293
TFLOP/s, before attention products, recurrence and the joint-schema head. This describes the
target's arithmetic requirement; it does not establish this machine's maximum throughput.

### Compiler and pipeline hints

`golden/perf-pipeline-20261004/` tests explicit threadgroup limits, execution-width
multiples, required threadgroup dimensions and the macOS 27 persistent-kernel optimization
hint. The dispatch uses 128 threads throughout. The two large shapes use the current
production tiles: 64x128 for the expansion and 32x256 for the down projection. Each variant
alternates with the baseline after ten initial warm-ups; two paired warm-ups precede six
measured calls per path. The reported values here are means, not medians.

Threadgroup hints leave the 4,510x5,120x34,816 expansion near 25 ms, with no material gain.
The down-projection results vary in both directions, including about 5% movement in the
unchanged descriptor control. They do not support a production change.

The persistent hint helps the experimental 160-group scheduler on the expansion, but it
still takes 26.633 ms against a paired production baseline of 25.466 ms. The 320-group
variants are slower. Applying the hint directly to the production kernel changes the
expansion from 25.748 to 25.354 ms and the down projection from 13.042 to 13.008 ms. The
down-projection persistent variants include noisy small gains and regressions. These
measurements do not establish a dependable improvement worth integrating.

A separate probe compiles only GEMMs with Safe, Relaxed and Fast math modes. The expansion
takes 25.486 ms in Relaxed mode against 25.511 ms for its Safe baseline, and 25.272 ms in
Fast mode against 25.518 ms. The down projection changes from 13.149 to 12.885 ms in Relaxed
mode, but Fast regresses slightly, from 13.111 to 13.180 ms. These probes do not establish a
material gain; production retains Safe math.

All tested large-shape outputs match their baseline bits and pass 64 sampled float64 dot
products. The persistent and math-mode probes also pass the ragged 33x63x65 case. Each final
correctness invocation begins with NaN-filled output and guards, checks every output is
finite, and verifies the guard remains untouched. No engine source or binary changed.

The reproducible sources are `gemm-pipeline.m`, `persistent-hints.m` with
`persistent.metal`, and `gemm-mathmode.m` in that directory. Compile each Objective-C source
with `clang -O2 -fobjc-arc SOURCE -framework Metal -framework Foundation -o BINARY`, then
run `BINARY T K N` from the repository root. Corresponding logs record each candidate and
its paired baseline. `summary.json` records file hashes and results.

### Native tensor layouts

A separate probe compares inline buffer views with buffer-backed `MTLTensor` handles and
device-allocated tensors whose layout Metal chooses. It copies the original BF16 bits into
the device tensor and checks exact readback. Activations remain FP16 and outputs FP32. Each
method alternates with its baseline after ten initial warm-ups, with two paired warm-ups and
eight measured calls. Values below are means for the device-allocated variant:

| Tokens x input x output | Paired inline baseline | Device tensor |
|---|---:|---:|
| 1,382 x 5,120 x 34,816 | 7.867 ms | 7.852 ms |
| 1,382 x 17,408 x 5,120 | 4.003 ms | 4.229 ms |
| 4,510 x 5,120 x 34,816 | 25.470 ms | 25.663 ms |
| 4,510 x 17,408 x 5,120 | 13.112 ms | 12.919 ms |

Neither tensor-handle approach establishes a material improvement. All output bits match the
corresponding inline kernel on these shapes and the ragged 33x63x65 case. Outputs remain
finite after NaN initialization, guards remain untouched, and 64 float64 samples per case
pass. These are isolated GEMMs, not a validated replacement engine. No production source or
binary changed. Sources, logs, exact hashes and the profile summary are under
`golden/perf-medium-20261004/`.

### Transposed weight layout

`golden/gemm-transpose-20261004/` stores the same BF16 weight bits as K-by-N rather than
N-by-K and disables the right-operand transpose in MPP. Activations remain FP16 and outputs
FP32. CPU transposition and input preparation are excluded from timings. This is a
standalone layout probe, not a changed model format or production engine.

All 20 shapes pass both multiply and residual-accumulate checks. They cover ragged
dimensions, row-tile boundaries, packed offsets of 31 rows, exact weight/output bits,
poisoned guards, repeated accumulation and 64 sampled float64 dot products per mode. The
initial compiler failure came from passing temporary tensor views to MPP's lvalue
parameters; named views fix it. The failed source and log remain in `compile-v1-*`.

Each mode uses two paired warm-ups followed by twelve measured calls per arm in six ABBA
quartets. Timing leaves residual buffers on the GPU between accumulating calls. The table
selects modes corresponding to the projections' normal use:

| T x K x N | Accumulate | Existing layout ms | Transposed layout ms | Reduction |
|---|---|---:|---:|---:|
| 1,382 x 4,096 x 24,576 | no | 4.407 | 4.534 | -2.88% |
| 1,382 x 12,288 x 4,096 | yes | 2.337 | 2.441 | -4.43% |
| 1,382 x 5,120 x 34,816 | no | 8.324 | 8.175 | +1.79% |
| 1,382 x 17,408 x 5,120 | yes | 4.177 | 4.401 | -5.37% |
| 1,382 x 5,120 x 16,480 | no | 3.752 | 3.815 | -1.67% |
| 4,510 x 17,408 x 5,120 | yes | 15.084 | 15.747 | -4.40% |
| 8,072 x 4,096 x 24,576 | no | 29.855 | 30.191 | -1.12% |
| 8,072 x 12,288 x 4,096 | yes | 15.926 | 16.363 | -2.74% |
| 16,347 x 5,120 x 34,816 | no | 114.908 | 116.251 | -1.17% |
| 16,347 x 17,408 x 5,120 | yes | 62.636 | 63.882 | -1.99% |

Most shapes are slower with transposed weights. The 1,382-token 27B expansion improves by
1.79% in this screen, with five of six quartets favoring it; the corresponding 16,347 shape
is 1.17% slower. The result does not support a general layout conversion, so none is
retained. `analysis.json` contains every raw pair and quartet, and `manifest.json` pins the
corrected sources and binary.

### FP16 weight operands

The isolated probe did not show a large throughput difference between BF16 and FP16 weight
operands. Model weights and production code remain unchanged.

The kernel is the qualified engine's non-accumulating 64×128 `gemm_x`, with the weight type
made a template parameter. Activations are FP16 in both arms. Every synthetic weight is
exactly representable in both formats; the host verifies that conversion preserves its
value. There is no model-weight conversion or quantization experiment here.

Each shape used four warm calls followed by four ABBA quartets, giving eight GPU timings per
arm. All output bits matched across types, every output was finite, guards were intact, and
both arms passed the sampled float64 bound.

| Tokens | K → N | BF16 weights, median GPU ms | FP16 weights, median GPU ms | Observed reduction |
|---:|---:|---:|---:|---:|
| 2,048 | 4,096 → 24,576 | 6.3052 | 6.3051 | 0.00% |
| 2,048 | 5,120 → 34,816 | 11.2576 | 11.1513 | 0.94% |
| 8,192 | 4,096 → 24,576 | 26.5130 | 26.0827 | 1.62% |
| 8,192 | 5,120 → 34,816 | 48.1628 | 47.9020 | 0.54% |

These are one-run, isolated-kernel observations with visible timing variation. They do not
establish repeatable whole-model improvements or a hardware ceiling. The probe covers
expansion GEMMs only, with one tile and no residual accumulation; it uses the Metal
compiler's default math options. Before pursuing a conversion, an improvement would need to
survive the production dispatch and compiler settings, and every real weight value and
full-model result would need validation.

The current evidence does not justify implementing a model-weight conversion for this route.
Exact conversion for the full BF16 exponent range would require more than casting all
weights to FP16; no range analysis or scaling scheme has been implemented.

Evidence: `golden/weight-type-20261005/result.json`, `probe.jsonl`, `probe.log`,
`manifest.json`, `probe.m`, and `probe.metal`.

### CPU/GPU column split

Computing a small share of matrix output columns on the CPU did not establish a useful speed
gain. None of 28 cases passed the frozen threshold of at least 5% lower median wall time
with improvement in all six paired timing quartets. The experiment is rejected. No
production source, binary or dispatch changed.

#### Question and method

The article profile attributes 87.5% of its measured GPU categories to matrix operations.
Earlier counters show high neural-accelerator activity on selected shapes. This experiment
asks whether the CPU can contribute enough independent matrix work to shorten those
operations despite limited measured GPU headroom.

The GPU uses snapshots of the existing production GEMM kernels and computes a prefix of
output columns. The CPU computes the remaining columns using Accelerate's [single-precision
matrix
product](https://developer.apple.com/documentation/accelerate/cblas_sgemm(_:_:_:_:_:_:_:_:_:_:_:_:_:_:)).
Its input values are exact FP32 expansions of the same FP16 activations and BF16 weights.
CPU operand conversion, joining the split outputs and synchronization with surrounding model
layers are all excluded. This favors the candidate and makes the result a feasibility
screen, not an engine benchmark.

CPU shares target 1%, 2%, 4% and 8% of columns, rounded to the production column tile.
Duplicate shares are skipped. Both automatic and single-threaded BLAS modes run on the same
calling thread through Apple's [threading
API](https://developer.apple.com/documentation/accelerate/blassetthreading(_:)). The thread
submits the GPU command, computes the CPU suffix, then waits for GPU completion. Total wall
time includes submission and completion; GPU and CPU intervals are recorded separately.

Every case has one warm-up ABBA quartet and six measured ABBA quartets. The baseline
computes the complete matrix on the GPU. The candidate computes every output element exactly
once across CPU and GPU. The 8K down projection uses residual accumulation; expansion cases
use multiplication. Synthetic inputs have the model's projection dimensions.

#### Results

The table selects the best median candidate for each shape after measurement, so these are
favorable selections, not independently confirmed gains. Each baseline is paired with the
candidate in its row.

| Tokens x input width x output width | CPU columns / threading | Full GPU ms | Split ms | Time reduction | Faster quartets |
|---|---|---:|---:|---:|---:|
| 1,300 x 5,120 x 34,816 | 640 / single | 7.538 | 7.395 | 1.90% | 4/6 |
| 10,331 x 5,120 x 34,816 | 640 / single | 68.588 | 68.441 | 0.21% | 2/6 |
| 8,072 x 17,408 x 5,120 | 256 / automatic | 25.216 | 42.888 | -70.08% | 0/6 |
| 16,347 x 5,120 x 34,816 | 384 / automatic | 109.339 | 108.197 | 1.04% | 3/6 |

The down projection's CPU portion takes 42.881 ms while its reduced GPU portion takes 22.764
ms. Moving 5% of columns to the CPU lengthens that case. Smaller CPU shares in the
expansions leave the GPU dominant and fail to produce a consistent gain. These measurements
reject this split and backend under the tested conditions; they do not establish a ceiling
for every heterogeneous algorithm or prove that no further GPU optimization exists.

#### Correctness and decision

All 28 cases pass finite-output, NaN-guard and sampled float64 checks. Across the cases,
7,651,915,264 GPU output comparisons match the full GPU computation bit for bit. The CPU
portion contains 306,055,680 checked finite elements; 297,849,972 differ in bits from the
GPU result. The maximum scaled float64 error over 3,584 sampled outputs is 1.68e-7, below
the frozen 2e-6 bound.

Passing these kernel checks would not establish model quality. CPU reduction rounding
differs, so adoption would require model parity, batch invariance and labeled evaluation in
addition to a realistic-cost speed test. The optimistic performance screen fails first, so
none of that further integration work runs.

Artifacts are under `golden/gemm-cpu-share-20261005/`: `bench.m`, the unchanged GEMM shader
snapshot, `run.py`, `freeze.json`, 784 timing samples in `samples.jsonl`, `result.json` and
`review.json`.

### Concurrent GEMM and recurrence

Requesting concurrent execution of independent GEMM and DeltaNet scan kernels did not
establish enough benefit to justify a new batch scheduler. None of 16 fixture/order cases
passed the frozen screen of at least 5% lower median GPU time and improvement in all six
paired timing quartets. Production scheduling, source and binaries remain unchanged.

#### Motivation and scope

The real article profile puts 87.5% of measured GPU categories in matrix work and 5.7% in
the recurrence scan. Both run serially in the current compute pass. An independent record
could supply recurrence work while another record uses the tensor units for a matrix
operation. Unlike the rejected CPU split, this would preserve each production kernel's
arithmetic.

The test compiles an unchanged snapshot of `metal/clef.metal`. One task runs the production
FP16/BF16 expansion GEMM; the other runs the production FP32 `gdn_scan_8` on separate
buffers. Each task retains its complete input and output. Record lengths stay below the 27B
chunked-recurrence boundary, so the chosen scan matches production. Packed token counts
select the existing GEMM tile and grouped order.

The baseline encodes GEMM then scan in a serial compute pass. Two candidates use [Metal's
concurrent dispatch
mode](https://developer.apple.com/documentation/metal/mtldispatchtype/concurrent), one in
each command order. Disjoint buffers eliminate inter-task dependencies. This requests
concurrency; the test does not collect a per-kernel execution trace and does not prove
simultaneous physical execution. The measured result is the time for the complete pair under
each encoding policy.

The fixture set contains both models' projection dimensions and value-head counts, with one,
four or eight 1,300-token records and one 2,235-token record. These are synthetic operands.
An actual scheduler would also have to arrange dependencies across all layers and account
for extra activation memory. This screen grants the pair ideal independence before
attempting that integration.

#### Measurements

Each fixture/order uses one warm-up ABBA quartet, then six measured ABBA quartets. The table
selects the faster candidate median for each fixture after measurement; these favorable
selections are not separate confirmation runs. Every baseline is paired with its row's
candidate.

| Model | Records x tokens | Command order in concurrent pass | Serial pair ms | Concurrent pair ms | Reduction | Faster quartets |
|---|---|---|---:|---:|---:|---:|
| Flash | 1 x 1,300 | GEMM first | 5.109 | 5.289 | -3.53% | 1/6 |
| Flash | 4 x 1,300 | Scan first | 20.126 | 20.005 | 0.60% | 4/6 |
| Flash | 8 x 1,300 | GEMM first | 40.481 | 40.289 | 0.48% | 4/6 |
| Flash | 1 x 2,235 | GEMM first | 8.551 | 9.023 | -5.52% | 1/6 |
| 27B | 1 x 1,300 | GEMM first | 8.849 | 8.710 | 1.57% | 6/6 |
| 27B | 4 x 1,300 | GEMM first | 34.324 | 34.358 | -0.10% | 3/6 |
| 27B | 8 x 1,300 | Scan first | 78.698 | 76.226 | 3.14% | 4/6 |
| 27B | 1 x 2,235 | GEMM first | 15.058 | 15.111 | -0.35% | 1/6 |

The single-record 27B case has a repeatable small pair-level benefit in this screen, but its
0.139 ms median difference is not an end-to-end model gain. The larger packed candidate has
mixed paired results. Neither supports the additional complexity of a scheduler under the
frozen criterion. These results do not rule out every scheduling design or establish a
hardware ceiling.

#### Verification and evidence

All 5,329,633,280 compared output elements match their serial reference bit for bit, across
checks before and after timing. Every fixture passes finite-output and NaN-guard checks. The
baseline also passes sampled float64 checks, with maximum scaled GEMM error 7.29e-8 and
maximum absolute scan error 9.90e-6, within the existing kernel bounds. No model evaluation
follows a failed speed screen.

The requested upstream `qwen-perf` branch was also checked live with `git ls-remote`. Its
head remains
[`b7093a23410620fccae796188332f69028724b7c`](https://github.com/kernelpool/ds4/tree/b7093a23410620fccae796188332f69028724b7c),
the commit covered by the [existing review](#simd-group-grouping-in-the-sequential-scan).
There is no new upstream patch in that branch to apply.

Artifacts are in `golden/gemm-gdn-overlap-20261005/`: `upstream.json`, the exact shader
snapshot, `bench.m`, `run.py`, `freeze.json`, `cases.jsonl`, `result.json` and
`review.json`. The raw file includes 448 timing samples across eight fixtures and two
concurrent orders.

### Other tile orders

Of six tile-assignment orders screened on 18 shapes, only interleaving four tile rows on
long 27B expansions was retained. Full column-major traversal was slower on every large
shape and larger groups severely regressed long down-projections; see
[grouped expansion tile order](performance-history.md#grouped-27b-expansion-tile-order-2026-10-05).

### 16-row FP16 projection tiles, 2026-10-14

A private probe reduced FP16 projection tiles from 32 to 16 rows to cut padding on short Flash
requests. At the tested 146x4096x24576 shape, 16x128 and 16x256 tiles kept exact tile/packed
output, sampled FP64 error bounds and NaN guards, but showed no clear isolated win. Forcing
16x128 tiles in complete Flash inference (one resident engine, two warm and eight measured calls
per arm in four ABBA quartets) kept every logit exact and regressed every quartet:

| Flash tokens | Existing dispatch, median ms | 16x128, median ms | Latency increase |
|---|---:|---:|---:|
| 146 | 70.048 | 80.797 | 15.3% |
| 300 | 100.606 | 123.750 | 23.0% |
| 594 | 185.326 | 246.290 | 32.9% |

Rejected; no tile or shader change was retained. 27B was not run through the full-request probe.
Evidence: `golden/dense-perf-20261014/tile16/`.

### Static outer extents for FP32 interior tiles, 2026-10-14

A private shader specialization gave only complete interior tiles of the FP32 32x128 multiply
and accumulate kernels static outer tensor extents. K stayed dynamic; edge tiles, the FP16/BF16
paths, FP32 16-row tiles, arithmetic and dispatch geometry were unchanged. All 1,452 paired
cases were bit-identical across 93,880,122 output elements, covering both dense geometries,
every packed row shift, ragged K, cancellation, nonzero residuals and poisoned tails.

Five of six frozen single-GEMM gates failed. Median candidate/baseline ratios for T146/300/2235
were 1.002078/0.993634/1.068260 at K4096,N4096 and 1.022428/1.001913/1.002906 at K6144,N5120,
each from two warmups and six ABBA quartets per arm. Advancement required at least 3%
improvement and five faster quartets at both longer lengths for each geometry, with at most 2%
extra at T146. Flash T2235 varied substantially, so the screen does not establish a precise
slowdown. No complete-request test followed. Evidence:
`golden/dense-fp32-static-slice-20261014/`.

## FFN

### Fused gate/up projection with SwiGLU

The prototype holds gate and up projections in cooperative registers, then computes the
existing SwiGLU expression and `act16` conversion without writing the FP32 projection
buffer. Four tile sizes passed exact output, overflow-flag and NaN-guard checks on the two
model widths at 346 and 4,510 tokens, including BF16 output and a lowered FP16 overflow
limit.

The apparent short-input microbenchmark gain did not survive complete inference. A
same-process ABBA comparison, with four warm-ups and six samples per mode, produced these
medians:

| Model / tokens | Current | Fused FFN |
|---|---:|---:|
| Flash / 346 | 112.25 ms | 114.05 ms |
| Flash / 600 | 177.35 ms | 180.60 ms |
| Flash / 4,510 | 1,576.65 ms | 1,612.65 ms |
| 27B / 346 | 417.20 ms | 431.80 ms |
| 27B / 600 | 707.35 ms | 740.15 ms |
| 27B / 4,510 | 5,523.20 ms | 6,055.00 ms |

Every logit matched, but every tested complete-request median was worse. This implementation
was rejected. This does not rule out a different fusion design.

## DeltaNet

### Decay precompute and loop unrolling

A second prototype precomputes `exp(g)` once per token and value head in `gdn_prep` instead
of evaluating it repeatedly in `gdn_scan`. Explicit scan-loop unrolling was tested
separately and together with that change. All four modes returned identical logits on both
models at 4,510 tokens. Request-level differences varied in sign across models and variants.

A warm isolated Flash scan at 4,510 tokens measured 3.3286 ms for the current LPC=8 kernel,
3.2973 ms with precomputed decay, 3.3301 ms with unrolling, and 3.2985 ms with both. Every
FP32 output bit and output guard matched for LPC=2, 4 and 8. The approximately 0.03 ms
saving per scan would total less than 1 ms across Flash's 24 DeltaNet layers. It does not
explain the roughly 2% difference in the first request-level trial. No scan change was
retained.

### Lane counts and other block sizes

While developing the chunked recurrence, sixteen or thirty-two lanes per value column in
the sequential scan, an eight-token SIMDgroup-matrix block and a 64-token MPP block were
all slower than the retained 32-token FP32 block; see
[chunked DeltaNet](performance-history.md#chunked-27b-deltanet-2026-10-04).

### SIMD-group grouping in the sequential scan

The reviewed snapshot is [kernelpool/ds4 at
b7093a2](https://github.com/kernelpool/ds4/tree/b7093a23410620fccae796188332f69028724b7c).
The comparison base is `0aaea5a238fb41a35106a551e73c8409dfb751ac`. The branch changes
Qwen3.8 inference, including quantized MoE kernels and speculative verification. Its
published token rates are not an equivalent benchmark for dense BF16 Clef classification.

#### Transferable ideas

The [DeltaNet
dispatch](https://github.com/kernelpool/ds4/blob/b7093a23410620fccae796188332f69028724b7c/ds4_metal.m#L49022)
can group several independent SIMD groups into a threadgroup. Each group retains its own
recurrence and reductions. The upstream default uses four groups for one large M3 Ultra
prefill shape, with one group elsewhere. That tuning needs measurement on the M5 Max and
Clef's different lane assignment.

The [command submission
change](https://github.com/kernelpool/ds4/commit/29eed3655b9f6e8bc2f950b774dca3eda0ae9450)
submits work every two layers to overlap GPU execution with host encoding. Cleffa currently
encodes ordinary inference into one command buffer. Its warm short-request traces show
approximately 1.4 ms of host encoding, so overlapping encoding would save at most that
component on those requests, before accounting for additional submissions. This is a
candidate for measurement, not an established Cleffa speedup.

The [completion-wait
change](https://github.com/kernelpool/ds4/commit/967ed75b722f6c02aa58964076ba5f01842f111d)
polls command-buffer status for up to 100 ms before blocking. Its commit describes an
approximately 0.3 ms decode wake-up cost. Cleffa waits once per ordinary forward pass, so
that observation does not imply a large full-input gain here. Polling also occupies a CPU
core; it has not been adopted.

#### Kernels that require a different implementation

The [fused DeltaNet
kernel](https://github.com/kernelpool/ds4/blob/b7093a23410620fccae796188332f69028724b7c/metal/qwen4.metal#L1241)
combines normalization, gates, recurrence and output normalization for at most 16 tokens.
Its threadgroup arrays have that fixed limit. It uses `h % Hk` for key-head mapping and a
sigmoid output gate. Clef uses `h / (Hv / Hk)` and SiLU. Its recurrence also distributes the
state and reduces partial products differently from Cleffa. Copying the kernel would change
the model's computation; a full-input implementation would need its own design and numerical
qualification.

The new Q8 multi-row matrix-vector paths preserve the one-token reduction order during
speculative verification. Cleffa performs full-input matrix multiplication with exact BF16
weights and has no autoregressive verification loop. The quantized expert tile, router,
hyperconnection and sparse-index changes do not directly fit its architecture.

#### Cleffa experiment

`golden/ds4-qwen-perf-20261004/` contains the pinned upstream clone, an isolated shader
generator and a benchmark derived from Cleffa's existing recurrence. The candidate changes
only the assignment of existing LPC=8 SIMD groups to threadgroups. It compares the original
kernel with groups of 1, 2, 4, 8 and 16, using four warm-ups and eight measured dispatches
per mode in alternating order.

The fixtures cover 248, 1,382, 4,510 and 8,072 tokens, a mixed 4,510 + 346-token batch, and
eight 1,382-token records, each with 32 and 48 value heads. Every dispatch starts with
poisoned output, checks exact bits against the original and checks prefix/tail guards. A
sampled float64 recurrence checks each record with zero, weak, strong and overflowing decay
inputs. Timing excludes host validation and compilation.

This experiment leaves production kernels unchanged. The 27B already uses a qualified
chunked recurrence on records of at least 4,096 tokens, so its long sequential-kernel
measurements alone would not establish a gain over production.

All twelve fixtures pass exact-bit, poison-guard and sampled float64 checks. The largest
sampled absolute difference from float64 is below `1.83e-5`, within the existing `2e-5 * (1
+ abs(reference))` recurrence bound. Selected median dispatch times are:

| Value heads | Records | Groups per threadgroup | Original | Grouped |
|---|---|---:|---:|---:|
| 32 | 4,510 tokens | 2 | 3.4523 ms | 3.2447 ms |
| 32 | 8,072 tokens | 2 | 6.4253 ms | 6.1058 ms |
| 48 | 1,382 tokens | 4 | 1.4529 ms | 1.3935 ms |
| 32 | 8 × 1,382 tokens | 4 | 8.7871 ms | 7.7521 ms |
| 48 | 8 × 1,382 tokens | 4 | 13.9771 ms | 12.7435 ms |

Larger groups are not consistently faster. Four groups slow the mixed 4,510 + 346-token case
from 4.6861 to 5.1448 ms with 32 heads and from 5.9553 to 7.6252 ms with 48 heads. Eight and
sixteen groups also regress several single-record shapes. The full sweep is in
`groups.jsonl` and `groups-summary.json`; these results measure the sequential scan alone,
excluding the model's projections, attention and classification head.

The first whole-engine trial uses one loaded model, two warm-ups per arm followed by two
ABBA quartets, or four measured calls per arm. All raw logits match exactly on every
repeated and candidate call. Timing includes inference and the CPU head and excludes model
loading, request encoding and HTTP.

| Model | Records | Original median | Grouped median | Median time reduction |
|---|---|---:|---:|---:|
| Flash | 4,510 tokens | 1,462.3 ms | 1,468.5 ms | -0.42% |
| Flash | 8 × 1,382 tokens | 3,963.8 ms | 3,959.0 ms | 0.12% |
| 27B | 1,382 tokens | 1,588.3 ms | 1,575.0 ms | 0.83% |
| 27B | 8 × 1,382 tokens | 13,698.9 ms | 13,637.2 ms | 0.45% |

Paired quartet means are more informative than the overall medians under clock drift. The
single Flash quartets disagree on direction; both packed Flash quartets slightly favor the
original. The packed 27B quartets also disagree. Only single-record 27B favors grouping in
both quartets, by 0.33% and 1.38%, requiring another measurement before retention. Samples
and paired differences are in `engine-summary.json`.

As a scale check, the 1,382-token isolated scan difference multiplied by the 27B's 48
DeltaNet layers is approximately 2.9 ms. This estimate assumes the isolated timing carries
over unchanged. The observed whole-request difference is 13.3 ms, which is another reason to
verify it in a separate run rather than attribute all of it to grouping.

#### Separate repeat and decision

A second 27B run checks three single-record lengths with two warm-ups per arm and four ABBA
quartets, or eight measured calls per arm. All raw logits remain identical.

| Tokens | Original median | Four-group median | Median time reduction | Paired quartet reductions |
|---|---:|---:|---:|---|
| 594 | 618.3 ms | 624.3 ms | -0.97% | +0.89%, -2.52%, -2.97%, +4.68% |
| 1,382 | 1,484.9 ms | 1,473.4 ms | 0.77% | +2.02%, -1.65%, +1.34%, +3.22% |
| 2,235 | 2,462.6 ms | 2,415.6 ms | 1.91% | +3.75%, +1.46%, +0.08%, +0.60% |

The original and grouped calls share a loaded model and alternate within each quartet.
Positive paired values favor grouping. The mixed signs and large variation at the shorter
lengths prevent a general speedup claim. The 2,235-token result favors grouping in all four
quartets and is a candidate for further testing around that length. It does not qualify an
entire dispatch range, particularly near the existing 4,096-token chunked-scan boundary.
`repeat-summary.json` contains every sample and paired result.

No production shader or dispatch change from this branch review has been retained. Broad
enablement is rejected; a narrower 27B prefill dispatch remains unqualified. Across the two
engine runs, all 108 forward calls preserve raw logits against their corresponding baseline.
Those targeted checks do not replace a complete corpus, poison, overflow and
batch-invariance qualification if a candidate is later retained.

The initial review used the validated keep-warm build in
`golden/perf-idle-residency-20261004/production-validation.json`.

#### Medium-context follow-up

A later trial uses the qualified parallel CPU head and limits the grouped scan to a single
27B record with 2,048–4,095 tokens. Six synthetic checkout fixtures bracket that range. Four
warm-ups and three ABBA quartets per fixture give 96 calls, all with raw logits identical
across arms and repeats. The 2,047- and 4,096-token controls use the same kernel in both
arms.

| Full tokens | Original median | Grouped median | Paired quartet reductions |
|---:|---:|---:|---|
| 2,047, control | 2,221.471 ms | 2,195.437 ms | 2.88%, 0.84%, -1.73% |
| 2,048 | 2,214.701 ms | 2,197.903 ms | -3.55%, 1.83%, 2.43% |
| 2,235 | 2,425.748 ms | 2,427.573 ms | 0.79%, -0.45%, 0.74% |
| 3,072 | 3,425.570 ms | 3,406.545 ms | 0.16%, 2.70%, 1.29% |
| 4,095 | 4,823.038 ms | 4,781.864 ms | 0.96%, -0.23%, 1.39% |
| 4,096, control | 4,642.306 ms | 4,633.309 ms | 0.18%, 0.02%, -0.12% |

This does not reproduce a reliable 2,235-token benefit or establish the proposed range. The
grouping change remains isolated. Sources, exact encoded lengths, samples and summaries are
in `golden/gdn-medium-20261004/`.

### Packed chunked scans

The article workload's recurrent scan accounts for about 6% of its serialized GPU profile.
Production processes short records together with its sequential recurrence. Its existing
chunked recurrence is selected only for individual 27B records of at least 4,096 tokens, and
processes those records serially.

This isolated experiment tests whether dispatching chunked scans for multiple records
together makes chunking worthwhile on shorter records. It copies the existing FP32 chunk
arithmetic and changes record and scratch addressing. Each record retains its own 32-token
block boundaries and recurrent state. The new prepass skips uniform threadgroups beyond a
shorter record's final block.

No production code changes. The candidate is rejected at the kernel screen, so no model
quality evaluation or engine integration is started.

#### Correctness

Twelve cases cover both model head counts with a single 1,300-token record, the eight
observed article lengths, eight 512-token records, eight 2,048-token records, a mixed
short/4,097-token batch, and tails of 1, 7, 31, 32, 33, 63, 64 and 65 tokens. The first
packed record begins at offset 31.

Both packed value-tile variants match the existing record-by-record chunked implementation
over all 749,568,000 compared FP32 outputs. Output and scratch guards stay poisoned, and all
active output/scratch elements are finite. All four modes also pass sampled float64
recurrence checks, including beta 0 and 1, weak/strong decay, -1e8 decay and overflowing
negative decay sums. The largest sampled absolute error across modes is 1.60e-5, inside the
existing `2e-5 * (1 + abs(reference))` bound.

These exact comparisons concern packed versus serial **chunked** arithmetic. Chunking is not
bit-identical to the production **sequential** recurrence for short records. A retained
change would therefore require model accuracy and batch-invariance validation. Kernel checks
alone do not establish task quality.

#### Performance

Each candidate has its own sequential control, two paired warm-ups, and six ABBA quartets
with twelve measured calls per arm. Times are GPU command-buffer intervals including
preprocessing. They exclude model projections and the head.

Selected 27B-head-count results:

| Records | Packed value columns | Sequential median | Candidate median | Reduction | Quartets faster |
|---|---:|---:|---:|---:|---:|
| 1 x 1,300 | 16 | 1.363 ms | 1.396 ms | -2.44% | 0/6 |
| 1 x 1,300 | 32 | 1.364 ms | 1.495 ms | -9.64% | 0/6 |
| 8 actual article lengths | 16 | 10.151 ms | 9.544 ms | 5.97% | 6/6 |
| 8 actual article lengths | 32 | 10.149 ms | 9.430 ms | 7.09% | 6/6 |
| 8 x 512 | 16 | 3.982 ms | 3.614 ms | 9.25% | 6/6 |
| 8 x 2,048 | 32 | 16.292 ms | 14.945 ms | 8.27% | 6/6 |

The eight-article shape totals 10,331 tokens and needs another 796.5 MiB of scratch for the
27B head count, or 531.0 MiB for Flash. Eight 2,048-token records need 1,251 MiB and 834 MiB
respectively. Flash's article scan improves by about 6%, but its single-record scan also
regresses. Uneven batches can regress too; long-record timings against the sequential
control do not represent the current 27B production selection.

Neither packed variant passes the frozen next-stage criterion: at least 15% less scan time
on the article and eight-2,048-token shapes, plus all six paired quartets faster on both the
single-1,300 and article shapes. At the measured article profile's scan share, a 7.09% scan
reduction would save only about 0.4% of total GPU category time. That is an estimate, not a
measured request speedup. The single-record regression, extra scratch and changed
short-record arithmetic make this an unsuitable production change.

#### Evidence

`golden/gdn-packed-chunks-20261005/` contains the source generator, standalone benchmark,
generated shader, build hashes, frozen criteria, twelve raw timing records and aggregate
results. The numerical inputs are synthetic; only article lengths were used.

### Interval-decay reuse in chunk preparation, 2026-10-15

Status: the candidate passed its isolated kernel screen, but its whole-engine screen stopped
before any candidate inference, so it is not adopted. Production `gdn_chunk_prep` in
`metal/clef.metal` still evaluates `gdn_interval_decay` for each element.

The candidate caches the inclusive interval-decay triangle once per full 32-token block in
`gdn_chunk_prep`; KE, U and A reuse those identical values. The table lives in the existing
short-tail scratch, and partial blocks keep the original expressions. Interval summation,
gamma/E, reductions, allocations, dispatches and barriers are unchanged.

The synthetic discriminator passed all 133 cases: Hv32/48, every partial tail after a full
block, all 32 placement offsets at T63 and three long Hv48 lengths. It compared every padded
W/U/KE/A/E byte and both VC16/VC32 scan outputs across arms under two poison patterns, with
input/output guards and finite checks. All 9,576 sampled float64 checkpoints passed; the worst
absolute error was `5.6718843e-8`. Six timing series of one warm and six measured ABBA quartets
favored the candidate in every measured quartet. Reductions are medians of the six
within-quartet mean ratios:

| Tokens / value heads | Preparation reduction | Preparation + scan reduction | Combined median paired saving |
|---|---:|---:|---:|
| 4,096 / 48 | 32.15% | 7.64% | 0.9516 ms |
| 8,072 / 48 | 16.91% | 5.20% | 0.9254 ms |
| 16,347 / 48 | 17.09% | 12.82% | 3.4581 ms |

All series passed the frozen advancement gate: a preparation median ratio of at most 0.90 with
at least five faster quartets, and a combined ratio of at most 1.00. Each interval is one
command buffer and excludes compilation, allocation and checks. Static threadgroup memory is
unchanged (8,704 bytes for preparation, 12,288/24,576 bytes for the two scans); register
occupancy was not measured. Multiplying the combined savings by 48 layers gives conditional
estimates of 44.4 ms at 8K and 166.0 ms at 16K, about 0.48% and 0.79% of historical 27B
full-call medians; synthetic kernel timings do not establish request gains. Production Flash
does not use the chunked route, so the Hv32 cases are coverage, not a Flash gain.

The whole-engine screen stopped in its fresh Flash baseline. The private builds omitted the
`clef-media` helper, which the decoder resolves beside the real CLI executable, so encoded-video
records `vd005` and `vd006` returned explicit errors and the qualifier correctly rejected
incomplete coverage despite exit status zero. The other 44 baseline records (99 questions, 329
logits) were byte-exact to the retained build. No candidate, 27B or timing run followed, and the
screen was closed rather than repaired and continued. A further attempt needs a package that
includes `clef-media`. Evidence: `golden/dense-gdn-decay-reuse-20261015/`
(`runs/local-score.json`, `engine-stage-decision.json`).

## Workload-level changes

For the batched classification workload (seven-question rubrics on the 27B, see the
[performance history](performance-history.md#batched-classification-workload)), four
application-level alternatives were evaluated on 66 operator-labeled items with the article
text unchanged. All were rejected under a predeclared no-regression condition:

- **Shorter seven-field rubric** (65 fewer tokens): 1.06× throughput, but two more rejected
  stories missed by full-score routing in the first label round.
- **Warning-only pass** (three of the seven questions, 696 instead of 1,300 tokens): 1.92×
  throughput and the same rejected-story catch counts, but one more publish-worthy story
  sent to review, two borderline routes changed, and no full audit or numeric score.
- **Cached warning-then-confirm cascade** (warning pass first, full rubric via a prefix-cache
  hit only when a warning fires): 16.4% less total time on the 66 labeled items with every
  warning set preserved, 14.0% less on 134 further items but two full-rubric warnings
  missed there; only seven of nine matched blocks were faster on the pilot.
- **Schema before state** (so the rubric could become a reusable cached prefix): twenty
  full-audit routes and thirteen warning flag sets changed and AUC fell from 0.841 to
  0.707 on round one. The head pools question and option hidden states that, in canonical
  order, condition on the article; that conditioning is what the reordering removes.

Keep the original rubric. The existing prefix cache cannot reuse a state-dependent schema
across unrelated records.

## Vision follow-up, 2026-10-07

Evidence lives in `golden/vision-exhaust-2026-10-07/`; `experiments.json` records the acceptance
rule. No quantization, resolution reduction, decision regression or relaxed tolerance was
allowed. The following candidates passed their isolated exact-output comparisons but did
not earn a production dispatch change:

- Compensated GEMM plus bias/GELU epilogues. Medium block-up shapes sometimes improved in
  isolation, while small and merger shapes regressed. Paired complete requests failed to
  establish a broad gain; the 27B medium-image median was 2.15% slower.
- Thirteen backbone tile/order variants at twelve actual matrix shapes. These covered
  16/32/64/128-row tiles, 64/128/256-column tiles and grouped/column-major traversal. All
  screened outputs matched exactly. The short-request candidate's apparent gain on `v001`
  did not generalize: `v000` was 5.08% slower. A separate long-Flash expansion candidate
  was 0.73% slower on `v009` and 5.46% slower on `v010` in its paired request runs.
- FP32 vision attention with a multiply-only score product, removing explicit accumulator
  clearing. All output bits matched at 2,048, 2,052, 4,092, 4,096 and 4,100 patches, including
  poisoned tails, narrow outputs and overflow checks. Timing varied by about 1% with no
  consistent improvement.
- Compiling the image pipeline with `-O3` instead of `-O2`. The three PNG screening fixtures
  were approximately 5–9% slower, so the existing compiler optimization setting stays.

Desktop activity caused substantial absolute timing variation. These are screening results,
not hardware limits. The experiments remain reproducible in the evidence directory; none
justifies claiming that every possible kernel optimization has been exhausted.

## Dense precision controls, 2026-10-14 and 2026-10-15

These private candidates changed one local rounding or operator-grouping step at a time in Flash
and 27B, to test whether the complete model moves closer to the FP32 reference. Candidates that
reached a model screen used the canonical 46-request text/image/video corpus (103 questions and
339 logits per model) unless the entry says otherwise. The frozen rule was to keep every FP32
decision and the 0.05 logit / 0.002 probability ceilings, and to lower both the mean
question-maximum logit error and the mean question-maximum probability error on both models. The
user approved at most 2% extra full-request latency for a demonstrated accuracy gain. No
candidate was integrated, and none established a labeled task-accuracy gain. Repeatedly, better
local arithmetic did not carry through to final scores.

### FP32 attention output, 2026-10-14

The compensated attention kernel kept its gated output in FP32 instead of rounding it to FP16,
reusing the gate buffer after its final read, and the existing FP32 residual GEMM projected it.
This changes both storage precision and the projection implementation. It passed 422 kernel
comparisons (separate and in-place outputs, packed records, prefix suffixes, poison guards,
overflow and direct BF16 calls), and rounding its output to FP16 reproduced baseline bits.

On the canonical corpus both models kept all 103 decisions and the ceilings. Mean
question-maximum probability error fell 19.01% on Flash and 5.94% on 27B, but individual errors
were mixed: 27B's maximum rose from 0.001136 to 0.001796, and Flash's `r000` final-norm relative
L2 worsened. On a separate 24-request numerical fixture (all 96 decisions per model kept),
Flash's request-mean probability error fell 38.73% with a negative paired bootstrap interval and
a lower maximum. 27B's fell 6.22%, but its 95% interval crossed zero and its maximum rose 5.47%
(0.00024736 to 0.00026089), failing the predeclared improvement criterion. That fixture had
informed earlier choices and is not untouched validation.

Paired fresh-request timings, with two warmups and six ABBA quartets per fixture in one loaded
engine, exceeded the 2% limit everywhere; 35 of 36 quartets were slower:

| Model | Tokens | Baseline median ms | FP32 median ms | Latency increase |
|---|---:|---:|---:|---:|
| Flash | 146 | 75.020 | 78.544 | 4.70% |
| Flash | 300 | 124.194 | 131.173 | 5.62% |
| Flash | 2235 | 708.976 | 748.521 | 5.58% |
| 27B | 146 | 194.930 | 204.387 | 4.85% |
| 27B | 300 | 341.857 | 360.827 | 5.55% |
| 27B | 2235 | 2327.556 | 2444.859 | 5.04% |

Evidence: `golden/dense-attn-output-20261014/` and `golden/dense-attn-output-f32-20261014/`.

### Two-plane attention output, 2026-10-14

The follow-up stored the gated output as two FP16 planes, `hi = half(value)` and `lo =
half((value - float(hi)) * 2048)`, combined two FP32 projection products as `product(hi, W) +
product(lo, W) / 2048`, then added the FP32 residual once, in every full-attention layer of both
dense models. It reused dead scratch and kept overflow flags and finite saturation; Omni and
BF16 fallback were unchanged. It passed the 422 kernel comparisons and its attention error fell
on 1,420 FP64 samples. On six real captures its sampled squared and maximum projection errors
were below both FP16 and direct FP32 projection.

Both models kept all 103 decisions, the ceilings and the `r000` layer checks, and control mode
reproduced all 339 baseline logits. The final-score rule still rejected it:

| Model | Mean max logit error, baseline → candidate | Change | Mean max probability error, baseline → candidate | Change |
|---|---:|---:|---:|---:|
| Flash | 0.00129242 → 0.000866845 | −32.93% | 0.0000714094 → 0.0000748107 | +4.76% |
| 27B | 0.00134961 → 0.00151639 | +12.36% | 0.000154015 → 0.000161028 | +4.55% |

Flash had 60 improved and 43 worse probability errors; 27B had 49 improved and 54 worse. Both
`r000` final-norm relative L2 values improved, but that single trace does not describe the other
requests. No latency was measured. Evidence: `golden/dense-attn-output-comp-20261014/`
(`decision.json`).

### FP32 attention output at the final block, 2026-10-14

The same FP32 output kernel ran only in the final decoder block, layer 31 on Flash and 63 on
27B, positions chosen from the architectures before evaluation. Kernel checks passed, control
mode reproduced all 339 retained logits per model, and every stage through the final-block input
stayed byte-exact. Flash's mean question-maximum logit/probability errors rose 0.865%/0.212% and
its maximum probability error 0.898%; 27B's means fell only 0.145%/0.112%. Rejected; no latency
was measured. Evidence: `golden/dense-final-attn-20261014/`.

A matched-input diagnostic then fed one unchanged native final block either its own input or the
exact FP32 reference input, for one 300-token request per model, with control outputs and all
seven logits byte-exact to the baseline. Final-block output RMS error fell from 0.007916 to
0.001442 on Flash and from 0.036555 to 0.001939 on 27B; on the union of question and option
spans it fell from 0.005059 to 0.001419 and from 0.014898 to 0.001728. The large final-block
discrepancy on this request therefore depends strongly on upstream error. This is not an
additive decomposition and does not justify a final-MLP precision change. One token accounted
for 79.0% of 27B's control squared error. Evidence:
`golden/dense-final-block-diagnostic-20261014/`.

### FP32 attention output at the earliest two blocks, 2026-10-14

Following that diagnostic, a fixed policy kept FP32 output only at blocks 3 and 7, the earliest
full-attention blocks in both dense topologies, selected before any outputs were seen. All
captures passed coverage, ceilings and decisions; mode 0 reproduced every retained logit, and
`r000` stages through block 2 stayed byte-exact. Flash's mean logit/probability errors fell
23.976%/1.781% while its worst logit/probability errors rose 5.924%/18.889%. On 27B mean logit
error fell 5.606%, but mean probability error rose 5.834% and worst probability error 35.361%.
Probability error worsened on 47 Flash and 56 27B questions. Rejected; no latency was measured.
Evidence: `golden/dense-early-attn-20261014/`.

### DeltaNet query normalization boundary, 2026-10-14

A control materialized the normalized FP32 query before applying the existing dk scale, matching
the pinned reference's operator grouping; norm reduction, epsilon, key normalization, beta,
gates, buffers and dispatch geometry were unchanged. A staged GPU control confirmed every query
bit across 823,296 elements, 235,296 of which differ from production, and other outputs and
eight packed-record comparisons stayed exact. Synthetic query SSE against FP64 nevertheless rose
11.56%: matching the reference's grouping is not more accurate mathematics.

The canonical screen passed ceilings and decisions on both models. Flash's mean question-maximum
logit/probability errors fell 11.04%/0.50%, but 27B's rose 7.96%/6.79%, and its maximum
probability error rose 27.05% while staying below 0.002. Rejected. Evidence:
`golden/dense-arithmetic-controls-20261014/`.

### Vision attention scale rounded once, 2026-10-14

A host-only mode computed the vision scale as `(float)(1.0 / sqrt((double)head_dim))` instead of
`1.0f / sqrtf((float)head_dim)`. At head dimension 72 this changes one bit, is about 17 times
closer to the exact scalar and matches the pinned reference, yet local synthetic
attention-output SSE against FP64 rose 9.02%.

The canonical screen first passed: mean question-maximum probability error fell 6.79% on Flash
and 0.04% on 27B, with all bootstrap intervals crossing zero. A fresh prospective screen
therefore used 16 synthetic image/video requests (four each of small and large images and
videos; 40 questions, 120 logits and 12,924 tokens per model), with token IDs, positions, media
grids and processed FP32 patches checked exact against the reference first. It rejected the
change: Flash's mean question-maximum logit and probability errors rose 15.21% and 22.04%, and
its maximum probability error 9.14%; 27B's logit mean improved 4.92% but its probability mean
worsened 3.94%. Both versions kept every ceiling and all 40 decisions per model. Evidence:
`golden/dense-arithmetic-controls-20261014/media16/`.

### DeltaNet output precision, 2026-10-15

**Attribution.** A matched-input CPU oracle held native normalized q/k/v, beta, log gate and
weights fixed and separated recurrence/state rounding, producer FP32 arithmetic, final FP16
conversion, decay sensitivity and output-GEMM/residual rounding, projected into common
residual-boundary units. The cells were `r000` and `v000` at layers 0/16 on Flash and 0/32 on
27B: eight cells sharing four trajectories, all on the sequential `gdn_scan_8`. Long chunked 27B
records, video, prefixes, overflow fallback and other depths are outside this result. Final FP16
operand storage dominated every cell: its RMS was 360.94–2369.59 times the recurrence RMS and
2608.67–3057.52 times the producer-arithmetic RMS, and it exceeded both nuisance terms, meeting
the prospectively fixed factor-two separation rule. Its relative L2 error against the
double-precision projected operand was 0.0190–0.0296%, so a large component ratio alone does not
establish task relevance. The first shadow replay failed its bit-exact scan check because the
appended shadow kernels inherited a later file-scope `#pragma clang fp contract(off)`; restoring
`contract(on)` before them fixed it, and the attribution is conditional on that corrected
instrument. Evidence: `golden/dense-upstream-attribution-20261015/`
(`revision-2/reports/result.json`).

**Candidate.** Keeping the FP32 producer values in the dead scan O buffer and projecting them
with the existing accumulating FP32-input GEMM (`gemm_f32_acc_32x128`) preserved the producer
expression, contraction context, `act16` overflow flags, finite saturation, allocations and
checkpoint layout. The route was selected uniformly for dense DeltaNet blocks when the engine
opened; default, BF16-retry and Omni routes were unchanged. It passed 914 local GPU dispatches,
where the eight real cells matched both the original FP16 operands and the FP32 replay bits
exactly, and 30 public API calls per model across eleven engine configurations. Those covered
packed records with nonzero offsets, prefix reuse, the 27B chunk16/chunk32 routes, checkpoint
plans, prefix invalidation and forced prefix-to-plain-to-BF16 fallback. All 39 selected default
logits changed in each model, so the route was not a no-op.

**Cost.** The fourteen-case latency screen rejected it before any accuracy campaign. Changes are
medians of six ABBA quartet ratios, each comparing the mean of two candidate calls with the mean
of two baseline calls. Calls include GPU inference, readback and the CPU head, excluding model
load, encoding and cache fill; cached cases use separate entries per arm.

| Request | Flash latency change | 27B latency change |
|---|---:|---:|
| Fresh, 146 tokens | +10.22% | +14.27% |
| Fresh, 300 tokens | +13.67% | +15.83% |
| Fresh, 2,235 tokens | +12.57% | +13.52% |
| Fresh, 8,072 tokens | +11.04% | +12.34% |
| Fresh, 16,347 tokens | +9.40% | +10.91% |
| Cached, 8,072 tokens, 7,808 reused | +10.55% | +11.17% |
| Cached, 16,347 tokens, 16,064 reused | +9.35% | +10.34% |

All 84 quartets were slower and every case exceeded the 2% limit; all 400 non-reference calls
matched their own-arm uncached logits. Dispatch counters show the FP16 projection route replaced
by `gemm_f32_acc_32x128`: FP32 doubles activation operand width, gives up the larger half-only
tiles some baseline cases use, and the producer writes FP32 in place. These are plausible
causes, not a measured attribution; profiling was off. The DeltaNet output projection in
`clef_metal.m` still reads the 16-bit operand `g->A`. Evidence:
`golden/dense-gdn-f32-feasibility-20261015/` (`cost-result-revision-2.json`,
`root-cost-rejection-revision-2.json`).

### CPU head pooling and MLP residual controls, 2026-10-14

On six saved head inputs (38 logits, 16 questions), FP64 span and lexical means increased
downstream logit RMS error against the pinned FP64 head graph: Flash 3.303e-7 to 3.787e-7 and
27B 2.123e-7 to 2.295e-7. FP64 weighted option summaries changed no logits. This holds native
normalized hidden states and memory K/V fixed, so it isolates CPU head arithmetic, not
whole-model error. Evidence: `golden/dense-pooling-audit-20261014/`.

An unchanged-kernel control compared MPP `multiply_accumulate` with `multiply` followed by one
CPU FP32 residual add. All 7,299,072 outputs matched across 48 synthetic down-projection cases,
including cancellation and large residuals, and 4,608 sampled FP64 checks on identical rounded
operands gave identical errors. This is observed equality on those inputs, not a property of MPP
or a test of lost FP32 activation precision. No candidate followed. Evidence:
`golden/dense-mlp-residual-control-20261014/`.

### CPU head GELU tail conditioning, 2026-10-15

A model-free screen compared the retained `0.5f*x*(1.0f+erff(x*c))` with `0.5f*x*erfcf(-(x*c))`,
keeping the FP32 argument and multiplication boundaries. The alternative recovers small
negative-tail values lost when `erff` rounds to minus one (the largest omitted real value on the
synthetic grid was `8.25e-8`), but some positive inputs became less accurate. Four fixed scalar
CPU workloads had median candidate/baseline time ratios of 2.22–3.56, with none of 24 ABBA
quartets faster. These are scalar loop costs, not head or request latency. Actual head
activations and downstream benefit were not established, no model ran, and the GELU helper is
unchanged. Evidence: `golden/dense-head-gelu-screen-20261015/` (`CORRECTED-ANALYSIS.md`).

## Dense RMSNorm and head-memory buffers, 2026-10-14

### Virtual lanes for RMSNorm

A private probe replaced a 1,024-thread row reduction with 512 threads, each computing two
logical lane sums, to keep the per-lane and SIMD-group reduction order while reducing
synchronization. Under production safe math it changed one BF16 output out of 598,016 at 146
rows by 4,096 columns, failing the exact-output condition before timing. This does not show
worse accuracy, which was not measured. Evidence: `golden/dense-norm-20261014/decision.json`.

### Decoder scratch reuse for head memory

A host-only change reused the dead decoder `O` and `Xc` allocations for the dense head's memory
projection and FP32 normalized memory, with independent capacity checks and allocation
fallbacks; arithmetic, shaders and Omni were unchanged. For the released models at capacity
16,384 the logical reduction is 128 MiB with the FP32 head or 64 MiB with the BF16 head, not a
measured RSS saving. All 26 correctness and recovery jobs passed, as did the ASan/UBSan
allocation test (32 layouts, 862 initial plus 862 growth failure points).

The frozen latency gate, a median ABBA candidate/baseline ratio of at most 1.01, failed at
27B/2,235 tokens (1.011211). A prospectively fixed repeat of all six model/length cases,
requiring every repeat median and every median pooled over the twelve original and repeat
quartets to stay within 1.01, failed at Flash/2,235: 1.028103 in the repeat and 1.011232 pooled.
Every call had exact outputs. Both allocation sets stayed resident, so allocator cost and RSS
were not measured, and no precise general slowdown is inferred. Rejected: `mem` and `mn32`
remain separate allocations in `clef_metal.m`. Evidence:
`golden/dense-head-scratch-reuse-20261014/runtime-audit/`.

## Omni

Clef-Omni experiments from 2026-10-12 to 2026-10-16. None of them is in the working default.
Gates are those of the [Omni verification policy](omni.md#numerical-verification-policy): exact
tokens and argmax, residual relative L2 at most 0.01 at every captured stage, logit error at most
0.05 and probability error at most 0.002. "FP32 screen" means the 43 requests and 93 questions of
the short corpora against the FP32 reference (blocked attention); the working default fails 12
of those questions, with a mean absolute probability error of 0.000708464 per question.
Components marked "Kept" are part of the unapplied FP32-compatible build
([status](omni.md#status)); its retained speed steps and their timings are in
[the lineage table](omni.md#fp32-compatible-build). Evidence directories are under `golden/`.

### FP64 precision prototypes

From 2026-10-12 to 2026-10-13 the work tried to make native arithmetic more accurate, measured
against independent CPU FP64 references ([FP64 controls](omni.md#fp64-controls)), while the FP32
screen remained the acceptance gate. Unless the table says otherwise, each change passed its own
local operator tests (precision bounds, packing, poison, nonfinite inputs); the columns give the
whole-model results. Row 1 stands alone. Rows 2–8 are variants of one full-compensation prototype
(row 8 builds on row 6); rows 10–15 change one operator family at a time on the row-8 prototype,
cumulatively from row 11. `r020` is the 8,069-token text control.

| # | Change | Independent FP64 result | FP32 screen, failed of 93 | Latency | Decision | Evidence |
|---:|---|---|---:|---|---|---|
| 1 | Vision GEMMs as 32-element FP32×BF16 products with two-component sums (2026-10-12; alone) | `oav004` 12/50 → 0/50 failed stages; `ot003` 13/50 → 26/50 (worst 0.0520371 → 0.148930); `ov001` 9/50 unchanged | 17 | 1.5–2.5% slower on three fixtures | Rejected, removed | `omni-vision-accuracy-20261012/` |
| 2 | Full compensation: dense/expert products, decoder attention, small-grid vision attention, precise vision rotary | Seven selected controls pass | — | +37.8–53.5% | Not adopted | `omni-decoder-accuracy-20261012/rotary/` |
| 3 | The same restricted to Omni vision kernels | Fresh two-image `ov003`: 7 failed stages (0.0310381); the default passes | 18 | +2.2–6.9% on media fixtures | Rejected | `omni-decoder-accuracy-20261012/vision-precise/` |
| 4 | Full compensation with unrolled sums and 16×128 expert/dense tiles | Nine selected controls pass | 19 | 13–20% faster than full compensation; still +11–26% | Not adopted; a 32×64 expert tile was stopped at 29–39% slower expert products | `omni-compensated-performance-20261013/` |
| 5 | Scoped to Omni entry points; complete audio corpus | `oa002` fails 3 stages (0.0207346); the default passes all 300 | — | — | Regression found | `omni-reference-agreement-20261013/` |
| 6 | Decoder rotary coefficients from CPU FP64, stored as FP32 high/low pairs (1 KiB per packed token) | 17 controls and the complete video corpus pass; audio/video `oav003` fails 7 stages (0.0367556) | 18 | +9.0% to +21.3% on six fixtures | Not adopted | `omni-decoder-rounding-20261013/rotary-v2/`, `omni-av-qualification-20261013/` |
| 7 | FP64 learned-position coefficients with compensated addition | `oav002` gains 7 failed stages (0.0433678) | — | — | Rejected | `omni-vision-position-20261013/precise/` |
| 8 | Vision tanh-GELU in an exponential-sigmoid form, on row 6 | All 23, then 41, controls pass; `r020` fails `final_norm` (0.0201982) | 18 | 9.97–22.62% slower | Not adopted; kept as the diagnostic prototype | `omni-vision-gelu-20261013/`, `omni-8k-text-qualification-20261013/` |
| 9 | That GELU alone on default arithmetic | `ov003` and `ot004` gain 7 failed stages each | 17 | — | Rejected | `omni-production-gelu-20261013/` |
| 10 | Q/K normalization with compensated squares, refined reciprocal root, low bits through rotary | `r020` 1 → 16 failed stages (0.0471469); changed expert sets 799 → 5,988 | — | — | Rejected | `omni-qk-norm-20261013/` |
| 11 | QKV with 16-term partial sums; QKV with FMA-compensated products and sums | 16-term misses the local `3e-8` bound; compensated: `r020` 1 → 16 (0.0403813) | — | — | Rejected | `omni-router-normalization-20261013/` |
| 12 | Compensated QKV with a separate FP32 Q/K store; then accurate decoder RMS | 16 failed stages (0.0431581); with RMS, 1 (0.0201982309) | — | — | Not qualified | `omni-qk-store-20261013/` |
| 13 | Attention: low components kept through the final division; compensated probability–value products | Division fails the local exact-constant test; products: `r020` 16 failed (0.0433346) | — | — | Rejected | `omni-prefix-attention-20261013/` |
| 14 | Attention: compensated scores, exponential and online softmax | Local error 8.8-fold lower; `r020` 16 failed (0.0404776) | — | — | Rejected | `omni-attention-scores-20261013/` |
| 15 | Compensated selected-expert weight normalization | On the combined precision profile, `r020` 16 → 17 failed (0.0418617) | 15 on default arithmetic; 18 on the combined profile | — | Rejected | `omni-route-weights-20261013/` |
| 16 | GPU-written indirect dispatch grids for expert tiles (arithmetic unchanged) | Captures unchanged | Unchanged | Prototype: 0.40–3.02% lower on five fixtures, `r019` +1.06% and +0.34%; default: 0.96–3.17% lower | Not kept for the prototype; the default variant deferred | `omni-precision-performance-20261013/` |

What this established:

- **Local accuracy does not predict the whole request.** On `r020`, replacing only the QKV
  projection with its correctly rounded result restores the reference expert at the first flip
  (layer 22, token 5,145). Yet the native Q/K, QKV, attention and routing-weight corrections each
  moved the first flip earlier, to near-ties such as layer 12/token 5,606 (reference score gap
  `3.31363e-8`), and raised the failed stages from 1 to 16 or 17; accurate RMS left the single
  failure in place. Exact layer arithmetic cannot undo state drift that is already sufficient for
  the flip, and the row-8 prototype's native arithmetic partly cancels earlier error at that tie.
- **FP32 storage is not the obstacle.** From the embedding, ideal FP64 arithmetic with FP32 stores
  at the native kernel boundaries (Q/K normalization stored separately) passes all 50 `r020`
  residual checks, worst relative L2 `0.000168495`. Fusing the Q/K normalization store flips layer
  12 again, so fewer rounding boundaries do not guarantee the reference path.
- **FP64 agreement moves away from the FP32 gate.** The combined precision prototype passes all 91
  shared final checks against FP64 but fails against FP32 on exactly the 17 questions where the FP32
  reference itself exceeds the limits against FP64. The FP32 gate was reaffirmed as the acceptance
  target, and the work turned to reproducing the reference's arithmetic.
- Do not retry compensated vision GEMMs on GEMM error or `oav004` alone, selected-weight
  compensation on its local rounding result, or any single-operator precision change without a
  whole-request FP64 and FP32 check.

### FP32 compatibility components

2026-10-13. Each change reproduces a pinned PyTorch 2.11 MPS operation and was first shown
byte-exact on identical captured inputs, with offsets, poisoned scratch, guards and explicit error
cases. Rows are cumulative in order; "alone" means applied to the working default.

| Component | Change | Same-input check | FP32 screen: failed of 93 (mean absolute probability error) | Decision | Evidence |
|---|---|---|---|---|---|
| Router projection | `MPSNDArrayMatrixMultiplication` once per complete record on cached FP32 expansions of the BF16 router weights | 38,144 saved scores exact | 14 (0.00113228) alone | Kept with later rows | `omni-reference-divergence-20261013/` |
| Vision patch embedding | Contraction in spatial-then-channel (DHWC) order | 1,290,240 convolution outputs exact; the original order differs on 1,161,174 | 16 (0.000882867) alone; 15 (0.00147004) with the router | Kept | `omni-fp32-vision-stages-20261013/` |
| Learned-position interpolation | MPS coordinate formation and separately rounded weighted products | 63,286,272 values exact | 16 (0.000832705) alone | Kept | `omni-fp32-vision-position-20261013/` |
| Vision LayerNorm | MPS reduction: four contiguous elements per thread, sum and squared sum, variance clamp, precise reciprocal root, local `contract(on)` | 46,517,760 values exact | 17 (0.000848623) alone; the four rows together 12 (0.000794738) | Kept | `omni-fp32-vision-normalization-20261013/` |
| Vision rotary and attention | Separately rounded rotary products; MPSGraph QK product, FP32 scale, softmax, 1,024-key value blocks and block sum | Exact through attention, residual, norm2 and first MLP projection | 10 (0.000741431) | Kept | `omni-fp32-vision-attention-20261013/` |
| Vision GELU | MPSGraph tanh and erf GELU in the pinned operation order | All 44 compared `ot001` vision stages exact | 7 (0.000542382) | Kept | `omni-fp32-vision-gelu-20261013/` |
| Decoder Q/K preparation | Materialized squares, MPSGraph mean, `1 / precise::sqrt`, separate stores; rotary table from the pinned FP32 initialization (it differs from `powf` at one of 64 frequencies) | 86,615,040 values exact | 7 (0.000427291) | Kept | `omni-fp32-r019-decoder-20261013/` |
| Decoder RMS | The same recipe at input, post-attention and final norms; or post-attention only | 71,804,928 values exact | 11 (0.000543154); post-attention only 7 (0.000371090) | All-boundary form kept with the next row; post-attention-only dropped | `omni-fp32-r019-layer5-20261013/` |
| Decoder causal attention | Blocked FP32 as in the reference: whole eight-head GQA groups, 64-million-element score budget, 1,024-key value blocks, the reference's sorted-stride views for query slices | 192,552,960 values exact | 7 (0.000453197) | Kept | `omni-fp32-decoder-attention-20261013/` |
| Router weights | Materialized softmax, top 8 by sort, sum and division | 183,816 weights and IDs exact | 7 (0.000395057) | Kept | `omni-fp32-router-weights-20261013/` |
| Per-record reduction shapes | RMS and key normalization reduced at each record's own shape | Packing `r003` with `r019` had moved `r003`'s logits by up to 0.1150105; all six packing checks now exact | 7; unbatched outputs unchanged | Kept; required for batch invariance | `omni-fp32-router-weights-20261013/` |
| Small-expert products and SiLU | FP32 graph products for experts with fewer than 16 assignments in the complete record (chunk-local counts would select the wrong groups); SiLU as the reference's graph operations | 57,481,216 values exact | 3 (0.000259015); all 1,600 text, image and video stages exact | Kept; this graph profile ran at 25.00x and 4.73x default latency | `omni-fp32-expert-block-20261013/` |
| Audio CNN and positions | NCHW/OIHW convolution graph with bias and separate GELU; sinusoid table regenerated from the CPU FP32 formula (1,530 of 16,640 values had differed) | 290,856,960 values exact | Audio records: 0 of 22 questions, 13 of 550 stages (11 new on `oav003`) | Kept | `omni-fp32-audio-isolation-20261013/` |
| Audio LayerNorm | MPS kernel arithmetic with 320 threads at width 1,280 | 12,990,720 values exact | Audio: 1 of 550 stages (`oav000/final_norm` 0.0139364) | Kept | `omni-fp32-audio-norm-20261013/` |
| Short audio projections | MPSNDArray products when the complete matrix has fewer than 16 rows | 15,089,664 values exact | Audio: 4 of 550 (three new on `oa003`) | Rejected as this profile; included in the final composition with the next two rows | `omni-fp32-oav000-isolation-20261013/` |
| Audio attention and GELU | Interleaved Q/K views, separate scale and softmax, 1,024-element padded value reduction; separately materialized GELU | 3,244,800 and 3,686,400 values exact | With the frontend: 0 of 550 stages, 0 of 22 questions | Kept | `omni-fp32-audio-attention-20261013/` |
| Audio frontend | PocketFFT with contraction on, through a C ABI bridge; dense FP32 mel product; SLEEF-derived `log10` with explicit FMA | 6,447,654 stage values exact over 22 cases | As above | Kept | `omni-fp32-audio-boundary-20261013/` |

Notes:

- **Why a locally exact router regressed.** On `ot001`, the MPS router first changes an expert at
  layer 4/token 99 and fails 22 stages, although its same-input scores match the reference
  exactly. Native visual embeddings and DeepStack features together were sufficient for that
  flip under an FP32 decoder; either alone was not. Each component helped only once the inputs
  reaching it matched as well.
- **Audio boundary.** Supplying the FP32 tower's output on canonical features made all 50
  `oav000` decoder stages byte-exact, while the FP32 tower on native features and the native tower
  on canonical features each still failed. The remaining error was in both the frontend and the
  tower; the substitution itself was a diagnostic and never part of inference.
- **Cost.** The composition is graph-based and was slow before the speed work: the vision/decoder
  profile ran at 25.00x the default's latency on 298 tokens, and the complete composition with
  audio took 6,506.796 ms against the default's 246.702 ms. Its first full 43-request pass came
  with the first retained speed step, which left every output unchanged.

### FP32-compatible speed candidates

Candidates for the FP32-compatible profile that were not retained, 2026-10-14 to 2026-10-16.
Every candidate kept its parent's arithmetic or reproduced it bit for bit, and every timed response
was checked byte for byte. "Model screen" means poisoned captures of `r003` (298 tokens) and
`r019` (2,231 tokens, final 512 rows): 100 saved stages and five questions against the FP32
reference, plus four packed responses.

The 2026-10-14 candidates were timed in production-bracketed resident processes (two warmups,
three measured requests) and judged on whether both paired blocks favored them. From 2026-10-15,
criteria were frozen before each run and a failed screen was not repeated. Early 2026-10-15 jobs
used ABBA with the same counts and required both paired ratios below 1 and the pooled ratio below 0.99 on each
request. After an identical-binary control proved unstable (see diagnostics below), jobs used four
resident processes with five warmups and five measured requests each: the target request needed
both paired ratios below 0.97 and the pooled ratio below 0.95, and the other request was a
regression control with paired and pooled ratios below 1.03. Operator-only screens applied 5%
pooled and 3% paired benefit thresholds to GPU device spans, which include internal submission
gaps. Ratios are candidate/parent; "paired" values are the two ABBA pairs.

| Candidate | Numerical result | Speed against its criterion | Decision | Evidence |
|---|---|---|---|---|
| Q/K helper's three callback commands combined into one MPS wrapper | 125,460,480 local values exact; whole-model outputs checked on `r003` and `r019` only | 298 tokens 1.92% lower; 2,231 tokens 0.69% higher, then 1.12% lower in a repeat | Not promoted | `omni-fp32-qk-async-20261014/` |
| Empty Q/K-to-attention submission removed | Local checks exact | −0.03% and +0.91% | Not promoted | `omni-fp32-qk-handoff-20261014/` |
| Unused expert activation rows trimmed | Model screen exact | No short gain; 2,231 tokens 14.8% slower, both pairs regress | Rejected | `omni-activation-extent-20261014/` |
| Activation appended to the tiny-expert projection command | Two text screens and packing pass | Short 2.5% lower; long mixed; the fixed follow-up aborted before any pair | Not accepted | `omni-expert-projection-activation-20261014/` |
| Four-value tiny-expert weight scanner | Operator controls pass | Four of six shapes have a slower pair | Rejected at the operator screen | `omni-tiny-scan4-20261015/` |
| Router helper preparation overlapped with the incoming command | Model screen exact | Short 1.54% faster; long 0.41% slower, both pairs regress | Rejected | `omni-router-input-timing-20261015/` |
| Tiny-expert output validation grouped into one dispatch per batch | Operator exact | Three of six shapes fail; down-8 pooled 8.10% higher | Rejected at the operator screen | `omni-tiny-output-batch-20261015/` |
| First record's RMS square appended to its producer command | Model screen exact | 298: 0.997153 / 1.002500; 2,231: 1.006217 / 0.961494 | Rejected | `omni-rms-producer-transfer-timing-20261015/` |
| Native RMS mean with fixed reduction trees for 298 and 2,231 rows | Model screen exact | 298: 1.179361 / 1.010239; 2,231: 1.003119 / 0.939316 | Rejected | `omni-rms-native-timing-20261015/` |
| Square, native mean and normalization fused into the caller's encoder | Model screen exact | 298 passes (0.897161 / 0.984416); 2,231: 0.908767 / 1.014257 | Rejected | `omni-rms-inline-timing-v2-20261015/` |
| Probability padding and layout conversion moved into the softmax graph | Model screen exact | +10.95% (1.135499 / 1.137081) and +29.04% (1.309278 / 1.274372) | Rejected | `omni-attention-layout-timing-20261015/` |
| Query chunks capped at 1,024 rows, scores computed only for reachable keys | Operator within local limits; model screen exact | 298: +3.005% (pooled 1.030049); 2,231: +5.444% (1.077669 / 1.039833) | Rejected | `omni-attention-causal-timing-20261015/` |
| Expert chunks of 4,096 rows | Model screen exact | 2,231: 2.229% lower (1.009965 / 0.952693); would add up to another 498.25 MiB | Rejected | `omni-chunk4096-timing-v2-20261015/` |
| Eight-head attention groups promoted to sixteen at 2,049–2,896 tokens | Model screen exact | 2,231: +3.409% (1.065104 / 1.013053); 298 pairs 1.090409 / 0.963882; scratch 605,279,296 → 1,210,558,592 bytes | Rejected | `omni-attention-head-batch-timing-20261015/` |
| Waits deferred until the last attention head group | Model screen exact | 298 pair 1.038457 over the bound; 2,231: +0.794% (0.985890 / 1.010713) | Rejected | `omni-attention-deferred-timing-20261016/` |
| Probability–value product and block reduction in one graph | Operator exact | 2,231-token device span 0.273% lower (0.994495 / 0.998635); 512-token pairs 1.091277 / 1.069289 | Rejected at the operator screen | `omni-attention-product-reduce-operator-timing-20261016/` |
| Native 64×128 expert tile for local counts of at least 64 | Operator exact | Uniform up 1.216104, concentrated down 1.409023, both pairs slower | Rejected at the operator screen | `omni-native-expert-tile64-20261016/` |
| Expert-local tile order visiting both rows of a column first | Operator exact | Up-dense128 5.886% lower but one pair fails; up-concentrated 5.645% higher | Rejected at the operator screen | `omni-native-expert-order2-20261016/` |
| Native FP32 MPP Q/K score product | Four synthetic fixtures inexact within local limits (worst `2.51422e-7` relative L2); model screen exact | 2,231: 0.831% lower (0.996309 / 0.981740) | Rejected | `omni-attention-native-score-timing-20261016/` |
| The same, skipping score tiles wholly beyond the last live query | As above | 2,231: 4.363% lower (0.971015 / 0.933574) | Rejected; adding the scale/mask epilogue passed (lineage step 15) | `omni-attention-causal-native-score-timing-20261016/` |
| Native score product reading Q/K directly instead of packed copies | Model screen exact | Against the default only: 1.909017 and 1.155806 | Not promoted; its own effect unmeasured | `omni-direct-qk-default-timing-20261016/` |
| Eight-word BF16 exponent scanner alone | Model screen exact | 298: 4.9946% lower, pooled 0.950053743 against < 0.95 | Rejected; with direct array views it passed (step 17) | `omni-tiny-bits8-timing-20261016/` |
| Two equal-row tiny-expert products per call | Model screen exact | 298: 1.8813% lower (0.980954 / 0.982271) | Rejected | `omni-paired-expert-timing-20261016/` |
| Native `vector4x32-fma` tiny-expert products | Within local limits, but `r019` fails 25 of 50 stages (final 0.0840002) and both questions (logit 0.0610426, probability 0.0124707) | Not timed | Rejected on numerics | `omni-tiny-native-v4-rejection-20261016/` |
| Native dot4-FMA tiny-expert products with separate validation dispatches | Bit-exact to MPS; model screen exact | 298: 3.009% lower (0.969367 / 0.970368); 2,231 pair 1.039992 over the bound | Rejected; fusing the validation passed (step 18) | `omni-tiny-native-dot4-timing-20261016/` |
| Tiny experts grouped by row count, one dispatch per bucket | Model screen exact | 298: 1.854% lower (0.987848 / 0.979228) | Rejected | `omni-tiny-grouped-validation-timing-20261016/` |
| Two output columns per tiny-expert thread | Model screen exact | 298: 0.547% lower (0.991710 / 0.996270) | Rejected | `omni-tiny-two-column-timing-20261016/` |
| Native expert tiles compacted to skip empty and tiny-expert tiles | Operator exact | Dense pooled device ratios 1.028551–1.125057 | Rejected at the operator screen | `omni-native-expert-compact-20261016/` |
| Native 32×256 expert tile | Operator exact | Up-dense128 1.090327; down 1.433477 and 1.522702 | Rejected at the operator screen | `omni-native-expert-wide256-20261016/` |

### Default-path speed candidates

**Compensated FP16 expert operands, 2026-10-17.** FP32-operand tensor-op GEMMs run at about a
quarter of the FP16 rate, so the expert gather and SwiGLU wrote each operand as FP16 high and
residual (scaled by 2048) planes in the existing scratch, and the expert GEMM ran two
FP32-accumulating products against the same BF16 weights, with per-record overflow flags and an
FP32 rerun. Logits moved by at most 4.5e-6 on `r003`. Expert time rose from 164.4 to 245.1 ms at
298 tokens (wall 245 → 335 ms), 218.7 → 290.1 ms at 586 and 736.3 → 822.9 ms at 2,231; it fell
only at 8,069 (2,437.8 → 2,385.1 ms) and 16,344 (5,127.8 → 4,756.5 ms). At short lengths the
expert GEMMs are bound by streaming each expert's weights once per chunk, and the second product
reads them again. Rejected and removed; the profile is in
[performance history](performance-history.md#omni-category-profile-2026-10-17).

**Compensated FP16 QKV and output projections, 2026-10-17.** The dense projections, unlike the
experts, are compute-bound on FP32 operands (`bench/mixed_bench.m`: about 15 TFLOPS against 59
for FP16 here). Splitting their operands into FP16 high/residual planes with per-token-record
overflow flags halved the `gemm` category (45.7 → 24.5 ms at 298 tokens, 262.7 → 141.9 ms at
2,231) but failed the FP64-bounded check on `r019` (logit error 0.0531) and `r021` (probability
error 0.0115 against the FP32 reference's 0.0099), with 9 questions beyond the fixed limits;
the same attention change without it passed. The 22-bit Q/K operands feed long-context scores.
Rejected and removed.

**Plain FP32 attention accumulation, 2026-10-17.** Summing every probability-value product in
plain FP32 cut attention to 187 ms at 2,231 tokens and 3.38 s at 8,069 (from 322 ms and 4.24 s)
and passed the FP64-bounded check, but drifts by about n * eps / 2 on flat rows (1.56e-5 at 8,193
keys against 2.98e-8 compensated), about 2e-3 at the 64k context where no reference checks it.
Not kept.

**Block-compensated attention, 2026-10-17.** Summing each 32-key block's four products in plain
FP32 and folding the block into the two-float accumulators once kept flat-row error bounded
(7.45e-8 at both 2,049 and 8,193 keys), cut attention to 262 ms and 3.62 s, made whole requests
3.7%, 5.4% and 9.3% faster at 2,231, 8,069 and 16,344 tokens, and passed the FP64-bounded check
on all 43 requests. It was merged, then reverted: compared directly with the previous default
against FP64 (`tests/compare_fp64_errors.py`), mean logit error over all 93 questions rose
27.2% and mean probability error 40.8%, against a 7.2% / 6.7% spread between the two FP32
PyTorch references, and every long-request question got worse (`r020/service` |d p| 0.000816 →
0.009748, `r021/service` 0.002549 → 0.007324). That comparison is now part of the acceptance
check ([policy](omni.md#numerical-verification-policy)). Evidence for both:
`golden/omni-attention-block-20261017/`.

**Expert GEMM bandwidth.** `bench/expert_bench.m` reproduces the expert GEMMs at 298 tokens: 128
experts of about 19 rows. FP32 and FP16 operands run at 330–375 GB/s of weight traffic against a
520–545 GB/s streaming read on this machine; 32×64 tiles and 2- or 4-way split-K do not raise it.
The MPP expert product is bandwidth-bound below its ceiling; closing that gap needs a different
kernel, not another operand type.

### Timing method and attribution diagnostics

- **Identical-binary control** (2026-10-15). The same build under both labels on the 2,231-token
  request: paired ratios 1.003772 and 1.013625 lie inside [0.98, 1.02], but the four arm medians
  (2,453.345 to 2,540.457 ms) spread by 1.035507, beyond the 1.03 limit. Small whole-request gains
  are not resolvable with this protocol; the later 0.97/0.95 margins are an engineering screen, not
  a noise bound. `omni-whole-request-timing-control-20261015/`
- **Synthetic operator timing** (2026-10-15). A calibration with identical-work, deliberately slower
  and bracketed controls found overlapping command-buffer intervals in all 144 quartets (717 of 720
  adjacent pairs). A serialized protocol then failed calibration: identical-work ratios 1.10931
  and 0.91471, and 29 of 96 scanner brackets within 0.90–1.10. The output-batch screen had also run
  106 extra correctness commands before timing in the candidate arm only. Small speed claims
  therefore need complete-request timing. `omni-timing-calibration-failure-20261015/`,
  `omni-timing-serial-calibration-20261015/`, `omni-timing-method-audit-20261015/`
- **Counters.** The M5 Max offers stage sampling and timestamp counters but no dispatch-boundary
  sampling, so kernels inside a shared encoder cannot be sampled individually.
  `omni-dispatch-counter-probe-20261016/`
- **Where the fused build's time goes.** Expert-containing commands take 193.154 ms of a 276.981 ms
  observed GPU interval union at 298 tokens; at 2,231 tokens expert and attention commands take
  765.445 and 480.498 ms of 1,594.456 ms. Sampled dense up/down projection medians sum to
  100.803 and 625.660 ms for the two requests and tiny up/down medians to 76.219 and 48.211 ms;
  dense up is the largest category. In attention, probability–value products measure 9.862 and
  183.760 ms, scores 3.497 and 94.636 ms, tail padding 5.304 and 34.532 ms. Gaps between commands
  are not proven idle. `omni-fused-command-profile-v2-20261016/`,
  `omni-fused-expert-stage-profile-20261016/`, `omni-fused-attention-stage-profile-20261016/`
- **Tiny-product sensitivity.** One extra fresh weight scan per tiny product raises 298-token latency
  from 473.917 to 539.929 ms (13.929%); repeating each tiny product, with matched extra checks in
  both arms, raises it from 477.015 to 535.068 ms (12.1702%). These are marginal costs, not
  removable latency. Reusing projection descriptors saves 2.522729 ms of CPU per synthetic request
  and was deprioritized. `omni-tiny-scan-sensitivity-20261016/`,
  `omni-tiny-product-sensitivity-20261016/`, `omni-tiny-metadata-cpu-20261016/`
- **MPS tiny-product arithmetic.** Twenty guessed FP32 reduction recipes all passed local tolerances
  and none matched MPS bits. Synthetic cancellation probes and a reconstructed addition tree then
  gave the exact recipe: a four-term FMA cell, separately rounded outer accumulation and an
  adjacent 32-lane reduction, matching all 1,326,080 retained MPS words; separate multiplication
  differs in 381,526 up and 491,988 down words. Coverage is one layer of one request, 27 experts
  and two incoming states, without real 10- or 14-row cases. `omni-tiny-subgroup-replay-20261016/`,
  `omni-tiny-mps-tree-20261016/`, `omni-tiny-dot4-replay-20261016/`
- **MPS RMS mean.** The MPSGraph mean's rounding depends on the row count: the 298 shared rows of
  the 298-row and 2,529-row fixtures differ on 76 means. Fixture-matching reduction trees use 256
  lanes at sampled counts 7–298, 64 at 511–513 and 32 at 1,023–4,097; no general dispatch rule is
  established. `omni-rms-shape-validation-20261015/`
