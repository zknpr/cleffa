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
