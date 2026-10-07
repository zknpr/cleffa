# Performance history

The retained engine changes since the `db38cfc` baseline, in order, with the measurements
that qualified each one. Rejected variants are in [rejected experiments](rejected-experiments.md);
the attention and cache work has its own reports ([attention](attention.md),
[prefix cache](prefix-cache.md)); hosted-API measurements are in the
[hosted comparison](hosted-comparison.md).

All measurements are from the one tested M5 Max (40 GPU cores, 128 GB, macOS 27), one GPU
job at a time. Unless stated otherwise, a time is CLI inference including the CPU head and
excluding request encoding, model load and HTTP, taken as the median of paired calls in
which the two arms alternate A, B, B, A ("ABBA quartets") inside one resident process.
Absolute times from different runs are not comparable: GPU clock and temperature move them
by several percent, so each change was judged only against its own paired baseline. Every
retained change reproduces the previous build's logits bit for bit unless the entry says
otherwise.

## Status, 2026-10-05

Retained results at the 2026-10-05 stopping point:

| Workload | Verified result | Qualification |
|---|---|---|
| Fresh 16,347-token requests | Latest retained attention change reduces latency 8.6% on Flash and 7.4% on 27B against the FP32 control | Sustained paired tests; excludes model loading, encoding and HTTP |
| Changed tail after a shared 15K preamble | About 0.28–0.35 s on Flash and 0.94–1.21 s on 27B | Opt-in checkpoint cache; first fills still process the full input |
| Actual article audit | About 13% higher throughput than the older build on 96 saved requests | No observed decision, routing or flag changes on that comparison |
| Labeled quality | All 6,256 ContractNLI decisions unchanged; another 170 long-document decisions evaluated | Finite task coverage; probabilities are not bitwise identical to FP32 |
| Public full-context replay | All 46 FP32 decisions retained per model, with full token counts | 22 requests per model, through 16,347 tokens |

These measurements use different workloads and timing protocols; their absolute times should
not be combined into a single before/after curve.

The current serialized category profile attributes about 70% of long Flash GPU time to
matrix projections, 18% to attention and 6% to recurrence. Matrix work accounts for 87.5% of
the article workload's profiled GPU categories. The separate counter experiments show high
accelerator utilization, while complete-request tracing finds less than 1% gaps inside the
GPU span. These observations support stopping speculative tuning; they do not prove that
every kernel is optimal.

Requirements the work was held to, with the evidence and its limits:

| Requirement | Verified evidence | Limit |
|---|---|---|
| Preserve accepted context | CLI and server reject truncation by default. The current public replay checks full counts for every request through 16,347 tokens. | Low-level encoding defaults retain reference-compatible behavior; explicit CLI/server truncation remains an opt-in. |
| Preserve original weights | The retained engine uses the original BF16 model weights. | Compensated tensor-op attention changes arithmetic. It is not bitwise equivalent to the FP32 control. |
| Preserve task quality | 6,256 ContractNLI predictions are unchanged across the compared attention modes. The current public replay matches all 46 FP32 decisions per model. | Probability errors move in both directions. Only three ContractNLI test documents reach 8,192 tokens; this is not universal long-context accuracy proof. |
| Qualify cache reuse | Both attention modes and both models pass checkpoint checks. The CPU planner audit covers 6,016 requests, allocation failures and a corrupted-token negative control. | Prefix reuse needs shared leading input. It adds memory and requires tenant-isolated server keys. |
| Improve fresh requests | Paired sustained tests at 16,347 tokens show 8.6% lower latency for Flash and 7.4% for 27B from the retained attention change. | These gains do not close the gap to saved full-count hosted observations. |
| Improve repeated long prefixes | A new tail after a shared 15K preamble takes about 0.28 to 0.35 seconds on Flash and 0.94 to 1.21 seconds on 27B in the audited paired experiment. | First fills still process the full input. These timings are not comparable to unrelated-article throughput. |
| Evaluate both article workflows | Both the seven-field rewrite and three-field warning path were tested on 66 existing operator labels with unchanged article state. | Neither passed the frozen no-regression criterion; neither was adopted. |
| Avoid paid testing | This audit uses saved hosted responses and makes zero API calls. | The hosted evidence describes its capture date, not a current deployment guarantee. |

## Attention kernel unrolling, 2026-10-03

Unrolling the three loops that access the `attention_fa` accumulator makes each fragment
index a compile-time constant. All three loops must be unrolled together; unrolling only the
update did not improve performance. The Q.K loop stays compact. The standalone kernel
measured 2.23–2.42× faster at 8k/16k tokens with identical output bits. Arithmetic order,
precision, weights, and record boundaries are unchanged.

Measured against `db38cfc` on the same M5 Max 128 GB, macOS 27.0.1, with default FP16
activations:

| Model | Input tokens | Before | After | Speedup |
|---|---:|---:|---:|---:|
| clef-flash | 8,072 | 4.43 s | 3.40 s | 1.30× |
| clef-flash | 16,347 | 13.40 s | 8.80 s | 1.52× |
| Clef 27B | 8,072 | 14.51 s | 12.07 s | 1.20× |
| Clef 27B | 16,347 | 40.97 s | 29.49 s | 1.39× |

These are per-request medians of three runs per binary, in before/after order B A A B B A.
Each run starts a process, warms it with three short requests, then runs the 22-request
corpus back to back at batch size 1. `--time --logits` measures inference including the CPU
head; these are neither HTTP round trips nor `CLEF_PROFILE` timings. GPU temperature and
clock changes still affect the results. Short requests vary in both directions and show no
consistent gain.

For a synthetic seven-question rubric with 1,266–1,341 tokens per record and batch size 8,
16 requests took 26.94 s before and 26.45 s after (sum of per-batch medians, three runs
each, same B A A B B A order, one eight-record warm-up per process). That is 773 to 787
tokens/s, only 1.019× and small relative to run-to-run variation. Eight independent
1.3k-token records do not have the attention cost of one 10k-token record. The CPU head took
about 0.2 s per batch in a separate instrumented run, around 1–2% of total time. Most schema
tokens follow the state, so their causal representations cannot be reused exactly across
different states. Separate `CLEF_PROFILE=1` runs on this shape put 84% of baseline GPU time
in GEMMs and 7% in attention. Attention fell from about 0.92 s to 0.49 s per batch; GEMMs
still took about 11 s.

Every corpus logit is byte-identical to the baseline in all six runs for both models. Both
still match all 46 FP32-oracle decisions, with maximum probability errors of 0.0002 (flash)
and 0.0010 (27B). Batch invariance and NaN-poison checks pass for both; the flash
overflow/fallback suite also passes. This optimization preserves the existing numerical
accuracy.

`bench/attention_bench.m` compiles the production shader directly, checks sampled outputs
against float64, tests packed records in both output precisions, and checks unwritten
outputs and tail guards. It can compare timing and every output bit against an older source:

```sh
make attention-bench
./attention-bench 8072 16347
./attention-bench --baseline path/to/older-clef.metal 8072 16347
```

## GEMM tile tuning, 2026-10-03

FP16 backbone GEMMs generally use 64x128 tiles at 1,024 packed tokens or more. Short inputs
retain 32-row tiles, with the later width tuning and long 27B down-projection exception
described below. Weights, activation precision and FP32 accumulation are unchanged. BF16
fallback and FP32 head projections retain their existing tiles.

With the attention optimization already enabled, a controlled 27B rubric run improved from
14.51 s to 13.07 s per eight-record batch, 1.110x throughput. It repeated the same
10,387-token batch in one process: four warm-up batches followed by four samples per tile in
the order 32, 64, 64, 32, 64, 32, 32, 64. Every logit was identical. A separate six-process
comparison gave only 1.026x, so the size of the gain varies with the measurement conditions.
These are batch timings, not single-request latency targets.

The tested FP16 GEMMs have identical output bits across the tile sizes, including residual
accumulation and shifted row offsets. `bench/gemm_tiles.m` compiles the production shader,
checks sampled outputs against float64, and poisons input tails and output guards. Its
default suite covers every row count from 1 through 65 with ragged K/N, plus both models'
main FFN shapes and the 27B DeltaNet input projection at 1,023, 1,024 and 1,025 rows. The
long 27B down-projection boundary is also checked at 4,095, 4,096 and 4,097 rows. The suite
now includes grouped expansion tiles at that boundary and partial groups, for 190 shape/mode
checks.

The October 5 follow-up interleaves four tile rows for long 27B FP16 expansion projections
(`T >= 4096`, `K >= 5120`, `N >= 4*K`). This changes tile scheduling while preserving each
element's reduction. Two paired full-request runs at 16,347 tokens measured 3.15% and 1.03%
lower latency with identical logits. Gains at 4.5K and 8K were not established. Flash
retains its previous order because its isolated GEMM improvement did not carry through to
requests. See the [tile-order results](#grouped-27b-expansion-tile-order-2026-10-05).

```sh
make test-gemm
./gemm-tiles 10387 17408 5120  # exact checks plus alternating GPU timings
```

After this change, both models pass the full 22-record batch and NaN-poison suites and all
46 FP32-oracle decisions, with unchanged maximum probability errors. Forced-overflow checks
pass on the first 20 flash corpus records, including mixed FP16/BF16 batches across the tile
threshold. Host and error-propagation tests pass. A 50-record replay of the existing rubric
job preserves every saved response byte for byte.

## Short 27B GEMM tiles, 2026-10-04

Below 1,024 packed tokens, FP16 GEMMs with both K and N at least 5,120 now use 32x256 tiles.
This covers the 27B backbone; flash keeps 32x128 at these lengths. The large-batch 64x128
path and both models' BF16 fallback and FP32 head GEMMs are unchanged.

Two single-process experiments compared the old tile with the wider tile on identical
requests, with CPU-head packing enabled throughout. All logits were byte-identical. Each
process warmed with six requests, then interleaved three modes: original tiles, wider tiles,
and an expansion-only variant. The first experiment measured six calls per mode; the second
measured twelve with balanced permutations of the three modes. The full wider-tile variant
was retained:

| 27B input tokens | First run, original → wider | Second run, original → wider |
|---|---:|---:|
| 146 | not measured | 218.55 → 194.45 ms |
| 300 | 364.85 → 357.35 ms | 364.15 → 363.65 ms |
| 346 | 425.60 → 401.75 ms | 447.90 → 422.40 ms |
| 600 | 796.40 → 675.85 ms | 960.65 → 722.90 ms |

These are per-mode medians of CLI inference time including the CPU head, not HTTP latency.
The 300-token case shows no consistent gain. The 346-token case improves about 6% in both
runs; at 600 tokens the throughput gain varies from 18% to 33%. Absolute timing varies with
GPU clock/temperature, so neither the largest gain nor these timings should be generalized
to every request. The expanded `make test-gemm` compares all three production tiles in both
multiply and accumulate modes, including shifted packed rows and ragged edges.

## Long 27B down-projection tiles, 2026-10-04

At 4,096 or more packed tokens, FP16 down projections with N >= 5,120 and K >= 2*N use
32x256 tiles. This selects the 27B FFN down projection. In same-process comparisons that
alternate current/wider/wider/current dispatch, complete inference improved modestly:

| 27B input tokens | Previous median | Wider down projection |
|---|---:|---:|
| 4,510 | 5,704.75 ms | 5,616.35 ms |
| 16,347 | 27,716.20 ms | 27,289.85 ms |

Each trial discards four warm-ups. The 4,510-token trial measures six calls per mode; the
16,347-token trial measures four. All logits match, and every measured four-call round
improves. This is about a 1.5% latency reduction in both trials. Flash's result varied in
sign, so its down-projection dispatch was retained. The wider tile preserves the existing
operands and reduction order; the full-input latency target remains unmet.

All 166 GEMM shape/mode checks and both 46-question FP32-oracle suites pass. The 27B's
22-record normal and NaN-poisoned batch-1/batch-8 outputs are byte-identical to the saved
baseline. Long-request residual rows and forced BF16 fallback also match that baseline.

See [rejected experiments](rejected-experiments.md#gemm) for the rejected persistent, specialized and attention
variants, profiling evidence and validation details.

### Experiment record

The complete-request comparison loads each model once and alternates
current/wider/wider/current dispatch. Four warm-ups precede twelve measured requests, six
per mode. The wider tile is selected only for FP16 matrices with at least 1,024 rows and an
input width at least twice their output width. At 4,510 tokens:

| Model | Current median | Wider down-projection median |
|---|---:|---:|
| Flash | 1,499.75 ms | 1,474.55 ms |
| 27B | 5,704.75 ms | 5,616.35 ms |

Every logit matched across all 16 calls per model. Flash's three measured quartets varied in
sign; 27B improved in all three. A follow-up at 16,347 tokens uses the same order with four
warm-ups and eight measured calls. The 27B median falls from 27,716.20 to 27,289.85 ms, with
identical logits across all 12 calls and improvement in both measured quartets.

The production selection is narrower than the experimental switch: at least 4,096 packed
tokens, FP16 input, N >= 5,120 and K >= 2*N. This retains the measured 27B down-projection
gain while leaving Flash on its existing dispatch. The approximately 1.5% latency reduction
does not meet the overall latency target. `make test-gemm` now includes the new boundary at
4,095, 4,096 and 4,097 rows, in both multiply and accumulate modes with shifted packed rows.

The rebuilt engine passes all 166 GEMM cases and both FP32-oracle suites with first-record
layer dumps: 46/46 decisions per model, with unchanged maximum probability errors of 0.0002
for Flash and 0.0010 for 27B. All 22 normal 27B outputs match the saved baseline byte for
byte. NaN-poisoned batch-1 and batch-8 runs also match that baseline exactly. On the
4,510-token fixture paired with a short request, the last 33 residual rows of every layer
match in normal mode, and forced all-producer overflow at batch 1 and batch 8 matches the
baseline's BF16 output. Commands and checks are recorded in
`golden/perf-persistent-20261004/retained-validation.json`; the paired performance samples
and rejected probes are in `experiment-summary.json` there. Its inputs and responses stay in
memory; only the aggregate result is saved in `replay.json`.

## Chunked 27B DeltaNet, 2026-10-04

27B records with at least 4,096 tokens use a 32-token block formulation of the gated delta
rule. All preprocessing and state products remain FP32. The selection uses each record's
length, so packing it with another request preserves its results. Flash retains the
sequential recurrence because the corrected block implementation was slower in complete
inference.

On the M5 Max, the qualification build alternates sequential/chunked/chunked/sequential
dispatch in one process, with four warm-ups and four measured requests per mode:

| 27B input tokens | Sequential recurrence | Chunked recurrence |
|---|---:|---:|
| 4,510 | 5,634.15 ms | 5,492.65 ms |
| 8,072 | 11,663.55 ms | 11,480.65 ms |
| 16,347 | 26,248.35 ms | 25,792.45 ms |

Both measured rounds improve at every length, by about 1.6–2.5% overall. The arithmetic
order changes, so logits can differ from the sequential kernel. Both models retain 46/46
FP32-oracle decisions; the 27B maximum probability error falls from 0.00104 to 0.00074.
These corpus results do not establish a general accuracy improvement.

The prepass sums decay intervals directly. Subtracting large FP32 prefix sums can erase
subsequent small decays or produce NaN after overflow. `make test-gdn` covers that
regression, all 32 record offsets and tails, NaN guards, sampled float64 results, and
failure/recovery of each scratch allocation. The two value-column tiles return identical
bits. Scratch grows atomically, is reused between records and layers, and costs about 1.3 GB
at 16,347 tokens. The final production-dispatch comparison repeats the long case at
26,325.30 versus 25,708.25 ms, a 2.3% reduction, with both measured rounds improving. See
[the production selection record](#production-selection) for the localhost timings.

### Experiment record

The experiment under `golden/perf-delta-20261004/` replaces the sequential rank-one
recurrence with 32-token blocks. A prepass forms the block's lower-triangular inverse and
transformed keys and values; a second kernel carries the FP32 state between blocks using
full-precision MPP products. Neither weights nor operands use reduced precision. A float64
implementation of the equations matches the sequential recurrence within 1e-12 on the tested
lengths and decay distributions.

Increasing the original scan from eight lanes per value column to sixteen or thirty-two was
slower. An eight-token SIMDgroup-matrix implementation and a 64-token MPP implementation
were also slower. The best 32-token implementation reads Q/K/V through strided device tensor
views, avoiding an additional threadgroup copy. Its prepass explicitly masks padded rows
before multiplying coefficients; multiplying a zero mask by an undefined NaN is
insufficient. Selected ragged lengths from 1 to 65 pass after that correction.

The isolated scan comparison includes both preprocessing and state propagation. Four
warm-ups precede eight measured calls per variant in alternating order. Representative
medians are:

| Tokens / value heads | Production scan | Chunked scan |
|---|---:|---:|
| 4,510 / 32 | 3.3212 ms | 2.9361 ms |
| 4,510 / 48 | 4.9115 ms | 4.2533 ms |
| 16,347 / 32 | 12.5433 ms | 10.9356 ms |
| 16,347 / 48 | 19.8609 ms | 15.4133 ms |

These synthetic cases check finite outputs, NaN guards and sampled float64 recurrence
results. Sampled absolute error falls from roughly 1e-6 to below 5e-8, but output bits
differ from the production recurrence. Two accidentally overlapping lane-probe runs were
discarded and repeated serially; the reported measurements use one GPU job at a time.

Both models retain 46/46 FP32-oracle decisions and pass the existing logit and probability
error bounds. Mean absolute probability error changes from 2.19e-5 to 2.10e-5 for Flash and
4.99e-5 to 4.52e-5 for 27B. Mean absolute logit error increases from 4.43e-4 to 4.87e-4 and
5.88e-4 to 5.97e-4 respectively. These results do not establish an overall quality
improvement.

The first whole-engine prototype uses single-record dispatch, so those initial measurements
do not establish packed-batch invariance. At 16,347 tokens on 27B, preprocessing needs
approximately 1.3 GB of extra scratch. The initial prototype was kept out of production.

The initial complete-request comparisons load each model once, then use four warm-ups
followed by two ABBA rounds, giving four measured calls per mode. Truncation is disabled,
token counts match the full fixtures, and repeated logits are identical within each mode.

| Model / tokens | Production recurrence | Chunked recurrence |
|---|---:|---:|
| Flash / 4,510 | 1,558.55 ms | 1,550.30 ms |
| 27B / 4,510 | 5,891.15 ms | 5,807.05 ms |
| Flash / 16,347 | 8,067.30 ms | 7,955.05 ms |
| 27B / 16,347 | 26,397.80 ms | 25,766.30 ms |

The initial longest 27B case improves by 2.4%, with gains in both rounds. Flash's shorter
case has one effectively unchanged round. A subsequent numerical stress test invalidated
this version for production: after a -1e8 decay, subtracting FP32 prefix sums erases the
following -0.02 decays. The original recurrence passes the float64 bound while the chunked
output differs by 0.00210 on a reference value of 0.00570. `model-time.json` contains every
measured sample and the experimental binary hash; `model-quality.json` contains the separate
FP32-oracle comparison.

The corrected prepass sums each decay interval directly. This avoids cancellation and also
handles finite negative decays whose prefix sum overflows. Both the -1e8 and -FLT_MAX stress
cases now pass, with maximum sampled error 1.08e-8 versus production's 1.26e-7. All 32 tail
lengths pass on both head counts before and after the correction, including finite-output,
NaN-guard and sampled float64 checks. Its 16,347-token, 48-head isolated scan takes 17.7318
ms versus 20.2673 ms for production in the same run, including preprocessing.

Both models still pass all 46 FP32-oracle decisions. For the corrected 27B candidate, mean
absolute logit error is 5.82e-4 versus production's 5.88e-4, and mean absolute probability
error is 4.34e-5 versus 4.99e-5. A fresh complete-request ABBA comparison at 16,347 tokens
measures 26,248.35 ms for production and 25,792.45 ms for the corrected candidate, a 1.7%
reduction with improvement in both measured rounds. Those results are in
`model-quality-stable.json` and `model-time-stable.json`. The correction's measurements
supersede the earlier 2.4% result.

A subsequent experimental host selects the algorithm by each record's length, binds record
offsets explicitly and reuses scratch across serial record dispatches. Both models'
22-record batch-8 outputs match the corrected standalone logits exactly, including with all
activation and new preprocessing scratch buffers poisoned. A short/long/short fixture also
returns the candidate's BF16 reference exactly under forced all-producer overflow at batch
sizes 1 and 8. The last 33 rows of all 66 saved residual tensors for the 27B 16,347-token
request pass against the safe FP32 oracle, with maximum relative L2 error 0.003253 below the
0.01 bound. This experimental validation precedes the production integration below. Evidence
is in `packed-validation.json`, `long-layer-validation.json` and
`tail-validation-stable.json`.

### Production selection

Further corrected-kernel ABBA trials cover the other dispatch ranges, with four warm-ups and
four measured calls per mode. At 4,510 tokens, 27B improves from 5,634.15 to 5,492.65 ms; at
8,072 tokens, it improves from 11,663.55 to 11,480.65 ms. Both measured rounds improve in
each case. Flash instead regresses at 16,347 tokens, from 7,963.55 to 8,134.15 ms, with both
rounds slower. `model-time-dispatch.json` records the samples.

The production selection therefore uses chunked recurrence only for Hv=48 records with at
least 4,096 tokens. Each record gets 16-column state tiles below 8,192 tokens and 32-column
tiles thereafter. All other records keep the sequential recurrence. The host selects by
individual record length in both ordinary and BF16-fallback forwards. A batch containing a
selected record dispatches scans separately, reusing scratch in the serial encoder; batches
of short records retain their original combined scan dispatch.

`ensure_gdn_capacity` allocates all five scratch buffers before replacing the previous set,
checks the rounded row count and device buffer limit, and propagates failure. The existing
poison hook covers the new buffers, and model close releases them. The peak extra allocation
is approximately 1.3 GB at 16,347 tokens on 27B; Flash allocates none.

The tracked `make test-gdn` suite passes 144 shape/head cases at all 32 record offsets, with
sampled float64 recurrence checks, beta endpoints, weak/strong decays, overflowing negative
prefix sums, NaN guards, and exact equality between 16- and 32-column chunk tiles. The same
regression fails on the earlier prefix-subtraction shader because it produces non-finite
output. `tests/test_gdn_buffers.m` injects each of the five allocation failures and checks
buffer identity, capacity preservation, recovery, length overflow and dispatch boundaries.

The rebuilt production CLI passes `make test`, `make test-errors`, both 46-question FP32
parity suites with first-record residual dumps, and exact batch-1/batch-8/poison comparisons
on both 22-record corpora. It matches the qualified corrected candidate on 27B and the saved
sequential baseline on Flash. Forced all-producer overflow on a short/long/short fixture
matches each model's BF16 reference at batch sizes 1 and 8. The final 33 residual rows of
all 66 tensors for the long 27B case pass the safe FP32 oracle, with maximum relative L2
error 0.003253. Evidence for this build is under `golden/perf-delta-retained-20261004/`.

A final same-process ABBA build wraps the production dispatch, changing only whether
`gdn_chunked` selects the new path. Four warm-ups precede four measured calls per mode at
16,347 tokens. Median latency is 26,325.30 ms with the sequential recurrence and 25,708.25
ms with chunking, a 2.3% reduction. Both rounds improve, and each mode's logits match its
independently validated output. `production-time.json` contains every sample and the paired
binary hash. This repeated gain remains small relative to the full-input hosted-API target.

`retained-validation.json` records successful commands and hashes. All checks completed
before the concluding hash comparison; no production code changed during validation.

The rebuilt localhost server also completes the public checkout fixtures with full token
counts and deterministic repeated responses. This follow-up uses six measured calls for the
exact blog request and two shuffled passes over each padded fixture:

| Fixture | Full tokens | Flash median | 27B median |
|---|---:|---:|---:|
| Blog example | 346 | 135.00 ms | 416.38 ms |
| 8 KiB, outage at end | 1,382 | 479.65 ms | 1,545.20 ms |
| 16 KiB, outage at end | 2,424 | 871.68 ms | 2,832.88 ms |
| 32 KiB, outage at end | 4,510 | 1,631.06 ms | 5,444.30 ms |

These HTTP measurements are a fresh absolute snapshot, not an A/B speedup calculation
against older runs under different thermal conditions. They use synthetic filler, not the
unavailable external chart payloads.

## CPU head: packed weights, scorer batching and two workers, 2026-10-04

The CPU decision head prepares padded, transposed copies of its multi-row projection weights
once at model load. This speeds up Accelerate's small GEMMs. Single-row calls and ragged
weight dimensions keep the original layout because transposing those can change BLAS
reduction order. Weights and activations retain the same FP32 values.

An instrumented build alternated the original and packed layouts on the same GPU-produced
head inputs. Each forward ran four warm-up calls followed by six measured calls per layout,
in repeated original/packed/packed/original order. The table gives medians of these
per-forward means across four short forwards or three long forwards:

| Model | Input tokens | Original CPU head | Packed CPU head | Time saved |
|---|---:|---:|---:|---:|
| flash | 346 | 11.67 ms | 8.97 ms | 2.70 ms |
| 27B | 346 | 12.55 ms | 9.46 ms | 3.10 ms |
| flash | 4,510 | 25.28 ms | 22.43 ms | 2.85 ms |
| 27B | 4,510 | 29.11 ms | 25.06 ms | 4.05 ms |

These are CPU-head savings; the backbone still dominates total latency. This layout change
added 376.25 MiB for flash and 392.75 MiB for the 27B, freed with the model. Preparation
adds startup work but no per-request packing. The layout change alone kept every logit
bit-identical across all 16 head calls for both 22-record corpora and the checkout fixtures.
`make test` now includes 78 layout checks covering single-row fallback, ragged dimensions,
padded tails, large row counts, allocation failure and the decoder's shared-weight
ownership.

The residual scorer now batches options within each record when there are at least four
options. Its extra packed FP32 projection adds 16.5 MiB per model, bringing total
packed-weight overhead to 392.75 MiB for Flash and 409.25 MiB for 27B. A direct comparison
on identical GPU-produced inputs saves 0.4–1.2 ms of CPU-head time on the 300-, 362- and
594-token fixtures, with improvement in all 36 alternating comparisons. Complete inference
timings show a small Flash gain; 27B timings are noisier. This does not close the long-input
GPU gap.

Both models retain 46/46 FP32 decisions and exact batch/poison/overflow invariance. The
changed BLAS reduction moves public-corpus logits by at most 9.54e-7. Decisions, scores,
confidence and noul outputs all remain unchanged. Six additional CPU tests compare the
scorer with float64 arithmetic. See the [measurement and validation
record](#cpu-head-packed-weights-scorer-batching-and-two-workers-2026-10-04).

Large CPU head projections and attention calls now use two workers. On paired measurements
from 346 to 16,347 input tokens, head time falls by 28–39%, saving roughly 2–18 ms. The
backbone dominates complete-request time: Flash improves modestly in these samples; a
whole-request 27B improvement is not established. See the [head measurements and
checks](#cpu-head-packed-weights-scorer-batching-and-two-workers-2026-10-04).

### Host processing and residual-scorer batching

A standalone CPU probe measures JSON parsing, strict request encoding, response formatting
and cleanup. It links only libSystem, loads tokenizer metadata from the GGUF, and rejects
input truncation. The sources are isolated copies of the current host files. Model startup,
network I/O, queueing, the CPU head and GPU inference are outside these measurements.
Formatting uses uniform synthetic probabilities and includes their allocation; it does not
use real model predictions.

| Public fixture | Tokens | Host processing median |
|---|---:|---:|
| Checkout blog example | 346 | 0.054 ms |
| Checkout 8 KiB, outage at end | 1,382 | 0.249 ms |
| Checkout 32 KiB, outage at end | 4,510 | 0.798 ms |
| Corpus r020 | 8,072 | 0.279 ms |
| Corpus r021 | 16,347 | 0.514 ms |

All nine checkout fixtures and 22 corpus fixtures have three warm-ups and fifteen measured
repetitions. All 558 repeated encodings preserve exact token IDs, question/option IDs,
types, spans and formatted response bytes. Token counts also match `clef-tool encode-strict`
for every fixture. None of these host-processing medians reaches 0.8 ms, so this path does
not explain the observed inference gap. Sources, raw timings, source hashes and the
validating `summarize.py` are in `golden/perf-host-20261004/`.

A separate CPU-only prototype batches the residual scorer across options within one record.
The previous head applied its 4,096-to-1,024 projection separately to every option, followed
by GELU and a scalar projection. The prototype assembles the same features in the existing
scratch space and applies those two projections to the option matrix. Records with fewer
than four options retain the single-option path. The packed first projection costs another
16.5 MiB of exact FP32 weights; no weight or activation precision changes.

On synthetic features and actual model weights, nine-option scorer medians fall from 0.803
to 0.283 ms for Flash and from 1.013 to 0.300 ms for 27B. At 32 options they fall from 2.692
to 0.348 ms and from 3.065 to 0.361 ms respectively. Two-option batching is slower, which
motivates the small-record fallback. The scorer output changes by at most 1.91e-6 across the
measured one-to-64-option cases; these are arithmetic-order changes, not exact bit parity
between the original and candidate.

The complete head is replayed from the existing FP32 `final_norm` tensors for r000–r002,
with hidden normalization and memory K/V preprocessing also computed on the CPU. Alternating
baseline/candidate calls give the following head-only medians:

| Model | r000 | r001 | r002 |
|---|---:|---:|---:|
| Flash, original → batched | 8.394 → 7.948 ms | 7.905 → 7.753 ms | 8.401 → 8.005 ms |
| 27B, original → batched | 8.599 → 8.240 ms | 8.075 → 7.898 ms | 8.676 → 8.330 ms |

Both variants agree with all 16 reference decisions. Candidate maximum probability error
against FP32 is 1.43e-7; maximum candidate-versus-original logit change is 4.77e-7. These
checks cover six existing reference records, not the full model corpus. AddressSanitizer and
UndefinedBehaviorSanitizer also pass on those records and fourteen additional synthetic head
cases spanning two to 64 options. All twenty cases preserve exact candidate logits when
memory rows move by 31 positions with NaN padding before and after them. Repeated calls
within each mode are exact and all outputs are finite.

The isolated experiment is under `golden/perf-head-score-20261004/`. Its engine supports
`CLEF_BATCH_SCORE=1` and builds separately from the main worktree. The qualified scorer
batching was subsequently retained in production without that experimental environment
switch.

All 46 FP32 decisions pass for each model. The maximum change from the previously validated
production logits is 9.536743e-7 for each model. All 22 records have exact candidate logits
across batch sizes 1 and 8, with both clean and NaN-poisoned buffers. A short/long/short
packed batch also preserves the per-record BF16 overflow fallback exactly. The 14 completed
public checks, source hashes and logs are saved in `validation.json` and its sibling files.

An alternating ABBA benchmark uses the same loaded engine and switches only scorer mode.
Each fixture has two warm-ups and twelve measured calls per mode. It times `clef_run_ex`,
including GPU inference and the CPU head, but excludes loading, encoding and HTTP transport.

| Model | Fixture / tokens | Original median | Batched median | Batched wins / six quartets |
|---|---|---:|---:|---:|
| Flash | r000 / 300 | 121.882 ms | 120.475 ms | 5/6 |
| Flash | r009 / 362 | 145.914 ms | 145.454 ms | 5/6 |
| Flash | r018 / 594 | 229.245 ms | 227.629 ms | 5/6 |
| 27B | r000 / 300 | 413.462 ms | 412.787 ms | 3/6 |
| 27B | r009 / 362 | 481.563 ms | 478.484 ms | 4/6 |
| 27B | r018 / 594 | 744.850 ms | 743.651 ms | 4/6 |

The Flash run supports a small improvement. The 27B differences are noisy, so their median
changes are not a firm effect-size estimate. Direct head timings on identical inputs from a
real GPU forward separate the change from that GPU timing noise:

| Model | Tokens | Original head | Batched head | Saving |
|---|---:|---:|---:|---:|
| Flash | 300 | 8.522 ms | 8.115 ms | 0.407 ms |
| Flash | 362 | 6.575 ms | 5.913 ms | 0.662 ms |
| Flash | 594 | 11.441 ms | 10.325 ms | 1.116 ms |
| 27B | 300 | 8.849 ms | 8.438 ms | 0.411 ms |
| 27B | 362 | 6.946 ms | 6.170 ms | 0.777 ms |
| 27B | 594 | 11.839 ms | 10.645 ms | 1.195 ms |

All six ABBA quartets improve for every row, 36/36 overall. Each mode has two warm-ups and
twelve measured head calls. The instrumentation returns the original mode's logits, which
exactly match the saved production baseline, and does not change GPU work. Samples and
analysis are in `head-time-summary.json` and its sibling logs.

A separate in-memory diagnosis confirms that only one option-probability entry changes after
four-decimal rounding, by 0.0001. Its raw logit change is at most 4.77e-7 and its unrounded
softmax probability change at most 9.82e-8. No choice, score, confidence, noul output or
argmax changes. The numeric change is within the FP32 parity contract, so the optimization
is retained with this rounding difference documented.

The extended CPU linear check passes all 78 layout cases and six residual-scorer cases
against a float64 calculation, including NaN padding and output bounds. It is now part of
`make test`, which passes on the production rebuild. Production-to-candidate binding checks
are recorded in `production-validation.json`.

### Two workers

The CPU head now splits large linear projections across two workers by output columns, and
splits larger attention calls by independent heads. It processes every input token, keeps
FP32 arithmetic, and leaves the GPU backbone unchanged.

This is a small part of complete-request latency. It does not resolve the backbone's context
scaling or establish Cloudflare-equivalent full-input latency.

#### Implementation

`lin_cols` retains the original input reduction and output stride. Two workers handle
32-column-aligned halves when the output width is divisible by 64 and the weight matrix has
at least 1,048,576 elements. Other shapes use one BLAS call. Bias is added after both
workers finish. The tests compare packed and original weight layouts against unsplit BLAS.

Attention splits even head counts when `query_rows * keys * heads >= 65,536`. Each worker
has separate score scratch, reads the same K/V buffers and writes disjoint output columns.
The BLAS calls and softmax within each head are unchanged. This doubles score scratch for
selected calls; allocation failure still propagates before output is written.

The original candidate read its debug threshold through an unsynchronized static flag. An
eight-thread first-use regression reproduced a ThreadSanitizer race. `dispatch_once` fixes
that race. `CLEF_DEBUG_HEAD_SPLIT_MIN=0` disables the linear split only; the setting is read
once per process.

#### Measurements

M5 Max, 40 GPU cores, 128 GB RAM. Each case reuses one backbone result and alternates the
original and parallel head in ABBA order: four warm-ups, then twelve measurements per arm.
Every measured raw logit matches. All sixty paired quartets favor the parallel head.

| Model | Input tokens | Original head | Parallel head |
|---|---:|---:|---:|
| Flash | 346 | 9.076 ms | 6.466 ms |
| Flash | 1,382 | 12.660 ms | 8.203 ms |
| Flash | 2,424 | 14.962 ms | 10.029 ms |
| Flash | 8,072 | 29.971 ms | 18.543 ms |
| Flash | 16,347 | 53.492 ms | 35.511 ms |
| 27B | 346 | 8.736 ms | 6.193 ms |
| 27B | 1,382 | 11.883 ms | 7.782 ms |
| 27B | 2,424 | 15.255 ms | 10.192 ms |
| 27B | 8,072 | 30.316 ms | 18.707 ms |
| 27B | 16,347 | 53.680 ms | 35.373 ms |

Separate complete-request comparisons use four measured calls per arm:

| Model | Input tokens | Original request | Parallel request |
|---|---:|---:|---:|
| Flash | 346 | 127.078 ms | 124.774 ms |
| Flash | 1,382 | 463.544 ms | 458.350 ms |
| 27B | 346 | 377.420 ms | 384.499 ms |
| 27B | 1,382 | 1,513.932 ms | 1,509.139 ms |

Both Flash quartets improve at each length. The 27B 346-token quartets regress by 0.61% and
2.19%; the 1,382-token quartets have opposite signs. The head component improves
consistently, but these samples do not establish a complete-request speedup for 27B. The GPU
is unchanged and dominates those requests. Do not apply the head's percentage reduction to
total latency or interpret it as a new long-context performance result.

#### Qualification

The isolated candidate passed:

- All 22 public records on each model, at batch sizes 1 and 8 and with NaN poisoning: exact
  raw logits against the preceding production build, preserving its 46/46 FP32 reference
  decisions per model.
- A mixed 4,510 + 346-token batch on each model with forced FP16 overflow and poisoning:
  exact raw logits against the preceding build's BF16 fallback.
- Flash HTTP/CLI parity for 23 cases, 32 concurrent clients, protocol/limit checks and idle
  keep-warm passes, with activation poisoning enabled.
- `make test`: 78 linear layout/reference checks, the existing scorer float64 checks, ten
  new attention cases, first-use concurrency, JSON/tokenizer and existing host checks.
- The new attention cases cover both sides of the parallel threshold, odd head counts,
  strided NaN padding, output guards, float64 samples and score allocation failure/recovery.
- ThreadSanitizer first-use check: original candidate fails with exit 70; corrected
  initialization passes. `make test-head-tsan` preserves this check.

The final source only shortens implementation comments relative to that candidate.
Production build verification is recorded separately below.

It does not erase the older scorer change's documented last-decimal probability difference.

#### Production build

The main checkout was rebuilt after applying the head source and tests. `make test`, `make
test-head-tsan`, `make test-errors` and the nine tokenizer truncation-boundary cases pass.
Both models' complete 22-record public corpora again produce identical raw logits at batch
size eight. The Metal source and generated shader hashes are unchanged.

The CLI also now rejects over-limit states by default, matching the server. `--truncate`
explicitly restores the reference behavior; `--no-truncate` remains supported. A new CLI
regression fails on the old binary and passes on the rebuilt one. It injects failure of the
first GPU command buffer to prove default rejection happens before inference, while explicit
truncation reaches inference with 16,384 tokens. Flag ordering is checked too.

Exact commands, logs and source hashes are in `production-validation.json` and its
associated logs under the experiment directory.

## Attention: shared probabilities, prefetch and rescaling, 2026-10-04

Reuse and prefetch attention now scale each row's accumulator elements directly. Every
accepted request still processes its full context; weights, operand precision and the 32-key
softmax/value-product order are unchanged. The original short-request attention kernel is
retained.

A same-process paired prototype measures 16,347-token request medians of 7,593.16 to
7,470.61 ms for Flash and 24,436.21 to 24,296.91 ms for 27B, reductions of 1.61% and 0.57%.
Four measured calls per arm follow four warm-ups. Medium-length results are mixed, so these
figures do not imply the same gain at every context length.

The retained kernel body matches the timed prototype; final-binary request timing has not
been repeated. See [the rescaling report](attention.md#direct-fragment-rescaling) for paired
samples, the focused 4,510-token repeat and compiler assumptions.

These three FP32-path changes (shared probabilities for long single requests and packed
batches, 64-key score prefetch, direct fragment rescaling) are reported in full, with their
kernel and request timings, in the [attention report](attention.md).

## Keep-warm for sparse traffic, 2026-10-04

Before this change, idle gaps added latency. On clef-flash with a ~260-token request:

| Gap since the previous request | Latency |
|---|---|
| back to back | 95 ms |
| 0.2–0.5 s | 117–120 ms |
| 1–3 s | 209–287 ms |

On the 27B, same request:

| Gap since the previous request | Latency |
|---|---|
| back to back | 306–311 ms |
| 0.3 s | 309–313 ms |
| 2 s | 572–600 ms |

A startup warm-up alone did not remove the penalty. Stage timings located the step before
GPU execution, and periodic buffer references remove it.

The server now keeps model and activation buffers active while idle. Its worker submits
read-only GPU passes every 500 ms, using a separate 16-byte output buffer. This removes a
large delay before GPU execution on the M5 Max used here. `--no-keep-warm` disables it;
`--keep-warm` explicitly selects the default.

### Measured effect

The public test request contains 248 tokens and two questions. Each arm starts its own
server, sends three initial requests, then measures three requests after five-second idle
gaps. The control runs first with `--no-keep-warm`; the second server uses the default. All
complete response bytes match within and across arms. These are localhost HTTP times,
including encoding and the CPU head, with model startup excluded.

| Model | Keep-warm off, median HTTP | Keep-warm on, median HTTP | Reduction |
|---|---:|---:|---:|
| Flash | 220.5 ms | 104.8 ms | 52.5% |
| 27B | 535.7 ms | 263.3 ms | 50.8% |

Flash's individual samples were 220.3, 220.5 and 221.3 ms with keep-warm off, versus 104.8,
107.9 and 102.5 ms with it on. The 27B samples were 531.5, 540.3 and 535.7 ms, versus 260.4,
263.3 and 265.7 ms. Three samples in sequential arms demonstrate the large idle-gap effect
here; they do not establish a precise percentage across workloads.

`CLEF_STAGE_TIME=1` records host encoding time, commit-to-completion wall time, GPU
execution time, and CPU head time. Subtracting GPU execution time from the
commit-to-completion wait estimates the delay before GPU execution, including host
scheduling overhead. Its median fell from 125.3 to 1.0 ms on Flash and from 283.3 to 1.1 ms
on the 27B.

The original worktree's control experiments found that a pass referencing the weights
removed the delay while the same pass referencing a small unrelated buffer did not. That is
consistent with buffer residency behavior, but the driver mechanism is unconfirmed. The
earlier explanation attributing the whole idle penalty to clock ramp-up was too strong.
Thermal and clock changes can independently affect kernel execution time.

### Implementation and limits

The same worker performs both idle passes and inference, so they cannot overlap within an
engine. It releases the request-queue mutex during GPU work and checks the queue again
afterwards. A request arriving during an idle pass waits for that pass to finish. The passes
read one element of each allocated buffer and write only the separate output buffer. They
work before the first inference, when activation buffers are still unallocated.

Missing command buffers or encoders and GPU execution failures propagate to the caller. The
server logs an idle-pass failure once and continues accepting requests; inference retains
its existing error handling.

This reduces latency for sparse traffic. It does not accelerate continuously busy CLI
batches or change the model's arithmetic, weights, precision, input length, or truncation
policy. It also does not establish a new equal-input comparison with Cloudflare: this
248-token fixture was not sent to the hosted service. Power and energy cost have not been
established for this combined build. Keeping the buffers active is a deliberate serving
policy; use `--no-keep-warm` to disable it.

### Validation and artifacts

The isolated combined candidate passes:

- Both models' complete 22-record corpus in batches of eight, normally and with NaN
  poisoning, with every raw logit identical to the preceding production build. That build
  agrees with all 46 FP32 decisions per model.
- Exact preservation of all allocated activation bytes across an idle pass, including
  padding; calls before the first forward; missing command-buffer and encoder errors;
  injected execution errors and successful recovery.
- Flash HTTP/CLI byte equality on all 23 cases, including over-length rejection, all 32
  concurrent clients, and protocol/limit handling, with activation poisoning enabled.
- All host checks. The initial isolated `make test` stopped because its copy omitted
  `tools/`; after copying the unchanged directory, the remaining two checks passed.
- The earlier scorer's documented last-decimal difference against older frozen outputs is
  unchanged.

`production-validation.json` records the retained build's checks separately.

The retained production build also passes a complete `make test` and exact-logit checks on
all 22 records for each model. Repeating the Flash idle test with that build gives 219.8 ms
without keep-warm and 102.3 ms with the default, again with identical responses.

### Reproduce

```sh
make
/usr/bin/lockf -k "$PWD/golden/.gpu.lock" .venv/bin/python -B tests/test_keep_warm.py gguf/clef-flash.gguf
/usr/bin/lockf -k "$PWD/golden/.gpu.lock" .venv/bin/python -B tests/test_keep_warm.py gguf/clef.gguf
```

The test requires its control to show at least 40 ms of median idle delay; otherwise it
returns exit 2 instead of claiming success without exercising the behavior. Every warm
sample must have less than 20 ms of estimated start delay. A GPU error or response-byte
difference fails the test. Stage timings describe an unsplit forward pass; do not combine
them with `CLEF_PROFILE=1`, which deliberately serializes kernel groups.

## Combined comparison, 2026-10-04

An instrumented build switches between `db38cfc`'s attention kernel, 32x128 GEMM tiles and
original CPU head layout, and all three optimized paths. It uses one process and the same
loaded model, four warm-up requests, then six measured calls per mode in repeated
original/optimized/optimized/ original order. The checkout fixtures below supply the inputs.
Every logit is byte-identical across both modes on all six model/input combinations.

| Model | Input tokens | Original | Optimized | Throughput gain |
|---|---:|---:|---:|---:|
| flash | 346 | 116.65 ms | 112.70 ms | 1.04× |
| flash | 600 | 183.55 ms | 178.65 ms | 1.03× |
| flash | 4,510 | 1,784.10 ms | 1,591.80 ms | 1.12× |
| 27B | 346 | 421.95 ms | 395.85 ms | 1.07× |
| 27B | 600 | 832.05 ms | 672.55 ms | 1.24× |
| 27B | 4,510 | 7,170.15 ms | 5,624.75 ms | 1.27× |

These are medians of CLI inference time including the CPU head. The separate HTTP
measurements below include encoding and transport and were made in a different run. They
should not be subtracted from these values to estimate overhead. GEMMs remain the main cost.
Read-only GEMM resource metadata, untracked/shared or private weight storage, and smaller
attention threadgroups were also measured; none established a useful repeatable gain, so
they are not enabled.

The [performance follow-up](attention.md#shared-probabilities-for-long-single-requests) adds shared attention probabilities for single requests of at
least 4,096 tokens. At 16,347 tokens, complete-request medians improve from 8.39 to 7.49 s
for Flash and 28.76 to 27.25 s for 27B, with identical logits. Reuse also applies to packed
batches of up to eight records with at least 4,096 total tokens. Two 8,072-token records
improve by 3.9% on Flash and 4.2% on 27B in the paired full-engine check; eight 1,382-token
records improve by only 0.9% and 0.3%. Shorter or larger batches retain the previous
dispatch. The same report compares MLX, MPSGraph, fused FFNs and DeltaNet changes; those
substitutions did not establish a further useful gain with the precision contract preserved.
These measurements do not establish the hardware ceiling, and the full-input hosted-API
latency target remains unmet.

A [further context-scaling pass](attention.md#context-scaling-and-64-key-prefetch) retains
64-key score prefetch for single requests of at least 1,024 tokens. Each original 32-key
softmax and value-product update stays separate, preserving exact tested logits. Paired
prototype runs reduce the 16,347-token inference-plus-head median by another 1.7–1.8% on
both models. Shorter-input gains are small and noisier; packed requests keep their prior
dispatch. No context is discarded.

Before the attention-reuse follow-up, the combined build passed 46/46 FP32-oracle decisions
per model, with maximum probability errors unchanged at 0.0002 (flash) and 0.0010 (27B).
Both full 22-record batch-invariance and NaN-poison suites passed, as did host/error tests,
the 160 GEMM shape/mode checks and 72 CPU layout checks. Flash overflow/fallback tests
passed on 20 records; a three-record 27B forced-fallback check matched BF16 byte-for-byte in
single and packed runs. Flash HTTP responses matched the CLI on all 23 cases (including
over-length rejection), and all 32 concurrent clients received their own expected response.
The follow-up report records validation of the new attention path.

## Checkout example over localhost HTTP, 2026-10-04

The exact three-question checkout example from [Cloudflare's
announcement](https://blog.cloudflare.com/clef-decision-models/) contains 63 state bytes and
346 encoded input tokens. With all the optimizations above, it measures 112.5 ms on flash
and 389.4 ms on the 27B, medians of 12 back-to-back localhost HTTP calls after three
warm-ups. These include request encoding, inference, the CPU head and the HTTP round trip.
Cloudflare's published 38.8/209.3 ms medians cover its evaluation suite; the post does not
specify hardware or per-request token counts, so those figures do not establish equivalent
local latency.

For input scaling, the same questions were kept and synthetic ASCII status text was added
before the outage sentence. These are our fixtures, not the unavailable payloads behind the
external API chart. Each size has three measured calls in varied order, after the short
example:

| State bytes | Full input tokens | Local flash | Local 27B |
|---|---:|---:|---:|
| 128 B | 356 | 123 ms | 445 ms |
| 512 B | 404 | 135 ms | 489 ms |
| 2 KiB | 600 | 191 ms | 702 ms |
| 8 KiB | 1,382 | 425 ms | 1,614 ms |
| 16 KiB | 2,424 | 756 ms | 2,809 ms |
| 32 KiB | 4,510 | 1,481 ms | 5,911 ms |

Truncation was disabled. Every response's input-token count matches the full strict
encoding, and repeated calls return identical answers. Both models flag the outage at the
end as urgent and route it to the technical team at every size. At 16/32 KiB they also
select Critical as the highest-probability severity, with the outage at either the beginning
or end. The repeating filler is a latency fixture, not a broad long-context accuracy
benchmark. State bytes are not token counts, and a hosted endpoint that truncates text does
less work than the full-input path measured here.

Reproduce with `bench/checkout_latency.py`. It starts and stops its own localhost servers in
sequence, verifies full token counts and deterministic responses, and saves timing samples,
responses and the server binary's SHA-256. Absolute times vary across runs; use interleaved
comparisons when attributing a difference to an optimization.

```sh
.venv/bin/python -B bench/checkout_latency.py golden/checkout-latency.json
```

The same payloads were later sent to Cloudflare's hosted models; see the
[hosted comparison](hosted-comparison.md).

## Grouped 27B expansion tile order, 2026-10-05

The earlier persistent-scheduling experiment changed how many output tiles each threadgroup
computed. This screen instead keeps one tile per threadgroup and changes only the order in
which independent output tiles are assigned. The operands, tensor layouts, tile sizes and
per-element K reductions stay unchanged.

The screen covers 18 shapes, six orders and both multiply and residual-accumulate modes, for
216 shape/mode/order cases. Every case passes full output-bit equality, packed-row
alignment, poisoned guards, repeated accumulation and sampled float64 checks. Group 1
preserves the existing row-major order. Group 0 visits every tile row before the next tile
column; groups 2, 4, 8 and 16 interleave that many tile rows. Each result has its own paired
baseline, two paired warm-ups and twelve measured calls per arm in six ABBA quartets.

Most orders regress. Full column-major traversal is slower on every measured large shape,
and larger groups severely regress long down-projections. Group 4 improves the 27B's
16,347-token expansion by 2.39% in multiply mode and 2.05% in accumulate mode, with six of
six quartets favoring it in both. Independent repeats check that candidate and two nearby
shapes:

| Expansion shape T x K x N | Mode | Baseline ms | Group 4 ms | Reduction | Quartets faster |
|---|---|---:|---:|---:|---:|
| 16,347 x 5,120 x 34,816, repeat 1 | multiply | 94.075 | 90.693 | 3.60% | 6/6 |
| 16,347 x 5,120 x 34,816, repeat 1 | accumulate | 96.140 | 91.680 | 4.64% | 6/6 |
| 8,072 x 5,120 x 34,816 | multiply | 46.249 | 44.966 | 2.77% | 6/6 |
| 16,347 x 4,096 x 24,576 | multiply | 59.566 | 56.616 | 4.95% | 6/6 |
| 16,347 x 5,120 x 34,816, repeat 2 | multiply | 97.083 | 95.332 | 1.80% | 6/6 |
| 16,347 x 5,120 x 34,816, repeat 2 | accumulate | 107.316 | 104.084 | 3.01% | 6/6 |

The unchanged-order control on the 27B's 16K expansion measures 0.28% and 0.39% reductions
for those two modes in its own paired run. All ten repeated mode cases pass the same
numerical and buffer checks. Absolute times from separate processes are not paired
comparisons; only each row's baseline and candidate are paired.

An isolated engine first tests group 4 only on FP16 expansion projections with at least
4,096 rows, K at least 4,096 and N at least four times K. Down-projections and BF16 fallback
keep their existing dispatch. Its local tile-index formula avoids flattening the entire grid
into a signed int and matches the screened mapping across 1,813 CPU grid cases, including
partial final groups.

The full-request screen uses four warm-up calls and eight measured calls per case, four
measured calls per arm in two ABBA quartets. These are same-process engine call times
including the CPU head, not HTTP latency. All 72 calls preserve raw logits exactly.

| Model | Tokens | Baseline ms | Group 4 ms | Reduction | Quartet reductions |
|---|---:|---:|---:|---:|---|
| Flash | 4,510 | 1,463.142 | 1,453.793 | +0.64% | -2.47%, +1.87% |
| Flash | 8,072 | 3,247.459 | 3,196.316 | +1.57% | +2.45%, -0.10% |
| Flash | 16,347 | 7,616.224 | 7,657.392 | -0.54% | -0.26%, -1.05% |
| 27B | 4,510 | 5,429.539 | 5,439.534 | -0.18% | -0.42%, +0.16% |
| 27B | 8,072 | 10,524.282 | 10,398.712 | +1.19% | +1.42%, +1.03% |
| 27B | 16,347 | 25,131.669 | 24,340.049 | +3.15% | +3.55%, +1.79% |

Flash does not establish a request-speed benefit. A second isolated build therefore narrows
dispatch to K at least 5,120, excluding Flash, and uses overflow-safe positive dimension
ceiling divisions. It repeats the two long 27B cases in a separate run with the same timing
protocol. All 24 calls preserve every raw logit exactly:

| Tokens | Baseline ms | Group 4 ms | Reduction | Quartet reductions |
|---:|---:|---:|---:|---|
| 8,072 | 11,166.194 | 11,184.756 | -0.17% | -2.90%, +1.01% |
| 16,347 | 24,743.401 | 24,487.480 | +1.03% | +1.37%, +1.26% |

The 16K benefit repeats, with different magnitudes. The 8K benefit does not repeat; neither
4.5K nor 8K has an established gain. The production-form isolated build removes the A/B hook
and adds permanent GEMM cases for partial four-row groups and the 4,096-token expansion
boundary. All eleven qualification checks pass: host/error tests, 190 GEMM shape/mode cases,
both models' exact 175 public logits at batches 1, 8 with NaN poison, and 22, mixed
overflowing/clean records matching the qualified BF16 fallback, and both FP32-oracle suites
with first-record layer dumps.

Those exact source files are now retained and rebuilt in the root checkout. The host,
shader, generated include and GEMM benchmark match the qualified candidate hashes. The
timing above comes from the paired isolated engines, not a new timing run of these root
binaries.

`production-manifest.json` and `production-qualification.json` record the root build and
final checks. No tensor-attention numerical change is included in this retention.

Sources, hashes, raw samples and analyses are in `golden/gemm-swizzle-20261004/`.
`analysis.json` covers the initial screen; `repeat-analysis.json` covers the independent
repeats. `engine-manifest.json` pins the isolated engine and its qualified root source
baseline. `engine-analysis.json` and `confirm-engine-analysis.json` contain the full request
samples; `narrow-manifest.json` pins the repeat build and `selected-manifest.json` pins the
production-form candidate.

## Flash 64-row tiles from 768 tokens, 2026-10-05

Template timing exposed a Flash regression at 1,024–1,025 tokens: the GEMM dispatcher
switched to 64-row tiles at 1,024 processed rows, and a template hit removing 32 rows
crossed back below it. The isolated fresh-input experiment selected a narrower rule, now in
main: Flash uses 64-row tiles at 768–1,023 tokens when 64-row padding equals 32-row padding
(`T % 64 == 0` or `T % 64 > 32`). Paired reductions were 2.5–6.9% at the affected lengths
with exact logits; 27B showed no consistent benefit and keeps its dispatch.
`tests/test_gemm_dispatch.py` covers both sides of the boundaries. The measurements are in
the [prefix cache report](prefix-cache.md#gemm-dispatch-near-1k-tokens).

## Compensated attention default, 2026-10-05

Compensated tensor-unit attention became the default after a ten-variant precision screen, a
6,256-prediction labeled evaluation with unchanged decisions, and paired fresh-input timing
on the adopted build: 8.60% lower latency on Flash and 7.40% on 27B at 16,347 tokens, with
all four quartets improving at 2,048 tokens and above on both models. Mean probability error
against the public FP32 corpus rose from 0.00003576 to 0.00004576 (Flash) and from
0.00007982 to 0.00009400 (27B) while every decision and absolute bound held.
`CLEF_ATTN_TU=0` keeps the FP32 path. Full evidence: [attention](attention.md).

## Prefix cache, checkpoints and template entry, 2026-10-05

Opt-in exact prefix reuse, periodic recurrent-state checkpoints and the fixed-template
entry were integrated with exact-logit qualification on both models. Matching-prefix hits
at 16,347 tokens take 210 ms on Flash and 650 ms on 27B against 7.4 s and 23.8 s full
passes; a changed tail after a shared 15K preamble takes 0.28–0.35 s and 0.94–1.21 s.
Uncached requests did not demonstrate a speedup, and a small cost at 2,235 tokens on 27B is
not ruled out. Full evidence: [prefix cache](prefix-cache.md).

## Profiles and GPU activity

### Component time by context length

The production CLI before CPU head parallelization was profiled on both models at 1,382,
2,235, 4,510, 8,072 and 16,347 tokens. `CLEF_PROFILE=1` serializes dispatch categories.
These measurements identify component costs; they are not normal API latency or a hardware
ceiling. The 1,382-, 2,235-, 8,072- and 16,347-token cases match frozen qualified logits.
The 4,510-token case checks equality between repeated normal and profiled runs.

| Model | Full tokens | GEMM time | Attention time | Other GPU time | Attention share |
|---|---:|---:|---:|---:|---:|
| Flash | 1,382 | 319.1 ms | 19.6 ms | 52.5 ms | 5.0% |
| Flash | 2,235 | 532.4 ms | 50.3 ms | 88.8 ms | 7.5% |
| Flash | 4,510 | 1,099.8 ms | 163.1 ms | 184.3 ms | 11.3% |
| Flash | 8,072 | 2,636.2 ms | 677.2 ms | 404.2 ms | 18.2% |
| Flash | 16,347 | 5,575.4 ms | 2,974.4 ms | 854.1 ms | 31.6% |
| 27B | 1,382 | 1,142.4 ms | 59.7 ms | 147.6 ms | 4.4% |
| 27B | 2,235 | 1,879.6 ms | 151.9 ms | 247.5 ms | 6.7% |
| 27B | 4,510 | 4,021.0 ms | 491.2 ms | 480.3 ms | 9.8% |
| 27B | 8,072 | 8,469.3 ms | 1,859.9 ms | 991.5 ms | 16.4% |
| 27B | 16,347 | 17,448.9 ms | 7,752.8 ms | 2,015.5 ms | 28.5% |

Projections dominate at every measured length. Attention's share increases substantially as
the context grows. Faster attention can therefore help long inputs more, but attention alone
cannot account for a large improvement around 1–2K tokens.

Separate unprofiled calls include one warm-up and two measurements per length. The long
Flash calls drifted from 8.01 s on the first 16K call to 9.21 and 9.90 s. The two measured
27B 16K calls took 29.49 and 28.34 s. These are observations of variation, not evidence that
a code change caused it. Candidate comparisons use paired ordering instead of comparing
these times with earlier runs in different conditions.

The exact CLI hash and raw data are in `golden/head-parallel-20261004/medium-profile.json`,
`long-profile.json` and their associated logs.

### GPU activity and the remaining work

A bounded, read-only `powermetrics` capture during repeated 27B inference showed 95–100% GPU
active residency after the first sample and nominal reported thermal pressure throughout.
The requested GPU performance state was P13. Hardware frequency reached 1,620 MHz, then
varied down to 1,359 MHz later in the capture. These measurements establish sustained GPU
activity; they do not establish neural-accelerator utilization or a hardware throughput
ceiling.

A separate serialized profile isolates the final norm and GPU head work. At 4,510 tokens,
that work takes about 10.5 ms on Flash and 13.0 ms on 27B. The backbone GEMMs account for
991 ms and 4,002 ms respectively in those profiled calls. Profiling changes command-buffer
execution, so these are component measurements, not normal request latencies. They make the
large backbone matrix operations the main remaining optimization target.

The evidence rules out these particular substitutions as a substantial solution. It does not
prove that a better full-precision algorithm or a different execution strategy cannot be
faster. The external API chart still lacks the original request/response artifacts, and its
long points report truncation; it is not an equal-work latency target for our full-input
measurements.

### Medium-input profile

The hosted comparison identifies a remaining 27B gap at 1,382 full input tokens. Three
`CLEF_PROFILE=1` passes over that public fixture attribute about 84% of GPU time to GEMMs.
The last pass reports 1,086.7 ms GEMM, 63.4 ms attention, 72.6 ms recurrence and 64.8 ms
other GPU work. This mode serializes dispatches, so those values locate work rather than
measure normal request latency.

An existing-tile sweep at 1,382 tokens covers the expansion, down projection, DeltaNet
input, output projection and attention input. The current 64x128 tile is competitive with
32x256 on all five shapes. The down projection shows a small isolated advantage for 32x256,
4.204 versus 4.334 ms in accumulate mode; the other results vary with shape and mode. This
does not establish a material full-request gain, so the medium-input dispatch is unchanged.
All ten shape/mode cases pass exact tile/packing parity, sampled float64 and NaN guards.

### Measured neural-accelerator utilization

After the Apple analysis components were installed, Instruments exposed the Metal System
Trace "Performance Limiters" counter set. The saved `PerformanceLimiters.tracetemplate`
collects neural-accelerator, cache and occupancy counters. `xctrace` successfully launched
only the isolated tensor-layout benchmark at 4,510 x 5,120 x 34,816; it exited with status
0. The trace contains all 73 expected command buffers. Counter definitions identify Neural
Accelerator Utilization as executed GEMM work relative to peak accelerator performance, and
Limiter as attempted work relative to that peak.

`dispatch-counters.py` matches counter samples to the target process's actual compute
intervals and the benchmark's alternating call schedule. It excludes host gaps, initial
warm-ups, paired warm-ups and correctness-only dispatches. The following values are medians
across the measured dispatches' sample medians:

| Method | Measured dispatches | NAX utilization | NAX limiter | Last-level-cache utilization |
|---|---:|---:|---:|---:|
| Production inline kernel | 32 | 98.29% | 98.37% | 100% |
| Buffer-backed tensor handle | 8 | 98.11% | 98.19% | 100% |
| Device-allocated tensor | 8 | 97.47% | 97.58% | 100% |

The production dispatch medians range from 97.27% to 98.76% NAX utilization. Median kernel
occupancy is only 16.67%, but the accelerator remains busy; low occupancy alone therefore
does not demonstrate unused GEMM throughput. These counters support treating this expansion
as close to accelerator saturation under the measured conditions. They are not a whole-model
hardware ceiling, do not cover every projection shape or context length, and do not
establish that full-input Cloudflare latency is attainable. Profiling affects execution, and
GPU counters are device-wide even though the sample selection uses this benchmark's
execution intervals.

The trace, exported counter definitions, samples and per-dispatch analysis are retained in
the same experiment directory. `parse-counters.py` reads 8,549,696 counter rows with
streaming XML parsing and saves selected time series; `dispatch-counter-summary.json`
contains the 73 schedule assignments and per-dispatch statistics. Reproduce the capture from
the repository root with:

```sh
xcrun xctrace record \
  --template golden/perf-medium-20261004/PerformanceLimiters.tracetemplate \
  --output golden/perf-medium-20261004/expansion.trace --time-limit 20s \
  --launch -- "$PWD/golden/perf-medium-20261004/tensor-layout" 4510 5120 34816
```

Use a new output path to preserve the existing trace. No engine change or additional hosted
inference was made during this investigation.

### Medium projections and the long down-projection counter check

`golden/perf-counter-shapes-20261004/` profiles three production GEMM functions on synthetic
FP16 activations and BF16 weights. Each shape has twelve warm-ups, eight measured dispatches
and a correctness dispatch. The trace contains all 63 labeled command buffers and exits 0.
The first probe uses the non-accumulating functions for all three shapes; the down
projection's actual residual accumulation is measured separately below.

| T x K x N | Tile | Unprofiled GPU median | NAX median | NAX mean |
|---|---|---:|---:|---:|
| 1,382 x 5,120 x 34,816 | 64x128 | 7.709 ms | 99.80% | 97.70% |
| 1,382 x 17,408 x 5,120 | 64x128 | 4.001 ms | 99.76% | 94.39% |
| 4,510 x 17,408 x 5,120 | 32x256 | 13.060 ms | 97.80% | 88.83% |

NAX medians are medians of dispatch sample medians; means are means of dispatch sample
means. Both use only samples inside the target's compute intervals. They show why a
near-100% median alone should not be interpreted as uniform saturation: the longer down
projection has dips that lower its average. The analyzer also reports samples with no
observed overlapping GPU interval from another process. Those exclusions preserve the
medium-shape conclusion, with 97.89% and 94.50% means, but do not isolate the cause of the
long-shape variation.

Counter families end at different times in this short capture. All eight measured dispatches
have complete NAX coverage. The last-level-cache and external-memory counters fully cover
only five of the eight long down-projection dispatches. `analyze.py` reports coverage and
excludes partial dispatches from those aggregates; it does not fill in missing data.
Streaming analysis reads 2,280,922 counter rows. Every shape passes finite-output,
output-guard, exact poisoned repeatability and 64 sampled float64 checks, with maximum
scaled error below 6.7e-8.

The follow-up in `golden/perf-counter-down-20261004/` compares the actual FP32-residual
accumulation kernels at 4,510 x 17,408 x 5,120. It holds the operand buffers fixed and uses
ABBA order: eight total warm-up dispatches followed by twenty-four measurements, twelve per
tile. Timing reuses the GPU-written residual instead of resetting it from the CPU between
dispatches. Separate correctness calls start from identical residuals and check full output
bit equality, finite values, output guards and 64 float64 samples. All pass; maximum scaled
error is 3.66e-8. The final accumulated outputs are finite and the guards remain untouched.

| Accumulating tile | Unprofiled GPU median | NAX median | NAX mean |
|---|---:|---:|---:|
| Current 32x256 | 12.814 ms | 97.59% | 90.02% |
| Previous 64x128 | 13.263 ms | 94.40% | 88.17% |

The current tile is faster in all six unprofiled ABBA quartets. These isolated measurements
support retaining the existing dispatch; they are not a new request-level speedup. The trace
contains all 34 expected command buffers, exits 0, and provides complete counter coverage
for all measured dispatches after a 200 ms process tail allows counter collection to finish.
Analysis reads 1,986,044 counter rows. The raw traces, exported metadata, per-dispatch
counter statistics, separate unprofiled samples and analyzer sources are retained in both
directories.

The data supports limited idle-accelerator headroom for these medium GEMMs and favors the
current long down-projection tile. It does not establish a whole-engine optimum or explain
every slow dispatch. Profiling and other GPU activity affect execution; neither these
timings nor utilization percentages establish attainable full-request API latency. No
production source, generated shader or binary changed.

### Long Flash requests

A 13,876-token Flash request spends about 70% of its serialized category profile on matrix
projections, 18% on attention and 6% on the recurrent scan, and a full-request trace finds
less than 0.8% scheduling gaps inside the GPU span. See
[long-request timing](long-request-timing.md).

## Batched classification workload

A batch classification job on the 27B (seven-question rubrics of about 1.3K tokens per
record, `--batch 8`) motivated much of this work. Its shape differs from the long-context
corpus requests: eight independent 1.3K records do not have the attention cost of one 10K
record, and the repeated question schema follows the state in the prompt
(`prefix | state | schema | suffix`), so its hidden states depend on the state and cannot be
reused across records.

On the `db38cfc` baseline the 27B held 775–805 tokens/s on every such workload (821 at
235-token records), which is about 84% of the measured BF16 tensor-op GEMM rate; batch size
barely moved it (1.8K to 10K tokens per batch: 821 to 781 tokens/s), and the 5,310-request
run held 795–807 tokens/s in every tenth of its 81 GPU-minutes, so steady batched load
does not decay. The current engine is about 13% faster than that baseline on a saved
96-request comparison (755 to 850 tokens/s) and processes about 0.65 records/s.

A serialized category profile of eight actual requests on the current engine:

| GPU category | Time | Share of measured categories |
|---|---:|---:|
| GEMM | 10,501 ms | 87.5% |
| DeltaNet scan | 681 ms | 5.7% |
| Attention | 249 ms | 2.1% |
| Other categories combined | 571 ms | 4.8% |

Profiling serializes GPU work, so these category measurements are not ordinary request
latency. In particular, its 12.46-second stage named `encode` includes profiling waits; it
is not JSON/tokenizer time. CPU head execution accounts for about 0.9% of total batch
inference time across the separate unprofiled pilot. Eliminating attention or overlapping
the CPU head cannot produce a large gain on this shape. This profile does not prove a global
hardware ceiling or exhaust all possible GEMM improvements.

### Batch-size and matrix-dispatch follow-up

A resident-engine test processes the same 16 actual population requests (20,632 tokens, 528
logits) at batch sizes 1, 4, 8 and 16. A full batch-16 warm-up allocates maximum buffers
before timing. Four measured passes per size use a balanced Williams order. All 8,448
compared logits match the warm-up bit for bit; no cache, input shortening or rubric change
is involved.

| Batch size | Median time for 16 articles | Articles/s | Time reduction vs batch 8 |
|---|---:|---:|---:|
| 1 | 22.802 s | 0.702 | 2.59% |
| 4 | 23.271 s | 0.688 | 0.59% |
| 8 | 23.409 s | 0.683 | Baseline |
| 16 | 23.699 s | 0.675 | -1.24% |

Batch one wins all four paired rounds, by 2.44–3.31%, but fails the frozen 5% threshold for
changing the recommendation. A larger batch does not produce a throughput breakthrough.
These are engine-call times on a different subset from the application pilot; their absolute
rates are not a paired comparison with it.

An isolated diagnostic build also records each matrix's dimensions and selected kernel while
processing eight actual requests. It changes logging only and matches all eight saved
current-build response lines at both batch sizes tested. At batch eight, the 256 backbone
projections all use FP16 activation kernels; there is no BF16 overflow rerun. Their
serialized GPU intervals break down as:

| Projection | Share of backbone matrix time | Measured TFLOP/s |
|---|---:|---:|
| Feed-forward gate/up | 46.24% | 53.21 |
| Feed-forward down | 23.90% | 51.47 |
| DeltaNet input | 16.74% | 52.18 |
| Attention and DeltaNet output | 8.32% | 52.16 |
| Attention Q/K/V | 4.80% | 52.77 |

The trace rules out an accidental slow-precision path on these inputs and identifies the
feed-forward projections as the largest target. It does not prove the chip's ceiling.
Batch-one profiling shows higher matrix rates, but the runs are serialized, unpaired
diagnostics; use the balanced unprofiled test above for the measured batch-size benefit.

Application-level alternatives for this workload (a shorter rubric, a warning-only pass, a
cached warning-then-confirm cascade, and placing the schema before the state) were all
tested and rejected on quality; see
[rejected experiments](rejected-experiments.md#workload-level-changes).
