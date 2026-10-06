# Attention

How the attention kernels went from the baseline FA kernel to the current default, with
the measurements behind each retained change. The engine fixes its attention mode when the
model is opened: compensated tensor-unit attention (`attention_tu`) by default, or the
tiled FP32 path with `CLEF_ATTN_TU=0`. `CLEF_ATTN_REF=1` selects the simple reference
kernel for debugging. Every path processes the full accepted input.

All timings are from the one tested M5 Max (40 GPU cores, 128 GB), one GPU job at a time.
Complete-request times include the CPU head and exclude request encoding, model load and
HTTP unless stated. "ABBA quartets" means the two arms alternate A, B, B, A in one resident
process so clock drift affects both equally.

## Kernel selection

| Mode | Dispatch | Kernel |
|---|---|---|
| compensated (default) | all requests | `attention_tu`: high and residual FP16 planes, three FP32-accumulating products per QK/PV product, 32-query/128-key tiles aligned to each record |
| FP32 control, `CLEF_ATTN_TU=0` | single request of at least 1,024 tokens | `attention_prefetch_64`: four SIMDgroups share 32 query rows, 64-key score prefetch, separate 32-key softmax and value updates |
| FP32 control | packed batch of up to eight records with at least 4,096 total tokens | `attention_reuse_4`: four SIMDgroups share 32 query rows, 32-key blocks |
| FP32 control | other requests | `attention_fa`: the original kernel with unrolled accumulator loops |
| `CLEF_ATTN_REF=1` | all requests | `attention`: reference kernel |

The compensated path omits the residual-times-residual product, so it approximates FP32
attention rather than reproducing it bit for bit. The per-record FP16 overflow rerun (BF16
activations) always uses the FP32 path. `make test-attention` compares every kernel on exact
output bits, packed-record invariance, overflow flags, poisoned output and tail guards and
sampled float64 values, on the original 16-bit value grid and on full FP32 mantissas.
The 2026-10-03 unrolling of the FA accumulator loops (2.2–2.4× kernel speedup with identical
bits) is recorded in the [performance history](performance-history.md#attention-kernel-unrolling-2026-10-03).

## FP32 path: shared probabilities and packed reuse

### Shared probabilities for long single requests

The new `attention_reuse_4` kernel assigns four SIMDgroups to 32 query rows of one head.
Each group computes scores and softmax for eight rows, then computes 64 output columns for
all 32 rows. This reuses probabilities and value fragments while retaining 32 accumulator
fragments per SIMDgroup. Keeping only one key fragment's probabilities live at a time avoids
the slowdown seen when all four probability fragments were held together.

The computation remains FP32, with the same 32-key blocks, softmax reductions and ordered
MMA updates as `attention_fa`. Groups beyond the final query row still participate in
barriers; they load zero Q, so the existing eight-row tail padding remains sufficient.
Epilogue stores use scratch disjoint from the reciprocal diagonals that other groups are
still reading.

A same-process comparison loaded each model once and alternated the current kernel,
two-block reuse and four-block reuse in the order A/B/C/C/B/A. Six warm-up requests were
followed by twelve measured requests, four per variant. Every logit matched across all 18
calls per case. These are complete CLI inference times, including the CPU head, for the
existing full-input corpus records; they do not reproduce the external API chart's missing
payloads.

| Model | Input tokens | Previous median | Four-block reuse | Latency reduction |
|---|---:|---:|---:|---:|
| 27B | 8,072 | 11,710.75 ms | 11,247.40 ms | 4.0% |
| 27B | 16,347 | 28,755.45 ms | 27,246.95 ms | 5.2% |
| Flash | 16,347 | 8,387.45 ms | 7,493.20 ms | 10.7% |

Both measured six-call rounds improve for every table row. The 27B timings are noisy, so the
medians should not be read as sub-millisecond estimates. The 4,510-token trials showed much
smaller differences. The host selects the new kernel only for a single request of at least
4,096 tokens; shorter requests and packed batches retain their previous dispatch. A speedup
for packing many short records has not been established.

`make test-attention` compares the new kernel with the existing kernel on identical inputs.
It covers all 32 query-row offsets, ragged tails, packed records, FP16 and BF16 output, a
sampled float64 oracle, unwritten-output/tail NaNs, and mixed overflowing/clean records with
exact per-record flag and output comparisons. It forces the new kernel on small cases that
the host would normally send to the existing kernel.

The final source's isolated 8,072/16,347-token attention checks measured 1.27–1.37×
throughput with identical output bits. These kernel gains do not apply to the remaining
GEMMs or the rest of the request. Both models pass `tests/test_parity.py --dump` against
their FP32 goldens: 46/46 decisions, with unchanged maximum probability errors of 0.0002
(Flash) and 0.0010 (27B). Both full 22-record batch-invariance and NaN-poison suites pass,
as does `make test-errors`. For each model, a 4,510-token request paired with a 346-token
request also matches the saved baseline's logits in normal and BF16 modes. The last 33
residual rows of every layer are byte-identical in both modes. Forced attention-only and
all-producer overflow, at batch 1 and batch 8, produces the baseline BF16 results exactly.
Commands, timings, source/binary hashes and aggregate validation results are recorded in
`golden/perf-attention-20261004/attention-summary.json`.

### Packed attention reuse

Production now enables `attention_reuse_4` for batches of up to eight records containing at
least 4,096 total tokens, including the previously enabled long single requests. The shader
and arithmetic are unchanged. Larger batches retain the original dispatch: many tiny records
can share a query tile, and that performance regime has not been qualified.

A standalone experiment first compared both kernels on identical inputs in ABBA order with
two warm-ups and twelve measured dispatches per mode. Every output bit and overflow flag
matches across all calls, and all eight sampled float64/oracle and guard checks pass.

| Packed records | Query heads | Existing kernel | Reuse kernel | Kernel speedup |
|---|---:|---:|---:|---:|
| 8 × 300 tokens | 16 | 1.050 ms | 1.008 ms | 1.042× |
| 8 × 300 tokens | 24 | 1.464 ms | 1.411 ms | 1.038× |
| 8 × 1,382 tokens | 16 | 16.977 ms | 14.253 ms | 1.191× |
| 8 × 1,382 tokens | 24 | 25.055 ms | 20.742 ms | 1.208× |
| 4,510 + 346 tokens | 16 | 23.420 ms | 18.578 ms | 1.261× |
| 4,510 + 346 tokens | 24 | 34.028 ms | 27.194 ms | 1.251× |
| 2 × 8,072 tokens | 16 | 143.842 ms | 111.217 ms | 1.293× |
| 2 × 8,072 tokens | 24 | 232.006 ms | 186.692 ms | 1.243× |

These are isolated attention-dispatch timings. A boundary check on 8 × 512 tokens also
favored reuse for both head counts, but the 16-head series had substantial clock drift; its
absolute timings are not used as an engine-speed claim.

The full-engine comparison uses the same loaded model, one warm-up per mode and two ABBA
quartets per fixture, for four measured calls per mode. It includes the CPU head and GPU
inference, excluding model loading, encoding and HTTP. Every repeated and candidate logit
matches the first baseline exactly.

| Model | Packed records | Baseline median | Reuse median | Time reduction |
|---|---|---:|---:|---:|
| Flash | 8 × 1,382 tokens | 3,800.9 ms | 3,767.0 ms | 0.89% |
| Flash | 2 × 8,072 tokens | 6,693.1 ms | 6,431.5 ms | 3.91% |
| 27B | 8 × 1,382 tokens | 12,921.7 ms | 12,878.1 ms | 0.34% |
| 27B | 2 × 8,072 tokens | 22,786.0 ms | 21,839.5 ms | 4.15% |

Both quartets favor reuse in all four cases. The medium-record improvement is small, and two
quartets do not establish a precise effect size across workloads. This change improves
packed inference; it does not change single-request dispatch or establish a new hosted-API
latency result.

Qualification covers all 22 public records for both models in batches of eight, normally and
with NaN poisoning: every raw logit matches the preceding scorer build's single-record
result. On the mixed 4,510 + 346-token batch, forced FP16 overflow with poisoning matches
baseline BF16 exactly for both models. The permanent attention benchmark adds 16 long packed
cases: four record layouts, both head counts and FP16/BF16 output, with every candidate
output and its tail guard poisoned before execution. All output bits, overflow flags and
sampled float64 checks pass. Existing small packed and overflow cases also pass.

The previously documented scorer-rounding difference against the older frozen outputs
remains unchanged.

The experiment, timing samples and validation reports are under
`golden/perf-packed-attention-20261004/`. `production-validation.json` confirms unchanged
sources apart from the narrowed dispatch condition and exact qualified logits for all 22
production outputs per model. The production build and `git diff --check` pass.

## Context scaling and 64-key prefetch

Profiles by context length are in the [performance history](performance-history.md#component-time-by-context-length).

### Larger key-block experiment

An isolated FP32 attention variant computes 64 or 128 keys per block instead of 32. It does
not shorten input or reduce operand precision. It changes the online softmax and
accumulation grouping. Twelve measurements per arm, in six ABBA quartets after warm-up, show
roughly 9–18% lower attention time across 1,382–16,347 tokens. Every measured quartet favors
its candidate. These are kernel times, not whole-request gains.

Both variants pass the isolated sampled float64 checks and packed-record invariance,
including all query offsets, key-block boundaries, FP16/BF16 output, overflow flags and NaN
output guards. They change output bits relative to the original kernel.

The full-model check shows why decision agreement alone is insufficient. Flash still agrees
with all 46 FP32 reference decisions, but its mean probability error rises from 0.00003576
to 0.00004813 with 128-key blocks and 0.00005265 with 64-key blocks. Maximum probability
error also rises, from 0.00024937 to 0.00029869 and 0.00036496. On 27B, mean probability
error rises from 0.00007982 to 0.00009872 with 128-key blocks and 0.00010909 with 64-key
blocks; maximum error rises from 0.00073920 to 0.00125170 and 0.00113315. Both retain 46/46
decisions. The 128-key variant passes exact standalone/batch-eight and NaN-poison invariance
on both models, but the precision tradeoff prevents treating it as a quality-preserving
improvement.

This variant remains isolated. A second experiment computes larger score tiles while
retaining each original 32-key softmax and value-product update. For 64-key score tiles, all
tested output bits and overflow flags match the original kernel. The packed tests cover
every query offset, key-block boundaries, FP16/BF16 output and mixed overflow.

| Full tokens | Query heads | Original attention | 64-key score prefetch |
|---|---:|---:|---:|
| 1,382 | 16 | 1.965 ms | 1.785 ms |
| 1,382 | 24 | 2.808 ms | 2.537 ms |
| 2,235 | 16 | 4.597 ms | 4.187 ms |
| 2,235 | 24 | 6.736 ms | 6.122 ms |
| 4,510 | 16 | 19.177 ms | 17.251 ms |
| 4,510 | 24 | 29.282 ms | 26.778 ms |
| 8,072 | 16 | 66.525 ms | 60.251 ms |
| 8,072 | 24 | 108.316 ms | 97.924 ms |
| 16,347 | 16 | 293.710 ms | 262.108 ms |
| 16,347 | 24 | 430.631 ms | 396.496 ms |

These are isolated kernel medians from twelve calls per arm. Fifty-nine of sixty paired
quartets improve. The 128-key prefetch variant also preserves bits but is generally less
effective.

The 64-key prefetch engine now passes both models' complete 22-record public corpus with raw
logits identical to the qualified production build at batch sizes 1, 8 and 22, and at batch
size eight with NaN poisoning. A mixed 4,510 + 346-token batch with forced FP16 overflow and
poisoning matches production's BF16 fallback exactly on both models. This preserves all 46
FP32 reference decisions and probability errors per model. The prototype selects prefetch
only for at least 1,024 packed tokens and at most eight records; the larger batch also
verifies the original fallback path.

The prefetch kernel was isolated for the following complete-request timing and retention
checks. Its kernel speedup must not be reported as a request-level speedup. All experiment
sources and raw results are under `golden/attention-keys-20261004/`.

The first complete Flash comparison has four warm-ups and two measured ABBA quartets per
length, with exact logits on all 72 calls. At 16,347 tokens the median improves from
7,485.483 to 7,356.337 ms (1.73%); both paired quartet means favor prefetch, by 1.52% and
1.74%. At 1,382–8,072 tokens the paired comparisons have mixed directions, so this run does
not establish a reliable improvement there. The 346-token control uses the same kernel in
both modes and also varies, reinforcing the need for paired evidence. The 27B comparison
also finishes with exact logits on all 72 calls:

| Full tokens | Production median | Prefetch median | Paired quartet reductions |
|---:|---:|---:|---:|
| 1,382 | 1,457.914 ms | 1,477.015 ms | -1.69%, -2.77% |
| 2,235 | 2,435.524 ms | 2,379.593 ms | 2.20%, 2.15% |
| 4,510 | 5,218.440 ms | 5,186.311 ms | 0.46%, 0.74% |
| 8,072 | 10,444.888 ms | 10,327.587 ms | 1.43%, 1.51% |
| 16,347 | 24,553.386 ms | 24,108.506 ms | 1.83%, 2.37% |

The 346-token 27B control uses identical kernels in both modes yet differs by 3.77% and
4.32% in its paired quartets, illustrating the variation in short requests. The 1,382-token
regression required a separate check before enabling the prototype's broad dispatch.
Production below 4,096 tokens uses `attention_fa`; the original isolated comparisons used
`attention_reuse_4`.

A direct comparison against `attention_fa` now covers 346, 512, 1,024, 1,382, 2,048, 2,235,
3,072 and 4,095 tokens with both head counts. All 96 paired quartets favor the prefetch
kernel, with median kernel reductions of about 18–32%. Output bits, packed invariance,
overflow flags and sampled float64 checks pass. This rules out the kernel itself being
slower in that isolated test. It does not erase the request-level result.

A separate whole-request diagnostic reverses the order of three-call blocks. Eight blocks
follow four warm-ups, giving twelve measured calls per arm at 346 and 1,382 tokens. All 112
calls across both models preserve exact logits. The 346-token control uses identical
dispatch in both arms.

| Model | Full tokens | Original median | Prefetch median | Paired block reductions |
|---|---:|---:|---:|---|
| Flash | 346 | 110.476 ms | 110.492 ms | -0.02%, 0.51%, 0.06%, 0.05% |
| Flash | 1,382 | 431.904 ms | 423.522 ms | 3.07%, 0.17%, 5.96%, 0.64% |
| 27B | 346 | 400.359 ms | 399.125 ms | 3.60%, -1.89%, 1.80%, -2.08% |
| 27B | 1,382 | 1,612.017 ms | 1,602.099 ms | 0.37%, 0.29%, 1.12%, 0.66% |

These figures include the first call of each block. Reporting those calls separately does
not reveal a repeatable switch penalty. At 1,382 tokens, median GPU execution time falls
from 422.915 to 415.020 ms on Flash and from 1,602.050 to 1,592.715 ms on 27B. The repeat
supports a modest benefit, with substantial variation between runs. A cache, clock or
thermal explanation for the earlier regression has not been established. Raw stage timings
and both inclusive and remaining-call summaries are in `*-warm-pairs.log`,
`*-warm-pairs.jsonl` and `warm-analysis.json`.

The eight-group, 64-query-row variant also preserves output bits, but improves isolated
attention by only about -1.8% to 4.2% against the old reuse kernel. The four-group variant
is consistently faster, so the eight-group variant is not selected. See `prefetch-q8.log`
and `analysis.json`.

### Packed-request timing and dispatch decision

Three packed fixtures use four warm-ups and two measured ABBA quartets per model. All 72
calls preserve exact logits. Timing includes inference and the CPU head, excluding model
load, encoding and HTTP.

| Model | Full tokens per record | Original median | Prefetch median | Paired quartet reductions |
|---|---|---:|---:|---|
| Flash | 8 × 1,382 | 3,665.375 ms | 3,683.420 ms | -5.68%, 5.69% |
| Flash | 4,510 + 346 | 1,686.639 ms | 1,653.677 ms | 1.17%, 1.64% |
| Flash | 2 × 8,072 | 6,694.746 ms | 6,731.626 ms | 1.13%, -0.55% |
| 27B | 8 × 1,382 | 12,929.035 ms | 12,986.002 ms | -0.49%, 1.18% |
| 27B | 4,510 + 346 | 5,900.277 ms | 5,896.488 ms | 0.30%, -0.10% |
| 27B | 2 × 8,072 | 20,904.353 ms | 20,684.321 ms | 1.00%, 1.00% |

Two fixtures improve in both paired quartets, but the remaining results do not support broad
packed enablement. The retention decision therefore limits prefetch to single requests with
at least 1,024 tokens. Packed requests keep the previously qualified dispatch:
`attention_reuse_4` from 4,096 total tokens for up to eight records, otherwise
`attention_fa`. This decision does not claim a packed-throughput improvement from prefetch.

`*-packed-pairs.jsonl`, `*-packed-pairs.log` and `packed-analysis.json` contain the raw
samples and summaries.

### Retained implementation and final qualification

`metal/clef.metal` now specializes one attention-reuse template for 32- and 64-key score
blocks. The 64-key specialization is the qualified four-group prefetch algorithm, with
identical code tokens to the experimental template apart from the function name, comments
and whitespace. The 32-key specialization supplies the existing packed path. `clef_metal.m`
enables prefetch only for a single request of at least 1,024 tokens and keeps both prior
packed fallbacks. It zeroes 64 K/V tail rows within the existing 64-row allocation. No
weight, operand precision, softmax grouping or input limit changes.

`make test` and `make test-attention` pass. The latter now includes `--prefetch 64`: 24
small packed checks, 32 long packed checks and 80 single-shape/head checks across the three
kernel selections. These cover exact baseline bits, record invariance, FP16/BF16 output,
overflow flags, poisoned output/tail guards and sampled float64 values. The new single-shape
cases bracket the 64-key and 1,024-token boundaries.

The generalized kernel build passes both complete 22-record corpora at batch sizes 1, 8 and
22, and batch eight with poisoning, plus mixed 4,510 + 346-token forced-overflow checks.
After restricting dispatch to single requests, the final CLI repeats both corpora at batch
sizes 1, 8 with poisoning, and 22. Every raw logit matches the preceding qualified
production build. Both final mixed overflow/poison runs also match its BF16 fallback
exactly. Thus all 46 FP32-reference decisions and the existing probability errors are
unchanged on each model.

The Flash server passes the 23-case HTTP suite and 32-client concurrency check with poisoned
activations and idle touches enabled.

The retained kernel's whole-request evidence is the paired timing above, measured on the
instrumented prototype. It gives about 1.7–1.8% lower median inference-plus-head time at
16,347 tokens. Shorter-input improvements are small and noisier. No packed-throughput
benefit is claimed for this change. Projections remain the main cost, especially at 1–2K.

The generated Metal include was rebuilt with Make.

Final SHA-256:

## Direct fragment rescaling

Direct fragment rescaling is retained for the reuse and prefetch kernels. The original FA
kernel remains unchanged.

The paired prototype reduces 16,347-token request medians by 1.61% on Flash and 0.57% on
27B. Medium-length request results are mixed; a focused repeat does not reproduce the
initial 4,510-token slowdown and does not establish a gain at that length. This is an
incremental long-context improvement, not Cloudflare-equivalent full-input latency.

### Changes examined

The current attention kernel rescales each accumulator row by multiplying it by an 8x8
diagonal matrix. Its off-diagonal entries are zero. The candidates are:

- `epilogue`: normalize each value after the final fragment store, before gating.
- `identity`: skip the diagonal multiply when all eight row scales equal one. Keep the
  original multiply when any row needs rescaling.
- `init`: initialize the off-diagonal zeros once per query tile instead of every key block.
- `identity-init`: combine identity skipping and one-time initialization.
- `broadcast-v2`: load a matrix containing the row scale in every column, then multiply the
  corresponding local accumulator elements directly. It includes the epilogue change.

The first broadcast prototype did not compile because the compiler exposes matrix storage as
a 64-element vector, which has no matching `fma` overload. The corrected prototype accesses
the two local elements with a `float2` reference. It does not hard-code a lane-to-row
mapping: the loaded broadcast matrix supplies the matching scale for each local accumulator
element.

This uses the installed compiler's `thread_elements()` accessor, also used by the local MLX
Steel implementation. The public MSL specification does not document that accessor. The
prototype assumes a 32-lane Apple SIMDgroup with two local elements per 8x8 fragment. That
compiler dependency remains a consideration before production retention.

### Kernel checks and screening

All five working candidates pass all three attention dispatch families on both synthetic
value distributions. The original grid and the full FP32 mantissa fixture each test
FP16/BF16 output, packed-record invariance, overflow flags, poisoned output and tail guards,
and sampled float64 accuracy. Single-request lengths include block boundaries and extend
through 16,347 tokens. Every compared output bit and overflow flag matches the baseline.

Epilogue-only and initialization-only changes show no consistent speed benefit. Combining
initialization with identity skipping does not establish an additional gain. Identity
skipping helps long reuse/prefetch kernels, but does not show a consistent FA benefit.
Direct fragment rescaling is the most consistent medium/long candidate in this screen. The
original FA path has no established short-request improvement.

The follow-up uses two warm-up pairs and twelve measured calls per arm, ordered as six ABBA
quartets. This table contains GPU kernel medians for the production 64-key prefetch path,
with full FP32 synthetic inputs. These are not complete request latencies.

| Full tokens | Query heads | Baseline ms | Direct rescale ms | Reduction | Quartets faster |
|---|---:|---:|---:|---:|---:|
| 1,024 | 16 | 1.010 | 0.960 | 4.90% | 6/6 |
| 1,024 | 24 | 1.455 | 1.396 | 4.07% | 6/6 |
| 1,382 | 16 | 1.785 | 1.693 | 5.16% | 6/6 |
| 1,382 | 24 | 2.548 | 2.410 | 5.42% | 6/6 |
| 2,235 | 16 | 4.191 | 3.961 | 5.49% | 6/6 |
| 2,235 | 24 | 6.137 | 5.791 | 5.64% | 5/6 |
| 4,510 | 16 | 16.213 | 15.929 | 1.75% | 4/6 |
| 4,510 | 24 | 25.015 | 23.366 | 6.60% | 6/6 |
| 8,072 | 16 | 54.492 | 51.775 | 4.99% | 6/6 |
| 8,072 | 24 | 89.666 | 84.188 | 6.11% | 6/6 |
| 16,347 | 16 | 262.636 | 248.796 | 5.27% | 6/6 |
| 16,347 | 24 | 392.691 | 372.475 | 5.15% | 6/6 |

The 32-key reuse path also improves in all twelve repeated shapes, by 3.44% to 6.35% in
kernel median time. That benchmark uses single records to compare like-for-like kernels;
packed-record correctness is checked separately. Identity skipping reduces 16,347-token
prefetch medians by 3.96% and 6.34% for 16 and 24 query heads, respectively, but its benefit
is less consistent around 1,024 and 1,382 tokens.

### Full-model evaluation

Each candidate runs both complete 22-record corpora at batch size one with strict encoding
and truncation disabled. Identity skipping and direct rescaling both reproduce every saved
qualified production logit exactly. Both preserve all 46 FP32-reference decisions per model
and the existing numerical errors:

| Model | Mean question-maximum probability error | Maximum probability error |
|---|---:|---:|
| Flash | 3.5755436448543436e-5 | 0.0002493696689521707 |
| 27B | 7.982075535551896e-5 | 0.0007391979189106945 |

The mean averages each question's maximum absolute option-probability error. These results
establish exact equality on the public corpus, not accuracy on unseen requests.

### Complete-request comparison

The same process alternates baseline and direct rescaling for each fixture. Four total
warm-up calls precede eight measured calls, arranged as two ABBA quartets with four calls
per arm. All 120 calls across both models match raw logits with `memcmp`, including
warm-ups. This uses complete inputs and the normal unprofiled inference path.

| Model | Full tokens | Baseline median ms | Direct rescale median ms | Reduction |
|---|---:|---:|---:|---:|
| Flash | 1,382 | 400.8 | 396.7 | +1.01% |
| Flash | 2,235 | 719.8 | 708.6 | +1.55% |
| Flash | 4,510 | 1532.9 | 1591.3 | -3.81% |
| Flash | 8,072 | 3160.7 | 3130.2 | +0.96% |
| Flash | 16,347 | 7593.2 | 7470.6 | +1.61% |
| 27B | 1,382 | 1508.5 | 1503.0 | +0.37% |
| 27B | 2,235 | 2455.2 | 2401.4 | +2.19% |
| 27B | 4,510 | 5293.1 | 5293.7 | -0.01% |
| 27B | 8,072 | 10532.1 | 10478.9 | +0.51% |
| 27B | 16,347 | 24436.2 | 24296.9 | +0.57% |

Both quartets favor the candidate on both models at 8,072 and 16,347 tokens. At 16,347,
Flash's quartet reductions are 1.77% and 1.25%; 27B's are 0.60% and 0.57%. Four samples per
arm do not establish a general latency distribution. Smaller-case results include opposing
quartet directions, and Flash's 4,510-token case is slower in both quartets. A focused
repeat of that case is recorded below, with four ABBA quartets after warm-up.

The isolated kernel gain therefore translates into a much smaller request gain. These
measurements do not establish a uniform speedup across lengths or a cause for the timing
variation. The retained change preserves the existing dispatch thresholds and the FA kernel
used for short requests and larger packed batches.

The focused 4,510-token Flash repeat has eight measured calls per arm. Baseline and
candidate medians are 1,642.04 and 1,646.36 ms, a 0.26% increase. Quartet reductions are
+5.40%, +0.85%, -0.28% and -2.61%; every raw logit remains identical across all twenty
calls. This does not reproduce the first run's 3.81% slowdown, and it does not establish a
benefit at this length. Absolute times and quartet directions vary. The separate batch,
poison and forced-fallback qualification subsequently passes against the same pinned build.

### Retained build and qualification

Only the reuse/prefetch body is copied from the qualified candidate; its code is unchanged.
The FA function remains byte-identical to the preceding production source. The host checks
that both changed pipelines use a 32-lane SIMDgroup before allowing inference. Make
regenerates the embedded shader and rebuilds both binaries.

The final build passes:

- `make test test-attention test-errors`, including both synthetic attention distributions.
- Both complete 22-record corpora at batch sizes 1, 8 with poisoning, and 22, with exact
  saved logits and unchanged FP32-reference error metrics.
- Mixed 4,510 + 346-token forced-overflow/poison runs against the preceding BF16 fallback.
- Flash HTTP byte parity for 23 cases, 32 concurrent clients, limits and idle touches, with
  activation poisoning enabled.

The paired timing measurements above belong to the isolated prototype. The retained
reuse/prefetch source is the same, but a new whole-request timing comparison of the final
binaries has not been run. `production-qualification.json` pins the final build and checks.

## Compensated attention precision screen

None of the ten tensor-attention candidates was retained by this screen. Every candidate
keeps all 46 decisions on both public corpora, but each increases mean probability error
against the FP32 oracle on both models. That is a measured regression on this fixed corpus,
not proof of worse task accuracy on unseen requests. Passing the existing absolute error
ceilings also does not prove equivalent quality. The later control and real-activation
results below qualify how this screen should be interpreted.

The screen used the prefetch revision recorded in [the context-scaling
report](#context-scaling-and-64-key-prefetch). Production subsequently retained [direct
attention rescaling](#direct-fragment-rescaling), with exact public logits.

### Method and candidates

The experiment freezes a read-only snapshot of the prototype worktree's compensated FP16
tensor-attention prototype. Q, K and V are split into high and residual FP16 planes. Three
FP32-accumulating tensor products approximate each full QK/PV product, omitting the
residual-times-residual term. All candidates read the complete input. The root CLI and
thread-safe CPU head are copied into isolated experiment directories.

`c` is the original 128-key prototype. `128r` uses the unrounded probability denominator.
`32o` restores 32-key blocks and ascending per-lane softmax summation, but sums rounded
high-plus-low probabilities. `32r` also uses the unrounded denominator.

The second set uses a probability scale of 32768 instead of 1024 and an unrounded
denominator. Suffix `p` retains unscaled residual planes and three products. Suffix `s`
scales residual planes by 2048 before FP16 storage and rescales their contributions. Suffix
`f` adds the fourth, residual-times-residual product. Prefix 32 or 128 selects the key block
size; the 32-key variants retain ascending per-lane summation.

Both sets use an atomic shared overflow flag. Controls with tensor attention disabled match
the qualified root's raw logits exactly on both 22-record corpora. The original `c` control
also matches the prototype worktree's saved prototype outputs exactly. Each candidate runs
the complete corpus with batch size one and strict encoding. Goldens are unchanged.

### Whole-model results

The mean below averages each question's maximum absolute option-probability error, using the
same evaluator and FP32 references for every arm. The relative change is against current
production. Every row has 46/46 matching decisions on each model, and every row passes the
existing maximum logit-error and probability-error ceilings. These are corpus results, not
proof of unchanged accuracy on unseen inputs.

| Variant | Flash mean error | Change | 27B mean error | Change |
|---|---:|---:|---:|---:|
| baseline | 3.5755436e-05 | +0.00% | 7.9820755e-05 | +0.00% |
| c | 4.5761814e-05 | +27.99% | 9.3999334e-05 | +17.76% |
| 128r | 4.5490989e-05 | +27.23% | 9.3980741e-05 | +17.74% |
| 32o | 4.3463531e-05 | +21.56% | 8.4110153e-05 | +5.37% |
| 32r | 4.1413666e-05 | +15.82% | 9.3309535e-05 | +16.90% |
| 128p | 4.4321844e-05 | +23.96% | 9.3078089e-05 | +16.61% |
| 128s | 4.0938297e-05 | +14.50% | 0.00010636393 | +33.25% |
| 128f | 4.475064e-05 | +25.16% | 0.00012114502 | +51.77% |
| 32p | 4.2773156e-05 | +19.63% | 8.8524373e-05 | +10.90% |
| 32s | 3.8897162e-05 | +8.79% | 0.00010868173 | +36.16% |
| 32f | 4.0238807e-05 | +12.54% | 8.7815994e-05 | +10.02% |

Several variants improve maximum probability error on one model while worsening the mean. No
variant improves the mean on either model. The 32-key variants also cost more in isolated
attention timings; adding the fourth product costs more again. Those timings are sequential
screening measurements, not sustained paired whole-request benchmarks. No speedup is claimed
or retained from these experiments.

### What the isolated checks missed

All ten candidates pass the original sampled float64 checks, split-plane checks, packed
record invariance, overflow checks and poisoned-output/tail checks. Those are necessary
implementation checks, but they do not establish whole-model numerical quality.

The synthetic input generator uses a 16-bit numerical grid. That undersamples the mantissas
and magnitudes encountered in FP32 activations. A separate CPU experiment with 200,000
values per distribution shows why this matters: for uniform values in [-0.01, 0.01], scaling
the residual plane by 2048 increases exact two-plane reconstruction from 1,431 to 147,926
values and reduces mean representation error from 1.49e-8 to 1.12e-10. Yet those scaled
variants still worsen whole-model mean probability error. Better representation in isolation
is not enough to predict the final model result.

The benchmark now accepts `--values fp32`, preserving the original grid by default and
adding full FP32 mantissas with varied exponents. The new production check passes exact
prefetch-versus-original output comparisons, sampled float64, packing, FP16/BF16 output,
overflow flags and poisoned output/tail guards. It covers 8 small packed checks, 16 long
packed checks and 24 single-shape/head checks through 16,347 tokens. `make test-attention`
includes this check.

The original compensated tensor variant, `128s` and `32f` also pass the stronger fixture
with the existing tolerance, along with the FP32 control. Thus widening the synthetic
distribution does not expose the whole-model error by itself. The coarse grid is a coverage
limitation, but its causal contribution to the observed model error remains unproven. The
full-model eval is still required for numerical changes.

These follow-up logs and exact commands are in
`golden/attention-groups-20261004/full-values-results.json`.

### Control and real-activation evidence, 2026-10-05

A CPU-only audit of the prototype worktree's saved public outputs reproduces its control
results. Its tiled FP32 control has exactly the same raw logits as this screen's baseline on
both models. The scalar FP32 reference kernel changes summation order; its mean probability
error changes by -0.21% on Flash and +47.78% on 27B. Thus even a full-FP32 attention
implementation can score worse against the whole-model FP32 golden after the rest of this
engine's rounded operations.

The compensated tensor kernel in that control run reproduces the `c` result above. Removing
one complete request at a time, including all of its questions, gives:

| Model | Mean error change on all 22 requests | Questions worse / better | Range after omitting one request |
|---|---:|---:|---:|
| Flash | +27.99% | 26 / 20 | +9.12% to +39.52% |
| 27B | +17.76% | 22 / 24 | -8.32% to +39.39% |

The Flash increase survives every single-request omission. On 27B the direction depends on
which request is omitted. These are descriptive checks of a fixed corpus, not confidence
intervals or equivalence tests. Questions in one request share a forward pass; counting all
46 as independent observations would overstate the available evidence. The corpus has also
been used repeatedly to choose implementations, so it is not an independent held-out
validation set.

A separate run replayed real Q/K/V activations from four lengths and three layers per model
against sampled float64 attention. Reading those saved logs confirms that the compensated
kernel has lower FP32 relative RMS error in all 12 cases on each model. Its gated FP16
relative RMS error is equal or slightly lower at the logs' printed precision. Those are
sampled kernel results, not an independent rerun here and not a whole-model accuracy
guarantee. They nevertheless prevent treating the corpus mean as a complete ranking of the
attention arithmetic.

Both observations should remain visible: the final probability error increased on the tested
corpus, while the sampled local attention result improved against float64. Neither matching
decisions nor a small-corpus average alone establishes unchanged task quality. The root
checkout keeps its qualified FP32 attention path. A numerical replacement needs broader
independent end-to-end evidence as well as the real-activation checks; a non-significant
difference would not by itself establish equivalence.

The audit script and per-question/request results are under
`golden/eval-sensitivity-20261005/`. `results.json` pins the input artifact hashes and
records the audit's scope. It runs entirely on the CPU and does not regenerate goldens,
invoke an engine, or send hosted requests.

### Fresh validation fixtures

`golden/heldout-20261005/fixture/` freezes 24 new synthetic requests and 96 labels before
model inference. The cases cover incident status, invoices, access policies and deployment
state, with noul, choice and score questions. Measured input lengths run from 743 to 16,300
tokens. Relevant facts occur at the start, middle and end; the native tokenizer accepts
every complete request without truncation. These controlled fixtures are new to the
implementation selection process, but they are not a random production sample.

The evaluation freezes the qualified root CLI and the parallel compensated-attention CLI. It
first checks that disabling tensor attention in the latter reproduces the root's logits,
then compares the compensated variant with fresh FP32 references. The report separates FP32
decision agreement and probability error from accuracy, log loss and Brier score against the
predeclared synthetic labels. The faster single-half value variant is excluded.

The initial 24-request streamed Flash reference ran out of MPS memory at the first attention
layer after completing layers 0 through 2. It produced no golden logits. The failure and
partial output directory are preserved. The retry uses at most four requests and 20,000
total tokens per process, with each 16K request run alone. A forward hook synchronizes MPS
and releases unused cache after each decoder-layer call; it returns no replacement output
and changes no tensor operation. Each part verifies the pinned model snapshot before import.
Merging requires complete question coverage, finite logits and exact native/reference token
IDs.

Both bounded references completed all ten parts and each merged 24 requests and 96 questions
with exact full token coverage. Both quality comparisons are complete. Both references get
all 96 predeclared labels right. The 27B reference's smallest top-two probability margin is
0.88687, so this fixture's lack of close decisions applies to both models. Existing goldens,
`ref/corpus.py` and the reference computation files remain unchanged.

#### Flash results on the fresh fixtures

The frozen candidate with tensor attention disabled reproduces all baseline output bytes.
Both the baseline and compensated variant match all 96 FP32 decisions and all 96 predeclared
labels, including the four longest requests. Both pass the existing 0.05 maximum logit-error
and 0.002 maximum probability-error limits.

| Metric | Baseline | Compensated |
|---|---:|---:|
| Mean per-question maximum probability error | 2.19764e-5 | 2.68080e-5 |
| Maximum probability error | 0.00082453 | 0.00066834 |
| Maximum raw logit error | 0.029460 | 0.038038 |
| Mean label log loss | 0.02783768 | 0.02783395 |
| Mean label Brier score | 0.00226941 | 0.00226691 |

The mean probability error rises 21.99%, while the worst probability error falls. Fifty
questions move closer to the FP32 probabilities and 46 move farther away. The label-based
scores improve slightly, by 0.0134% in log loss and 0.1100% in Brier score. These small
changes on controlled fixtures do not establish better calibration or general task accuracy.
All labels being correct also limits what this set can tell us about decision-boundary
regressions. Full-input coverage and matching answers are confirmed here; general quality
equivalence is not.

The FP32 top-two probability margin is at least 0.70739 on every fresh Flash question
(median 0.96234). None is below 0.1. `eval-clef-flash/margins.json` records this diagnostic:
the set checks long-context evidence handling, but is insensitive to close decisions.

The fixture, oracle, engine and output hashes are in
`golden/heldout-20261005/eval-clef-flash/results.json`; `analysis.json` breaks results down
by family and input-length target. These are quality runs, not interleaved timing
measurements, so their runtimes are not used as speedup evidence.

#### 27B results on the fresh fixtures

The FP32 control again reproduces the frozen baseline output bytes. Baseline and compensated
attention both match all 96 reference decisions and all 96 labels, and both pass the
existing numerical bounds.

| Metric | Baseline | Compensated |
|---|---:|---:|
| Mean per-question maximum probability error | 9.84351e-6 | 9.62836e-6 |
| Maximum probability error | 0.00032277 | 0.00024736 |
| Maximum raw logit error | 0.00784481 | 0.00703001 |
| Mean label log loss | 0.01095988 | 0.01095921 |
| Mean label Brier score | 0.000310162 | 0.000310114 |

Mean probability error falls 2.19%; 46 questions move closer to the FP32 probabilities and
50 move farther away. Label log loss and Brier score change by less than 0.02%. These
results do not establish a task accuracy gain, particularly with the large decision margins
noted above. They also show why the older corpus's increase in average probability error is
not a universal ranking of the two implementations. Artifacts are in
`golden/heldout-20261005/eval-clef/`, with per-family and per-length breakdowns in
`analysis.json`.

The copied candidate shader was newer than the frozen candidate binary: the prototype
worktree had edits that were not in that build. The binary comparison above remains valid
and its FP32 control is exact. The original frozen copy is preserved. Any integration must
rebuild a consistent source tree and validate that resulting binary; the newer copied shader
is not the source of the reported result.

`golden/attention-integration-20261005/` now contains such a reconstructed build. It
combines the recovered shader and frozen host with the retained grouped 27B expansion GEMMs.
It uses the atomic rescale flag from the earlier order experiments and excludes the
lower-precision value-product instantiation. Its shader is verified verbatim inside the
rebuilt binary. Root production is unchanged.

Qualification compares its public poisoned batches and all fresh-fixture logits with the
saved evaluated outputs, checks mixed BF16 fallback, and runs grid/full-FP32 attention
fixtures against float64. All seven Flash and four 27B checks pass. The Flash run includes
host/error tests and all 190 GEMM shape/mode cases. Both models reproduce the saved raw
logits exactly in the FP32 control and compensated arms; both also reproduce the qualified
BF16 fallback on the mixed 4,510+346-token overflow fixture. A separate same-process ABBA
harness selects FP32 or compensated attention before each complete request while keeping the
grouped GEMMs enabled in both arms. Timing is gated on qualification; no speedup or
production acceptance is established by the reconstruction itself.

The first reconstructed Flash qualification stopped in the snapshot-verifier unit test: the
isolated build copied the verifier but omitted its nested trusted-manifest directory. The
failed log is preserved as `host-gemm-errors-missing-manifests.log`. The copy step now
includes the two checked-in manifests, with hashes recorded in `support-fix.json`. No
inference artifact changed. The complete Flash qualification then passed; the 27B inference
qualification was independent of this fixture issue.

#### Paired request timing and adoption decision

The rebuilt candidate completed 72 calls per model across six full-input lengths. Each
length has four warm-ups followed by two ABBA quartets, giving four measured calls per arm.
The same process, model and buffers serve both arms; grouped 27B GEMMs remain enabled. Each
arm's repeated outputs are bit-identical, and decisions agree across arms. Times include the
CPU head but exclude tokenization and HTTP.

| Tokens | Flash FP32 → compensated, ms | Reduction | 27B FP32 → compensated, ms | Reduction |
|---:|---:|---:|---:|---:|
| 346 | 120.43 → 119.47 | 0.80% | 422.37 → 424.16 | -0.42% |
| 1,382 | 454.50 → 448.34 | 1.36% | 1,577.58 → 1,556.81 | 1.32% |
| 2,235 | 735.85 → 717.23 | 2.53% | 2,559.68 → 2,523.48 | 1.41% |
| 4,510 | 1,604.41 → 1,560.92 | 2.71% | 5,520.24 → 5,370.65 | 2.71% |
| 8,072 | 3,144.22 → 3,037.87 | 3.38% | 10,664.44 → 10,193.36 | 4.42% |
| 16,347 | 7,518.09 → 6,890.62 | 8.35% | 24,501.67 → 22,736.41 | 7.20% |

Both quartets improve at 8K and 16K for both models. The 27B's shortest case is unresolved:
one quartet is slightly slower and the other slightly faster. These are back-to-back engine
timings, not cool-start API latencies. No token is omitted.

A CPU-only comparison with the saved, hash-pinned Cloudflare journal requires no new API
calls. Among the 39 questions with matching reported token counts, both local arms retain
39/39 FP32 decisions; hosted Flash has 39/39 and hosted 27B 38/39. The candidate's mean
probability errors are 0.00004270 (Flash) and 0.00009687 (27B), versus hosted 0.00117843 and
0.00152408. The qualified production errors on that subset are smaller still, 0.00003088 and
0.00005786. Local metrics use float64 softmax of raw logits while hosted probabilities have
API rounding. Matching counts do not prove token identity, and agreement is not labeled
accuracy.

**The default remains FP32 attention.** The candidate offers a repeatable long-input gain
and passes the existing absolute bounds, but mean FP32 probability error rises on both Flash
sets and on the older 27B set. This does not establish worse general task accuracy; it is
insufficient evidence to call the change a quality-neutral default under the requested
constraint.

`adoption-decision.json` records this decision. The prepared source patch is unapplied, and
its application helper checks that decision before modifying production. The qualified
engine, all existing goldens and the default context handling are unchanged.

The final residual check also passes the existing 0.01 relative-L2 bound at every saved
stage of the first public request: 34 stages on Flash and 66 on 27B, each at 300 tokens.
Maximum relative L2 changes from 0.001232 to 0.001391 on Flash and from 0.004054 to 0.004804
on 27B. Final-norm relative L2 changes from 0.0007378 to 0.0007017 on Flash and from
0.002398 to 0.002494 on 27B. These mixed layer results do not change the non-adoption
decision. `residual-qualification.json` pins both engines and reference files. This
experiment has no remaining queued GPU work.

The next independent direction is preserving the existing FP32 arithmetic while reusing an
exact matching prefix. The parallel cache currently requires tensor attention, so it cannot
be adopted unchanged under that decision. A consistent read-only source snapshot and the
required FP32 layout changes are recorded in `golden/prefix-fp32-20261005/feasibility.json`.
An isolated adaptation is now built. Its Flash smoke check matches production byte for byte,
and 48 preparation layouts plus 1,080 cached-attention layouts preserve full-pass bits and
memory guards. The ordinary encoding path passed 132 byte-identical comparisons against
production. See [FP32 prefix validation](prefix-cache.md#exact-fp32-prefix-reuse) for the
candidate and pending checks. Any such speedup would apply to matching or growing contexts;
an unrelated first request still needs its complete forward pass.

## Labeled evaluation on ContractNLI

Compensated attention preserves all 6,256 labeled predictions in the completed development
and test comparisons on both models. Probability-score changes are small and mixed. Its mean
error against some FP32 reference corpora increases, so the task result does not establish
numerical equivalence. The earlier non-adoption decision in
`golden/attention-integration-20261005/adoption-decision.json` predates these labeled
results. The later [integration review](#adoption-as-the-default) supported adopting the
qualified implementation in main.

This evaluation adds externally labeled document tasks. It does not change the engine,
replace numerical reference tests, or establish Cloudflare leaderboard parity. All four
development/test comparisons have completed. Their frozen outputs support further analysis
without repeating inference.

### Dataset and input preservation

Source: Yuta Koreeda and Christopher D. Manning, [ContractNLI: A Dataset for Document-level
Natural Language Inference for Contracts](https://aclanthology.org/2021.findings-emnlp.164),
Findings of EMNLP 2021. The [official dataset](https://stanfordnlp.github.io/contract-nli/)
is distributed under CC BY 4.0. The archive and its license are retained locally under
`golden/contractnli-20261005/`.

`source.json` records the archive hash and upstream commit
`eced6528dd3c1d14d73f9a87df8f7bdbc03126f9`. The development split contains 61 documents,
each with 17 labeled hypotheses. This adapter places the complete contract in `state` and
all hypotheses in one choice-question schema. The choices are Entailment, Contradiction and
NotMentioned. Reference labels are never part of the requests. Evidence-span extraction is
outside this evaluation.

All 61 documents fit the accepted context without truncation: 2,698–8,658 full tokens,
280,281 tokens total. Both models' encoders, including the candidate's encoder, produce
exactly the same encoded records. There are 1,037 decisions: 519 Entailment, 95
Contradiction and 423 NotMentioned. No document was filtered by length, model prediction or
confidence. The adapter was fixed before evaluating the test split and was not tuned from
its results.

This is a local NLI adapter. Cloudflare's published Decision Index prompt and scoring
protocol have not been recovered, so its numbers are not directly comparable to this
evaluation. The leaderboard's direct HTTP fetch returned 403; the dataset was downloaded
from its official public repository.

### Frozen comparison

`evaluation-manifest.json` pins the inputs, scoring code, declared metrics and two copied
executables:

- `baseline-clef`: the qualified main FP32 build, with caching disabled.
- `candidate-clef`: the previously qualified compensated-attention build, with
  `CLEF_ATTN_TU=1`.

Each arm uses `--strict --no-truncate --batch 1 --logits --time`. Every reported
processed-token count must match the full CPU encoding, and every request must produce
exactly the expected questions and three finite logits per question. Missing or malformed
results fail the run. All inference is local.

The report records correct count, accuracy, macro-F1 across the three labels, mean
multiclass Brier score and mean negative log likelihood. It retains every prediction change,
distinguishing correct-to-wrong from wrong-to-correct. Brier score sums the three squared
probability errors before averaging decisions. CLI durations are descriptive; the arm order
is unsuitable for a paired speedup claim. Independent latency evidence remains in the
attention experiment.

Five initial CPU scorer tests passed: hand-calculated probabilities and metrics, an
imbalanced confusion matrix, malformed/nonfinite/incomplete output rejection, comparison
direction and identity, and logit shift/extreme-value behavior.

### Scoring correction and first result

The first run exposed an adapter bug: the scorer used request insertion order for logits,
but the encoder sorts choice IDs. It therefore swapped Entailment and Contradiction.
Original metric reports at the artifact directory's top level are invalid and retained for
diagnosis; their raw model outputs are valid. `scoring-error.json` records the error.

The corrected scorer reads each question's actual `option_ids` from the encoder. All 1,037
mappings were verified with both models and both encoders. Seven CPU tests pass, including a
regression that fails with the original scorer and passes with the corrected one, and
rejection of malformed option metadata. Corrected reports live in `corrected/`. Re-scoring
uses the existing hash-checked outputs; no inference is repeated.

| Flash metric | FP32 control | Compensated attention |
|---|---:|---:|
| Correct decisions | 924 / 1,037 | 924 / 1,037 |
| Accuracy | 89.1032% | 89.1032% |
| Macro-F1 | 0.843171 | 0.843171 |
| Mean negative log likelihood | 0.322401513 | 0.322401069 |
| Mean multiclass Brier score | 0.168996712 | 0.168998042 |

No Flash decision changes. Probability metrics move slightly in opposite directions:
negative log likelihood improves by 4.44e-7, while Brier worsens by 1.33e-6. These results
do not demonstrate a calibration improvement or resolve the existing mixed numerical
evidence. This development result alone did not settle adoption; the frozen test comparison
and integration checks followed.

The corrected 27B run also preserves every prediction:

| 27B metric | FP32 control | Compensated attention |
|---|---:|---:|
| Correct decisions | 911 / 1,037 | 911 / 1,037 |
| Accuracy | 87.8496% | 87.8496% |
| Macro-F1 | 0.824810 | 0.824810 |
| Mean negative log likelihood | 0.365823504 | 0.365828363 |
| Mean multiclass Brier score | 0.187685085 | 0.187688111 |

Both probability metrics worsen slightly on 27B: NLL by 4.859e-6 and Brier by 3.026e-6.
Across both models, all 2,074 development predictions are unchanged. These are task-accuracy
results on the frozen cohort, not proof of equality on unseen requests.

#### FP32 summation-order control

A separate run of three Flash arms used the qualified main-plus-attention build: current
FP32 attention, compensated attention, and the scalar FP32 reference kernel. The last arm
retains operand precision while changing the summation order.
`parallel-control/verified.json` records a read-only capture and CPU verification: all
processed token counts match, every reported arm metric recomputes exactly, and the
current-FP32 and compensated raw outputs are byte identical to our corresponding arms. No
extra inference was run for this audit.

All three arms get 924/1,037 decisions correct and make identical predictions. The scalar
FP32 control changes mean negative log likelihood by +1.539e-5 and Brier by +1.000e-5,
versus −4.443e-7 and +1.330e-6 for compensated attention. Mean per-question maximum
probability drift is 6.154e-5 for scalar FP32 and 6.199e-5 for compensated attention.
Maximum drift is 0.000933 and 0.000834, respectively. Thus the probability movement is
comparable to a change in FP32 summation order on this cohort; its sign alone does not
establish lost task accuracy.

That script also reports 95% bootstrap intervals from resampling whole documents 10,000
times. For compensated attention, the NLL interval is [−1.661e-5, +1.512e-5] and the Brier
interval is [−8.024e-6, +1.059e-5]. We reviewed the document-level resampling code and
independently recomputed its point estimates, not its bootstrap draws. Intervals containing
zero are not an equivalence test, and an empirical zero interval for observed decision
changes does not establish zero risk on unseen requests.

The completed 27B control was captured and checked the same way in
`parallel-control-27b/verified.json`: all 61 full token counts match, all metrics recompute,
and the FP32 and compensated raw outputs are byte identical to our corresponding arms. All
three arms get 911/1,037 decisions correct with identical predictions. Relative to current
FP32, scalar FP32 changes NLL by +1.854e-5 and Brier by +1.486e-5. Its mean per-question
maximum probability drift is 0.000170, versus 0.000171 for compensated attention; maxima are
0.004375 and 0.001966. The reported document-bootstrap intervals for compensated attention
are [−2.997e-5, +4.193e-5] for NLL and [−2.297e-5, +3.002e-5] for Brier, with the same
interpretation limits as the Flash intervals.

### Interpretation limits

This adds one domain and one adapter, with correlated labels within each document and
possible model training exposure. Agreement or equal aggregate accuracy on these documents
would not prove unchanged accuracy on arbitrary requests. Any changed decisions need
inspection alongside probability scores and the existing numerical evidence. The development
split is an experiment set; the separately frozen test split provides confirmation using the
same adapter.

Run each model under the lock:

```sh
/usr/bin/lockf -k golden/.gpu.lock .venv/bin/python -B golden/contractnli-20261005/corrected/run.py clef-flash
/usr/bin/lockf -k golden/.gpu.lock .venv/bin/python -B golden/contractnli-20261005/corrected/run.py clef
```

The runner refuses to overwrite an existing or partial evaluation. Results are saved as
`clef-flash-evaluation.json` and `clef-evaluation.json` in the artifact directory. The
original 27B attempt stopped before loading the model: its broad `clef-*.log` guard matched
existing `clef-flash-*.log` files. No 27B inference was performed by that attempt.
`failure-model-prefix.json` preserves the failure. The corrected guard checks exact expected
filenames; three additional CPU tests cover this collision and every partial/complete output
path.

Flash was re-scored with `corrected/rescore.py clef-flash`. The new 27B run uses
`corrected/run.py clef` and directly writes valid scores, logs and raw outputs under
`corrected/`. Both binaries and the requests remain the originally frozen ones.

### Frozen test split

The unchanged adapter has also been prepared on all 123 test documents, totaling 2,091
labels and 548,020 full tokens. Lengths span 2,563–10,230 tokens; no document was truncated
or rejected. The development and test document IDs are disjoint. `test-cohort/cohort.json`
records the inputs and actual encoder option IDs, and `test-cohort/evaluation-manifest.json`
pins the same two binaries and corrected scorer. Its seven scorer tests and three
output-guard tests pass.

Flash has completed all 123 full documents in both arms:

| Flash test metric | FP32 control | Compensated attention |
|---|---:|---:|
| Correct decisions | 1,836 / 2,091 | 1,836 / 2,091 |
| Accuracy | 87.8049% | 87.8049% |
| Macro-F1 | 0.836825 | 0.836825 |
| Mean negative log likelihood | 0.329585041 | 0.329584147 |
| Mean multiclass Brier score | 0.173951872 | 0.173952090 |

Every prediction is unchanged. NLL improves by 8.939e-7 and Brier worsens by 2.183e-7. This
confirms unchanged observed task decisions on the held-out Flash cohort, with tiny mixed
probability-score changes. It is not proof of numerical identity or equivalence on arbitrary
inputs.

27B has also completed all 123 documents:

| 27B test metric | FP32 control | Compensated attention |
|---|---:|---:|
| Correct decisions | 1,830 / 2,091 | 1,830 / 2,091 |
| Accuracy | 87.5179% | 87.5179% |
| Macro-F1 | 0.831890 | 0.831890 |
| Mean negative log likelihood | 0.373401323 | 0.373394899 |
| Mean multiclass Brier score | 0.192736988 | 0.192732870 |

Again, every prediction is unchanged. NLL improves by 6.424e-6 and Brier by 4.117e-6. Across
both models and both splits, **all 6,256 predictions remain unchanged**.
`quality-summary.json` freezes the four completed reports and their hashes. Together with
the existing absolute numerical bounds and FP32 summation-order controls, this supports
qualifying the candidate for integration. It establishes no observed task-decision loss on
these cohorts, not a universal accuracy guarantee.

### Task error analysis

`error-analysis/` groups the completed runs by hypothesis and full input length, using the
saved decisions. It does not rerun inference or change prompts. The hardest hypothesis on
both development models and the Flash test run is `nda-20`: whether the receiving party may
retain some confidential information after its return or destruction. Flash gets 83/123 test
decisions correct on that question (67.48%), compared with 87.80% over all questions.
Third-party sharing (`nda-7`) is next at 91/123. Both numerical variants make the same
mistakes.

The Flash test split contains 61 documents below 4,096 tokens, 59 at 4,096–8,191, and only
three at 8,192 or more. Correct decisions are respectively 901/1,037, 892/1,003 and 43/51.
Those buckets differ in content and labels; they do not isolate a context-length effect. In
particular, the three longest documents are insufficient to establish broad long-context
accuracy.

The held-out models also make different mistakes despite similar totals. They both answer
1,749 questions correctly and both miss 174; Flash alone is correct on 87, while 27B alone
is correct on 81. Flash's net advantage of six therefore does not justify substituting it
for 27B under a no-quality-loss requirement. `error-analysis/compare_models.py` verifies the
frozen report hashes and computes this overlap without inference. Both attention
implementations retain the same overlap because all their predictions match.

### Additional long documents

The current compensated-attention implementation preserves all 170 decisions across both
models on five additional full documents of 8,349–13,876 tokens. Flash answers 81/85
correctly in both modes; 27B answers 79/85 correctly in both. This adds long-context
evidence to the earlier development/test evaluation, whose test split had only three
documents above 8,192 tokens. It introduces no production change and does not establish
universal accuracy equivalence.

#### Cohort and comparison

The frozen cohort includes every document from the previously unused ContractNLI training
split whose complete encoded request has at least 8,192 tokens. The unchanged adapter
supplies the entire contract and all 17 hypotheses without labels. Of 423 documents, five
meet the length criterion, 418 fall below it, and none is rejected. Selection precedes
inference and uses no model outputs. The five selected texts are unique and their IDs and
exact text hashes are disjoint from the previous development/test cohorts.

The five requests contain 50,755 tokens and 85 labels: 59 Entailment, 9 Contradiction and 17
NotMentioned. CPU verification reconstructs requests and labels from the pinned public
archive, repeats the length selection over all 423 documents, and checks identical complete
encodings across both models and strict mode. Every inference call reports the expected full
token count.

This uses the dataset's training split only as additional evaluation data; no training or
tuning occurs. Possible exposure during the models' original training is unknown. Five
documents from one domain provide limited coverage, and this cohort does not reach 16K
tokens.

Both arms use the current root CLI, original model weights, strict mode, no truncation and
no cache. `CLEF_ATTN_TU=0` selects the FP32-attention control; `CLEF_ATTN_TU=1` selects the
current default compensated attention. Other engine arithmetic is unchanged. These are
attention-mode comparisons, not comparisons against an entirely FP32 model.

Two CLI processes remain resident, but only one inference request is outstanding. Each warms
on the longest document. Each document then receives one ABBA quartet, alternating which arm
goes first by document. There are 22 forward calls per model including warmups. Every
repeated raw response must match its earlier response within that arm.

#### Accuracy and probability scores

| Metric | Flash control | Flash default | 27B control | 27B default |
|---|---:|---:|---:|---:|
| Correct decisions | 81/85 | 81/85 | 79/85 | 79/85 |
| Accuracy | 95.2941% | 95.2941% | 92.9412% | 92.9412% |
| Macro-F1 | 0.949485 | 0.949485 | 0.903425 | 0.903425 |
| Mean negative log likelihood | 0.261499946 | 0.261504964 | 0.309251686 | 0.309259559 |
| Mean multiclass Brier score | 0.115439954 | 0.115436202 | 0.146664968 | 0.146655241 |

Neither model changes any prediction, including its errors. Probability metrics move
slightly in opposite directions: NLL worsens by 5.02e-6 on Flash and 7.87e-6 on 27B; Brier
improves by 3.75e-6 and 9.73e-6 respectively. These changes do not establish a calibration
improvement. Flash's higher correct count on this small cohort does not qualify it as a
replacement for 27B.

#### Descriptive timing

Each cell below averages two measured CLI inference times. One quartet per document is
insufficient to replace the larger paired performance qualification. Two resident engine
contexts and the run order also limit comparisons with earlier absolute latencies.

| Full tokens | Flash control → default | Time reduction | 27B control → default | Time reduction |
|---|---:|---:|---:|---:|
| 10,645 | 4.335 → 4.045 s | 6.69% | 15.116 → 15.180 s | −0.42% |
| 8,770 | 3.505 → 3.368 s | 3.91% | 11.784 → 11.986 s | −1.71% |
| 8,349 | 3.438 → 3.279 s | 4.63% | 11.615 → 11.086 s | 4.55% |
| 9,115 | 3.789 → 3.623 s | 4.38% | 12.690 → 12.253 s | 3.44% |
| 13,876 | 6.242 → 6.440 s | −3.18% | 20.838 → 19.516 s | 6.34% |

Flash improves on four documents and regresses on the longest; 27B improves on three and
regresses on two. The measurements do not establish a uniform speed gain. This run's purpose
is extending labeled accuracy coverage, and it does not alter the existing dispatch or
adoption decision.

A [six-quartet follow-up](long-request-timing.md#six-quartet-reproduction) investigates the
13,876-token Flash reversal. It finds a lower default median with mixed paired results, and
a separate stage observation locates the reproduced slower quartet primarily inside the GPU
execution interval. The original measurements above remain unchanged.

## Adoption as the default

The reviewed candidate preserves the full accepted input and the original BF16 weights. It
passes both models' numerical, batching, prefix-cache and template checks against the saved
evaluated implementations. **Compensated attention is now the main default**, with
`CLEF_ATTN_TU=0` retaining the FP32 control. The main checkout rebuilt without warnings with
the measured template-eligibility rule below.

### Quality evidence

Compensated attention uses high and residual FP16 planes with FP32 accumulation. Three
tensor products approximate each attention product; the residual-times- residual term is
omitted. This changes numerical results. It does not quantize weights or shorten context.

All 6,256 ContractNLI predictions across the frozen development and test splits remain
unchanged. The small probability-score shifts are mixed and comparable in scale to the
measured FP32 summation-order controls. The earlier public FP32-reference comparison remains
less favorable to the candidate:

| Model | FP32 attention mean probability error | Compensated mean error | Compensated maximum error |
|---|---:|---:|---:|
| Flash | 0.00003576 | 0.00004576 | 0.00023393 |
| 27B | 0.00007982 | 0.00009400 | 0.00113585 |

Both paths retain all 46 public oracle decisions and pass the existing absolute error
bounds. These numerical and labeled-task measurements answer different questions; neither
establishes identical probabilities or universal accuracy. Only three ContractNLI test
documents exceed 8,192 tokens. Separate synthetic fixtures exercise full 16,347-token
inputs, but do not replace broad task coverage. See [the precision
experiments](#compensated-attention-precision-screen) and [the labeled task
evaluation](#labeled-evaluation-on-contractnli).

### Implementation and qualification

The integration fixes a synchronization issue in the newer cache prototype: each SIMD group
now has one writer to its own rescaling-vote slot. A barrier precedes combining the votes.
All 32 tested kernel shapes retain exact output bits, with additional split-plane,
cached-layout, poison and overflow checks. The original task-evaluation candidate already
used atomic votes.

The engine fixes its attention mode when opened. Prefix entries store either FP32 K/V or two
half planes with the same total byte size; capacity growth copies the corresponding layout.
Query blocks remain aligned to the record's first token, preserving exact standalone, packed
and resumed results within each mode. Overflow reruns use the original
BF16-input/FP32-attention path. Error propagation, entry ownership and unsupported-mode
bypasses remain explicit.

Both models pass nine numerical comparisons against frozen outputs: public packed poison,
fresh full-context requests, an eight-document ContractNLI subset, and independently
referenced attention-only overflow. Both also pass five cache and server suites covering
fill/hit poison, capacity transitions, failure recovery, template reuse, HTTP bytes, key
isolation, budgets and GEMM boundaries. The ContractNLI subset checks implementation
identity; it is not another independent task evaluation. No full evaluation was repeated for
this integration.

### Final paired measurements

Fresh-input timing uses one resident model and switches modes only between complete uncached
requests. Both arms keep the selected Flash GEMM dispatch and the existing 27B GEMMs. Each
length has four warm-ups and four ABBA quartets, eight measured calls per arm. Times include
the CPU head and exclude model loading, request encoding and HTTP.

| Full tokens | Flash FP32 → compensated, ms | Reduction | 27B FP32 → compensated, ms | Reduction |
|---:|---:|---:|---:|---:|
| 346 | 123.76 → 122.13 | 1.32% | 440.23 → 438.75 | 0.34% |
| 594 | 210.46 → 210.16 | 0.15% | 718.07 → 714.81 | 0.45% |
| 1,024 | 342.62 → 341.50 | 0.33% | 1,163.15 → 1,151.83 | 0.97% |
| 1,382 | 471.96 → 476.18 | -0.89% | 1,598.07 → 1,579.95 | 1.13% |
| 2,048 | 712.28 → 691.74 | 2.88% | 2,338.73 → 2,305.97 | 1.40% |
| 2,235 | 783.42 → 770.81 | 1.61% | 2,564.18 → 2,525.17 | 1.52% |
| 4,510 | 1,705.12 → 1,645.42 | 3.50% | 5,506.69 → 5,352.58 | 2.80% |
| 8,072 | 3,397.84 → 3,237.39 | 4.72% | 10,618.21 → 10,169.65 | 4.22% |
| 16,347 | 8,063.81 → 7,370.00 | 8.60% | 24,311.82 → 22,512.07 | 7.40% |

All four quartets on both models improve at 2,048 tokens and above. Shorter results are
small or mixed. The 1,382-token medians regress despite three positive quartets, which
illustrates why a single median is insufficient to select a length cutoff. Template and
fixed-mode prefix-hit comparisons are complete on both models. Each 21-length template
comparison preserves exact logits on 420 timed calls. Flash regresses by 5–6% at several
lengths near 1K where ordinary and cached GEMMs choose different row tiles. Preserving the
ordinary tile in an isolated experiment did not recover that loss, so that backend change
was rejected. A second comparison checks ten padding boundaries on each model: all four
quartets improve at every length where removing 32 tokens eliminates a padded 64-row tile.
The measured reductions there are 2.6–5.1% on Flash and 3.0–6.1% on 27B. Non-saving
boundaries are flat or mixed. This supports restricting reuse by tile/padding geometry; see
[the template policy](prefix-cache.md#fixed-template-entry). Every path retains the full
input.

Runtime qualification now passes all five lanes: four CLI error tests, nine CLI
allocation/cleanup cases, command-buffer error propagation and recovery,
cross-model/reopened-engine cache rejection, and idle keep-warm behavior. The
byte-preservation test includes the new attention block-list buffer. On Flash, keep-warm
reduces the median post-idle GPU start delay from 111 ms to 0.8 ms with identical response
bytes. HTTP qualification also passes: 23/23 CLI response comparisons (including
overlong-input rejection), 32/32 concurrent clients, and the request-method, path, header,
body-limit and keep-alive checks.

Matching-prefix hits also improve in a separate paired comparison:

| Full tokens | Flash FP32 → compensated hit, ms | 27B FP32 → compensated hit, ms | Reused tokens |
|---:|---:|---:|---:|
| 1,382 | 121.68 → 119.96 | 380.26 → 370.31 | 1,056 |
| 2,235 | 110.87 → 106.55 | 360.33 → 350.25 | 1,952 |
| 8,072 | 144.77 → 132.17 | 587.30 → 547.72 | 7,808 |
| 16,347 | 199.37 → 171.00 | 753.53 → 669.23 | 16,064 |

Each arm has its own fixed-mode engine and cache entry, checked against an independent full
pass in that mode. There are four warm-up hits and four ABBA quartets per length, plus one
fill observation per arm. These are paired hit latencies, not fresh-input times. All four
Flash quartets improve at 2,235, 8,072 and 16,347 tokens; the 1,382-token result is mixed.
All four 27B quartets improve at every tested length. Cache byte counts are identical
between modes.

Artifacts are under `golden/attention-main-review-20261005/`: `reviewed/` is the frozen
build, `reviewed-check-manifest.json` pins its inputs, the `reviewed-numeric-*` and
`reviewed-cache-*` reports record completed checks, and `timing/` preserves the raw samples
and paired drivers. The final checks include 110 exact template-boundary logits, 175 public
logits in each attention mode, 240 fresh full-context logits, 195 GEMM-boundary/fallback
logits and 350 public cache fill/hit logits per model, plus HTTP fallback and isolation
checks. Flash also runs the host, runtime-error and attention-kernel suites. Existing golden
data is unchanged.

## Independent attention comparison with MLX

The isolated attention experiment uses identical unrounded FP32 Q/K/V arrays, causal GQA,
head dimension 256 and both diffuse and sharp score distributions. Cleffa's attention core
was exposed through an MLX custom kernel with FP32 output before gating and activation
rounding. MLX used its [scaled-dot-product attention
API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html).

At 4,510 tokens and 24 query heads / 4 KV heads:

| Path | Median | Maximum sampled absolute error against float64 |
|---|---:|---:|
| Cleffa core, default-mode comparison | 33.01 ms | 5.89e-6 |
| MLX SDPA, default TF32 | 22.33 ms | 2.27e-3 |
| Cleffa core, full-precision comparison | 32.72 ms | 5.89e-6 |
| MLX SDPA, `MLX_ENABLE_TF32=0` | 49.16 ms | 6.85e-6 |

Each comparison has four warm-ups and six measured calls per path in alternating order, then
64 float64 samples. The default MLX path fails the sampled FP32 error bound of `2e-5 * (1 +
abs(reference))`. Disabling TF32 passes that bound but is slower. This experiment does not
validate packed-record invariance or complete-model quality for an MLX replacement; no
replacement was adopted.

## Artifacts

Local, git-ignored evidence directories under `golden/`:

- `perf-attention-20261004/`: shared-probability kernel, timings and validation.
- `perf-packed-attention-20261004/`: packed reuse experiment and qualification.
- `attention-keys-20261004/`, `head-parallel-20261004/`: key-block experiment, profiles and
  whole-request prefetch comparisons.
- `attention-retention-20261004/`: retained prefetch build and final qualification.
- `attention-scale-20261004/`: rescaling screen, paired samples and final build.
- `attention-groups-20261004/`: SIMDgroup-split experiment and the FP32-mantissa fixture.
- `attention-order-20261004/`: compensated-attention precision screen.
- `eval-sensitivity-20261005/`: CPU audit of the control results and leave-one-request-out
  sensitivity.
- `heldout-20261005/`: 24 fresh synthetic requests, their FP32 references and both models'
  results.
- `attention-integration-20261005/`: reconstructed candidate, qualification, paired timing
  and the initial non-adoption decision.
- `contractnli-20261005/`, `contractnli-long-20261005/`: dataset archive, frozen cohorts,
  corrected scorer, raw outputs and completed reports.
- `attention-main-review-20261005/`: reviewed cache-capable build, numeric and cache checks,
  paired timing and the adoption record.
