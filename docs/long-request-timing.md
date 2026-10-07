# Long-request timing variation

A 13,876-token ContractNLI document ran slower with the default compensated attention than
with the FP32 control in one quartet of the
[long-document accuracy check](attention.md#additional-long-documents). This report records
the follow-up: a six-quartet reproduction with stage timing, synchronized GPU clock and
power telemetry, an Instruments limiter-counter trace, a shader-timeline capture with the
built-in category profile, and a sustained 27B batch run with fan and clock readings.

The conclusion: the default has the lower median in every repeat, the reversals coincide
with lower GPU clocks in the default arm, the GPU is active for more than 99% of each
request, and matrix projections dominate. No dispatch change was supported, and none of
this establishes a hardware ceiling.

## Six-quartet reproduction

The 13,876-token document that was slower with default attention in the small accuracy run
does not show a consistent slowdown under repeated measurement. Default attention has a
lower median in both follow-ups, but some matched quartets remain slower. No dispatch change
is supported by these results.

Stage timing locates the reproduced regression primarily inside the GPU command buffer's
execution interval. It is not explained by CPU head work, host encoding or an overflow
rerun. GPU scheduling and clock or power behavior were not measured, so this does not
identify a particular kernel or establish a hardware ceiling.

### Fixed reproduction

The request is the complete public ContractNLI document 622 with all 17 original hypotheses,
encoded as 13,876 tokens. It is unchanged from the [long-document accuracy
check](attention.md#additional-long-documents). Both arms use the current root CLI and
original Flash weights. The control sets `CLEF_ATTN_TU=0`; the default sets it to `1`. No
cache, truncation or source modification is used.

Two CLI processes remain resident with only one request outstanding. Four warmups in ABBA
order precede six measured quartets, alternating ABBA and BAAB. Each arm has 12 measured
calls. Every response must match its saved prior-mode logits, and repeated raw response text
must be identical within each arm.

Before inference, the gate requires at least a 2% median difference and all six quartets in
the same direction to establish a consistent improvement or slowdown. Other results are
inconclusive. This is a diagnostic gate, not a new production adoption criterion.

| Run | FP32-attention median | Default median | Median reduction | Faster quartets |
|---|---:|---:|---:|---:|
| Plain timing | 6.229 s | 5.944 s | 4.57% | 4/6 |
| Existing stage timers enabled | 6.418 s | 6.085 s | 5.20% | 5/6 |

The plain run's quartet reductions are +6.67%, +4.32%, −5.05%, −6.74%, +5.88% and +6.69%.
The stage run's are +6.74%, +1.30%, −5.74%, +1.56%, +7.39% and +7.02%. Both fail the
consistency gate. These are separate runs; their absolute medians are not a paired test of
instrumentation overhead.

### Stage attribution

The follow-up enables the existing `CLEF_STAGE_TIME=1`. It leaves the ordinary
command-buffer structure intact and does not enable the serializing category profiler. Each
request records host encoding, commit-to-completion wait, the GPU-reported command-buffer
interval and CPU head time. All 28 calls, including warmups, have exactly one GPU pass.

| Measured stage | Control median | Default median |
|---|---:|---:|
| Host encoding | 0.17 ms | 0.17 ms |
| GPU execution interval | 6,280.98 ms | 5,911.63 ms |
| Wait outside GPU interval | 105.58 ms | 104.78 ms |
| CPU head | 77.38 ms | 77.56 ms |

Component medians are calculated separately and need not sum to the total median. Wait minus
GPU interval includes submission/completion and host rescheduling overhead; it is not
exclusively queue delay. The GPU interval may include scheduling or preemption and is not a
measurement of exclusive device occupancy.

In the slower third quartet, default attention adds 361.50 ms to total request time. The GPU
interval increases by 416.73 ms, while wait outside that interval falls by 55.17 ms. CPU
head time differs by −0.10 ms and encoding by −0.02 ms. This narrows the investigation to
behavior during GPU execution. It does not justify reducing context, changing the rubric, or
switching attention by input length.

Across measured calls, the default's GPU interval ranges from 5,499.48 to 6,657.35 ms; the
control ranges from 5,931.67 to 6,422.44 ms. Clock, power and per-process GPU scheduling
telemetry would be needed to distinguish their contributions. No such causal attribution is
claimed from these timings.

## GPU clocks during the reversals

The intermittent 13.9K Flash reversals coincide with lower GPU clocks in the
default-attention arm. In the two slower matched quartets, the default runs at about 1.07
GHz while the FP32-attention control runs at 1.22 GHz. In quartets where the clocks are
similar, the default is 6.7–8.1% faster. This supports clock variation as a contributor to
the timing reversals; it does not identify the policy or physical limit responsible for
changing the clocks.

### Synchronized observation

The input and six-quartet protocol are the same as the previous [timing
investigation](#six-quartet-reproduction). Four warmups precede six alternating ABBA/BAAB
quartets. Each arm has 12 measured calls. The existing nonserializing stage timers remain
enabled.

This run adds persisted request start/end times using both wall and monotonic clocks.
Read-only `powermetrics` captures GPU clocks, requested performance state, activity,
estimated power and thermal pressure. The existing SMC helper records fans and Foundation
power/thermal state.

Power sample timestamps have whole-second precision. To avoid mixing adjacent requests, a
sample is joined only when its entire conservative interval `[timestamp - elapsed_seconds,
timestamp + 1 second]` fits inside the request. Every measured call retains at least three
interior power samples. Clock means weight sample duration and GPU active fraction; power
means weight duration. Fan readings are point observations inside the request interval.

### Result

The default median is 6.228 s versus 6.502 s for the control, a 4.22% reduction. Four
quartets improve and two regress, so the frozen all-quartets consistency gate remains
inconclusive. This is an instrumented observation, not a new production performance
qualification.

| Quartet | Control time | Default time | Control active clock | Default active clock | Default time reduction |
|---|---:|---:|---:|---:|---:|
| 1 | 6.130 s | 5.631 s | 1,269 MHz | 1,274 MHz | 8.15% |
| 2 | 6.841 s | 6.327 s | 1,132 MHz | 1,138 MHz | 7.51% |
| 3 | 6.485 s | 6.633 s | 1,216 MHz | 1,072 MHz | −2.28% |
| 4 | 6.349 s | 6.673 s | 1,225 MHz | 1,068 MHz | −5.11% |
| 5 | 6.527 s | 6.090 s | 1,198 MHz | 1,166 MHz | 6.69% |
| 6 | 6.589 s | 6.145 s | 1,172 MHz | 1,164 MHz | 6.75% |

Table entries average the two calls per arm within each quartet; they are distinct from the
overall medians. Across 12 measured calls per arm, the descriptive Pearson correlation
between GPU interval and interior active clock is −0.995 for the control and −0.987 for the
default. Sequential observations are not independent trials, and correlation does not
establish causation or a linear speed prediction.

All retained interior samples report 100% GPU activity and 100% requested P13, the highest
requested state shown by the tool. Actual average active clocks still vary from 1,127–1,306
MHz for control calls and 1,049–1,281 MHz for default calls. This is no evidence that
requesting a higher software performance state would fix the reversals.

The measured GPU interval accounts for most request time. Each call has one GPU pass; CPU
head medians remain about 77 ms and host encoding about 0.17 ms. The read-only telemetry
also reports lower estimated GPU power in the slower default quartets: about 33–34 W versus
43–44 W for the control. Estimated power is an observation on this device, not a
device-to-device efficiency comparison or proof of a particular power limit.

### Cooling and configured mode

Fans rise from roughly 25% of their reported maxima during the first quartet to full speed
during the sixth. Both first exceed 98% at 141.2 seconds from the first warm request and
remain there for the final 33.6 seconds of inference. The last quartet still favors the
default by 6.75%. This does not measure a pre-cooling intervention or show that external
cooling would improve sustained performance.

Foundation reports low-power mode off throughout. Its thermal state changes from nominal to
fair; the power sampler's separate pressure labels reach Moderate and Heavy later in the
run. The two slower default quartets retain Nominal labels in their joined samples. These
coarse labels therefore do not identify what caused their clock differences.

The Mac is on AC power with `pmset` reporting AC `powermode 2`. A subsequent read-only
System Settings inspection, verified through accessibility and a screenshot, shows **On
power adapter: High Power** and **On battery: Automatic**. The connected adapter reports 140
W. No preference was changed.

`system_profiler SPPowerDataType` instead reports AC HighPowerMode No and LowPowerMode Yes.
That conflicts with the configured UI mode and Foundation's low-power indicator. The report
is retained with charger identifiers omitted; the discrepancy is not treated as proof that
low-power mode is active or that a particular macOS defect caused the timings.

Apple documents that [High Power mode permits higher fan
speeds](https://support.apple.com/en-euro/101613) and may improve intensive workloads. It
does not promise a fixed GPU clock. Apple's [GPU profiling
guidance](https://developer.apple.com/documentation/xcode/optimizing-gpu-performance) also
identifies thermals and system settings as factors in GPU performance state. Neither source
establishes the cause of this machine's observed changes.

## Full-request limiter trace

Cleffa is marked active for more than 99.2% of each recorded GPU interval on this complete
13,876-token Flash request. The two warm passes contain only 40.1 and 43.7 ms of scheduling
gaps across 5.28 and 5.47 seconds. This capture does not show a large host-submission or
GPU-scheduling gap to remove.

Matrix-accelerator utilization has a high median, but a lower average across the whole
request. Those are different observations. They do not establish a whole-model hardware
ceiling or a bound on possible kernel improvements.

### Capture and validation

The unchanged current CLI processes public ContractNLI document 622 three times with default
compensated attention, strict encoding, truncation disabled and no prefix cache. All three
responses retain 13,876 tokens and match the saved default-mode logits. The first pass
includes allocation and warmup.

Instruments uses the Performance Limiters counter set with its induced performance-state
setting left at Default. No system preference, production source, binary or model changed.

The original wrapper then failed while parsing a response file because `xctrace` merged
diagnostics with JSON output. `recover.py` validates the saved stream, all three outputs and
the original capture artifacts without repeating inference. Its
`capture-recovered-result.json` is the authoritative capture result.

### Execution intervals

Each request has one submitted command buffer and one compute encoder. Its Active interval
segments are merged before calculating elapsed time and gaps. These segments are scheduling
intervals, not individual kernel durations.

| Measurement | First pass, warmup | Second pass | Third pass |
|---|---:|---:|---:|
| Recorded GPU span | 4,521.09 ms | 5,278.71 ms | 5,469.35 ms |
| Target Active time | 4,487.99 ms | 5,238.65 ms | 5,425.66 ms |
| Gaps within GPU span | 33.09 ms | 40.07 ms | 43.69 ms |
| Active share of GPU span | 99.27% | 99.24% | 99.20% |
| Largest gap | 0.493 ms | 0.504 ms | 0.341 ms |
| CPU submission end to first Active interval | 153.24 ms | 1.31 ms | 1.78 ms |
| CPU head time | 150.72 ms | 79.35 ms | 80.41 ms |

The GPU spans agree with the CLI's recorded GPU intervals within 0.003 ms. Other
applications have active GPU work during 34.49 and 38.34 ms of the two warm passes' gaps.
Overlap does not establish why the target was inactive. Removing every observed gap would
cover less than 0.8% of these GPU spans; this arithmetic is not a measured optimization or a
limit on faster kernels.

These are profiled requests from one process, not an ordinary latency benchmark or a
comparison with FP32 attention. The shorter first pass must not be used as evidence that
allocation makes inference faster.

### Counters and desktop overlap

The export contains 52,515,898 counter rows. The streaming parser retains 23 named counters.
Every selected counter series spans all three requests. Within the warm request spans,
maximum sample gaps are about 0.024 ms for the accelerator counters and 0.151 ms for the
bandwidth/cache counter family.

Ghostty, WindowServer and Activity Monitor also use the GPU during the capture. Their
combined Active intervals overlap 2.742 and 2.779 seconds of the target's Active time in the
warm passes. The analysis reports all target-active samples and separately excludes
timestamps inside any observed other-process Active interval. Approximately 105,000 and
111,000 accelerator samples remain.

| Counter, excluding observed other-process overlap | Second pass mean / median | Third pass mean / median |
|---|---:|---:|
| Neural Accelerator Utilization | 76.21% / 97.50% | 75.87% / 97.99% |
| Neural Accelerator Limiter | 76.39% / 97.62% | 76.03% / 98.09% |
| Kernel Occupancy | 22.39% / 16.67% | 22.11% / 16.67% |
| Last Level Cache Utilization | 88.20% / 100.00% | 77.59% / 77.47% |

The exported counter description defines Neural Accelerator Utilization as executed GEMM
work relative to the accelerator's peak performance. The high median supports substantial
accelerator use during many samples. The lower mean includes phases with little or no
accelerator use. A 16.67% occupancy median therefore does not by itself demonstrate idle
matrix throughput.

These are sample statistics, not time-weighted integrals. The counters are device-wide, and
excluding overlap at a timestamp does not prove exclusive occupancy throughout that
counter's averaging window. Desktop overlap also changes which execution phases remain in
the sample. Neither mean nor median is a reliable whole-model speedup bound.

The literal trace performance-state labels shift from mostly Maximum in the first pass to
mostly Medium in later passes. This capture does not contain a validated mapping from those
labels to numeric clocks. The separate [synchronized clock
observation](#gpu-clocks-during-the-reversals) provides the measured association between
clocks and request times.

### Consequence for further work

The small scheduling gaps do not support prioritizing host-submission changes for this
request. The counter averages leave execution phases that need attribution before selecting
another kernel change. Shader Timeline was disabled, and the exported shader-interval table
has zero rows. Compiled shader names cannot supply missing execution durations.

A subsequent trace with individual kernel timing could identify which phases have low
accelerator use and distinguish attention, recurrence and other operations. This capture
alone does not qualify a new optimization. Existing production defaults remain unchanged.

This long-request evidence is separate from the [article workflow
evaluation](performance-history.md#batched-classification-workload), where dense matrix
operations account for 87.5% of the serialized GPU category profile and neither tested
rubric reduction meets the no-regression condition.

## Shader timeline and category profile

The current 13,876-token Flash workload spends about 70% of its serialized GPU category
profile on matrix projections, 18% on attention and 6% on the recurrent scan. All profiled
outputs match the previously saved default-mode logits. These component timings identify
optimization targets; they are not an ordinary request-latency comparison.

### Shader timeline coverage

A separate Instruments template enables Shader Timeline, disables the counter set and leaves
the induced performance state at Default. Encoding is strict, truncation is disabled, and no
cache is used. The first call includes allocation and warmup.

Apple describes the shader timeline in its [Metal profiling
guidance](https://developer.apple.com/documentation/xcode/analyzing-the-performance-of-your-metal-app/).
The new export contains 10,504 shader intervals, including 6,466 attributed to the target
process. It confirms execution of the expected matrix, attention, recurrent and elementwise
kernels. Interval counts are not dispatch counts.

| Measurement | First pass | Second pass | Third pass |
|---|---:|---:|---:|
| GPU span | 4,628.29 ms | 5,348.89 ms | 5,549.32 ms |
| Target Active time | 4,520.59 ms | 5,229.03 ms | 5,423.23 ms |
| Union of attributed shader intervals | 1,933.57 ms | 2,167.65 ms | 2,151.15 ms |
| Attributed share of target Active time | 42.77% | 41.45% | 39.67% |

Every target shader interval lies inside a target Active interval. Overlap between different
shader names is negligible. Nonetheless, the exported timeline leaves most active time
unattributed, so normalizing the visible shader durations into a complete workload breakdown
would be misleading. The missing attribution is not evidence that the GPU was idle.

GPU spans match the CLI's GPU timer within 0.005 ms. Desktop GPU activity is reported
separately. No counters from the earlier [limiter capture](#full-request-limiter-trace) are
joined to this different recording.

The recording, target and TOC export succeeded. The wrapper expected the disabled
counter-set label to be `None`, while this Instruments version exports `(null)`. CPU-only
recovery validates the saved template and capture, all full token counts and every prior
logit. It preserves the original failed wrapper and does not repeat inference.
`capture-recovered-result.json` is the authoritative result.

### Current built-in category profile

To fill the attribution gap, a separate resident current CLI processes the same three
requests with `CLEF_PROFILE=1`. This profiler ends and waits for each category's command
buffer. It covers all categories, but changes submission timing and may affect clocks and
GPU overhead.

| GPU category | Second pass | Third pass | Share across those passes |
|---|---:|---:|---:|
| Matrix projections | 3,579.1 ms | 3,700.2 ms | 70.38–70.63% |
| Attention | 893.5 ms | 948.3 ms | 17.63–18.04% |
| Recurrent scan | 296.6 ms | 307.5 ms | 5.85% |
| SwiGLU | 104.0 ms | 104.9 ms | 2.00–2.05% |
| Convolution and recurrence preparation | 67.5 ms | 70.2 ms | 1.33–1.34% |
| Remaining categories combined | 126.8 ms | 126.6 ms | 2.41–2.50% |

The profiler's stage called `encode` includes its synchronous GPU waits. It must not be
interpreted as tokenizer or CPU command-encoding overhead. The table includes the GPU half
of the head in the remaining categories; the CPU head is outside these category sums.

Attention remains the largest target after matrix projections. This led to an isolated
[register-softmax screen](rejected-experiments.md#register-softmax). It preserved compared
output bits but failed its performance gate. No engine change was adopted from either
profiling run.

## Sustained 27B batches and cooling

Both fans reach their reported maximum speeds under the existing settings. For the final 90
seconds of observed inference, both remain above 98% of those maxima. A manual maximum-fan
setting would not provide appreciably more fan speed during that part of the run. This
observation does not measure the effect of pre-cooling, different ambient conditions or
external cooling.

The Mac is on AC power. The AC profile reports `powermode 2`, and Foundation reports
low-power mode disabled throughout. No fan or power setting was changed.

### Fixed workload

One resident current 27B process executes 18 identical eight-article batches. Every batch
contains 10,331 tokens and all seven original questions. There is no prefix cache,
truncation or prompt change. The runner checks all 144 response lines against the saved
current-build control, including reported token usage.

| Inference measurement | Time per eight articles |
|---|---:|
| First batch | 10.528 s |
| Median of batches 3 to 6 | 11.956 s |
| Slowest batch, number 7 | 12.791 s |
| Median of batches 15 to 18 | 12.074 s |

The late median is about 1% slower than the early measured window, and about 15% slower than
the first batch. This distinguishes the initial burst from sustained throughput. It does not
demonstrate continuously worsening inference over a long article job.

Read-only `powermetrics` and an SMC sampler run alongside inference. The SMC program reads
key metadata and values only; it contains no write command. GPU summaries below include
samples with at least 95% activity and more than 10 W reported GPU power. Times start at the
first such sample.

| Elapsed window | Mean active GPU clock | Mean GPU power | Mean left / right fan reading |
|---|---:|---:|---:|
| 0 to 30 s | 1,382 MHz | 61.0 W | 455 / 489 RPM |
| 30 to 75 s | 1,212 MHz | 45.7 W | 2,416 / 2,608 RPM |
| 75 to 120 s | 1,201 MHz | 42.9 W | 4,032 / 4,353 RPM |
| 120 to 165 s | 1,261 MHz | 47.4 W | 5,345 / 5,771 RPM |
| 165 s to end of busy interval | 1,225 MHz | 44.4 W | Approximately 5,350 / 5,775 RPM |

The reported fan maxima are 5,349 and 5,777 RPM. Both reach 98% of those values after 121.5
seconds and remain there through the final busy sample, 90.5 seconds later. Foundation's
thermal state moves from nominal to fair around that point. The GPU frequency histogram
includes a 1,620 MHz state, but sustained activity often runs below it even while the
requested state is the highest one.

This trace cannot distinguish local temperature limits, package power limits or other
clock-management behavior. GPU frequency alone also does not establish tensor throughput or
a linear request-speed prediction. It supplies no evidence for an additional large sustained
gain from merely selecting maximum fans under these conditions.

The run's telemetry shutdown failed after inference completed, so per-batch timestamps were
lost; the recovered result keeps the frozen batch-latency windows and reports GPU and fan
data in separate elapsed-time windows rather than inventing per-batch joins.

## Artifacts

Local, git-ignored directories under `golden/`: `flash-long-timing-20261005/` and
`flash-long-stage-20261005/` (six-quartet runs, frozen runners, per-call events, verifiers),
`flash-long-telemetry-20261005/` (synchronized power and fan samples, `analysis.json`),
`flash-long-trace-20261005/` (limiter capture, exported counters, recovery),
`flash-long-shaders-20261005/` (shader timeline capture and category profile) and
`thermal-article-20261005/` (sustained batch run and recovered observation).
