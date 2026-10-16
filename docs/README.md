# Reports

The detail behind the top-level README, which stays a short current-state summary, and the
measurement and qualification records behind its numbers and the invariants in `CLAUDE.md`. Each
file covers one topic; add new evidence to the matching report rather than starting a dated
file. Raw artifacts (sources, binaries, samples, manifests) are
local and git-ignored under `golden/<experiment>-<date>/`, named in each report's artifacts
section.

| Report | Contents |
|---|---|
| [usage.md](usage.md) | Request inputs (images, video, Omni audio) with every limit and its reason, the HTTP server and its limits, the template and prefix caches, CLI options and diagnostic switches |
| [accuracy.md](accuracy.md) | Evaluation metrics; dense-model agreement with the FP32 engineering reference; why FP16 activations and how their range is guarded; the PyTorch long-input attention defect; host byte parity; image and video parity |
| [security.md](security.md) | Untrusted-input bounds and the measurements behind them, the media worker's isolation, server exposure, reference code, strict mode, truncation, hosted number serialization and the deliberate divergences from the reference |
| [testing.md](testing.md) | The test command catalog, parity gates, how the FP32 references are produced, debug hooks and the Omni qualification entry point |
| [performance-history.md](performance-history.md) | Every retained change since `db38cfc`, in order, with its paired measurements; status at 2026-10-05; the 2026-10-03 PyTorch comparison; GPU profiles and accelerator counters; the batched classification workload; stock PyTorch controls and current native/hosted timings; Omni decoder performance; dense CPU-head workers, head memory and head-loader packing |
| [attention.md](attention.md) | Kernel selection; the FP32 path (shared probabilities, 64-key prefetch, direct rescaling); the compensated tensor-unit precision screen, the ContractNLI labeled evaluation and the adoption measurements |
| [prefix-cache.md](prefix-cache.md) | Exact FP32 prefix reuse, recurrent-state checkpoints, the fixed-template entry and the Flash GEMM rule near 1K tokens |
| [hosted-comparison.md](hosted-comparison.md) | The 2026-10-10 Flash/27B/Omni core comparison with stock BF16 controls, labeled BANKING77 and CLINC150+OOS results, the corrected BF16 (MPS causal-attention) comparison, video transport and number-serialization probes; the 2026-10-03 Jev comparison; earlier Workers AI measurements on the checkout fixtures and the public corpus, the truncation observations, and the local replay against saved hosted responses |
| [long-request-timing.md](long-request-timing.md) | Why a 13.9K-token Flash request sometimes ran slower with the default attention: stage timing, GPU clocks, limiter and shader traces, sustained thermal behavior |
| [rejected-experiments.md](rejected-experiments.md) | Attention, GEMM, FFN, DeltaNet, workload-level and vision changes, dense precision controls and buffer candidates, and Omni candidates that were measured and not retained, with the criterion each failed |
| [vision.md](vision.md) | Images: the preprocessing and tower pipeline, host byte parity, FP32-oracle parity of both operand modes, costs, cache behavior, limits and divergences |
| [video.md](video.md) | Video: frame-array input, encoded MP4/MOV uploads through the optional `clef-media` worker, sampling, timestamps and frame pairs, the offline converter and its isolation, host and FP32-oracle parity on the video corpus, costs, limits and divergences |
| [omni.md](omni.md) | Clef-Omni: status and the open adoption decision between the fast default and the FP32-compatible branches, running it, architecture, the verification policy, results against the FP32 and FP64 references, where the error comes from, latency, hosted and labeled agreement |

**Evidence archive.** `tools/evidence_archive.py OUT.tar.gz` packs the checkable part of
`golden/` for publication as a release asset (`cleffa-evidence-<date>.tar.gz`): result,
manifest, sample and log files, experiment sources, saved logits, and the small files of the
oracle directories so `tests/test_parity.py` runs without a PyTorch pass. It drops binaries,
Instruments traces, tensors, environments, the ds4 upstream clone, Cloudflare account dumps
and every private-workload file, rewrites the local checkout path, home directory and
account name to placeholders, records before/after hashes per file in `manifest.json`, and
fails if any forbidden string survives. `--list` shows the selection without writing.

Conventions shared by every report: one GPU job at a time on the one tested M5 Max; paired
ABBA quartets inside one resident process for any claimed engine speedup (comparisons with
PyTorch or hosted endpoints state their own, sequential, ordering); exact logit bits required
wherever arithmetic was unchanged; FP32-oracle probability error, not argmax agreement, for any
numerical change; and every accepted request processed in full, with truncation disabled.
