# Reports

Measurement and qualification records behind the numbers in the top-level README and the
invariants in `CLAUDE.md`. Each file covers one topic; add new evidence to the matching report
rather than starting a dated file. Raw artifacts (sources, binaries, samples, manifests) are
local and git-ignored under `golden/<experiment>-<date>/`, named in each report's artifacts
section.

| Report | Contents |
|---|---|
| [performance-history.md](performance-history.md) | Every retained change since `db38cfc`, in order, with its paired measurements; current status; GPU profiles and accelerator counters; the batched classification workload |
| [attention.md](attention.md) | Kernel selection; the FP32 path (shared probabilities, 64-key prefetch, direct rescaling); the compensated tensor-unit precision screen, the ContractNLI labeled evaluation and the adoption measurements |
| [prefix-cache.md](prefix-cache.md) | Exact FP32 prefix reuse, recurrent-state checkpoints, the fixed-template entry and the Flash GEMM rule near 1K tokens |
| [hosted-comparison.md](hosted-comparison.md) | Workers AI measurements on the checkout fixtures and the public corpus, the truncation observations, and the local replay against saved hosted responses |
| [long-request-timing.md](long-request-timing.md) | Why a 13.9K-token Flash request sometimes ran slower with the default attention: stage timing, GPU clocks, limiter and shader traces, sustained thermal behavior |
| [rejected-experiments.md](rejected-experiments.md) | Attention, GEMM, FFN, DeltaNet and workload-level changes that were measured and not retained, with the criterion each failed |

**Evidence archive.** `tools/evidence_archive.py OUT.tar.gz` packs the checkable part of
`golden/` for publication as a release asset (`cleffa-evidence-<date>.tar.gz`): result,
manifest, sample and log files, experiment sources, saved logits, and the small files of the
oracle directories so `tests/test_parity.py` runs without a PyTorch pass. It drops binaries,
Instruments traces, tensors, environments, the ds4 upstream clone, Cloudflare account dumps
and every private-workload file, rewrites the local checkout path, home directory and
account name to placeholders, records before/after hashes per file in `manifest.json`, and
fails if any forbidden string survives. `--list` shows the selection without writing.

Conventions shared by every report: one GPU job at a time on the one tested M5 Max; paired
ABBA quartets inside one resident process for any speed claim; exact logit bits required
wherever arithmetic was unchanged; FP32-oracle probability error, not argmax agreement, for any
numerical change; and every accepted request processed in full, with truncation disabled.
