# Prefix cache, checkpoints and the fixed-template entry

Three opt-in mechanisms reuse backbone state for an exact token prefix. None omits input:
the tokens before the reused boundary stay represented by the retained state, and every
token after it is processed. Reuse is exact within one attention mode and one DeltaNet
kernel class, so a hit returns the same logit bits as the uncached pass of the same engine.

- **Exact prefix cache** (`--prefix-cache` on the CLI, `--prefix-cache-mb` plus an
  `X-Clef-Prefix-Cache` key on the server): one entry per key holds attention and
  head-memory K/V once per token and a recurrent-state snapshot before the question schema.
- **Recurrent-state checkpoints**: each entry also keeps up to twelve periodic snapshots
  (every 2,048 tokens, spacing doubled for longer requests) plus one learned at an observed
  divergence, so a changed tail or middle resumes from the last matching checkpoint.
- **Fixed-template entry** (`--template-cache`): reuses the first 32 tokens of Clef's fixed
  system prompt and user header across unrelated requests, within measured tile and
  padding ranges up to 2,048 tokens.

The README's "Repeated, growing and edited contexts" and "Reuse the fixed prompt" sections
give the user-facing numbers. This document records how each mechanism was qualified. The
implementation notes an agent needs are in `CLAUDE.md`.

Keyed entries also support images: exact owned patch bytes, geometry, placement and image order
identify reusable vision features. Those CPU copies and the GPU features count toward the cache
budget. Changed images invalidate the associated backbone prefix. See [vision](vision.md#caches)
for the matching rules, changed-question timing and image-specific regressions.

## Exact FP32 prefix reuse

The candidates under `golden/prefix-fp32-20261005/` adapt the parallel cache implementation
to the qualified root's FP32 attention arithmetic. It does not use compensated tensor
attention, reduce precision or omit any accepted input. Existing weight and activation
formats remain unchanged.

The selected implementation is the clean `integrated32/` build: 32-token snapshot intervals,
a 128-token minimum eligible prefix, engine ownership checks and cached buffer keep-warm.
The earlier `engine/` and `integrated/` builds use 128-token snapshot intervals and remain
frozen as controls. Results below distinguish those builds; a pass on one does not imply a
pass on another.

The cache applies to repeated or growing text. A first request still processes the whole
input, and filling the entry adds allocation and state-storage work. The benchmarks below
report cold requests, initial fills and cache hits separately. This is not evidence of a
cold-request speedup. A small uncached cost remains possible at 2K on 27B; the measurements
below preserve that limitation.

### Original 128-token candidate

`feasibility.json` pins the parallel source snapshot and the qualified root. `build.py`
constructs a separate engine; `manifest.json` records its source and binary hashes.

The ordinary attention functions are copied unchanged from the qualified root. Cached
attention uses a separate function with compact query rows and FP32 K/V heads whose stride
is the cache capacity. It retains the original sequence of 32-key softmax and value updates.
Reused prefixes end on a 128-token boundary, which aligns the original query tiles and
DeltaNet blocks. A 27B request that crosses the 4,096-token DeltaNet algorithm boundary
cannot reuse the old state.

The adaptation also preserves the qualified grouped GEMM dispatch. Cache growth copies
complete FP32 K/V rows. After replacement or shrinkage, the host clears unused rows that a
masked attention tile may read, while preserving the next head's live prefix. The final head
has explicit padding. Failed passes invalidate the usable entry; overflowing FP16
activations use the existing whole-record BF16 fallback.

The main candidate snapshot predates the parallel implementation's cache keep-warm change.
Separate `idle-cache/` and `ownership/` variants provided that change and the
engine-identity guard for the clean builds. The ownership guard passed both its CPU
regression and a populated real-model test. Current CLI/server callers use one engine.

### Validation status

Passed without GPU execution:

- The isolated CLI, server and kernel harness compile without warnings.
- `cpu-record-check.json`: 132 encoded records match qualified production byte for byte:
  both models, all 22 public requests, and parity, strict and no-truncation modes.
- `boundary-fixture.json`: full-input counts of 4,095, 4,096, 4,097, 8,191, 8,192 and 8,193
  tokens are verified by the host encoder. Growing fixtures retain exact token prefixes.

- `smoke-result.json`: ordinary, cached and poisoned Flash outputs match qualified root
  output byte for byte. The repeated and changed-question requests reuse 1,024 tokens; a
  repeated replacement request reuses 1,920 tokens.
- `kernel-result.json`: all 48 preparation layouts preserve Q/K/V/G bits, untouched prefix
  rows and output guards. All 1,080 attention layouts match the full-pass bits, preserve
  poisoned guards and pass the existing sampled float64 bounds.

Both base public checks passed: all 175 logits per model match the saved qualified root in
poisoned packed execution. Both base transition checks also passed all 15 steps and 120
logits, both boundary checks passed 72 logits, and both safety checks passed overflow,
bypass and allocation recovery. Both base heldout checks passed all 480 fill/hit logits per
model. The checks are:

| Check | Evidence required |
|---|---|
| `smoke.py` | Flash fill, hit, changed questions and replacement match root raw output, including poisoned buffers |
| `check_kernels.py` | 48 prep layouts preserve Q/K/V/G bits and prefix/guard memory; 1,080 attention layouts preserve full-pass output bits across cache strides and tile boundaries |
| `qualify.py MODEL public` | All 175 public logits match the saved qualified root in poisoned packed execution |
| `qualify.py MODEL transitions` | Growth, shrinkage, changed middle text/questions and bypass preserve full cold results |
| `qualify.py MODEL boundaries` | Exact full-pass parity around the 4,096-token algorithm and 8,192-token tile boundaries |
| `qualify.py MODEL safety` | Overflow, unsupported activation mode and allocation failure preserve fallback/recovery behavior |
| `qualify.py MODEL heldout` | First visits and repeat hits reproduce all 240 saved logits across 24 fresh contexts |

Saved FP32 references are reused; no oracle regeneration or hosted API calls are required. A
passed compile or encoder test does not qualify inference. Both models and the HTTP behavior
now pass; complete-request performance is still being measured before adoption.

### Cache ownership follow-up

The copied library API has no model-instance check before using an entry. A caller that
reuses one entry across engines could therefore apply model-dependent GPU state to another
model. The current server uses one engine; this is a library integration concern, not an
observed HTTP failure.

`ownership/` adds a monotonic engine identity, binds an entry before its first
model-dependent allocation, and rejects mismatched identities from inference and cache
keep-warm. Engine addresses can be recycled after close, so pointer equality alone is
insufficient. The identity allocator is atomic and refuses to wrap. No inference shader or
arithmetic changes.

`ownership-result.json` records the successful CPU-only regression: distinct engines and
recycled addresses fail explicitly, the failed call leaves the entry's owner and caller
output intact, and touching an empty entry performs no GPU work. It does not open a model or
Metal device. The source is separate from the queued numerical candidate; a final
integration requires a full header-dependent rebuild and runtime verification.

`ownership-model-result.json` now records a passed real-engine check. A populated Flash
entry reused 1,024 of 1,382 tokens with exact logits; the 27B engine and a reopened Flash
engine rejected it explicitly. The original Flash entry remained usable and its keep-warm
call preserved logits. The allocator did not recycle the address in this run; the CPU
regression separately covers that case.

### Reference-order regression

The first Flash transition check failed at the three changed-question records. The fault was
in `qualify.py`: cold-reference deduplication serialized requests with sorted keys and then
sent those reordered requests to the model. Question and option order affect encoded inputs.
The cached and cold arms were therefore receiving different schemas, with a maximum logit
difference of 1.79786.

`failure-reference-order/` preserves the original script, outputs, logs and failed
regression. `check_reference_order.py` executes the actual deduplication function with the
real CPU encoder at the inference boundary. It failed on six of 18 records, including
deliberately reordered questions; after removing key sorting, all 18 records retain their
token IDs, spans and option order. No candidate engine source or binary changed. The
corrected Flash transition check has now passed all 15 steps and 120 logits, including
growth, changed questions and text, short-input bypass, shrinkage and NaN poison
(`clef-flash-transitions.json`).

### Clean integrated build

`integrated/` combines the frozen FP32 cache with the ownership guard, corrected startup
logging and cache-buffer keep-warm. `build_integrated.py` copied no object files: all
translation units were compiled against the final header. Both CLI and server embed the same
FP32 shader as the main candidate. This isolated build did not itself change production;
final source adoption is recorded below.

`integrated-check-manifest.json` pins the resulting sources, objects, binaries, test scripts
and saved public references. `check_integrated.py` was prepared to check host/error behavior
once on Flash, 44 poisoned fill/hit requests against all 350 saved public logits per model,
and each model's HTTP cache isolation, eviction, header validation and ordinary keep-warm
behavior. These focused integration checks supplement the main transition, boundary,
fallback and heldout checks; neither set is a substitute for the other.

The integrated `make test` and `make test-errors` checks have passed. These cover host
processing and head behavior, allocation failures, command-buffer errors and keep-warm
preservation. Flash's 44 poisoned fill/hit requests also preserve all 350 public logits
exactly. The initial integration driver incorrectly compared JSON whitespace against
formatted saved references; `logit_checks.py` revalidated the already-captured outputs as
finite FP32 bits, without rerunning inference. Five comparator regressions cover formatting,
one-ULP differences, signed zero, nonfinite/missing values and option order. The original
driver and output remain preserved; subsequent checks use `check_integrated_v2.py`. The
27B's 44 poisoned fill/hit requests now also preserve all 350 public logits exactly. Both
models' HTTP checks passed response parity, key isolation, unkeyed behavior, eviction and
malformed keys. Ordinary keep-warm also preserves every checked response and removes the
tested five-second idle start delay in both models. `prefix-pair-integrated` is compiled
against this clean build; its timing runner requires both base qualification and the
integration checks before measuring.

### Finer resume-point probe

The FP32 cached attention uses 32 query rows per tile and preserves separate 32-key updates.
The original candidate nevertheless required 128-token snapshots.
`prefix32-attention-checks` tests whether 32-, 64-, 96-token and last-32-aligned offsets
preserve full-pass bits with the unchanged shader. It adds 72 preparation and 1,440
attention layouts, including full FP32 values, output guards and ragged tails. All 72
preparation and 1,440 attention layouts passed (`kernel32-result.json`).

`integrated32/` tests 32-token snapshot intervals while retaining the 128-token minimum
eligible prefix, preserving short-input bypass behavior. The only shader change is its
alignment comment; no kernel arithmetic changed. The library's bounds and snapshot
calculation now allow 32-token alignment. A clean build regenerated the embedded shader and
recompiled every object. Both models' full-model transition checks passed all 15 steps and
120 logits. Several requests reused 32 or 64 more tokens than the 128-token candidate; the
two snapshot intervals have not been timed against each other. Both models also pass the
72-logit boundary check, the 175-logit poisoned public check, all three safety checks and
all 480 fill/hit logits on the 24 fresh contexts. The 128-token candidates remain frozen;
the 32-token source is now integrated in production.

`qualify32.py` reuses a cold reference only after the corresponding main qualification has
completed with the current script and frozen engine. It records the reference hash and
ordered payload hash. It does not regenerate the FP32 oracle or silently compare against
reordered requests.

### Final measurements and permanent regression preparation

`check_server32.py` checks HTTP response bytes, key isolation, unkeyed requests, eviction
and malformed headers on the finer-snapshot candidate. It requires that model's five
qualification lanes to pass first. `run_pair_integrated32.py` then measures the same six
lengths and ABBA schedule described above against the final build, with exact logits
required on every call. Both final HTTP checks passed. Both paired full/hit runs completed;
the 27B table appears below.

| Full tokens | Flash full-pass median | Flash hit median | Initial fill, one observation | Reused tokens |
|---:|---:|---:|---:|---:|
| 346 | 110.18 ms | 109.86 ms (bypass) | 110.19 ms (bypass) | 0 |
| 1,382 | 443.88 ms | 129.70 ms | 410.31 ms | 1,056 |
| 2,235 | 725.47 ms | 113.29 ms | 707.20 ms | 1,952 |
| 4,510 | 1,543.03 ms | 142.85 ms | 1,578.95 ms | 4,192 |
| 8,072 | 3,088.17 ms | 153.82 ms | 3,096.06 ms | 7,808 |
| 16,347 | 7,375.78 ms | 210.16 ms | 7,423.50 ms | 16,064 |

Every one of the 72 engine calls preserved raw logit bits. Both measured quartets favor the
hit path at each eligible length; the short bypass has mixed directions. At 16,347 tokens
the two reductions are 97.09% and 97.18%. The entry uses 1,932,787,712 GPU bytes there. This
is a matching-prefix speedup: all accepted context remains represented by exact retained
state and newly processed rows. Initial fills have one observation each and are not a
measured fill-speed gain. These calls include the CPU head but exclude encoding, model
loading and HTTP. Evidence: `pair-integrated32-clef-flash-result.json` and its 72-sample
JSONL.

`compare_uncached32.py` separately compares qualified production with the finer candidate
while prefix caching is disabled in both. Two resident CLI processes take turns, with two
warm-ups and four measured calls per arm at each of the same six lengths. Every response
must retain finite, exact FP32 logits. The CLI's internal timer includes inference and the
CPU head, while excluding model load, request encoding and response serialization. This
checks uncached processing in warm engines; it is not startup latency. Both GPU comparisons
completed with exact logits; their results and focused follow-ups appear below.

`idle32-control/clef-server` was linked from the exact `integrated32/` objects with only the
server's cache-buffer idle-touch loop removed. Ordinary engine keep-warm remains enabled in
both arms. `compare_idle32.py` uses this matched control to avoid confounding snapshot
geometry with idle behavior. It measures three initial calls (the first fills the entry) and
three hits after five-second gaps at 16,347 full tokens, requiring exact HTTP bytes and
complete processed-plus-reused token counts. Its ordered arms do not control for thermal or
run-order effects. Both model comparisons completed with exact HTTP responses; results
appear below.

`proposed-tests/` prepared the regressions now copied into the source tree; the frozen
engine builds remain intact. The CLI test uses the same 15 transition cases and bounded
fallback/recovery cases as the qualification, with 32-token alignment and the correct
FP32-cache activation bypass. The older copied test still assumes 128-token alignment and
tensor-attention eligibility, so it must not be adopted unchanged. The prepared test
deduplicates cold references while preserving schema order. Nine CPU acceptance checks pass,
and 18 real encoded records remain byte identical through that deduplication. The ownership
and attention tests compile without warnings; the CPU ownership test passes. Their model/GPU
executions still use the separately qualified frozen harnesses until source-tree adoption.

The permanent `test_prefix_public.py` additionally covers all public requests as poisoned
fill/hit pairs. Its standalone main-build Flash run passed all 350 logits; the production
validator checked the same full set on both models.
`proposed-support/integration-support.patch` adds the permanent tests, a model-free
ownership check and the CPU eval checks to `make test`, plus an explicit `make
test-prefix-attention` target. The support patch and the separate nine-file engine patch
both passed `git apply --check` and are now applied. The clean main-worktree build has
completed without warnings.

#### Final 27B paired timing and Flash uncached comparison

`pair-integrated32-clef-result.json` completes the same 72-call protocol on the 27B. Every
output remains bitwise equal to its full-input reference. These are engine plus CPU-head
times in a warm process, excluding encoding, loading and HTTP. Initial fills have only one
sample per length.

| Full tokens | Full median, ms | Hit median, ms | Initial fill, ms | Reused tokens |
|---:|---:|---:|---:|---:|
| 346 | 378.13 | 367.73 | 341.46 | 0 |
| 1,382 | 1,500.27 | 413.22 | 1,519.10 | 1,056 |
| 2,235 | 2,462.47 | 354.92 | 2,446.78 | 1,952 |
| 4,510 | 5,236.46 | 452.37 | 5,299.83 | 4,192 |
| 8,072 | 10,139.31 | 465.99 | 10,290.31 | 7,808 |
| 16,347 | 23,829.64 | 649.64 | 23,906.03 | 16,064 |

The 346-token case bypasses caching, so its timing difference is not a cache benefit. Both
measured ABBA quartets favor hits at every eligible length. The 16,347-token entry retains
3,111,780,352 GPU bytes on 27B, versus 1,932,787,712 on Flash; these are retained cache
allocations in addition to the engine's buffers.

`uncached32-clef-flash.json` compares the qualified production CLI with the candidate, with
caching disabled in both. All 72 responses have exact logits and complete token counts. The
16,347-token medians are 7,453.9 and 7,442.6 ms. At 8,072 tokens the candidate is 1.16%
slower (3,129.7 versus 3,166.0 ms), with both quartets slower by 0.54–1.23%. The apparent
8.84% improvement at 1,382 tokens is not stable across quartets (1.06% and 9.89%), so it is
not evidence of a cold-input speedup. Other median differences are below 0.5%. These
measurements do not establish a cold-input improvement or statistical equivalence.

`validate_root.py` completed all eight lanes against the rebuilt source tree. Its frozen
production manifest covers the host/error checks, exact attention and ownership tests, saved
FP32 parity, poisoned packed and cached public requests, cache transitions/recovery, and
HTTP. It does not regenerate oracles. Source adoption, the clean build and the production
validation sequence are complete.

The final Flash idle comparison is complete in `idle32-clef-flash.json`. Every HTTP response
is byte-identical to the qualified full-context reference. With ordinary engine keep-warm
enabled in both arms, adding cache-buffer touches changes the median after five-second gaps
from 223.80 to 211.71 ms. The median GPU wait minus execution interval changes from 22.57 to
1.08 ms. These are three post-gap samples per arm in one control-then-candidate run; thermal
and order effects remain confounded. The 27B comparison also retains exact response bytes:
post-gap HTTP medians change from 604.77 to 550.98 ms, and median GPU wait minus execution
from 35.14 to 1.18 ms, with the same sampling and ordering limitation.

The first 27B uncached comparison stopped before loading a model: its broad
`uncached32-clef-*.log` overwrite guard matched completed Flash logs. The idle runner had
the same filename ambiguity. Both now check their exact two arm filenames.
`failure-model-name-prefix/` preserves the original scripts, observed false matches and
eight passing guard checks. No completed result was overwritten; Flash reports retain the
hashes of the scripts actually used. The corrected 27B comparison is complete: all 72
responses retain exact logits and full token counts.

Because both primary Flash 8K quartets were slightly slower, `compare_uncached8k.py`
repeated that fixture with four warm calls and four measured ABBA quartets. Exact logits
were retained. Production and candidate medians were 3,200.10 and 3,193.75 ms; quartet
reductions were -1.06%, +1.19%, +1.37% and +0.98%. The original slowdown did not repeat.
These small differences do not establish an uncached speedup.

The complete 27B uncached comparison reports:

| Full tokens | Production median, ms | Candidate median, ms | Candidate time change |
|---:|---:|---:|---:|
| 346 | 392.85 | 396.00 | +0.80% |
| 1,382 | 1,742.45 | 1,746.35 | +0.22% |
| 2,235 | 2,645.25 | 2,675.60 | +1.15% |
| 4,510 | 5,637.30 | 5,653.10 | +0.28% |
| 8,072 | 10,748.70 | 10,763.70 | +0.14% |
| 16,347 | 24,511.65 | 24,569.90 | +0.24% |

The two 2,235-token quartets were both slower, by 0.57% and 1.15%. The focused four-quartet
repeat in `compare_uncached2235.py` reports 2,582.95 versus 2,592.30 ms, a 0.36% slower
candidate median. Quartet reductions were -2.64%, +1.19%, -1.04% and -1.10%. A small
uncached cost is therefore not ruled out; these results are not described as a cold speedup
or proof of unchanged latency. The primary 8K and 16K quartets changed direction.

The opt-in cache was integrated for its large, repeatable matching-prefix gains and exact
outputs, with the uncached timing limitation retained in the README.
`root-integration-applied.json` pins the evidence behind that decision. The independent
exact-alpha experiment was rejected on inconsistent kernel timings and is not part of this
integration.

### Main-worktree verification

The rebuilt prefix-cache checkpoint completed all eight validation lanes. All 97 frozen
artifacts matched `root-integration-manifest.json` before the subsequent fixed-template
extension was applied. The final 27B HTTP lane passed byte parity, key isolation,
malformed-key rejection and budget eviction, as the earlier isolated candidate did on both
models.

| Rebuilt-main check | Flash | 27B |
|---|---|---|
| Host, encoder, tokenizer and error suites | Pass; shared code | Same shared code |
| Exact attention layouts and populated cross-model ownership | Pass; shared checks | Covered by ownership check |
| Saved FP32 parity, poisoned packed and cached public outputs | Pass | Pass |
| Cache transitions, overflow, unsupported-mode bypass and allocation recovery | Pass | Pass |
| HTTP bytes, key isolation, malformed keys and budget eviction | Pass | Pass |

Both models retain all 46 public oracle decisions. Relative to the saved qualified engine,
each model's packed outputs preserve 175 raw FP32 logits and the poisoned fill/hit sequence
preserves 350. Each transition sequence preserves 120 logits. The permanent Flash public
wrapper also passed independently. These are output-preservation checks, not evidence of
improved labeled accuracy.

The ordinary-recurrence diagnostic in `golden/gdn-cold-20261005/` is complete. At the 27B
2,235-token shape it preserves exact bits, poisoned guards and sampled float64 bounds.
Old/current GPU medians are 2.471857/2.473500 ms per dispatch; the four paired quartets
favor the old kernel by 0.09–1.10%. This synthetic result does not explain the whole-request
cold cost or justify duplicating the original kernel. No change was adopted from it.

## Recurrent-state checkpoints

Periodic recurrent-state checkpoints make exact prefix reuse useful when a long request
changes near its end or in its middle. The qualified candidate has been integrated in main
and rebuilt without warnings. Final verification passed on both models: ten checks on Flash
and nine on 27B. It does not accelerate an unrelated fresh input.

The previous cache kept one recurrent-state snapshot just before the question schema. A
change before that snapshot forced a full recomputation. The new implementation keeps
periodic snapshots and learns a snapshot just before an observed divergence, so it can
resume from matching tokens before the edit. Attention and head memory still include the
full context. Reuse requires exact token identity and the same DeltaNet kernel class.

### Completed Flash comparison

The parallel implementation was compared with a build whose source files match qualified
main. Both used `--prefix-cache --batch 1 --strict --no-truncate`. Each sequence ran in a
separate process after 45 seconds idle, in the order baseline, candidate, candidate,
baseline. There are two samples per arm per request. Timings cover the forward pass, CPU
head and cache operations, excluding model loading, encoding, HTTP and the idle period.

| Scenario | Full tokens | Baseline | Candidate | Reduction in mean latency |
|---|---:|---:|---:|---:|
| First changed tail | 15,068 | 5.20–5.82 s | 339–351 ms | 93.7% |
| Later changed tails | 15,068–15,140 | 5.84–6.36 s | 284–347 ms | 94.5–95.2% per request |
| First middle edit | 16,351 | 6.45–6.54 s | 4.32–4.33 s | 33.3% |
| Later middle edit / restored input | 16,347–16,351 | 6.66–6.78 s | 3.67–3.74 s | 44.8–45.2% per request |

The two 16K first-fill controls were essentially unchanged: approximately 5.7–5.8 seconds.
Existing same-state hits were about 150–175 ms in both arms. Two samples per arm cannot
establish a small improvement or a tight overhead bound; the large edit/tail improvements
separated from every baseline sample.

The timing runner required identical raw logits across all four arms. It did not retain
those raw logits, so that equality is enforced by the pinned runner rather than
independently re-scored by this audit. The independent CPU audit reconstructed all twelve
public requests and verified every full encoded token count against the reported counts in
every arm. No request was truncated. These are synthetic shared-prefix workloads, not
fresh-input or Cloudflare API comparisons.

### Memory and qualification

The completed 27B comparison used the same requests, order and idle interval:

| Scenario | Full tokens | Baseline | Candidate | Reduction in mean latency |
|---|---:|---:|---:|---:|
| First changed tail | 15,068 | 19.81–21.14 s | 1.11–1.21 s | 94.3% |
| Later changed tails | 15,068–15,140 | 20.00–23.62 s | 0.94–1.09 s | 95.4–95.7% per request |
| First middle edit | 16,351 | 21.77–21.97 s | 14.53–14.56 s | 33.5% |
| Later middle edit / restored input | 16,347–16,351 | 22.09–23.73 s | 12.26–12.48 s | 45.7–46.8% per request |

The first-fill samples overlap. Existing-hit means were 2.5% slower for one request and 5.5%
faster for the other, also with overlapping samples. Thus the large edit/tail gains are
supported; a small existing-hit cost remains possible. The CPU audit verifies every full
token count and the unchanged build hashes for 27B too. Its timing runner likewise required
identical logits across all arms.

The final tail-sequence entry grew from about 1,815 MB to 2,237 MB, a 422 MB increase. The
unchanged-state sequence used about 369 MB more. These are retained entry sizes, in decimal
MB, and exclude ordinary engine buffers. The existing server budget is a post-request
retained-cache budget, not a peak allocation cap. On 27B, tail entries grew from 2,927 MB to
4,182 MB and middle-edit entries from 3,112 MB to 4,367 MB. A 4 GiB budget cannot retain
that measured middle-edit entry; it still serves the full request, then evicts the entry. A
fixed budget may now hold fewer long entries, so cache-hit rate remains a workload-dependent
tradeoff.

The default-attention gate passed on both Flash and 27B, including checkpoint transitions,
poisoned buffers, overflow and allocation recovery. Source and binary hashes match the
captured candidate. An independent CPU test of the actual planner passed 6,016 requests
under AddressSanitizer and UndefinedBehaviorSanitizer, including failure injection and a
corrupt-token negative control.

The original checkpoint runner discarded inherited `CLEF_*` settings. Thus its default-mode
gate did not establish checkpoint behavior with `CLEF_ATTN_TU=0`. An isolated adaptation now
requires an explicit attention mode; CPU checks confirm that it selects FP32 even when the
inherited setting requests tensor-unit attention. Permanent planner and mode-selection
regressions are installed in main. The mode-selector checks fail with the original runner
and pass with the fix. The reviewed implementation, regressions and updated transition
expectation are installed. Final integration checks passed on both rebuilt models against
frozen fresh outputs and the qualified checkpoint binary, including both attention modes,
template and server paths, failure recovery and budget eviction. Flash also passed `make
test`. The final audit verified all 214 pinned artifacts and both models' check logs. No
extra task-quality inference is justified solely by the completed bit-exact cache tests.

## Fixed-template entry

This opt-in feature reuses the first 32 tokens of the fixed system prompt and user header
across unrelated requests. Every byte of the new request is still encoded and processed. The
main checkout now restricts reuse to measured beneficial tile/padding ranges, with an upper
limit of 2,048 tokens. Other lengths run normally and preserve the entry for a later
eligible request. The final rebuild has no compiler warnings and both models pass
verification: 110 exact poisoned logits over 22 requests per model, overflow, HTTP isolation
and allocation recovery. Model-free ownership checks cover eligible and bypassed lengths,
invalid owners and reopened engines. This feature does not accelerate the first fill or
fresh long contexts.

### Current eligibility and final measurements

For a full encoded request of `T` tokens with at least 32 tokens before its schema, apply
these rules in order:

| Condition | Action |
|---|---|
| `T > 2048` | Run the ordinary full-input path. |
| Flash, `768 <= T <= 1024`, and `T % 64 == 0` or `T % 64 > 32` | Run the ordinary path; reuse would switch from a faster 64-row GEMM tile to a 32-row tile. |
| Either model, `T > 1056`, and `T % 64 == 0` or `T % 64 > 32` | Run the ordinary path; removing 32 tokens saves no padded GEMM rows. |
| Other eligible lengths | Fill or reuse the fixed 32-token prefix. |

The existing span, prefix-ID, engine-owner and backend-mode checks still apply. The host
selector changes no arithmetic. Invalid owners are rejected before any token or GPU-buffer
access, including when the length would bypass reuse. The permanent test enumerates outcomes
on both sides of these boundaries, checks reuse after bypasses, and uses an eligible
1,025-token request to exercise allocation failure above 1,024 rows and subsequent recovery.
The pre-policy binary fails that new reuse assertion, as expected.

The final compensated-attention comparison uses two unrelated inputs per length, independent
full-pass references, four warm-ups and four ABBA quartets. It records 420 timed calls at 21
lengths on each model, with exact raw logits. A follow-up covers ten padding boundaries with
another 200 timed calls per model, also exact. Times include inference and the CPU head,
excluding encoding, model load and HTTP. Selected follow-up results:

| Full tokens | Flash ordinary → template, ms | 27B ordinary → template, ms | Current eligibility |
|---:|---:|---:|---|
| 1,056 | 326.408 → 314.245 | 1,100.042 → 1,032.554 | reuse |
| 1,088 | 339.837 → 340.486 | 1,106.298 → 1,103.516 | bypass |
| 1,089 | 344.921 → 330.355 | 1,149.515 → 1,113.813 | reuse |
| 1,120 | 343.705 → 332.724 | 1,150.217 → 1,110.373 | reuse |
| 1,121 | 348.475 → 347.036 | 1,152.238 → 1,148.870 | bypass |
| 1,184 | 382.740 → 363.808 | 1,229.467 → 1,164.177 | reuse |
| 1,312 | 404.712 → 383.983 | 1,373.364 → 1,297.077 | reuse |
| 1,568 | 491.335 → 466.559 | 1,634.807 → 1,572.401 | reuse |
| 2,016 | 617.660 → 601.359 | 2,117.840 → 2,054.252 | reuse |
| 2,048 | 621.218 → 618.048 | 2,124.499 → 2,123.945 | bypass |

Every eligible row improves in all four quartets on both models, with median reductions of
2.6–5.1% on Flash and 3.0–6.1% on 27B. The bypassed rows are flat or mixed. At 346 and 594
tokens, the 21-length comparison improves by 4.3–5.3% on Flash and 4.6–4.9% on 27B. Some
Flash lengths near 1K instead regress by 5–6%. An isolated attempt to keep the full-pass
tile during template hits preserves exact logits but does not recover the loss; it was
rejected.

Artifacts are under `golden/attention-main-review-20261005/`: `timing/template-*`,
`template-alignment/`, `template-geometry/`, and `template-policy/` record the measurements,
rejected experiment, source backup, expected regression failure and final selector.
`root-manifest.json` freezes the rebuilt code and test inputs;
`root-final-qualification.json` verifies the completed reports and their logs. The
historical measurements below explain how the policy developed; their older unrestricted
2,048-token eligibility is superseded.

### Source and correctness

`golden/template-review-20261005/capture.json` freezes the parallel `main-tpl` prototype and
its completed `batch35` results. Only `clef.c`, `clef_engine.h`, `clef_main.c` and
`clef_server.c` differ from the main FP32 build. The shader, Metal dispatch and head
implementation match. The core factors the existing prefix path into a helper that accepts
the snapshot row; ordinary prefix caching retains its 128-token eligibility threshold.

The CPU encoder check covers 162 records: both models, three encoding modes, the public
corpus and changed/empty/control-token/Unicode/structured states. Each preserves the same
36-token template and keeps all head spans after the 32-token snapshot. The inference entry
still compares the actual prefix IDs before reuse and binds itself to one engine instance.

Only fixed public state is reused. The entry also retains request-dependent suffix K/V and
head-memory rows as scratch space; the next pass overwrites its live suffix and clears
masked attention padding before reading. The prototype's claim that the entry contains no
request data was inaccurate. The isolated reviewed candidate corrects that comment, logs
initial allocation fallback, documents the CLI option and rejects incompatible CLI cache
flags explicitly. It does not change GPU arithmetic.

### Prototype and reviewed-build checks

The captured parallel run passes ordinary/cached/poisoned/forced-overflow checks on all 22
public requests for both models, eight HTTP response comparisons per model, and the main
prefix transition/recovery test on Flash. These results apply to the captured prototype. The
reviewed build passes on both models: 11 unrelated/replaced/shrunk requests preserve 55 raw
logits per model with poisoned buffers, including exact lengths around 1,024 and 2,048.
Forced overflow, HTTP bytes, keyed/unkeyed interleaving and allocation fallback/recovery
also pass. Each model's complete targeted check also passes on HTTP bytes, key isolation,
allocation failure and recovery.

The first Flash run reached the final log assertion with every response comparison passing,
then failed because the harness expected an error without the propagated `metal:` prefix.
`failure-metal-prefix/` preserves that run and its exact test. The corrected acceptance
passes six CPU cases, including duplicate/missing errors, and the complete corrected Flash
run passes. No engine code changed for this fix.

### Earlier FP32 integration

`adopt.py` verified all eight completed main prefix-cache lanes, both reviewed template
checks, their source/model identities, and the patch before applying it.

The new main build matches all qualified inference source files and embeds the same FP32
shader verbatim in both binaries. `root-manifest.json` freezes 69 main source, test,
documentation and binary artifacts. `qualify_root.py` completed the permanent template
checks on each rebuilt model against the backed-up ordinary CLI; the Flash lane also passed
`make test` and the new CLI flag regression.

Each timing below is a range of two arm medians from one plain/template/template/ plain
sequence. An arm runs eight calls and reports the last six, with a 30-second cooldown before
the process starts. Times cover the CLI's inference and head timer, not HTTP, encoding or
model load. Repeated timing inputs do not measure varying-content latency; the separate
equality tests cover changed inputs.

| Model | Full tokens | Plain arm medians, ms | Template arm medians, ms |
|---|---:|---:|---:|
| Flash | 346 | 109.4–110.0 | 103.3–103.5 |
| Flash | 594 | 174.8–174.9 | 166.6–166.7 |
| 27B | 346 | 333.6–335.5 | 316.4–316.5 |
| 27B | 594 | 565.6–566.2 | 541.2–541.4 |

This is a roughly 4–6% short-request lead, not a statistical bound or a first-fill speedup.

### Varied-input timing and cutoff review

`pair-clef-flash.json` uses two unrelated inputs at each length, each checked against its
own independently computed ordinary raw logits. After four warm-up calls, four ABBA quartets
run ordinary A, template A, template B, ordinary B in one resident engine. Every call
retains the full input. Times include inference and the CPU head, excluding encoding, model
load and HTTP.

| Flash full tokens | Ordinary median, ms | Template median, ms | Time reduction |
|---:|---:|---:|---:|
| 346 | 109.049 | 102.814 | 5.72% |
| 594 | 174.185 | 171.954 | 1.28% |
| 1,382 | 411.057 | 426.146 | −3.67% |
| 2,048 | 692.400 | 693.216 | −0.12% |

The corresponding 27B run also preserves every raw logit:

| 27B full tokens | Ordinary median, ms | Template median, ms | Time reduction |
|---:|---:|---:|---:|
| 346 | 367.810 | 346.445 | 5.81% |
| 594 | 641.857 | 619.553 | 3.47% |
| 1,382 | 1,437.762 | 1,456.364 | −1.29% |
| 2,048 | 2,145.136 | 2,146.964 | −0.09% |

The four within-quartet reductions at 1,382 tokens are −6.41%, −4.95%, −4.85% and +0.81%;
this is a reason to restrict eligibility, despite exact outputs. At 594 tokens the quartets
improve by 0.99–5.55%; temporal drift makes the aggregate median difference less
representative there. On 27B all four quartets regress at 1,382 tokens, by 0.02–3.95%. The
original unrestricted 2,048-token eligibility therefore needed revision.

The completed `boundary-1024/` comparison uses the same protocol:

| Full tokens | Flash ordinary → template, ms | 27B ordinary → template, ms |
|---:|---:|---:|
| 1,023 | 370.124 → 352.515 | 1,075.876 → 1,050.088 |
| 1,024 | 320.923 → 357.517 | 1,079.546 → 1,046.117 |
| 1,025 | 335.805 → 369.634 | 1,127.734 → 1,076.035 |
| 1,152 | 345.671 → 343.961 | 1,204.178 → 1,198.251 |

Flash regresses in all four quartets at both 1,024 and 1,025, while improving in every
quartet at 1,023. Its GEMM dispatch switches tile shape at 1,024 **processed** rows; reusing
32 rows crosses back below that threshold. 27B's short-matrix tile differs, and it does not
show the same regression. This identifies a concrete dispatch difference, not yet its
contribution to total latency. An isolated whole-engine comparison under
`golden/gemm-boundary-20261005/` subsequently qualified a restricted rule for fresh requests
between 768 and 1,024, with unchanged-dispatch controls. That rule is now in main; see [the
GEMM report](#gemm-dispatch-near-1k-tokens). The later compensated-attention comparisons
above determine the final template eligibility without another backend geometry change.

### Memory and serving

The option defaults off. The CLI uses a dedicated entry and requires batch one with no
residual dump. The server uses one worker-owned entry for unkeyed single requests; keyed
requests continue through their own entries, and packed requests use the ordinary batch
path. The template entry is additional to the keyed cache budget.

`memory.json` calculates retained GPU allocations from the pinned model configs and
`prefix_reserve` formulas:

| Model | Capacity 1,024 rows | Capacity 2,048 rows |
|---|---:|---:|
| Flash | 163.25 MiB | 275.25 MiB |
| 27B | 327.625 MiB | 503.625 MiB |

These exclude ordinary engine buffers, allocator overhead and temporary overlap during
capacity growth. Reusing 32 tokens does not mean allocating only 32 rows: the same buffers
also hold the current request's suffix.

## GEMM dispatch near 1K tokens

Template timing exposed a sharp Flash regression at 1,024 and 1,025 full tokens. The GEMM
dispatcher uses 64-row tiles from 1,024 processed rows; a template hit removes 32 rows and
crosses back below that threshold. This is a verified path difference, not a proven
explanation of the whole regression.

The isolated experiment in `golden/gemm-boundary-20261005/` compares thresholds of 1,024 and
768 on **fresh, uncached** inputs. It changes only host dispatch to existing qualified
kernels. Weights, arithmetic, attention and context limits are unchanged. The selected Flash
rule described below has now been applied to main after both isolated model qualifications
passed. Rebuilt-main checks also pass on both models.

### First whole-engine comparison

Each model stays resident. Two unrelated inputs share each exact length. Four warm-up calls
precede four ABBA quartets: original A, candidate A, candidate B, original B. Every result
is compared with its own independently computed original raw logits. All 100 calls per model
match exactly. Times include inference and the CPU head, excluding encoding, model load and
HTTP.

| Full tokens | Flash original → candidate, ms | 27B original → candidate, ms |
|---:|---:|---:|
| 594, unchanged dispatch | 195.485 → 194.154 | 666.637 → 678.298 |
| 768 | 266.257 → 258.599 | 834.566 → 830.762 |
| 992 | 330.833 → 328.767 | 1,075.489 → 1,091.139 |
| 1,023 | 349.466 → 335.802 | 1,085.227 → 1,091.446 |
| 1,024, unchanged dispatch | 328.411 → 325.878 | 1,073.948 → 1,074.052 |

Flash improves in all four quartets at 768 tokens (0.30–4.73%) and at 1,023 (1.94–16.49%,
with one unusually large difference). At 992 the reductions are 0.22%, 0.77%, 1.64% and
−0.06%. The unchanged-dispatch controls are noisy: their median reductions are 0.68% and
0.77%, with mixed signs within quartets. This supports further checking of a modest Flash
gain, not a general speedup estimate.

27B does not show a useful consistent benefit. Its 992-token case regresses in three
quartets, and the other changed cases are mixed. Its short matrices use 32×256 tiles where
Flash commonly uses 32×128, so a shared threshold is not justified. The candidate is not
selected for 27B.

### Completed focused check and selected rule

`repeat-flash/` uses the same frozen binary and paired protocol at 767, 768, 800, 832, 896,
960, 992, 1,023 and 1,024 tokens. The first and last lengths leave dispatch unchanged. This
covers the affected range before selecting a production threshold. All 180 calls preserve
exact raw logits. Results:

| Full tokens | Original → candidate, ms | Median reduction |
|---:|---:|---:|
| 767, unchanged dispatch | 280.722 → 272.873 | 2.80% |
| 768 | 272.384 → 260.896 | 4.22% |
| 800 | 278.818 → 277.411 | 0.50% |
| 832 | 293.181 → 280.055 | 4.48% |
| 896 | 312.775 → 303.013 | 3.12% |
| 960 | 337.375 → 324.595 | 3.79% |
| 992 | 347.097 → 345.868 | 0.35% |
| 1,023 | 351.742 → 346.964 | 1.36% |
| 1,024, unchanged dispatch | 337.822 → 336.750 | 0.32% |

All four quartets improve at 768, 832, 896 and 960. The unchanged-dispatch 767-token control
also has a 2.80% median difference, so aggregate medians alone are insufficient evidence.
Its quartet reductions are 0.32%, 2.92%, 1.21% and −0.08%; the four 768-token reductions are
3.69–4.52%. The small differences at 800 and 992 remain mixed across quartets.

At 800 and 992, 64-row tiles cover 32 more padded rows than 32-row tiles. The next isolated
candidate therefore limits the earlier switch to Flash shapes whose padded row counts match:
`T >= 768`, `T < 1024`, and `T % 64` is zero or greater than 32. The existing selection from
1,024 rows remains intact, as does 27B's dispatch. This rule is an experimental selection
from the measurements, not an established explanation of the separate template-cache
regression.

The completed `selected/` comparison includes new partial-tile lengths 801 and 865. All 200
calls preserve exact raw logits:

| Full tokens | Original → selected, ms | Median reduction |
|---:|---:|---:|
| 767, unchanged dispatch | 262.536 → 258.770 | 1.43% |
| 768 | 259.882 → 253.496 | 2.46% |
| 800, unchanged dispatch | 281.874 → 281.438 | 0.15% |
| 801 | 277.580 → 266.793 | 3.89% |
| 832 | 285.670 → 268.172 | 6.13% |
| 865 | 306.507 → 292.629 | 4.53% |
| 960 | 329.340 → 315.226 | 4.29% |
| 992, unchanged dispatch | 352.425 → 352.646 | −0.06% |
| 1,023 | 358.082 → 333.569 | 6.85% |
| 1,024, unchanged dispatch | 318.941 → 323.915 | −1.56% |

All four quartets improve at 801, 832, 865 and 1,023. The 768 and 960 cases each have one
slower quartet. Together with the earlier runs, this supports qualifying the restricted
rule; it does not establish a uniform gain at every affected length. The controls still show
timing noise.

`selected/production/` contains a separately built CLI with the rule enabled and the
experimental setter removed. `selected/decision.json` selects it for exact CLI
qualification, with main adoption still false. `selected/qualify.py` checks ordinary,
packed, poisoned-buffer, prefix, template and forced-overflow paths against the frozen
original binary. Flash passes all nine checks: 100 exact logits per
ordinary/packed/template/overflow arm, plus 50 per repeated-prefix arm. 27B passes the same
nine checks. No gain above 1,024 is claimed, and the separate template-cache performance
cutoff remains unresolved.

`selected/integration.patch` prepares the dispatcher change and a permanent
`test_gemm_dispatch.py` regression, covering both sides of the partial-tile boundaries and
forced fallback. That patch is now applied to main, with a clean rebuild and no shader,
weight, context-handling or attention-default change. The permanent boundary regression
passes on both rebuilt models: 60 exact logits in each of the single, packed and template
arms, and 15 in forced BF16 fallback. `make test` also passes. `selected/root-manifest.json`
freezes the resulting 70 main artifacts. The FP32 attention default remains unchanged.

### Template interaction after the dispatch change

The completed `selected/template-check/` run covers 16 lengths and preserves exact raw
logits on all 320 calls. Selected FP32-path observations:

| Full tokens | Ordinary → template, ms | Median reduction |
|---:|---:|---:|
| 346 | 129.578 → 118.757 | 8.35% |
| 767 | 288.462 → 274.154 | 4.96% |
| 768 | 263.974 → 278.136 | −5.36% |
| 992 | 348.768 → 324.336 | 7.01% |
| 1,024 | 354.965 → 352.398 | 0.72% |
| 2,048 | 673.873 → 695.422 | −3.20% |

The 346-, 767- and 992-token cases improve in all four quartets. At 768 and 2,048, three
quartets regress. The 1,024-token median is slightly positive but three quartets regress;
both ordinary and cached GEMM choices at that length are unchanged by the selected rule, so
its changed timing relative to the earlier run cannot be credited to this dispatch change.
The full artifact retains every quartet and sample. These results reinforce the need to
revise template eligibility, and do not support claiming a general gain through 2K. The
newer compensated-attention integration subsequently completed both models at 21 template
lengths plus ten padding boundaries, with exact outputs on every call. Preserving the
ordinary GEMM tile in a separate experiment did not recover the Flash loss, so that change
was rejected. Main now bypasses those measured Flash tile cliffs and, above 1,056 tokens,
reuses the template only when removing 32 tokens saves a padded 64-row tile. See [the
template policy](#fixed-template-entry). The selected fresh-input GEMM rule remains
unchanged.

## Artifacts

Local, git-ignored evidence directories under `golden/`:

- `prefix-fp32-20261005/`: the FP32 cache candidates (`engine/`, `integrated/`,
  `integrated32/`), ownership and reference-order regressions, qualification reports, paired
  full/hit and uncached timing samples, idle comparisons and the integration manifest.
- `gdn-cold-20261005/`: the ordinary-recurrence diagnostic.
- `prefix-checkpoint-review-20261005/`: checkpoint source capture, planner audit, both
  attention-mode gates, timing audits and the adoption record.
- `template-review-20261005/`: template prototype capture, reviewed candidate and root
  qualification.
- `attention-main-review-20261005/timing/template-*`, `template-alignment/`,
  `template-geometry/`, `template-policy/`: the compensated-mode template measurements and
  the final eligibility selector.
- `gemm-boundary-20261005/`: the Flash 1K boundary experiment, repeats, selected rule and
  its integration record.
