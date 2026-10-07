# Hosted comparison

Measurements against Cloudflare's hosted `clef-flash` and `clef` endpoints on Workers AI,
using the public checkout fixtures and the public 22-request corpus. A hosted response is
evidence of that deployment's behavior on the capture date; Cloudflare does not return a
weight revision, and reported token counts do not prove token identity. Agreement is not
labeled accuracy, and a hosted time that covers fewer tokens is not an equal-work latency.
The comparison against Jev's hosted API is in the README.

Local timings here are localhost HTTP, including encoding and the CPU head. Hosted timings
include the network from the measuring machine. The two were collected in sequential runs,
not interleaved, so they are absolute observations rather than paired comparisons.

## Checkout fixtures over Workers AI, 2026-10-04

`bench/cloudflare_checkout.py` uses the same public fixtures as `bench/checkout_latency.py`.
Its default mode writes a request plan without network access. Adding `--run` sends those
fixtures to the fixed Cloudflare Workers AI endpoint, with the token supplied through an
environment variable. Three default passes plus one warm-up per model make 56 calls.

The journal includes each exact payload and its SHA-256, the local strict-encoding token
count, every response and reported token count, latency, and whether the reported count
matches. Missing usage is unknown, and a smaller count is not treated as equivalent work.
The script preserves answers so outage-at-start and outage-at-end results can be assessed
alongside latency. A matching count alone is not proof that the provider processed all text.
Failures are recorded before collection stops; previous output files cannot be overwritten.

After Wrangler login was refreshed, the selected account completed all 56 calls: one warm-up
and three shuffled passes over nine fixtures for each model. All returned HTTP 200 and
deterministic answers. A fresh local run used the same payload bytes, verified by all 18
SHA-256 pairs, with three short warm-ups per model, six measured blog calls and three
shuffled passes over the padded fixtures. Only one local model ran at a time. Both use
keep-alive connections; hosted measurements include network and service time. These are
sequential runs, not an interleaved optimization comparison, and three hosted samples give
noisy medians.

| Fixture | Full tokens | Hosted tokens | Hosted Flash | Local Flash | Hosted 27B | Local 27B |
|---|---:|---:|---:|---:|---:|---:|
| Blog example | 346 | 346 | 310 ms | 113 ms | 547 ms | 346 ms |
| 128 B, outage at end | 356 | 356 | 326 ms | 126 ms | 553 ms | 472 ms |
| 512 B, outage at end | 404 | 404 | 353 ms | 132 ms | 565 ms | 516 ms |
| 2 KiB, outage at end | 600 | 600 | 479 ms | 191 ms | 470 ms | 748 ms |
| 8 KiB, outage at end | 1,382 | 1,382 | 395 ms | 416 ms | 756 ms | 1,717 ms |
| 16 KiB, outage at end | 2,424 | 2,382 | 633 ms | 768 ms | 899 ms | 3,042 ms |
| 32 KiB, outage at end | 4,510 | 2,382 | 542 ms | 1,443 ms | 577 ms | 5,980 ms |
| 16 KiB, outage at start | 2,424 | 2,382 | 582 ms | 742 ms | 673 ms | 3,007 ms |
| 32 KiB, outage at start | 4,510 | 2,382 | 699 ms | 1,417 ms | 660 ms | 5,972 ms |

The last four rows are **not equivalent-work latency comparisons**. Both hosted models
report 2,382 tokens for both lengths and both outage positions. For an outage at the end,
Flash returns urgency 0.0180 and severity 0.0615/3; 27B returns 0.0098 and 0.2196/3. Each
model's entire answer object is identical between 16 and 32 KiB. Moving the outage to the
beginning raises hosted urgency to 0.9396 (Flash) and 0.9851 (27B). Local inference consumes
all tokens, with trailing-outage urgency 0.9592/0.9612 for Flash and 0.9856/0.9866 for 27B.
This is strong evidence of prefix truncation on these hosted fixtures; it does not identify
the internal limit or establish a general model-quality ranking. Cloudflare's [model
documentation](https://developers.cloudflare.com/workers-ai/models/clef/) says text can be
truncated, but does not document this observed 2,382-token total or a request parameter to
raise the text limit.

At the largest fixture with matching counts, local Flash's median is 5.5% higher, while
local 27B's is 2.27 times the hosted median. Hosted 27B samples span 450–1,083 ms there, so
neither ratio is a precise service-performance estimate. The shorter 346–404-token fixtures
are faster locally in this run.

The free-only run was bounded before collection. Cloudflare's
[pricing](https://developers.cloudflare.com/workers-ai/platform/pricing/) provides 10,000
free neurons per day on both Free and Paid plans. The plan estimated 1,536.42 neurons from
full inputs and checked account usage before and between models, reserving another 1,000
neurons through the run's own guard, kept with its artifacts (`bench/cloudflare_checkout.py`
has no budget guard; the later corpus collector in `bench/cloudflare_corpus.py` reserves 2,000). The returned token counts imply 1,145.82 neurons at the published rates. Analytics
initially returned no rows, then reported 967.07 neurons at 12:36 UTC; this delayed figure
is not a final billing total. No plan changes, deployments or additional inference calls
were made.

`golden/perf-cloudflare-20261004/` contains the public request/response journal, fresh local
samples, budget checks, and `summary.json`. Its `summarize.py` validates call counts,
response success, repeated answers, token counts, request hashes and the local binary hash
without sending requests. The five standalone collector regression tests pass for
wrapped/direct responses, truncated or missing usage, failed/incomplete responses, and
recording a transport failure without leaking the token or retrying.

## Newer saved observations, 2026-10-05

A read-only check of a separately collected 2026-10-05 API journal changes the earlier
truncation comparison. The same long Flash payload sometimes returns the full reported token
count and detects the outage sentence at the end, and sometimes reports 2,382 tokens and
misses that outage. The 27B reports the full count and detects the final outage in all six
measured long calls in this journal.

| Model | Full local token count | Hosted samples reporting the full count, ms | Samples reporting 2,382 tokens, ms |
|---|---:|---|---|
| Flash | 7,975 | 644, 710 | 703 |
| Flash | 15,612 | 984 | 359, 352 |
| 27B | 7,975 | 1,477, 1,625, 2,178 | none |
| 27B | 15,612 | 2,628, 3,025, 3,593 | none |

These are small groups of network-inclusive observations, not a stable service latency
guarantee. Full-count Flash calls return urgency around 0.96; the three short-count calls
return 0.018 despite identical submitted payload bytes. The 27B full-count calls return
urgency around 0.98. The reason for Flash's changing behavior is unknown. Reported counts
and tail detection do not prove identical tokenization or reveal internal precision, caching
or hardware.

`saved-hosted-long/verify.py` independently checks request hashes, all 20 call records,
response/usage consistency and all six local encoded input counts using the main host-only
encoder. It preserves the public payloads and answers without account identifiers in
`public-snapshot.json`, and pins the original journal hash in `verified.json`. It makes no
network or GPU calls. Full-count and short-count samples must stay separate in further
comparisons. The earlier 2,382-token observations remain valid for those calls; they do not
describe every current hosted request.

The completed parallel localhost comparison uses those same payload bytes. Its server binary
hash was independently checked against the still-present build. Each arm has three samples,
with 30 seconds idle before each request and keep-warm enabled. Arms run sequentially, so
this table describes absolute observations; the paired measurements above remain the
evidence for adopting the new kernel.

| Full tokens | Model | Local FP32 attention, ms | Local compensated, ms | Local matching-prefix hit, ms |
|---:|---|---:|---:|---:|
| 7,975 | Flash | 2,483.4 | 2,360.6 | 175.5 |
| 7,975 | 27B | 8,992.2 | 8,507.7 | 454.1 |
| 15,612 | Flash | 5,693.2 | 5,218.1 | 197.2 |
| 15,612 | 27B | 20,543.8 | 18,995.6 | 523.5 |

Cached answers exactly equal that prototype's uncached compensated API responses. Hits reuse
7,648 or 15,296 tokens, respectively; they follow an earlier keyed fill. They must not
replace uncached latency in the goal comparison. These are results from the parallel
prototype before the reviewed vote correction and Flash short GEMM dispatch, not a claim
that the final checkout has passed qualification. The fixtures and idle protocol also differ
from the back-to-back 8,072/16,347-token measurements.
`saved-hosted-long/local-verified.json` pins both completed reports, the server binary hash,
all samples, full token counts and cache-answer checks.

## Corpus response reference, 2026-10-04

The public 22-request corpus now has a separate reference captured from Cloudflare's hosted
`clef-flash` and `clef` endpoints. The existing FP32 goldens remain the numerical acceptance
target. A hosted response is evidence of that deployment's behavior on this date; Cloudflare
did not return a weight revision. Agreement is not labeled task accuracy.

Each model received one warm-up and two shuffled passes over all 22 requests. All 90 calls
succeeded, with identical answers across the two measured passes. The local production CLI
then processed the same payloads once per model with `--strict --no-truncate`. No inference
source or binary changed for this work.

### Results

Both local models agree with all 46 FP32 decisions. The hosted Flash model agrees with 45/46
and hosted 27B with 44/46 across the full corpus. Those totals include requests for which
the hosted service reports a smaller input count, so the table below uses only the 39
questions whose reported input counts match the full local encoding.

| Model and execution | Decisions matching FP32 | Mean maximum probability error | Maximum probability error |
|---|---:|---:|---:|
| Flash, local | 39/39 | 0.0000488 | 0.000210 |
| Flash, hosted | 39/39 | 0.0011784 | 0.007610 |
| 27B, local | 39/39 | 0.0000792 | 0.000387 |
| 27B, hosted | 38/39 | 0.0015241 | 0.020204 |

For each question, probability error is the largest absolute difference over its options
relative to the softmax of the saved FP32 logits. The mean averages those question-level
errors. Both local and hosted API probabilities are rounded to four decimal places; these
figures therefore differ slightly from comparisons using raw local logits.

On this subset, mean error is about 24 times lower locally for Flash and 19 times lower for
27B. This establishes closer reproduction of the pinned FP32 computation on these inputs. It
does not establish higher task accuracy or identify the hosted deployment's precision. A
reported token count match also does not prove token identity.

The 27B disagreement with matching input counts is `r019/service`. Local and FP32 select
`cdn-edge`; hosted selects `db-primary`. FP32's top two logits differ by only about 0.004.
The full-corpus Flash disagreement is `r021/service`; 27B also differs there. This request
has a large input-count mismatch and is not evidence of numerical error on equal input.

### Input and response differences

| Request | Full local tokens | Hosted tokens, both models |
|---|---:|---:|
| r001, invoice JSON | 260 | 258 |
| r017, numbers and nested JSON | 200 | 198 |
| r020, long logs | 8,072 | 2,340 |
| r021, longer logs | 16,347 | 2,340 |

The two-token differences for structured JSON are not treated as truncation proof. The large
text differences are consistent with the prefix truncation observed in the separate checkout
benchmark. Every local response reports its complete input count.

The unmodified corpus's first request received HTTP 422 from the hosted validator. A single
probe established that supplying the omitted `instructions` field with its reference
default, the question ID, makes it acceptable. `bench/cloudflare_corpus.py` applies that
adaptation to missing instructions and verifies exact equality of all encoded token IDs,
spans, option IDs and question metadata before sending anything. `ref/corpus.py` and the
FP32 goldens are unchanged. The rejected request and successful compatibility probe are
retained separately.

Hosted `confidence` also differs from the pinned response formatter. The pinned
implementation and Cleffa return the maximum option probability. Across this hosted sample,
confidence is consistent within 0.000228 with `(n * sum(p*p) - 1) / (n - 1)`, using the
rounded returned probabilities. That formula is an inference from observed responses, not a
documented contract. Raw confidence differences are retained but are not counted as
probability error. Score values, option probabilities and decisions are available separately
in the detailed comparison.

### Free allowance and evidence

The account reported 967.07 daily neurons before collection. At the [documented
rates](https://developers.cloudflare.com/workers-ai/platform/pricing/), the 90-call run
reserved 1,952.28 neurons using full submitted input counts, plus a 2,000-neuron margin for
delayed analytics and other account activity. The guard checked daily usage before each
model. There were no retries or account/plan changes. Returned token counts imply 767.70
neurons for the completed 90-call run. The final account-wide observation was 1,574.49
neurons; analytics were still delayed, so it is not presented as a reconciled bill. All
estimates fit the daily 10,000-neuron free allowance, including the two short preliminary
calls.

Local artifacts are under `golden/cloudflare-corpus-20261004/`, which is git-ignored:

- `hosted-explicit.jsonl`: plan, request hashes, all raw responses and usage checks.
- `cloudflare-clef-flash.responses.jsonl` and `cloudflare-clef.responses.jsonl`: first-pass
  response goldens with request IDs, hashes and full/reported token counts.
- `comparison.json`: per-question distributions, decisions, scores, repeat stability and
  grouped comparisons against FP32 and the local replay.
- `reference-manifest.json`: reference hashes and budget evidence.
- `local-manifest.json`, `clef-flash.jsonl`, `clef.jsonl`: local command, binary hash and
  responses.
- `hosted.jsonl`, `explicit-instructions-probe.jsonl`: initial rejection and compatibility
  probe.

Its full-corpus maximum response-probability error is 0.000299 for Flash and 0.000702 for
27B, including rounding.

## Current local replay, 2026-10-05

The current CLI, after the qualified attention and prefix-checkpoint integration, replayed
all 22 requests for each model against the saved 4 October hosted journal. This made no API
calls. It does not refresh evidence about Cloudflare's current deployment. Both local runs
used `--strict --no-truncate --batch 1`, and every response reported the complete local
input count, including 8,072 and 16,347 tokens.

Both current local models still match all 46 FP32 decisions. On the same 39 questions with
matching hosted and local reported counts:

| Model and execution | Decisions matching FP32 | Mean maximum probability error | Maximum probability error |
|---|---:|---:|---:|
| Flash, current local | 39/39 | 0.0000589 | 0.000231 |
| Flash, saved hosted | 39/39 | 0.0011784 | 0.007610 |
| 27B, current local | 39/39 | 0.0001210 | 0.001138 |
| 27B, saved hosted | 38/39 | 0.0015241 | 0.020204 |

Current local mean error is about 20 times lower for Flash and 13 times lower for 27B than
the saved hosted responses. It is higher than the older local build's error in the original
table. This is numerical distance on the public corpus, not evidence that either
implementation has better labeled task accuracy. Current local agreement with saved hosted
decisions remains 45/46 for Flash and 44/46 for 27B across the full corpus, or 39/39 and
38/39 on the count-matched subset.

The new artifacts are under `golden/performance-audit-20261005/`:

- `current-local.json` pins the commands, binary, source, runner and saved journal.
- `clef-flash.jsonl` and `clef.jsonl` contain the current public responses.
- `cloudflare-current-comparison.json` contains the grouped and per-question comparisons.

## Reproduce

Use a new output path for each capture; the collector refuses to overwrite evidence. Default
execution makes no network requests. Explicit collection uses the named account in the
current Wrangler login and stops if live usage cannot be verified or the bounded run would
exceed its conservative free budget.

```sh
.venv/bin/python -B bench/cloudflare_corpus.py golden/cloudflare-plan.jsonl
.venv/bin/python -B bench/cloudflare_corpus.py golden/cloudflare-new.jsonl --run --account ACCOUNT
.venv/bin/python -B bench/compare_cloudflare.py golden/cloudflare-new.jsonl golden/cloudflare-new-comparison.json
.venv/bin/python -B -m unittest discover -s tests -p '*cloudflare*.py'
```

For a local response comparison, add `--local-dir DIR` to the comparison command. That
directory must contain `clef-flash.jsonl` and `clef.jsonl`, one response per request in plan
order, with the matching model selector and full input counts, and each row must carry the
planned request's `id` and `request_sha256`: position in the file is not identity. The plan
itself carries an `input_ids_sha256` per request that must equal the FP32 oracle encoding's.
The 2026-10-04 journal and its local files predate both fields; the comparator refuses them
unless `--allow-unhashed-plan` is passed, and then records `input_ids_verified: false` and
`local_bound: false` per model in the summary. The saved `capture-local.py`
in this run's artifact directory reproduces the sequential local replay and checks for other
GPU workers first. Local replay is not performed by the hosted collector.

All 11 offline tests pass. They cover failed calls, missing usage, free-budget boundaries,
explicit-instruction adaptation, invalid probability distributions, rounded choice ties, and
missing or duplicate measured passes. The comparison rejects incomplete collections instead
of silently reducing its denominator.
