# Hosted comparison

Measurements against Cloudflare's hosted Flash, 27B and Omni endpoints on Workers AI: the
2026-10-10 core corpus with stock PyTorch controls, labeled BANKING77 and CLINC150+OOS subsets,
the public checkout fixtures and the public 22-request corpus. A hosted response is
evidence of that deployment's behavior on the capture date; Cloudflare does not return a
weight revision, and reported token counts do not prove token identity. Agreement is not
labeled accuracy, and a hosted time that covers fewer tokens is not an equal-work latency.
The 2026-10-03 comparison against Jev's hosted API is [at the end](#jevs-hosted-api-2026-10-03).

Local timings here are localhost HTTP, including encoding and the CPU head. Hosted timings
include the network from the measuring machine. The two were collected in sequential runs,
not interleaved, so they are absolute observations rather than paired comparisons.

## Fresh core comparison, 2026-10-10

The plan has 135 requests: 46 each for Flash and 27B, 43 for Omni. C is the integration branch's
strict localhost HTTP server, sent the same serialized bytes as hosted (encoded MOV videos
included), with media soft limits raised to admit the fixtures and startup warm-up and keep-warm
disabled. H1 and H2 are hosted passes at least 60 seconds apart; all successful answer objects
were identical between them, so H2 is not pooled. B is stock BF16 on MPS with PyTorch 2.11.0
([stock controls](performance-history.md#stock-pytorch-controls-2026-10-10)), no longer a
precision-noise control after the [causal-attention
finding](#corrected-bf16-comparison-2026-10-11). F is the historical FP32 reference used for
attribution; Omni's F includes native-input routing arbitration and is not independent numerical
qualification. Failed calls count as failed coverage. Probability statistics are the mean,
linearly interpolated p99 and maximum of each question's maximum absolute option difference.
Wilson intervals treat questions as units and are nominal, since several questions can share a
request.

| Model | C/H1 decisions | Wilson 95% | Mean / p99 / max probability difference | H1/H2 decisions | B/H1 decisions |
|---|---:|---:|---:|---:|---:|
| Flash | 80/81 | 93.33–99.78% | 0.02719 / 0.12300 / 0.13180 | 81/81 | 79/81 |
| 27B | 80/81 | 93.33–99.78% | 0.03353 / 0.24742 / 0.33790 | 81/81 | Unmeasured: stock load guard |
| Omni | 86/93 | 85.27–96.31% | 0.04378 / 0.39530 / 0.47470 | 93/93 | Unmeasured: stock load guard |

Hosted rejects ten of each dense model's 46 requests, carrying 22 questions (hence 81, not 103):
`v011`, `v012`, `vd001` and `vd004` for explicit `media_kwargs`, the other six video requests for
audio/video input. No processor settings were removed to admit them; Omni hosted accepts every
modality. Stock Flash completes 38 requests (87 questions), failing its eight encoded-video
requests in the processor; stock 27B and Omni stop while loading. `r001` and `r017` keep
two-token hosted count differences on all three models ([number
probe](#object-state-number-probe-2026-10-10)). On questions with matching reported counts and
verified C/F token IDs, C/H1 agreement is 74/75 for each dense model (92.83–99.76%) and 83/90 for
Omni (84.81–96.18%). The fresh long requests have full hosted counts, so the older truncation
observations below do not carry over.

| Model / modality | C/H1 decisions | Mean / p99 / max probability difference |
|---|---:|---:|
| Flash text | 45/46 | 0.02023 / 0.09855 / 0.10120 |
| Flash images | 35/35 | 0.03634 / 0.12806 / 0.13180 |
| Flash video | No hosted answers | Unmeasured |
| 27B text | 45/46 | 0.01720 / 0.09900 / 0.09940 |
| 27B images | 35/35 | 0.05498 / 0.29945 / 0.33790 |
| 27B video | No hosted answers | Unmeasured |
| Omni text | 44/46 | 0.01037 / 0.10749 / 0.14300 |
| Omni images | 15/15 | 0.03216 / 0.10977 / 0.11250 |
| Omni audio | 12/12 | 0.01102 / 0.02752 / 0.02800 |
| Omni video | 6/10 | 0.21903 / 0.46693 / 0.47470 |
| Omni audio/video | 9/10 | 0.07892 / 0.19810 / 0.20110 |

Probability differences are material even where decisions agree, and Omni's video gap survives
the [transport probe](#omni-encoded-video-transport-probe-2026-10-10). Neither agreement with F
nor the absence of a stock arm identifies hosted as wrong. The precision band frozen before C/H
scoring (95th percentile of B/F differences over 87 Flash control questions, 0.0822865884) was
withdrawn after the causal-attention finding; it remains in `precision-bands-wire.json` and
`comparison-wire.json`, and no replacement was fitted after seeing C/H. Every disagreement is
therefore unexplained, including small F margins without B evidence:

| Model / disagreement | Native → hosted | FP32 top-two probability margin | Max probability difference | Stock control |
|---|---|---:|---:|---|
| Flash `r004/urgency` | 1 → 0 | 0.01451 | 0.04770 | Failed its correctness audit |
| 27B `r019/service` | cdn-edge → db-primary | 0.00170 | 0.03460 | Unavailable |
| Omni `r015/language` | en → ja | 0.04051 | 0.04330 | Unavailable |
| Omni `r021/service` | billing → db-primary | 0.02622 | 0.06410 | Unavailable |
| Omni `ot001/direction` | left → still | 0.01998 | 0.35540 | Unavailable |
| Omni `ot003/moving` | true → false | 0.17703 | 0.17360 | Unavailable |
| Omni `ot003/direction` | right → still | 0.15740 | 0.38840 | Unavailable |
| Omni `ot004/direction` | right → still | 0.57130 | 0.47470 | Unavailable |
| Omni `oav002/audible` | false → true | 0.05661 | 0.03070 | Unavailable |

All have matching reported counts and verified C/F token IDs; hosted token IDs and media pixels
are not exposed.

| Control comparison | Decisions agreeing | Mean / p99 / max probability difference |
|---|---:|---:|
| Flash B/H1, same values for B/H2 | 79/81 | 0.03676 / 0.27944 / 0.58040 |
| Flash C/B | 84/87 | 0.02278 / 0.18295 / 0.44860 |
| Flash C/F | 103/103 | 0.000084 / 0.000393 / 0.000876 |
| Flash B/F | 84/87 | 0.02278 / 0.18326 / 0.44948 |
| Flash H1/F, same values for H2/F | 80/81 | 0.02718 / 0.12288 / 0.13092 |
| 27B C/F | 103/103 | 0.000172 / 0.001017 / 0.001138 |
| 27B H1/F, same values for H2/F | 80/81 | 0.03351 / 0.24710 / 0.33783 |
| Omni C/F, historical arbitrated reference | 93/93 | 0.000034 / 0.000049 / 0.000050 |
| Omni H1/F, same values for H2/F | 86/93 | 0.04378 / 0.39532 / 0.47473 |

Evidence is in `golden/evaluation-standard-2026-10-10/`: `wire-plan.json`, `protocol.json`,
`precision-bands-wire.json`, `hosted-wire-*.jsonl`, the native and stock captures, and
`comparison-wire.json` (coverage failures, every question and every disagreement).

## Labeled evaluation coverage, 2026-10-10

`bench/labeled_eval.py` builds BANKING77 and CLINC150+OOS requests from pinned public test splits
with the community Decision Index 0.2.1 templates and option ordering, and scores that harness's
macro-F1, averaged over labels observed in gold or predictions. It reproduces the public request
construction, not Cloudflare's unpublished hosted evaluation row for row; BFCL and Amazon ESCI
have not been run. Pairs require identical serialized bytes, so captures with literal Unicode and
with ASCII escapes are never paired. Hosted has one pass each: 385 BANKING77 and 453
CLINC150+OOS requests for Flash, 329 and 70 for Omni (fewer than half of CLINC's classes, so no
full-suite estimate), none for 27B. Native Flash (integration branch, strict HTTP) answered all
838 paired requests with matching body bytes and reported token counts, and every answer equals
the released engine's.

| Flash subset | Examples / classes | Native accuracy / macro-F1 | Hosted accuracy / macro-F1 | Stock BF16 accuracy / macro-F1 | C/H agreement, Wilson 95% |
|---|---:|---:|---:|---:|---:|
| BANKING77 | 385 / 77 | 0.8727 / 0.8709 | 0.8468 / 0.8503 | 0.8805 / 0.8778 | 366/385, 92.42–96.82% |
| CLINC150+OOS | 453 / 151 | 0.5453 / 0.6156 | 0.5453 / 0.6110 | 0.6093 / 0.6657 | 289/453, 59.27–68.09% |

| Flash subset | Correct only native / only hosted | Exact McNemar p | Native minus hosted macro-F1, paired 95% interval | Mean / p99 / max probability difference |
|---|---:|---:|---:|---:|
| BANKING77 | 12 / 2 | 0.0129 | +0.0206, [0.0037, 0.0441] | 0.12882 / 0.44429 / 0.71380 |
| CLINC150+OOS | 17 / 17 | 1.0 | +0.0047, [-0.0226, 0.0265] | 0.14106 / 0.44050 / 0.55910 |

Macro-F1 intervals use 2,000 paired request-bootstrap replicates, seed 20261010, without label
stratification; p-values are not adjusted for multiple comparisons. BANKING77 shows a native
advantage on this sample. CLINC establishes neither superiority nor equivalence; its equal
accuracy hides many rows where both arms miss the label differently. Subset scores do not
reproduce full-split leaderboard scores. Stock Flash answered all 838 requests without errors or
runtime overrides (339/385 BANKING77 and 276/453 CLINC correct, against native's 336/385 and
247/453) on the affected PyTorch 2.11 attention path; the [corrected
comparison](#corrected-bf16-comparison-2026-10-11) shows no significant CLINC advantage.

| Flash subset / pair | Decision agreement, Wilson 95% | Correct only first / only second | Exact McNemar p | First minus second macro-F1, paired 95% interval | Mean / p99 / max probability difference |
|---|---:|---:|---:|---:|---:|
| BANKING77 C/B | 377/385, 95.95–98.94% | 1 / 4 | 0.375 | -0.0069, [-0.0221, 0.0055] | 0.03595 / 0.18338 / 0.27200 |
| BANKING77 H/B | 358/385, 89.99–95.14% | 3 / 16 | 0.00443 | -0.0275, [-0.0553, -0.0064] | 0.14291 / 0.45772 / 0.68440 |
| CLINC150+OOS C/B | 311/453, 64.24–72.75% | 5 / 34 | 0.00000243 | -0.0500, [-0.0830, -0.0307] | 0.09253 / 0.34548 / 0.49740 |
| CLINC150+OOS H/B | 371/453, 78.09–85.17% | 11 / 40 | 0.0000570 | -0.0547, [-0.0900, -0.0304] | 0.12754 / 0.50769 / 0.73550 |

Stock chooses the native answer on all 19 BANKING77 C/H disagreements; of the 164 CLINC ones it
matches hosted on 102, native on 42 and neither on 20 (`labeled/decision-relationships.json`).
These are decision relationships, not precision explanations: the labeled rows have no FP32
margin, and the core precision band does not transfer. Native Omni answered all 399 available
paired requests through strict HTTP with matching body bytes and token counts:

| Omni subset | Examples / observed gold classes | Native accuracy / macro-F1 | Hosted accuracy / macro-F1 | C/H agreement, Wilson 95% |
|---|---:|---:|---:|---:|
| BANKING77 | 329 / 77 | 305/329 (0.9271) / 0.9254 | 306/329 (0.9301) / 0.9287 | 327/329, 97.81–99.83% |
| CLINC150+OOS | 70 / 70 of 151 | 69/70 (0.9857) / 0.9718 | 69/70 (0.9857) / 0.9718 | 70/70, 94.80–100% |

| Omni subset | Correct only native / only hosted | Exact McNemar p | Native minus hosted macro-F1, paired 95% interval | Mean / p99 / max probability difference |
|---|---:|---:|---:|---:|
| BANKING77 | 0 / 1 | 1.0 | -0.0033, [-0.0144, 0.0008] | 0.01212 / 0.07632 / 0.10830 |
| CLINC150+OOS | 0 / 0 | 1.0 | 0.0000, [0.0000, 0.0000] | 0.00938 / 0.03545 / 0.03600 |

The BANKING77 disagreements are `test:1751` (native `option_39`, hosted and gold `option_43`) and
`test:2675` (native `option_45`, hosted `option_39`, gold `option_66`); without stock Omni
answers, neither has a stock attribution. The zero-width CLINC interval reflects identical
decisions on 70 rows, not a zero population difference. These text-task results do not bear on
the core video discrepancies.

Full-split local runs on the released `fce959f` engine have no hosted arm. Flash on all 3,080
BANKING77 examples reaches accuracy **0.9078** and macro-F1 **0.9075**, request-bootstrap 95%
interval **[0.8965, 0.9163]** (2,000 replicates, seed 20261010): close to the reported Flash
leaderboard macro-F1 of 0.909, but not paired with Cloudflare's unpublished run. Full
CLINC150+OOS against stock BF16 PyTorch on MPS covers all 5,500 requests, including 1,000 OOS
examples, with identical request bytes:

| Full CLINC150+OOS arm | Accuracy | Macro-F1 | Macro-F1 95% interval |
|---|---:|---:|---:|
| Released native `fce959f` | 0.6318 | 0.6661 | [0.6506, 0.6740] |
| Stock BF16 PyTorch, MPS | 0.6478 | 0.7110 | [0.6970, 0.7188] |

Decisions agree on 3,956/5,500 rows; native alone is correct on 212 and stock alone on 300
(`p=0.0001164`, exact McNemar). Native minus stock macro-F1 is `-0.04485`, 95% interval
`[-0.05258, -0.03871]` (1,000 replicates). This favors the affected stock implementation and
cannot establish the quality of correctly masked BF16 inference; the corrected 5,500-row eager
run and a full current-branch rerun have not been collected.

Diagnostics on the same Flash requests leave the defaults unchanged. A stock FP32 control, not
the BF16 baseline, agrees with the released engine on all 384 byte-matched BANKING77 rows.
Sorting CLINC request keys gives the same 453 native responses (not admitted as another hosted
pair), so it does not explain the C/H gap. On CLINC, `CLEF_HEAD_BF16=1` changes no decision (247
correct) and adding `CLEF_ACT_F16=0` changes three (245 correct): the stock CLINC shift is the
attention defect below, not a missing native rounding switch.

Sources and labels are pinned in `golden/labeled-evals-20261010/`. Scores are in
`golden/evaluation-standard-2026-10-10/labeled/`: `released-flash-banking77-full-score.json`,
`released-flash-clinc-full-CB.json`, `score-clef-*-CH.json` for current Flash and Omni C/H, and
`score-clef-flash-*-CHB.json` for Flash's three arms.

## Expanded Omni labeled results, 2026-10-11

Completed local Omni and 27B subsets and an extended saved hosted Omni collection were rescored
with zero new hosted calls; Omni's numerical qualification remains incomplete. Native combines
strict HTTP and CLI captures (329 HTTP plus 56 CLI rows on BANKING77, 70 HTTP plus 262 CLI rows
on the hosted CLINC subset), whose 399 overlapping answer objects are identical;
`BANKING77:test:170` uses its HTTP response, because the CLI request has literal Unicode and the
hosted request ASCII escapes.

| Exact-byte paired subset | Native correct / macro-F1 | Hosted correct / macro-F1 | Decision agreement, Wilson 95% | Exact McNemar p |
|---|---:|---:|---:|---:|
| BANKING77, 385 rows | 355/385 / 0.9216 | 356/385 / 0.9243 | 383/385, 98.13–99.86% | 1.0 |
| CLINC150+OOS, 332 rows | 327/332 / 0.9854 | 326/332 / 0.9828 | 331/332, 98.31–99.95% | 1.0 |

Native minus hosted macro-F1 is `-0.002624` on BANKING77, interval `[-0.011222, 0.000742]`, and
`0.002649` on CLINC, interval `[0, 0.011029]` (1,000 pooled paired-request replicates, seed
20261010); neither establishes an accuracy advantage or equivalence. CLINC covers all 151 gold
classes but only 332 of 453 planned rows, from one hosted pass; the missing 121 hosted responses
are not inferred. Mean / p99 / maximum probability differences on four-decimal responses are
`0.012604 / 0.083528 / 0.1083` for BANKING77 and `0.011024 / 0.069828 / 0.1383` for CLINC. The
two BANKING77 disagreements are unchanged; CLINC adds `CLINC150+OOS:test:264`, native correct and
hosted wrong.

| Local model | BANKING77 accuracy / macro-F1, 385 rows | CLINC accuracy / macro-F1, 453 rows |
|---|---:|---:|
| Omni, integration branch | 0.9221 / 0.9216 | 0.9868 / 0.9866 |
| 27B, producer-reported released engine | 0.9169 / 0.9138 | 0.9735 / 0.9735 |

The 27B handoff names the released `fce959f` engine but has no per-run binary record, so its
provenance is producer-reported. Evidence:
`golden/evaluation-standard-2026-10-10/labeled/omni-completed-20261011/manifest.json`. A
2026-10-17 CLI recapture with the exact-rotary Omni default, scored against these saved hosted
answers, reproduces every native decision on both subsets and the core 86/93 agreement with the
same seven disagreements; ground truth for those is in [the Omni
report](omni.md#hosted-and-labeled-agreement), evidence in `golden/omni-recheck-20261017/`.

## Corrected BF16 comparison, 2026-10-11

The stock Flash CLINC shift comes from PyTorch 2.11 MPS `scaled_dot_product_attention(...,
is_causal=True)` on FP16/BF16 inputs, which the pinned HF path uses for unpadded records.
`ref/mps_sdpa_causal.py` compares implicit and explicit causal masking with CPU FP64 on exactly
the same dtype-rounded operands. Across nine cases (head dimensions 64/128/256, eight heads, 512
tokens, FP32/FP16/BF16), implicit FP16/BF16 causal attention has relative L2 error 0.402–0.432;
explicit masking gives about 0.00020 for FP16 and 0.00160 for BF16, and implicit FP32 stays below
4.53e-7. With cuts at tokens 1, 2, 7, 17 and 256, **all six implicit FP16/BF16 cases leak future
values**: changing values from token 1 onward changes token 0's output by as much as 3.9296875,
while all three FP32 cases and all nine explicit-mask controls keep exact prefix independence
(the midpoint shows no change, so an earlier midpoint-only probe missed the defect). This proves
leakage on the tested shapes, not the upstream fix. `ref/safe_attn.py` uses explicit masks
with `is_causal=False`; the independent FP32 and FP64 Omni qualification failures are separate.
Evidence under `golden/evaluation-standard-2026-10-10/`: `mps-causal-audit.json` and
`mps-causal-early-boundaries-summary.json`.

The finding changes evaluation attribution, not Cleffa inference or its numerical limits.
Historical B arms, the old Flash BF16 golden and their timings remain available but cannot
represent pure BF16 precision variation. The source-bound audit revokes the frozen control
without altering its threshold; recomputing every statistic from the original captures
(`comparison-wire-audited.json`, which binds `stock-control-correctness-audit.json`) leaves them
all unchanged. BF16 with eager attention, on the same pinned dependencies, is scored as a
separately named corrected variant, not a silent replacement for stock B:

| Exact-byte paired subset | Native correct | Eager BF16 correct | Decision agreement | Exact McNemar p | Native / eager macro-F1 |
|---|---:|---:|---:|---:|---:|
| Current branch, BANKING77 | 335/384 | 336/384 | 383/384 | 1.0 | 0.8709 / 0.8733 |
| Current branch, CLINC150+OOS | 247/453 | 250/453 | 449/453 | 0.25 | 0.6156 / 0.6222 |
| Released engine, full BANKING77 subset | 336/385 | 337/385 | 384/385 | 1.0 | 0.8709 / 0.8733 |

The current-branch BANKING77 pairing excludes `BANKING77:test:170`, whose saved native request
uses ASCII escapes while the eager and released-engine requests use literal Unicode. On current
CLINC, native minus eager macro-F1 is `-0.00657`, 95% interval `[-0.01473, 0.00003]` (1,000
pooled paired-request replicates, seed 20261010). Mean/max probability difference is
`0.007679 / 0.0753` on CLINC and `0.004292 / 0.0287` on the 384 BANKING77 rows. There is no
significant accuracy difference on these subsets; that is not an equivalence test. The 5,500-row
corrected baseline and stock inference on a newer PyTorch remain unmeasured. Evidence:
`golden/evaluation-standard-2026-10-10/labeled/eager-corrected/manifest.json`.

## Omni encoded-video transport probe, 2026-10-10

Five lossless MOV variants vary source frame rate, duration and frame count; both hosted passes
received the local encoder's bodies. Native C-worker RGB bytes were first checked against the
pinned PyAV loader and native token IDs against the pinned processor. Doubling the source rate
while repeating every frame (`vp000` versus `vp003`) gives identical local RGB, patches,
positions, tokens and processor tensors.

| Variant | Source frames / FPS | Duration | Locally sampled frames | Local / hosted tokens | Hosted moving probability |
|---|---:|---:|---:|---:|---:|
| `vp000`, baseline | 6 / 2 | 3 s | 6 | 652 / 652 | 0.4009 |
| `vp001`, slower source | 6 / 1 | 6 s | 6 | 652 / 652 | 0.4009 |
| `vp002`, faster source | 6 / 4 | 1.5 s | 4, including odd-frame duplication | 512 / 512 | 0.2564 |
| `vp003`, duplicated at double rate | 12 / 4 | 3 s | 6 | 652 / 652 | 0.4009 |
| `vp004`, longer duplicated sequence | 12 / 2 | 6 s | 12 | 1,072 / 1,072 | 0.1583 |

Responses were identical across two passes at least 60 seconds apart, and the locally equivalent
`vp000`, `vp001` and `vp003` returned identical hosted answer objects. Token counts follow the
sampled sequence length: no evidence of frame-count or FPS transport loss on these fixtures,
though hosted pixel values and resize arithmetic are not exposed. Hosted selects `still` for all
five variants, so transport alone does not explain the video decision gap. Evidence:
`golden/evaluation-standard-2026-10-10/` (`video-transport-plan.json`,
`hosted-video-transport.jsonl`, `video-transport-summary.json`, `free-budget-ledger.jsonl`).

## Object-state number probe, 2026-10-10

Six hosted Flash calls, including the warm-up, are in
`golden/cloudflare-serialization-probe-20261010/run.jsonl`.

| Payload | Local tokens | Hosted tokens | Hosted answers |
|---|---:|---:|---|
| Original `r001`, including warm-up repeat | 260 | 258 | Both identical |
| `r001`, change `1250.0` to `1250` | 258 | 258 | Identical to original |
| `r001`, change to `1250.5` | 260 | 260 | Changed |
| Original `r017` | 200 | 198 | Baseline |
| `r017`, JavaScript-style numeric representation | 198 | 198 | Identical to original |

Identical means equal returned answer objects, with probabilities rounded to four decimals; the
changed-value control shows this precision can distinguish a nearby value on `r001`. The results
support normalization of integral floats and signed zero before hosted encoding. The `r017`
variant also replaced `12345678901234567890` with `12345678901234567000`, as a JavaScript JSON
round trip would, but bundled with the signed-zero normalization, so hosted large-integer
rounding remains an inference; Cleffa and the pinned Python encoder keep the exact integer. Only
Flash received this probe, and hosted token IDs and the server's serializer are not exposed. The
core comparison keeps its original payloads. The user chose to keep Cleffa's Python-compatible
numeric behavior; no hosted-compatible serialization mode will be added for this integration.

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

The offline tests cover failed calls, missing usage, free-budget boundaries,
explicit-instruction adaptation, invalid probability distributions, rounded choice ties, and
missing or duplicate measured passes. The comparison rejects incomplete collections instead
of silently reducing its denominator.

## Jev's hosted API, 2026-10-03

Jev is Typesafe's own model behind the same SystemOne API, not Clef. The cleffa columns below are
from the 2026-10-03 baseline build; current engine timings are in [the README](../README.md#speed). On 2026-10-03 the corpus ran
against Jev's API (`jev-latest`, which answered as `jev-1.13.0`) and against `clef-server` on
localhost, with `bench/jev_compare.py`. Each endpoint got one warm-up request, then three
back-to-back passes on one keep-alive connection, from an M5 Max in Italy.

Jev's times include the network. An authenticated `GET /v1/models` took 192 ms (median) from
here, so most of Jev's time is round trip, and its numbers depend on where the client is.

| Requests (Clef tokens) | Jev (incl. network) | cleffa 27B | cleffa flash |
|---|---:|---:|---:|
| 17 requests, 153–363 tokens | 230–261 ms | 236–484 ms | 78–155 ms |
| 594 tokens, 6 questions | 234 ms | 826 ms | 231 ms |
| 2,235 | 273 ms | 3.03 s | 0.83 s |
| 8,072 | 287 ms | 14.6 s | 4.26 s |
| 16,347 | 334 ms | 42.3 s | 12.9 s |
| **median (21 requests)** | **240 ms** | **398 ms** | **139 ms** |

The 27B column is in the heat-soaked regime (the long requests run in every pass). Jev reported
the full input (16,445 tokens for the 16k request), so it doesn't truncate.

Jev rejects one corpus request: a `noul` question with neither instructions nor criteria, which
the Clef reference accepts (`Noul question must have criteria or instructions`). The table and
the comparison below cover the other 21.

The corpus has no labels, so this measures agreement, not accuracy:

| Pair | Same decision | Median total-variation distance |
|---|---:|---:|
| Jev vs cleffa 27B | 43/45 questions | 0.040 |
| Jev vs cleffa flash | 38/45 | 0.041 |
| cleffa 27B vs cleffa flash | 41/46 | 0.024 |

Jev and the 27B differ on two low-margin questions: a double-charge ticket's urgency (Jev "this
week" at 0.56, the 27B "today" at 0.62), and the most-affected service in an 8k-token log of
uniformly random lines, where Jev's pick has the most 5xx statuses and the 27B's the most ERROR
lines. For labeled accuracy, see Cloudflare's model card, where Clef and Jev each lead on
different benchmarks.

Jev's probabilities are rounded to two decimals and changed by up to 0.08 between passes; its
decisions didn't. cleffa's responses were byte-identical across passes.

Against Jev from here, local flash is faster on every request up to 363 tokens and level at 594.
The 27B is slower on all but the shortest (153 tokens: 236 vs 254 ms). Jev's lead grows with
length: 3× (flash) to 11× (27B) at 2k tokens, 39× to 127× at 16k.
