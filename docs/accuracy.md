# Accuracy

How cleffa's outputs are evaluated, how close the dense models are to the FP32 engineering
reference and why, the precision choices behind that, the reference defects found along the way,
and host-side byte parity. Clef-Omni's status is in [the Omni report](omni.md).

## Evaluation metrics

Hosted decision agreement and paired labeled accuracy are the primary evaluation
metrics. The fresh 2026-10-10 core comparison gives 80/81 decisions agreeing with
hosted for Flash, 80/81 for 27B, and 86/93 for Omni. Omni's video discrepancies
remain unresolved. On exact-body paired labeled text subsets, Omni agrees with
hosted on 383/385 BANKING77 and 331/332 CLINC decisions (mixed CLI/HTTP native captures);
Flash scores higher than hosted on BANKING77. The earlier stock BF16 CLINC
advantage used PyTorch 2.11 MPS with incorrect implicit causal attention. The corrected
eager-attention comparison finds no significant accuracy difference from Cleffa
on either labeled subset. This does not establish equivalence or general
superiority. See the [hosted and labeled results](hosted-comparison.md#fresh-core-comparison-2026-10-10)
for confidence intervals, coverage, failures and exact comparison arms.

## Dense models against the FP32 reference

The tables below measure agreement with a local HF FP32 engineering
reference, not semantic accuracy or Cloudflare's serving implementation. The
BF16 control uses the released `joint_schema_model.py` on Transformers 5.10.2 and
MPS; the FP32 reference upcasts the same source weights.
The historical Flash BF16 control below uses the affected implicit causal SDPA
path; its distance from FP32 is not a measurement of BF16 rounding alone.
Cleffa's dense backbone matmuls use FP16 activation operands with FP32 accumulation.
Those operands are 8× finer than the BF16 the reference feeds its Linears, at the
same tensor-op rate (`bench/mixed_bench.m`). Attention uses high and residual FP16
planes with FP32 accumulation; the omitted residual-times-residual products make
this an approximation. `CLEF_ATTN_TU=0` restores tiled FP32 attention. The residual
stream and CPU head remain FP32. Weights are the exact BF16 values.

Measured over the 22-request / 46-question corpus in `ref/corpus.py`. Inputs range from 146 to
16,347 tokens and cover all question types and tokenizer edge cases.
The mean probability-error column averages each question's largest option error, as in
`tests/compare3.py`.

### clef-flash

| vs FP32 reference | argmax agreement | mean \|Δp\| | max \|Δp\| |
|---|---|---|---|
| **default compensated attention** | **46/46** | **0.00004576** | **0.00023393** |
| FP32 attention control | 46/46 | 0.00003576 | 0.00024937 |
| HF BF16 on MPS | 44/46 | 0.0098 | 0.0956 |

With the FP32 attention control, hidden states are 28–289× closer to FP32 than the BF16 reference at every layer
(final norm: 7.4e-4 vs 1.8e-1 relative L2). The engine agrees with FP32 on both
questions where the historical BF16 output differs. This historical
BF16 comparison includes that attention defect; no labeled correctness claim follows.

### Clef (27B)

| vs FP32 reference | argmax agreement | mean \|Δp\| | max \|Δp\| |
|---|---|---|---|
| **default compensated attention** | **46/46** | **0.00009400** | **0.00113585** |
| FP32 attention control | 46/46 | 0.00007982 | 0.00073920 |
| HF BF16 on MPS, head-split SDPA for r021 | 46/46 | 0.0072 | 0.0437 |

For the FP32 attention control, the closest call is `r019/service`, where the oracle's
top two logits are 0.004 apart. That control is
0.0002 from FP32 there; BF16 is 0.025 away but retains the same argmax.
On `r000`, the engine's hidden states are 44–351× closer to FP32 than BF16's at every
layer (final norm: 2.4e-3 vs 3.0e-1).

### Compensated attention

The compensated path increases mean error against this small FP32 corpus while
keeping all decisions within the existing numerical bounds. In the separate
frozen ContractNLI development and test evaluation, all 6,256 predictions across
both models remain unchanged, with small mixed probability-score shifts. This is
observed task stability, not equal probabilities or an accuracy guarantee on
unseen inputs. See [the integration report](attention.md#adoption-as-the-default)
for paired fresh-input timings and [the task evaluation](attention.md#labeled-evaluation-on-contractnli)
for labeled accuracy and cohort limits.

## Why FP16 activations

With BF16 activations, the engine scored 45/46 on the 27B, with
mean |Δp| 0.0007, and missed `r019/service`. Rounding each backbone matmul's input to BF16 was
its main error source.

Switching one producer class to FP16 also reached 46/46 on this corpus without lowering the
error, which is how near-ties behave. So the measure is the error itself, and only FP16 on all
four classes cut it about 7× on both models:
- the RMSNorm outputs;
- the attention, DeltaNet and SwiGLU outputs.

The head now takes the f32 hidden state, where HF's BF16 path rounds it. Its small GEMMs run on
f32 activations.

## FP16 range guard

FP16 tops out at 65,504.
- **How close real inputs come:** the largest matmul input over the corpus (`ref/act_stats.py`)
  is 3,648 on the 27B, the MLP down-projection input in layer 63. Every other input is ≤ 112.
- **What happens past the limit:** if any value of a record would exceed it, or is NaN, the
  engine reruns that record alone with BF16 activations.
- **The discarded pass stays finite.** In that pass, the out-of-range value is written as
  ±65,504, never `inf`. Attention multiplies other records' rows by exact zeros, and 0 × `inf` is
  NaN: an earlier version leaked NaN into co-batched records this way. A code review caught it,
  and the test now covers it.
- **Last line of defence:** a non-finite logit for any reason fails the batch with an error
  instead of returning a NaN decision. The server then retries each record alone.
- **Batch effects:** the decision depends only on the record's own rows. Batching stays
  bitwise invariant, and one request cannot change another's numbers.
- **Worst case:** a crafted request can at most double its own GPU time, and requests batched
  with it wait for that rerun. Those batches are capped by `--batch-tokens`; larger requests run
  alone.
- **Test:** `tests/test_f16_overflow.sh` forces the path with a lowered limit, which takes the
  same write path as a real overflow. It covers:
  - each producer class's flag;
  - a limit that mixes overflowing and clean records in one batch;
  - `--dump`.

`CLEF_ACT_F16=0` (BF16 activations) together with `CLEF_HEAD_BF16=1` (BF16-rounded head input)
selects the reference's operand rounding. Long 27B requests still use the chunked FP32
DeltaNet recurrence described below.

## The PyTorch reference on long 27B inputs

The PyTorch reference is wrong on long 27B inputs unless run with `--safe-attn`.
`scaled_dot_product_attention` on MPS (torch 2.11) returns wrong output for every head whose
score-matrix offset `h·T²` reaches 2^32, a 32-bit overflow.
- **Who it hits:** the 27B (24 heads) from 13,666 tokens. clef-flash (16 heads) never reaches it
  within its 16,384-token window.
- **Reproduction:** `ref/mps_sdpa_bug.py`, with random tensors and no model.
- **Upstream:** this is [pytorch/pytorch#179352](https://github.com/pytorch/pytorch/issues/179352),
  closed as an Apple MPSGraph bug (Apple FB22437937). It is still present in torch 2.11.0 on
  macOS 27.0.1.
- **Workaround:** `--safe-attn` on the oracles (`ref/safe_attn.py`), which splits the heads so
  each call stays below the limit. With it, that layer's attention is 7.6e-6 from float64.
  Without it, heads 17–23 are 0.27–0.72 off.

The first 27B references were generated without the workaround and were wrong for `r021`
(16,347 tokens). The FP32 reference was also wrong for `r020` (8,072 tokens, below the limit),
but only in the full-corpus run. Run alone, together with `r021`, or in the full corpus with
`--safe-attn`, `r020` comes out bitwise identical every time. Why the first run differed is not
established. The 27B numbers above use references regenerated with `--safe-attn`.

## Host byte parity

Host-side pieces must match Python byte for byte, and they do:

| Component | Result |
|---|---|
| Tokenizer vs HF | 25,677 / 25,677 strings, including an NFC control arm |
| `json.dumps` / `repr(float)` / `round()` | 404k floats, 240k roundings, 20k documents |
| Request encoding (`encode_record`) | 3,089 requests, 43 with images (ids, spans, image runs, 3D positions) |
| Response building (`systemone_answer`) | 1,338 responses |
| Image decoding, resizing, normalization, patches, position interpolation | 100 PNG/JPEG images of every color type, scan layout, JPEG colour-space marking and quantization-table placement, byte-identical to PIL, torchvision and the processor |
| Video sampling, resizing, patches, timestamp tokens, 3D positions | 14 frame-array requests, byte-identical to the processor (`tests/test_video.py`) |

## Images

The vision tower runs its 27 layers before any text is read, so operand rounding there
compounds. Its producers retain f32 outputs. Non-residual GEMMs use compensated high/residual
FP16 operands with FP32 accumulation; residual GEMMs keep direct FP32 inputs. On the
16-request / 41-question vision corpus (`ref/corpus_vision.py`, 1 to 3 images per request, 16 to
1,024 image tokens) against the FP32 oracle, clef-flash answers 41/41 with max |Δp| 0.0009,
and the 27B 41/41 with max |Δp| 0.0010; with
16-bit tower operands (`CLEF_VIS_F32=0`) two of clef-flash's questions exceed the limits (max |Δp|
0.0025, features 2.9e-3 away). The text corpus is unchanged on both models. Details and costs:
[docs/vision.md](vision.md).

## Video

On the separate 8-request / 16-question synthetic video corpus (`ref/corpus_video.py`)
against the streamed FP32 oracle, clef-flash answers 16/16 with max |Δp| 0.0002 and the 27B 16/16
with max |Δp| 0.0007. That is parity on synthetic clips, not measured accuracy on real video.
These GPU results predate the branch's rebase onto the final image-path changes; see
[docs/video.md](video.md#numerical-parity) for scope.
