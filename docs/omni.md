# Clef-Omni

Native inference for the pinned Cloudflare Clef-Omni release
(`Cloudflare/clef-omni@0db1cd2607d76a7bdb2a382f659e7b313079f84b`) through the existing `clef` CLI
and `clef-server` SystemOne endpoint: text, images, silent video, audio and video with
soundtracks. Model matrices stay BF16, never quantized; activations and residuals are FP32. The
reference disables speech generation and returns typed decisions only, so the talker is not
ported. Numbers are from the one tested M5 Max (128 GB), one GPU job at a time.

## Status

All five input kinds run through the CLI and HTTP, with batch, poison, failure-recovery and
resource-limit checks. Since 2026-10-16 Omni's numerical acceptance is **FP64-bounded**: against
the independent FP64 reference, each question's error must be within the larger of the fixed
limits and the FP32 PyTorch reference's own error ([policy](#numerical-verification-policy)).
The working default (the current tree) **passes** it on every request that has an FP64
reference: 42 requests, 91 questions, all decisions equal to FP64 ([results](#against-fp64)).
Against the fixed limits alone it exceeds them on 8 of those questions, the FP32 reference on 17.
Coverage stops at 8,069 tokens: the 16,344-token `r021` reference is being generated, and nothing
near the 64,000-token default context has a reference ([not done](#not-done)).

Under the earlier FP32-only gates, which no longer qualify Omni, the default reproduces all 43
token sequences and 93 decisions of the short corpora but fails 207 of 2,150 residual checks and
12 final-output checks ([results](#working-default-against-fp32-diagnostic)).

An FP32-compatible build, made to pass those FP32-only gates, exists on separate branches, not merged. It replaces the working default's own
compensated router and attention arithmetic, and several vision, decoder and audio operators,
with implementations that reproduce the pinned PyTorch 2.11 MPS reference operation for
operation. It passes every gate on the short corpora and the packing, poison, recovery and HTTP
checks. It changes none of the 93 decisions, its labeled and hosted agreement has not been
measured, and it is slower than the working default:

| Build | Short-corpus FP32 gates | 298 tokens, default / build | 2,231 tokens, default / build | Branch |
|---|---|---:|---:|---|
| Working default | Fail: 207/2,150 residual, 12/93 final-output checks; 93/93 decisions | — | — | This branch |
| Merged FP32-compatible build, 2026-10-14 | Pass: 43 requests, 93 questions, 2,150 stages (2,050 byte-exact) | 246.259 / 512.645 ms (2.08173) | 1,507.743 / 2,253.775 ms (1.49480) | `omni-fp32-compatible-20261014` |
| Fused-validation candidate, 2026-10-16 | Pass: same counts and byte-exact stages | 244.688 / 382.566 ms (1.563483) | 1,573.819 / 1,784.752 ms (1.134026) | `omni-fp32-fused-20261016` |

Each ratio comes from its own job pairing that build with the working default; latencies from
different jobs are not comparable. Each branch is one commit applying the build's evidence
patch (`golden/omni-working-integration-20261014/from-root.patch` and
`golden/omni-tiny-fused-validation-portable-tests-20261016/from-current-root.patch`) unchanged
to the engine commit. The fused-validation build descends from the merged build through the steps in
[the lineage table](#fp32-compatible-build); it adds up to 249.125 MiB of calculated expert
buffer capacity, and no Flash/27B regression is recorded for it. Both add the header-only C++
PocketFFT library behind a C ABI bridge and a SLEEF-derived logarithm for the audio frontend.

These builds match PyTorch's MPS rounding rather than FP64, cost 1.13–2.08× the default's
latency, and are not needed under the FP64-bounded rule; they stay unmerged for reference. The
hosted and labeled results below were measured on the working default.

## Running it

```sh
./download_models.sh clef-omni
make
./clef -m gguf/clef-omni.gguf requests.jsonl
./clef-server -m gguf/clef-omni.gguf --port 8080
.venv/bin/python -B tools/audio_request.py clip.flac --request request.json > audio.jsonl
.venv/bin/python -B tools/video_request.py clip.mp4 --model-family qwen3_omni_moe \
  --request request.json > video.jsonl
```

The setup command produces `gguf/clef-omni.gguf`: 1433 tensors, 63.659 GB, every value checked
against the pinned snapshot. Development exports made before the video processor metadata
(`clef.vision.video_processor` and its pixel bounds) was added lack those keys and must be
reconverted.

**Audio.** Native `audio` is a list of finite mono 16 kHz sample arrays, WAV data URLs, or
`{"content_type":"audio/wav","base64":"..."}` envelopes. WAV accepts PCM16 or IEEE float32 with
one channel at 16 kHz. URLs, file paths and bare audio base64 strings are rejected. The helper
`tools/audio_request.py` accepts local WAV, FLAC, MP3, Ogg and MP4/MOV, resamples in a bounded
subprocess, and emits float32 WAV without quantizing samples. Decoder parity was checked with
PyAV 19.0.1; a decoder upgrade needs requalification. The checked build links libavcodec/libavformat
63.1.102 and libswresample 7.1.102; the video helper uses Homebrew FFmpeg 9.0.2. PyAV is a Python
reference/conversion dependency.

**Encoded video.** The optional `clef-media` C worker links the codec libraries; the inference
executables do not. Build it with `make media`, keep it beside the CLI, server and host tool,
and start the server with `--allow-encoded-video` (the CLI accepts uploads by default). It
accepts MP4/MOV base64 envelopes or data URLs in `videos`, preserving soundtracks. No Python
runtime is needed for this path. See [the video report](video.md#encoded-uploads) for bounds and
isolation.

**Frame arrays.** Omni `videos` contain frames already sampled at 2 FPS:
`{"frames":["data:image/png;base64,..."],"fps":2,"audio":"data:audio/wav;base64,..."}`. The `audio`
member is optional, but all videos in a record must consistently have soundtracks or
consistently omit them; the engine refuses the mixed case, which the reference resolves by
discarding tracks. Standalone clips precede soundtracks in shared frontend padding. Video rows
win timestamp ties during temporal merging. When overriding `media_kwargs.videos_kwargs`, include
`fps:2`, `do_sample_frames:false`, and `use_audio_in_video:true` for soundtracks. The converter
sets these fields and preserves frame/audio association automatically.

**Limits.** CLI/server defaults are four audio clips, including soundtracks, and 4,800,000 total
samples per request. `--max-audio` and `--max-audio-samples` override those limits; zero removes
only that soft limit. Context, body, finite-value and integer bounds still apply.
`--max-image-requests` also bounds concurrently held audio requests. The default 8 MiB HTTP body
limit can be reached before the sample cap: a full 4,800,000-sample PCM16 request passes with
`--max-body 16777216`, and the default cap rejects it with 413.

**Context and caches.** Omni defaults to 64,000 tokens with a configured ceiling of 65,536;
numerical qualification of the long-context range is pending. Prefix/template-cache requests run
through the ordinary native path without Omni prefix reuse. `--dump-routes FILE` writes the
first request's router logits, selected experts and weights as JSONL (`--dump-last` bounds the
rows).

## Architecture

### Model family and loading

Omni is a family inside the existing engine, not a separate executable: it shares GGUF parsing,
tokenization, JSON/schema handling, the joint head, request packing and serving, and gets its own
decoder, MoE, media positions and audio path. `general.architecture` stays `clef`. Omni exports
add `clef.backbone = "qwen3_omni_moe"`, `clef.format_version = 1`, `clef.capabilities` (1 text,
3 with vision, 7 with audio), `clef.context_default` and `clef.context_length = 65536`.
`clef_model.c` requires all five keys together. A file with none of them is the Qwen3.5 layout;
a file with Omni tensors but no metadata is rejected rather than guessed. `clef.c` then pins the
decoder, head, vision and DeepStack geometry before any size product or allocation, and
`clef_audio_config_load` checks the audio tower and Whisper frontend configuration field by field.

| Component | Pinned geometry (checked at load) |
|---|---|
| Decoder | 48 full-attention layers, hidden width 2048 |
| Attention | 32 query heads, 4 KV heads, head dimension 128, Q/K RMS normalization, no query gate |
| Norm | Plain RMSNorm weight, epsilon 1e-6, without Qwen3.5's `1 + weight` |
| Rotary | All 128 dimensions, theta 1,000,000, interleaved sections `[24, 20, 20]` |
| Experts | 128 per layer, top 8, intermediate width 768, normalized selected probabilities, no shared expert |
| Embedding/output | BF16 `[152064, 2048]` |
| Joint head | Width 1024, 2 routing layers, 4 decoder layers, 16 heads, FFN 4096 |
| Vision | 27 layers, width 1152, 16 heads, FFN 4304, patch 16, temporal patch 2, merge 2 |
| DeepStack | Extra mergers after vision blocks 8, 16, 24; injected after decoder blocks 0, 1, 2 |
| Audio frontend | Mono 16 kHz, 128 mel bins, FFT 400, hop 160, no dither |
| Audio tower | Three stride-2 Conv2d layers, 32 transformer layers, width 1280, 20 heads, FFN 5120, output 2048 |
| Video timing | 2 FPS sampling; 13 temporal position units per second |

`tools/convert_omni.py` exports the `thinker.*` tensors and the joint head and excludes
`talker.*` and `code2wav.*`. It streams individual source matrices instead of materializing an
expert bank. Q/K/V are stored as `[q | k | v]` with no gate rows; experts are rank-3 BF16 in
expert-major `[expert, output, input]` order with `[gate; up]` inside each expert; the router is
BF16, executed with FP32 operands; plain RMSNorm weights are exact BF16-to-FP32 conversions. The
extra mergers come from the checkpoint names `thinker.visual.merger_list.{0,1,2}.*`.
`tests/verify_omni_gguf.py` derives the expected inventory and transformations independently of
the converter and rejects missing, extra, duplicate or malformed tensors. A text-only development
export declares its capabilities, and media requests to it fail explicitly.

### Decoder and experts

`clef_metal_omni.m` and the Omni section of `metal/clef.metal`. Every operand is FP32 (the FP16
activation path is disabled for Omni), so the FP16-overflow rerun does not apply. Per layer:

- RMSNorm, then the fused QKV GEMM on FP32 activations against BF16 weights.
- `omni_attn_prep`: Q/K RMS normalization and interleaved M-RoPE. Rotary pair `i` uses position
  axis `i % 3` for `i < 60`; pairs 60–63 use the temporal axis, because the `[24, 20, 20]` layout
  has only 20 complete triplets. Text cannot expose this; `make test-omni-attention` checks it
  against an FP64 RMS/RoPE oracle that includes the four temporal-only pairs.
- `omni_attention`: tiled FP32 attention (8 queries × 32 keys) with two-float compensated
  accumulators and record-aligned key blocks, then the output projection into the residual.
- Router: `omni_router` keeps each of the 128 logits as a high/low FP32 pair, compensating
  products and sums, because expert boundaries can be closer than a plain FP32 dot product's
  error. `omni_route` ranks the pairs before the FP32 softmax, takes eight, breaks exact ties by
  the lowest expert ID (the pinned `torch.topk` has no stable tie rule) and renormalizes the
  selected weights. Nonfinite logits set a status bit checked at completion.
- Experts: each record is processed in chunks of up to 1,024 tokens, positioned relative to the
  record so GEMM shapes do not depend on batch neighbors. Assignment counts, offsets and the
  stable scatter run in parallel, one SIMD group per expert, preserving the serial bucket order.
  Each expert's run is padded to 32 rows; one GEMM grid covers all runs, and each 32×128 tile
  finds its expert by an upper-bound search over the offsets. Products are FP32 activations times
  BF16 weights on Metal tensor ops. `omni_combine` sums the eight contributions in ascending
  expert ID and adds the residual once, as eager PyTorch does. The expert arena is 298.02 MiB,
  independent of context ([performance history](performance-history.md#omni-expert-chunks-2026-10-12)).

### Vision and DeepStack

Omni reuses the vision tower code with Omni-specific steps: projection biases follow the pinned
Linear-plus-bias, then residual rounding order (`omni_vision_residual`) instead of being folded
into the next norm, and all tower GEMMs use direct FP32 operands (`CLEF_VIS_COMP=0` is therefore
a no-op for Omni). The main merger normalizes each 1152-value patch before grouping four; the
extra DeepStack mergers group first and then normalize 4608 values. Their outputs are added only
to visual rows after decoder blocks 0, 1 and 2. The reference encoder overrides the processor's
image bounds to 65536–1048576 pixels.

### Audio

`clef_audio.c` decodes WAV after validating the whole RIFF chunk table, checks finiteness and
sample-count arithmetic before allocation, and computes features per clip: periodic Hann window,
a fixed-scratch 400-point mixed-radix FFT with FP64 accumulation and FP32 storage, sparse Slaney
mel filters from `make audio-tables`, `log10` with the floor and dynamic-range clipping of the
pinned Whisper extractor. `clef_metal_audio.m` runs the tower in FP32: three Conv2d/GELU stages
over bounded im2col batches, sinusoidal positions, 32 transformer layers with bidirectional
attention inside per-clip windows of at most 104 rows, final norm and the projection to 2048.
Token counts follow the pinned chunk-length formula, not a rounded tokens-per-second estimate.

### Positions and media layout

`clef_record_omni.c` places visual and audio tokens and computes 3D positions. Video timestamps
use 13 temporal units per second. With soundtracks, audio and video rows are interleaved by a
stable merge of temporal indices with video first on ties, following Transformers 5.10.2's
`Qwen3OmniMoeProcessor` and `get_rope_index` (whose replacement method ignores
`seconds_per_chunk`). Audio padding is per record and includes associated soundtracks. The full
media token sequence is budgeted before any feature allocation and never truncated inside a
media run.

### Source files and tests

| File | Owns |
|---|---|
| `clef_model.c` | Family/format/capability metadata; audio configuration checks |
| `clef.c` | Omni geometry checks and exact-rank tensor binding |
| `clef_metal_omni.m` | Decoder attention dispatch, router, expert assignment and products, route traces |
| `clef_metal.m` | Shared lifecycle, vision tower including Omni mergers, DeepStack injection |
| `clef_metal_audio.m` | Audio tower |
| `clef_audio.c` | Sample decoding and frontend |
| `clef_record_omni.c` | Audio placement, visual tokens and positions |
| `clef_media.c`, `clef_encoded_video.c` | Isolated MP4/MOV worker and its host side |
| `tools/convert_omni.py`, `tests/verify_omni_gguf.py` | Export and independent verification |

Model-free checks: `make test-omni-attention test-omni-router test-omni-moe test-omni-assign
test-omni-deepstack test-omni-audio test-omni-buffers test-audio`. The Python suites are
`tests/test_omni_*.py`; `tests/test_omni_qualification.py` is the qualification runner below.

## Numerical verification policy

Hosted decision agreement and labeled accuracy are the headline evaluation metrics. Numerical
changes are gated against the independent CPU FP64 reference, bounded by the FP32 PyTorch
reference (adopted 2026-10-16). Neither reference describes Cloudflare's serving arithmetic.

**Why not fixed FP32 gates.** Top-8 routing over 128 experts has near-ties finer than FP32
resolves (reference score gaps down to `3.31363e-8`). Any rounding difference flips some expert
choices and the difference propagates, so no FP32 implementation stays within fixed per-layer
limits of any single reference: the blocked and SDPA FP32 references differ from each other by
0.114956 relative L2 on text, and against FP64 the FP32 reference itself exceeds the final-output
limits on 17 of 91 questions. Passing fixed FP32 gates required reproducing PyTorch's MPS
rounding (the FP32-compatible branches), not computing the model more accurately.

**Gates.** Exact token IDs. For every question, against the FP64 reference: finite outputs, and
a maximum absolute logit error and maximum probability error each within the larger of the fixed
limit (0.05 and 0.002) and the FP32 reference's own error on that question; the argmax must equal
FP64's or the FP32 reference's. `tests/test_parity.py gguf/clef-omni.gguf FP64_GOLDEN --bound-by
FP32_GOLDEN` implements this and requires both references to have encoded identical token IDs.
Per-layer relative L2 (0.01) is reported as a diagnostic; non-finite residuals still fail. The
fixed limits are never relaxed beyond this, and the SDPA noise-floor control never supplies a
tolerance. The earlier FP32-only gates remain a diagnostic. `tests/test_omni_qualification.py`
still applies fixed limits to whichever reference it is given, so it does not implement this rule.

**FP32 reference.** The bound uses the "FP32 reference (blocked attention)":
`ref/oracle_f32_stream.py --omni-attention blocked`, from layer zero for every corpus. It bounds
query storage and uses blocked value sums even on short inputs. `--omni-attention sdpa` is the
independent FP32 noise-floor control with identical inputs; its spread is compared on the same
question only, never transferred between requests or modalities. The reference pages weights so
only the current module is resident, and `--stage-layers N` runs fresh subprocesses to bound
PyTorch MPS graph-cache growth (three new routing shapes raised driver memory from 7.31 GiB to
18.53 GiB with one fixed expert layer). Checkpoints bind arguments and source hashes across every
stage. Dense `--safe-attn` keeps its committed head-only splitting byte for byte; a regression
compares it with the branch base. Long text references keep the last 512 rows per stage
(`--dump-last`); media references keep all rows.

**Routing is a diagnostic.** The reference always computes expert choices, outputs and mixture
weights from its own activations; native logits, activations or choices never enter it. Reports
give ordered-choice and selected-set difference rates, neither of which is a gate. The FP64
arbitration options (`--arbitrate-routes`, `--arbitrate-activations`, `--arbitrate-rounding`,
`--arbitrate-input-shift`) remain available as disclosed diagnostics; the qualification runner
rejects arbitrated or mixed-policy references. Earlier arbitrated goldens
(`golden/clef-omni-text-f32-arbitrated`, `golden/clef-omni-vision-f32-residual-order`,
`golden/clef-omni-video-f32-input-shift`, `golden/clef-omni-audio-f32-arbitrated`,
`golden/clef-omni-av-f32-arbitrated`) predate this policy and do not qualify.

**Audio frontend contract.** Sample bytes, padding, masks, token counts and positions must be
exactly equal. Features may differ from the pinned FP32 processor by at most `5e-5` absolute and
`5e-6` relative L2 per clip, with separate FFT, power, mel and log stage checks. The tower and
decoder use the ordinary gates.

**Independent FP64 lane.** Authorized on 2026-10-11 as a separate method and the acceptance
reference since 2026-10-16. `ref/oracle_fp64.py` (and `ref/oracle_fp64_bounded.py`, which streams validated
captures in 1 MiB chunks instead of holding a record in RAM) computes the decoder, routing, media
towers, rotary and joint head in CPU FP64 from the pinned encoder's tokens, with the processor's
FP32 image/audio preprocessing and the release loader's BF16 weight rounding before upcasting.
`ref/omni_fp64_policy.py` fixes the method: no native inputs, online softmax over 128-query by
2,048-key tiles, paged weights, per-layer captures and runtime dtype checks against downcasts;
`tests/test_omni_fp64.py` compares it with dense FP64 calculations. Qualification against it
requires an explicit `--reference fp64`, with the same limits. FP64 agreement cannot establish
superiority to hosted serving.

```sh
lockf -t 1 -k golden/.gpu.lock .venv/bin/python -B tools/memory_guard.py \
  --max-gib 6 --reserve-gib 32 --timeout 600 \
  --report golden/omni-bounded-reproduction-memory.json -- \
  .venv/bin/python -B ref/oracle_fp64_bounded.py model-omni --corpus text \
  --only r003 r013 --dump-last 0 --output golden/omni-bounded-reproduction
```

**Resources.** One GPU job at a time under `golden/.gpu.lock`, inside
`tools/memory_guard.py --max-gib 24 --reserve-gib 32 --report REPORT -- COMMAND`; the verifier's
default `--max-memory-gib 24` sets a 16 GiB MPS allocator limit and an 8 GiB resident-weight
limit. The guard stops the process group on a budget violation and leaves incomplete evidence;
sampling cannot stop one oversized allocation between checks. The machine's 128 GiB is total RAM,
not a model budget.

**Runner.** `tests/test_omni_qualification.py` maps `text`, `images`, `video`, `audio`,
`audio_video` and `long` to golden and native-capture directories. It builds the CLI, server,
host tool and media worker first, requires captures from that exact CLI and GGUF, records all
four executable hashes, and cannot produce a complete report from missing records, stale
captures, failed commands or interrupted runs. `bench/omni_latency.py` times repeated full
requests after model load.

## Results

### Against FP64

Acceptance run of 2026-10-16 on the working default (engine commit `32b5ef8`), with
`tests/test_parity.py --bound-by` per FP64 reference and the matching FP32 blocked-attention
reference as the bound. Maxima are the script's four-decimal summaries. Evidence:
`golden/omni-fp64-bounded-20261016/`.

| FP64 reference | Requests / questions | Max logit / probability error vs FP64 | Beyond fixed limits: native / FP32 reference | FP64-bounded |
|---|---:|---:|---:|---|
| Text up to 1,024 tokens | 19 / 40 | 0.0082 / 0.0004 | 0 / 3 | Pass |
| `r019`, 2,231 tokens | 1 / 2 | 0.0146 / 0.0037 | 1 / 1 | Pass |
| `r020`, 8,069 tokens | 1 / 2 | 0.0345 / 0.0038 | 1 / 1 | Pass |
| Images | 5 / 15 | 0.1291 / 0.0019 | 1 / 1 | Pass |
| Video | 5 / 10 | 0.0430 / 0.0061 | 3 / 4 | Pass |
| Audio | 6 / 12 | below 0.0001 / below 0.0001 | 0 / 1 | Pass |
| Audio/video | 5 / 10 | 0.0868 / 0.0065 | 2 / 6 | Pass |
| All | 42 / 91 | | 8 / 17 | Pass |

Every decision equals FP64's. The eight native questions beyond the fixed limits are
`r019/service`, `r020/service`, `ov001/brightness`, `ot002/moving`, `ot003/moving`,
`ot003/direction`, `oav000/sound` and `oav000/audible`; on each, the FP32 reference's error on the
exceeded measure is at least as large.

### Working default against FP32 (diagnostic)

Independent FP32 (blocked attention) results, 2026-10-11, no arbitration. Logit/probability maxima
are CLI summaries rounded to four places; acceptance uses unrounded values. Evidence:
`golden/evaluation-standard-2026-10-10/independent-short-summary.json`.

| Corpus | Exact tokens | Exact decisions | Failed residual checks | Worst relative L2 | Max logit error | Max probability error |
|---|---:|---:|---:|---:|---:|---:|
| Text | 22/22 | 46/46 | 119/1,100 | 0.102507 | 0.1150 | 0.0066 |
| Images | 5/5 | 15/15 | 2/250 | 0.0125241 | 0.0084 | 0.0003 |
| Video | 5/5 | 10/10 | 20/250 | 0.0331750 | 0.0405 | 0.0114 |
| Audio | 6/6 | 12/12 | 17/300 | 0.105272 | 0.0480 | 0.0022 |
| Video with audio | 5/5 | 10/10 | 49/250 | 0.130862 | 0.0566 | 0.0199 |

The blocked/SDPA control on identical inputs, compared per question, not as a tolerance:

| Corpus | Max residual relative L2 | Max logit difference | Max probability difference | Exact decisions |
|---|---:|---:|---:|---:|
| Text | 0.114956 | 0.0888067 | 0.0129141 | 46/46 |
| Images | 0 | 0 | 0 | 15/15 |
| Video | 0 | 0 | 0 | 10/10 |
| Audio | 2.60676e-6 | 6.31809e-6 | 3.87182e-7 | 12/12 |
| Video with audio | 2.51449e-6 | 5.24521e-6 | 1.30369e-6 | 10/10 |

The twelve failed final-output checks are `r003/urgency`, `r003/outage`, `r013/mentions_audio`,
`r020/service`, `r021/incident`, `r021/service`, `ot002/direction` (probability error 0.0113844),
`oa005/audible` (0.00220567), and `oav001/sound`, `oav001/audible`, `oav002/sound`,
`oav002/audible` (largest 0.0198715). The text control's nonzero spread is on `r019`–`r021`. Only
`r020/service`'s probability error (0.00660822) lies inside its control spread (0.0129141); its
logit error (0.091547) does not, and every other failure lies outside its spread. Images pass all
final-output limits but fail `ov001/layer.47` (0.0125241) and `ov001/final_norm` (0.0112986).

Routing differences over the captured rows (diagnostic):

| Corpus | Ordered-choice differences / rows | Selected-set differences / rows |
|---|---:|---:|
| Text | 17,603/341,712 | 8,147/341,712 |
| Images | 83/98,064 | 22/98,064 |
| Video | 1,607/133,248 | 687/133,248 |
| Audio | 2,668/93,888 | 1,594/93,888 |
| Video with audio | 10,593/143,184 | 5,564/143,184 |

The native frontend plus poisoned audio tower passes 262 stage checks across six records against
the reference, worst relative L2 `8.37845481e-5`. The 64k native capture completes and reproduces
an earlier full capture byte for byte; there is no 64k reference to compare it with.

### FP32-compatible build

The working default's differences from the FP32 reference come from small same-input
differences in many operators (for example `4.69578e-7` relative L2 in vision patch projection,
`3.01668e-7` in vision GELU and about `3.95e-7` in decoder attention), which near-tied top-8
routing decisions then amplify ([below](#where-the-error-comes-from)). Matching the reference
therefore required matching its arithmetic, not making it more accurate. The build reproduces
these pinned PyTorch MPS operations: the router projection (`MPSNDArrayMatrixMultiplication`) and
its softmax/top-8/renormalization; vision patch contraction order, learned-position
interpolation, LayerNorm reduction, rotary product rounding, blocked attention and GELU; decoder
Q/K preparation and RMS (MPSGraph mean, `1 / precise::sqrt`, separate stores, the pinned default
rotary table), blocked causal attention with the reference's tensor strides, SiLU, and the
different arithmetic MPS uses for experts with fewer than 16 rows in the complete record; audio
convolutions and GELU, sinusoid table, LayerNorm, short projections and attention; and a frontend
using PocketFFT, a dense FP32 mel product and a SLEEF-derived `log10`. RMS and key normalization
reduce at each record's own shape, because MPS mean rounding depends on the reduction's outer
size; without that, packing `r003` with `r019` moved `r003`'s logits by up to 0.1150105. The
component experiments and their individual effects are in
[rejected experiments](rejected-experiments.md#fp32-compatibility-components).

From step 2 of the lineage below, every build that completed the short corpora passes all 93 final
questions and all 2,150 saved stages; 2,050 stages are byte-exact to the reference, and the 100
inexact ones belong to the mixed-audio records `oa003` and `oav002`. The fused-validation
candidate's maxima are 0.005391403 relative L2, 0.01115802 logit error and 0.000916208
probability error. All 43 encodings and all 93 decisions are unchanged from the working default.
Four long text records are compared only on their saved final 512 rows; unsaved prefix rows and
64k are unqualified.

The merged build (step 10) and the fused-validation candidate (step 18) each also pass forward and
reversed NaN-poisoned batch-2 packing of seven mixed-media requests (14 exact responses), three
injected CLI command failures followed by six exact recoveries, and bounded HTTP methods including
a calibrated media failure, recovery and uncached fallback (merged build: four jobs, six methods,
37 HTTP responses; fused: four methods, 27 HTTP responses, seven CLI references). The fused patch
clean-builds with portable expert regressions for all 15 native row kernels. The merged build and
the dense-head merge (step 13) pass bounded Flash and 27B checks: three requests and eight
questions each, exact bytes, packed poison and sampled `r000` residuals.

Lineage. Steps 1–13 were timed in resident processes with two warmups and three measured requests
each: steps 1–7 in order bracketed by the working default ("production"), steps 8–13 in ABBA order
against the parent, or against the working default for steps 10 and 13. Steps 14–18 used frozen
ABBA jobs of four resident processes with five warmups and five measured requests each (criteria in
[rejected experiments](rejected-experiments.md#fp32-compatible-speed-candidates)). Every timed
response is byte-exact to its build's expected output.

| # | Step | 298 tokens | 2,231 tokens | Evidence (`golden/`) |
|---:|---|---|---|---|
| 1 | Vision and decoder components through expert occupancy, graph-based (audio components followed) | 6,138.53 ms, 25.00x production | 7,081.39 ms, 4.73x | `omni-fp32-expert-block-20261013/` |
| 2 | Expert expansion, product and check encoded in the serial encoder instead of three waited submissions | 2.8793x production | 1.7733x | `omni-fp32-expert-qualification-20261014/` |
| 3 | RMS square, mean and scale in one MPS command wrapper | 2.7538x | 1.7380x | `omni-fp32-rms-async-20261014/` |
| 4 | Attention callbacks combined per chunk; every native root checked | 2.5696x | 1.5587x | `omni-fp32-decoder-root-status-20261014/` |
| 5 | RMS incoming wait deferred | 2.93% lower than 4 | 0.77% lower | `omni-fp32-rms-overlap-20261014/` |
| 6 | Tiny-expert products on BF16 weights directly, sticky weight scan (plus audio pipeline-registry and encoder-cleanup fixes) | 5.35% lower than 5 | 1.58% lower | `omni-fp32-expert-mixed-20261014/` |
| 7 | Clean authored-source build | 2.3795x production | 1.4451x | `omni-source-qualification-20261014/` |
| 8 | Skip native expert products the compatibility path overwrites | 0.93742 of 7 | 0.98384 | `omni-expert-skip-qualification-20261014/` |
| 9 | Gate packing, SiLU and up multiplication in one callback | 2.63% lower than 8 | 3.77% lower | `omni-expert-activation-qualification-20261014/` |
| 10 | Merged with the current dense head (`from-root.patch`) | 2.08173x working CLI | 1.49480x | `omni-working-integration-20261014/` |
| 11 | Expert metadata built from router IDs on the host, removing a readback wait | 3.67% lower than the qualified build | 3.10% lower | `omni-expert-host-layout-20261015/` |
| 12 | Router graph helpers chained, five commands to two | 3.90% lower than 11 | 3.98% lower | `omni-router-chain-20261015/` |
| 13 | Merged with the newer dense head (42-path patch) | 475.161 vs default 245.038 ms | 2,176.001 vs 1,523.676 ms | `omni-current-head-default-timing-20261015/` |
| 14 | Expert chunks of 2,048 rows | neutral | 5.053% lower than 13 | `omni-chunk2048-service-20261015/` |
| 15 | Attention score scale and causal mask applied in the native product epilogue | neutral | 6.071% lower than 14 | `omni-attention-score-epilogue-corpus-20261016/` |
| 16 | Probability–value blocks read directly from softmax and GQA values | unchanged (475.283 vs 476.086 ms) | 6.735% lower than 15 | `omni-attention-direct-pv-corpus-20261016/` |
| 17 | Tiny-expert array views built directly; eight-word BF16 scanner | 5.3220% lower than 16 | within regression bound | `omni-tiny-direct-array-corpus-20261016/` |
| 18 | Native dot4-FMA tiny-expert products reproducing MPS bit for bit, with fused weight/output validation | 14.232% lower than 17 | 3.747% lower (pooled) | `omni-tiny-fused-validation-corpus-20261016/` |

Steps 11 and 12 passed the corpus, packing and recovery checks; their HTTP checks ran after the
merge in step 13. Step 16 also has a default-relative measurement: 473.879 versus 247.248 ms at
298 tokens and 1,827.819 versus 1,556.233 ms at 2,231 tokens. Step 18's MPS reproduction rests on
a CPU reconstruction: a four-term FMA cell, separately rounded outer accumulation and an adjacent
32-lane reduction match all 1,326,080 retained MPS words, where separate multiplication differs in
381,526 up and 491,988 down words despite passing local tolerances. An earlier native recipe that
passed local tolerances failed the model gate on `r019` (25 of 50 stages), so bitwise reproduction
of MPS is a requirement for this build, not an optimization detail.

### FP64 controls

The independent FP64 lane separates native error from FP32-reference drift. Against it, the
working default fails these checks; `r021` (16,344 tokens) has no FP64 reference.

| FP64 inventory | Stages / questions | Working default: failed stages / questions |
|---|---:|---:|
| Images, 5 requests | 250 / 15 | 17 / 1 |
| Video, 5 requests | 250 / 10 | 29 / 3 |
| Audio, 6 requests (plus 262 tower stages) | 300 / 12 | 0 / 0 (tower 0) |
| Audio/video, 5 requests | 250 / 10 | 44 / 2 |
| Text up to 1,024 tokens, 19 requests | 950 / 40 | 1 / 0 (`r009/final_norm`, 0.0116134) |
| `r019`, 2,231 tokens | 50 / 2 | 14 / 1 |
| `r020`, 8,069 tokens | 50 / 2 | 26 / 1 |
| All 42 requests | 2,100 / 91 | 131 / 8 |

The FP32 reference is itself far from FP64 on some questions. On the 91 shared questions it exceeds
the final-output limits against FP64 on 17, while the working default does so on 8; the SDPA control
exceeds the blocked reference's limits on 4 of 93. For eleven questions the FP32/FP64 separation
exceeds twice a limit, so by the triangle inequality no output can pass against both references.
On nine selected controls the FP32 reference fails 100 of 450 residual and 11 of 21 final checks
against FP64, versus 44 and 4 for the working default. The FP32-compatible build was not compared
with the FP64 references. A private FP64-oriented prototype passed 41 FP64 controls but failed
`r020` and 18 of 93 FP32 checks; it is recorded in
[rejected experiments](rejected-experiments.md#fp64-precision-prototypes).

Evidence: `golden/omni-short-text-qualification-20261013/`,
`golden/omni-mid-text-qualification-20261013/`, `golden/omni-8k-text-qualification-20261013/`,
`golden/omni-reference-agreement-20261013/`, `golden/omni-route-weights-20261013/`; references in
`golden/clef-omni-*-f64-v1/`, `golden/clef-omni-text-bounded-f64-v2/` and
`golden/clef-omni-text-8k-f64-v2/`.

### Where the error comes from

- **Near-tied routing amplifies small differences.** The first changed expert set on `r003` is
  layer 7/token 127: the FP32 reference prefers expert 75 over 68 by `3.81470e-6`, independent FP64
  prefers 68, and native chooses 68. On `r013` the reference's two candidate scores tie exactly.
  Replaying all 48 layers of `r003` and `r013` in CPU FP64 from token IDs, native passes every
  limit (worst relative L2 `0.00379202`), while the MPS FP32 reference reaches `0.0953938`.
  Changing only the reference's layer-7 router dot product to FP64 removes all 15 of `r003`'s FP64
  residual failures and both question failures.
- **Late divergence is propagated state.** At eight sampled late rows, native and reference agree
  within `2.55469e-6` relative L2 given identical incoming states, while the original different
  inputs reach `1.37993`. All 450 local stage checks against CPU FP64 pass on the sampled rows.
- **Media inputs alone can cause the first split.** For the MPS-router profile's first changed expert
  on `ot001`, native visual embeddings and DeepStack features together reproduce it under an FP32
  decoder while either alone does not. Under an FP64 decoder, native vision alone reproduces
  `oav004`'s first split and native audio does not. No single defective operator was isolated in
  either case.
- **Matching operators locally was not sufficient.** The MPS router projection alone reproduces
  all 38,144 saved reference scores exactly yet raises FP32 final-output failures from 12 to 14,
  because upstream states still differ. Each component helped only once its inputs matched too;
  the order is in [rejected experiments](rejected-experiments.md#fp32-compatibility-components).
- **More accurate arithmetic does not approach the FP32 gate.** Compensated, more-accurate
  operators moved the working default toward FP64 and away from the FP32 reference, and several
  made whole requests worse by flipping an earlier near-tie
  ([rejected experiments](rejected-experiments.md#fp64-precision-prototypes)).

## Latency

The working default's native and hosted timings on identical HTTP bodies are in
[performance history](performance-history.md#current-native-and-hosted-timings); that table was
measured on 2026-10-10, before the expert-chunk, tile-dispatch and parallel-assignment changes of
2026-10-12/13 ([performance history](performance-history.md#omni-expert-chunks-2026-10-12)).
Hosted was faster than native Omni there on short text, images, audio and audio/video. Stock
PyTorch could not load Omni within the 32 GiB system reserve, so stock latency is unmeasured
([stock controls](performance-history.md#stock-pytorch-controls-2026-10-10)). An earlier custom
resident loader with eager experts, not stock PyTorch, was 9.1x–17.3x slower than native on five
fixtures.

The FP32-compatible builds' default-relative measurements:

| Build | 298 tokens, default / build | 2,231 tokens, default / build | Evidence (`golden/`) |
|---|---:|---:|---|
| Step 10, merged build | 246.259 / 512.645 ms | 1,507.743 / 2,253.775 ms | `omni-working-integration-20261014/` |
| Step 13, dense-head merge | 245.038 / 475.161 ms | 1,523.676 / 2,176.001 ms | `omni-current-head-default-timing-20261015/` |
| Step 16, direct PV | 247.248 / 473.879 ms | 1,556.233 / 1,827.819 ms | `omni-direct-pv-default-timing-v2-20261016/` |
| Step 18, fused validation | 244.688 / 382.566 ms | 1,573.819 / 1,784.752 ms | `omni-tiny-fused-validation-default-timing-20261016/` |

The default-relative screens for steps 13, 16 and 18 required both paired ratios below 1 and the
pooled ratio below 0.99. Every build is slower than the working default in both pairs on both
requests. The 2,231-token arms drift between jobs and within them (default arm medians 1,520.198
to 1,578.365 ms in the fused job). An identical-binary control on that request failed its own
stability criterion, so differences of a few percent are not resolvable with this protocol. Profiling of the fused build puts expert-containing commands at
193.154 ms of a 276.981 ms observed GPU interval union at 298 tokens, and expert and attention
commands at 765.445 and 480.498 ms of 1,594.456 ms at 2,231 tokens; dense expert projections are
the largest sampled category
([diagnostics](rejected-experiments.md#timing-method-and-attribution-diagnostics)).

## Hosted and labeled agreement

These measure the working default. Fresh core agreement with hosted Omni is 86/93 decisions
(Wilson 95% 85.27–96.31%), with identical answers across two hosted passes; video agrees on 6/10.
All seven differences remain unexplained because the stock BF16 control could not load
([fresh comparison](hosted-comparison.md#fresh-core-comparison-2026-10-10)). Matching token counts
do not establish identical inputs: hosted token IDs and revision are not exposed, and a number
probe supports a hosted serialization difference behind two records' two-token count differences.

On exact-body paired labeled subsets, native/hosted accuracy is 355/385 versus 356/385 on BANKING77
(decision agreement 383/385) and 327/332 versus 326/332 on CLINC150+OOS (331/332), exact McNemar
p = 1.0 for both, with no significant difference
([expanded results](hosted-comparison.md#expanded-omni-labeled-results-2026-10-11)). The earlier
stock BF16 CLINC advantage on Flash came from PyTorch 2.11's incorrect implicit causal attention on
MPS; the corrected eager comparison does not address Omni's hosted differences or its numerical
failures ([corrected comparison](hosted-comparison.md#corrected-bf16-comparison-2026-10-11)).

On the 20 longest selected API-Bank failures from the Decision Index suite, hosted repeats native's
wrong choice on 17. On the other three, the independent FP32 reference also makes native's choice,
and so does a bounded BF16 control on the one case it covers. These are selected failures, not an
accuracy estimate, and no faulty native operation was isolated. Evidence:
`golden/decision-index-suite/omni-gap-debug/REPORT.md`.

## Not done

- **Long context.** Acceptance covers inputs up to 8,069 tokens. The FP64 reference for `r021`
  (16,344 tokens) is being generated; nothing near the 64,000-token default context has an FP64
  or FP32 reference (the 64k FP32 run was stopped after layer 14). Long text references keep only
  their last 512 rows, so earlier rows of the four long records are not compared.
- **FP32-compatible branches.** Unmerged and no longer needed for acceptance. Merging one would
  also mean accepting C++ PocketFFT and SLEEF-derived code, absent from `THIRD_PARTY_NOTICES.md`.
- **Hosted differences.** Seven core disagreements, four of them on video, remain unexplained.
  The stock BF16 control and stock PyTorch latency are unmeasured because the stock loader crossed
  the 32 GiB memory reserve and was stopped.
- **Coverage.** Encoded video is checked with one generated H.264/AAC MOV; general codec coverage
  and decoded-pixel parity of encoded uploads are not established.
- **Working-default speed.** Indirect expert dispatch measured 0.96–3.17% lower latency on the
  working default's fixtures and was deferred without full qualification
  ([rejected experiments](rejected-experiments.md#fp64-precision-prototypes)).
- **Caches.** Omni prefix and template reuse is unsupported; such requests run in full.
