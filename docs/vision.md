# Vision

How images enter the engine, what was checked against the reference, and what it costs. Numbers
are from the one tested M5 Max (128 GB), one GPU job at a time, as the other reports.

## Pipeline

The reference (`joint_schema_model.py` on `transformers` 5.10.2) takes PIL images, runs
`Qwen2VLImageProcessor`, the Qwen3.5 vision tower (`Qwen3_5VisionModel`) and scatters the
features over the `<|image_pad|>` tokens that follow the template prefix; the decoder then runs
with the 3D rotary positions of `get_rope_index`. The engine does the same from base64:

1. **Decode** (`clef_record.c` `image_bytes`, `clef_image.c`, `third_party/iris`): an `images`
   entry in the hosted API's forms (its input schema, `src/content/workers-ai-models/clef.json`
   in Cloudflare's docs): a `data:` URL (prefix in any case) or `{"content_type", "base64"}`;
   a bare base64 string is an extension. PNG or JPEG by signature, to RGB as PIL's
   `convert("RGB")` would (alpha dropped, palette applied, gray replicated). A declared
   `content_type` must match the signature; `image/webp` is rejected.
2. **Resize**: `smart_resize` to multiples of 32 inside `[min_pixels, max_pixels]`
   (65,536 to 16,777,216 from `processor_config.json`), then PyTorch's native uint8
   antialiased bicubic CPU kernel, ported arithmetic for arithmetic (int16 weights, integer
   accumulation, width pass then height pass). That kernel is what torchvision's `resize` runs
   on Apple Silicon for uint8 input; it differs from PIL by ±1 in a few hundred pixels per image
   and from the float path by much more, so the port is specific to it.
3. **Normalize and patch**: `(x - 127.5) / 127.5` in f32, 16x16 patches in 2x2 merge-window
   order with the frame duplicated for the two temporal taps, `[n_patch][1536]`.
4. **Tower** (`metal/clef.metal` "vision tower", `clef_metal.m` `encode_vision`): patch
   embedding as one GEMM, bilinearly resampled learned positions, 27 blocks of LayerNorm,
   bidirectional attention with 2D rotary embedding, LayerNorm, GELU MLP; merger LayerNorm,
   four patches per row, GELU, projection to the backbone width. One dispatch sequence per
   image, independent of the batch.
5. **Backbone**: the features replace the placeholder rows in the embedding (`embed` kernel),
   and `attn_prep*` apply interleaved M-RoPE: pair `i` of the 32 rotary pairs turns by the
   token's temporal, row or column position for `i mod 3 = 0, 1, 2`, which is the plain RoPE for
   text tokens. `clef_record_positions` reproduces `get_rope_index`: an image's tokens share its
   start on the temporal axis and take their merged-grid row and column, and the text after it
   continues from start + max(rows, columns).

## Host parity

Byte parity with the reference, `make test`:

| Check | Test | Result |
|---|---|---|
| Decode, resize, patches, position interpolation | `tests/test_image.py` (60 random PNG/JPEG images of every color type, up- and downscaling, `media_kwargs` bounds, plus three DEFLATE variants) | 63/63 byte-identical to PIL, torchvision and the processor's `pixel_values` |
| Request encoding with images (ids, spans, image runs, 3D positions) | `tests/test_record.py` (41 image requests among 3,081) | 3,081/3,081 |
| Rejections | both | lone `media_kwargs` bound, other processor arguments, videos, non-list images, bad base64, truncated/unsupported images, placeholder text in parity mode |

Divergences from the reference (it answers or behaves differently; the engine errors):
videos; `media_kwargs` other than `min_pixels` and `max_pixels`, or only one of the two (the
processor silently ignores a lone bound); PNGs that are 16-bit, sub-8-bit grayscale or
interlaced; JPEGs that are CMYK, 12-bit or arithmetic-coded, or whose first component is
sampled below another (Y 1x1 under Cb 2x2, which libjpeg accepts); images over 64 MiB encoded,
16,384 pixels on a side or 64 megapixels decoded; a literal `<|image_pad|>` in request text in
parity mode (the reference raises on the placeholder count); the server's per-request image
and per-image token limits. EXIF orientation is ignored, as the reference ignores it.

Two input checks were added to the vendored JPEG decoder after an automated review of this
change (2026-10-07), each confirmed with a crafted file under AddressSanitizer against the
unmodified copy and rejected cleanly after: a progressive scan header with Se > 63 drove the
coefficient index past the zigzag table (global over-read feeding a heap write), and a legal
luma-under-chroma sampling layout read past the luma plane in the color conversion. Both
files are regression cases in `tests/test_image.py`.

## Numerical parity

Corpus: `ref/corpus_vision.py`, 16 requests / 41 questions with 1 to 3 images each, 16 to
1,024 image tokens, PNG and JPEG, gray and alpha sources, JSON and ~1K-token text states.
Golden data: `ref/oracle.py --corpus vision --dtype float32`. `tests/test_parity.py` with its
default limits (exact token ids, argmax agreement, max |Δlogit| ≤ 0.05, max |Δp| ≤ 0.002,
per-layer relative L2 ≤ 0.01).

**Initial FP32 implementation, 2026-10-07** (before the optimizations below)

| Model, vision tower operands | ids | argmax | max \|Δlogit\| | max \|Δp\| | `v000` image rows rel. L2 | `v000` final norm rel. L2 |
|---|---|---|---|---|---|---|
| **clef-flash, f32** | 16/16 | **41/41** | **0.0126** | **0.0016** | 1.1e-5 | 1.1e-3 |
| clef-flash, 16-bit (`CLEF_VIS_F32=0`, FP16 per `ACT_VIS`) | 16/16 | 41/41 | 0.0724 | 0.0025 | 2.9e-3 | 5.6e-3 |
| **Clef 27B, f32**, `golden/clef-vision-f32` (streamed, `--safe-attn`) | 16/16 | **41/41** | **0.0066** | **0.0010** | 4.0e-6 | 1.4e-3 |

The tower runs 27 layers before any text is read, so operand rounding there compounds: with
FP16 operands two of 41 questions exceed the limits (`v009/outdoors`, the 1,024-token image, and
`v011/subject`). With f32 operands the image feature rows are 1.1e-5 from the FP32 oracle and
every question is within the limits, for the GEMM cost below. The text corpus is unchanged by
the vision changes: clef-flash 22/22 ids, 46/46, max |Δp| 0.0002 (`golden/clef-flash-f32`);
27B 22/22, 46/46, max |Δp| 0.0011 (`golden/clef-f32`, as before).

The streamed FP32 oracle (`ref/oracle_f32_stream.py --corpus vision`), which the 27B needs, is
bitwise identical to the full FP32 oracle on clef-flash for the vision corpus (41/41 questions,
all dumped layers and vision features).

Invariants on the vision corpus (`tests/test_batch.sh`, `tests/test_poison.sh`): logits
bitwise identical between `--batch 1` and `--batch 8`, and under NaN-poisoned buffers, in
both operand modes and with the reference attention kernel. `--template-cache` and
`--prefix-cache` give the plain logits on every image request (keyed image identity and feature
reuse are described below). `tests/test_server_images.py`: HTTP bytes equal the CLI's for every
request, with and without a cache key.

## Cost

`./clef --time`, warm, single request, f32 tower operands, 2026-10-07:

| Request | Image | Image tokens | Total tokens | clef-flash | Clef 27B |
|---|---|---|---|---|---|
| `v001` (webcam frame) | 336x252 JPEG | 88 | 373 | 145 ms | 435 ms |
| `v009` | 1024x1024 JPEG | 1,024 | 1,363 | 883 ms | 2,053 ms |

The same requests on the final build of this change (after the attention, GEMM and fusion work
below), measured the same way on 2026-10-07 as the median of the last three of four back-to-back
passes: `v001` 134 ms on clef-flash and 416 ms on the 27B; `v009` 773 ms and 1,857 ms. The
sections below hold the paired measurements that justify each step.

The 27B's text-only 346-token request measures 439 ms in the README's table, so the webcam
frame costs it roughly what its 88 extra tokens would as text; the tower is the same size on both
models (the merger projects to 5,120 instead of 4,096).

Serialized profiles (`CLEF_PROFILE=1`, not latency): on `v001` the tower's attention takes
5 ms and its GEMMs about 19 ms more with f32 operands than with 16-bit ones; on `v009` the
attention takes 277 ms (4,096 patches, quadratic) and the f32 GEMMs 165 ms more. The first
attention kernel (one lane per key, `vis_attention`, kept as the `CLEF_ATTN_REF=1` reference)
took 42 ms and 4.8 s respectively; the simdgroup-matrix kernel (`vis_attention_mma`, after
`attention_fa`) replaced it.

An image costs its tokens in the backbone like text: one token per 32x32 pixels of the
resized image, up to the reference's 16,384 (16.7 megapixels). The server rejects, by
default, more than 4 images per request or an image above 1,024 tokens
(`--max-images`, `--max-image-tokens`), naming the `media_kwargs.max_pixels` that would fit,
rather than downscaling silently; the reference's own downscaling is available through
`media_kwargs` with both bounds. clef-webcam's 336-pixel frames are about 90 to 110 tokens.

## FP32 attention optimization, 2026-10-07

Images with at least 2,048 patches (512 image tokens) now use `vis_attention_mpp`.
It keeps Q, K, V and softmax probabilities in FP32, uses 32-query/128-key Metal matrix
primitive tiles, and retains the online softmax and bounded scratch memory. Smaller images
keep `vis_attention_mma`; `CLEF_VIS_MPP=0` restores that kernel for every image. The reference
attention override still takes precedence. No image is resized differently and no tokens are
removed. The wider tile changes floating-point reduction order, so output is not promised to
be bit-identical to the old kernel.

The selection depends only on the individual image's patch count. Q/K/V each have 128 slack
rows, rewritten before attention, so the final tile cannot consume stale or poisoned values.
This adds 81 KiB to vision scratch compared with the original three 32-row tails; it does not
allocate a patches-squared score matrix.

`make test-vision-attention` runs the production kernels on 16 patch counts, including ragged
32/128-row tails and both sides of the 2,048-patch dispatch boundary. It compares sampled
outputs against float64 on flat and sharp softmax distributions, checks every output for
finiteness, verifies output guards and FP16/BF16 rounding, and checks per-record overflow
flag isolation. The sharpest sampled case puts both FP32 implementations near 3.5e-5 maximum
absolute error; the test caps it at 5e-5 and permits at most a 5% increase in aggregate squared
error over the original kernel. Full-model parity limits are unchanged.

After stopping the three old localhost keep-warm servers, paired standalone kernel
measurements at 4,096 patches were 9.91 ms for the original kernel and 7.25 ms for MPP,
about 27% less time per layer. This is the attention kernel only, on the tested M5 Max.

`bench/vision_latency.py` alternates the old and new kernel in separate CLI processes, drops
two warmup requests per process, checks token counts and response repeatability, and records
five timed requests per arm per round. It refuses other active Clef processes by default;
`--allow-other-engines` explicitly marks results provisional. CLI timings cover inference and
response construction, excluding image decoding, request encoding and model loading.

The initial isolated two-round comparison measured large-image `v009` medians of
991.2 → 861.6 ms on Flash and 2,121.0 → 2,055.1 ms on 27B. The Flash control drifted between
rounds, so a separate three-round large-image confirmation is recorded alongside it. That
confirmation measured **1022.4 → 926.0 ms (9.4% less time)**,
with fifteen samples per arm. The two-round 27B result is **3.1% less time**. Small
`v001` images use the same kernel in both arms and have identical response bytes; their timing
variation illustrates the noise floor. These are warm measurements from one M5 Max, not a
claim about other devices or cold starts. Raw samples are in `latency-isolated.json` and
`latency-flash-confirm.json` in the evidence directory. The earlier `latency.json` is retained
but marked provisional because the three old servers were still running.

Validation: both vision corpora still pass 41/41 questions against FP32 with the default
logit/probability limits, including the original small-image layer dump. Maximum probability
error remains 0.0016 on Flash and 0.0010 on 27B. Vision batch invariance and NaN poisoning pass
in both operand modes, and reference-attention poisoning passes. The text corpora pass 46/46
on both models. `make test`, `make test-errors` and the HTTP image suite pass.

Only `v009` changes logits in the vision corpus. Mean absolute probability error across all
options changes from 6.906e-5 to 6.867e-5 on Flash and from 1.449e-4 to 1.499e-4 on 27B.
For `v009` itself, maximum probability error is 6.62e-5 → 6.89e-5 on Flash and
1.57e-4 → 3.84e-4 on 27B. Maximum logit error there is 0.000619 → 0.003949 and
0.001050 → 0.001492 respectively. All remain below the unchanged limits, but the 27B's
probability error increases slightly; unchanged decisions do not imply identical scores.
The per-request metrics and raw logits are in `quality.json` and `quality-*.jsonl`.

The additional `v009` layer references are generated separately under the evidence directory
from the verified pinned snapshots, using `oracle_f32_stream.py --corpus vision --only v009`
and `--safe-attn` for 27B. Both fresh references reproduce the existing golden logits bitwise.
Both large-image layer comparisons pass: embedding relative L2 is 2.03e-5 on Flash and
1.50e-5 on 27B; final-norm relative L2 falls from 0.00223 to 0.00203 on Flash and from
0.00977 to 0.00847 on 27B, below the existing 0.01 limit. The existing corpus and golden data are unchanged.

The raw kernel, parity, error, HTTP and timing logs are under
`golden/vision-opt-2026-10-07/`. Compensated half-operand attention was also tried, but regressed
some small shapes and was slower than the selected FP32 path on the large image. Materializing
bounded score slabs was slower as well. Neither experiment is enabled or retained in the
production source.

## GEMM and bias optimization, 2026-10-07

The default now compensates **non-residual** vision projections (patch, QKV, MLP up,
and merger). The initial implementation materialized each producer's FP32 output before
`vis_split_gemm` wrote `hi = half(x)` and
`lo = half((x - float(hi)) * 2048)` into two planes; `vis_gemm_comp` computes the two
FP32-accumulating products against the unchanged BF16 weights and combines them as
`hi_product + lo_product / 2048`. Scaling preserves small residuals across FP16's limited
exponent range. This is not bitwise FP32 arithmetic.

Residual projections remain direct FP32, using a 16x128 tile with the same per-element
reductions as the previous 32x128 tile. Applying compensation to these projections too
failed the Flash probability gate (0.00235 > 0.002), so that variant was rejected.
A three-plane variant passed parity but gave little full-request improvement and was
also rejected. Compensated tiles use 32 rows below 1,024 rows and 64 above, selected per
image. Tests require exact output across tile shapes and packed row offsets.

The separate residual-bias passes are folded into the following LayerNorm. Each thread
updates its own residual elements before computing the norm; the FP32 rounding and
writeback are unchanged. This removes 54 dispatches per image. Both complete vision
corpora have byte-identical logits before/after this fusion and FP32 tile change, with
the same compensated projections enabled.

`CLEF_VIS_COMP=0` restores direct FP32 products throughout the tower. Clearing `ACT_VIS`
in `CLEF_ACT_F16` or running a BF16 retry also disables compensation. The split reuses
the record-local saturation/overflow path: NaNs and magnitudes past the FP16 limit flag
only that record and write finite halves. The retry uses the original FP32 tower operands.
`CLEF_VIS_F32=0` retains the older plain 16-bit mode. No image is resized differently.
The `CLEF_VIS_COMP=0` control reproduces the saved starting binary's logits byte-for-byte
over the full vision corpus on both models.

The initial scratch was one reusable pair of half planes sized for the largest image's
largest input, 67.25 MiB at 4,096 patches. The subsequent producer fusion below reduces it
to 24 MiB. Capacity growth remains atomic on allocation failure.
The keep-warm buffer list is sized from its activation and KV arrays, including the new
scratch and all 16 KV slots; an ASan regression covers the maximum populated list.
Reinstating the old 64-slot bound makes that regression fail with a stack-buffer-overflow;
the derived bound passes.

On the same M5 Max, alternating saved starting binary / candidate, three rounds with
three warmups and seven measured requests per arm (21 samples per result):

| Model / image | Starting build, ms | New build, ms | Median reduction |
|---|---:|---:|---:|
| Flash, `v001` webcam | 142.4 | 132.4 | 7.0% |
| Flash, `v009` large image | 884.5 | 815.8 | 7.8% |
| 27B, `v001` webcam | 441.6 | 435.6 | 1.4% |
| 27B, `v009` large image | 1872.3 | 1811.9 | 3.2% |

The starting build already includes the attention optimization above. These are warm CLI
inference times, excluding image decoding, request encoding, loading, response serialization
and HTTP. The small 27B change is within run-to-run variation. Its large-image
rounds improved by 0.3–4.4%, so treat 3.2% as this run's median, not a guaranteed gain.
Evidence: `golden/vision-gemm-2026-10-07/latency-final.json` includes samples, request and
binary hashes, responses, and interference preflight. `timed-clef` preserves the measured
binary; the subsequent rebuild only adjusts keep-warm capacity and comments.
Reproduce against a saved binary with:

```sh
.venv/bin/python -B bench/vision_latency.py golden/vision-gemm-latency.json \
  --baseline-binary /path/to/saved-clef --rounds 3 --samples 7 --warmup 3
```

Both models retain 41/41 vision decisions and 46/46 text decisions against FP32. Host
tokenization is exact. Numerical error across every vision question:

| Metric | Flash starting → new | 27B starting → new |
|---|---:|---:|
| Mean absolute probability error | 6.86653e-5 → 6.35583e-5 | 1.49865e-4 → 1.39946e-4 |
| Maximum absolute probability error | 0.00161825 → 0.00089441 | 0.00100868 → 0.00099687 |
| Mean absolute logit error | 0.00106286 → 0.00123315 | 0.00096335 → 0.00084136 |
| Maximum absolute logit error | 0.01259571 → 0.01722068 | 0.00663567 → 0.00702220 |

Probability error improves on average, but this is **not a uniform numerical improvement**:
Flash logit error increases. Every unchanged acceptance limit still passes. The fresh `v009`
layer references also pass: image-embedding relative L2 is 2.26e-5 / 1.19e-5 (Flash / 27B),
and final-norm relative L2 is 0.00204 / 0.00953. The latter 27B value is close to the existing
0.01 limit. Per-request errors and raw outputs are in `quality.json` and `quality-*.jsonl`
in the same evidence directory; no oracle or acceptance threshold was changed.

Verification: `make test`, `make test-errors`, `make test-vision-attention`, the vision GEMM
kernel tests (121 shapes, FP64 samples, exact tile/packing parity, poisoned tails, finite
saturation and record-local flags), bias/LayerNorm fusion in all three output types,
allocation-failure recovery and ASan keep-warm coverage; vision batch and poison tests
on both models in default and plain 16-bit modes; vision-only forced overflow and dump
retry parity on both models; full text/vision parity and large-image layer dumps; HTTP
image/limit/cache checks on Flash and reference-attention poison checks. Commands:

```sh
make test-vision-gemm
.venv/bin/python -B tests/test_vision_gemm.py gguf/clef-flash.gguf golden/clef-flash-vision-f32/requests.jsonl
```

## Exact fusion follow-up, 2026-10-07

Norm and merger-GELU producers now write the compensated half planes directly. They retain
the standalone conversion's FP32 arithmetic, saturation and record-local overflow flags.
Only patch embedding needs separate split scratch, reducing that allocation from 67.25 to
24 MiB at 4,096 patches. This removes 56 conversion dispatches per image. QKV/RoPE preparation
also clears the attention kernel's masked tail, removing another 81 fill dispatches.
Neither change changes an output bit on either complete vision corpus.

Controlled repeated-dispatch probes measured producer/split fusion at 1.08–1.35x and
QKV/tail fusion at 1.01–1.87x across the tested shapes. Complete fresh-request timings were
noisy: an apparent regression in blocked runs did not persist in the interleaved baseline,
cache-only and fused comparison. No reliable whole-request speedup is claimed for these
fusions. Their verified benefits are fewer dispatches and 43.25 MiB less scratch at 4,096
patches. See `split-timing.log`, `tail-timing.log` and `paired-stage.json` in the evidence directory.

The producer regression compares every half-plane bit, residual writeback and overflow flag
against the separate producer and split, including large values, low forced limits and ragged
widths. The attention regression starts with NaN-filled Q/K/V storage, verifies the exact
cleared tail and untouched guard, then checks FP64 samples and both output precisions.

The corresponding ledger, source snapshots, raw outputs and timing runs are under
`golden/vision-exhaust-2026-10-07/`. Further GEMM/GELU fusion, thirteen backbone tile/order
variants and an attention accumulator-initialization shortcut were screened. Kernel parity
passed, but full-request latency or broader timing checks rejected them. Their experimental
builds remain only in the ignored evidence directory, not the shipped dispatch.

## Caches

Images sit right after the 36 template tokens, so the fixed-template entry still serves image
requests normally. Keyed caching (`--prefix-cache` on the CLI; `--prefix-cache-mb` plus
`X-Clef-Prefix-Cache` over HTTP) also retains merged image features and exact owned copies of
the canonical patch bytes. Reuse requires identical pixel bytes, image order, token placement,
patch dimensions and grid geometry within the same engine instance. Token placeholders alone
never establish a match.

A hit skips the vision tower even if changed text prevents backbone prefix reuse, or the last
checkpoint lies inside an image's token run. Feature rows retain the full image-list layout;
only suffix tokens read them. Changing images invalidates the backbone prefix too. A failed
or overflowing pass is never eligible for reuse, and the BF16 retry follows the ordinary path.
CPU patch copies and GPU features both count toward admission and eviction. Image requests can
cache at a 32-token boundary; text retains the 128-token minimum. Cache keys are client-visible
identifiers, not authentication: assign them at an authenticating proxy as described in README.

`tests/test_vision_cache.py` checks exact raw logits on every corpus fill/hit, same-shape pixel
changes, image order and resize changes, altered text/questions, text/image transitions, poisoned
scratch and overflow invalidation. `tests/test_server_images.py` checks HTTP response parity,
key isolation, replacement and rejection of entries above budget. Cache allocation failure,
recovery, accounting and keep-warm are covered in `tests/test_vision_buffers.m`.

Warm CLI timing while alternating two question schemas on the same image, two reversed-order
rounds, four warmups and eight measured requests per arm and round:

| Model / image | Uncached median | Cache-hit median | Speedup |
|---|---:|---:|---:|
| Flash, `v001` | 126.1 ms | 85.7 ms | 1.47x |
| Flash, `v009` | 797.2 ms | 96.1 ms | 8.30x |
| 27B, `v001` | 419.1 ms | 310.4 ms | 1.35x |
| 27B, `v009` | 1890.6 ms | 321.9 ms | 5.87x |

Every raw-logit response is byte-identical across the uncached and cached sequences. This
measures inference, excluding decode, request encoding, model loading, response serialization
and HTTP. Initial fills still compute the complete request and allocate/copy retained
state; these hit numbers do not describe fill latency or new frames. Evidence:
`golden/vision-exhaust-2026-10-07/latency-cache.json` and `cache-timed-clef`.
Reproduce with `.venv/bin/python -B bench/vision_cache_latency.py golden/vision-cache-latency.json`.

The final build was also measured through loopback HTTP, including image decoding, record
encoding and response serialization. The same warm server alternated plain and keyed requests,
reversing arm order each pair and alternating two question schemas. After four warmup pairs,
eighteen measured pairs gave these medians; every paired response was byte-identical:

| Model / image | Plain HTTP | Cache-hit HTTP | Speedup |
|---|---:|---:|---:|
| Flash, `v001` | 150.7 ms | 102.0 ms | 1.48x |
| Flash, `v009` | 837.8 ms | 115.8 ms | 7.23x |
| 27B, `v001` | 438.9 ms | 312.1 ms | 1.41x |
| 27B, `v009` | 1921.1 ms | 358.7 ms | 5.36x |

Model startup and client-side JSON serialization are excluded. These are warm hit figures,
not new-frame or initial-fill latency. Desktop load changed absolute timings between runs;
the arms above were interleaved within each run. The script, raw samples, server binary and
its hash are `http_latency.py`, `latency-http.json`, `final-timed-clef-server` and
`final-binaries.json` in the evidence directory.

## CPU image processing follow-up, 2026-10-07

Both new images and cache hits still need decoding and canonical patches. The host path now
uses table lookup for base64, system zlib for PNG inflate/CRC, transfers already-RGB decoder
buffers instead of copying them, skips identity-resize allocation, shares bicubic weights
across RGB, and looks up the 256 exact FP32 normalized values. Still-image temporal planes
are copied contiguously. Bicubic accumulation order, resize dimensions and every patch bit
are preserved; no image resolution or model precision is reduced.

PNG output remains bounded by validated dimensions. Inflate must consume one complete stream
and produce exactly the required bytes; checksum errors, dictionaries, truncated/extra output
and trailing compressed data are rejected. Base64's bulk path still rejects padding outside
the final quartet and all non-alphabet bytes. Tests cover every invalid byte/position, PNG
stored/fixed/dynamic blocks, malformed streams and corrupt IHDR/IDAT/IEND checksums.

CPU-only timings on the M5 Max, median of three rounds, each with four warmups and thirty
measured samples. Arm order reverses between rounds; totals include base64, decode and
preprocessing, excluding JSON parsing, inference and response serialization:

| Image | Previous total | Current total | Speedup |
|---|---:|---:|---:|
| 1024x1024 RGB gradient PNG | 16.67 ms | 6.49 ms | 2.57x |
| 1024x1024 RGB random PNG | 62.74 ms | 8.99 ms | 6.98x |
| 1920x1080 RGBA random PNG, resized to the 1M-pixel budget | 174.87 ms | 33.84 ms | 5.17x |
| Corpus `v001`, JPEG | 1.84 ms | 1.53 ms | 1.20x |
| Corpus `v009`, JPEG | 12.63 ms | 9.98 ms | 1.27x |
| Corpus `v013`, small PNG with upscaling | 0.67 ms | 0.43 ms | 1.56x |

These are host-stage gains, not multipliers for complete inference. The original large vision
fixture is JPEG, so it benefits from preprocessing but not the PNG decoder changes. Raw
per-stage timings, fixtures and before/after binaries are in `latency-host.json`, `host-*.log`
and `image-*-requests.jsonl` under `golden/vision-exhaust-2026-10-07/`. Reproduce the stage
measurement with `make image-bench` and `./image-bench REQUESTS.jsonl`.

## Follow-up validation

Both models retain all 41/41 vision decisions and 46/46 text decisions against the FP32
oracles, using unchanged tolerances. Every raw vision logit is byte-identical to the build
before this follow-up. The retained changes improve speed or memory use; they make no
evaluation-quality claim beyond preserving those outputs.

The GPU/cache changes passed batch invariance, NaN poison, vision-only overflow/BF16 retry,
reference-attention and narrow-activation checks, allocation failure/recovery, engine ownership,
cache budget/LRU/key isolation and HTTP parity. The subsequent host changes passed `make test`,
`make test-errors`, another complete vision/cache/HTTP run on both models, 203/203 additional
image cases against Pillow/torchvision, and 103/103 image cases under ASan. Base64's exhaustive
invalid-byte/position checks also passed ASan. The two command/result ledgers are
`validation.json` and `validation-host-final.json` in the evidence directory.

## Not done

- Videos (`videos`, `<|video_pad|>`, the video processor).
- WebP (the hosted API accepts it); 16-bit, low-bit and interlaced PNGs; CMYK JPEGs.
- Attention remains quadratic in patches. The large-image FP32 MPP path reduces its cost;
  further tensor-unit work remains open.
