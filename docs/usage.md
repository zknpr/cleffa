# Usage

Request inputs and their limits, the HTTP server, the two opt-in caches and the CLI options.
The README has the quick start.

## Images

A request's `images` list takes the two forms of Workers AI's `@cf/cloudflare/clef` input schema
(`src/content/workers-ai-models/clef.json` in Cloudflare's docs): a `data:` URL, or an object
`{"content_type": "image/png" | "image/jpeg", "base64": "..."}`. A bare base64 string is accepted
too, as an extension. The images go before the state, as the reference places them:

```sh
img=$(base64 -i receipt.png)
printf '{"model":"clef","state":{"task":"Review the attached receipt."},"images":[{"content_type":"image/png","base64":"%s"}],"questions":{"legible":{"type":"noul","instructions":"Is the receipt total legible?"}}}\n' "$img" \
  | ./clef -m gguf/clef-flash.gguf
```

The hosted schema also lists `image/webp`, which this build rejects (PNG and JPEG only), and it
caps each image at 4 MiB and 16 megapixels with 4 images per request; the engine's own caps are
in [the security notes](security.md#untrusted-inputs). A `content_type`, or a data URL media type, that contradicts the file's
signature is an error; a data URL without a media type is decoded by its signature.

The engine reproduces the reference's image processor byte for byte (PIL decoding, `smart_resize`,
torchvision's uint8 bicubic resize, normalization, patch layout) and runs the Qwen3.5 vision
tower on the GPU. An image costs one backbone token per 32x32 pixels of the resized image, from
64 tokens up to the reference's 16,384 (16.7 megapixels), so a raw phone photo is thousands of
tokens: shrink images before sending, or pass the reference's own bounds as
`"media_kwargs": {"min_pixels": 65536, "max_pixels": 1048576}` (both are required; the processor
silently ignores one alone, which the engine rejects instead). The server accepts at most 4 images
per request and 1,024 tokens per image by default (`--max-images`, `--max-image-tokens`; 0 = the
reference's limits) and rejects an image over the limit with the `max_pixels` that would fit. It
also refuses a source image above 16,777,216 pixels at its header, before decoding
(`--max-image-pixels`; that is the processor's own `max_pixels`, and the hosted API's cap is
16 megapixels), and decodes or holds at most 8 image requests at a time (`--max-image-requests`);
the rest wait with only their request body in memory. [The security notes](security.md#untrusted-inputs) give the reasoning.
Clients should therefore shrink large photos before sending, as the hosted API requires too (it
also caps an encoded image at 4 MiB). A 24-megapixel phone photo (5712x4284, 6.4 MB as JPEG) is
refused by the pixel limit, and its 8.5 MB of base64 is over the default `--max-body` of 8 MiB as
well. A local deployment that wants originals can raise `--max-image-pixels` and `--max-body`
together. Measured on Flash, eight concurrent 24-megapixel JPEGs peaked 1.20 GB above idle against
1.08 GB for eight 16.8-megapixel ones. Compressible PNGs are the expensive case: at the decoders'
64-megapixel cap a decode holds about 0.5 GB, so the slot count then bounds about 4 GB.
Measured parity, costs and the unsupported formats are in [docs/vision.md](vision.md).

## Video

A request's `videos` list holds ordered frame arrays, `{"frames": [...], "fps": 24}`. Each frame
takes the same PNG/JPEG forms as `images`, and `fps` is the source frame rate, in `(0, 1000]`.
Frames must share dimensions, at least 32 pixels a side.

After `make media`, MP4/MOV uploads are accepted as `{"content_type":"video/mp4","base64":"..."}`
or `data:video/quicktime;base64,...`: by the CLI unless it gets `--no-encoded-video`, by the
server only with `--allow-encoded-video`. The native `clef-media` worker decodes in a bounded child
process without an OS sandbox ([security](security.md#untrusted-inputs)); keep it beside the
executable. Omni includes the soundtrack; dense models use video only.
URLs and file paths are rejected. [Decoder limits and isolation](video.md#encoded-uploads)
apply before the normal frame, token and audio limits.

For callers that already use frame arrays, `tools/video_request.py` converts a local
clip with FFmpeg. Put `model`, `state` and
`questions` in a request template, then convert:

```sh
.venv/bin/python -B tools/video_request.py clip.mp4 --request request.json > video.jsonl
./clef -m gguf/clef-flash.gguf video.jsonl
# Or submit the same converted request to the server:
curl localhost:8080/v1/systemone -H 'Content-Type: application/json' --data-binary @video.jsonl
```

The engine reproduces the pinned video processor: it samples frames uniformly at a target 2 FPS
(at least four when available, at most 768), resizes them against one pixel budget for the whole
video, pairs them for the tower's two temporal taps (repeating an unpaired last frame), and writes
each pair's timestamp, placeholder tokens and M-RoPE positions as the reference does. All images
precede all videos, which precede the state. Existing vision GGUFs already contain the weights and
the video token id; no reconversion is needed.

Sampling and sizing controls live under `media_kwargs.videos_kwargs`:

```json
{"videos_kwargs":{"fps":2,"size":{"shortest_edge":4096,"longest_edge":1048576}}}
```

`size` is a pixel budget across sampled frames, with both positive bounds required; the reference
defaults are 4,096 and 25,165,824. `num_frames` selects a count instead of `fps` (it clears the
default sampling FPS, as `fps: null` does in the reference), and `do_sample_frames: false` keeps
every supplied frame. Other processor controls are rejected. Top-level
`media_kwargs.min_pixels`/`max_pixels` continue to control still images only.

A frame array without metadata may describe up to 18,000 source frames or 600 seconds. Presampled
frames carry their source positions instead: `frame_indices` (strictly increasing integers within
`total_num_frames`, one per frame) with the original average `fps` and `do_sample_frames: false`,
so their timestamps match a request carrying every frame. The engine rejects indexed frames when
sampling is enabled. Encoded uploads require the optional native codec worker.

The server allows one video, 32 sampled frames and 1,024 tokens per video by default
(`--max-videos`, `--max-video-frames`, `--max-video-tokens`; 0 uses the processor and context
bounds, as the CLI does by default). These limits reject requests rather than reduce resolution
or sampling rate, and are checked before resizing or allocating patches. `--max-image-pixels` also
bounds each source frame, and `--max-image-requests` counts video requests too. Prefix and template
caches serve videos, with exact patch identity and timestamp-token matching.

The converter accepts H.264, HEVC, ProRes and MJPEG in MP4/MOV with exactly one video track. Audio
is ignored; stored orientation is preserved and HDR is not tone-mapped. Clips need a known frame
count and FPS, at most 18,000 frames, 600 seconds and 16,777,216 pixels per frame. By default it
samples 2 FPS, at most 32 frames, into an 8 MiB request, matching the server's frame and body
defaults, and exceeding a cap fails explicitly: use `--num-frames` or `--fps` to reduce sampling,
and `--max-frames` / `--max-body` only when the receiving server allows larger requests. Set the
model's pixel budget through `media_kwargs.videos_kwargs.size` in the template. Its output keeps
the source `frame_indices`, `total_num_frames` and FPS. The converter's isolation is under
[the security notes](security.md#untrusted-inputs); sampling details, parity, costs and the converter's internals
are in [docs/video.md](video.md).

## Audio (Clef-Omni)

Omni runs text, images, silent video, audio and video with soundtracks through the same
CLI and HTTP endpoint. Use `./download_models.sh clef-omni` to create its GGUF.
Numerical qualification is incomplete: independent checks fail across all five modalities,
and the 64k qualification remains stopped. See the [current evidence](omni.md#numerical-verification-policy).
Native audio accepts mono 16 kHz WAV or sample arrays. For a local compressed clip:

```sh
.venv/bin/python -B tools/audio_request.py clip.flac --request request.json > audio.jsonl
./clef -m gguf/clef-omni.gguf audio.jsonl
.venv/bin/python -B tools/video_request.py clip.mp4 --model-family qwen3_omni_moe \
  --request request.json > video.jsonl
```

The Omni video helper preserves the soundtrack and emits frames sampled at 2 FPS.
Requests default to four audio clips and 300 seconds of total audio; the HTTP body
cap applies separately. [The Omni report](omni.md) specifies its media ordering,
limits, context, numerical checks and cache behavior. The image and video sections above
describe the Clef and Clef-Flash processors.

## Server

```sh
./clef-server -m gguf/clef-flash.gguf --port 8080          # binds 127.0.0.1 by default
curl -s localhost:8080/v1/systemone -d '{"model":"clef","state":"Our checkout is down",
  "questions":{"outage":{"type":"noul","instructions":"Is a service down?"}}}'
```

`POST /v1/systemone` takes a Jev/SystemOne request body and returns the same response body.
`GET /health` reports status. On startup the server runs one warm-up forward pass, which
faults in the weights and allocates buffers, plus, when the model has a vision tower, one pass
with an image sized to the per-image token limit so the first image request pays for neither the
vision scratch nor the tower's weight pages (`--no-warmup` to skip both). The server micro-batches concurrent requests into one forward pass,
up to `--batch` requests (default 8) and `--batch-tokens` tokens (default 4096). Because results
don't depend on batch composition, this is invisible to clients. A request larger than the token
budget runs alone, so short requests never share a forward pass with a long one
(`tests/test_server_hol.py`). Requests queued *behind* a long one still wait for it: one GPU, no
preemption.

The worker keeps model and activation buffers active while idle using read-only GPU passes
every 500 ms. This removes the measured delay before GPU execution for sparse traffic;
`--no-keep-warm` disables it. On a 248-token request after five seconds idle, median HTTP
latency falls from 220.5 to 104.8 ms on Flash and from 535.7 to 263.3 ms on 27B, with identical
responses. These are three samples per arm on the M5 Max; power cost is not established.
See [the idle-latency report](performance-history.md#keep-warm-for-sparse-traffic-2026-10-04) for scope and validation.

Limits:

| Limit | Value |
|---|---|
| Headers | 16 KiB |
| Body | `--max-body` (default 8 MiB) |
| Connections | `--max-conn` (default 256) |
| Request deadline | 30 s for headers and body together, measured monotonically (not per read) |
| Error-response drain | 1 s and 1 MiB |

Chunked bodies and duplicate `Content-Length` headers are refused. Error responses use a lingering
close so they aren't lost to a TCP reset. `tests/test_server.py` checks HTTP output byte for byte
against the CLI, 32 concurrent clients, and each limit.

## Reuse the fixed prompt for short requests

`--template-cache` reuses the first 32 tokens of Clef's fixed system prompt and
user header, including across unrelated inputs. The first eligible request fills
the entry. It reuses that entry at measured beneficial lengths up to 2,048 full
tokens; other lengths take the ordinary full-input path. All request content
remains included. Above 1,056 tokens, reuse requires that removing 32 prompt
tokens saves a padded 64-row matrix tile. Flash also bypasses shorter lengths
where reuse would switch to a slower tile; the linked report lists the rule.

```sh
./clef -m gguf/clef-flash.gguf --template-cache --time requests.jsonl
./clef-server -m gguf/clef-flash.gguf --template-cache
```

It defaults off. The CLI requires batch one and no dump; choose either
`--template-cache` or `--prefix-cache`. The server can enable both: keyed requests
use their own prefix entries, unkeyed single requests use the fixed prompt, and
packed requests use the ordinary batch path. No client key is needed for the
fixed prompt because only public template state is reused.

This adds up to about 275 MiB on Flash or 504 MiB on 27B, beyond the ordinary
engine buffers and any `--prefix-cache-mb` budget. Its buffers also retain the
current request's suffix as scratch space, overwritten before reuse. In the
measured 346- and 594-token cases, warmed template hits were about 4–6% faster.
At eligible 1–2K padding boundaries, the final paired comparison measured about
3–6% reductions. This does not speed up the first fill or fresh long contexts. See the
[template qualification and timing report](prefix-cache.md#fixed-template-entry).

## Repeated, growing and edited contexts

Prefix caching reuses the backbone state of an identical token prefix. It retains
the full accepted context, including the tokens before the reused boundary. New
questions and the remaining suffix are evaluated against that state. Fresh,
unrelated contexts still need a full forward pass.

Entries keep periodic recurrent-state checkpoints and a checkpoint before an
observed edit. A changed tail or middle can resume from the last matching
checkpoint instead of recomputing the whole request. The saved states belong to
exact token prefixes; no input is omitted or summarized.

For successive JSONL requests in one CLI process:

```sh
./clef -m gguf/clef-flash.gguf --prefix-cache --batch 1 --time requests.jsonl
```

For the server, enable a retained-cache budget and send a key with each request:

```sh
./clef-server -m gguf/clef-flash.gguf --prefix-cache-mb 4096
curl -s localhost:8080/v1/systemone \
  -H 'Content-Type: application/json' -H 'X-Clef-Prefix-Cache: conversation-1' \
  --data-binary @request.json
```

Repeated images use the same options. An entry retains exact processed-image bytes and merged
features, so changing questions or text can reuse the image tower's result. Pixel content,
image order, placement and grid geometry must match. These retained bytes count against the
budget; failed or overflowing passes cannot supply reusable features.

The server cache is off by default. Keyed requests run individually; requests
without a key continue through ordinary micro-batching. A key has 1–64 ASCII
letters, digits, dots, underscores or hyphens; the server validates the header whether or
not caching is enabled, so a malformed key is HTTP 400 either way. An authenticating proxy should
assign keys per tenant and conversation: callers sharing a key can observe one
another's cache hits through latency. Keys are not authentication credentials.

The server retains at most 32 entries and evicts least recently used entries
after a request exceeds the budget. The budget uses MiB and limits the cache's GPU buffers and owned image patches:
a request whose entry would exceed it is served uncached without allocating one, and an
entry that would grow past it is dropped first. It does not bound total process memory, and
capacity growth copies an entry's planes, so a growing entry transiently needs up to twice
its size. Each entry
can retain up to twelve recurrent-state checkpoints, adding 50.25 MiB per
checkpoint on Flash or 149.625 MiB on 27B, plus attention and head-memory buffers.
A 16K Flash entry with eight checkpoints measured about 2.14 GiB, beyond the
engine's own buffers; changing prefixes can allocate more checkpoints. All
retained buffers count toward the budget. Allocation failures fall back to an uncached
server request. See the [checkpoint measurements](prefix-cache.md#recurrent-state-checkpoints)
for shared-prefix speedups, memory costs and qualification.

Measured full-input versus matching-prefix-hit medians on the M5 Max:

| Full input tokens | Flash full → hit | 27B full → hit |
|---:|---:|---:|
| 1,382 | 444 → 130 ms | 1,500 → 413 ms |
| 2,235 | 725 → 113 ms | 2,462 → 355 ms |
| 8,072 | 3,088 → 154 ms | 10,139 → 466 ms |
| 16,347 | 7,376 → 210 ms | 23,830 → 650 ms |

These are warm engine plus CPU-head times, excluding encoding, loading and HTTP,
with four measured calls per arm and length on the earlier FP32-attention build.
Every cached output has the same logit bits as that build's full-input result. Initial fills take roughly a full pass; these hit timings do not
describe first-use latency. Uncached comparisons did not demonstrate a speedup;
the 27B 2,235-token medians were 0.36–1.15% slower in two runs, so a small cost
remains possible. See the [qualification and timing report](prefix-cache.md#exact-fp32-prefix-reuse)
for the full-input comparisons, memory costs and validation scope.

## CLI options

`--batch N` packs N requests per forward pass, `--time` prints latency to stderr,
`--logits` prints raw logits, and `--dump FILE` writes per-layer residuals for the first request.
`--max-images N`, `--max-image-tokens N` and `--max-image-pixels N` apply the server's image
limits (the CLI, as the test harness, defaults to the reference's, and to the decoders' 64-megapixel
cap for source images); `--max-videos N`, `--max-video-frames N` and `--max-video-tokens N` do the
same for videos (default 0, the processor and context bounds). For Omni, `--max-audio N` and
`--max-audio-samples N` bound clips per request (default 4, soundtracks included) and total samples
(default 4,800,000, 300 seconds at 16 kHz) on both the CLI and the server; 0 removes that limit
only. Every clip is padded to the request's longest, so `--max-audio` also bounds frontend memory
([Omni report](omni.md#running-it)).
`--no-encoded-video` makes the CLI refuse MP4/MOV uploads; the server refuses them unless started
with `--allow-encoded-video`.
For diagnostics, `CLEF_PROFILE=1` reports GPU time per kernel category (it serializes the GPU,
so don't use it for latency) and `CLEF_ATTN_REF=1` switches to the simple reference attention
kernel. `CLEF_ATTN_TU=0` selects the prior tiled FP32 attention path; the
default uses compensated tensor-unit attention with FP32 accumulation. The vision tower keeps
FP32 residual GEMMs and compensates non-residual projections; `CLEF_VIS_COMP=0` restores direct
FP32 products throughout. `CLEF_VIS_F32=0` selects plain 16-bit vision operands ([accuracy](accuracy.md#images)).
`CLEF_VIS_GROUP=0` gives every image or video frame pair its own tower dispatches instead of
sharing projections within a record. For Omni, `CLEF_OMNI_ROPE=f32` computes rotary angles in FP32 as
Hugging Face does instead of exactly ([omni.md](omni.md#decoder-and-experts)).
