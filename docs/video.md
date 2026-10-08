# Video

How video enters the engine as frame arrays, how a local MP4/MOV clip becomes one, what was
checked against the reference, and what it costs. Video reuses the image path described in
[vision.md](vision.md): its decoders, resize kernel, tower and caches. Numbers are from the one
tested M5 Max (128 GB), one GPU job at a time, as the other reports.

## Pipeline

The reference runs the pinned `Qwen3VLVideoProcessor` on decoded frames; its defaults, the same in
both snapshots, are in `clef_video.h`. The engine reproduces it from a frame array:

1. **Input** (`clef_record.c`): a `videos` entry is `{"frames": [...], "fps": F}`. Each frame is a
   PNG or JPEG in any form `images` accepts, read by the same decoders under the same limits.
   `fps` is the source frame rate, in `(0, 1000]`. Without metadata a video may describe up to
   18,000 source frames or 600 seconds. Selected frames must share dimensions, at least 32 pixels
   a side. Presampled input carries `frame_indices` (strictly increasing integers, one per frame,
   within `total_num_frames`) with the original average `fps` and
   `media_kwargs.videos_kwargs.do_sample_frames: false`. These indices, not a nominal sampling
   rate, determine the timestamps. Indexed frames with sampling enabled are refused, and so are
   compressed video uploads, with a pointer to the converter.
2. **Sample**: uniform, following the processor's NumPy `linspace`/`round` rule, at a target
   2 FPS with at least four frames when available and at most 768. `num_frames` selects a count
   instead and clears the default sampling FPS, as `fps: null` does in the reference.
3. **Resize** (`clef_video.c` holds only the geometry): one `smart_resize` against the whole
   video's pixel budget using the sampled frame count, with default bounds 4,096 and 25,165,824
   (`videos_kwargs.size`), then the image path's bicubic kernel. The host keeps frame zero after
   the geometry check and skips identity resize copies.
4. **Pair**: consecutive resized frames form one temporal group, the tower's two temporal taps,
   in channel/temporal/spatial patch order; an odd final frame is duplicated after normalization.
   Each pair becomes an independent `clef_image_ref` entry with a `video_group` and a timestamp.
   Internal `images`/`n_images` therefore include video groups and `n_image_tokens` counts all
   visual tokens, so GPU packing, overflow retries and exact patch caching apply to videos
   unchanged.
5. **Tokens and positions**: each pair gets a one-decimal timestamp and a `<|video_pad|>` run in
   its own wrapper, and the reference keeps an outer video wrapper around each video's pairs;
   token parity needs both. All images precede all videos, which precede the state. M-RoPE
   positions and cache token identity include the timestamps.
6. **Tower**: Qwen3.5 vision attention is already independent for each temporal group, so no
   Metal kernel or weight changes; existing vision GGUFs hold the video token id and need no
   reconversion. Equal-grid pairs or images within one record share GEMM and normalization
   dispatches up to 1,024 patches in total, while position resampling, rotary embedding and
   attention stay per pair. Q/K/V scratch and its tail are reused after each attention dispatch,
   within the existing minimum scratch allocation. Grouping never crosses a record's overflow
   slot, so packing other requests cannot change a record's computation. `CLEF_VIS_GROUP=0`
   restores one dispatch sequence per pair or image.

The server allows one video, 32 sampled frames and 1,024 tokens per video by default
(`--max-videos`, `--max-video-frames`, `--max-video-tokens`; 0 uses the processor and context
bounds, as the CLI does by default). These are independent of the still-image count and token
limits. They refuse rather than reduce resolution or sampling rate, and they are checked, with
the schema and any untruncatable state reserved, before resizing or allocating patches. Source
pixels per frame (`--max-image-pixels`) and concurrent media requests (`--max-image-requests`)
use the existing image limits.

## Converter

`tools/video_request.py` turns one local MP4/MOV into a frame-array request. It runs `ffprobe`
and `ffmpeg` as separate processes. No engine binary links or starts a video decoder, and
`make test` does not need FFmpeg.

- **Input:** H.264, HEVC, ProRes or MJPEG in MP4/MOV, with exactly one video track. Audio is
  ignored, stored orientation preserved, HDR not tone-mapped. Clips need a known frame count and
  FPS, at most 18,000 frames, 600 seconds and 16,777,216 pixels per frame.
- **Isolation:** regular local files only, at most 64 MiB, copied into a private temporary
  directory. MOV external data references are disabled and input protocols restricted to files.
  It runs with the caller's privileges, without an OS sandbox, and the server never invokes it.
  Probe and decode share one 30-second deadline; on timeout the child process is killed and
  reaped. Sampled RGB is capped at 256 MiB.
- **Output:** 2 FPS, at most 32 sampled frames and an 8 MiB request by default, matching the
  server's frame and body defaults; exceeding a cap fails explicitly. It keeps the source
  `frame_indices`, `total_num_frames` and average FPS and disables a second sampling pass, so
  timestamps match a request carrying every frame. The model's pixel budget comes from
  `media_kwargs.videos_kwargs.size` in the request template.
- **Encoding:** FFmpeg emits lossless filter-0 RGB PNGs directly, which spares the C decoder Paeth
  reconstruction, using at most four software threads per stage and fewer for large sources. The
  pipe is checked for dimensions, format, CRCs and exact frame count. If the result exceeds
  `--max-body`, the converter retries the compact Pillow encoding within the same deadline.
  Source pixels, sampled frame indices and timestamps are identical either way.

## Host parity

`tests/test_video.py` (in `make test`) checks 14 cases byte for byte against the processor:
decoded RGB through patch values, token ids, question spans, visual token runs and all three
position axes. Cases cover mixed images and videos, multiple videos, odd counts, single-frame
input, explicit sample counts, spatial downscaling and presampled source indices. It rejects 39
malformed inputs and checks strict-token behavior. Its budget regressions check that the schema
and untruncatable-state reservations refuse an oversized video before its second frame is
decoded, while truncatable state still yields to the video.

`tests/test_video_request.py` (optional; needs ffmpeg/ffprobe) checks five codec/container
combinations covering four codecs, comparing converter output with independently decoded
full-frame requests through C sampling, token encoding and patches. It also checks process
deadlines, output caps, the exact compact fallback at the body limit, PNG pipe validation,
malformed clips and explicit CLI failures. A MOV carrying local-file and loopback HTTP references
must decode only its embedded media, with no network connection.

`tests/test_resize.py` compares the integer NEON resize with torchvision on arbitrary dimensions,
including SIMD tails and edge taps. Host parity and the malformed inputs also pass ASan/UBSan, the
frame host suite passes with FFmpeg absent from PATH, and `otool -L` shows no FFmpeg linkage in
any of the three engine binaries.

Divergences from the reference (it answers or behaves differently; the engine errors):
compressed video; source frame counts, durations, FPS or sample counts outside the limits above;
indexed frames with sampling enabled; `videos_kwargs` other than `fps`, `num_frames`, `size` and
`do_sample_frames`, or a `size` without both positive bounds.

## Numerical parity

Corpus: `ref/corpus_video.py`, eight synthetic requests and sixteen questions, kept separate so
the image and text golden data stay unchanged. Golden data: `ref/oracle_f32_stream.py --corpus
video` from hash-verified snapshots (`golden/clef-flash-video-f32`, and `golden/clef-video-f32`
with `--safe-attn`). `tests/test_parity.py` with its default limits:

| Model | Token IDs | Decisions | Maximum logit error | Maximum probability error |
|---|---:|---:|---:|---:|
| Clef-Flash | 8/8 exact | 16/16 | 0.0012 | 0.0002 |
| Clef 27B | 8/8 exact | 16/16 | 0.0066 | 0.0007 |

The first request's embedding and layer comparisons pass on both models. Batch 1/8 logits and
NaN-poison checks are exact on both models, and video overflow recovery passes.
`tests/test_vision_groups.py` gives identical logits with grouping on and off for 12 requests in
six configurations: batch 1/8, NaN poison, both vision operand modes, reference attention and
forced overflow. On Flash, `tests/test_server_video.py` passes concurrent CLI parity, keyed cache
isolation, count/frame/token limits, strict placeholders and failure recovery. Sixteen cache
transitions under poisoning, changing FPS, pixels, frame order, text and mixed-media layouts, give
the raw logits of ordinary inference with both prefix and template caching. Both models keep
41/41 decisions on the unchanged image corpus, and the image HTTP and cache suites pass. These
checks establish parity on synthetic inputs, not accuracy on real-world video.

**Validation scope.** The model, batch, poison and HTTP results above ran before this branch was
rebased onto the final image-path changes, which altered request encoding for images and videos
(schema and state reservations before patch allocation). On the rebased tree `make test`, 2,071
image differential cases, 3,089 request-encoding cases, the truncation and converter checks and
3,746,819 sanitizer/differential fuzz executions pass; GPU inference and the HTTP suites have not
been rerun on it. The ledger is `golden/video-sync-2026-10-08/validation.json`.

## Cost

Measured 2026-10-08 against the saved pre-change engine and converter, with unchanged weights and
pixel/frame budgets. The changes measured are the direct PNG output, the host copies and NEON
resize, and dispatch grouping. `bench/video_latency.py` keeps both engines warm and alternates
individual baseline and candidate requests in reversed order: ten samples per arm over two rounds.
Each sample covers the file snapshot, probe, codec conversion, frame encoding, JSON transfer, host
processing and inference; it excludes model loading and Python module import. No other inference
job was running. The clips are four-second, 24 FPS H.264 `testsrc2` fixtures, sampled to eight
frames with a 2,097,152-pixel video budget. Every response has byte-identical raw logits.

| Model and clip | Before | After | Latency reduction |
|---|---:|---:|---:|
| Flash, 640×360 | 702.1 ms | 633.5 ms | 9.8% |
| Flash, 1280×720 | 935.4 ms | 717.8 ms | 23.3% |
| 27B, 640×360 | 1,639.0 ms | 1,517.3 ms | 7.4% |
| 27B, 1280×720 | 1,942.2 ms | 1,702.7 ms | 12.3% |

HD conversion alone fell from about 240 ms to 94 ms. Its request grew from 1.70 MB to 2.22 MB,
so these local-pipe measurements do not establish the same gain over a bandwidth-limited network.
For an eight-frame 128×96 frame-array request, warm inference alone fell from 132.9 to 116.9 ms
on Flash and 364.7 to 352.1 ms on 27B. Larger GPU-only measurements drifted during the run,
including slower candidate medians in a case that used no grouping; no larger-image GPU speedup
is claimed. An earlier 4,096-patch grouping experiment showed less than 2% improvement on larger
inputs while expanding scratch, so the retained cap is 1,024.

The CPU benchmark times the actual request encoder. In per-case ABBA runs, the median of two
nine-sample run medians changed from 21.27 to 13.17 ms for eight 640×360 PNG frames, 131.66 to
112.97 ms for sixteen 1280×720 PNG frames, and 261.81 to 235.22 ms for the JPEG equivalent.
A single 720p frame, formerly decoded and preprocessed twice, fell from 18.38 to 6.78 ms.
Desktop timing drift is visible in the raw runs; these are fixture measurements, not universal
bounds.

These changes passed `make test`, 187 arbitrary resize shapes and 163 randomized image cases
against torchvision/Pillow, all 14 video host cases, ASan/UBSan video/JPEG checks, the five
converter combinations and a 181-second sanitizer fuzz run of 460,043 executions without
findings, besides the model checks under Numerical parity. Neither existing golden corpus was
regenerated.

## Evidence and reproduction

Evidence is git-ignored under `golden/`: `video-2026-10-07/` (oracle, parity, batch, poison, HTTP,
linkage and `make test` logs), `video-perf-2026-10-07/` (`clips-final.json`, `gpu-final.json`,
`host-final.json`, the saved baseline binaries and converter, synthetic fixtures and validation
logs) and `video-sync-2026-10-08/` (the post-rebase host ledger).

```sh
.venv/bin/python -B tests/test_video.py
.venv/bin/python -B tests/test_video_request.py  # optional, requires ffmpeg/ffprobe
.venv/bin/python -B ref/oracle_f32_stream.py model-flash --name clef-flash-video-f32 --corpus video
.venv/bin/python -B ref/oracle_f32_stream.py model --name clef-video-f32 --corpus video --safe-attn
.venv/bin/python -B tests/test_parity.py gguf/clef-flash.gguf golden/clef-flash-video-f32 --dump
.venv/bin/python -B tests/test_parity.py gguf/clef.gguf golden/clef-video-f32 --dump
tests/test_batch.sh gguf/clef-flash.gguf golden/clef-flash-video-f32/requests.jsonl
tests/test_poison.sh gguf/clef-flash.gguf golden/clef-flash-video-f32/requests.jsonl
.venv/bin/python -B tests/test_vision_groups.py gguf/clef-flash.gguf
.venv/bin/python -B tests/test_server_video.py gguf/clef-flash.gguf golden/clef-flash-video-f32/requests.jsonl
.venv/bin/python -B tests/test_resize.py
make video-host-bench
./video-host-bench gguf/clef-flash.gguf REQUESTS.jsonl
.venv/bin/python -B bench/video_latency.py RESULTS.json \
  --baseline-engine SAVED_CLEF --baseline-adapter SAVED_VIDEO_REQUEST.py \
  --clips CLIP.mp4 CLIP.mov
```

Input forms, limits and sampling controls for callers are in [README.md](../README.md#video).

## Not done

- Compressed video in the engine or server; live or unknown-length clips in the converter.
- Converter codecs beyond H.264, HEVC, ProRes and MJPEG.
- Accuracy on real video: the corpus is synthetic and unlabeled.
