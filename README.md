<p align="center">
  <img src="cleffa.png" alt="cleffa logo" width="220">
</p>

# cleffa — native Metal inference for Cloudflare Clef

A small C + Metal engine for [Cloudflare/clef](https://huggingface.co/Cloudflare/clef) (27B),
[Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash) (9B) and
[Cloudflare/clef-omni](https://huggingface.co/Cloudflare/clef-omni) (30B-A3B mixture of experts).
It returns typed decisions (the Jev/SystemOne request and response format) from one prefill pass,
on Apple Silicon. BF16 weights, never quantized.

Clef and Clef-Flash take text, images and video. Clef-Omni adds audio and video soundtracks; it
passes its accuracy check up to 16,344 tokens, while longer inputs and its speed are still being
worked on ([status](docs/omni.md)).

cleffa is built after [ds4](https://github.com/antirez/ds4) by antirez and the ds4.c authors, and
would not exist without it (see [Acknowledgements](#acknowledgements)). It is an independent
project, not affiliated with or endorsed by Cloudflare.

To see it work, [cleffa showcase](https://github.com/zknpr/cleffa-showcase) is a browser client
for `clef-server`: live camera, video, batch triage and repeated questions on one image, with
every option's probability. Run it next to the server, or open
[cleffa.zknpr.xyz](https://cleffa.zknpr.xyz) and connect your own engine.

## Requirements

- **Tested hardware:** only an M5 Max with 128 GB, on macOS 27.
- **Metal 4:** the GEMMs use Metal 4 tensor ops (MetalPerformancePrimitives). Earlier Apple GPUs
  are untested.
- **Memory:** the weights are mapped whole. clef-flash's GGUF is 19 GB, so plan on 32 GB of RAM
  or more. The 27B's is 55 GB, so plan on 64 GB or more. Long inputs add a few GB of activations
  on top. Clef-Omni's GGUF is 63.7 GB and has only run on the 128 GB machine.
- **Tools:** Xcode command-line tools (`clang`, `xxd`) and macOS's system zlib (`-lz`).
  The optional MP4/MOV worker needs FFmpeg development libraries and pkg-config
  (`brew install ffmpeg pkg-config`, then `make media`). The Python conversion tools
  use `ffmpeg` and `ffprobe` in PATH.
  Python 3.12 is needed only for conversion and tests: `uv` if installed, otherwise `python3.12`,
  plus `requirements.txt`.

## Use

```sh
./download_models.sh clef-flash         # download (pinned), verify, convert -> gguf/clef-flash.gguf
make                                    # clef, clef-server, clef-tool
./clef -m gguf/clef-flash.gguf requests.jsonl      # one SystemOne request per line
echo '{"model":"clef","state":"Checkout is down","questions":{"outage":{"type":"noul"}}}' \
  | ./clef -m gguf/clef-flash.gguf
```

`./download_models.sh` creates `.venv` on first use, downloads the Hugging Face snapshot at a
pinned revision, checks its complete inventory, sizes and hashes offline against checked-in
manifests (`tools/snapshots/`), converts it to one GGUF and checks every tensor of the GGUF
against the safetensors. Use `clef` for the 27B, `clef-omni` for Omni, `all` for all three, and
`--skip-download` to verify and convert a snapshot already in `model/`, `model-flash/` or
`model-omni/`. The reference oracles import Cloudflare's `joint_schema_model.py` from the snapshot,
so verify a snapshot before running them; remove stale `__pycache__` bytecode first and run them
with Python `-B`.

### Inputs

- **Images:** `images` takes `data:` URLs or `{"content_type", "base64"}` objects, as Workers AI's
  schema does, in PNG or JPEG. The engine reproduces the reference's image processor byte for
  byte. An image costs one token per 32x32 pixels of the resized image, from 64 up to 16,384, so
  shrink photos or pass `"media_kwargs": {"min_pixels": ..., "max_pixels": ...}`. The server allows
  4 images and 1,024 tokens per image by default.
- **Video:** `videos` takes frame arrays, `{"frames": [...], "fps": 24}`, or MP4/MOV uploads when
  `make media` has built the `clef-media` worker (the server also needs `--allow-encoded-video`). `tools/video_request.py` converts a local clip
  to frames. Frames are sampled at 2 FPS and paired for the tower; the server allows one video,
  32 sampled frames and 1,024 tokens per video by default.
- **Audio (Omni):** `audio` takes finite mono 16 kHz sample arrays or inline WAV;
  `tools/audio_request.py` converts other files. Omni videos keep their soundtrack. Requests
  default to four clips and 300 seconds of audio.

```sh
img=$(base64 -i receipt.png)
printf '{"model":"clef","state":{"task":"Review the attached receipt."},"images":[{"content_type":"image/png","base64":"%s"}],"questions":{"legible":{"type":"noul","instructions":"Is the receipt total legible?"}}}\n' "$img" \
  | ./clef -m gguf/clef-flash.gguf
```

Request forms, every limit and its reason, and the CLI options are in
[docs/usage.md](docs/usage.md).

### Server

```sh
./clef-server -m gguf/clef-flash.gguf --port 8080          # binds 127.0.0.1 by default
curl -s localhost:8080/v1/systemone -d '{"model":"clef","state":"Our checkout is down",
  "questions":{"outage":{"type":"noul","instructions":"Is a service down?"}}}'
```

`POST /v1/systemone` takes a Jev/SystemOne request body and returns the same response body;
`GET /health` reports status. The server warms up on start, micro-batches concurrent requests
(`--batch` 8, `--batch-tokens` 4096) with results that do not depend on the batch, and keeps the
GPU warm while idle (`--no-keep-warm` disables it). By default it tokenizes request text strictly
and rejects over-long states instead of truncating them (see [Security](#security)). Limits and
flags are in [docs/usage.md](docs/usage.md#server).

### Caches

Both caches are exact and off by default. `--template-cache` reuses the fixed prompt's first 32
tokens for requests up to 2,048 tokens; warmed hits measured about 4–6% faster. `--prefix-cache`
(CLI) or `--prefix-cache-mb N` with an `X-Clef-Prefix-Cache` key (server) reuses the backbone
state of an identical token prefix, with checkpoints so edited or growing contexts resume from
the last match. A matching-prefix hit at 16,347 tokens measured 7,376 → 210 ms on Flash and
23,830 → 650 ms on the 27B. Cache hits are visible in latency, so an authenticating proxy must
assign keys per tenant. See [docs/usage.md](docs/usage.md#repeated-growing-and-edited-contexts)
and [docs/prefix-cache.md](docs/prefix-cache.md).

## Accuracy

Hosted decision agreement and labeled accuracy are the primary metrics. On the 2026-10-10 core
comparison, decisions agree with Cloudflare's hosted API on 80/81 questions for Flash, 80/81 for
the 27B and 86/93 for Omni; on the generated corpora's ground truth, native is right on three of
Omni's seven disagreements and hosted on two ([docs/omni.md](docs/omni.md#hosted-and-labeled-agreement)). On paired labeled text
subsets, a corrected BF16 PyTorch comparison finds no significant accuracy difference from
cleffa ([hosted and labeled results](docs/hosted-comparison.md#fresh-core-comparison-2026-10-10)).

Kernel changes are gated against a local HF FP32 engineering reference over the 22-request /
46-question corpus (`ref/corpus.py`, 146 to 16,347 tokens):

| vs FP32 reference | argmax agreement | mean \|Δp\| | max \|Δp\| |
|---|---|---|---|
| **clef-flash** | **46/46** | **0.00004576** | **0.00023393** |
| HF BF16 on MPS, Flash | 44/46 | 0.0098 | 0.0956 |
| **Clef 27B** | **46/46** | **0.00009400** | **0.00113585** |
| HF BF16 on MPS, 27B | 46/46 | 0.0072 | 0.0437 |

FP32 is an engineering gate, not semantic ground truth, and the Flash BF16 row used PyTorch 2.11's
defective MPS causal attention. Dense backbone matmuls take FP16 activation operands with FP32
accumulation, attention uses compensated FP16 planes, and the residual stream and CPU head stay
FP32. On the vision corpus both models answer 41/41 (max |Δp| 0.0009 and 0.0010); on the
synthetic video corpus 16/16 (0.0002 and 0.0007). Tokenizer, JSON, request encoding, response
text and image/video preprocessing match Python byte for byte. Clef-Omni is accepted differently: its
near-tied expert routing makes fixed FP32 gates unattainable for any FP32 implementation, so it is
checked against an FP64 reference, allowing each question the FP32 reference's own error. It
passes on all 43 requests with a reference, up to 16,344 tokens, every decision equal to FP64
([status](docs/omni.md)). Methodology, precision choices and the
PyTorch reference defects found along the way are in [docs/accuracy.md](docs/accuracy.md).

## Speed

M5 Max (40 GPU cores, 128 GB), one GPU job at a time, engine time including the CPU head and
excluding request encoding, model load and HTTP. Medians of eight warm calls per length on
2026-10-05 ([details](docs/attention.md#adoption-as-the-default)):

| Full input tokens | clef-flash | Clef 27B |
|---:|---:|---:|
| 346 | 122 ms | 439 ms |
| 594 | 210 ms | 715 ms |
| 1,382 | 476 ms | 1,580 ms |
| 2,235 | 771 ms | 2,525 ms |
| 4,510 | 1,645 ms | 5,353 ms |
| 8,072 | 3,237 ms | 10,170 ms |
| 16,347 | 7,370 ms | 22,512 ms |

- **Images and video:** a 336x252 webcam frame with three questions (373 tokens) takes 134 ms on
  clef-flash and 416 ms on the 27B; a 1024x1024 photo (1,363 tokens) 773 ms and 1,857 ms
  ([vision](docs/vision.md), [video](docs/video.md#cost)).
- **Against PyTorch:** on the 2026-10-03 baseline, before every optimization above, corpus
  medians were 116 ms against 199 ms on Flash and 389 ms against 579 ms on the 27B
  ([history](docs/performance-history.md#pytorch-comparison-at-the-baseline-2026-10-03)).
- **Against Cloudflare's hosted API:** the gap on long inputs is not closed
  ([hosted comparison](docs/hosted-comparison.md)).
- **Regimes:** GEMMs dominate GPU time at every length. Long requests heat-soak the GPU (a
  260-token 27B request goes from ~310 ms cool to 360–450 ms after the 8k/16k requests), and the
  server's keep-warm default removes the delay that idle gaps used to add
  ([long-request timing](docs/long-request-timing.md)). Say which regime a number comes from.

## Design

- **`tools/convert.py`, `tools/convert_omni.py`:** HF safetensors → one GGUF, byte-exact BF16
  matrices with per-layer projections concatenated into one GEMM each; `tests/verify_gguf.py`
  checks the result.
- **Host side, byte-exact with Python:** `clef_tok.c` (byte-level BPE), `clef_json.c` (a JSON DOM
  with Python semantics), `clef_record.c` (`encode_record`, `systemone()` validation,
  `systemone_answer`, image/video placeholders and 3D rotary positions), `clef_image.c` (base64,
  ds4's iris PNG/JPEG decoders and the reference image processor), `clef_video.c` (video resize
  geometry), `clef_audio.c` and `clef_record_omni.c` (Omni audio frontend, placement and positions).
- **Codecs out of process:** `clef_encoded_video.c` hands MP4/MOV uploads to the optional
  `clef-media` libav worker; `tools/video_request.py` and `tools/audio_request.py` are offline
  converters.
- **`metal/clef.metal`, `clef_metal.m`:** tensor-op GEMMs (BF16 weights × FP16 activations, f32
  accumulation, batch-invariant tiles), compensated or FP32 attention
  ([docs/attention.md](docs/attention.md)), Gated DeltaNet, norms, SwiGLU and the Qwen3.5 vision
  tower with interleaved M-RoPE ([docs/vision.md](docs/vision.md)).
- **`clef_metal_omni.m`, `clef_metal_audio.m`, `clef_model.c`:** Omni's full-attention MoE decoder
  with DeepStack visual injection, its audio tower, and backbone metadata validation.
- **`clef_head.c`:** the joint schema head in f32 on the CPU via Accelerate; its memory-side K/V
  projections run on the GPU.
- **Batching and caches:** packed variable-length batches, bitwise independent of composition
  (`tests/test_batch.sh`), and the two opt-in caches.

## Tests

```sh
make test          # host byte parity, image/video/audio pipelines, verifiers, planners, collectors; no GPU model run
make test-errors   # CLI allocation/output errors, HTTP errors, Metal failure propagation (clef-flash)
.venv/bin/python -B tests/test_parity.py gguf/clef-flash.gguf golden/clef-flash-f32 --dump   # vs the FP32 reference
tests/test_batch.sh gguf/clef-flash.gguf golden/clef-flash/requests.jsonl    # batch invariance, tenant isolation
tests/test_poison.sh gguf/clef-flash.gguf golden/clef-flash/requests.jsonl   # no read-before-write
```

The tests need `gguf/clef-flash.gguf` and `model-flash/`; parity tests also need golden data from
the PyTorch oracles. The full catalog, the parity gates, how the references are produced and the
debug hooks are in [docs/testing.md](docs/testing.md).

## Security

Requests and GGUF files are treated as untrusted. In summary:

- **Bounded parsing:** every GGUF count, offset, type and shape is checked; JSON depth is capped at
  512 and duplicate-key and NFC handling are linear-time; images, video frames and audio are
  capped before decoding or allocating, and the server bounds concurrently held media requests.
- **The video worker is not sandboxed:** uploaded MP4/MOV clips are decoded by libav in a child
  process with a deadline and size caps, but it runs as the caller's user. The server refuses
  uploads unless started with `--allow-encoded-video`; the CLI, as the test harness, accepts them
  unless given `--no-encoded-video`.
- **Server exposure:** the server binds to localhost. One GPU serves one FIFO queue and keep-alive
  connections can hold slots, so put it behind an authenticating proxy with per-client rate and
  connection limits before exposing it.
- **Prompt-template injection:** the reference turns a literal `<|im_end|>` in request text into a
  control token. The server tokenizes request text strictly by default (`--no-strict` for parity);
  the CLI, as the test harness, defaults to parity (`--strict` to opt in).
- **Silent truncation:** the reference silently drops the end of an over-long state. Both
  interfaces reject such requests by default (`--truncate` restores the reference's behavior).
- **Deliberate divergences:** in a listed set of cases, including media URLs and file paths, which
  Omni's reference fetches or reads, the engine errors where the reference answers.

The measurements behind each bound, the strict-mode and truncation checks and the full divergence
list are in [docs/security.md](docs/security.md).

## Not done yet

- Clef-Omni accuracy references beyond 16,344 tokens (64k has none), and its long-context speed:
  23.7 s at 16,344 tokens, mostly attention ([docs/omni.md](docs/omni.md)).
- WebP ([docs/vision.md](docs/vision.md)).
- Live or unknown-length clips, and video codecs beyond H.264, HEVC, ProRes and MJPEG
  ([docs/video.md](docs/video.md)).
- Full-input latency parity with Cloudflare's hosted API on long inputs
  ([hosted comparison](docs/hosted-comparison.md)).

## Acknowledgements

cleffa exists thanks to [ds4](https://github.com/antirez/ds4) by antirez and the ds4.c authors. It
took ds4's approach whole:
- a small, model-specific C + Metal engine instead of a general framework;
- one GGUF wrapped zero-copy as a Metal buffer;
- correctness established by testing against the reference implementation, not by inspection;
- a script that makes the model setup reproducible.

Parts of the code are adapted from ds4, notably the Qwen pre-tokenizer, and the PNG/JPEG
decoders are ds4's copy of iris ([THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)). The vision
path follows ds4's Qwen3-VL tower structure. Without ds4 there would be no cleffa.

Thanks to Cloudflare for releasing Clef, Clef-Flash and Clef-Omni under Apache-2.0, along with
the reference implementation this engine is tested against.

Like ds4, cleffa was developed with strong assistance from AI coding agents, with a human
directing the goals and design decisions and reviewing the results.

## License

MIT, see [LICENSE](LICENSE). Third-party notices (ds4, Unicode data, the Clef models) are in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The model weights are not part of this repository.
They are Cloudflare's, under Apache-2.0.
