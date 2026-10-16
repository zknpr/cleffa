# Testing

The command catalog for the dense models, the parity gates, how the FP32 references are
produced and the debug hooks the tests use. `make test` is model-free apart from the Flash
tokenizer files; everything else runs explicitly with a model and a golden directory.

## Commands

The tests need `gguf/clef-flash.gguf` and `model-flash/` (`./download_models.sh clef-flash`). The
parity tests also need golden data from the PyTorch oracles (below).

```sh
make test                                            # host parity (incl. images and video), HTTP write failures, verifier/parity, cache-planner and collector regressions
make test-errors                                     # CLI allocation/output errors, HTTP errors, Metal failures
make test-attention                                  # production attention vs float64 samples, packing and tail guards; no model needed
make test-gemm                                       # production GEMM tile/packing parity, float64 and NaN guards; no model needed
make test-gdn                                        # chunked recurrence, offsets/tails, float64, NaN guards and allocation recovery
.venv/bin/python -B tests/test_tokenizer.py gguf/clef-flash.gguf model-flash
.venv/bin/python -B tests/test_record.py gguf/clef-flash.gguf model-flash model-flash   # includes image requests and 3D positions
.venv/bin/python -B tests/test_image.py                                                  # decode/resize/patch parity with PIL, torchvision, the processor
.venv/bin/python -B ref/oracle.py model-flash --name clef-flash-vision-f32 --dtype float32 --corpus vision   # vision corpus, FP32
.venv/bin/python -B tests/test_parity.py gguf/clef-flash.gguf golden/clef-flash-vision-f32 --dump
tests/test_batch.sh gguf/clef-flash.gguf golden/clef-flash-vision-f32/requests.jsonl    # also with CLEF_VIS_F32=0
tests/test_poison.sh gguf/clef-flash.gguf golden/clef-flash-vision-f32/requests.jsonl   # also with CLEF_VIS_F32=0 and CLEF_ATTN_REF=1
.venv/bin/python -B tests/test_server_images.py gguf/clef-flash.gguf golden/clef-flash-vision-f32/requests.jsonl   # starts its own servers: HTTP bytes, limits, strict mode, caches
.venv/bin/python -B tests/test_vision_cache.py gguf/clef-flash.gguf golden/clef-flash-vision-f32/requests.jsonl    # repeat with clef/27B
.venv/bin/python -B tests/test_video_request.py                                          # MP4/MOV converter; needs ffmpeg/ffprobe, not in make test
.venv/bin/python -B ref/oracle_f32_stream.py model-flash --name clef-flash-video-f32 --corpus video   # video corpus, FP32
.venv/bin/python -B tests/test_parity.py gguf/clef-flash.gguf golden/clef-flash-video-f32 --dump
tests/test_batch.sh gguf/clef-flash.gguf golden/clef-flash-video-f32/requests.jsonl
tests/test_poison.sh gguf/clef-flash.gguf golden/clef-flash-video-f32/requests.jsonl
.venv/bin/python -B tests/test_vision_groups.py gguf/clef-flash.gguf                     # shared vs separate tower dispatches: exact logits, batch, poison, overflow
.venv/bin/python -B tests/test_server_video.py gguf/clef-flash.gguf golden/clef-flash-video-f32/requests.jsonl   # starts its own servers: limits, strict tokens, caches
.venv/bin/python -B ref/oracle.py model-flash --name clef-flash                  # BF16 oracle (MPS)
.venv/bin/python -B ref/oracle.py model-flash --name clef-flash-f32 --dtype float32
.venv/bin/python -B tests/test_parity.py gguf/clef-flash.gguf golden/clef-flash-f32 --dump   # vs FP32: vs BF16 it fails where BF16 is wrong
.venv/bin/python -B tests/compare3.py golden/clef-flash golden/clef-flash-f32 golden/engine_logits.jsonl
tests/test_batch.sh gguf/clef-flash.gguf golden/clef-flash/requests.jsonl    # batch invariance, tenant isolation
tests/test_poison.sh gguf/clef-flash.gguf golden/clef-flash/requests.jsonl   # no read-before-write (NaN-poisoned buffers)
tests/test_grow_fail.sh gguf/clef-flash.gguf golden/clef-flash/requests.jsonl # engine stays correct after a failed buffer growth
tests/test_f16_overflow.sh gguf/clef-flash.gguf golden/clef-flash/requests.jsonl # FP16 overflow -> per-record BF16 rerun
tests/test_gpu_fail.sh gguf/clef-flash.gguf golden/clef-flash/requests.jsonl  # a failed Metal command buffer is an error, never stale results
.venv/bin/python -B tests/test_strict.py gguf/clef-flash.gguf model-flash        # strict mode vs tokenizers reference
.venv/bin/python -B tests/test_truncation.py gguf/clef-flash.gguf model-flash    # over-long state rejected exactly when the reference would cut it
.venv/bin/python -B tests/verify_gguf.py model-flash gguf/clef-flash.gguf        # complete tensor inventory, shapes/types, byte-exactness
# against a running clef-server on PORT (test_server_hol needs --truncate; test_server_retry needs
# CLEF_DEBUG_FAIL_MULTI=1 on the server; test_server_slow needs CLEF_DEBUG_IO_TIMEOUT=2):
.venv/bin/python -B tests/test_server.py PORT gguf/clef-flash.gguf golden/clef-flash/requests.jsonl
.venv/bin/python -B tests/test_server_hol.py PORT golden/clef-flash/requests.jsonl
.venv/bin/python -B tests/test_server_headers.py PORT        # 16 KiB header limit holds for split and keep-alive reads
.venv/bin/python -B tests/test_server_retry.py PORT golden/clef-flash/requests.jsonl gguf/clef-flash.gguf
.venv/bin/python -B tests/test_server_slow.py PORT           # trickling clients are dropped at the request deadline
tests/test_lingering_close.sh gguf/clef-flash.gguf        # starts its own server: an error response drains a half-closed client's body
make test-prefix-attention                           # exact full vs resumed attention, poisoned guards, float64 bounds; no model needed
make test-prefix-model                               # both models: a populated cache entry is refused by another engine and by a reopened one
make test-head-tsan                                  # ThreadSanitizer check of the CPU head's first-use configuration
.venv/bin/python -B tests/test_gemm_dispatch.py gguf/clef-flash.gguf     # exact logits across the Flash 768-1,024 tile boundaries
.venv/bin/python -B tests/test_prefix_cache.py gguf/clef-flash.gguf golden/clef-flash/requests.jsonl    # cache transitions, overflow, bypass, allocation recovery
.venv/bin/python -B tests/test_prefix_public.py gguf/clef-flash.gguf golden/clef-flash/requests.jsonl   # poisoned fill/hit pairs keep the uncached logits
.venv/bin/python -B tests/test_prefix_checkpoints.py gguf/clef-flash.gguf golden/clef-flash/requests.jsonl --attention tu   # tails, edits, shrinkage, eviction; repeat with --attention fp32
.venv/bin/python -B tests/test_template_cache.py gguf/clef-flash.gguf    # template reuse: exact logits, eligibility boundaries, HTTP fallback
.venv/bin/python -B tests/test_server_prefix.py gguf/clef-flash.gguf golden/clef-flash/requests.jsonl   # starts its own servers: HTTP bytes, key isolation, budget eviction
.venv/bin/python -B tests/test_keep_warm.py gguf/clef-flash.gguf         # starts its own servers: idle delay removed, identical responses
```

## Parity gates and GPU tests

The GPU tests assume one GPU job at a time on the machine. Run them for both models before
a release; clef-flash is enough while iterating.

`test_parity.py` requires exact token ids and argmax agreement, finite logits/residuals, maximum
absolute logit error <= 0.05, maximum probability error <= 0.002, and (with `--dump`) per-layer
relative L2 error <= 0.01. These defaults target both models' FP32 references. Override them with
`--max-logit-error`, `--max-prob-error` and `--max-layer-rel-l2` for explicit precision experiments.

## References

The FP32 engineering reference for the 27B, which doesn't fit fully resident in FP32, comes from `ref/oracle_f32_stream.py`.
It upcasts one layer at a time and is bitwise identical to the full FP32 oracle on clef-flash.
27B references need `--safe-attn` ([accuracy](accuracy.md#the-pytorch-reference-on-long-27b-inputs)):

```sh
.venv/bin/python -B ref/oracle.py model --name clef --safe-attn
.venv/bin/python -B ref/oracle_f32_stream.py model --name clef-f32 --safe-attn
.venv/bin/python -B ref/oracle_f32_stream.py model --name clef-vision-f32 --safe-attn --corpus vision
.venv/bin/python -B ref/oracle_f32_stream.py model --name clef-video-f32 --safe-attn --corpus video
.venv/bin/python -B tests/compare3.py golden/clef golden/clef-f32 golden/engine_logits_clef.jsonl
.venv/bin/python -B ref/mps_sdpa_bug.py              # the MPS attention bug, standalone
```

## Debug hooks

Debug hooks for tests:
- `CLEF_DEBUG_POISON=1` fills every activation buffer with NaN before each forward.
- `CLEF_DEBUG_FAIL_MULTI=1` fails every multi-record forward.
- `CLEF_DEBUG_GROW_FAIL_ABOVE=N` fails activation-buffer growth past N tokens.
- `CLEF_DEBUG_F16_LIMIT=X` treats FP16 operands above X as overflow.
- `CLEF_DEBUG_NIL_CMDBUF=N` makes the Nth Metal command-buffer creation fail.
- `CLEF_DEBUG_IO_TIMEOUT=S` shortens the server's request deadline to S seconds.
- `CLEF_STAGE_TIME=1` logs host encoding, commit-to-completion wait, GPU execution and CPU head time per request.
- `CLEF_DEBUG_KEEPWARM_MS=N` sets the idle-pass interval (0 disables); `CLEF_DEBUG_KEEPWARM_NOWEIGHTS=1` omits the weights from idle passes.
- `CLEF_DEBUG_PREFIX_FAIL_ABOVE=N` fails cache-capacity growth above N tokens; `CLEF_DEBUG_PREFIX_CKPT_FAIL=N` fails the Nth new checkpoint allocation.
- `CLEF_DEBUG_HEAD_SPLIT_MIN=N` sets the CPU head's linear-split threshold (0 disables the linear split).
- `CLEF_DEBUG_SIMD_WIDTH_FOR=NAME` makes the named attention pipeline report a 16-lane SIMDgroup at open, which must be refused.

Golden directories written before `ref/oracle_f32_stream.py` produced `encoded.jsonl` can get one
with `ref/write_encoded.py MODEL_DIR GOLDEN_DIR`, which uses the tokenizer only.

Precision diagnostics: `ref/act_stats.py MODEL` (largest Linear input per projection over the
corpus) and `bench/mixed_bench.m` (GEMM rate and error for BF16, FP16 and FP32 activations).

## Clef-Omni

`make test` includes Omni's model-free checks: configuration, positions, audio frontend and
records, converter and verifier fixtures, reference staging, memory guards and qualification
bookkeeping. The model-backed matrix runs one gate at a time through
`tests/test_omni_qualification.py --model gguf/clef-omni.gguf --inventory INVENTORY.json --output DIR`;
its references, gates and current results are in [the Omni report](omni.md#numerical-verification-policy).
