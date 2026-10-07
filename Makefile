CC      ?= clang
CFLAGS  ?= -O2 -g -std=c11 -Wall -Wextra -Wno-unused-parameter -D_DARWIN_C_SOURCE -DACCELERATE_NEW_LAPACK
OBJCFLAGS = $(CFLAGS) -fobjc-arc
# Header dependencies come from the compiler (*.d files): every object rebuilds when any header it
# includes changes, directly or through clef_engine.h. A hand-written list missed the transitive
# ones, so a struct change could link objects built against different layouts (review #4).
DEPFLAGS = -MMD -MP
LDLIBS  += -lz
LDFLAGS ?=
FRAMEWORKS = -framework Metal -framework Foundation -framework Accelerate

HOST_OBJS = clef_gguf.o clef_json.o clef_tok.o clef_record.o clef_image.o

.PHONY: all clean test test-image-diff test-errors test-attention test-vision-attention test-vision-gemm test-gemm test-gdn test-head-tsan test-prefix-attention test-prefix-model unicode

ENGINE_OBJS = clef.o clef_head.o clef_metal.o

all: clef clef-server clef-tool

%.o: %.c
	$(CC) $(CFLAGS) $(DEPFLAGS) -c -o $@ $<

clef_metal_src.inc: metal/clef.metal
	{ printf 'static const char clef_metal_src[] = {'; xxd -i < $< ; printf '};\nstatic const unsigned long clef_metal_src_len = sizeof(clef_metal_src);\n'; } > $@

clef_metal.o: clef_metal.m clef_metal_src.inc
	$(CC) $(OBJCFLAGS) $(DEPFLAGS) -c -o $@ $<

clef: clef_main.o $(ENGINE_OBJS) $(HOST_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS) $(FRAMEWORKS)

clef-server: clef_server.o $(ENGINE_OBJS) $(HOST_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS) $(FRAMEWORKS) -lpthread

clef-tool: tests/clef_tool.o $(HOST_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)

tests/test-base64: tests/test_base64.o clef_image.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)

image-bench: bench/image_latency.c clef_image.o clef_json.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)

attention-bench: bench/attention_bench.m
	$(CC) $(OBJCFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS) -framework Metal -framework Foundation

# Compiles the production Metal source at runtime; no model snapshot is needed.
test-attention: attention-bench
	./attention-bench 1 7 8 9 31 32 33 260 2049
	./attention-bench --reuse 4 --baseline metal/clef.metal 1 7 8 9 15 16 17 31 32 33 260 2049 4095 4096 4097
	./attention-bench --prefetch 64 --baseline metal/clef.metal 1 7 8 9 31 32 33 63 64 65 1023 1024 1025 4095 4096 4097
	./attention-bench --values fp32 --prefetch 64 --baseline metal/clef.metal 31 32 33 63 64 65 1023 1024 1025 2235 8072 16347
	./attention-bench --tu attention_tu --baseline metal/clef.metal 1 7 31 32 33 127 128 129 260 2049 4095 4096 4097
	./attention-bench --values fp32 --tu attention_tu --baseline metal/clef.metal 31 32 33 127 128 129 2235 8072

vision-attention-bench: bench/vision_attention_bench.m
	$(CC) $(OBJCFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS) -framework Metal -framework Foundation

test-vision-attention: vision-attention-bench
	./vision-attention-bench

vision-gemm-bench: bench/vision_gemm_bench.m
	$(CC) $(OBJCFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS) -framework Metal -framework Foundation

tests/test-vision-buffers: tests/test_vision_buffers.m clef_metal.m clef_metal_src.inc clef.o clef_head.o $(HOST_OBJS)
	$(CC) $(OBJCFLAGS) $(DEPFLAGS) -fsanitize=address -o $@ tests/test_vision_buffers.m clef.o clef_head.o $(HOST_OBJS) $(LDFLAGS) $(LDLIBS) $(FRAMEWORKS)

test-vision-gemm: vision-gemm-bench tests/test-vision-buffers
	./vision-gemm-bench
	tests/test-vision-buffers

gemm-tiles: bench/gemm_tiles.m
	$(CC) $(OBJCFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS) -framework Metal -framework Foundation

gdn-bench: bench/gdn_bench.m
	$(CC) $(OBJCFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS) -framework Metal -framework Foundation

tests/test-gdn-buffers: tests/test_gdn_buffers.m clef_metal.m clef_metal_src.inc clef.o clef_head.o $(HOST_OBJS)
	$(CC) $(OBJCFLAGS) $(DEPFLAGS) -o $@ tests/test_gdn_buffers.m clef.o clef_head.o $(HOST_OBJS) $(LDFLAGS) $(LDLIBS) $(FRAMEWORKS)

test-gdn: gdn-bench tests/test-gdn-buffers
	tests/test-gdn-buffers
	./gdn-bench
	./gdn-bench 4095 4096 4097 8191 8192 8193 16347

test-gemm: gemm-tiles
	./gemm-tiles

# clef_unicode.inc is committed and regenerated only on purpose (`make unicode`): its Unicode version
# must match the tokenizer tests, and an automatic rule fired on checkout mtimes in a fresh clone.
unicode:
	.venv/bin/python tools/gen_unicode.py > clef_unicode.inc

tests/test-head-attend: tests/test_head_attend.o $(HOST_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS) -framework Accelerate

tests/test-head-linear: tests/test_head_linear.o $(HOST_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS) -framework Accelerate

tests/test-head-parallel: tests/test_head_parallel.o $(HOST_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS) -framework Accelerate

# Instrument the included head implementation and concurrent first-use regression.
tests/test-head-parallel-tsan: tests/test_head_parallel.c clef_head.c $(HOST_OBJS)
	$(CC) $(CFLAGS) $(DEPFLAGS) -O1 -fsanitize=thread -o $@ tests/test_head_parallel.c $(HOST_OBJS) $(LDFLAGS) $(LDLIBS) -framework Accelerate

test-head-tsan: tests/test-head-parallel-tsan
	tests/test-head-parallel-tsan --init-only

# Decode files from cjpeg, sips, ffmpeg and Pillow with cleffa and with Pillow, pixel for pixel; encoders
# that are not installed are skipped. Not part of make test (it depends on those tools).
test-image-diff: clef-tool
	.venv/bin/python -B tests/test_image_diff.py

# libFuzzer target for the image decode and preprocessing path (tests/fuzz_image.c). Apple's clang
# ships without libFuzzer, so this uses the newest Homebrew LLVM; not part of make test.
FUZZ_CC ?= $(lastword $(sort $(wildcard /opt/homebrew/opt/llvm/bin/clang /opt/homebrew/Cellar/llvm*/*/bin/clang)))
fuzz-image: tests/fuzz_image.c clef_image.c clef_image.h third_party/iris/jpeg.h third_party/iris/png.h
	@test -n "$(FUZZ_CC)" || { echo "fuzz-image: no Homebrew LLVM clang found (brew install llvm, or set FUZZ_CC)"; exit 1; }
	$(FUZZ_CC) -isysroot $$(xcrun --show-sdk-path) -O1 -g -std=c11 -D_DARWIN_C_SOURCE -fsanitize=fuzzer,address,undefined -fno-sanitize-recover=all -o $@ tests/fuzz_image.c -lz

# Differential target: JPEGs the vendored decoder accepts are decoded with libjpeg-turbo too and must
# match pixel for pixel when libjpeg reports no warning (tests/fuzz_jpeg_diff.c). Homebrew jpeg-turbo.
JPEG_TURBO ?= /opt/homebrew/opt/jpeg-turbo
fuzz-jpeg-diff: tests/fuzz_jpeg_diff.c clef_image.c clef_image.h third_party/iris/jpeg.h third_party/iris/png.h
	@test -n "$(FUZZ_CC)" || { echo "fuzz-jpeg-diff: no Homebrew LLVM clang found (brew install llvm, or set FUZZ_CC)"; exit 1; }
	@test -f "$(JPEG_TURBO)/lib/libjpeg.a" || { echo "fuzz-jpeg-diff: no libjpeg.a under $(JPEG_TURBO) (brew install jpeg-turbo, or set JPEG_TURBO)"; exit 1; }
	$(FUZZ_CC) -isysroot $$(xcrun --show-sdk-path) -O1 -g -std=c11 -D_DARWIN_C_SOURCE -I$(JPEG_TURBO)/include -fsanitize=fuzzer,address,undefined -fno-sanitize-recover=all -o $@ tests/fuzz_jpeg_diff.c $(JPEG_TURBO)/lib/libjpeg.a -lz

# Includes clef_image.c; crafted JPEGs under UBSan and ASan (each aborted before the jpeg.h fixes).
tests/test-jpeg-ub: tests/test_jpeg_ub.c clef_image.c clef_image.h third_party/iris/jpeg.h third_party/iris/png.h
	$(CC) $(CFLAGS) $(DEPFLAGS) -fsanitize=undefined,address -fno-sanitize-recover=all -o $@ tests/test_jpeg_ub.c $(LDFLAGS) $(LDLIBS)

# Includes clef.c; UBSan aborts on signed overflow while crafted GGUF headers are validated.
tests/test-vision-config: tests/test_vision_config.c clef.c clef_head.o clef_metal.o $(HOST_OBJS)
	$(CC) $(CFLAGS) $(DEPFLAGS) -fsanitize=undefined -fno-sanitize-recover=undefined -o $@ tests/test_vision_config.c clef_head.o clef_metal.o $(HOST_OBJS) $(LDFLAGS) $(LDLIBS) $(FRAMEWORKS)

tests/test-record-errors: tests/test_record_errors.o $(HOST_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)

tests/test-server-writes: tests/test_server_writes.o $(ENGINE_OBJS) $(HOST_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS) $(FRAMEWORKS) -lpthread

tests/test-cli-alloc: tests/test_cli_alloc.o $(ENGINE_OBJS) $(HOST_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS) $(FRAMEWORKS)

# The test includes the Metal backend to wrap its command queue; do not link clef_metal.o twice.
tests/test-metal-errors: tests/test_metal_errors.m clef_metal.m clef_metal_src.inc clef.o clef_head.o $(HOST_OBJS)
	$(CC) $(OBJCFLAGS) $(DEPFLAGS) -o $@ tests/test_metal_errors.m clef.o clef_head.o $(HOST_OBJS) $(LDFLAGS) $(LDLIBS) $(FRAMEWORKS)

# The ownership test includes clef.c to exercise its engine identity allocator.
# It opens no model or Metal device; do not link clef.o a second time.
tests/test-prefix-owner: tests/test_prefix_owner.o clef_head.o clef_metal.o $(HOST_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS) $(FRAMEWORKS)

# Includes clef.c with mocked GPU state; verifies token identity and failed-pass invalidation.
tests/test-prefix-planner: tests/test_prefix_planner.o clef_head.o clef_metal.o $(HOST_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS) $(FRAMEWORKS)

tests/test-prefix-owner-model: tests/test_prefix_owner_model.o $(ENGINE_OBJS) $(HOST_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS) $(FRAMEWORKS)

# Compares full and resumed attention with exact bits, float64 bounds and poisoned guards.
tests/test-prefix-attention: tests/test_prefix_attention.m bench/attention_bench.m
	$(CC) $(OBJCFLAGS) $(DEPFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS) -framework Metal -framework Foundation

test-prefix-attention: tests/test-prefix-attention
	tests/test-prefix-attention metal/clef.metal metal/clef.metal

# Opens both models: a populated entry is rejected by another engine and by a reopened one.
# The test takes one request; r019 (2,235 tokens) has a state long enough to fill an entry.
test-prefix-model: tests/test-prefix-owner-model
	.venv/bin/python -B -c "import json, sys; [sys.stdout.write(l) for l in open('golden/clef-flash/requests.jsonl') if json.loads(l)['id'] == 'r019']" > tests/owner-model-request.jsonl
	tests/test-prefix-owner-model gguf/clef-flash.gguf gguf/clef.gguf tests/owner-model-request.jsonl
	rm -f tests/owner-model-request.jsonl

test: clef-tool clef-server tests/test-base64 tests/test-vision-config tests/test-jpeg-ub tests/test-prefix-owner tests/test-prefix-planner tests/test-head-attend tests/test-head-linear tests/test-head-parallel tests/test-record-errors tests/test-server-writes
	tests/test-base64
	tests/test-jpeg-ub
	.venv/bin/python -B tests/vision_config_fixture.py gguf/clef-flash.gguf tests/vision-config
	tests/test-vision-config tests/vision-config/vision-ok.gguf ok tests/vision-config/vision-patch-2p24.gguf "unsupported vision shape" tests/vision-config/vision-temporal-2p24.gguf "unsupported vision shape" \
	    tests/vision-config/vision-mrope-16-8-8.gguf "unsupported M-RoPE layout" tests/vision-config/vision-mrope-missing.gguf "unsupported M-RoPE layout" \
	    tests/vision-config/vision-image-id-vocab.gguf "below the vocabulary size" tests/vision-config/vision-video-id-2p31.gguf "below the vocabulary size" \
	    tests/vision-config/vision-4-heads.gguf "unsupported vision shape" \
	    tests/vision-config/vision-processor-bilinear.gguf "does not implement" tests/vision-config/vision-processor-clip-mean.gguf "does not implement" \
	    tests/vision-config/vision-processor-max-pixels.gguf "does not implement" tests/vision-config/vision-processor-missing.gguf "no image_processor"
	rm -rf tests/vision-config
	tests/test-prefix-owner
	tests/test-prefix-planner
	tests/test-head-attend
	tests/test-head-linear
	tests/test-head-parallel
	tests/test-record-errors
	tests/test-server-writes
	.venv/bin/python -B tests/test_json.py
	.venv/bin/python -B tests/test_image.py
	.venv/bin/python -B tests/test_tokenizer.py gguf/clef-flash.gguf model-flash
	.venv/bin/python -B tests/test_verify_gguf.py
	.venv/bin/python -B tests/test_verify_snapshot.py
	.venv/bin/python -B tests/test_parity_checks.py
	.venv/bin/python -B tests/test_prefix_cache_checks.py
	.venv/bin/python -B tests/test_prefix_checkpoint_checks.py
	.venv/bin/python -B tests/test_cloudflare_checkout.py
	.venv/bin/python -B tests/test_cloudflare_corpus.py
	.venv/bin/python -B tests/test_compare_cloudflare.py
	.venv/bin/python -B tests/test_checkout_latency.py
	.venv/bin/python -B tests/test_evidence_archive.py

test-errors: all tests/test-metal-errors tests/test-cli-alloc
	.venv/bin/python -B tests/test_cli_errors.py gguf/clef-flash.gguf
	tests/test-cli-alloc gguf/clef-flash.gguf
	tests/test-metal-errors gguf/clef-flash.gguf

clean:
	rm -f *.o *.d tests/*.o tests/*.d tests/test-vision-buffers tests/test-base64 tests/test-vision-config tests/test-jpeg-ub fuzz-image fuzz-jpeg-diff clef clef-server clef-tool attention-bench vision-attention-bench vision-gemm-bench image-bench gemm-tiles gdn-bench tests/test-gdn-buffers tests/test-head-attend tests/test-head-linear tests/test-head-parallel tests/test-head-parallel-tsan tests/test-record-errors tests/test-metal-errors tests/test-server-writes tests/test-cli-alloc tests/test-prefix-owner tests/test-prefix-planner tests/test-prefix-owner-model tests/test-prefix-attention clef_metal_src.inc

-include $(wildcard *.d tests/*.d)
