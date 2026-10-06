CC      ?= clang
CFLAGS  ?= -O2 -g -std=c11 -Wall -Wextra -Wno-unused-parameter -D_DARWIN_C_SOURCE -DACCELERATE_NEW_LAPACK
OBJCFLAGS = $(CFLAGS) -fobjc-arc
# Header dependencies come from the compiler (*.d files): every object rebuilds when any header it
# includes changes, directly or through clef_engine.h. A hand-written list missed the transitive
# ones, so a struct change could link objects built against different layouts (review #4).
DEPFLAGS = -MMD -MP
LDFLAGS ?=
FRAMEWORKS = -framework Metal -framework Foundation -framework Accelerate

HOST_OBJS = clef_gguf.o clef_json.o clef_tok.o clef_record.o

.PHONY: all clean test test-errors test-attention test-gemm test-gdn test-head-tsan test-prefix-attention unicode

ENGINE_OBJS = clef.o clef_head.o clef_metal.o

all: clef clef-server clef-tool

%.o: %.c
	$(CC) $(CFLAGS) $(DEPFLAGS) -c -o $@ $<

clef_metal_src.inc: metal/clef.metal
	{ printf 'static const char clef_metal_src[] = {'; xxd -i < $< ; printf '};\nstatic const unsigned long clef_metal_src_len = sizeof(clef_metal_src);\n'; } > $@

clef_metal.o: clef_metal.m clef_metal_src.inc
	$(CC) $(OBJCFLAGS) $(DEPFLAGS) -c -o $@ $<

clef: clef_main.o $(ENGINE_OBJS) $(HOST_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(FRAMEWORKS)

clef-server: clef_server.o $(ENGINE_OBJS) $(HOST_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(FRAMEWORKS) -lpthread

clef-tool: tests/clef_tool.o $(HOST_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

attention-bench: bench/attention_bench.m
	$(CC) $(OBJCFLAGS) -o $@ $< $(LDFLAGS) -framework Metal -framework Foundation

# Compiles the production Metal source at runtime; no model snapshot is needed.
test-attention: attention-bench
	./attention-bench 1 7 8 9 31 32 33 260 2049
	./attention-bench --reuse 4 --baseline metal/clef.metal 1 7 8 9 15 16 17 31 32 33 260 2049 4095 4096 4097
	./attention-bench --prefetch 64 --baseline metal/clef.metal 1 7 8 9 31 32 33 63 64 65 1023 1024 1025 4095 4096 4097
	./attention-bench --values fp32 --prefetch 64 --baseline metal/clef.metal 31 32 33 63 64 65 1023 1024 1025 2235 8072 16347
	./attention-bench --tu attention_tu --baseline metal/clef.metal 1 7 31 32 33 127 128 129 260 2049 4095 4096 4097
	./attention-bench --values fp32 --tu attention_tu --baseline metal/clef.metal 31 32 33 127 128 129 2235 8072

gemm-tiles: bench/gemm_tiles.m
	$(CC) $(OBJCFLAGS) -o $@ $< $(LDFLAGS) -framework Metal -framework Foundation

gdn-bench: bench/gdn_bench.m
	$(CC) $(OBJCFLAGS) -o $@ $< $(LDFLAGS) -framework Metal -framework Foundation

tests/test-gdn-buffers: tests/test_gdn_buffers.m clef_metal.m clef_metal_src.inc clef.o clef_head.o $(HOST_OBJS)
	$(CC) $(OBJCFLAGS) $(DEPFLAGS) -o $@ tests/test_gdn_buffers.m clef.o clef_head.o $(HOST_OBJS) $(LDFLAGS) $(FRAMEWORKS)

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
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) -framework Accelerate

tests/test-head-linear: tests/test_head_linear.o $(HOST_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) -framework Accelerate

tests/test-head-parallel: tests/test_head_parallel.o $(HOST_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) -framework Accelerate

# Instrument the included head implementation and concurrent first-use regression.
tests/test-head-parallel-tsan: tests/test_head_parallel.c clef_head.c $(HOST_OBJS)
	$(CC) $(CFLAGS) $(DEPFLAGS) -O1 -fsanitize=thread -o $@ tests/test_head_parallel.c $(HOST_OBJS) $(LDFLAGS) -framework Accelerate

test-head-tsan: tests/test-head-parallel-tsan
	tests/test-head-parallel-tsan --init-only

tests/test-record-errors: tests/test_record_errors.o $(HOST_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

tests/test-server-writes: tests/test_server_writes.o $(ENGINE_OBJS) $(HOST_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(FRAMEWORKS) -lpthread

tests/test-cli-alloc: tests/test_cli_alloc.o $(ENGINE_OBJS) $(HOST_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(FRAMEWORKS)

# The test includes the Metal backend to wrap its command queue; do not link clef_metal.o twice.
tests/test-metal-errors: tests/test_metal_errors.m clef_metal.m clef_metal_src.inc clef.o clef_head.o $(HOST_OBJS)
	$(CC) $(OBJCFLAGS) $(DEPFLAGS) -o $@ tests/test_metal_errors.m clef.o clef_head.o $(HOST_OBJS) $(LDFLAGS) $(FRAMEWORKS)

# The ownership test includes clef.c to exercise its engine identity allocator.
# It opens no model or Metal device; do not link clef.o a second time.
tests/test-prefix-owner: tests/test_prefix_owner.o clef_head.o clef_metal.o $(HOST_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(FRAMEWORKS)

# Includes clef.c with mocked GPU state; verifies token identity and failed-pass invalidation.
tests/test-prefix-planner: tests/test_prefix_planner.o clef_head.o clef_metal.o $(HOST_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(FRAMEWORKS)

tests/test-prefix-owner-model: tests/test_prefix_owner_model.o $(ENGINE_OBJS) $(HOST_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(FRAMEWORKS)

# Compares full and resumed attention with exact bits, float64 bounds and poisoned guards.
tests/test-prefix-attention: tests/test_prefix_attention.m bench/attention_bench.m
	$(CC) $(OBJCFLAGS) $(DEPFLAGS) -o $@ $< $(LDFLAGS) -framework Metal -framework Foundation

test-prefix-attention: tests/test-prefix-attention
	tests/test-prefix-attention metal/clef.metal metal/clef.metal

test: clef-tool tests/test-prefix-owner tests/test-prefix-planner tests/test-head-attend tests/test-head-linear tests/test-head-parallel tests/test-record-errors tests/test-server-writes
	tests/test-prefix-owner
	tests/test-prefix-planner
	tests/test-head-attend
	tests/test-head-linear
	tests/test-head-parallel
	tests/test-record-errors
	tests/test-server-writes
	.venv/bin/python -B tests/test_json.py
	.venv/bin/python -B tests/test_tokenizer.py gguf/clef-flash.gguf model-flash
	.venv/bin/python -B tests/test_verify_gguf.py
	.venv/bin/python -B tests/test_verify_snapshot.py
	.venv/bin/python -B tests/test_parity_checks.py
	.venv/bin/python -B tests/test_prefix_cache_checks.py
	.venv/bin/python -B tests/test_prefix_checkpoint_checks.py
	.venv/bin/python -B tests/test_cloudflare_checkout.py
	.venv/bin/python -B tests/test_cloudflare_corpus.py
	.venv/bin/python -B tests/test_compare_cloudflare.py
	.venv/bin/python -B tests/test_evidence_archive.py

test-errors: all tests/test-metal-errors tests/test-cli-alloc
	.venv/bin/python -B tests/test_cli_errors.py gguf/clef-flash.gguf
	tests/test-cli-alloc gguf/clef-flash.gguf
	tests/test-metal-errors gguf/clef-flash.gguf

clean:
	rm -f *.o *.d tests/*.o tests/*.d clef clef-server clef-tool attention-bench gemm-tiles gdn-bench tests/test-gdn-buffers tests/test-head-attend tests/test-head-linear tests/test-head-parallel tests/test-head-parallel-tsan tests/test-record-errors tests/test-metal-errors tests/test-server-writes tests/test-cli-alloc tests/test-prefix-owner tests/test-prefix-planner tests/test-prefix-owner-model tests/test-prefix-attention clef_metal_src.inc

-include $(wildcard *.d tests/*.d)
