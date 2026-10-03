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

.PHONY: all clean test unicode

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

# clef_unicode.inc is committed and regenerated only on purpose (`make unicode`): its Unicode version
# must match the tokenizer tests, and an automatic rule fired on checkout mtimes in a fresh clone.
unicode:
	.venv/bin/python tools/gen_unicode.py > clef_unicode.inc

tests/test-head-attend: tests/test_head_attend.o $(HOST_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) -framework Accelerate

test: clef-tool tests/test-head-attend
	tests/test-head-attend
	.venv/bin/python tests/test_json.py
	.venv/bin/python tests/test_tokenizer.py gguf/clef-flash.gguf model-flash

clean:
	rm -f *.o *.d tests/*.o tests/*.d clef clef-server clef-tool tests/test-head-attend clef_metal_src.inc

-include $(wildcard *.d tests/*.d)
