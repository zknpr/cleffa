CC      ?= clang
CFLAGS  ?= -O2 -g -std=c11 -Wall -Wextra -Wno-unused-parameter -D_DARWIN_C_SOURCE -DACCELERATE_NEW_LAPACK
OBJCFLAGS = $(CFLAGS) -fobjc-arc
LDFLAGS ?=
FRAMEWORKS = -framework Metal -framework Foundation -framework Accelerate

HOST_OBJS = clef_gguf.o clef_json.o clef_tok.o clef_record.o

.PHONY: all clean test unicode

ENGINE_OBJS = clef.o clef_head.o clef_metal.o

all: clef clef-server clef-tool

clef_metal_src.inc: metal/clef.metal
	{ printf 'static const char clef_metal_src[] = {'; xxd -i < $< ; printf '};\nstatic const unsigned long clef_metal_src_len = sizeof(clef_metal_src);\n'; } > $@

clef_metal.o: clef_metal.m clef_metal_src.inc clef_engine.h
	$(CC) $(OBJCFLAGS) -c -o $@ $<

clef.o: clef.c clef_engine.h
clef_head.o: clef_head.c clef_engine.h

clef: clef_main.c $(ENGINE_OBJS) $(HOST_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(FRAMEWORKS)

clef-server: clef_server.c $(ENGINE_OBJS) $(HOST_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(FRAMEWORKS) -lpthread

clef_tok.o: clef_tok.c clef_tok.h clef_gguf.h clef_unicode.inc
clef_json.o: clef_json.c clef_json.h
clef_gguf.o: clef_gguf.c clef_gguf.h
clef_record.o: clef_record.c clef_record.h clef_json.h clef_tok.h

clef-tool: tests/clef_tool.c $(HOST_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# clef_unicode.inc is committed and regenerated only on purpose (`make unicode`): its Unicode version
# must match the tokenizer tests, and an automatic rule fired on checkout mtimes in a fresh clone.
unicode:
	.venv/bin/python tools/gen_unicode.py > clef_unicode.inc

tests/test-head-attend: tests/test_head_attend.c clef_head.c $(HOST_OBJS)
	$(CC) $(CFLAGS) -o $@ tests/test_head_attend.c $(HOST_OBJS) $(LDFLAGS) -framework Accelerate

test: clef-tool tests/test-head-attend
	tests/test-head-attend
	.venv/bin/python tests/test_json.py
	.venv/bin/python tests/test_tokenizer.py gguf/clef-flash.gguf model-flash

clean:
	rm -f *.o clef clef-server clef-tool tests/test-head-attend clef_metal_src.inc
