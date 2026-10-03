#!/bin/sh
# Regression (review #4): a command buffer Metal fails to create must fail the forward pass. A nil
# command buffer turns every later encoder call into a no-op and has no .error, so the pass used to
# "succeed" without running and the head read the previous pass's activations: another request's.
# CLEF_DEBUG_NIL_CMDBUF=N makes the Nth command-buffer creation in the process return nil.
#   1. plain forward: the second request's command buffer is nil -> its response is an error, the
#      first request's response is unchanged;
#   2. --dump (one command buffer per layer): a nil mid-pass -> error;
#   3. CLEF_PROFILE=1 (a new command buffer at each kernel-category boundary): likewise.
# Usage: tests/test_gpu_fail.sh MODEL.gguf REQUESTS.jsonl
set -eu
model=$1; requests=$2
tmp=$(mktemp -d "${TMPDIR:-/tmp}/clef-gpufail.XXXXXX")
trap 'rm -rf "$tmp"' EXIT
head -2 "$requests" > "$tmp/two"
head -1 "$requests" > "$tmp/one"
./clef -m "$model" --logits "$tmp/two" > "$tmp/clean"

if CLEF_DEBUG_NIL_CMDBUF=2 ./clef -m "$model" --logits "$tmp/two" > "$tmp/nil2" 2>/dev/null; then rc=0; else rc=$?; fi
if [ "$(sed -n 1p "$tmp/nil2")" = "$(sed -n 1p "$tmp/clean")" ] && sed -n 2p "$tmp/nil2" | grep -q '^{"error"' && [ "$rc" -ne 0 ]; then
    echo "nil command buffer, second request: error reported, first request unchanged"
else
    echo "nil command buffer, second request: FAIL (exit $rc; second line: $(sed -n 2p "$tmp/nil2" | cut -c1-80))"; exit 1
fi

if CLEF_DEBUG_NIL_CMDBUF=3 ./clef -m "$model" --logits --dump "$tmp/dump" "$tmp/one" > "$tmp/nildump" 2>/dev/null; then rc=0; else rc=$?; fi
if grep -q '^{"error"' "$tmp/nildump" && [ "$rc" -ne 0 ]; then
    echo "nil command buffer mid-pass with --dump: error reported"
else
    echo "nil command buffer mid-pass with --dump: FAIL (exit $rc; output: $(cut -c1-80 "$tmp/nildump"))"; exit 1
fi

if CLEF_PROFILE=1 CLEF_DEBUG_NIL_CMDBUF=3 ./clef -m "$model" --logits "$tmp/one" > "$tmp/nilprof" 2>/dev/null; then rc=0; else rc=$?; fi
if grep -q '^{"error"' "$tmp/nilprof" && [ "$rc" -ne 0 ]; then
    echo "nil command buffer mid-pass with CLEF_PROFILE: error reported"
else
    echo "nil command buffer mid-pass with CLEF_PROFILE: FAIL (exit $rc; output: $(cut -c1-80 "$tmp/nilprof"))"; exit 1
fi
