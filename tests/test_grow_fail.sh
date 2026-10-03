#!/bin/sh
# Regression: a failed activation-buffer growth must leave the engine usable. Before the fix,
# growth replaced live buffers in place and kept the old capacity, so the next request that
# fit the old capacity ran against nil buffers and returned silently wrong logits.
# CLEF_DEBUG_GROW_FAIL_ABOVE=512 makes growth past 512 tokens fail.
# Usage: tests/test_grow_fail.sh MODEL.gguf REQUESTS.jsonl   (uses corpus lines 8, 20, 2)
set -eu
model=$1; requests=$2
tmp=$(mktemp -d "${TMPDIR:-/tmp}/clef-grow.XXXXXX")
trap 'rm -rf "$tmp"' EXIT
for i in 8 20 2; do sed -n "${i}p" "$requests"; done > "$tmp/req"
./clef -m "$model" --logits "$tmp/req" > "$tmp/clean"
CLEF_DEBUG_GROW_FAIL_ABOVE=512 ./clef -m "$model" --logits "$tmp/req" > "$tmp/fail" 2>/dev/null || true
grep -q '"error"' "$tmp/fail" || { echo "grow failure: hook did not trigger"; exit 1; }
if [ "$(sed -n 1p "$tmp/clean"; sed -n 3p "$tmp/clean")" = "$(sed -n 1p "$tmp/fail"; sed -n 3p "$tmp/fail")" ]; then
    echo "grow failure: failing request reported, requests before and after it correct"
else
    echo "grow failure: FAIL (request after the failed growth is wrong)"; exit 1
fi
