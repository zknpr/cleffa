#!/bin/sh
# Read-before-write guard: with every activation buffer filled with NaN before each forward
# (CLEF_DEBUG_POISON=1), logits must be bitwise identical to a clean run, at batch 1 and 8.
# Regression for review #2 M5: attention tiles read V rows past T that were never written
# this pass; masked probabilities are 0 but 0 * NaN = NaN (failed 22/22 before the fix).
# Usage: tests/test_poison.sh MODEL.gguf REQUESTS.jsonl
set -eu
model=$1; requests=$2
tmp=$(mktemp -d "${TMPDIR:-/tmp}/clef-poison.XXXXXX")
trap 'rm -rf "$tmp"' EXIT
./clef -m "$model" --logits "$requests" > "$tmp/clean"
for b in 1 8; do
    CLEF_DEBUG_POISON=1 ./clef -m "$model" --logits --batch "$b" "$requests" > "$tmp/poison$b"
    if cmp -s "$tmp/clean" "$tmp/poison$b"; then
        echo "poisoned buffers, batch $b: identical to clean"
    else
        echo "poisoned buffers, batch $b: FAIL ($(grep -c NaN "$tmp/poison$b") requests with NaN)"; exit 1
    fi
done
