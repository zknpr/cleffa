#!/bin/sh
# Batch invariance: packing several requests into one forward pass must not change any
# record's output (attention, conv and the delta-rule scan are per record), and responses
# must never carry another request's data (regression: shared getline buffer, review #2).
# Usage: tests/test_batch.sh MODEL.gguf REQUESTS.jsonl
set -eu
model=$1; requests=$2
tmp=$(mktemp -d "${TMPDIR:-/tmp}/clef-batch.XXXXXX")
trap 'rm -rf "$tmp"' EXIT
./clef -m "$model" --logits --batch 1 "$requests" > "$tmp/b1"
./clef -m "$model" --logits --batch 8 "$requests" > "$tmp/b8"
if cmp -s "$tmp/b1" "$tmp/b8"; then
    echo "batch invariance: --batch 1 and --batch 8 logits identical ($(wc -l < "$tmp/b1") requests)"
else
    echo "batch invariance: FAIL"; diff "$tmp/b1" "$tmp/b8" | head -10; exit 1
fi
# two tenants in one batch, second line longer (forces getline to reallocate)
printf '%s\n%s\n' \
  '{"model":"tenantA","state":"a","questions":{"qa":{"type":"choice","criteria":{"yes":"y","no":"n"}}}}' \
  "{\"model\":\"tenantB\",\"state\":\"$(printf 'b%.0s' $(seq 1 3000))\",\"questions\":{\"qb\":{\"type\":\"choice\",\"criteria\":{\"red\":\"r\",\"blu\":\"b\"}}}}" \
  > "$tmp/tenants"
./clef -m "$model" --batch 2 "$tmp/tenants" > "$tmp/out"
if head -1 "$tmp/out" | grep -q '"model":"tenantA","answers":{"qa"' && ! head -1 "$tmp/out" | grep -q tenantB; then
    echo "tenant isolation: OK"
else
    echo "tenant isolation: FAIL"; head -1 "$tmp/out" | cut -c1-200; exit 1
fi
