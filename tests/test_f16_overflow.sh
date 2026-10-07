#!/bin/sh
# FP16 overflow guard: a record whose FP16 GEMM operands would leave FP16's range (|x| > 65504)
# is recomputed alone with BF16 activations, and its discarded pass must not disturb the records
# packed with it. Real inputs stay far below the limit (27B corpus max 3648, ref/act_stats.py), so
# CLEF_DEBUG_F16_LIMIT lowers it to force the path; a flagged value is written exactly as a real
# overflow would be (act16 in metal/clef.metal).
#   1. each producer class flags its own overflow: with only class m in FP16 and a limit below
#      every activation, every record equals the BF16-activation run (CLEF_ACT_F16=0), for the
#      tensor-unit attention kernel (whose operand split in attn_prep flags too), the FP32
#      kernels (CLEF_ATTN_TU=0) and the reference kernel (CLEF_ATTN_REF=1);
#   2. all classes, limit below every activation: every record equals the BF16 run, batch 1 and 8;
#   3. limits that split the corpus: batch 1 == batch 8, each record equals either its FP16 or its
#      BF16 result, and some batch of 8 mixes both kinds. The DeltaNet-only arm overflows early in
#      the stack, so the flagged record's values pass through later attention layers that also
#      hold its neighbours' rows (review #3: an inf there became NaN in the neighbours);
#   4. --dump of an overflowing record holds the BF16 rerun's residuals, not the discarded pass.
# Usage: tests/test_f16_overflow.sh MODEL.gguf REQUESTS.jsonl [SPLIT_LIMIT [GDN_SPLIT_LIMIT]]
# The defaults split the clef-flash corpus: 1250 with all classes (12 of 22 records exceed it) and
# 80 with DeltaNet outputs only. 3000 splits the 27B corpus with all classes; its DeltaNet outputs
# peak at token 0, which every request shares (same template prefix, causal model), so no limit
# splits it: pass "-" to skip that arm there (the attention kernel path is the same code).
set -eu
model=$1; requests=$2; split=${3:-1250}; gdn_split=${4:-80}
tmp=$(mktemp -d "${TMPDIR:-/tmp}/clef-f16.XXXXXX")
trap 'rm -rf "$tmp"' EXIT
# Env prefixes go on ./clef itself: before a shell function, POSIX sh keeps the assignment after the
# call, which leaked CLEF_DEBUG_F16_LIMIT into later arms.
clef="./clef"
"$clef" -m "$model" --logits "$requests" > "$tmp/f16"
CLEF_ACT_F16=0 "$clef" -m "$model" --logits "$requests" > "$tmp/bf16"
if cmp -s "$tmp/f16" "$tmp/bf16"; then echo "FP16 and BF16 activations give identical logits: the test cannot tell them apart"; exit 1; fi

for m in 1 2 4 8; do
    CLEF_ACT_F16=$m "$clef" -m "$model" --logits --batch 8 "$requests" > "$tmp/only$m"
    if cmp -s "$tmp/only$m" "$tmp/bf16"; then echo "class $m alone in FP16 equals BF16: its flag cannot be tested"; exit 1; fi
    CLEF_ACT_F16=$m CLEF_DEBUG_F16_LIMIT=1e-6 "$clef" -m "$model" --logits --batch 8 "$requests" > "$tmp/lim$m"
    if ! cmp -s "$tmp/lim$m" "$tmp/bf16"; then echo "class $m: FAIL (an overflow in this class is not flagged)"; exit 1; fi
done
# BF16 passes never take the tensor-unit kernel, so $tmp/bf16 is the FP32 kernels' reference too
CLEF_ATTN_TU=0 CLEF_ACT_F16=2 "$clef" -m "$model" --logits --batch 8 "$requests" > "$tmp/fa_only2"
if cmp -s "$tmp/fa_only2" "$tmp/bf16"; then echo "FP32 attention kernels in FP16 equal BF16: their flag cannot be tested"; exit 1; fi
if cmp -s "$tmp/fa_only2" "$tmp/only2"; then echo "CLEF_ATTN_TU=0 did not change the attention kernel"; exit 1; fi
CLEF_ATTN_TU=0 CLEF_ACT_F16=2 CLEF_DEBUG_F16_LIMIT=1e-6 "$clef" -m "$model" --logits --batch 8 "$requests" > "$tmp/fa_lim2"
if ! cmp -s "$tmp/fa_lim2" "$tmp/bf16"; then echo "FP32 attention kernels: FAIL (their overflow is not flagged)"; exit 1; fi
CLEF_ATTN_REF=1 CLEF_ACT_F16=0 "$clef" -m "$model" --logits --batch 8 "$requests" > "$tmp/ref_bf16"
CLEF_ATTN_REF=1 CLEF_ACT_F16=2 "$clef" -m "$model" --logits --batch 8 "$requests" > "$tmp/ref_only2"
if cmp -s "$tmp/ref_only2" "$tmp/ref_bf16"; then echo "reference attention in FP16 equals BF16: its flag cannot be tested"; exit 1; fi
CLEF_ATTN_REF=1 CLEF_ACT_F16=2 CLEF_DEBUG_F16_LIMIT=1e-6 "$clef" -m "$model" --logits --batch 8 "$requests" > "$tmp/ref_lim2"
if ! cmp -s "$tmp/ref_lim2" "$tmp/ref_bf16"; then echo "reference attention: FAIL (its overflow is not flagged)"; exit 1; fi
echo "each producer class flags its own overflow (rmsnorm, attention tensor-unit/FP32/reference, DeltaNet, SwiGLU)"

for b in 1 8; do
    CLEF_DEBUG_F16_LIMIT=1e-6 "$clef" -m "$model" --logits --batch "$b" "$requests" > "$tmp/all$b"
    if cmp -s "$tmp/all$b" "$tmp/bf16"; then
        echo "every record overflowing, batch $b: all recomputed with BF16 activations (bitwise)"
    else
        echo "every record overflowing, batch $b: FAIL (output differs from the BF16-activation run)"; exit 1
    fi
done

# split_arm MASK LIMIT: some records overflow and some don't, packed together at --batch 8
split_arm() {
    mask=$1; lim=$2
    CLEF_ACT_F16=$mask "$clef" -m "$model" --logits "$requests" > "$tmp/ref$mask"
    CLEF_ACT_F16=$mask CLEF_DEBUG_F16_LIMIT=$lim "$clef" -m "$model" --logits --batch 1 "$requests" > "$tmp/s1"
    CLEF_ACT_F16=$mask CLEF_DEBUG_F16_LIMIT=$lim "$clef" -m "$model" --logits --batch 8 "$requests" > "$tmp/s8"
    if ! cmp -s "$tmp/s1" "$tmp/s8"; then
        echo "classes $mask, limit $lim: FAIL (batch 1 and batch 8 differ: an overflowing record changed its neighbours)"
        diff "$tmp/s1" "$tmp/s8" | grep -c '^>' | xargs echo "  records differing:"; exit 1
    fi
    n_f16=0; n_bf16=0; i=0; kinds=""; mixed=0
    while IFS= read -r line; do
        i=$((i + 1))
        if [ "$line" = "$(sed -n "${i}p" "$tmp/ref$mask")" ]; then n_f16=$((n_f16 + 1)); k=f
        elif [ "$line" = "$(sed -n "${i}p" "$tmp/bf16")" ]; then n_bf16=$((n_bf16 + 1)); k=b
        else echo "classes $mask, limit $lim: FAIL (record $i matches neither its FP16 nor its BF16 result)"; exit 1; fi
        kinds="$kinds$k"
        if [ $((i % 8)) -eq 0 ]; then case "$kinds" in *f*b*|*b*f*) mixed=1;; esac; kinds=""; fi
    done < "$tmp/s1"
    case "$kinds" in *f*b*|*b*f*) mixed=1;; esac
    if [ "$mixed" -eq 0 ]; then
        echo "classes $mask, limit $lim: no batch of 8 mixes overflowing and clean records ($n_bf16 BF16, $n_f16 FP16): pick another limit"; exit 1
    fi
    echo "classes $mask, limit $lim: $n_bf16 records recomputed with BF16, $n_f16 kept FP16, mixed in one batch; batch 1 == batch 8"
}
split_arm 15 "$split"
if [ "$gdn_split" = "-" ]; then echo "classes 4: skipped (no limit given)"; else split_arm 4 "$gdn_split"; fi

head -1 "$requests" > "$tmp/first"
"$clef" -m "$model" --logits --dump "$tmp/dump_f16" "$tmp/first" > /dev/null
CLEF_ACT_F16=0 "$clef" -m "$model" --logits --dump "$tmp/dump_bf16" "$tmp/first" > "$tmp/first_bf16"
CLEF_DEBUG_F16_LIMIT=1e-6 "$clef" -m "$model" --logits --dump "$tmp/dump_ovf" "$tmp/first" > "$tmp/first_ovf"
if cmp -s "$tmp/dump_f16" "$tmp/dump_bf16"; then echo "--dump: FP16 and BF16 dumps identical, the check cannot tell them apart"; exit 1; fi
if cmp -s "$tmp/dump_ovf" "$tmp/dump_bf16" && cmp -s "$tmp/first_ovf" "$tmp/first_bf16"; then
    echo "--dump of an overflowing record: dump and logits are the BF16 rerun's (bitwise)"
else
    echo "--dump of an overflowing record: FAIL (dump or logits differ from the BF16 run)"; exit 1
fi
