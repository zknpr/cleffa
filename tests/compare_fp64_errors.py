"""Non-inferiority of an Omni build against a baseline, measured against the FP64 reference.

Usage: compare_fp64_errors.py BASE_DIR CANDIDATE_DIR [--golden-root golden]

Each directory holds `test_parity.py --bound-by ... --report NAME.jsonl` files, one per FP64
reference. The FP64-bounded acceptance check allows every question the FP32 reference's own
error, which on about a fifth of the questions is far above the fixed limits; a faster build
could therefore lose accuracy there and still pass. This compares the two builds directly.

The noise scale is the spread between two faithful FP32 implementations: the blocked-attention
reference and the independent SDPA control, both measured against FP64 on the same questions.
A candidate is non-inferior when its mean logit error and mean probability error exceed the
baseline's by no more than the larger of 5% and that FP32 spread (relative), and the number of
questions beyond the fixed limits grows by no more than the two FP32 references differ in it. Every question must still pass its bound
(test_parity.py exits non-zero otherwise). Exit status 1 means not shown non-inferior.

The reports must cover every question of the FP64 references in CORPUS, and both FP32 references
every one of those: on a subset the spread, and so the allowance, depends on which questions were
picked (the two FP32 references are bitwise equal on most short requests). Each question's bound
is recomputed from the blocked FP32 reference rather than taken from the report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

FIXED_LOGIT, FIXED_PROB = 0.05, 0.002
# FP64 reference name -> corpus of the FP32 references for the same requests
CORPUS = {"text-16k-f64-v2": "text", "text-short-f64-v1": "text", "text-mid-f64-v1": "text",
          "text-8k-f64-v2": "text", "vision-f64-v1": "vision", "video-f64-v1": "video",
          "audio-f64-v1": "audio", "av-f64-v1": "av"}


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def load(directory: Path) -> dict[tuple[str, str], dict]:
    rows = {}
    for path in sorted(directory.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            r = json.loads(line)
            key = (r["id"], r["question"])
            if key in rows:
                sys.exit(f"duplicate question {key} in {directory}")
            rows[key] = r
    if not rows:
        sys.exit(f"no reports in {directory}")
    return rows


def reference_errors(root: Path) -> tuple[set[tuple[str, str]], dict[str, dict]]:
    """Every FP64-referenced question, and the two FP32 references' errors against FP64 on it."""
    inventory, out = set(), {"blocked": {}, "sdpa": {}}
    for name, corpus in CORPUS.items():
        fp64_dir = root / f"clef-omni-{name}"
        if not fp64_dir.exists():
            sys.exit(f"missing FP64 reference {fp64_dir}")
        fp64 = load_file(str(fp64_dir / "logits.safetensors"))
        inventory |= {tuple(k.split("/", 1)) for k in fp64}
        for arm, suffix in (("blocked", "blocked-d2"), ("sdpa", "sdpa-control-d2")):
            ref = load_file(str(root / f"clef-omni-{corpus}-f32-{suffix}" / "logits.safetensors"))
            for k, g in fp64.items():
                if k not in ref:
                    continue
                g64, r = g.astype(np.float64), ref[k].astype(np.float64)
                rid, q = k.split("/", 1)
                out[arm][(rid, q)] = (float(np.abs(r - g64).max()), float(np.abs(softmax(r) - softmax(g64)).max()))
    return inventory, out


def summary(errors: list[tuple[float, float]]) -> tuple[float, float, int]:
    dl = np.array([e[0] for e in errors]); dp = np.array([e[1] for e in errors])
    return float(dl.mean()), float(dp.mean()), int(((dl > FIXED_LOGIT) | (dp > FIXED_PROB)).sum())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base", type=Path)
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--golden-root", type=Path, default=Path(__file__).resolve().parents[1] / "golden")
    args = ap.parse_args()
    base, cand = load(args.base), load(args.candidate)
    if base.keys() != cand.keys():
        sys.exit(f"question sets differ: {len(base)} baseline, {len(cand)} candidate")
    keys = set(base)
    inventory, refs = reference_errors(args.golden_root)
    if keys != inventory:
        sys.exit(f"reports cover {len(keys & inventory)} of the {len(inventory)} FP64-referenced questions"
                 f" (and {len(keys - inventory)} others); non-inferiority is judged on all of them")
    if refs["blocked"].keys() != inventory or refs["sdpa"].keys() != inventory:
        sys.exit("FP32 references cover only part of the FP64-referenced questions")
    shared = sorted(keys)

    def row(label: str, errs: list[tuple[float, float]]) -> tuple[float, float, int]:
        ml, mp, beyond = summary(errs)
        print(f"{label:22s} mean |d logit| {ml:.6f}  mean |d p| {mp:.6f}  beyond fixed {beyond}/{len(errs)}")
        return ml, mp, beyond

    print(f"{len(keys)} questions; {len(shared)} with both FP32 references")
    b = row("baseline", [(base[k]["logit_error"], base[k]["prob_error"]) for k in sorted(keys)])
    c = row("candidate", [(cand[k]["logit_error"], cand[k]["prob_error"]) for k in sorted(keys)])
    rb = row("FP32 blocked (shared)", [refs["blocked"][k] for k in shared])
    rs = row("FP32 SDPA (shared)", [refs["sdpa"][k] for k in shared])
    spread_l = abs(rb[0] - rs[0]) / max(min(rb[0], rs[0]), 1e-30)
    spread_p = abs(rb[1] - rs[1]) / max(min(rb[1], rs[1]), 1e-30)
    allow_l, allow_p = max(0.05, spread_l), max(0.05, spread_p)
    grow_l, grow_p = c[0] / b[0] - 1, c[1] / b[1] - 1
    print(f"FP32 implementation spread: logit {spread_l:+.1%}, probability {spread_p:+.1%}")
    print(f"candidate vs baseline:      logit {grow_l:+.1%} (allowed {allow_l:.1%}), "
          f"probability {grow_p:+.1%} (allowed {allow_p:.1%}), beyond fixed {c[2] - b[2]:+d}")
    worse = sorted(keys, key=lambda k: cand[k]["prob_error"] - base[k]["prob_error"], reverse=True)[:5]
    for k in worse:
        print(f"  largest increase {k[0]}/{k[1]}: |d p| {base[k]['prob_error']:.6f} -> {cand[k]['prob_error']:.6f}, "
              f"|d logit| {base[k]['logit_error']:.6f} -> {cand[k]['logit_error']:.6f}")
    # test_parity.py's bound, from the reference: a report's own limits may have been relaxed.
    unbounded = [k for k in keys if cand[k]["logit_error"] > max(FIXED_LOGIT, refs["blocked"][k][0]) or
                 cand[k]["prob_error"] > max(FIXED_PROB, refs["blocked"][k][1])]
    allow_count = abs(rb[2] - rs[2])
    print(f"beyond-fixed allowance from the FP32 references: {allow_count}")
    ok = grow_l <= allow_l and grow_p <= allow_p and c[2] - b[2] <= allow_count and not unbounded
    print("non-inferior" if ok else "NOT shown non-inferior" + (f" ({len(unbounded)} questions outside the bound)" if unbounded else ""))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
