"""Three-way precision comparison: engine vs HF-BF16 vs HF-FP32 (the closest thing to truth).

Usage: compare3.py GOLDEN_BF16 GOLDEN_F32 ENGINE_LOGITS.jsonl [ENGINE_DUMP.bin]
For each layer (first request) and each question, reports the distance of the engine and of
the BF16 reference from the FP32 reference. If the engine is consistently closer to FP32
than BF16-HF is, the engine-vs-BF16 differences are the reference's rounding noise; a
layer where the engine is farther from FP32 than BF16-HF is points at an engine bug.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def rel(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def main() -> None:
    gb, gf, elog = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    dump = sys.argv[4] if len(sys.argv) > 4 else None
    reqs = [json.loads(l) for l in open(gb / "requests.jsonl")]
    if dump:
        rid = reqs[0]["id"]
        lb, lf = load_file(str(gb / "layers" / f"{rid}.safetensors")), load_file(str(gf / "layers" / f"{rid}.safetensors"))
        H = lb["embed"].shape[-1]
        T = lb["embed"].shape[0]
        n = sum(1 for k in lb if k.startswith("layer."))
        eng = np.fromfile(dump, dtype=np.float32).reshape(n + 2, T, H)
        names = ["embed"] + [f"layer.{i:02d}" for i in range(n)] + ["final_norm"]
        print(f"{'layer':12s} {'engine->f32':>12s} {'bf16HF->f32':>12s}  ratio")
        for i, nm in enumerate(names):
            f = lf[nm].astype(np.float64)
            e, b = rel(eng[i].astype(np.float64), f), rel(lb[nm].astype(np.float64), f)
            print(f"{nm:12s} {e:12.2e} {b:12.2e}  {e / b if b else float('nan'):5.2f}{'  <-- engine worse' if e > b * 1.5 and e > 1e-4 else ''}")
    LB, LF = load_file(str(gb / "logits.safetensors")), load_file(str(gf / "logits.safetensors"))
    rows = [json.loads(l) for l in open(elog)]
    e_dp, b_dp = [], []
    flips: list = []
    agree_e = agree_b = n_q = 0
    print(f"\n{'question':28s} {'|dp| eng':>9s} {'|dp| bf16':>9s} {'margin':>7s}  argmax f32/eng/bf16")
    for req, row in zip(reqs, rows):
        for qid, lg in row.items():
            k = f"{req['id']}/{qid}"
            pf, pb, pe = softmax(LF[k].astype(np.float64)), softmax(LB[k].astype(np.float64)), softmax(np.array(lg))
            de, db = float(np.abs(pe - pf).max()), float(np.abs(pb - pf).max())
            e_dp.append(de); b_dp.append(db)
            n_q += 1
            agree_e += int(pe.argmax() == pf.argmax())
            agree_b += int(pb.argmax() == pf.argmax())
            # FP32 top-2 logit margin: a flip on a near-tie (small margin) is precision, not a bug
            srt = np.sort(LF[k].astype(np.float64))[::-1]
            margin = float(srt[0] - srt[1]) if len(srt) > 1 else float("inf")
            mark = "" if pe.argmax() == pf.argmax() == pb.argmax() else "  *"
            if pe.argmax() != pf.argmax():
                mark += f" ENGINE FLIP (fp32 margin {margin:.3f})"
                flips.append((k, margin))
            print(f"{k:28s} {de:9.4f} {db:9.4f} {margin:7.3f}  {pf.argmax()}/{pe.argmax()}/{pb.argmax()}{mark}")
    print(f"\nvs FP32: engine argmax {agree_e}/{n_q}, mean|dp| {np.mean(e_dp):.4f}, max|dp| {np.max(e_dp):.4f}")
    print(f"vs FP32: BF16-HF argmax {agree_b}/{n_q}, mean|dp| {np.mean(b_dp):.4f}, max|dp| {np.max(b_dp):.4f}")
    if flips:
        print("engine argmax flips vs FP32 (with the FP32 top-2 logit margin):", ", ".join(f"{k} ({m:.3f})" for k, m in flips))


if __name__ == "__main__":
    main()
