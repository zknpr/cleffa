"""End-to-end parity of the native engine against the PyTorch oracle (ref/oracle.py).

Usage: test_parity.py MODEL.gguf GOLDEN_DIR [--dump]
  1. token ids        : clef-tool encode == golden encoded.jsonl (exact)
  2. residual stream  : engine --dump of the first request vs the oracle's layer hooks
                        (relative L2 error and cosine per layer)
  3. decisions        : engine --logits vs golden logits: max |d logit|, max |d p| after
                        softmax, and argmax agreement per question

Use an FP32 golden (golden/clef-flash-f32 or golden/clef-f32). Any argmax disagreement,
non-finite value, or numerical error above its tolerance fails. Defaults: max absolute
logit error 0.05, max probability error 0.002, and per-layer relative L2 error 0.01.
The --max-logit-error, --max-prob-error and --max-layer-rel-l2 flags override these limits.
BF16 references contain their own rounding error and are not the acceptance target.
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

ROOT = Path(__file__).resolve().parent.parent


def tolerance(text: str) -> float:
    value = float(text)
    if not np.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("tolerance must be finite and non-negative")
    return value


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gguf")
    parser.add_argument("golden", type=Path)
    parser.add_argument("--dump", action="store_true")
    # FP32 is the reference. Defaults allow the measured error of both models,
    # while guarding confidence/score drift even when argmax stays unchanged.
    parser.add_argument("--max-logit-error", type=tolerance, default=0.05)
    parser.add_argument("--max-prob-error", type=tolerance, default=0.002)
    parser.add_argument("--max-layer-rel-l2", type=tolerance, default=0.01)
    args = parser.parse_args()
    gguf, golden, do_dump = args.gguf, args.golden, args.dump
    requests = golden / "requests.jsonl"
    failures = 0

    # 1. token ids
    # JSONL uses LF, not Unicode line separators that can occur inside JSON strings.
    with requests.open() as source:
        request_lines = source.readlines()
    encoded = subprocess.run([ROOT / "clef-tool", "encode", gguf], input="".join(request_lines),
                             capture_output=True, text=True, check=True).stdout
    enc_c = list(io.StringIO(encoded))
    with (golden / "encoded.jsonl").open() as source:
        enc_g = [json.loads(l) for l in source]
    n_req = len(request_lines)
    if not n_req or not enc_g:
        sys.exit("empty request or encoded corpus")
    # zip() stops at the shorter list: missing output must fail, not shrink the comparison (review #4)
    if not (len(enc_c) == len(enc_g) == n_req):
        print(f"token ids: FAIL ({len(enc_c)} encoded by clef-tool, {len(enc_g)} golden, {n_req} requests)")
        failures += 1
    bad = sum(json.loads(c)["input_ids"] != g["input_ids"] for c, g in zip(enc_c, enc_g))
    print(f"token ids: {len(enc_g) - bad}/{len(enc_g)} identical")
    failures += bad

    # 2. residual stream of the first request
    layers_path = golden / "layers" / f"{enc_g[0]['id']}.safetensors"
    if do_dump and not layers_path.exists():
        sys.exit(f"missing requested layer reference: {layers_path}")
    if do_dump and layers_path.exists():
        ref = load_file(str(layers_path))
        dump = ROOT / "golden" / "engine_dump.bin"
        first = request_lines[0]
        subprocess.run([ROOT / "clef", "-m", gguf, "--dump", dump], input=first, capture_output=True, text=True, check=True)
        T = len(enc_g[0]["input_ids"])
        n_layer = sum(1 for k in ref if k.startswith("layer."))
        H = ref["embed"].shape[-1]
        eng = np.fromfile(dump, dtype=np.float32).reshape(n_layer + 2, T, H)
        names = ["embed"] + [f"layer.{i:02d}" for i in range(n_layer)] + ["final_norm"]
        print(f"residual stream ({T} tokens):")
        prev = 0.0
        for i, n in enumerate(names):
            r, e = ref[n].astype(np.float64), eng[i].astype(np.float64)
            if not (np.isfinite(r).all() and np.isfinite(e).all()):
                print(f"  {n}: FAIL non-finite residuals")
                failures += 1
                continue
            rel = np.linalg.norm(e - r) / (np.linalg.norm(r) + 1e-30)
            cos = float((e * r).sum() / (np.linalg.norm(e) * np.linalg.norm(r) + 1e-30))
            flag = "  <-- jump" if i > 0 and rel > 4 * prev + 0.01 else ""
            print(f"  {n:12s} rel_l2={rel:.2e} cos={cos:.6f}{flag}")
            if not np.isfinite(rel) or rel > args.max_layer_rel_l2:
                print(f"  {n}: FAIL relative L2 exceeds {args.max_layer_rel_l2}")
                failures += 1
            prev = rel

    # 3. decisions
    out = subprocess.run([ROOT / "clef", "-m", gguf, "--logits", requests], capture_output=True, text=True)
    if out.returncode != 0:
        print(out.stderr)
        sys.exit("engine failed")
    golden_logits = load_file(str(golden / "logits.safetensors"))
    rows = list(io.StringIO(out.stdout))
    max_dl = max_dp = 0.0
    n_q = agree = 0
    worst = None
    # every request must answer exactly the golden questions with the golden option counts; a short
    # or partial output used to pass as "agreement 0/0" (review #4)
    if len(rows) != n_req:
        print(f"decisions: FAIL ({len(rows)} responses for {n_req} requests)")
        failures += 1
    for req, enc, line in zip(map(json.loads, request_lines), enc_g, rows):
        got = json.loads(line)
        if isinstance(got.get("error"), str):
            print(f"  {req['id']}: engine error {got['error']}")
            failures += 1
            continue
        want = {q["id"]: len(q["option_ids"]) for q in enc["questions"]}
        have = {qid: len(v) for qid, v in got.items()}
        if have != want:
            print(f"  {req['id']}: questions/options {have} differ from the golden {want}")
            failures += 1
            continue
        for qid, logits in got.items():
            g = golden_logits[f"{req['id']}/{qid}"].astype(np.float64)
            e = np.array(logits, dtype=np.float64)
            if e.shape != g.shape or e.size == 0 or not (np.isfinite(e).all() and np.isfinite(g).all()):
                print(f"  {req['id']}/{qid}: FAIL invalid shape or non-finite logits")
                failures += 1
                continue
            dl = float(np.abs(e - g).max())
            dp = float(np.abs(softmax(e) - softmax(g)).max())
            if not (np.isfinite(dl) and np.isfinite(dp)) or dl > args.max_logit_error or dp > args.max_prob_error:
                print(f"  {req['id']}/{qid}: FAIL |d logit|={dl:.6g} (limit {args.max_logit_error}), "
                      f"|d p|={dp:.6g} (limit {args.max_prob_error})")
                failures += 1
            n_q += 1
            agree += int(e.argmax() == g.argmax())
            if dp > max_dp:
                worst = (req["id"], qid, dp)
            max_dl, max_dp = max(max_dl, dl), max(max_dp, dp)
    print(f"decisions: argmax agreement {agree}/{n_q}; max |d logit| {max_dl:.4f}; max |d p| {max_dp:.4f}"
          + (f" (worst {worst[0]}/{worst[1]})" if worst else ""))
    if agree != n_q or n_q == 0:
        failures += max(1, n_q - agree)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
