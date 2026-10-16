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

--bound-by FP32_GOLDEN (Clef-Omni): GOLDEN is the independent FP64 reference. Top-8 expert
routing has near-ties finer than FP32 resolves, so no FP32 implementation, the FP32 PyTorch
reference included, stays within fixed limits of FP64 everywhere. A question passes when its
logit and probability errors against FP64 are each within the larger of the limit and
FP32_GOLDEN's own error on that question, and its argmax matches FP64 or FP32_GOLDEN. Both
references must have encoded identical token ids. Layer relative L2 is then a diagnostic;
non-finite values still fail.
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
    parser.add_argument("--bound-by", type=Path, metavar="FP32_GOLDEN",
                        help="GOLDEN is FP64; allow each question FP32_GOLDEN's own error (Omni)")
    parser.add_argument("--report", type=Path, metavar="JSONL",
                        help="write each question's errors (and the bound's) for comparing builds")
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
    if args.bound_by:
        # The bound is meaningful only for the same input: a reference that tokenized the
        # request differently says nothing about this question's achievable error.
        with (args.bound_by / "encoded.jsonl").open() as source:
            bound_enc = {e["id"]: e["input_ids"] for e in map(json.loads, source)}
        mismatched = [g["id"] for g in enc_g if bound_enc.get(g["id"]) != g["input_ids"]]
        if mismatched:
            print(f"bound: FAIL {len(mismatched)} requests missing or encoded differently in {args.bound_by}")
            failures += len(mismatched)

    # 2. residual stream of the first request
    layers_path = golden / "layers" / f"{enc_g[0]['id']}.safetensors"
    if do_dump and not layers_path.exists():
        sys.exit(f"missing requested layer reference: {layers_path}")
    if do_dump and layers_path.exists():
        ref = load_file(str(layers_path))
        T = len(enc_g[0]["input_ids"])
        start = 0
        manifest_path = golden / "manifest.json"
        if manifest_path.exists():
            records = json.loads(manifest_path.read_text()).get("records", [])
            matches = [r for r in records if r.get("id") == enc_g[0]["id"]]
            if len(matches) != 1:
                sys.exit("missing or duplicate layer manifest record")
            record = matches[0]
            start = record.get("dump_start")
            if type(start) is not int or not 0 <= start < T or record.get("tokens") != T:
                sys.exit("invalid layer suffix geometry")
        n_layer = sum(1 for k in ref if k.startswith("layer."))
        names = ["embed"] + [f"layer.{i:02d}" for i in range(n_layer)] + ["final_norm"]
        if not set(names) <= ref.keys() or ref["embed"].ndim != 2:
            sys.exit("missing or invalid decoder layer reference")
        H = ref["embed"].shape[-1]
        if H <= 0 or any(ref[n].shape != (T-start, H) for n in names):
            sys.exit("layer shapes differ from explicit suffix geometry")
        dump = ROOT / "golden" / "engine_dump.bin"
        first = request_lines[0]
        command = [ROOT / "clef", "-m", gguf, "--dump", dump]
        if start:
            command += ["--dump-last", str(T-start)]
        subprocess.run(command, input=first, capture_output=True, text=True, check=True)
        if dump.stat().st_size != (n_layer+2)*(T-start)*H*4:
            sys.exit("native layer dump size differs from suffix geometry")
        eng = np.memmap(dump, dtype=np.float32, mode="r", shape=(n_layer+2, T-start, H))
        print(f"residual stream ({T-start} rows from {T} tokens):")
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
            if not np.isfinite(rel):
                print(f"  {n}: FAIL non-finite relative L2")
                failures += 1
            elif rel > args.max_layer_rel_l2:
                # Under --bound-by, routing flips make layer drift expected; the final outputs
                # carry the acceptance decision.
                print(f"  {n}: {'diagnostic' if args.bound_by else 'FAIL'} relative L2 exceeds {args.max_layer_rel_l2}")
                failures += 0 if args.bound_by else 1
            prev = rel

    # 3. decisions
    out = subprocess.run([ROOT / "clef", "-m", gguf, "--logits", requests], capture_output=True, text=True)
    if out.returncode != 0:
        print(out.stderr)
        sys.exit("engine failed")
    golden_logits = load_file(str(golden / "logits.safetensors"))
    bound_logits = load_file(str(args.bound_by / "logits.safetensors")) if args.bound_by else None
    report = args.report.open("w") if args.report else None
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
            limit_l, limit_p, agrees = args.max_logit_error, args.max_prob_error, e.argmax() == g.argmax()
            if bound_logits is not None:
                b = bound_logits.get(f"{req['id']}/{qid}")
                if b is None or b.shape != g.shape or not np.isfinite(b).all():
                    print(f"  {req['id']}/{qid}: FAIL missing, mis-shaped or non-finite bound logits")
                    failures += 1
                    continue
                b = b.astype(np.float64)
                limit_l = max(limit_l, float(np.abs(b - g).max()))
                limit_p = max(limit_p, float(np.abs(softmax(b) - softmax(g)).max()))
                agrees = agrees or e.argmax() == b.argmax()
            if report:
                report.write(json.dumps({"id": req["id"], "question": qid, "logit_error": dl, "prob_error": dp,
                                         "limit_logit": limit_l, "limit_prob": limit_p,
                                         "argmax_equal": bool(e.argmax() == g.argmax())}) + "\n")
            if not (np.isfinite(dl) and np.isfinite(dp)) or dl > limit_l or dp > limit_p:
                print(f"  {req['id']}/{qid}: FAIL |d logit|={dl:.6g} (limit {limit_l:.6g}), "
                      f"|d p|={dp:.6g} (limit {limit_p:.6g})")
                failures += 1
            n_q += 1
            agree += int(agrees)
            if dp > max_dp:
                worst = (req["id"], qid, dp)
            max_dl, max_dp = max(max_dl, dl), max(max_dp, dp)
    print(f"decisions: argmax agreement {agree}/{n_q}; max |d logit| {max_dl:.4f}; max |d p| {max_dp:.4f}"
          + (f" (worst {worst[0]}/{worst[1]})" if worst else ""))
    if report:
        report.close()
    if agree != n_q or n_q == 0:
        failures += max(1, n_q - agree)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
