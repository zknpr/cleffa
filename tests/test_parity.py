"""End-to-end parity of the native engine against the PyTorch oracle (ref/oracle.py).

Usage: test_parity.py MODEL.gguf GOLDEN_DIR [--dump]
  1. token ids        : clef-tool encode == golden encoded.jsonl (exact)
  2. residual stream  : engine --dump of the first request vs the oracle's layer hooks
                        (relative L2 error and cosine per layer)
  3. decisions        : engine --logits vs golden logits: max |d logit|, max |d p| after
                        softmax, and argmax agreement per question

The oracle runs the HF model in BF16, so it is itself an approximation; the engine keeps
f32 where the oracle rounds to BF16. Per-layer error therefore reflects BF16 rounding
noise in the reference plus any engine bug: a bug shows up as a jump at one layer, noise
as a smooth drift. Use an FP32 golden (golden/clef-flash-f32): any argmax disagreement fails,
and against the BF16 golden the engine disagrees exactly where BF16 itself is wrong.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

ROOT = Path(__file__).resolve().parent.parent


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def main() -> None:
    gguf, golden = sys.argv[1], Path(sys.argv[2])
    do_dump = "--dump" in sys.argv
    requests = golden / "requests.jsonl"
    failures = 0

    # 1. token ids
    enc_c = subprocess.run([ROOT / "clef-tool", "encode", gguf], stdin=open(requests), capture_output=True,
                           text=True, check=True).stdout.splitlines()
    enc_g = [json.loads(l) for l in open(golden / "encoded.jsonl")]
    bad = sum(json.loads(c)["input_ids"] != g["input_ids"] for c, g in zip(enc_c, enc_g))
    print(f"token ids: {len(enc_g) - bad}/{len(enc_g)} identical")
    failures += bad

    # 2. residual stream of the first request
    layers_path = golden / "layers" / f"{enc_g[0]['id']}.safetensors"
    if do_dump and layers_path.exists():
        ref = load_file(str(layers_path))
        dump = ROOT / "golden" / "engine_dump.bin"
        first = open(requests).readline()
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
            rel = np.linalg.norm(e - r) / (np.linalg.norm(r) + 1e-30)
            cos = float((e * r).sum() / (np.linalg.norm(e) * np.linalg.norm(r) + 1e-30))
            flag = "  <-- jump" if i > 0 and rel > 4 * prev + 0.01 else ""
            print(f"  {n:12s} rel_l2={rel:.2e} cos={cos:.6f}{flag}")
            prev = rel

    # 3. decisions
    out = subprocess.run([ROOT / "clef", "-m", gguf, "--logits", requests], capture_output=True, text=True)
    if out.returncode != 0:
        print(out.stderr)
        sys.exit("engine failed")
    golden_logits = load_file(str(golden / "logits.safetensors"))
    rows = out.stdout.splitlines()
    max_dl = max_dp = 0.0
    n_q = agree = 0
    worst = None
    for req, line in zip((json.loads(l) for l in open(requests)), rows):
        got = json.loads(line)
        if "error" in got:
            print(f"  {req['id']}: engine error {got['error']}")
            failures += 1
            continue
        for qid, logits in got.items():
            g = golden_logits[f"{req['id']}/{qid}"].astype(np.float64)
            e = np.array(logits, dtype=np.float64)
            dl = float(np.abs(e - g).max())
            dp = float(np.abs(softmax(e) - softmax(g)).max())
            n_q += 1
            agree += int(e.argmax() == g.argmax())
            if dp > max_dp:
                worst = (req["id"], qid, dp)
            max_dl, max_dp = max(max_dl, dl), max(max_dp, dp)
    print(f"decisions: argmax agreement {agree}/{n_q}; max |d logit| {max_dl:.4f}; max |d p| {max_dp:.4f}"
          + (f" (worst {worst[0]}/{worst[1]})" if worst else ""))
    if agree != n_q:
        failures += n_q - agree
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
