"""Largest |input| of every Linear in the model over the corpus (PyTorch BF16 on MPS).

Purpose: FP16 holds magnitudes up to 65504. Before feeding FP16 activations into the engine's
GEMMs, measure how close each GEMM input class comes to that limit on real requests. This is
evidence, not a guarantee: the engine still needs a guard for inputs outside the corpus.

Usage: act_stats.py MODEL_DIR [--safe-attn] [--out FILE.json]
Prints, per projection kind, the max over layers and requests, with the layer, request and token.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import corpus  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--safe-attn", action="store_true", help="head-chunked SDPA (MPS >= 2^32 offset bug, ref/safe_attn.py)")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    sys.path.insert(0, str(args.model_dir.resolve()))
    from joint_schema_model import collate_records, encode_record, load_release_model

    model, processor = load_release_model(args.model_dir, device="mps", dtype=torch.bfloat16)
    if args.safe_attn:
        import safe_attn
        safe_attn.install(model)
    tok = processor.tokenizer
    stats: dict[str, dict] = {}     # module name -> {max, req, token}
    cur = {"req": ""}

    def hook(name):
        def fn(mod, inp):
            x = inp[0].detach()
            a = x.abs().float().reshape(-1, x.shape[-1]).amax(dim=-1)   # per token row
            v, t = a.max(dim=0)
            v = float(v)
            s = stats.get(name)
            if s is None or v > s["max"]:
                stats[name] = {"max": v, "req": cur["req"], "token": int(t)}
        return fn

    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            mod.register_forward_pre_hook(hook(name))

    with torch.inference_mode():
        for req in corpus.build():
            cur["req"] = req["id"]
            enc = encode_record(tok, req)
            model(collate_records([enc], tok.pad_token_id, torch.device("mps")))
            print(f"{req['id']} tokens={len(enc.input_ids)}", flush=True)

    # group by projection kind: strip the layer index
    kinds: dict[str, tuple] = {}
    for name, s in stats.items():
        m = re.search(r"layers\.(\d+)\.", name)
        kind = re.sub(r"layers\.\d+\.", "layers.N.", name)
        if kind not in kinds or s["max"] > kinds[kind][0]:
            kinds[kind] = (s["max"], m.group(1) if m else "-", s["req"], s["token"])
    print(f"\n{'projection (input of)':72s} {'max |x|':>10s}  layer request token")
    for kind, (v, layer, req, t) in sorted(kinds.items(), key=lambda kv: -kv[1][0]):
        print(f"{kind:72s} {v:10.1f}  {layer:>5s} {req:7s} {t}")
    if args.out:
        args.out.write_text(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
