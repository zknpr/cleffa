"""Verify a complete converted GGUF against its HF source: tensor inventory, shapes,
types, and every byte (matrices, fused concatenations, head tensors) or value
(folded f32 norms, A = -exp(A_log)). Tokenizer-only exports are not complete models.

Usage: verify_gguf.py HF_DIR MODEL.gguf
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from gguf import GGMLQuantizationType, GGUFReader
from safetensors import safe_open

P = "model.language_model."


def main() -> None:
    hf, path = Path(sys.argv[1]), sys.argv[2]
    idx = json.loads((hf / "model.safetensors.index.json").read_text())["weight_map"]
    handles: dict = {}

    def src(name: str) -> torch.Tensor:
        shard = idx[name]
        if shard not in handles:
            handles[shard] = safe_open(hf / shard, "pt")
        return handles[shard].get_tensor(name)

    head = safe_open(hf / "joint_head.safetensors", "pt")
    cfg = json.loads((hf / "config.json").read_text())["text_config"]

    # Derive completeness from the source, independently of the converter and the
    # output table. Checking only tensors present made even an empty export pass.
    names = {"token_embd.weight", "output.weight", "output_norm.weight"}
    names.update("head." + k for k in head.keys())
    if len(cfg["layer_types"]) != cfg["num_hidden_layers"]:
        sys.exit("source config: layer_types does not match num_hidden_layers")
    common = {"attn_norm.weight", "ffn_norm.weight", "ffn_gate_up.weight", "ffn_down.weight"}
    full = {"attn_qkv.weight", "attn_output.weight", "attn_q_norm.weight", "attn_k_norm.weight"}
    linear = {"ssm_in.weight", "ssm_out.weight", "ssm_conv1d.weight", "ssm_dt.bias", "ssm_a", "ssm_norm.weight"}
    for i, kind in enumerate(cfg["layer_types"]):
        if kind not in ("full_attention", "linear_attention"):
            sys.exit(f"source config: unknown layer type {kind!r}")
        names.update(f"blk.{i}.{n}" for n in common | (full if kind == "full_attention" else linear))

    def expected(name: str) -> torch.Tensor:
        if name.startswith("head."):
            t = head.get_tensor(name[5:])
            return t.reshape(1) if t.dim() == 0 else t
        if name == "token_embd.weight":
            return src(P + "embed_tokens.weight")
        if name == "output.weight":
            return src("lm_head.weight")
        if name == "output_norm.weight":
            return 1.0 + src(P + "norm.weight").float()
        _, i, rest = name.split(".", 2)
        p = f"{P}layers.{i}."
        cat = lambda *ns: torch.cat([src(p + n) for n in ns], 0)  # noqa: E731
        table = {
            "attn_norm.weight": lambda: 1.0 + src(p + "input_layernorm.weight").float(),
            "ffn_norm.weight": lambda: 1.0 + src(p + "post_attention_layernorm.weight").float(),
            "ffn_gate_up.weight": lambda: cat("mlp.gate_proj.weight", "mlp.up_proj.weight"),
            "ffn_down.weight": lambda: src(p + "mlp.down_proj.weight"),
            "attn_qkv.weight": lambda: cat("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"),
            "attn_output.weight": lambda: src(p + "self_attn.o_proj.weight"),
            "attn_q_norm.weight": lambda: 1.0 + src(p + "self_attn.q_norm.weight").float(),
            "attn_k_norm.weight": lambda: 1.0 + src(p + "self_attn.k_norm.weight").float(),
            "ssm_in.weight": lambda: cat("linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight",
                                         "linear_attn.in_proj_b.weight", "linear_attn.in_proj_a.weight"),
            "ssm_out.weight": lambda: src(p + "linear_attn.out_proj.weight"),
            "ssm_conv1d.weight": lambda: src(p + "linear_attn.conv1d.weight").float().reshape(-1, cfg["linear_conv_kernel_dim"]),
            "ssm_dt.bias": lambda: src(p + "linear_attn.dt_bias").float(),
            "ssm_a": lambda: -src(p + "linear_attn.A_log").float().exp(),
            "ssm_norm.weight": lambda: src(p + "linear_attn.norm.weight").float(),
        }
        return table[rest]()

    reader = GGUFReader(path)
    counts = Counter(t.name for t in reader.tensors)
    missing = names - counts.keys()
    unexpected = counts.keys() - names
    duplicates = {n for n, count in counts.items() if count != 1}
    if missing or unexpected or duplicates:
        for label, entries in (("MISSING", missing), ("UNEXPECTED", unexpected), ("DUPLICATE", duplicates)):
            for name in sorted(entries):
                print(f"{label} {name}")
        sys.exit("GGUF tensor inventory does not match the source model")
    bad = 0
    nbytes = 0
    for t in reader.tensors:
        want = expected(t.name)
        raw = np.asarray(t.data)
        nbytes += raw.nbytes
        shape = tuple(int(d) for d in reversed(t.shape))  # GGUF lists fastest-varying dimensions first
        dtype = GGMLQuantizationType.BF16 if want.dtype == torch.bfloat16 else GGMLQuantizationType.F32
        if shape != tuple(want.shape) or t.tensor_type != dtype:
            bad += 1
            print(f"MISMATCH {t.name}: shape/type {shape}/{t.tensor_type.name}, expected {tuple(want.shape)}/{dtype.name}")
            continue
        if want.dtype == torch.bfloat16:
            ok = raw.dtype == np.uint8 or raw.dtype == np.uint16
            w = want.contiguous().view(torch.int16).numpy().view(np.uint16).reshape(-1)
            got = raw.reshape(-1).view(np.uint16)
            ok = ok and got.size == w.size and np.array_equal(got, w)
        else:
            w = want.float().contiguous().numpy().reshape(-1)
            got = raw.reshape(-1).view(np.float32)
            ok = got.size == w.size and np.array_equal(got, w)
        if not ok:
            bad += 1
            print(f"MISMATCH {t.name}")
    print(f"{len(reader.tensors) - bad}/{len(reader.tensors)} tensors exact ({nbytes / 1e9:.1f} GB compared)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
