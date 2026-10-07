"""Convert a Clef / Clef-Flash HF snapshot into a single GGUF for the native engine.

Text backbone + vision tower + joint head + tokenizer + image-processor parameters.

Precision policy: nothing is quantized and no stored value is rounded.
  - Matrices stay BF16, byte-identical to the safetensors source.
  - Qwen3.5 RMSNorm weights are stored zero-centred and applied as (1 + w) in f32
    (Qwen3_5RMSNorm.forward). We fold 1 + w here in f32 and store F32, which is
    exactly what the reference computes; folding in BF16 would round.
  - The DeltaNet gated norm (Qwen3_5RMSNormGated) uses plain w, and A_log enters as
    -exp(A_log.float()); both are stored F32 (bf16 -> f32 is exact; the exp is the
    same f32 op the reference runs).
  - Joint head tensors keep their stored dtype.
  - Vision tower matrices stay BF16 (the patch projection is the Conv3d weight viewed as
    [1152, 3*2*16*16], a byte-exact reshape in the processor's patch order); its LayerNorm
    weights/biases and linear biases are stored F32 (bf16 -> f32 is exact). The learned
    position table stays BF16 and is interpolated in f32 by the engine, as the FP32 reference
    does with the BF16 values.
  - Per-layer projections that consume the same input are concatenated along rows
    (exact byte concatenation) so each becomes one GEMM: attention [q|k|v], DeltaNet
    [qkv|z|b|a], MLP [gate|up].

Tensor layouts are not permuted. In particular in_proj_qkv rows stay [q | k | v] with
repeat_interleave head mapping (v-head h reads k-head h // (Hv / Hk)), and q_proj rows
stay per-head [query(head_dim) | gate(head_dim)]. The engine indexes the HF layout directly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gguf
import numpy as np
import torch
from safetensors import safe_open

ARCH = "clef"
PREFIX = "model.language_model."


class Source:
    """Lazy reader over a sharded safetensors checkpoint."""

    def __init__(self, model_dir: Path) -> None:
        index = json.loads((model_dir / "model.safetensors.index.json").read_text())
        self.weight_map: dict[str, str] = index["weight_map"]
        self.model_dir = model_dir
        self.handles: dict = {}

    def _handle(self, name: str):
        shard = self.weight_map[name]
        if shard not in self.handles:
            self.handles[shard] = safe_open(self.model_dir / shard, "pt")
        return self.handles[shard]

    def weight_shape(self, name: str) -> list[int]:
        return self._handle(name).get_slice(name).get_shape()

    def get(self, name: str) -> torch.Tensor:
        return self._handle(name).get_tensor(name)


def bf16_bytes(t: torch.Tensor) -> np.ndarray:
    if t.dtype != torch.bfloat16:
        raise ValueError(f"expected bf16 matrix, got {t.dtype}")
    # Reinterpret, never convert: the GGUF payload must equal the source bytes.
    return t.contiguous().view(torch.int16).numpy().view(np.uint16)


def f32(t: torch.Tensor) -> np.ndarray:
    return t.float().contiguous().numpy()


def write_tokenizer(w: gguf.GGUFWriter, model_dir: Path) -> None:
    tok = json.loads((model_dir / "tokenizer.json").read_text())
    model = tok["model"]
    if model["type"] != "BPE":
        raise ValueError(f"unsupported tokenizer model {model['type']}")
    vocab: dict[str, int] = model["vocab"]
    added = {a["id"]: a for a in tok["added_tokens"]}
    size = max(max(vocab.values()), max(added) if added else 0) + 1
    tokens = [""] * size
    types = [gguf.TokenType.UNUSED] * size
    for text, idx in vocab.items():
        tokens[idx] = text
        types[idx] = gguf.TokenType.NORMAL
    for idx, a in added.items():
        tokens[idx] = a["content"]
        types[idx] = gguf.TokenType.CONTROL if a["special"] else gguf.TokenType.USER_DEFINED
    merges = [m if isinstance(m, str) else " ".join(m) for m in model["merges"]]

    cfg = json.loads((model_dir / "tokenizer_config.json").read_text())
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("qwen35")
    w.add_token_list(tokens)
    w.add_token_types(types)
    w.add_token_merges(merges)
    ids = {a["content"]: i for i, a in added.items()}
    w.add_eos_token_id(ids[cfg["eos_token"]])
    w.add_pad_token_id(ids[cfg["pad_token"]])
    # The exact pre-tokenizer / normalizer definitions, so the engine can refuse a mismatch.
    w.add_string("clef.tokenizer.pre_tokenizer", json.dumps(tok["pre_tokenizer"], sort_keys=True))
    w.add_string("clef.tokenizer.normalizer", json.dumps(tok["normalizer"], sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--vocab-only", action="store_true", help="metadata + tokenizer only (for tokenizer tests)")
    args = parser.parse_args()

    config = json.loads((args.model_dir / "config.json").read_text())
    if config.get("model_type") != "qwen3_5":
        raise ValueError(f"unsupported model_type {config.get('model_type')}")
    text = config["text_config"]
    head_config = json.loads((args.model_dir / "joint_head_config.json").read_text())
    if head_config["hidden_size"] != text["hidden_size"]:
        raise ValueError("joint head hidden_size does not match backbone")
    rope = text["rope_parameters"]
    if rope.get("rope_type", "default") != "default":
        raise ValueError(f"unsupported rope_type {rope.get('rope_type')}")
    if text.get("attention_bias") or not text.get("attn_output_gate", False):
        raise ValueError("expected attention_bias=false and attn_output_gate=true")
    if text["hidden_act"] != "silu" or text.get("mlp_only_layers"):
        raise ValueError("unexpected MLP configuration")

    vision = config["vision_config"]
    if vision.get("model_type") != "qwen3_5_vision" or vision.get("in_channels", 3) != 3:
        raise ValueError("unsupported vision tower")
    if vision.get("deepstack_visual_indexes"):
        raise ValueError("deepstack injection is not implemented")
    # The rotary kernels pick each frequency's position axis as lane % 3: interleaved M-RoPE with
    # section [11, 11, 10]. The GGUF records the section (checked again at load), not the
    # interleaving, so refuse anything else here.
    if rope.get("mrope_interleaved") is not True or list(rope.get("mrope_section", [])) != [11, 11, 10]:
        raise ValueError("only interleaved M-RoPE with mrope_section [11, 11, 10] is implemented")
    if vision.get("hidden_act") != "gelu_pytorch_tanh" or vision["hidden_size"] % vision["num_heads"]:
        raise ValueError("unexpected vision block configuration")
    if vision["out_hidden_size"] != text["hidden_size"]:
        raise ValueError("vision out_hidden_size does not match the backbone")
    proc = json.loads((args.model_dir / "processor_config.json").read_text())
    ip = proc["image_processor"]
    # The engine reproduces this preprocessing exactly (clef_image.c); refuse anything else.
    if (ip["image_processor_type"] != "Qwen2VLImageProcessor" or ip["resample"] != 3 or
            ip["image_mean"] != [0.5] * 3 or ip["image_std"] != [0.5] * 3 or
            abs(ip["rescale_factor"] - 1 / 255) > 1e-12 or not ip["do_convert_rgb"] or
            not ip["do_resize"] or not ip["do_rescale"] or not ip["do_normalize"] or
            ip["patch_size"] != vision["patch_size"] or ip["merge_size"] != vision["spatial_merge_size"] or
            ip["temporal_patch_size"] != vision["temporal_patch_size"]):
        raise ValueError("unsupported image processor configuration")
    for key in ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id"):
        if not isinstance(config.get(key), int):
            raise ValueError(f"missing {key}")

    src = None if args.vocab_only else Source(args.model_dir)
    w = gguf.GGUFWriter(args.output, ARCH)
    w.add_name(args.model_dir.name)
    n_layer = text["num_hidden_layers"]
    w.add_block_count(n_layer)
    w.add_embedding_length(text["hidden_size"])
    w.add_feed_forward_length(text["intermediate_size"])
    w.add_head_count(text["num_attention_heads"])
    w.add_head_count_kv(text["num_key_value_heads"])
    w.add_key_length(text["head_dim"])
    w.add_value_length(text["head_dim"])
    w.add_layer_norm_rms_eps(text["rms_norm_eps"])
    w.add_rope_freq_base(float(rope["rope_theta"]))
    w.add_rope_dimension_count(int(text["head_dim"] * rope.get("partial_rotary_factor", 1.0)))
    w.add_array(f"{ARCH}.rope.mrope_section", [int(x) for x in rope.get("mrope_section", [])])
    w.add_vocab_size(text["vocab_size"])
    w.add_array(f"{ARCH}.layer_types", [1 if t == "full_attention" else 0 for t in text["layer_types"]])
    w.add_uint32(f"{ARCH}.ssm.conv_kernel", text["linear_conv_kernel_dim"])
    w.add_uint32(f"{ARCH}.ssm.k_heads", text["linear_num_key_heads"])
    w.add_uint32(f"{ARCH}.ssm.v_heads", text["linear_num_value_heads"])
    w.add_uint32(f"{ARCH}.ssm.k_head_dim", text["linear_key_head_dim"])
    w.add_uint32(f"{ARCH}.ssm.v_head_dim", text["linear_value_head_dim"])
    for key in ("width", "routing_layers", "layers", "heads", "feedforward"):
        w.add_uint32(f"{ARCH}.head.{key}", int(head_config[key]))
    w.add_uint32(f"{ARCH}.vision.block_count", vision["depth"])
    w.add_uint32(f"{ARCH}.vision.embedding_length", vision["hidden_size"])
    w.add_uint32(f"{ARCH}.vision.feed_forward_length", vision["intermediate_size"])
    w.add_uint32(f"{ARCH}.vision.attention.head_count", vision["num_heads"])
    w.add_uint32(f"{ARCH}.vision.patch_size", vision["patch_size"])
    w.add_uint32(f"{ARCH}.vision.spatial_merge_size", vision["spatial_merge_size"])
    w.add_uint32(f"{ARCH}.vision.temporal_patch_size", vision["temporal_patch_size"])
    w.add_uint32(f"{ARCH}.vision.position_embeddings", vision["num_position_embeddings"])
    w.add_uint32(f"{ARCH}.vision.image_token_id", config["image_token_id"])
    w.add_uint32(f"{ARCH}.vision.start_token_id", config["vision_start_token_id"])
    w.add_uint32(f"{ARCH}.vision.end_token_id", config["vision_end_token_id"])
    w.add_uint32(f"{ARCH}.vision.video_token_id", config["video_token_id"])
    # smart_resize bounds: pixels of the resized image (processor size.shortest_edge / longest_edge)
    w.add_uint32(f"{ARCH}.vision.image.min_pixels", int(ip["size"]["shortest_edge"]))
    w.add_uint32(f"{ARCH}.vision.image.max_pixels", int(ip["size"]["longest_edge"]))
    w.add_string(f"{ARCH}.vision.image_processor", json.dumps(ip, sort_keys=True))
    write_tokenizer(w, args.model_dir)
    if args.vocab_only:
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.close()
        print(f"wrote {args.output} (vocab only)")
        return

    # Two passes so peak memory is one tensor, not the model (the 27B is ~55 GB):
    # declare every tensor's shape/dtype, write the header, then stream the data.
    plan: list[tuple[str, tuple[int, ...], str, object]] = []

    def matrix(name: str, *sources: str) -> None:
        """BF16 matrix; several sources are concatenated along rows (exact byte concat)."""
        shapes = [tuple(src.weight_shape(s_)) for s_ in sources]
        if any(sh[1:] != shapes[0][1:] for sh in shapes):
            raise ValueError(f"{name}: cannot concatenate {shapes}")
        shape = (sum(sh[0] for sh in shapes),) + shapes[0][1:]
        plan.append((name, shape, "bf16", lambda: torch.cat([src.get(s_) for s_ in sources], 0)))

    def f32_tensor(name: str, shape: tuple[int, ...], fn) -> None:
        plan.append((name, shape, "f32", fn))

    def norm_1p(name: str, target: str) -> None:
        f32_tensor(target, tuple(src.weight_shape(name)), lambda: 1.0 + src.get(name).float())

    matrix("token_embd.weight", PREFIX + "embed_tokens.weight")
    norm_1p(PREFIX + "norm.weight", "output_norm.weight")
    matrix("output.weight", "lm_head.weight")

    for i, layer_type in enumerate(text["layer_types"]):
        p = f"{PREFIX}layers.{i}."
        b = f"blk.{i}."
        norm_1p(p + "input_layernorm.weight", b + "attn_norm.weight")
        norm_1p(p + "post_attention_layernorm.weight", b + "ffn_norm.weight")
        # [gate; up]: one GEMM, the SwiGLU kernel reads both halves
        matrix(b + "ffn_gate_up.weight", p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight")
        matrix(b + "ffn_down.weight", p + "mlp.down_proj.weight")
        if layer_type == "full_attention":
            a = p + "self_attn."
            # [q_proj (per head: query | gate); k_proj; v_proj]
            matrix(b + "attn_qkv.weight", a + "q_proj.weight", a + "k_proj.weight", a + "v_proj.weight")
            matrix(b + "attn_output.weight", a + "o_proj.weight")
            norm_1p(a + "q_norm.weight", b + "attn_q_norm.weight")
            norm_1p(a + "k_norm.weight", b + "attn_k_norm.weight")
        elif layer_type == "linear_attention":
            a = p + "linear_attn."
            # [in_proj_qkv; in_proj_z; in_proj_b; in_proj_a]
            matrix(b + "ssm_in.weight", a + "in_proj_qkv.weight", a + "in_proj_z.weight",
                   a + "in_proj_b.weight", a + "in_proj_a.weight")
            matrix(b + "ssm_out.weight", a + "out_proj.weight")
            cshape = tuple(src.weight_shape(a + "conv1d.weight"))
            f32_tensor(b + "ssm_conv1d.weight", (cshape[0], cshape[-1]),
                       lambda a=a, cshape=cshape: src.get(a + "conv1d.weight").float().reshape(cshape[0], cshape[-1]))
            f32_tensor(b + "ssm_dt.bias", tuple(src.weight_shape(a + "dt_bias")),
                       lambda a=a: src.get(a + "dt_bias").float())
            f32_tensor(b + "ssm_a", tuple(src.weight_shape(a + "A_log")),
                       lambda a=a: -src.get(a + "A_log").float().exp())
            f32_tensor(b + "ssm_norm.weight", tuple(src.weight_shape(a + "norm.weight")),
                       lambda a=a: src.get(a + "norm.weight").float())
        else:
            raise ValueError(f"layer {i}: unknown layer type {layer_type}")

    V = "model.visual."
    E, P, Tp = vision["hidden_size"], vision["patch_size"], vision["temporal_patch_size"]
    # Conv3d [E, 3, Tp, P, P] -> [E, 3*Tp*P*P]: the processor flattens each patch as
    # (channel, temporal, y, x), the same order, so this is a byte-exact view.
    plan.append(("v.patch_embd.weight", (E, 3 * Tp * P * P), "bf16",
                 lambda: src.get(V + "patch_embed.proj.weight").reshape(E, 3 * Tp * P * P)))
    f32_tensor("v.patch_embd.bias", (E,), lambda: src.get(V + "patch_embed.proj.bias").float())
    matrix("v.pos_embd.weight", V + "pos_embed.weight")
    for i in range(vision["depth"]):
        p = f"{V}blocks.{i}."
        b = f"v.blk.{i}."
        for hf_name, name in (("norm1", "ln1"), ("norm2", "ln2")):
            for part in ("weight", "bias"):
                f32_tensor(f"{b}{name}.{part}", tuple(src.weight_shape(f"{p}{hf_name}.{part}")),
                           lambda p=p, hf_name=hf_name, part=part: src.get(f"{p}{hf_name}.{part}").float())
        for hf_name, name in (("attn.qkv", "attn_qkv"), ("attn.proj", "attn_out"),
                              ("mlp.linear_fc1", "ffn_up"), ("mlp.linear_fc2", "ffn_down")):
            matrix(f"{b}{name}.weight", f"{p}{hf_name}.weight")
            f32_tensor(f"{b}{name}.bias", tuple(src.weight_shape(f"{p}{hf_name}.bias")),
                       lambda p=p, hf_name=hf_name: src.get(f"{p}{hf_name}.bias").float())
    for part in ("weight", "bias"):
        f32_tensor(f"v.post_ln.{part}", tuple(src.weight_shape(f"{V}merger.norm.{part}")),
                   lambda part=part: src.get(f"{V}merger.norm.{part}").float())
    for hf_name, name in (("linear_fc1", "mm.0"), ("linear_fc2", "mm.2")):
        matrix(f"v.{name}.weight", f"{V}merger.{hf_name}.weight")
        f32_tensor(f"v.{name}.bias", tuple(src.weight_shape(f"{V}merger.{hf_name}.bias")),
                   lambda hf_name=hf_name: src.get(f"{V}merger.{hf_name}.bias").float())

    head = safe_open(args.model_dir / "joint_head.safetensors", "pt")
    for name in sorted(head.keys()):
        sl = head.get_slice(name)
        shape = tuple(sl.get_shape()) or (1,)  # GGUF has no rank-0 tensors (logit scales, gate)
        kind = {"BF16": "bf16", "F32": "f32"}[sl.get_dtype()]
        plan.append(("head." + name, shape, kind, lambda name=name, shape=shape: head.get_tensor(name).reshape(shape)))

    for name, shape, kind, _ in plan:
        n = int(np.prod(shape))
        if kind == "bf16":
            w.add_tensor_info(name, shape, np.dtype(np.uint16), n * 2, raw_dtype=gguf.GGMLQuantizationType.BF16)
        else:
            w.add_tensor_info(name, shape, np.dtype(np.float32), n * 4)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for i, (name, shape, kind, fn) in enumerate(plan):
        t = fn()
        if tuple(t.shape) != shape:
            raise ValueError(f"{name}: produced {tuple(t.shape)}, declared {shape}")
        w.write_tensor_data(bf16_bytes(t) if kind == "bf16" else f32(t))
        if i % 50 == 0 or i == len(plan) - 1:
            print(f"  [{i + 1}/{len(plan)}] {name}", flush=True)
    w.close()
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
