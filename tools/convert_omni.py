"""Lossless Omni thinker export. No talker/code2wav execution or weight conversion.

Expert tensors have [expert, output, input] order, with [gate; up] inside each
expert. Stream individual source matrices instead of materializing an expert bank.
The GGUF library writes the header and tensor table; payloads use the same alignment
but allow several bounded source slices per declared tensor.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import gguf
import numpy as np
from safetensors import safe_open

from convert import Source, bf16_bytes, f32, write_tokenizer


class OmniSource(Source):
    def _handle(self, name):
        shard = self.weight_map[name]
        if shard not in self.handles:
            # Closing old mappings bounds resident source pages, not just Python
            # allocations. A currently yielded tensor retains its own mapping.
            self.handles.clear()
        return super()._handle(name)


@dataclass
class Tensor:
    name: str
    shape: tuple[int, ...]
    kind: str
    sources: list[tuple[str, tuple[int, ...]]]
    capability: int


def validate_config(config, proc, head):
    if config.get("model_type") != "qwen3_omni_moe":
        raise ValueError("expected qwen3_omni_moe")
    c = config["thinker_config"]
    t, v, a = (c[k + "_config"] for k in ("text", "vision", "audio"))
    for key in ("hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
                "head_dim", "moe_intermediate_size", "num_experts", "num_experts_per_tok", "vocab_size"):
        if type(t[key]) is not int or not 0 < t[key] < 2**31:
            raise ValueError(f"invalid text {key}")
    checks = dict(attention_bias=False, hidden_act="silu", norm_topk_prob=True,
                  decoder_sparse_step=1, shared_expert_intermediate_size=0,
                  mlp_only_layers=[], use_qk_norm=True, use_sliding_window=False,
                  sliding_window=None, tie_word_embeddings=False)
    for key, value in checks.items():
        if t.get(key, value) != value:
            raise ValueError(f"unsupported {key}: {t.get(key)}")
    rope = t["rope_scaling"]
    if (rope.get("rope_type", "default") != "default" or not rope.get("mrope_interleaved") or
            rope.get("mrope_section") != [24, 20, 20] or t["head_dim"] != 128 or
            t["rope_theta"] != 1000000 or t["max_position_embeddings"] != 65536):
        raise ValueError("unsupported Omni rotary/context configuration")
    if (t["num_attention_heads"] % t["num_key_value_heads"] or
            t["num_experts_per_tok"] > t["num_experts"] or head["hidden_size"] != t["hidden_size"] or
            not math.isfinite(t["rms_norm_eps"]) or t["rms_norm_eps"] <= 0):
        raise ValueError("inconsistent Omni text/head dimensions")
    if (v["model_type"] != "qwen3_omni_moe_vision_encoder" or
            v["hidden_act"] != "gelu_pytorch_tanh" or v.get("in_channels", 3) != 3 or
            v.get("in_chans", 3) != 3 or not v.get("apply_vit_abs_pos_embed", True) or
            v["hidden_size"] % v["num_heads"] or v["out_hidden_size"] != t["hidden_size"] or
            v["image_size"] % v["patch_size"]):
        raise ValueError("unsupported vision configuration")
    ds = v["deepstack_visual_indexes"]
    if len(ds) != 3 or ds != sorted(set(ds)) or any(type(i) is not int or not 0 <= i < v["depth"] for i in ds):
        raise ValueError("unsupported vision deepstack configuration")
    ip = proc["image_processor"]
    if (ip["image_processor_type"] != "Qwen2VLImageProcessor" or ip["resample"] != 3 or
            ip["image_mean"] != [.5]*3 or ip["image_std"] != [.5]*3 or
            abs(ip["rescale_factor"] - 1/255) > 1e-12 or
            not all(ip[k] for k in ("do_convert_rgb", "do_resize", "do_rescale", "do_normalize")) or
            ip["patch_size"] != v["patch_size"] or ip["merge_size"] != v["spatial_merge_size"] or
            ip["temporal_patch_size"] != v["temporal_patch_size"]):
        raise ValueError("unsupported image processor")
    vp = proc.get('video_processor', {})
    shared = ('resample','image_mean','image_std','rescale_factor','do_convert_rgb','do_resize',
              'do_rescale','do_normalize','patch_size','merge_size','temporal_patch_size')
    if (vp.get('video_processor_type') != 'Qwen2VLVideoProcessor' or
            any(vp.get(k) != ip[k] for k in shared) or vp.get('do_sample_frames') is not False or
            vp.get('return_metadata') is not False or
            vp.get('size') != dict(shortest_edge=3136,longest_edge=12845056)):
        raise ValueError('unsupported video processor')
    ap = proc["feature_extractor"]
    for key, value in dict(feature_extractor_type="WhisperFeatureExtractor", sampling_rate=16000,
                           feature_size=128, n_fft=400, hop_length=160, dither=0.,
                           padding_side="right", padding_value=0.).items():
        if ap.get(key) != value:
            raise ValueError(f"unsupported audio processor {key}")
    if (a["model_type"] != "qwen3_omni_moe_audio_encoder" or a["num_mel_bins"] != 128 or
            a["output_dim"] != t["hidden_size"] or a["activation_function"] != "gelu" or
            a["d_model"] % a["encoder_attention_heads"] or a["n_window"] != 50 or
            a["n_window_infer"] != 800 or a["conv_chunksize"] != 500 or a.get("scale_embedding", False)):
        raise ValueError("unsupported audio encoder configuration")
    for key in ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id",
                "audio_token_id", "audio_start_token_id", "audio_end_token_id"):
        if type(c.get(key)) is not int or not 0 <= c[key] < t["vocab_size"]:
            raise ValueError(f"invalid {key}")
    if c["position_id_per_seconds"] != 13 or c["seconds_per_chunk"] != 2:
        raise ValueError("unsupported temporal position configuration")
    return c, t, v, a


def layout(t, v, a):
    plan = []

    def add(name, shape, kind, sources, cap=1):
        plan.append(Tensor(name, tuple(shape), kind, [(n, tuple(s)) for n, s in sources], cap))

    def one(name, source, shape, kind="bf16", cap=1):
        add(name, shape, kind, [(source, shape)], cap)

    h, f, e, d = (t[k] for k in ("hidden_size", "moe_intermediate_size", "num_experts", "head_dim"))
    q, kv = t["num_attention_heads"]*d, t["num_key_value_heads"]*d
    one("token_embd.weight", "thinker.model.embed_tokens.weight", (t["vocab_size"], h))
    one("output.weight", "thinker.lm_head.weight", (t["vocab_size"], h))
    one("output_norm.weight", "thinker.model.norm.weight", (h,), "f32")
    for i in range(t["num_hidden_layers"]):
        p, b = f"thinker.model.layers.{i}.", f"blk.{i}."
        for src, dst in (("input_layernorm", "attn_norm"), ("post_attention_layernorm", "ffn_norm")):
            one(b+dst+".weight", p+src+".weight", (h,), "f32")
        add(b+"attn_qkv.weight", (q+2*kv, h), "bf16",
            [(p+"self_attn."+n+"_proj.weight", (rows, h)) for n, rows in (("q", q), ("k", kv), ("v", kv))])
        one(b+"attn_output.weight", p+"self_attn.o_proj.weight", (h, q))
        for n in ("q", "k"):
            one(b+f"attn_{n}_norm.weight", p+f"self_attn.{n}_norm.weight", (d,), "f32")
        one(b+"ffn_router.weight", p+"mlp.gate.weight", (e, h))
        add(b+"ffn_gate_up_exps.weight", (e, 2*f, h), "bf16",
            [(p+f"mlp.experts.{j}.{n}_proj.weight", (f, h)) for j in range(e) for n in ("gate", "up")])
        add(b+"ffn_down_exps.weight", (e, h, f), "bf16",
            [(p+f"mlp.experts.{j}.down_proj.weight", (h, f)) for j in range(e)])
    p = "thinker.visual."
    vh, vf, m = v["hidden_size"], v["intermediate_size"], v["spatial_merge_size"]**2*v["hidden_size"]
    patch = (vh, 3, v["temporal_patch_size"], v["patch_size"], v["patch_size"])
    add("v.patch_embd.weight", (vh, math.prod(patch[1:])), "bf16", [(p+"patch_embed.proj.weight", patch)], 2)
    one("v.patch_embd.bias", p+"patch_embed.proj.bias", (vh,), "f32", 2)
    one("v.pos_embd.weight", p+"pos_embed.weight", ((v["image_size"]//v["patch_size"])**2, vh), cap=2)
    for i in range(v["depth"]):
        for src, dst, shape in (("norm1", "ln1", (vh,)), ("norm2", "ln2", (vh,)),
                                ("attn.qkv", "attn_qkv", (3*vh, vh)), ("attn.proj", "attn_out", (vh, vh)),
                                ("mlp.linear_fc1", "ffn_up", (vf, vh)), ("mlp.linear_fc2", "ffn_down", (vh, vf))):
            for part in ("weight", "bias"):
                sh = shape if part == "weight" else (shape[0],)
                one(f"v.blk.{i}.{dst}.{part}", p+f"blocks.{i}.{src}.{part}", sh,
                    "bf16" if len(sh) > 1 else "f32", 2)
    for k in range(-1, len(v["deepstack_visual_indexes"])):
        src = p+("merger." if k < 0 else f"merger_list.{k}.")
        dst = "v." if k < 0 else f"v.deepstack.{k}."
        for n, target, shape in (("ln_q", "post_ln", (vh if k < 0 else m,)),
                                  ("mlp.0", "mm.0", (m, m)), ("mlp.2", "mm.2", (h, m))):
            for part in ("weight", "bias"):
                sh = shape if part == "weight" else (shape[0],)
                one(dst+target+"."+part, src+n+"."+part, sh, "bf16" if len(sh)>1 else "f32", 2)
    ah, af, ac = a["d_model"], a["encoder_ffn_dim"], a["downsample_hidden_size"]
    ashapes = {"conv_out.weight": (ah, ac*((a["num_mel_bins"]+7)//8))}
    pairs = [("conv2d1", (ac, 1, 3, 3)), ("conv2d2", (ac, ac, 3, 3)),
             ("conv2d3", (ac, ac, 3, 3)), ("ln_post", (ah,)), ("proj1", (ah, ah)), ("proj2", (h, ah))]
    for i in range(a["encoder_layers"]):
        pairs.extend((f"layers.{i}."+n, shape) for n, shape in (
            ("self_attn.q_proj", (ah, ah)), ("self_attn.k_proj", (ah, ah)),
            ("self_attn.v_proj", (ah, ah)), ("self_attn.out_proj", (ah, ah)),
            ("self_attn_layer_norm", (ah,)), ("final_layer_norm", (ah,)),
            ("fc1", (af, ah)), ("fc2", (ah, af))))
    for n, shape in pairs:
        ashapes[n+".weight"] = shape
        ashapes[n+".bias"] = (shape[0],)
    for n, shape in ashapes.items():
        one("a."+n, "thinker.audio_tower."+n, shape, "bf16" if len(shape)>1 else "f32", 4)
    return plan


def convert_omni(model_dir: Path, output: Path, *, vocab_only=False, modalities="all") -> None:
    caps = {"text": 1, "vision": 3, "all": 7}[modalities]
    config = json.loads((model_dir/"config.json").read_text())
    proc = json.loads((model_dir/"processor_config.json").read_text())
    head_config = json.loads((model_dir/"joint_head_config.json").read_text())
    c, t, v, a = validate_config(config, proc, head_config)
    w = gguf.GGUFWriter(output, "clef")
    w.add_name(model_dir.name)
    w.add_string("clef.backbone", "qwen3_omni_moe")
    for key, value in {"format_version": 1, "capabilities": caps, "context_default": 64000,
                       "context_length": t["max_position_embeddings"], "block_count": t["num_hidden_layers"],
                       "embedding_length": t["hidden_size"], "feed_forward_length": t["moe_intermediate_size"],
                       "attention.head_count": t["num_attention_heads"], "attention.head_count_kv": t["num_key_value_heads"],
                       "attention.key_length": t["head_dim"], "attention.value_length": t["head_dim"],
                       "rope.dimension_count": t["head_dim"], "vocab_size": t["vocab_size"],
                       "moe.expert_count": t["num_experts"], "moe.expert_used_count": t["num_experts_per_tok"]}.items():
        w.add_uint32("clef."+key, value)
    w.add_bool("clef.moe.normalize_topk", True)
    w.add_float32("clef.attention.layer_norm_rms_epsilon", t["rms_norm_eps"])
    w.add_float32("clef.rope.freq_base", float(t["rope_theta"]))
    w.add_array("clef.rope.mrope_section", t["rope_scaling"]["mrope_section"])
    w.add_array("clef.layer_types", [1]*t["num_hidden_layers"])
    for key in ("width", "routing_layers", "layers", "heads", "feedforward"):
        w.add_uint32("clef.head."+key, head_config[key])
    for key, value in {"block_count": v["depth"], "embedding_length": v["hidden_size"],
                       "feed_forward_length": v["intermediate_size"], "attention.head_count": v["num_heads"],
                       "patch_size": v["patch_size"], "spatial_merge_size": v["spatial_merge_size"],
                       "temporal_patch_size": v["temporal_patch_size"],
                       "position_embeddings": (v["image_size"]//v["patch_size"])**2,
                       "image_token_id": c["image_token_id"], "video_token_id": c["video_token_id"],
                       "start_token_id": c["vision_start_token_id"], "end_token_id": c["vision_end_token_id"],
                       "image.min_pixels": 65536, "image.max_pixels": 1048576}.items():
        w.add_uint32("clef.vision."+key, value)
    w.add_key_value("clef.vision.deepstack_layers", v["deepstack_visual_indexes"],
                    gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.INT32)
    w.add_string("clef.vision.image_processor", json.dumps(proc["image_processor"], sort_keys=True))
    w.add_string("clef.vision.video_processor", json.dumps(proc["video_processor"], sort_keys=True))
    w.add_uint32('clef.vision.video.min_pixels',128*32*32)
    w.add_uint32('clef.vision.video.max_pixels',768*32*32)
    w.add_string("clef.audio.config", json.dumps(a, sort_keys=True))
    w.add_string("clef.audio.processor", json.dumps(proc["feature_extractor"], sort_keys=True))
    for key in ("audio_token_id", "audio_start_token_id", "audio_end_token_id", "position_id_per_seconds", "seconds_per_chunk"):
        w.add_uint32("clef.audio."+key, c[key])
    write_tokenizer(w, model_dir)
    if vocab_only:
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.close()
        return
    src = OmniSource(model_dir)
    plan = layout(t, v, a)
    known = {name for item in plan for name, _ in item.sources}
    unexpected = set(src.weight_map) - known
    unexpected = {n for n in unexpected if not n.startswith(("talker.", "code2wav."))}
    if unexpected:
        raise ValueError(f"unrecognized source tensors: {sorted(unexpected)}")
    plan = [item for item in plan if item.capability & caps]
    # Fail shape/dtype validation before opening the output. No full matrices are read.
    for item in plan:
        for name, shape in item.sources:
            sl = src._handle(name).get_slice(name)
            dtype = sl.get_dtype()
            if tuple(sl.get_shape()) != shape:
                raise ValueError(f"{name}: shape {sl.get_shape()}, expected {shape}")
            if dtype not in (("BF16",) if item.kind == "bf16" else ("BF16", "F32")):
                raise ValueError(f"{name}: expected {'BF16' if item.kind == 'bf16' else 'BF16/F32'}, got {dtype}")
    head = safe_open(model_dir/"joint_head.safetensors", "pt")
    for name in sorted(head.keys()):
        sl = head.get_slice(name)
        kind = {"BF16": "bf16", "F32": "f32"}[sl.get_dtype()]
        shape = tuple(sl.get_shape()) or (1,)
        plan.append(Tensor("head."+name, shape, kind, [(name, shape)], 1))
    for item in plan:
        nbytes = math.prod(item.shape)*(2 if item.kind == "bf16" else 4)
        w.add_tensor_info(item.name, item.shape, np.dtype(np.uint16 if item.kind == "bf16" else np.float32),
                          nbytes, raw_dtype=gguf.GGMLQuantizationType.BF16 if item.kind == "bf16" else None)
    try:
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_ti_data_to_file()
        if len(w.fout) != 1 or not np.little_endian:
            raise ValueError("Omni streaming requires a single little-endian output")
        fp = w.fout[0]
        for i, item in enumerate(plan):
            w.write_padding(fp, fp.tell())
            written = 0
            for name, _ in item.sources:
                tensor = head.get_tensor(name) if item.name.startswith("head.") else src.get(name)
                data = bf16_bytes(tensor) if item.kind == "bf16" else f32(tensor)
                data.tofile(fp)
                written += data.nbytes
                del data, tensor
            if written != math.prod(item.shape)*(2 if item.kind == "bf16" else 4):
                raise ValueError(f"{item.name}: incorrect streamed payload length")
            w.write_padding(fp, written)
            if i % 50 == 0 or i == len(plan)-1:
                print(f"  [{i+1}/{len(plan)}] {item.name}", flush=True)
    finally:
        w.close()
    print(f"wrote {output} (capabilities={caps})")
