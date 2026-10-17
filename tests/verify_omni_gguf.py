"""Independently verify Omni format v1 against a safetensors snapshot.

Do not import converter code here: inventory, fusion order, dtype promotion, and
metadata checks must catch converter mistakes as well as corrupted artifacts.
"""

import json
import math
from collections import Counter
from pathlib import Path

import gguf
import numpy as np
import torch
from safetensors import safe_open


def verify_omni(model_dir: Path, path: Path) -> None:
    r = gguf.GGUFReader(path)
    cfg = json.loads((model_dir/"config.json").read_text())["thinker_config"]
    t, v, a = (cfg[k+"_config"] for k in ("text", "vision", "audio"))
    hc = json.loads((model_dir/"joint_head_config.json").read_text())
    pc = json.loads((model_dir/"processor_config.json").read_text())

    def metadata(key, want, typ):
        f = r.fields.get(key)
        if f is None or f.types[0] != typ or f.contents() != want:
            raise ValueError(f"{key}: invalid or missing metadata (expected {want!r})")

    u32, f32, string = gguf.GGUFValueType.UINT32, gguf.GGUFValueType.FLOAT32, gguf.GGUFValueType.STRING
    metadata("general.architecture", "clef", string)
    metadata("clef.backbone", "qwen3_omni_moe", string)
    metadata("clef.format_version", 1, u32)
    field = r.fields.get("clef.capabilities")
    caps = field.contents() if field else None
    if caps not in (1, 3, 7):
        raise ValueError("clef.capabilities: unsupported export")
    metadata("clef.capabilities", caps, u32)
    integers = {"context_default": 64000, "context_length": t["max_position_embeddings"],
                "block_count": t["num_hidden_layers"], "embedding_length": t["hidden_size"],
                "feed_forward_length": t["moe_intermediate_size"], "attention.head_count": t["num_attention_heads"],
                "attention.head_count_kv": t["num_key_value_heads"], "attention.key_length": t["head_dim"],
                "attention.value_length": t["head_dim"], "rope.dimension_count": t["head_dim"],
                "vocab_size": t["vocab_size"], "moe.expert_count": t["num_experts"],
                "moe.expert_used_count": t["num_experts_per_tok"]}
    integers.update({"head."+k: hc[k] for k in ("width", "routing_layers", "layers", "heads", "feedforward")})
    integers.update({"vision."+k: value for k, value in {
        "block_count": v["depth"], "embedding_length": v["hidden_size"], "feed_forward_length": v["intermediate_size"],
        "attention.head_count": v["num_heads"], "patch_size": v["patch_size"], "spatial_merge_size": v["spatial_merge_size"],
        "temporal_patch_size": v["temporal_patch_size"], "position_embeddings": (v["image_size"]//v["patch_size"])**2,
        "image_token_id": cfg["image_token_id"], "video_token_id": cfg["video_token_id"],
        "start_token_id": cfg["vision_start_token_id"], "end_token_id": cfg["vision_end_token_id"],
        "image.min_pixels": 65536, "image.max_pixels": 1048576,
        "video.min_pixels": 131072, "video.max_pixels": 786432}.items()})
    integers.update({"audio."+k: cfg[k] for k in ("audio_token_id", "audio_start_token_id", "audio_end_token_id",
                                                 "position_id_per_seconds", "seconds_per_chunk")})
    for key, value in integers.items():
        metadata("clef."+key, value, u32)
    metadata("clef.moe.normalize_topk", t["norm_topk_prob"], gguf.GGUFValueType.BOOL)
    metadata("clef.rope.freq_base", float(np.float32(t["rope_theta"])), f32)
    metadata("clef.attention.layer_norm_rms_epsilon", float(np.float32(t["rms_norm_eps"])), f32)
    for key, values in (("rope.mrope_section", t["rope_scaling"]["mrope_section"]),
                         ("layer_types", [1]*t["num_hidden_layers"]),
                         ("vision.deepstack_layers", v["deepstack_visual_indexes"])):
        metadata("clef."+key, values, gguf.GGUFValueType.ARRAY)
        if r.fields["clef."+key].types[-1] != gguf.GGUFValueType.INT32:
            raise ValueError(f"clef.{key}: expected INT32 array")
    for key, value in (("vision.image_processor", pc["image_processor"]),
                        ("vision.video_processor",pc['video_processor']), ("audio.config", a),
                        ("audio.processor", pc["feature_extractor"])):
        metadata("clef."+key, json.dumps(value, sort_keys=True), string)
    # Tokenizer content is independently reconstructed, including unused ID gaps.
    tok = json.loads((model_dir/"tokenizer.json").read_text())
    tc = json.loads((model_dir/"tokenizer_config.json").read_text())
    added = {x["id"]: x for x in tok["added_tokens"]}
    vocab = tok["model"]["vocab"]
    n = max(max(vocab.values()), max(added, default=0))+1
    tokens, types = [""]*n, [int(gguf.TokenType.UNUSED)]*n
    for text, idx in vocab.items():
        tokens[idx], types[idx] = text, int(gguf.TokenType.NORMAL)
    for idx, x in added.items():
        tokens[idx], types[idx] = x["content"], int(gguf.TokenType.CONTROL if x["special"] else gguf.TokenType.USER_DEFINED)
    for key, values in (("tokens", tokens), ("token_type", types),
                         ("merges", [x if isinstance(x, str) else " ".join(x) for x in tok["model"]["merges"]])):
        metadata("tokenizer.ggml."+key, values, gguf.GGUFValueType.ARRAY)
    for key in ("eos", "pad"):
        gguf_key = "padding" if key == "pad" else key
        metadata("tokenizer.ggml."+gguf_key+"_token_id", next(i for i, x in added.items() if x["content"] == tc[key+"_token"]), u32)
    metadata("tokenizer.ggml.model", "gpt2", string)
    metadata("tokenizer.ggml.pre", "qwen35", string)
    for key in ("normalizer", "pre_tokenizer"):
        metadata("clef.tokenizer."+key, json.dumps(tok[key], sort_keys=True), string)

    index = json.loads((model_dir/"model.safetensors.index.json").read_text())["weight_map"]
    handles = {}

    def source(name):
        shard = index[name]
        if shard not in handles:
            handles.clear()
            handles[shard] = safe_open(model_dir/shard, "pt")
        return handles[shard].get_tensor(name)

    # Each target is a sequence of source slices, optionally stacked by expert.
    expected = {}
    known = set()

    def add(dst, names, *, fp32=False, experts=False, flatten=False, enabled=True):
        known.update(names)
        if enabled:
            expected[dst] = (names, fp32, experts, flatten)

    add("token_embd.weight", ["thinker.model.embed_tokens.weight"])
    add("output.weight", ["thinker.lm_head.weight"])
    add("output_norm.weight", ["thinker.model.norm.weight"], fp32=True)
    for i in range(t["num_hidden_layers"]):
        p, b = f"thinker.model.layers.{i}.", f"blk.{i}."
        for suffix, name in (("input_layernorm", "attn_norm"), ("post_attention_layernorm", "ffn_norm"),
                              ("self_attn.q_norm", "attn_q_norm"), ("self_attn.k_norm", "attn_k_norm")):
            add(b+name+".weight", [p+suffix+".weight"], fp32=True)
        add(b+"attn_output.weight", [p+"self_attn.o_proj.weight"])
        add(b+"attn_qkv.weight", [p+"self_attn."+x+"_proj.weight" for x in ("q", "k", "v")])
        add(b+"ffn_router.weight", [p+"mlp.gate.weight"])
        add(b+"ffn_gate_up_exps.weight", [p+f"mlp.experts.{j}.{x}_proj.weight"
            for j in range(t["num_experts"]) for x in ("gate", "up")], experts=True)
        add(b+"ffn_down_exps.weight", [p+f"mlp.experts.{j}.down_proj.weight" for j in range(t["num_experts"])], experts=True)
    vp = "thinker.visual."
    add("v.patch_embd.weight", [vp+"patch_embed.proj.weight"], flatten=True, enabled=caps & 2)
    add("v.patch_embd.bias", [vp+"patch_embed.proj.bias"], fp32=True, enabled=caps & 2)
    add("v.pos_embd.weight", [vp+"pos_embed.weight"], enabled=caps & 2)
    for i in range(v["depth"]):
        for dst, name in (("ln1", "norm1"), ("ln2", "norm2"), ("attn_qkv", "attn.qkv"),
                           ("attn_out", "attn.proj"), ("ffn_up", "mlp.linear_fc1"), ("ffn_down", "mlp.linear_fc2")):
            for part in ("weight", "bias"):
                add(f"v.blk.{i}.{dst}.{part}", [vp+f"blocks.{i}.{name}.{part}"],
                    fp32=part == "bias" or dst.startswith("ln"), enabled=caps & 2)
    mergers = [("v.", "merger.")] + [(f"v.deepstack.{i}.", f"merger_list.{i}.") for i in range(len(v["deepstack_visual_indexes"]))]
    for dst, src in mergers:
        for dn, sn in (("post_ln", "ln_q"), ("mm.0", "mlp.0"), ("mm.2", "mlp.2")):
            for part in ("weight", "bias"):
                add(dst+dn+"."+part, [vp+src+sn+"."+part], fp32=dn == "post_ln" or part == "bias", enabled=caps & 2)
    anames = [n+"."+part for n in ("conv2d1", "conv2d2", "conv2d3", "ln_post", "proj1", "proj2") for part in ("weight", "bias")]
    anames.append("conv_out.weight")
    for i in range(a["encoder_layers"]):
        anames.extend(f"layers.{i}.{n}.{part}" for n in (
            "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.out_proj",
            "self_attn_layer_norm", "final_layer_norm", "fc1", "fc2") for part in ("weight", "bias"))
    for n in anames:
        add("a."+n, ["thinker.audio_tower."+n], fp32=n.endswith(".bias") or "layer_norm." in n or n.startswith("ln_post."), enabled=caps & 4)
    unknown = {n for n in index if not n.startswith(("talker.", "code2wav."))} - known
    if unknown:
        raise ValueError(f"unexpected source tensors: {sorted(unknown)}")
    head = safe_open(model_dir/"joint_head.safetensors", "pt")
    for name in head.keys():
        add("head."+name, [name])
    counts = Counter(x.name for x in r.tensors)
    missing, unexpected = set(expected)-counts.keys(), counts.keys()-set(expected)
    duplicates = {n for n, count in counts.items() if count != 1}
    if missing or unexpected or duplicates:
        raise ValueError(f"inventory: missing={sorted(missing)} unexpected={sorted(unexpected)} duplicate={sorted(duplicates)}")
    total = 0
    for tensor in r.tensors:
        names, fp32, experts, flatten = expected[tensor.name]
        get = head.get_tensor if tensor.name.startswith("head.") else source
        shapes = []
        offset = 0
        expected_dtype = None
        for name in names:
            x = get(name)
            if x.dtype not in (torch.bfloat16, torch.float32):
                raise ValueError(f"{name}: unsupported source dtype")
            if fp32:
                x = x.float()
            elif not tensor.name.startswith("head.") and x.dtype != torch.bfloat16:
                raise ValueError(f"{name}: expected BF16 source matrix")
            shapes.append(tuple(x.shape) or (1,))
            kind = gguf.GGMLQuantizationType.BF16 if x.dtype == torch.bfloat16 else gguf.GGMLQuantizationType.F32
            if tensor.tensor_type != kind or (expected_dtype is not None and kind != expected_dtype):
                raise ValueError(f"{tensor.name}: dtype mismatch")
            expected_dtype = kind
            # Compare bits for both types: as floats, +0.0 equals -0.0.
            want = x.contiguous().view(torch.int16).numpy().view(np.uint16).reshape(-1) if x.dtype == torch.bfloat16 else x.numpy().reshape(-1).view(np.uint32)
            raw = tensor.data.reshape(-1).view(want.dtype)
            if offset+want.size > raw.size:
                raise ValueError(f"{tensor.name}: shape/payload too short")
            # Bound the comparison allocation even for the embedding matrix.
            for k in range(0, want.size, 1 << 20):
                if not np.array_equal(raw[offset+k:offset+min(k+(1 << 20), want.size)], want[k:k+(1 << 20)]):
                    raise ValueError(f"{tensor.name}: payload mismatch in {name} at element {k}")
            offset += want.size
            del x, want, raw
        shape = (sum(s[0] for s in shapes),)+shapes[0][1:]
        if any(s[1:] != shapes[0][1:] for s in shapes):
            raise ValueError(f"{tensor.name}: incompatible source shapes")
        if experts:
            shape = (t["num_experts"], shape[0]//t["num_experts"], *shape[1:])
        elif flatten:
            shape = (shape[0], math.prod(shape[1:]))
        if tuple(reversed(tensor.shape)) != shape or offset != tensor.n_elements:
            raise ValueError(f"{tensor.name}: shape mismatch (expected {shape})")
        total += tensor.n_bytes
    print(f"{len(r.tensors)}/{len(r.tensors)} Omni tensors exact ({total/1e9:.3f} GB compared), capabilities={caps}")
