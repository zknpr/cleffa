"""Small, complete Omni checkpoint; no model downloads or remote code imports."""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file


def snapshot(path: Path):
    text = dict(model_type="qwen3_omni_moe_text", hidden_size=8, num_hidden_layers=1,
                num_attention_heads=2, num_key_value_heads=1, head_dim=128,
                intermediate_size=4, moe_intermediate_size=4, num_experts=3,
                num_experts_per_tok=2, norm_topk_prob=True, decoder_sparse_step=1,
                shared_expert_intermediate_size=0, mlp_only_layers=[], hidden_act="silu",
                attention_bias=False, use_qk_norm=True, use_sliding_window=False,
                sliding_window=None, rms_norm_eps=1e-6, rope_theta=1000000,
                rope_scaling=dict(rope_type="default", mrope_interleaved=True,
                                  mrope_section=[24, 20, 20]),
                vocab_size=16, max_position_embeddings=65536, tie_word_embeddings=False)
    vision = dict(model_type="qwen3_omni_moe_vision_encoder", depth=3, hidden_size=8,
                  intermediate_size=12, num_heads=2, out_hidden_size=8,
                  patch_size=2, temporal_patch_size=2, spatial_merge_size=2,
                  image_size=4, in_channels=3, in_chans=3, apply_vit_abs_pos_embed=True,
                  hidden_act="gelu_pytorch_tanh", deepstack_visual_indexes=[0, 1, 2])
    audio = dict(model_type="qwen3_omni_moe_audio_encoder", d_model=8,
                 encoder_layers=1, num_hidden_layers=1, encoder_attention_heads=2,
                 encoder_ffn_dim=16, downsample_hidden_size=2, num_mel_bins=128,
                 output_dim=8, activation_function="gelu", n_window=50,
                 n_window_infer=800, conv_chunksize=500, max_source_positions=1500,
                 scale_embedding=False)
    thinker = dict(text_config=text, vision_config=vision, audio_config=audio,
                   image_token_id=6, video_token_id=7, vision_start_token_id=8,
                   vision_end_token_id=9, audio_token_id=10, audio_start_token_id=11,
                   audio_end_token_id=12, position_id_per_seconds=13, seconds_per_chunk=2)
    config = dict(model_type="qwen3_omni_moe", thinker_config=thinker)
    ip = dict(image_processor_type="Qwen2VLImageProcessor", resample=3,
              image_mean=[.5]*3, image_std=[.5]*3, rescale_factor=1/255,
              do_convert_rgb=True, do_resize=True, do_rescale=True, do_normalize=True,
              patch_size=2, merge_size=2, temporal_patch_size=2,
              size=dict(shortest_edge=3136, longest_edge=12845056))
    ap = dict(feature_extractor_type="WhisperFeatureExtractor", sampling_rate=16000,
              feature_size=128, n_fft=400, hop_length=160, dither=0.,
              padding_side="right", padding_value=0., return_attention_mask=True)
    vp={k:v for k,v in ip.items() if k!='image_processor_type'}
    vp.update(video_processor_type='Qwen2VLVideoProcessor',do_sample_frames=False,return_metadata=False)
    proc = dict(image_processor=ip, video_processor=vp, feature_extractor=ap)
    hc = dict(hidden_size=8, width=8, routing_layers=1, layers=1, heads=2, feedforward=16)
    tok = dict(model=dict(type="BPE", vocab={"a": 0, "b": 1}, merges=[["a", "b"]]),
               added_tokens=[dict(id=2, content="<eos>", special=True),
                             dict(id=3, content="<pad>", special=True)],
               pre_tokenizer=None, normalizer=None)
    for name, data in (("config.json", config), ("processor_config.json", proc),
                       ("joint_head_config.json", hc), ("tokenizer.json", tok),
                       ("tokenizer_config.json", dict(eos_token="<eos>", pad_token="<pad>"))):
        (path / name).write_text(json.dumps(data))
    weights = {}

    def add(name, shape):
        # Different offsets expose expert swaps even when all expert shapes match.
        x = torch.arange(int(np.prod(shape)), dtype=torch.float32).reshape(shape)
        weights[name] = ((x % 23 - 11) / 16 + len(weights) / 8).to(torch.bfloat16)

    p = "thinker.model."
    add(p + "embed_tokens.weight", (16, 8))
    add(p + "norm.weight", (8,))
    add("thinker.lm_head.weight", (16, 8))
    p += "layers.0."
    for n in ("input_layernorm", "post_attention_layernorm"):
        add(p+n+".weight", (8,))
    for n, shape in (("q_proj", (256, 8)), ("k_proj", (128, 8)), ("v_proj", (128, 8)),
                     ("o_proj", (8, 256)), ("q_norm", (128,)), ("k_norm", (128,))):
        add(p+"self_attn."+n+".weight", shape)
    add(p+"mlp.gate.weight", (3, 8))
    for e in range(3):
        for n, shape in (("gate_proj", (4, 8)), ("up_proj", (4, 8)), ("down_proj", (8, 4))):
            add(p+f"mlp.experts.{e}.{n}.weight", shape)
    p = "thinker.visual."
    add(p+"patch_embed.proj.weight", (8, 3, 2, 2, 2))
    add(p+"patch_embed.proj.bias", (8,))
    add(p+"pos_embed.weight", (4, 8))
    for i in range(3):
        for n, shape in (("norm1", (8,)), ("norm2", (8,)), ("attn.qkv", (24, 8)),
                         ("attn.proj", (8, 8)), ("mlp.linear_fc1", (12, 8)),
                         ("mlp.linear_fc2", (8, 12))):
            for part in ("weight", "bias"):
                add(p+f"blocks.{i}.{n}.{part}", shape if part == "weight" else (shape[0],))
    for k in range(-1, 3):
        m = "merger." if k < 0 else f"merger_list.{k}."
        for n, shape in (("ln_q", (8 if k < 0 else 32,)),
                         ("mlp.0", (32, 32)), ("mlp.2", (8, 32))):
            for part in ("weight", "bias"):
                add(p+m+n+"."+part, shape if part == "weight" else (shape[0],))
    p = "thinker.audio_tower."
    for n, shape in (("conv2d1", (2, 1, 3, 3)), ("conv2d2", (2, 2, 3, 3)),
                     ("conv2d3", (2, 2, 3, 3)), ("ln_post", (8,)),
                     ("proj1", (8, 8)), ("proj2", (8, 8))):
        add(p+n+".weight", shape)
        add(p+n+".bias", (shape[0],))
    add(p+"conv_out.weight", (8, 32))
    for n, shape in (("self_attn.q_proj", (8, 8)), ("self_attn.k_proj", (8, 8)),
                     ("self_attn.v_proj", (8, 8)), ("self_attn.out_proj", (8, 8)),
                     ("self_attn_layer_norm", (8,)), ("final_layer_norm", (8,)),
                     ("fc1", (16, 8)), ("fc2", (8, 16))):
        add(p+"layers.0."+n+".weight", shape)
        add(p+"layers.0."+n+".bias", (shape[0],))
    add("talker.ignored.weight", (1,))
    add("code2wav.ignored.weight", (1,))
    save_file(weights, path / "weights.safetensors")
    (path / "model.safetensors.index.json").write_text(json.dumps(
        dict(weight_map={n: "weights.safetensors" for n in weights})))
    save_file({"scale": torch.tensor(0.5), "proj.weight": torch.ones(8, 8, dtype=torch.bfloat16)},
              path / "joint_head.safetensors")
    return config, weights
