"""Is the FP32 oracle's long-context attention right? (diagnostic for the 27B at 16k tokens)

Runs one corpus request through layers 0..L-1 in f32 exactly as oracle_f32_stream.py does,
then computes the first full-attention layer's attention output for the last N queries
three ways: (1) the HF attention interface the oracle used (SDPA on MPS, f32), (2) the same
on CPU, (3) a float64 CPU reference. Prints each one's relative error against float64.

Usage: check_attn_long.py MODEL_DIR REQUEST_ID [--last 512]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import corpus  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("request_id")
    ap.add_argument("--last", type=int, default=512)
    ap.add_argument("--safe-attn", action="store_true", help="head-chunked SDPA (MPS >= 2^32 offset bug, ref/safe_attn.py)")
    args = ap.parse_args()
    sys.path.insert(0, str(args.model_dir.resolve()))
    from joint_schema_model import collate_records, encode_record, load_release_model
    from transformers.masking_utils import create_causal_mask
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb, ALL_ATTENTION_FUNCTIONS, eager_attention_forward

    dev = torch.device("mps")
    model, processor = load_release_model(args.model_dir, device="mps", dtype=torch.bfloat16)
    if args.safe_attn:
        import safe_attn
        safe_attn.install(model)
    tok = processor.tokenizer
    tm = model.language_model.model.language_model
    req = next(r for r in corpus.build() if r["id"] == args.request_id)
    b = collate_records([encode_record(tok, req)], tok.pad_token_id, dev)

    with torch.inference_mode():
        h = tm.embed_tokens(b["input_ids"]).float()
        T = h.shape[1]
        pos = torch.arange(T, device=dev).view(1, 1, -1).expand(4, 1, -1)
        text_pos, pos3 = pos[0], pos[1:]
        mask = create_causal_mask(config=tm.config, inputs_embeds=h, attention_mask=b["attention_mask"],
                                  past_key_values=None, position_ids=text_pos)
        lin_mask = tm._update_linear_attn_mask(b["attention_mask"], None)
        tm.rotary_emb.float()
        pe = tm.rotary_emb(h, pos3)
        L = tm.config.layer_types.index("full_attention")
        for li in range(L):
            layer = tm.layers[li].float()
            h = layer(h, position_embeddings=pe, attention_mask=lin_mask, position_ids=text_pos,
                      past_key_values=None, use_cache=False)
            layer.to(torch.bfloat16)
        layer = tm.layers[L].float()
        x = layer.input_layernorm(h)
        att = layer.self_attn
        hd = att.head_dim
        shape = (1, T, -1, hd)
        q, gate = torch.chunk(att.q_proj(x).view(1, T, -1, hd * 2), 2, dim=-1)
        q = att.q_norm(q.reshape(shape)).transpose(1, 2)
        k = att.k_norm(att.k_proj(x).view(shape)).transpose(1, 2)
        v = att.v_proj(x).view(shape).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, *pe)
        n = args.last
        print(f"request {args.request_id}: T={T}, layer {L} ({att.config._attn_implementation}), "
              f"heads q={q.shape[1]} kv={k.shape[1]}, checking the last {n} queries")

        # (1) exactly what the oracle ran: the configured attention interface on MPS
        fn = ALL_ATTENTION_FUNCTIONS.get_interface(att.config._attn_implementation, eager_attention_forward)
        out_mps, _ = fn(att, q, k, v, mask, dropout=0.0, scaling=att.scaling)
        out_mps = out_mps[0, -n:].float().cpu()
        # (2) same interface on CPU, f32
        qc, kc, vc = q.cpu(), k.cpu(), v.cpu()
        mask_c = None if mask is None else mask.cpu()
        out_cpu, _ = fn(att, qc, kc, vc, mask_c, dropout=0.0, scaling=att.scaling)
        out_cpu = out_cpu[0, -n:].float()
        # (3) float64 reference for the last n queries
        rep = q.shape[1] // k.shape[1]
        q64 = qc[0, :, -n:].double()
        k64 = kc[0].double().repeat_interleave(rep, 0)
        v64 = vc[0].double().repeat_interleave(rep, 0)
        s = (q64 @ k64.transpose(1, 2)) * att.scaling
        qpos = torch.arange(T - n, T).view(1, n, 1)
        s = s.masked_fill(torch.arange(T).view(1, 1, T) > qpos, float("-inf"))
        ref = (torch.softmax(s, -1) @ v64).transpose(0, 1)          # [n, heads, hd]
        rel = lambda a: float((a.double() - ref).norm() / ref.norm())
        print(f"  MPS {att.config._attn_implementation} f32 vs float64: {rel(out_mps):.2e}")
        print(f"  CPU {att.config._attn_implementation} f32 vs float64: {rel(out_cpu):.2e}")
        per_head = [(float((out_mps[:, hh].double() - ref[:, hh]).norm() / ref[:, hh].norm())) for hh in range(ref.shape[1])]
        print("  MPS per-head rel error:", " ".join(f"{e:.1e}" for e in per_head))


if __name__ == "__main__":
    main()
