"""FP32 reference for models too large to hold in FP32 (Clef 27B: ~108 GB).

Same semantics as `oracle.py --dtype float32` (BF16 weights upcast exactly, every op in f32),
computed layer by layer: the model stays in BF16, each decoder layer is upcast to f32, all
requests' hidden states pass through it, and it is cast back. Peak extra memory is one layer
in f32 plus the hidden states. Mirrors Qwen3_5Model.forward (image features scattered into the
embeddings, 3D rotary positions from get_rope_index), Qwen3_5TextModel.forward and
ClefModel.forward, one record per batch. With --corpus vision the vision tower (small) is upcast
whole and run per request in f32, as oracle.py --dtype float32 runs it.

Writes golden/<name>/{requests.jsonl, encoded.jsonl, logits.safetensors, layers/<id>.safetensors}.
Validate against oracle.py --dtype float32 on a model that fits (tests: clef-flash).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).parent))
import corpus  # noqa: E402
import corpus_vision  # noqa: E402
import golden_io  # noqa: E402
from oracle import with_pil_images  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--name", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dump-layers", type=int, default=3)
    ap.add_argument("--safe-attn", action="store_true", help="head-chunked SDPA (MPS >= 2^32 offset bug, ref/safe_attn.py)")
    ap.add_argument("--only", nargs="*", help="request ids to run (default: all); all of them are dumped")
    ap.add_argument("--dump-last", type=int, default=0, help="dump only the last N token rows per layer")
    ap.add_argument("--corpus", default="text", choices=["text", "vision"])
    args = ap.parse_args()

    sys.path.insert(0, str(args.model_dir.resolve()))
    from joint_schema_model import collate_records, encode_record, load_release_model
    from transformers.masking_utils import create_causal_mask

    out = Path(__file__).parent.parent / "golden" / args.name
    (out / "layers").mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)
    t0 = time.perf_counter()
    model, processor = load_release_model(args.model_dir, device=args.device, dtype=torch.bfloat16)
    if args.safe_attn:
        import safe_attn
        safe_attn.install(model)
    print(f"loaded in {time.perf_counter() - t0:.1f}s", flush=True)
    tok = processor.tokenizer
    full = model.language_model.model                        # Qwen3_5Model: visual + get_rope_index + language model
    tm = full.language_model                                 # Qwen3_5TextModel
    base = model.language_model

    requests = corpus.build() if args.corpus == "text" else corpus_vision.build()
    if args.only:
        requests = [r for r in requests if r["id"] in set(args.only)]
        args.dump_layers = len(requests)
    tail = (lambda t: t[-args.dump_last:]) if args.dump_last > 0 else (lambda t: t)
    encoded = [encode_record(tok, with_pil_images(r), processor=processor) for r in requests]
    batches = [collate_records([e], tok.pad_token_id, dev) for e in encoded]
    positions = [None] * len(requests)

    with torch.inference_mode():
        if any(e.media is not None for e in encoded):
            full.visual.float()
        hidden, extras = [], []
        for i, (e, b) in enumerate(zip(encoded, batches)):
            h = tm.embed_tokens(b["input_ids"]).float()
            T = h.shape[1]
            media = b["media"]
            dump_vision = {}
            if e.media is not None:
                # Qwen3_5Model.forward: image features over the placeholder rows, positions from get_rope_index
                feats = full.visual(media["pixel_values"].float(), grid_thw=media["image_grid_thw"]).pooler_output
                for k, f in enumerate(torch.split(feats, (media["image_grid_thw"].prod(-1) // 4).tolist())):
                    dump_vision[f"vision.{k}"] = f.float().cpu().clone()
                mask_img = (b["input_ids"] == full.config.image_token_id).unsqueeze(-1).expand_as(h)
                h = h.masked_scatter(mask_img, feats.to(h.dtype))
                pos3, _ = full.get_rope_index(b["input_ids"], media["mm_token_type_ids"], image_grid_thw=media["image_grid_thw"],
                                              attention_mask=b["attention_mask"])
                pos3 = pos3.to(dev)
                positions[i] = pos3[:, 0].cpu().tolist()
                text_pos = None   # a (3, bs, T) position_ids leaves Qwen3_5TextModel's text_position_ids None
            else:
                pos = torch.arange(T, device=dev).view(1, 1, -1).expand(4, 1, -1)
                text_pos, pos3 = pos[0], pos[1:]
            mask = create_causal_mask(config=tm.config, inputs_embeds=h, attention_mask=b["attention_mask"],
                                      past_key_values=None, position_ids=text_pos)
            lin_mask = tm._update_linear_attn_mask(b["attention_mask"], None)
            tm.rotary_emb.float()
            pe = tm.rotary_emb(h, pos3)
            hidden.append(h)
            extras.append((mask, lin_mask, pe, text_pos, dump_vision))
        dumps = [dict(embed=tail(h[0]).cpu().clone(), **ex[4]) for h, ex in zip(hidden[: args.dump_layers], extras)]

        for li, layer in enumerate(tm.layers):
            layer.float()
            for i, (mask, lin_mask, pe, text_pos, _) in enumerate(extras):
                lm = lin_mask if tm.config.layer_types[li] == "linear_attention" else mask
                hidden[i] = layer(hidden[i], position_embeddings=pe, attention_mask=lm,
                                  position_ids=text_pos, past_key_values=None, use_cache=False)
                if i < args.dump_layers:
                    dumps[i][f"layer.{li:02d}"] = tail(hidden[i][0]).cpu().clone()
            layer.to(torch.bfloat16)
            if args.device == "mps":
                torch.mps.empty_cache()
            print(f"layer {li} done", flush=True)

        tm.norm.float()
        model.head.float()
        out_w = base.get_output_embeddings().weight.float()
        logits_out = {}
        for i, (req, enc, b) in enumerate(zip(requests, encoded, batches)):
            last = tm.norm(hidden[i])
            if i < args.dump_layers:
                dumps[i]["final_norm"] = tail(last[0]).cpu().clone()
            logits = model.head(last, b["input_ids"], b["attention_mask"], b["records"], out_w)[0]
            for q, ql in zip(enc.questions, logits):
                logits_out[f"{req['id']}/{q.question_id}"] = ql.float().cpu().contiguous()

    with open(out / "requests.jsonl", "w") as f:
        for r in requests:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # tests/test_parity.py needs it (token ids, expected questions); it was missing here (review #4)
    with open(out / "encoded.jsonl", "w") as f:
        for r, enc, pos in zip(requests, encoded, positions):
            f.write(golden_io.encoded_line(r, enc, pos))
    save_file(logits_out, out / "logits.safetensors")
    for req, d in zip(requests, dumps):
        save_file({k: v.contiguous() for k, v in d.items()}, out / "layers" / f"{req['id']}.safetensors")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
