"""PyTorch reference oracle for Clef / Clef-Flash.

Writes, per request, everything the native engine is checked against:
  golden/<name>/requests.jsonl      the request corpus (--corpus text: ref/corpus.py; vision: ref/corpus_vision.py)
  golden/<name>/encoded.jsonl       input_ids, question/option spans, option ids; with images also
                                    the image token runs and the 3D rotary positions
  golden/<name>/logits.safetensors  per-question logits (f32), keyed "<id>/<question>"
  golden/<name>/layers/<id>.safetensors   (first --dump-layers requests) the embeddings the decoder
                                          reads (image features scattered in), every decoder layer
                                          output, the final-normed hidden state, and per image its
                                          vision-tower features ("vision.<k>")
  golden/<name>/latency.json        warm single-request latency on this device

Images in a request are base64 strings (optionally data URLs), as the engine takes them; the
oracle decodes them with PIL and hands the processor the PIL images, as the reference does.

The model code is imported from the pinned, reviewed snapshot directory.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).parent))
import corpus  # noqa: E402
import corpus_vision  # noqa: E402
import golden_io  # noqa: E402


def with_pil_images(request: dict) -> dict:
    """The record the reference encodes: base64 images decoded with PIL, as a client would hand them over."""
    if not request.get("images"):
        return request
    from PIL import Image

    return dict(request, images=[Image.open(io.BytesIO(base64.b64decode(s.split(",", 1)[-1]))) for s in request["images"]])


def sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--dump-layers", type=int, default=3)
    parser.add_argument("--latency-reps", type=int, default=3)
    parser.add_argument("--only", nargs="*", help="request ids to run (default: all)")
    parser.add_argument("--safe-attn", action="store_true", help="head-chunked SDPA (MPS >= 2^32 offset bug, ref/safe_attn.py)")
    parser.add_argument("--dump-last", type=int, default=0, help="dump only the last N token rows per layer")
    parser.add_argument("--corpus", default="text", choices=["text", "vision"])
    args = parser.parse_args()

    sys.path.insert(0, str(args.model_dir.resolve()))
    from joint_schema_model import collate_records, encode_record, load_release_model

    out = Path(__file__).parent.parent / "golden" / args.name
    (out / "layers").mkdir(parents=True, exist_ok=True)

    dtype = getattr(torch, args.dtype)
    start = time.perf_counter()
    model, processor = load_release_model(args.model_dir, device=args.device, dtype=dtype)
    if args.safe_attn:
        import safe_attn
        safe_attn.install(model)
    load_s = time.perf_counter() - start
    print(f"loaded {args.model_dir} on {args.device}/{args.dtype} in {load_s:.1f}s", flush=True)
    tokenizer = processor.tokenizer
    text_model = model.language_model.model.language_model
    device = torch.device(args.device)

    tail = (lambda t: t[-args.dump_last:]) if args.dump_last > 0 else (lambda t: t)
    requests = corpus.build() if args.corpus == "text" else corpus_vision.build()
    full_model = model.language_model.model   # Qwen3_5Model: vision tower + get_rope_index + language model
    if args.only:
        requests = [r for r in requests if r["id"] in set(args.only)]
    logits_out: dict[str, torch.Tensor] = {}
    with open(out / "requests.jsonl", "w") as rf, open(out / "encoded.jsonl", "w") as ef:
        for index, request in enumerate(requests):
            rf.write(json.dumps(request, ensure_ascii=False) + "\n")
            encoded = encode_record(tokenizer, with_pil_images(request), processor=processor)
            position_ids = None
            if encoded.media is not None:
                ids = torch.tensor([list(encoded.input_ids)])
                types_ = torch.zeros_like(ids)
                offset = encoded.media["token_offset"]
                types_[0, offset:offset + len(encoded.media["mm_token_type_ids"])] = torch.tensor(encoded.media["mm_token_type_ids"])
                pos, _ = full_model.get_rope_index(ids, types_, image_grid_thw=encoded.media["image_grid_thw"])
                position_ids = pos[:, 0].tolist()
            ef.write(golden_io.encoded_line(request, encoded, position_ids))

            captured: dict[str, torch.Tensor] = {}
            hooks = []
            if index < args.dump_layers:
                # the token embeddings for a text record; with images, the embeddings handed to the
                # decoder after the vision features replaced the placeholder rows
                hooks.append(text_model.embed_tokens.register_forward_hook(
                    lambda m, i, o: captured.__setitem__("embed", tail(o[0]).float().cpu())))
                hooks.append(text_model.register_forward_pre_hook(
                    lambda m, a, kw: captured.__setitem__("embed", tail(kw["inputs_embeds"][0]).float().cpu())
                    if kw.get("inputs_embeds") is not None else None, with_kwargs=True))
                def capture_vision(m, a, kw, o):   # a hook must return None, or it replaces the output
                    for k, f in enumerate(torch.split(o.pooler_output, (kw["grid_thw"].prod(-1) // 4).tolist())):
                        captured[f"vision.{k}"] = f.float().cpu()
                hooks.append(full_model.visual.register_forward_hook(capture_vision, with_kwargs=True))
                for li, layer in enumerate(text_model.layers):
                    hooks.append(layer.register_forward_hook(
                        lambda m, i, o, li=li: captured.__setitem__(
                            f"layer.{li:02d}", tail((o[0] if isinstance(o, tuple) else o)[0]).float().cpu())))
                hooks.append(text_model.norm.register_forward_hook(
                    lambda m, i, o: captured.__setitem__("final_norm", tail(o[0]).float().cpu())))

            batch = collate_records([encoded], tokenizer.pad_token_id, device)
            with torch.inference_mode():
                logits = model(batch)[0]
            for hook in hooks:
                hook.remove()
            for q, ql in zip(encoded.questions, logits):
                logits_out[f"{request['id']}/{q.question_id}"] = ql.float().cpu().contiguous()
            if captured:
                save_file({k: v.contiguous() for k, v in captured.items()}, out / "layers" / f"{request['id']}.safetensors")
            print(f"{request['id']} tokens={len(encoded.input_ids)}", flush=True)

    save_file(logits_out, out / "logits.safetensors")

    # Warm single-request latency, end to end from encoded record to probabilities on host.
    latencies: dict[str, list[float]] = {}
    for _ in range(args.latency_reps):
        for request in requests:
            encoded = encode_record(tokenizer, with_pil_images(request), processor=processor)
            sync(args.device)
            t0 = time.perf_counter()
            batch = collate_records([encoded], tokenizer.pad_token_id, device)
            with torch.inference_mode():
                logits = model(batch)[0]
            [ql.float().softmax(-1).tolist() for ql in logits]
            sync(args.device)
            latencies.setdefault(request["id"], []).append((time.perf_counter() - t0) * 1000)
    per_request = {
        rid: {"tokens": len(encode_record(tokenizer, with_pil_images(r), processor=processor).input_ids),
              "median_ms": statistics.median(latencies[rid])}
        for rid, r in zip((r["id"] for r in requests), requests)
    }
    medians = sorted(v["median_ms"] for v in per_request.values())
    summary = {
        "device": args.device,
        "dtype": args.dtype,
        "torch": torch.__version__,
        "load_s": load_s,
        "median_ms": statistics.median(medians),
        "p95_ms": medians[min(len(medians) - 1, int(round(0.95 * (len(medians) - 1))))],
        "per_request": per_request,
    }
    (out / "latency.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_request"}, indent=2))


if __name__ == "__main__":
    main()
