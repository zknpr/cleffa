"""Parity of the C request encoder and response builder against joint_schema_model.

Usage: test_record.py MODEL.gguf HF_TOKENIZER_DIR REFERENCE_CODE_DIR
  encode : C encode_request == Python systemone() validation + encode_record (ids, spans, option ids);
           with images also the image token runs and the 3D rotary positions of Qwen3_5Model.get_rope_index
  respond: C response JSON == Python systemone_answer over the same float32 probabilities
"""

from __future__ import annotations

import base64
import io
import json
import random
import struct
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ref"))
import corpus  # noqa: E402

KEYS = ["a", "b", "B", "é", "é", "10", "2", "id with space", "顧客", "🚨", "", "\"q\"", "x" * 40]
VALUES = [None, True, False, 0, -0.0, 1e-9, 3, 12345678901234567890, "", "text", "Ünï\ncode", [1, "x"], {"z": 1, "a": [None]}]


def rand_question(rng: random.Random) -> dict:
    t = rng.choice(["noul", "choice", "score", "noul", "choice", "score", "bogus"])
    q: dict = {"type": t}
    r = rng.random()
    if r < 0.25:
        pass
    elif r < 0.35:
        q["instructions"] = ""
    elif r < 0.45:
        q["instructions"] = None
    elif r < 0.55:
        q["instructions"] = rng.choice(VALUES[1:])
    else:
        q["instructions"] = rng.choice(["Is it urgent?", "Pick one.", "Rate it", "顧客の意図は?"])
    if t == "choice":
        keys = rng.sample(KEYS, rng.randint(0, 6))
        q["criteria"] = {k: rng.choice(VALUES + ["desc " + k]) for k in keys}
    elif t == "score":
        q["criteria"] = [rng.choice(VALUES + ["level"]) for _ in range(rng.randint(0, 6))]
    elif t == "noul" and rng.random() < 0.4:
        q["criteria"] = rng.choice([{}, {"true": "Yes."}, {"false": None}, {"true": 1, "false": [2], "other": 3}])
    return q


def rand_request(rng: random.Random) -> dict:
    qs = {}
    for _ in range(rng.randint(0, 5)):
        qs[rng.choice(KEYS + ["q1", "q2", "q3"])] = rand_question(rng)
    return {
        "model": rng.choice(["clef", "clef-flash"]),
        "state": rng.choice(VALUES + ["Short state.", {"nested": {"b": 2, "a": 1}}, "word " * rng.randint(0, 3000)]),
        "questions": qs,
    }


def rand_image(rng: random.Random):
    """A small deterministic PNG or JPEG in one of the forms the engine takes: the hosted API's data
    URL (prefix in any case) or {"content_type", "base64"} object, or a bare base64 string."""
    h, w = rng.randint(1, 120), rng.randint(1, 120)
    arr = np.array([[((x * 37 + y * 11 + rng.randint(0, 3)) % 256, (x ^ y) & 255, (x * y) % 256) for x in range(w)] for y in range(h)], np.uint8)
    buf = io.BytesIO()
    jpeg = rng.random() < 0.3
    if jpeg:
        Image.fromarray(arr).save(buf, "JPEG", quality=rng.randint(60, 95))
    else:
        Image.fromarray(arr).save(buf, "PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    form = rng.random()
    if form < 0.3:
        return rng.choice(["data:", "DATA:", "Data:"]) + ("image/jpeg" if jpeg else "image/png") + ";base64," + b64
    if form < 0.6:
        return {"content_type": "image/jpeg" if jpeg else "image/png", "base64": b64}
    return b64


def image_payload(item) -> bytes | None:
    """The engine's rule for one images entry: decoded bytes, or None where it rejects
    (unsupported or mislabeled content_type or data URL media type, a non-base64 data URL)."""
    if isinstance(item, str):
        mtype = ""
        if item[:5].lower() == "data:":
            head, _, payload = item.partition(",")
            if not head.endswith(";base64"):
                return None
            mtype = head[5:].split(";", 1)[0].lower()
            if mtype not in ("", "image/png", "image/jpeg"):   # WebP and anything else: unsupported
                return None
            item = payload
        try:
            data = base64.b64decode(item, validate=True)
        except Exception:
            return None
        if mtype:   # a declared type must match the signature, like content_type below
            is_png, is_jpeg = data[:8] == b"\x89PNG\r\n\x1a\n", data[:2] == b"\xff\xd8"
            if (mtype == "image/png") != is_png or (mtype == "image/jpeg") != is_jpeg:
                return None
        return data
    if not isinstance(item, dict) or not isinstance(item.get("content_type"), str) or not isinstance(item.get("base64"), str):
        return None
    if item["content_type"] not in ("image/png", "image/jpeg") or item["base64"][:5].lower() == "data:":
        return None
    try:
        data = base64.b64decode(item["base64"], validate=True)
    except Exception:
        return None
    is_png, is_jpeg = data[:8] == b"\x89PNG\r\n\x1a\n", data[:2] == b"\xff\xd8"
    if (item["content_type"] == "image/png") != is_png or (item["content_type"] == "image/jpeg") != is_jpeg:
        return None
    return data


def image_requests(rng: random.Random) -> list[dict]:
    """Requests with images: 1..3 images, sizes that upscale, downscale and hit odd grids, optional
    media_kwargs bounds, and the placeholder-injection and video cases the engine must reject."""
    out = []
    for i in range(40):
        req = {"model": "clef-flash", "state": rng.choice(["A photo.", {"task": "Review the attached receipt."}, ""]),
               "images": [rand_image(rng) for _ in range(rng.randint(1, 3))],
               "questions": {"legible": {"type": "noul", "instructions": "Is it legible?"},
                             "kind": {"type": "choice", "criteria": {"photo": None, "diagram": "A diagram", "text": None}}}}
        if i % 4 == 3:
            req["media_kwargs"] = {"max_pixels": rng.choice([65536, 102400, 200000]), "min_pixels": rng.choice([1024, 65536])}
            req["media_kwargs"]["min_pixels"] = min(req["media_kwargs"]["min_pixels"], req["media_kwargs"]["max_pixels"])
        out.append(req)
    out.append({"model": "clef-flash", "state": "<|image_pad|> injected", "images": [rand_image(rng)], "questions": {"q": {"type": "noul"}}})
    out.append({"model": "clef-flash", "state": "s", "videos": [[0]], "questions": {"q": {"type": "noul"}}})
    out.append({"model": "clef-flash", "state": "s", "images": "notalist", "questions": {"q": {"type": "noul"}}})
    out.append({"model": "clef-flash", "state": "s", "images": [], "questions": {"q": {"type": "noul"}}})
    out.append({"model": "clef-flash", "state": "s", "images": None, "questions": {"q": {"type": "noul"}}})
    out.append({"model": "clef-flash", "state": "s", "images": ["bm90IGFuIGltYWdl"], "questions": {"q": {"type": "noul"}}})
    out.append({"model": "clef-flash", "state": "s", "images": [rand_image(rng)], "media_kwargs": {"patch_size": 14}, "questions": {"q": {"type": "noul"}}})
    out.append({"model": "clef-flash", "state": "s", "images": [rand_image(rng)], "media_kwargs": {"max_pixels": 65536}, "questions": {"q": {"type": "noul"}}})
    png = base64.b64encode(_png_bytes()).decode()
    out.append({"model": "clef-flash", "state": "s", "images": [{"content_type": "image/webp", "base64": png}], "questions": {"q": {"type": "noul"}}})
    out.append({"model": "clef-flash", "state": "s", "images": [{"content_type": "image/jpeg", "base64": png}], "questions": {"q": {"type": "noul"}}})   # mislabeled
    out.append({"model": "clef-flash", "state": "s", "images": [{"content_type": "image/png"}], "questions": {"q": {"type": "noul"}}})
    out.append({"model": "clef-flash", "state": "s", "images": [{"content_type": "image/png", "base64": "data:image/png;base64," + png}], "questions": {"q": {"type": "noul"}}})
    out.append({"model": "clef-flash", "state": "s", "images": ["data:image/png," + png], "questions": {"q": {"type": "noul"}}})   # not base64-marked
    out.append({"model": "clef-flash", "state": "s", "images": [{"content_type": "image/png", "base64": png}, "DATA:image/png;base64," + png], "questions": {"q": {"type": "noul"}}})
    # review #3: the data URL's media type is checked like content_type (mislabeled, WebP, other types
    # rejected; no type leaves the signature to decide), media_kwargs integers are range-checked, and an
    # image whose resized grid cannot fit a request is refused before any resize or patch allocation
    out.append({"model": "clef-flash", "state": "s", "images": ["data:image/jpeg;base64," + png], "questions": {"q": {"type": "noul"}}})   # mislabeled data URL
    out.append({"model": "clef-flash", "state": "s", "images": ["data:IMAGE/PNG;base64," + png], "questions": {"q": {"type": "noul"}}})    # MIME types are case-insensitive
    out.append({"model": "clef-flash", "state": "s", "images": ["data:image/gif;base64," + png], "questions": {"q": {"type": "noul"}}})
    out.append({"model": "clef-flash", "state": "s", "images": ["data:image/webp;base64," + png], "questions": {"q": {"type": "noul"}}})
    out.append({"model": "clef-flash", "state": "s", "images": ["data:;base64," + png], "questions": {"q": {"type": "noul"}}})
    out.append({"model": "clef-flash", "state": "s", "images": [png], "media_kwargs": {"min_pixels": 99999999999999999999, "max_pixels": 99999999999999999999}, "questions": {"q": {"type": "noul"}}})
    out.append({"model": "clef-flash", "state": "s", "images": [png], "media_kwargs": {"min_pixels": 1.5, "max_pixels": 65536}, "questions": {"q": {"type": "noul"}}})
    out.append({"model": "clef-flash", "state": "s", "images": [png], "media_kwargs": {"min_pixels": 67108864, "max_pixels": 67108864}, "questions": {"q": {"type": "noul"}}})   # 40x40 -> 8192x8192
    return out


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.zeros((40, 40, 3), np.uint8)).save(buf, "PNG")
    return buf.getvalue()


def pil_images(request: dict) -> list:
    payloads = [image_payload(item) for item in request.get("images") or []]
    if any(p is None for p in payloads):
        raise ValueError("an images entry the engine rejects")
    return [Image.open(io.BytesIO(p)) for p in payloads]


def py_positions(encoded, spatial_merge_size: int = 2) -> list[list[int]]:
    """Qwen3_5Model.get_rope_index on the encoded record, with the model class's own code."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

    stub = types.SimpleNamespace(config=types.SimpleNamespace(vision_config=types.SimpleNamespace(spatial_merge_size=spatial_merge_size)))
    stub.get_vision_position_ids = lambda *a, **k: Qwen3_5Model.get_vision_position_ids(stub, *a, **k)
    ids = torch.tensor([list(encoded.input_ids)])
    types_ = torch.zeros_like(ids)
    media = encoded.media
    offset = media["token_offset"]
    types_[0, offset:offset + len(media["mm_token_type_ids"])] = torch.tensor(media["mm_token_type_ids"])
    pos, _ = Qwen3_5Model.get_rope_index(stub, ids, types_, image_grid_thw=media["image_grid_thw"])
    return pos[:, 0].tolist()


def py_encode(tok, request, processor=None):
    from joint_schema_model import encode_record, QUESTION_TYPES

    questions = request.get("questions")
    if not isinstance(request.get("model"), str) or "state" not in request:
        return "ERR"
    if not isinstance(questions, dict) or not questions:
        return "ERR"
    for question in questions.values():
        if question.get("type") not in QUESTION_TYPES:
            return "ERR"
        if question["type"] != "noul" and not question.get("criteria"):
            return "ERR"
        # documented divergence: score criteria must be a list in C
        if question["type"] == "score" and not isinstance(question["criteria"], list):
            return "ERR"
    for qid, question in questions.items():
        # documented divergence: an empty id with no instructions is an empty span (NaN in the reference)
        ins = question.get("instructions")
        if qid == "" and (ins is None or ins == ""):
            return "ERR"
    # documented divergences: videos, non-list images, processor arguments other than the pixel bounds
    if request.get("videos"):
        return "ERR"
    if request.get("images") and not isinstance(request["images"], list):
        return "ERR"
    media = request.get("media_kwargs") or {}
    media_keys = set(media.keys())
    if media_keys - {"min_pixels", "max_pixels"} or len(media_keys) == 1:   # a lone bound is ignored by the processor; rejected here
        return "ERR"
    # documented divergence: the bounds must be positive integers within int32 (range-checked in C)
    if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 or v > 2**31 - 1 for v in media.values()):
        return "ERR"
    try:
        images = pil_images(request)
        # documented divergence: an image whose resized grid exceeds the context is refused before any
        # resize; the reference would build the patches and then fail on length
        if images:
            from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
            for im in images:
                h, w = smart_resize(im.height, im.width, 32, media.get("min_pixels", 65536), media.get("max_pixels", 16777216))
                if (h // 32) * (w // 32) > 16384:
                    return "ERR"
        rec = dict(request, images=images)
        e = encode_record(tok, rec, max_length=16384, processor=processor)
        if e.media is not None:
            # the reference fails later, scattering the image features over a mismatched token count
            if list(e.input_ids).count(processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")) != int(e.media["pixel_values"].shape[0]) // 4:
                return "ERR"
    except Exception:
        return "ERR"
    out = {
        "input_ids": list(e.input_ids),
        "questions": [
            {"id": q.question_id, "type": q.question_type, "span": list(q.question_span),
             "option_spans": [list(s) for s in q.option_spans], "option_ids": list(q.option_ids)}
            for q in e.questions
        ],
    }
    if e.media is not None:
        grids = e.media["image_grid_thw"].tolist()
        starts, t = [], e.media["token_offset"]
        for g in grids:
            t += 1   # <|vision_start|>
            starts.append([t, g[1], g[2]])
            t += g[1] * g[2] // 4 + 1
        out["images"] = starts
        out["position_ids"] = py_positions(e)
    return out


def f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", x))[0]


def main() -> None:
    gguf_path, tok_dir, code_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    sys.path.insert(0, code_dir)
    from joint_schema_model import encode_record, systemone_answer
    from transformers import AutoProcessor, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tok_dir)
    processor = AutoProcessor.from_pretrained(tok_dir)
    rng = random.Random(11)
    requests = corpus.build() + [rand_request(rng) for _ in range(3000)] + image_requests(rng)
    # states far beyond max_length: the C encoder stops tokenizing early (review #2, M3); the
    # kept prefix must equal Python's tokenize-everything-then-truncate result
    for n in (20000, 60000):
        words = ["alpha", "é", "e\u0301", "顧客", "  ", "\n", "1234", "!!", "<|im_end|>", "don't"]
        big = "".join(rng.choice(words) + rng.choice([" ", "", "\t"]) for _ in range(n))
        requests.append({"model": "clef", "state": big, "questions": {"q": {"type": "noul"}}})
        requests.append({"model": "clef", "state": {"log": big}, "questions": {"q": {"type": "score", "criteria": ["a", "b"]}}})
    # schema larger than max_length must be rejected the same way
    requests.append({"model": "clef", "state": "s", "questions": {
        f"q{i}": {"type": "choice", "criteria": {f"o{j}": "x " * 50 for j in range(40)}} for i in range(12)}})

    lines = [json.dumps(r, ensure_ascii=False) for r in requests]
    out = subprocess.run([ROOT / "clef-tool", "encode", gguf_path], input="\n".join(lines) + "\n",
                         capture_output=True, text=True, check=True).stdout.split("\n")[: len(lines)]
    bad = 0
    n_ok = 0
    n_img = 0
    for i, (req, got) in enumerate(zip(requests, out)):
        want = py_encode(tok, req, processor)
        n_img += want != "ERR" and "images" in want
        g = "ERR" if got.startswith("ERR") else json.loads(got)
        if g != want:
            bad += 1
            if bad <= 8:
                print(f"  encode mismatch #{i}: python={str(want)[:200]}\n    c={got[:200]}")
        elif want != "ERR":
            n_ok += 1
    print(f"encode: {len(requests) - bad}/{len(requests)} match ({n_ok} encoded, {n_img} with images and 3D positions, rest rejected by both)")

    # Responses over float32 probabilities, as model(...).float().softmax(-1).tolist() yields.
    resp_lines, wants = [], []
    for req in requests:
        enc = py_encode(tok, req, processor)
        if enc == "ERR":
            continue
        probs = []
        for q in enc["questions"]:
            raw = [rng.random() ** rng.choice([1, 4, 12]) for _ in q["option_ids"]]
            if rng.random() < 0.2 and len(raw) > 1:
                raw[1] = raw[0]  # ties: first maximum must win
            s = sum(raw) or 1.0
            probs.append([f32(x / s) for x in raw])
        answers = {
            q["id"]: systemone_answer(req["questions"][q["id"]], dict(zip(q["option_ids"], p)))
            for q, p in zip(enc["questions"], probs)
        }
        wants.append(json.dumps({"model": req["model"], "answers": answers,
                                 "usage": {"input_tokens": len(enc["input_ids"]), "output_tokens": 0}},
                                ensure_ascii=False, separators=(",", ":")))
        resp_lines.append(json.dumps({"request": req, "probs": probs}, ensure_ascii=False))
    got = subprocess.run([ROOT / "clef-tool", "respond", gguf_path], input="\n".join(resp_lines) + "\n",
                         capture_output=True, text=True, check=True).stdout.split("\n")[: len(resp_lines)]
    rbad = 0
    for w, g in zip(wants, got):
        if w != g:
            rbad += 1
            if rbad <= 5:
                print(f"  respond mismatch\n    python={w[:300]}\n    c     ={g[:300]}")
    print(f"respond: {len(wants) - rbad}/{len(wants)} match")
    mem_bad = image_budget_memory(gguf_path)
    if bad or rbad or mem_bad:
        sys.exit("FAIL")
    print("PASS")


def image_budget_memory(gguf_path: str) -> int:
    """Review #3: an image that cannot fit with the prompt and the images before it is refused
    before its resize and patches are allocated. Before the fix, a 4096x4096 image (exactly the
    16,384-token context) reached 542 MB resident and two 4096x2048 images 492 MB; after, 139 MB
    and 291 MB (the first of the two fits alone and is preprocessed). The schema counts too: a
    14,336-token image fits the prompt alone, but not with a 2,500-token question, and was
    preprocessed before the final length check refused it (review #3, Codex on 600ddfe); so does a
    state when truncation is refused, as the server refuses it by default (Codex on a99a8a9). Peak
    RSS of clef-tool, which applies no per-image token limit, as the CLI does by default."""
    def png(w: int, h: int) -> str:
        y, x = np.indices((h, w))
        arr = np.stack([x * 255 // (w - 1), y * 255 // (h - 1), (x + y) % 256], -1).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, "PNG")
        return base64.b64encode(buf.getvalue()).decode()

    q = {"q": {"type": "noul", "instructions": "Is it blue?"}}
    long_q = {"q": {"type": "noul", "instructions": "Is it blue? " + "alpha " * 2500}}
    half = png(4096, 2048)
    near = png(4096, 3584)
    cases = [("one 16,384-token image", [png(4096, 4096)], q, "x", "encode", 300),
             ("two 8,192-token images", [half, half], q, "x", "encode", 400),
             ("a 14,336-token image with a 2,500-token schema", [near], long_q, "x", "encode", 300),
             ("a 14,336-token image with a 2,500-token state, truncation refused", [near], q, "alpha " * 2500, "encode-notrunc", 300)]
    failures = 0
    for name, images, questions, state, mode, limit_mb in cases:
        line = json.dumps({"model": "clef-flash", "state": state, "images": images, "questions": questions}) + "\n"
        r = subprocess.run(["/usr/bin/time", "-l", str(ROOT / "clef-tool"), mode, gguf_path], input=line,
                           capture_output=True, text=True)
        rss = next((int(l.split()[0]) for l in r.stderr.splitlines() if "maximum resident set size" in l), None)
        refused = r.stdout.startswith("ERR") and "cannot fit" in r.stdout
        mb = rss / 1e6 if rss is not None else float("inf")
        ok = refused and mb < limit_mb
        failures += not ok
        print(f"image budget: {name} {'refused' if refused else 'NOT refused: ' + r.stdout[:120]} at {mb:.0f} MB peak "
              f"(limit {limit_mb} MB){'' if ok else '  FAIL'}")
    return failures


if __name__ == "__main__":
    main()
