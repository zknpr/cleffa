"""Fixed SystemOne request set with images: the vision parity and latency corpus.

Deterministic synthetic images (no binary fixtures in git), chosen to cover what the native
vision path must reproduce: PNG and JPEG decoding, RGB/gray/alpha sources, upscaling below the
pixel floor, downscaling, odd grids, several images per request, media_kwargs bounds, images
with JSON and long text states, and one 1,024-token image. Requests carry images as base64
strings, some as data URLs (the hosted API's object form is covered by tests/test_record.py and
tests/test_server_images.py, which decode to the same bytes). Changing this file invalidates
every golden/*vision* directory.
"""

from __future__ import annotations

import base64
import io
import json

import numpy as np
from PIL import Image, ImageDraw


def _png(arr: np.ndarray, mode: str | None = None, data_url: bool = False) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr, mode).save(buf, format="PNG")
    s = base64.b64encode(buf.getvalue()).decode()
    return "data:image/png;base64," + s if data_url else s


def _jpeg(arr: np.ndarray, quality: int = 85, progressive: bool = False, subsampling: int = 2) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=quality, progressive=progressive, subsampling=subsampling)
    return base64.b64encode(buf.getvalue()).decode()


def _gradient(h: int, w: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, x = np.indices((h, w), dtype=np.float64)
    r = (x / max(w - 1, 1)) * 255
    g = (y / max(h - 1, 1)) * 255
    b = ((np.sin(x / 9.0 + rng.random() * 6) + np.cos(y / 7.0)) * 0.25 + 0.5) * 255
    return np.stack([r, g, b], -1).round().clip(0, 255).astype(np.uint8)


def _scene(h: int, w: int, seed: int) -> np.ndarray:
    """Sky, ground, a sun and a few boxes: enough structure for photo-like questions."""
    rng = np.random.default_rng(seed)
    img = Image.fromarray(_gradient(h, w, seed))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, w, h * 0.55], fill=(120, 170, 230))
    d.rectangle([0, h * 0.55, w, h], fill=(70, 140, 60))
    d.ellipse([w * 0.7, h * 0.08, w * 0.85, h * 0.08 + w * 0.15], fill=(250, 220, 80))
    for _ in range(5):
        x0, y0 = rng.integers(0, w * 0.8), rng.integers(h * 0.5, h * 0.9)
        d.rectangle([x0, y0, x0 + rng.integers(4, max(5, w * 0.2)), y0 + rng.integers(4, max(5, h * 0.3))],
                    fill=tuple(int(v) for v in rng.integers(0, 256, 3)))
    return np.asarray(img)


def _receipt(h: int, w: int, seed: int) -> np.ndarray:
    """A pale page with dark text-like bars and a total line."""
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (w, h), (248, 246, 240))
    d = ImageDraw.Draw(img)
    y = 12
    while y < h - 30:
        n = rng.integers(2, 6)
        x = 10
        for _ in range(n):
            width = int(rng.integers(8, 40))
            d.rectangle([x, y, x + width, y + 6], fill=(30, 30, 30))
            x += width + 6
        y += 14
    d.line([8, h - 24, w - 8, h - 24], fill=(0, 0, 0), width=2)
    d.rectangle([w - 70, h - 18, w - 12, h - 10], fill=(200, 20, 20))
    return np.asarray(img)


def _noise(h: int, w: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def _log(lines: int, seed: int) -> str:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(lines):
        out.append(f"2026-10-01T08:{i // 60:02d}:{i % 60:02d}Z {['INFO', 'WARN', 'ERROR'][rng.integers(0, 3)]} "
                   f"camera-{rng.integers(1, 5)} frame={i} motion={rng.random():.2f} lux={rng.integers(0, 2000)}")
    return "\n".join(out)


WEBCAM_QUESTIONS = {
    "wearing_glasses": {"type": "noul", "instructions": "Wearing glasses?"},
    "mood": {"type": "choice", "criteria": {"happy": None, "neutral": None, "tired": None}},
    "energy": {"type": "score", "criteria": ["low", "medium", "high"]},
}

SCENE_QUESTIONS = {
    "outdoors": {"type": "noul", "instructions": "Is the scene outdoors?"},
    "subject": {"type": "choice", "instructions": "Main subject of the image?",
                "criteria": {"landscape": "Sky and ground", "person": None, "document": "A page of text", "abstract": "Noise or pattern"}},
    "brightness": {"type": "score", "criteria": ["dark", "dim", "normal", "bright"]},
}


def build() -> list[dict]:
    requests: list[dict] = []
    add = requests.append
    # README-style receipt (model card example)
    add({"model": "clef-flash", "state": {"task": "Review the attached receipt."}, "images": [_png(_receipt(320, 240, 1))],
         "questions": {"legible": {"type": "noul", "instructions": "Is the receipt total legible?"},
                       "document": {"type": "choice", "criteria": {"receipt": "A receipt", "photo": "A photograph", "other": None}}}})
    # webcam frame at clef-webcam's 336-pixel cap, JPEG
    add({"model": "clef-flash", "state": "A live webcam frame from a laptop.", "images": [_jpeg(_scene(252, 336, 2))],
         "questions": WEBCAM_QUESTIONS})
    # tiny image: upscaled to the 65,536-pixel floor
    add({"model": "clef-flash", "state": "A photo.", "images": [_png(_gradient(16, 16, 3), data_url=True)], "questions": SCENE_QUESTIONS})
    # noise, progressive JPEG with 4:4:4 chroma
    add({"model": "clef-flash", "state": "A photo.", "images": [_jpeg(_noise(100, 150, 4), quality=92, progressive=True, subsampling=0)],
         "questions": SCENE_QUESTIONS})
    # two images of different sizes
    add({"model": "clef-flash", "state": "Two frames from the same camera, one minute apart.",
         "images": [_png(_scene(120, 160, 5)), _jpeg(_scene(200, 90, 6))],
         "questions": {"changed": {"type": "noul", "instructions": "Did the scene change between the frames?"},
                       "count": {"type": "score", "criteria": ["0", "1", "2", "3 or more"], "instructions": "How many boxes are visible?"}}})
    # wide aspect ratio
    add({"model": "clef-flash", "state": "A panorama.", "images": [_png(_gradient(96, 640, 7))],
         "questions": {"quality": {"type": "score", "criteria": ["poor", "fair", "good", "excellent"]}}})
    # grayscale PNG and RGBA PNG with varying alpha
    add({"model": "clef-flash", "state": "A scanned page.", "images": [_png(np.asarray(Image.fromarray(_receipt(200, 300, 8)).convert("L")), "L")],
         "questions": {"legible": {"type": "noul"}, "language": {"type": "choice", "criteria": {"en": None, "de": None, "unknown": None}}}})
    rgba = np.concatenate([_scene(128, 128, 9), np.linspace(0, 255, 128 * 128).reshape(128, 128, 1).astype(np.uint8)], -1)
    add({"model": "clef-flash", "state": "An icon with transparency.", "images": [_png(rgba, "RGBA")],
         "questions": {"transparent": {"type": "noul", "instructions": "Does the image have transparent areas?"}}})
    # JSON state and many questions
    add({"model": "clef-flash",
         "state": {"ticket": "Customer sent a photo of the damaged package.", "order": {"id": "A-1042", "items": 2, "insured": True}},
         "images": [_png(_scene(240, 320, 10))],
         "questions": {"damaged": {"type": "noul", "instructions": "Does the photo show damage?"},
                       "refund": {"type": "choice", "criteria": {"full": "Refund everything", "partial": "Refund part", "none": None}},
                       "severity": {"type": "score", "criteria": ["none", "minor", "major", "total loss"]},
                       "insured": {"type": "noul", "instructions": "Is the order insured?"},
                       "needs_photo": {"type": "noul", "instructions": "Is another photo needed?"},
                       "team": {"type": "choice", "criteria": {"claims": None, "logistics": None, "support": None}}}})
    # 1,024-token image (1024x1024 -> 64x64 patches -> 32x32 merged)
    add({"model": "clef-flash", "state": "A high-resolution photo.", "images": [_jpeg(_scene(1024, 1024, 11), quality=80)],
         "questions": SCENE_QUESTIONS})
    # image with a long text state (~1k tokens)
    add({"model": "clef-flash", "state": _log(40, 12), "images": [_png(_scene(192, 256, 13))],
         "questions": {"incident": {"type": "noul", "instructions": "Do the logs and the frame indicate an incident?"},
                       "camera": {"type": "choice", "criteria": {f"camera-{i}": None for i in range(1, 5)}}}})
    # media_kwargs: a 500x400 photo capped to 64 tokens, and one with a raised floor (the processor
    # applies the bounds only when both are given)
    add({"model": "clef-flash", "state": "A photo.", "images": [_png(_scene(400, 500, 14))],
         "media_kwargs": {"min_pixels": 65536, "max_pixels": 65536}, "questions": SCENE_QUESTIONS})
    add({"model": "clef-flash", "state": "A photo.", "images": [_png(_scene(64, 48, 15))],
         "media_kwargs": {"min_pixels": 200000, "max_pixels": 16777216}, "questions": SCENE_QUESTIONS})
    # three small images and an empty state
    add({"model": "clef-flash", "state": "", "images": [_png(_noise(40, 40, 16)), _png(_gradient(33, 47, 17)), _jpeg(_scene(70, 50, 18))],
         "questions": {"same": {"type": "noul", "instructions": "Are all three images the same scene?"}}})
    # odd grid sizes: 7x5 merged (224x160) and 3x9 merged (96x288)
    add({"model": "clef-flash", "state": "A photo.", "images": [_png(_scene(224, 160, 19))], "questions": SCENE_QUESTIONS})
    add({"model": "clef-flash", "state": "A photo.", "images": [_png(_gradient(96, 288, 20))], "questions": SCENE_QUESTIONS})
    for i, request in enumerate(requests):
        request["id"] = f"v{i:03d}"
    return requests


if __name__ == "__main__":
    for request in build():
        print(json.dumps(request, ensure_ascii=False))
