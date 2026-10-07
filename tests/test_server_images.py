"""clef-server with images, against the CLI and the server's own limits.

Usage: test_server_images.py MODEL.gguf VISION_REQUESTS.jsonl
  - every vision corpus request over HTTP (default server: strict, 4 images, 1,024 tokens per image)
    == the CLI's response bytes for the same request (--strict --no-truncate)
  - the hosted API's two forms (a data URL, {"content_type", "base64"}) and the bare base64 extension
    answer identically; an unsupported or mislabeled content_type is a 400
  - limits: a fifth image is a 400 naming the limit; an image past the token limit is a 400 that
    names media_kwargs.max_pixels, and the same image with those media_kwargs is served; a lone
    media_kwargs bound, bad base64, a truncated PNG and a video are 400s; the limits can be raised
  - strict mode: a literal <|image_pad|> in the state is text and the request is served; with
    --no-strict the engine refuses it as the reference would fail (placeholder count)
  - --template-cache and --prefix-cache-mb: image requests answer byte-identically to the plain
    server, with and without a cache key, and repeated keyed images reuse validated features and prefix state
"""

from __future__ import annotations

import base64
import http.client
import io
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
model, corpus_path = sys.argv[1], sys.argv[2]
corpus = [json.loads(line) for line in open(corpus_path)]
REUSED = re.compile(r"prefix cache: reused (\d+) of (\d+) tokens")


def fail(msg: str) -> None:
    sys.exit("FAIL: " + msg)


class Server:
    def __init__(self, flags: list[str], tmp: Path, name: str):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]
        self.log_path = tmp / (name + ".log")
        self.log = self.log_path.open("w")
        env = {k: v for k, v in os.environ.items() if not k.startswith("CLEF_")}
        self.proc = subprocess.Popen([str(ROOT / "clef-server"), "-m", model, "--port", str(self.port), *flags],
                                     cwd=ROOT, stdout=self.log, stderr=self.log, env=env)
        deadline = time.monotonic() + 240
        while True:
            try:
                c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
                c.request("GET", "/health")
                if c.getresponse().status == 200:
                    break
            except OSError:
                pass
            if self.proc.poll() is not None or time.monotonic() > deadline:
                self.stop()
                fail(f"server did not start ({name})")
            time.sleep(0.5)

    def post(self, req: dict, key: str | None = None) -> tuple[int, bytes]:
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=600)
        headers = {"Content-Type": "application/json"}
        if key is not None:
            headers["X-Clef-Prefix-Cache"] = key
        body = json.dumps(req).encode()
        try:
            c.request("POST", "/v1/systemone", body, headers)
            r = c.getresponse()
            data = r.read()
        except OSError as e:
            fail(f"connection error posting {len(body)} bytes (images {len(req.get('images') or [])}): {e}")
        finally:
            c.close()
        return r.status, data

    def reused(self) -> list[int]:
        self.log.flush()
        return [int(m.group(1)) for m in REUSED.finditer(self.log_path.read_text())]

    def stop(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.log.close()


def png(h: int, w: int, seed: int = 0, smooth: bool = False) -> str:
    """Random pixels, or a gradient that compresses well (a large image must fit --max-body)."""
    rng = np.random.default_rng(seed)
    if smooth:
        y, x = np.indices((h, w))
        arr = np.stack([x * 255 // max(w - 1, 1), y * 255 // max(h - 1, 1), (x + y) % 256], -1).astype(np.uint8)
    else:
        arr = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def expect(status: int, body: bytes, want: int, text: str, what: str) -> None:
    if status != want or text not in body.decode(errors="replace"):
        fail(f"{what}: {status} {body[:200]!r}, expected {want} with {text!r}")


small = next(r for r in corpus if len(r["images"]) == 1 and len(json.dumps(r)) < 3000)
cli = subprocess.run([ROOT / "clef", "-m", model, "--strict", "--no-truncate"],
                     input="\n".join(json.dumps(r, ensure_ascii=False) for r in corpus) + "\n",
                     capture_output=True, text=True).stdout.splitlines()
if len(cli) != len(corpus):
    fail("CLI did not answer every request")

with tempfile.TemporaryDirectory() as t:
    tmp = Path(t)
    plain = Server([], tmp, "plain")
    try:
        want = {}
        for r, c in zip(corpus, cli):
            st, body = plain.post(r)
            if st != 200 or body.decode() != c:
                fail(f"{r['id']}: HTTP {st} {body[:160]!r} differs from the CLI {c[:160]!r}")
            want[r["id"]] = body
        print(f"images over HTTP: {len(corpus)}/{len(corpus)} byte-equal to the CLI")

        b64 = small["images"][0].split(",", 1)[-1]
        kind = "image/jpeg" if base64.b64decode(b64)[:2] == b"\xff\xd8" else "image/png"
        bare = plain.post(dict(small, images=[b64]))
        for form in ("data:%s;base64," % kind + b64, "DATA:%s;base64," % kind + b64, {"content_type": kind, "base64": b64}):
            if plain.post(dict(small, images=[form])) != bare:
                fail(f"image form {str(form)[:40]} answered differently from bare base64")
        st, body = plain.post(dict(small, images=[{"content_type": "image/webp", "base64": b64}]))
        expect(st, body, 400, "WebP is not supported", "webp content_type")
        other = "image/png" if kind == "image/jpeg" else "image/jpeg"
        st, body = plain.post(dict(small, images=[{"content_type": other, "base64": b64}]))
        expect(st, body, 400, "content_type says", "mislabeled image")
        tiny = png(40, 40)
        st, body = plain.post(dict(small, images=[tiny] * 5))
        expect(st, body, 400, "too many images", "five images")
        st, body = plain.post(dict(small, images=[tiny] * 4))
        if st != 200:
            fail(f"four images: {st} {body[:200]!r}")
        big = png(1536, 1536, 1, smooth=True)   # 2,304 tokens, ~30 KB encoded
        st, body = plain.post(dict(small, images=[big]))
        expect(st, body, 400, "media_kwargs.max_pixels <= 1048576", "oversized image")
        st, body = plain.post(dict(small, images=[big], media_kwargs={"min_pixels": 65536, "max_pixels": 1048576}))
        if st != 200:
            fail(f"oversized image with media_kwargs: {st} {body[:200]!r}")
        st, body = plain.post(dict(small, images=[tiny], media_kwargs={"max_pixels": 65536}))
        expect(st, body, 400, "both min_pixels and max_pixels", "lone bound")
        st, body = plain.post(dict(small, images=["not base64!"]))
        expect(st, body, 400, "images[0]", "bad base64")
        st, body = plain.post(dict(small, images=[base64.b64encode(base64.b64decode(tiny)[:40]).decode()]))
        expect(st, body, 400, "images[0]", "truncated PNG")
        st, body = plain.post(dict(small, videos=[[0]]))
        expect(st, body, 400, "videos are not supported", "video")
        st, body = plain.post(dict(small, state="<|image_pad|> in the state"))
        if st != 200:
            fail(f"strict mode: placeholder text was not served: {st} {body[:200]!r}")
        print("limits and errors: five images, 2,304-token image, lone media_kwargs bound, bad base64, truncated PNG, video, WebP and a mislabeled "
              "object rejected; data URL, object and bare forms equal; placeholder text served")
    finally:
        plain.stop()

    raised = Server(["--max-images", "5", "--max-image-tokens", "4096", "--no-strict"], tmp, "raised")
    try:
        st, body = raised.post(dict(small, images=[tiny] * 5))
        if st != 200:
            fail(f"--max-images 5: {st} {body[:200]!r}")
        st, body = raised.post(dict(small, images=[big]))
        if st != 200:
            fail(f"--max-image-tokens 4096: {st} {body[:200]!r}")
        st, body = raised.post(dict(small, state="<|image_pad|> in the state"))
        expect(st, body, 400, "image placeholder tokens in request content", "no-strict placeholder")
        print("raised limits served; --no-strict refuses placeholder text")
    finally:
        raised.stop()

    cached = Server(["--template-cache", "--prefix-cache-mb", "4096"], tmp, "cached")
    try:
        for r in corpus:
            st, body = cached.post(r)
            if st != 200 or body != want[r["id"]]:
                fail(f"{r['id']} with --template-cache: {st} differs from the plain server")
            st, body = cached.post(r, "tenant-" + r["id"])
            if st != 200 or body != want[r["id"]]:
                fail(f"{r['id']} with a cache key: {st} differs from the plain server")
        # Reserve one key for a large image and compare first fill, hit, a different
        # key, and changed pixels with the same placeholder count. No key can use another's entry.
        r = next(r for r in corpus if r['id'] == 'v009')
        for key, hit in [('repeat', False), ('repeat', True), ('separate', False), ('separate', True)]:
            before = len(cached.reused())
            st, body = cached.post(r, key)
            if st != 200 or body != want[r['id']]:
                fail(f"{key}: keyed image response differs")
            reused = cached.reused()[before:]
            if len(reused) != 1 or bool(reused[0]) != hit:
                fail(f"{key}: unexpected prefix reuse {reused}, expected hit={hit}")
        replacement = dict(r, images=[png(1024, 1024, smooth=True)])
        st, expected = cached.post(replacement)
        if st != 200:
            fail("uncached replacement image failed")
        before = len(cached.reused())
        for _ in range(2):
            if cached.post(replacement, 'repeat') != (200, expected):
                fail("same-shape replacement image reused stale content")
        reused = cached.reused()[before:]
        if len(reused) != 2 or reused[0] or not reused[1]:
            fail(f"image replacement did not invalidate then refill: {reused}")
        print(f"template/keyed caches: {len(corpus)} corpus responses exact; repeated images, key isolation and replacement PASS")
    finally:
        cached.stop()
    tiny_budget = Server(["--prefix-cache-mb", "1"], tmp, "tiny-budget")
    try:
        r = next(r for r in corpus if r['id'] == 'v009')
        for _ in range(2):
            if tiny_budget.post(r, 'over-budget') != (200, want[r['id']]):
                fail("budget bypass changed the image response")
        if tiny_budget.reused() or 'would exceed the budget' not in tiny_budget.log_path.read_text():
            fail("oversized image entry was not rejected before cache allocation")
        print("image cache budget: oversized entry served uncached exactly")
    finally:
        tiny_budget.stop()
print("PASS")
