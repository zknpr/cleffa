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
import threading
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

    def post(self, req: dict, key: str | None = None, raw: bytes | None = None) -> tuple[int, bytes]:
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=600)
        headers = {"Content-Type": "application/json"}
        if key is not None:
            headers["X-Clef-Prefix-Cache"] = key
        body = raw if raw is not None else json.dumps(req).encode()
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
        # review #3: a data URL's media type is checked like content_type
        st, body = plain.post(dict(small, images=["data:%s;base64," % other + b64]))
        expect(st, body, 400, "data URL media type says", "mislabeled data URL")
        st, body = plain.post(dict(small, images=["data:image/webp;base64," + b64]))
        expect(st, body, 400, "WebP is not supported", "webp data URL")
        tiny = png(40, 40)
        # review #3: a 40x40 PNG forced to 8192x8192 by media_kwargs must be refused before the
        # resize and the 1.6 GB patch allocation, not after (measured 1.85 GB resident before the fix)
        t0 = time.monotonic()
        st, body = plain.post(dict(small, images=[tiny], media_kwargs={"min_pixels": 67108864, "max_pixels": 67108864}))
        huge_s = time.monotonic() - t0
        expect(st, body, 400, "above the limit of 1024 per image", "tiny image with huge bounds")
        if huge_s > 2.0:
            fail(f"tiny image with huge bounds took {huge_s:.1f} s: the token check ran after the resize")
        st, body = plain.post(dict(small, images=[tiny], media_kwargs={"min_pixels": 99999999999999999999, "max_pixels": 99999999999999999999}))
        expect(st, body, 400, "must be a positive integer", "out-of-range media_kwargs")
        # review #3: a source image over --max-image-pixels (default 16,777,216) is refused at its
        # header. A compressible 8192x8192 PNG is ~0.4 MB as base64 but decoded first to several
        # times its 192 MiB of RGB (4.1 GB above baseline for eight such requests before the fix).
        buf = io.BytesIO()
        Image.new("RGB", (8192, 8192)).save(buf, format="PNG")
        huge_src = base64.b64encode(buf.getvalue()).decode()
        t0 = time.monotonic()
        st, body = plain.post(dict(small, images=[huge_src]))
        src_s = time.monotonic() - t0
        expect(st, body, 400, "8192x8192 is 67108864 pixels, above the limit of 16777216 per image", "huge source image")
        if src_s > 1.0:
            fail(f"huge source image took {src_s:.2f} s: it was decoded before the pixel limit applied")
        at_limit = png(4096, 4096, 2, smooth=True)   # exactly 16,777,216 pixels: allowed, resized by media_kwargs
        st, body = plain.post(dict(small, images=[at_limit], media_kwargs={"min_pixels": 65536, "max_pixels": 1048576}))
        if st != 200:
            fail(f"source image at the pixel limit: {st} {body[:200]!r}")
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
        # A repeated key collapses to one, as json.loads makes it before the processor ignores the lone
        # bound. Codex read the member count as counting copies (refuted on b1e7dd2); this keeps the
        # parser's merge and the count check tied together. Raw bytes: json.dumps cannot repeat a key.
        lone = json.dumps(dict(small, images=[tiny], media_kwargs={"max_pixels": 65536}))
        for repeated in ('{"max_pixels": 65536, "max_pixels": 102400}', '{"max_pixels": 65536, "max\\u005fpixels": 102400}'):
            raw = lone.replace('{"max_pixels": 65536}', repeated)
            if raw == lone:
                fail("repeated-bound fixture did not rewrite media_kwargs")
            st, body = plain.post({}, raw=raw.encode())
            expect(st, body, 400, "both min_pixels and max_pixels", f"repeated bound {repeated}")
        st, body = plain.post(dict(small, images=["not base64!"]))
        expect(st, body, 400, "images[0]", "bad base64")
        st, body = plain.post(dict(small, images=[base64.b64encode(base64.b64decode(tiny)[:40]).decode()]))
        expect(st, body, 400, "images[0]", "truncated PNG")
        st, body = plain.post(dict(small, videos=[[0]]))
        expect(st, body, 400, "videos are not supported", "video")
        st, body = plain.post(dict(small, state="<|image_pad|> in the state"))
        if st != 200:
            fail(f"strict mode: placeholder text was not served: {st} {body[:200]!r}")
        print("limits and errors: five images, 2,304-token image, lone or repeated media_kwargs bound, bad base64, truncated PNG, video, WebP and a mislabeled "
              "object rejected; mislabeled and WebP data URLs rejected; a tiny image with huge bounds refused in %.2f s before any resize; "
              "out-of-range media_kwargs rejected; an 8192x8192 source refused at its header in %.2f s, a 4096x4096 one served; "
              "data URL, object and bare forms equal; placeholder text served" % (huge_s, src_s))
    finally:
        plain.stop()
    log = plain.log_path.read_text()
    if "warm-up image pass" not in log:
        fail("the server did not warm the vision path at startup (review #3)")
    print("startup: " + next(l for l in log.splitlines() if "warm-up image pass" in l).strip())

    # review #3: image requests take one of --max-image-requests slots from before decoding until
    # their patches are freed. With a single slot, a burst mixing answered requests and every
    # rejection path must all finish (a leaked slot would block the rest) with the same bytes.
    one = Server(["--max-image-requests", "1", "--no-warmup"], tmp, "one-slot")
    try:
        mix = [(r, want[r["id"]]) for r in corpus[:6]]
        mix += [(dict(small, images=["not base64!"]), None), (dict(small, images=[tiny] * 5), None),
                (dict(small, images=[huge_src]), None), (dict(small, images=[big]), None)]
        mix += [(r, want[r["id"]]) for r in corpus[6:9]]
        results: list = [None] * len(mix)

        def run(k: int) -> None:
            results[k] = one.post(mix[k][0])

        threads = [threading.Thread(target=run, args=(k,), daemon=True) for k in range(len(mix))]
        for t in threads:
            t.start()
        deadline = time.monotonic() + 180
        for t in threads:
            t.join(max(0.0, deadline - time.monotonic()))
        if any(t.is_alive() for t in threads):
            fail("one-slot server: requests still blocked after 180 s (an image slot leaked)")
        for (req, expected), (st, body) in zip(mix, results):
            if expected is None:
                if st != 400:
                    fail(f"one-slot server: rejection answered {st} {body[:120]!r}")
            elif st != 200 or body != expected:
                fail(f"one-slot server: {req.get('id')} {st} differs from the unbounded server")
        print(f"one image slot: {len(mix)} concurrent requests (9 answered, 4 rejected) finished, bytes unchanged")
    finally:
        one.stop()

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
