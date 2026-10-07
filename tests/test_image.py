"""Byte parity of the C image pipeline (clef_image.c, via clef-tool image) with the reference:
PIL decoding (convert("RGB")), Qwen2VLImageProcessor.smart_resize, torchvision's uint8
antialiased bicubic resize, the f32 normalization and patch layout of the processor's
pixel_values, and the position-table interpolation indices and weights.

Usage: .venv/bin/python -B tests/test_image.py [--seed N] [--cases N]
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import struct
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.v2.functional as tvF
from PIL import Image
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor, smart_resize
from transformers.vision_utils import get_vision_bilinear_indices_and_weights

ROOT = Path(__file__).resolve().parent.parent
MIN_PIXELS, MAX_PIXELS = 65536, 16777216

# A 64x48 baseline JPEG from libjpeg-turbo's `cjpeg -quality 90 -sample 1x1,2x2,1x1`: luma sampled
# below chroma, which libjpeg and PIL decode. The vendored decoder sizes the luma plane from its
# own factors and read it at image resolution (heap over-read, found by review on 2026-10-07); it
# now refuses the layout.
LUMA_UNDER_CHROMA_JPEG = (
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQEBQoHBwYIDAoMDAsKCwsNDhIQDQ4RDgsLEBYQERMU"
    "FRUVDA8XGBYUGBIUFRT/2wBDAQMEBAUEBQkFBQkUDQsNFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQU"
    "FBQUFBQUFBT/wAARCAAwAEADAREAAiIBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUF"
    "BAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVW"
    "V1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi"
    "4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAEC"
    "AxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVm"
    "Z2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq"
    "8vP09fb3+Pn6/9oADAMBAAIRAxEAPwCHVLu0nvoTcWKwXoSK5juJ0YmdG2o4+QLjByAMgARhlUBstEK1LLMPX+qVKlGnNt0qqbTX"
    "LztU7qqpVKcH7SpUiuZKTnUvXly1ZVisJPH4Orm1GjO1XWpKcLTglGU5OOvI1KNNQnC8VFxpycpOalP0IcS4XD5nDK8U2o+1qK3O"
    "qcYSg5qPs/aVHGnUcOaMuacJ2mlJzVoH01TCJRxkcbBVFWbjCb5XCnGUVRjFSUpOKipqnKEZT0ScedzsuSjhJunUTk3B3i1bms23"
    "yyd07XX27+7aKlLlSAfZtTLappk9vbosHl7BcRNIGIAWOTaoZiMqvG4Y+ig/C1Vgcqw9TNcgrShGcqcfZ01NRhyVXCPu1IKatUXL"
    "aV1JJxjJ2SdYdUsXmFBYqjOP1f3FU5pKzSjG16lVqKTcHGDjUjpNqDlKU36eKpPC4vF4vBKToJwqpxsqk5KS5KXL8PNL3nS15akr"
    "SUPawvHwcPUqQo/WMLCly+7UlKahH45YiEvaSVRKddShKcfac7pxnLnmuWLl6GEwrnTj7a8puU+ZO6e/KtJNpaJyUuRPRwXxWJrK"
    "eUNbGJI2t1RWkhuZkEaAsgJcSF/3hEhYHI2hfvHapr6HA160HTlHnjDkqUqjnTUW041IRfNJxlP34y9pyyi3FUqaScWp8eMx+Kx+"
    "AgsnnVjOLpSfK/em7Jy+K0pKm6ekmqcrOnCcFShVadanFwjQhRt7SMnNKraDlGzv7ZVHOlyvmhGnUU/fc4zUqjcmqmDzCjiMV7FO"
    "MacZONSFTllCMY03KN480+enUcnyySpOFWrBNyfu+TNfUJKdCrztNO1nFw0va8VGEk7PmV7JrRrli5XBq1te6M+mSgXE0RMMVrsQ"
    "iFGmQBAu87zjHoQR1BBNceMy114KjC01XjCcqftU1GopxgvaWdOcUoR9+e0uf958EqkvOp5pmVapXpUqztTcpVeWUUpw5VBRtS5f"
    "cnKM4wprllKLo3UPZp0+yq6VHCYbG4XDReKpRjzUqnPBzdKcqjqKkpqzptShJqopxcKcJrmiuTnrRoUMXi6mNnKVOvH3XyuXtKic"
    "1alF8yf7lUuTSdNc1Llk5KnzdscJXo8uIw6mpK8veTV7J31STSsmny2k72lZXZjwTTCxt9N8u8tr64iInkYB5Ddh2IhMqO4QgIxI"
    "boduCNpavpMyxWFynF+0nKaq1Ju/vym5ynTUuVSm/ZJNLXmjyxado1Kahy/N4bJ8BTq05+xU6Tf7yMaLg3UqNwSpRioWlTsm48je"
    "G5rqUWk6nbXcFl9BQw/I6KcVTalK3OpUWuWouRxlKS1jZU4R9km41JyWM8Q54Cln2Axag6SlNvldN04ql7KMf3kpqcFUTnGU7S/5"
    "eRi+dIynh4yg1HWO3utOP2Yr3bc1k4rWEX7qlvFu8+rG9m8UWssVjb6diOaWKdWxak7cFlXcVwx8v5y33WAxnIOuV5pXUFlOfzp0"
    "7c1nye+ouDhBqMKSjOPsrRh+8jq7uMVClKHpwwVKsozrU40pwk3TvKc6kZKEZ1JTi2lNpydSSUXdQtGnLljKPHhMzxOBweEePqzn"
    "Ou3FOUqkZSnGrCEJxlySmk6LXNFRnUhTc5xg+XlIz2VDHZlRpVPaKdeMYqMJKm0rqip88IySceaVre5KNBRbvU9i2sv92WEnOLi7"
    "KUYxjttaHKm1yqMnLRq7+HmnK8T6bFqV6shkaJDKjNA0+bhUYIcDEakMA2V3AcDbjaMmsplhsRShUxWJcaqVObnKpKlNwScJ+9GC"
    "leS50+WUHKcsQ26MZwlHbNaMqlCpDD3VRqpCo46xcJ/vJzahLmldOFOLhThGEoSiopQdMyxmaYudZVaNH91SoyjzqhCNSPNDnlRj"
    "HnhOatUpQTjGMK03NyatdeTzwzOrOeGownS5pxcIXcFGzUV7OP72CjeCadvdapwVNcs36FKrN0qlX2cZPRO/wuKbdoJW93e7aUVa"
    "Tj7vLbUuYhaQ/YBbQJc2kaCKVrZEa1bJMUbShAPlRN652cMA24x7q2wWNxdR1swws5VZxhFtRcY1KdJL2ikm1GrWXPKSnGnCFKoq"
    "cvelZQZHAYytmFXB4ilKrKUGocsrvnqS51VqL2/PKzjU9m4WhOClO8YRckSxdWWMjhsJjOaqoLWNS6S5HBzi1KM1NxT2te05ubhz"
    "VYRnGOpVsPzTrSqxqJ0qftE41J3dOov3kIc1JVOW8oK1SfPSaglUahyYLSX1WjC0J25rLnTu1N6yajdOL0f2lKb92yIY7cXkMemQ"
    "28Vvql5E+wl1Dq6AArEEjwSnJ3naWGMcvz68c6xuEq1KlbERc5T5PeV5Rm4qNKco8nvRlCUXz1KlST5JTpVKtONNQ6cBmtLGzq4J"
    "1aVGpOm17zl7X2SleVSM/c9lL/l7s4wULyl7GPtX8zClSxNfD4hY1QUZc1GdNxjUtKUacHKi7VIt3jL3oTi4ud5NckZfQ1K1fLsX"
    "SeNp+1oyvKh7s41HVg+WMFOLkpRqJRUpq8KrqU7xi5ucuadWnGlUqz52kpWslFtxahFNxn7sdXpzOy5nKS5Ilm6nN9eWjQ+dfpIN"
    "r20yjKiMOcsGVdrgNHkfMT5rgnsPlsPmmHzRwweOnS9jVfJGnL2qcYxqJOUHSk4PljOrJQlU5m4tXlSlJ0/rsRXw08BKliFBYGbr"
    "Sbnac/axduVKE1HmkpShKCUpOcVT5op3p8+DxMcPB47BYn2klzJySVOTk+SMpqnJSUfedJQrcvs1aEKcFZylz5bmH1upSwHM4tXk"
    "pwi5QhV5pcsnNTi96cqcfcpygrU+ePLNUu+tVVNValWacoJtylraS5FJRUG1Hl5PdcpRabUpJtuTS7uZraCS0vbs3bQ2YW4ndUMj"
    "xfLKd+SxQNGn38DIkAG0nj52rRxFH27wrjTbftI0b8kIqUHJv/n5CfNJxUXOjfkjKpOD5D6CrKlh8zeKhOEpt4dKnaq3FtSi2pwa"
    "hJ3lyuq4yjLmqLlbnJ1sMf7OE6ePwVuepGlCV26V1VlLkm4OdW6i3JxXJzcyjLmqNSjDGnJxrzxMcJTqRcHKop1IVJwcIUp0k51V"
    "CLU/Z1IymrwcXOVOc1CV+SOEq/WOelQc5zu3G6to4pN7xlFa9IuTUeZO0iKDUZpbeRP3yi7Ys5aKMfLvLEQ7SQCCQdoH90KV+YH2"
    "pVMPKVCl7W9SOqi6cpqUaMpSjyRTXLFzqezk5ONSbinyqNc8zCUfr2Nq4R3d1zKM1U/iexv7Rya5YTipPlvHR1pSjzSpKJ1VcLg8"
    "bjI/2lBzoqom4SSU+WpJpKqp8rhJxowcnCUZOnSlConuuOWW4DAzw8K1ajRpulT5JTlD2sfZuVWMoN8slJz0cOeM4tc0VOmpKVUa"
    "PsZxoqfs563baV23dXu0rr4rWldJrmTdj//Z"
)


def progressive_se255(arr: np.ndarray) -> bytes:
    """A progressive JPEG whose first AC scan declares Se = 255: the coefficient index ran past the
    64-entry zigzag table (global over-read feeding a heap write) until the scan header was validated."""
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=90, progressive=True)
    data = bytearray(buf.getvalue())
    pos = 2
    while pos + 4 <= len(data):
        marker, seg = data[pos + 1], int.from_bytes(data[pos + 2:pos + 4], "big")
        if marker == 0xDA:
            ns = data[pos + 4]
            ss_off = pos + 5 + ns * 2
            if data[ss_off] != 0:   # the first AC scan
                data[ss_off + 1] = 0xFF
                return bytes(data)
            pos += 2 + seg
            while pos + 1 < len(data) and not (data[pos] == 0xFF and data[pos + 1] not in (0, *range(0xD0, 0xD8))):
                pos += 1
        else:
            pos += 2 + seg
    raise AssertionError("no AC scan found")


def processor(min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS):
    return Qwen2VLImageProcessor(min_pixels=min_pixels, max_pixels=max_pixels, patch_size=16, merge_size=2,
                                 temporal_patch_size=2, image_mean=[0.5] * 3, image_std=[0.5] * 3)


def synth(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    kind = rng.integers(0, 4)
    y, x = np.indices((h, w))
    if kind == 0:
        return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    if kind == 1:
        return np.stack([(x * 255 // max(w - 1, 1)), (y * 255 // max(h - 1, 1)), ((x + y) * 7) % 256], -1).astype(np.uint8)
    if kind == 2:
        img = np.full((h, w, 3), 240, np.uint8)
        for _ in range(6):
            y0, y1 = sorted(rng.integers(0, h + 1, 2))
            x0, x1 = sorted(rng.integers(0, w + 1, 2))
            img[y0:y1, x0:x1] = rng.integers(0, 256, 3)
        return img
    return np.where((x // 2 + y // 3) % 2 == 0, 255, 0).astype(np.uint8)[..., None].repeat(3, -1)


def encode(img: Image.Image, fmt: str, **kw) -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt, **kw)
    return base64.b64encode(buf.getvalue()).decode()


def png_stream(stream: bytes, width: int = 1, height: int = 1) -> str:
    """Wrap a controlled zlib stream in CRC-valid RGB PNG chunks."""
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    data = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", stream) + chunk(b"IEND", b""))
    return base64.b64encode(data).decode()


def run_tool(lines: list[str]) -> list[str]:
    out = subprocess.run([ROOT / "clef-tool", "image"], input="\n".join(lines) + "\n",
                         capture_output=True, text=True, check=True).stdout
    return out.split("\n")[: len(lines)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cases", type=int, default=60)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    failures = 0
    with tempfile.TemporaryDirectory(prefix="clef-image-") as tmp:
        cases = []   # (name, b64, expected PIL image, min_pixels, max_pixels)
        # decoder coverage: PNG color types, JPEG progressive/subsampling/grayscale, sizes
        for i in range(args.cases):
            h, w = (int(rng.integers(1, 97)), int(rng.integers(1, 97))) if i % 3 else (int(rng.integers(100, 700)), int(rng.integers(100, 700)))
            arr = synth(rng, h, w)
            base = Image.fromarray(arr)
            choice = i % 7
            if choice == 0:
                im, b64 = base, encode(base, "PNG")
            elif choice == 1:
                rgba = np.concatenate([arr, rng.integers(0, 256, (h, w, 1), dtype=np.uint8)], -1)
                im, b64 = Image.fromarray(rgba, "RGBA"), encode(Image.fromarray(rgba, "RGBA"), "PNG")
            elif choice == 2:
                gray = base.convert("L")
                im, b64 = gray, encode(gray, "PNG")
            elif choice == 3:
                la = Image.fromarray(np.concatenate([np.asarray(base.convert("L"))[..., None], rng.integers(0, 256, (h, w, 1), dtype=np.uint8)], -1), "LA")
                im, b64 = la, encode(la, "PNG")
            elif choice == 4:
                pal = base.convert("P", palette=Image.ADAPTIVE, colors=int(rng.integers(2, 256)))
                im, b64 = pal, encode(pal, "PNG")
            elif choice == 5:
                b64 = encode(base, "JPEG", quality=int(rng.integers(50, 100)), subsampling=int(rng.integers(0, 3)),
                             progressive=bool(rng.integers(0, 2)))
                im = Image.open(io.BytesIO(base64.b64decode(b64)))
            else:
                b64 = encode(base.convert("L"), "JPEG", quality=85, progressive=bool(rng.integers(0, 2)))
                im = Image.open(io.BytesIO(base64.b64decode(b64)))
            mn, mx = MIN_PIXELS, MAX_PIXELS
            if i % 5 == 4:   # media_kwargs-style bounds: force down- and up-scaling paths
                mn, mx = int(rng.choice([1024, 65536, 200000])), int(rng.choice([65536, 102400, 1048576]))
                mn = min(mn, mx)
            cases.append((f"case{i}", b64 if i % 2 else ("DATA:" if i % 4 else "data:") + "image/x;base64," + b64, im, mn, mx))
        # Exercise all DEFLATE block types independently of Pillow's compression defaults.
        arr = synth(np.random.default_rng(19), 256, 256)
        raw = b"".join(b"\0" + row.tobytes() for row in arr)
        for name, level, strategy in [("stored", 0, zlib.Z_DEFAULT_STRATEGY),
                                       ("fixed", 6, zlib.Z_FIXED), ("dynamic", 6, zlib.Z_DEFAULT_STRATEGY)]:
            compressor = zlib.compressobj(level, zlib.DEFLATED, zlib.MAX_WBITS, 8, strategy)
            stream = compressor.compress(raw) + compressor.flush()
            cases.append((name, png_stream(stream, 256, 256), Image.fromarray(arr), MIN_PIXELS, MAX_PIXELS))
        lines = [json.dumps({"image": b64, "out": f"{tmp}/{name}", "min_pixels": mn, "max_pixels": mx})
                 for name, b64, _, mn, mx in cases]
        results = run_tool(lines)
        for (name, _, im, mn, mx), got in zip(cases, results):
            ref_rgb = np.asarray(im.convert("RGB"))
            if got.startswith("ERR"):
                print(f"{name}: engine error: {got}")
                failures += 1
                continue
            meta = json.loads(got)
            H, W = ref_rgb.shape[:2]
            rh, rw = smart_resize(H, W, factor=32, min_pixels=mn, max_pixels=mx)
            if (meta["decoded"], meta["width"], meta["height"]) != ([W, H], rw, rh):
                print(f"{name}: size mismatch {meta} vs decoded {W}x{H} resized {rw}x{rh}")
                failures += 1
                continue
            dec = np.fromfile(f"{tmp}/{name}.rgb", np.uint8).reshape(H, W, 3)
            if not np.array_equal(dec, ref_rgb):
                print(f"{name}: decoded pixels differ ({int((dec != ref_rgb).sum())} values, max {int(np.abs(dec.astype(int) - ref_rgb).max())})")
                failures += 1
                continue
            t = tvF.pil_to_tensor(im.convert("RGB"))
            ref_resized = tvF.resize(t, [rh, rw], interpolation=tvF.InterpolationMode.BICUBIC, antialias=True).permute(1, 2, 0).numpy()
            got_resized = np.fromfile(f"{tmp}/{name}.resized", np.uint8).reshape(rh, rw, 3)
            if not np.array_equal(got_resized, ref_resized):
                d = np.abs(got_resized.astype(int) - ref_resized.astype(int))
                print(f"{name}: resized pixels differ ({W}x{H} -> {rw}x{rh}: {int((d > 0).sum())} values, max {int(d.max())})")
                failures += 1
                continue
            out = processor(mn, mx)(images=im, return_tensors="pt")
            ref_patches = out["pixel_values"].numpy()
            got_patches = np.fromfile(f"{tmp}/{name}.patches", np.float32).reshape(ref_patches.shape[0], -1)
            grid = out["image_grid_thw"][0].tolist()
            if grid != [1, meta["grid_h"], meta["grid_w"]] or got_patches.shape != ref_patches.shape or not np.array_equal(got_patches, ref_patches):
                print(f"{name}: patches differ (grid {grid} vs {meta}, max abs {float(np.abs(got_patches - ref_patches).max()) if got_patches.shape == ref_patches.shape else 'shape'})")
                failures += 1
                continue
            n = ref_patches.shape[0]
            raw = np.fromfile(f"{tmp}/{name}.pos", np.uint8)
            got_idx = raw[: n * 16].view(np.int32).reshape(n, 4)
            got_w = raw[n * 16:].view(np.float32).reshape(n, 4)
            ref_idx, ref_w = get_vision_bilinear_indices_and_weights(torch.tensor([grid]), 48, 2)
            ref_idx, ref_w = ref_idx.numpy().T, ref_w.numpy().T
            if not np.array_equal(got_idx, ref_idx) or not np.allclose(got_w, ref_w, rtol=0, atol=2e-7):
                print(f"{name}: position interpolation differs (idx equal {np.array_equal(got_idx, ref_idx)}, max |dw| {float(np.abs(got_w - ref_w).max())})")
                failures += 1
                continue
        print(f"images: {len(cases) - failures}/{len(cases)} byte-identical through decode, resize, patches and positions")

        # error handling: bad base64, truncated image, aspect ratio, unsupported bit depth, and the
        # two crafted JPEGs that overflowed the vendored decoder before its scan-header and sampling
        # checks (ASan-confirmed on the unmodified copy; both must be rejected, never decoded)
        buf = io.BytesIO(); Image.fromarray(np.zeros((4, 4), np.uint16)).save(buf, "PNG")
        wide = Image.fromarray(np.zeros((2, 500, 3), np.uint8))
        se255 = progressive_se255(np.random.default_rng(3).integers(0, 256, (48, 64, 3), dtype=np.uint8))
        bad = [json.dumps({"image": "abc", "out": f"{tmp}/bad0"}),
               json.dumps({"image": base64.b64encode(b"\x89PNG\r\n\x1a\nxx").decode(), "out": f"{tmp}/bad1"}),
               json.dumps({"image": encode(wide, "PNG"), "out": f"{tmp}/bad2"}),
               json.dumps({"image": base64.b64encode(buf.getvalue()).decode(), "out": f"{tmp}/bad3"}),
               json.dumps({"image": "data:image/png,notbase64", "out": f"{tmp}/bad4"}),
               json.dumps({"image": base64.b64encode(se255).decode(), "out": f"{tmp}/bad5"}),
               json.dumps({"image": LUMA_UNDER_CHROMA_JPEG, "out": f"{tmp}/bad6"})]
        raw = b"\0\x10\x20\x30"   # one RGB scanline, including its filter byte
        stream = zlib.compress(raw)
        corrupt_checksum = stream[:-1] + bytes([stream[-1] ^ 1])
        dictionary = zlib.compressobj(zdict=b"dictionary")
        malformed = [corrupt_checksum, stream[:-2], zlib.compress(raw[:-1]),
                     zlib.compress(raw + b"\0"), stream + b"trailing", stream + stream,
                     dictionary.compress(raw) + dictionary.flush()]
        bad.extend(json.dumps({"image": png_stream(s), "out": f"{tmp}/badpng{i}"})
                   for i, s in enumerate(malformed))
        png = base64.b64decode(png_stream(stream))
        for i, offset in enumerate((29, 41 + len(stream), len(png) - 4)):   # IHDR, IDAT, IEND CRCs
            corrupt = bytearray(png)
            corrupt[offset] ^= 1
            bad.append(json.dumps({"image": base64.b64encode(corrupt).decode(), "out": f"{tmp}/badcrc{i}"}))
        for line, got in zip(bad, run_tool(bad)):
            if not got.startswith("ERR"):
                print(f"expected an error for {line[:60]}: {got}")
                failures += 1
        # the crafted JPEGs' unmodified sources still decode, and PIL reads the unusual sampling layout
        Image.open(io.BytesIO(base64.b64decode(LUMA_UNDER_CHROMA_JPEG))).load()
        print("errors: bad base64, truncated PNG, aspect ratio, 16-bit PNG, a non-base64 data URL, a progressive scan "
              "with Se = 255, a luma-under-chroma JPEG, seven malformed zlib streams and three corrupt CRCs rejected"
              if not failures else "errors: see above")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
