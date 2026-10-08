"""Seed corpus for tests/fuzz_image.c: small PNGs of every color type and bit depth, and JPEGs
across subsampling, progressive, grayscale, restart intervals and quality, plus the crafted files
the regression tests keep (tests/test_image.py). Usage: fuzz_image_seeds.py OUT_DIR"""
import base64
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_image  # noqa: E402  (crafted JPEGs; its main() only runs as a script)

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(1)
n = 0


def save(data: bytes, name: str) -> None:
    global n
    (out / f"{n:03d}-{name}").write_bytes(data)
    n += 1


def arr(h: int, w: int, c: int) -> np.ndarray:
    y, x = np.indices((h, w))
    base = np.stack([(x * 37 + y * 11) % 256, (x ^ y) & 255, (x * y) % 256, (x + 3 * y) % 256], -1)
    return (base[..., :c] + rng.integers(0, 4, (h, w, c))).clip(0, 255).astype(np.uint8)


for h, w in ((1, 1), (3, 7), (17, 33), (40, 40)):
    for mode, c in (("L", 1), ("LA", 2), ("RGB", 3), ("RGBA", 4)):
        a = arr(h, w, c)
        img = Image.fromarray(a[..., 0] if c == 1 else a, mode)
        for opt in ({}, {"optimize": True}, {"compress_level": 0}):
            buf = io.BytesIO()
            img.save(buf, "PNG", **opt)
            save(buf.getvalue(), f"{mode}-{w}x{h}.png")
    for bits in (1, 2, 4):
        pal = Image.fromarray(arr(h, w, 1)[..., 0] >> (8 - bits), "L").convert("P", palette=Image.ADAPTIVE, colors=1 << bits)
        buf = io.BytesIO()
        pal.save(buf, "PNG", bits=bits)
        save(buf.getvalue(), f"P{bits}-{w}x{h}.png")
        buf = io.BytesIO()
        pal.save(buf, "PNG", bits=bits, transparency=0)
        save(buf.getvalue(), f"P{bits}t-{w}x{h}.png")
    for mode in ("L", "RGB"):
        a = arr(h, w, 3)
        img = Image.fromarray(a[..., 0] if mode == "L" else a, mode)
        for q, sub, prog in ((90, 0, False), (75, 1, False), (50, 2, False), (85, 2, True), (95, 0, True)):
            for kw in ({}, {"optimize": True}, {"restart_marker_blocks": 1}):
                buf = io.BytesIO()
                try:
                    img.save(buf, "JPEG", quality=q, subsampling=sub, progressive=prog, **kw)
                except (TypeError, OSError, ValueError):
                    continue
                save(buf.getvalue(), f"{mode}-q{q}-s{sub}{'-prog' if prog else ''}-{w}x{h}.jpg")
save(test_image.progressive_se255(arr(16, 16, 3)), "progressive-se255.jpg")
save(base64.b64decode(test_image.LUMA_UNDER_CHROMA_JPEG), "luma-under-chroma.jpg")
# progressive 4:2:0 with a DC scan per component: a layout PIL cannot write, which the fuzzer
# otherwise never sees (review #3 found the decoder mishandled it)
save(base64.b64decode(test_image.SEPARATE_DC_SCANS_37X21), "separate-dc-37x21.jpg")
save(base64.b64decode(test_image.SEPARATE_DC_SCANS_32X32), "separate-dc-32x32.jpg")
# RGB-coded JPEGs (Adobe transform 0; ids 'R','G','B'), which libjpeg copies instead of converting: the
# differential fuzzer never produced one from YCbCr seeds, and review #3 found them decoded wrong
for prog in (False, True):
    buf = io.BytesIO()
    Image.fromarray(arr(17, 33, 3)).save(buf, "JPEG", quality=85, subsampling=2, progressive=prog)
    save(test_image.with_colour_markers(buf.getvalue(), jfif=False, adobe=0), f"adobe0{'-prog' if prog else ''}.jpg")
    save(test_image.with_colour_markers(buf.getvalue(), jfif=False, ids=b"RGB"), f"rgb-ids{'-prog' if prog else ''}.jpg")
# Single blocks at the IDCT range boundary and with AC categories above 10, and a progressive file
# whose quantization table is redefined after its first scan (review #3, Codex on 3f3eb03)
for name, data in (("idct-511", test_image.single_coefficient_jpeg(0, 4088)), ("idct-513", test_image.single_coefficient_jpeg(0, 4104)),
                   ("ac-category-11", test_image.single_coefficient_jpeg(1, 1024)), ("ac-category-15", test_image.single_coefficient_jpeg(5, 16384))):
    save(data, f"{name}.jpg")
buf = io.BytesIO()
Image.fromarray(arr(16, 24, 3)).save(buf, "JPEG", quality=80, progressive=True)
segs = test_image.jpeg_segments(buf.getvalue())
first = next(i for i, (m, _) in enumerate(segs) if m == 0xDA)
save(b"".join(x for _, x in segs[:first + 1]) + test_image.dqt(0, bytes([1] * 64)) + b"".join(x for _, x in segs[first + 1:]),
     "dqt-redefined-after-scan.jpg")
save(base64.b64decode(test_image.AC_FIRST_OVERSHOOT_JPEG), "ac-first-overshoot-fuzz.jpg")
# a PNG with an empty IDAT before its data, which the decoder skips (Codex on 600ddfe)
buf = io.BytesIO()
Image.fromarray(arr(6, 8, 3)).save(buf, "PNG")
data = buf.getvalue()
first_idat = data.index(b"IDAT") - 4
empty = b"\x00\x00\x00\x00IDAT" + (0x35AF061E).to_bytes(4, "big")   # CRC-32 of "IDAT"
save(data[:first_idat] + empty + data[first_idat:], "empty-first-idat.png")
# a sequential frame with one scan per component (Codex on a82292d)
save(base64.b64decode(test_image.SEQUENTIAL_SCAN_PER_COMPONENT_JPEG), "sequential-scan-per-component.jpg")
save(test_image.with_scan_order(base64.b64decode(test_image.SEQUENTIAL_Y_THEN_CBCR_JPEG), 1, (1, 0)), "sequential-cr-then-cb.jpg")
print(f"{n} seeds in {out}")
