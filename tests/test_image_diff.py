"""Differential decode test: JPEGs and PNGs from several encoders, decoded by cleffa (clef-tool image)
and by Pillow (libjpeg-turbo / Pillow's PNG decoder), compared pixel for pixel.

tests/test_image.py checks parity on files Pillow itself wrote; a decoder can be wrong on layouts
Pillow never produces and no sanitizer notices (review #3: a valid progressive file with a DC scan
per component decoded with values off by 49). This builds files with cjpeg (sampling factors,
progressive scan scripts with spectral selection and successive approximation, restart intervals,
quantizer settings), jpegtran (grayscale conversion and lossless rotations of cjpeg output), macOS sips (Apple's ImageIO encoder), ffmpeg's MJPEG and PNG encoders and Pillow,
over several image contents and odd sizes. Encoders that are not installed are skipped and reported.

Outcomes per file: identical; refused by both; refused by cleffa for a documented reason read from
the file's own headers (arithmetic coding, lossless or 12-bit JPEG, four components, a first
component sampled below another, 16-bit or interlaced PNG); anything else fails: a pixel
difference, or a refusal of a file Pillow decodes with no documented reason.
Usage: .venv/bin/python -B tests/test_image_diff.py [--keep DIR]
"""
from __future__ import annotations

import argparse
import base64
import io
import itertools
import json
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def contents(h: int, w: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    y, x = np.indices((h, w))
    grad = np.stack([x * 255 // max(w - 1, 1), y * 255 // max(h - 1, 1), (x + y) * 127 // max(w + h - 2, 1)], -1)
    edges = np.where(((x // 4) + (y // 3)) % 2, 230, 20)[..., None].repeat(3, -1)
    edges[..., 1] = (edges[..., 1] + x * 3) % 256
    noise = rng.integers(0, 256, (h, w, 3))
    flat = np.full((h, w, 3), [200, 40, 90])
    return {k: v.astype(np.uint8) for k, v in {"grad": grad, "edges": edges, "noise": noise, "flat": flat}.items()}


SIZES = [(1, 1), (7, 5), (16, 16), (17, 15), (33, 31), (37, 100), (129, 257), (240, 320)]

SCAN_SCRIPTS = {
    # one DC scan per component, then each component's AC in full
    "dc-per-comp": "0: 0 0 0 0; 1: 0 0 0 0; 2: 0 0 0 0; 0: 1 63 0 0; 1: 1 63 0 0; 2: 1 63 0 0;",
    # interleaved DC with successive approximation, spectral bands, refinements
    "sa-bands": "0,1,2: 0 0 0 1; 0: 1 5 0 2; 2: 1 63 0 1; 1: 1 63 0 1; 0: 6 63 0 2; 0: 1 63 2 1; 0,1,2: 0 0 1 0; "
                "2: 1 63 1 0; 1: 1 63 1 0; 0: 1 63 1 0;",
    # luma DC alone, chroma DC together, AC in several bands
    "mixed-dc": "0: 0 0 0 0; 1,2: 0 0 0 0; 0: 1 9 0 0; 0: 10 63 0 0; 1: 1 63 0 0; 2: 1 63 0 0;",
    # DC refinement in separate per-component scans
    "dc-refine-sep": "0: 0 0 0 1; 1: 0 0 0 1; 2: 0 0 0 1; 0: 0 0 1 0; 1: 0 0 1 0; 2: 0 0 1 0; "
                     "0: 1 63 0 0; 1: 1 63 0 0; 2: 1 63 0 0;",
}
GRAY_SCRIPTS = {
    "gray-sa": "0: 0 0 0 1; 0: 1 5 0 2; 0: 6 63 0 2; 0: 1 63 2 1; 0: 0 0 1 0; 0: 1 63 1 0;",
}


def cjpeg_variants(tmp: Path, gray: bool):
    q = ["-quality", "85"]
    out = []
    if gray:
        out += [("gray", ["-grayscale", *q]), ("gray-prog", ["-grayscale", "-progressive", *q]),
                ("gray-restart", ["-grayscale", "-restart", "1B", *q])]
        for name, script in GRAY_SCRIPTS.items():
            p = tmp / f"{name}.txt"
            p.write_text(script)
            out.append((name, ["-grayscale", "-progressive", "-scans", str(p), *q]))
        return out
    for samp in ("1x1,1x1,1x1", "2x1,1x1,1x1", "1x2,1x1,1x1", "2x2,1x1,1x1", "4x1,1x1,1x1", "4x2,1x1,1x1",
                 "2x2,1x2,1x1", "1x1,2x2,1x1"):
        out.append((f"s{samp.replace(',', '-')}", ["-sample", samp, *q]))
        out.append((f"s{samp.replace(',', '-')}-prog", ["-sample", samp, "-progressive", *q]))
    for name, script in SCAN_SCRIPTS.items():
        p = tmp / f"{name}.txt"
        p.write_text(script)
        for samp in ("2x2,1x1,1x1", "2x1,1x1,1x1", "1x1,1x1,1x1"):
            out.append((f"{name}-{samp.replace(',', '-')}", ["-sample", samp, "-progressive", "-scans", str(p), *q]))
    out += [("restart-1", ["-sample", "2x2,1x1,1x1", "-restart", "1B", *q]),
            ("restart-2-prog", ["-sample", "2x2,1x1,1x1", "-restart", "2", "-progressive", *q]),
            ("optimize", ["-optimize", *q]), ("q1", ["-quality", "1"]), ("q100", ["-quality", "100"]),
            ("dct-float", ["-dct", "float", *q]), ("smooth", ["-smooth", "50", *q]),
            ("arithmetic", ["-arithmetic", *q]), ("12bit", ["-precision", "12", *q]), ("lossless", ["-lossless", "1"])]
    return out


def encode_all(tmp: Path) -> tuple[list[tuple[str, bytes]], dict[str, int]]:
    files: list[tuple[str, bytes]] = []
    used: dict[str, int] = {}
    have = {t: shutil.which(t) is not None for t in ("cjpeg", "jpegtran", "sips", "ffmpeg")}

    def add(enc: str, name: str, data: bytes) -> None:
        files.append((f"{enc}:{name}", data))
        used[enc] = used.get(enc, 0) + 1

    for (h, w), seed in zip(SIZES, itertools.count(1)):
        for cname, arr in contents(h, w, seed).items():
            if (h, w) in ((1, 1), (7, 5)) and cname != "grad":
                continue
            stem = f"{cname}-{w}x{h}"
            img = Image.fromarray(arr)
            # Pillow
            for sub, prog in ((0, False), (2, True), (1, False)):
                buf = io.BytesIO()
                img.save(buf, "JPEG", quality=80, subsampling=sub, progressive=prog)
                add("pillow", f"{stem}-s{sub}{'-prog' if prog else ''}.jpg", buf.getvalue())
            for mode in ("RGB", "RGBA", "L", "P"):
                buf = io.BytesIO()
                (img.convert(mode) if mode != "P" else img.convert("P", palette=Image.Palette.ADAPTIVE, colors=37)).save(buf, "PNG")
                add("pillow", f"{stem}-{mode}.png", buf.getvalue())
            ppm = tmp / f"{stem}.ppm"
            pgm = tmp / f"{stem}.pgm"
            img.save(ppm)
            img.convert("L").save(pgm)
            if have["cjpeg"]:
                for gray, srcf in ((False, ppm), (True, pgm)):
                    for vname, args in cjpeg_variants(tmp, gray):
                        r = subprocess.run(["cjpeg", *args, str(srcf)], capture_output=True)
                        if r.returncode == 0 and r.stdout:
                            add("cjpeg", f"{stem}-{vname}.jpg", r.stdout)
            if have["jpegtran"]:
                # lossless transforms of cjpeg output: grayscale conversion, and rotations that turn
                # 4:2:2 into 4:4:0
                for samp in ("2x2,1x1,1x1", "2x1,1x1,1x1"):
                    r = subprocess.run(["cjpeg", "-sample", samp, "-quality", "85", str(ppm)], capture_output=True)
                    if r.returncode or not r.stdout:
                        continue
                    for tname, targs in (("gray", ["-grayscale"]), ("gray-prog", ["-grayscale", "-progressive"]),
                                         ("transpose", ["-transpose", "-trim"]), ("rot90", ["-rotate", "90", "-trim"]),
                                         ("flip", ["-flip", "horizontal", "-trim"])):
                        t2 = subprocess.run(["jpegtran", *targs], input=r.stdout, capture_output=True)
                        if t2.returncode == 0 and t2.stdout:
                            add("jpegtran", f"{stem}-{samp.split(',')[0]}-{tname}.jpg", t2.stdout)
            if have["sips"] and w >= 2 and h >= 2:
                png = tmp / f"{stem}.png"
                img.save(png)
                for fmt, opts in (("jpeg", ["-s", "formatOptions", "70"]), ("jpeg", ["-s", "formatOptions", "best"]), ("png", [])):
                    dst = tmp / f"{stem}-sips-{fmt}-{opts[-1] if opts else 'd'}.{fmt[:3]}"
                    r = subprocess.run(["sips", "-s", "format", fmt, *opts, str(png), "--out", str(dst)], capture_output=True)
                    if r.returncode == 0 and dst.exists():
                        add("sips", dst.name, dst.read_bytes())
            if have["ffmpeg"] and w % 2 == 0 and h % 2 == 0:
                for pix in ("yuvj420p", "yuvj422p", "yuvj444p"):
                    dst = tmp / f"{stem}-{pix}.jpg"
                    r = subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(ppm), "-pix_fmt", pix, "-q:v", "3",
                                        "-frames:v", "1", str(dst)], capture_output=True)
                    if r.returncode == 0 and dst.exists():
                        add("ffmpeg", dst.name, dst.read_bytes())
                for pix in ("rgb24", "rgba", "gray", "pal8", "rgb48be", "gray16be"):
                    dst = tmp / f"{stem}-{pix}.png"
                    r = subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(ppm), "-pix_fmt", pix,
                                        "-frames:v", "1", str(dst)], capture_output=True)
                    if r.returncode == 0 and dst.exists():
                        add("ffmpeg", dst.name, dst.read_bytes())
    for t, ok in have.items():
        if not ok:
            used[f"{t} (not installed, skipped)"] = 0
    return files, used


def documented_refusal(data: bytes) -> str | None:
    """The engine's documented divergences, read from the file's headers."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        if len(data) >= 29 and data[12:16] == b"IHDR":
            depth, ctype, interlace = data[24], data[25], data[28]
            if depth == 16:
                return "16-bit PNG"
            if interlace:
                return "interlaced PNG"
            if ctype in (0, 4) and depth < 8:
                return "low-bit grayscale PNG"
        return None
    pos = 2
    while pos + 4 <= len(data):
        if data[pos] != 0xFF:
            pos += 1
            continue
        m = data[pos + 1]
        if m in (0xD8, 0x01) or 0xD0 <= m <= 0xD7 or m == 0xFF:
            pos += 2 if m != 0xFF else 1
            continue
        seg = struct.unpack(">H", data[pos + 2:pos + 4])[0]
        if 0xC0 <= m <= 0xCF and m not in (0xC4, 0xC8, 0xCC):
            if m >= 0xC9:
                return "arithmetic-coded JPEG"
            if m in (0xC3, 0xC7, 0xCB, 0xCF):
                return "lossless JPEG"
            if data[pos + 4] != 8:
                return "12-bit JPEG"
            nc = data[pos + 9]
            if nc == 4:
                return "CMYK JPEG"
            samp = [data[pos + 11 + 3 * i] for i in range(nc)]
            hs, vs = [s >> 4 for s in samp], [s & 15 for s in samp]
            if nc > 1 and (hs[0] != max(hs) or vs[0] != max(vs)):
                return "first component sampled below another"
            return None
        if m == 0xDA:
            return None
        pos += 2 + seg
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keep", type=Path, help="write the corpus and every failing file here")
    args = ap.parse_args()
    with tempfile.TemporaryDirectory(prefix="clef-image-diff-") as t:
        tmp = Path(t)
        files, used = encode_all(tmp)
        lines = [json.dumps({"image": base64.b64encode(data).decode(), "out": str(tmp / f"out{i}"),
                             "min_pixels": 64, "max_pixels": 1 << 20}) for i, (_, data) in enumerate(files)]
        r = subprocess.run([str(ROOT / "clef-tool"), "image"], input="\n".join(lines) + "\n", capture_output=True, text=True)
        results = r.stdout.splitlines()
        if len(results) != len(files):
            sys.exit(f"clef-tool image answered {len(results)} of {len(files)} lines: {r.stderr[-400:]}")
        counts: dict[str, int] = {}
        failures = []
        for i, ((name, data), got) in enumerate(zip(files, results)):
            ref_err = ""
            try:
                ref = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
            except Exception as e:   # noqa: BLE001  Pillow refused it
                ref, ref_err = None, str(e)
            ok = not got.startswith("ERR")
            if ref is None and not ok:
                key = "refused by both"
            elif ref is None:
                key = "decoded by cleffa only (Pillow refused)"
                failures.append((name, f"Pillow: {ref_err}"))
            elif not ok:
                reason = documented_refusal(data)
                key = f"refused, documented: {reason}" if reason else "refused, UNDOCUMENTED"
                if not reason:
                    failures.append((name, got))
            else:
                dec = np.fromfile(tmp / f"out{i}.rgb", np.uint8)
                if dec.size != ref.size:
                    key = "size differs"
                    failures.append((name, f"cleffa {dec.size} values, Pillow {ref.shape}"))
                else:
                    diff = np.abs(dec.reshape(ref.shape).astype(int) - ref.astype(int))
                    if diff.max() == 0:
                        key = "identical"
                    else:
                        key = "PIXELS DIFFER"
                        failures.append((name, f"{int((diff > 0).sum())} values differ, max {int(diff.max())}"))
            counts[key] = counts.get(key, 0) + 1
            if args.keep and key not in ("identical", "refused by both"):
                args.keep.mkdir(parents=True, exist_ok=True)
                (args.keep / name.replace(":", "_").replace("/", "_")).write_bytes(data)
        print(f"files: {len(files)} from " + ", ".join(f"{k} {v}" for k, v in sorted(used.items())))
        for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"  {v:5d}  {k}")
        for name, why in failures[:40]:
            print(f"  FAIL {name}: {why}")
        if len(failures) > 40:
            print(f"  ... {len(failures) - 40} more")
        if failures:
            sys.exit("FAIL")
        print("PASS")


if __name__ == "__main__":
    main()
