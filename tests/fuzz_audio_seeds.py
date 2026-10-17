"""Seed corpus for tests/fuzz_audio.c: WAV files in both accepted encodings with the chunk-table
variants the parser handles (extra and odd-sized chunks, data before fmt, an 18-byte fmt, extreme
and subnormal samples), record batches of clips with different lengths, requests in every audio
form (sample arrays, data URLs, base64 objects, video soundtracks) and synthetic records. Each seed
starts with the harness's mode byte. Usage: fuzz_audio_seeds.py OUT_DIR"""
import base64
import json
import math
import struct
import sys
from pathlib import Path

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
n = 0


def save(mode: int, data: bytes, name: str) -> None:
    global n
    (out / f"{n:03d}-{name}").write_bytes(bytes([mode]) + data)
    n += 1


def chunk(tag: bytes, body: bytes) -> bytes:
    return tag + struct.pack("<I", len(body)) + body + (b"\0" if len(body) & 1 else b"")


def fmt(float32: bool, extra: bool = False) -> bytes:
    width = 4 if float32 else 2
    body = struct.pack("<HHIIHH", 3 if float32 else 1, 1, 16000, 16000 * width, width, 8 * width)
    return chunk(b"fmt ", body + (b"\0\0" if extra else b""))


def samples(count: int, float32: bool, amp: float = 0.3, freq: float = 440) -> bytes:
    v = [amp * math.sin(2 * math.pi * freq * i / 16000) for i in range(count)]
    return struct.pack(f"<{count}f", *v) if float32 else struct.pack(f"<{count}h", *(round(x * 32767) for x in v))


def wav(*chunks: bytes) -> bytes:
    body = b"WAVE" + b"".join(chunks)
    return b"RIFF" + struct.pack("<I", len(body)) + body


def pcm(count: int) -> bytes:
    return wav(fmt(False), chunk(b"data", samples(count, False)))


def flt(count: int) -> bytes:
    return wav(fmt(True), chunk(b"data", samples(count, True)))


# mode 0: one WAV
save(0, pcm(800), "pcm16")
save(0, flt(401), "float32")
save(0, pcm(201), "pcm16-minimum")
save(0, flt(3200), "float32-long")
save(0, wav(chunk(b"data", samples(480, False)), fmt(False)), "data-before-fmt")
save(0, wav(chunk(b"LIST", b"INFOx"), fmt(False), chunk(b"junk", b""), chunk(b"data", samples(300, False))), "odd-chunk")
save(0, wav(fmt(True, extra=True), chunk(b"data", samples(256, True))), "fmt18")
save(0, wav(fmt(False), chunk(b"data", struct.pack("<4h", -32768, 32767, 0, -1) * 64)), "pcm16-extremes")
special = struct.pack("<4f", 1e-45, -0.0, 3.4e38, -1e30) * 64   # subnormal, signed zero, huge
save(0, wav(fmt(True), chunk(b"data", special)), "float32-extremes")


def prefixed(*files: bytes) -> bytes:
    return b"".join(struct.pack("<H", len(f)) + f for f in files)


# mode 1: the clips of one record (a short clip pads beside a longer one)
save(1, prefixed(pcm(150), pcm(900)), "short-and-long")
save(1, prefixed(flt(401), flt(401)), "equal")
save(1, prefixed(pcm(161), flt(1600), pcm(3300)), "three")


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def request(opts: bytes, body: dict) -> bytes:
    return opts + json.dumps(body, separators=(",", ":")).encode()


# mode 2: option bytes (sample cap and item limit, max_length, reserve), then the request.
# 0x13: 48,000-sample cap, four items; max_length 1 + 4 * 255; no reserve.
capped = bytes([0x13, 255, 0])
# 0x00: no sample cap or item limit; max_length 40; two reserved tokens.
open_ = bytes([0x00, 39, 2])
arr = [round(0.5 * math.sin(i / 7), 4) for i in range(250)] + [0, 1, -1]
for opts, tag in ((capped, "capped"), (open_, "open")):
    save(2, request(opts, {"audio": [arr]}), f"samples-{tag}")
    save(2, request(opts, {"audio": ["data:audio/wav;base64," + b64(pcm(640))]}), f"data-url-{tag}")
    save(2, request(opts, {"audio": [{"content_type": "audio/wav", "base64": b64(flt(500))}]}), f"object-{tag}")
    save(2, request(opts, {"audio": [arr[:150], "data:audio/x-wav;base64," + b64(pcm(1200))]}), f"two-{tag}")
    save(2, request(opts, {"audio": [arr], "videos": [{"frames": [], "audio": "data:audio/wave;base64," + b64(pcm(400))}]}),
         f"soundtrack-{tag}")
    save(2, request(opts, {"videos": [{"audio": {"content_type": "audio/wav", "base64": b64(pcm(320))}},
                                      {"audio": arr}]}), f"two-soundtracks-{tag}")

# mode 3: stills, videos, standalone clips, soundtrack flag, then grids and clip lengths
save(3, bytes([1, 2, 1, 1, 3, 4, 2, 1, 1, 5, 2, 7, 1, 200, 0, 3, 2]), "image-videos-tracks")
save(3, bytes([0, 1, 2, 1, 4, 0, 0, 90, 1, 44, 0, 0, 0]), "video-track-standalone")
save(3, bytes([2, 0, 3, 0, 1, 1, 5, 5, 100, 0, 101, 0, 7, 0, 4, 4]), "stills-standalone")
print(f"{n} seeds in {out}")
