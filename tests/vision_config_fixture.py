"""Write small header-only GGUFs for tests/test_vision_config.c: the clef.* keys of a converted
model with no tensors, and copies with one vision geometry field set to 2^24 (the most cfg_u32
admits). Usage: vision_config_fixture.py MODEL.gguf OUT_DIR"""
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFReader, GGUFValueType, GGUFWriter


def fields(path: str) -> list[tuple[str, object, GGUFValueType, GGUFValueType | None]]:
    out = []
    for name, f in GGUFReader(path).fields.items():
        if not name.startswith("clef."):
            continue   # tokenizer and general.* keys are not read by load_config
        vt = f.types[0]
        if vt == GGUFValueType.ARRAY:
            sub = f.types[1]
            if sub == GGUFValueType.STRING:
                val = [bytes(f.parts[i]).decode() for i in f.data]
            else:
                val = [f.parts[i][0].item() for i in f.data]
            out.append((name, val, vt, sub))
        elif vt == GGUFValueType.STRING:
            out.append((name, bytes(f.parts[f.data[0]]).decode(), vt, None))
        else:
            out.append((name, f.parts[f.data[0]][0].item(), vt, None))
    return out


def write(path: Path, kvs, override: dict, drop: frozenset = frozenset()) -> None:
    w = GGUFWriter(str(path), "clef")
    for name, val, vt, sub in kvs:
        if name not in drop:
            w.add_key_value(name, override.get(name, val), vt, sub_type=sub)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def main() -> None:
    src, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    kvs = fields(src)
    write(out / "vision-ok.gguf", kvs, {})
    write(out / "vision-patch-2p24.gguf", kvs, {"clef.vision.patch_size": 1 << 24})
    write(out / "vision-temporal-2p24.gguf", kvs, {"clef.vision.temporal_patch_size": 1 << 24})
    # review #3: an M-RoPE layout the kernels do not implement, or none, and token ids outside the
    # vocabulary, must be refused at load
    vocab = next(v for n, v, _, _ in kvs if n == "clef.vocab_size")
    write(out / "vision-mrope-16-8-8.gguf", kvs, {"clef.rope.mrope_section": [16, 8, 8]})
    write(out / "vision-mrope-missing.gguf", kvs, {}, frozenset({"clef.rope.mrope_section"}))
    write(out / "vision-image-id-vocab.gguf", kvs, {"clef.vision.image_token_id": vocab})
    write(out / "vision-video-id-2p31.gguf", kvs, {"clef.vision.video_token_id": 1 << 31})
    print(f"wrote 7 fixtures with {len(kvs)} clef.* keys to {out}")


if __name__ == "__main__":
    main()
