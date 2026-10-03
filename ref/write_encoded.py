"""Write golden/<name>/encoded.jsonl for a golden directory that lacks it.

The streamed FP32 oracle used to omit encoded.jsonl, which tests/test_parity.py requires (review
#4). The encoding depends only on the requests and the tokenizer, so it is recomputed here from
requests.jsonl with the snapshot's processor, the way the oracles build it, without loading
weights. Refuses to overwrite an existing file that differs.

Usage: write_encoded.py MODEL_DIR GOLDEN_DIR [--out FILE]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import golden_io  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("golden_dir", type=Path)
    ap.add_argument("--out", type=Path, help="write here instead of GOLDEN_DIR/encoded.jsonl")
    args = ap.parse_args()
    sys.path.insert(0, str(args.model_dir.resolve()))
    from joint_schema_model import encode_record
    from transformers import AutoProcessor

    tok = AutoProcessor.from_pretrained(args.model_dir).tokenizer   # as load_release_model builds it
    requests = [json.loads(l) for l in open(args.golden_dir / "requests.jsonl")]
    text = "".join(golden_io.encoded_line(r, encode_record(tok, r)) for r in requests)
    out = args.out or args.golden_dir / "encoded.jsonl"
    if out.exists() and out.read_text() != text:
        sys.exit(f"{out} exists and differs; not overwriting")
    out.write_text(text)
    print(f"wrote {out} ({len(requests)} requests)")


if __name__ == "__main__":
    main()
