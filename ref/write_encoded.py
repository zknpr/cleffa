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
    config = json.loads((args.model_dir/'config.json').read_text())
    is_omni = config.get('model_type') == 'qwen3_omni_moe'
    if is_omni:
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
        from verify_snapshot import load_manifest, verify_snapshot
        from omni import REVISION
        bad = verify_snapshot(args.model_dir,load_manifest(REVISION))
        if bad:
            raise ValueError('snapshot verification failed: '+'; '.join(bad))
    sys.path.insert(0, str(args.model_dir.resolve()))
    from joint_schema_model import encode_record
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model_dir)
    tok = processor.tokenizer   # as load_release_model builds it
    requests = [json.loads(l) for l in open(args.golden_dir / "requests.jsonl")]
    if is_omni:
        import torch
        from transformers import Qwen3OmniMoeThinkerConfig
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeThinkerForConditionalGeneration
        from joint_schema_model import load_image
        from omni_media import media_layout, encode_reference
        # Positions use only config and token/grid tensors. Meta construction
        # keeps this host oracle independent of the 61 GB inference weights.
        with torch.device('meta'):
            thinker = Qwen3OmniMoeThinkerForConditionalGeneration(Qwen3OmniMoeThinkerConfig(**config['thinker_config']))
        lines = []
        for request in requests:
            encoded = encode_reference(encode_record,request,processor)
            layout = media_layout(encoded,thinker,processor) if encoded.media else None
            lines.append(golden_io.encoded_line(request,encoded,media_layout=layout))
        text = ''.join(lines)
    else:
        text = "".join(golden_io.encoded_line(r, encode_record(tok, r)) for r in requests)
    out = args.out or args.golden_dir / "encoded.jsonl"
    if out.exists() and out.read_text() != text:
        sys.exit(f"{out} exists and differs; not overwriting")
    out.write_text(text)
    print(f"wrote {out} ({len(requests)} requests)")


if __name__ == "__main__":
    main()
