"""Grouped GEMMs must keep each image's attention and each record's overflow independent.

Usage: .venv/bin/python -B tests/test_vision_groups.py MODEL.gguf
"""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ref'))
import corpus_video


def main():
    cases = corpus_video.build()
    # Ragged 16-row GEMM tiles, the 1024-patch group boundary and unequal grids.
    for n, h, w in [(9, 64, 96), (33, 128, 128), (5, 192, 288)]:
        frames = [corpus_video.image(f) for f in corpus_video.frames(n, h, w)]
        cases.append(corpus_video.request({'frames': frames, 'fps': 8},
                     media_kwargs={'videos_kwargs': {'do_sample_frames': False}}))
    cases.append(dict(cases[0], images=[cases[0]['videos'][0]['frames'][0]] * 2,
                      media_kwargs={'min_pixels': 12288, 'max_pixels': 12288}))
    payload = ''.join(json.dumps(r) + '\n' for r in cases)
    clean = {k: v for k, v in os.environ.items() if not k.startswith('CLEF_')}
    modes = [{}, {'CLEF_VIS_COMP': '0'}, {'CLEF_VIS_F32': '0'}, {'CLEF_ACT_F16': '0'},
             {'CLEF_ATTN_REF': '1'}, {'CLEF_ACT_F16': '16', 'CLEF_DEBUG_F16_LIMIT': '1e-6'}]
    for mode in modes:
        expected = None
        for grouped, batch in [(False, 1), (True, 1), (True, 8)]:
            env = {**clean, **mode, 'CLEF_VIS_GROUP': str(int(grouped)), 'CLEF_DEBUG_POISON': '1'}
            run = subprocess.run([str(ROOT / 'clef'), '-m', sys.argv[1], '--logits', '--batch', str(batch)],
                                 input=payload, text=True, capture_output=True, check=True, env=env, timeout=600)
            rows = run.stdout.splitlines()
            assert len(rows) == len(cases) and all('error' not in json.loads(r) for r in rows), run.stdout
            if expected is None:
                expected = rows
            assert rows == expected, (mode, grouped, batch)
        print(f'vision groups: {mode or "default"}, {len(cases)} requests, batch 1/8 and poison: exact logits', flush=True)


if __name__ == '__main__':
    main()
