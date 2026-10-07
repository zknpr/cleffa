"""Exact vision prefix/feature cache reuse, identity transitions and overflow invalidation.

Usage: .venv/bin/python -B tests/test_vision_cache.py MODEL VISION_REQUESTS
"""
import argparse
import base64
import io
import json
import os
from pathlib import Path
import re
import subprocess

from PIL import Image, ImageOps

from test_prefix_cache import cold_reference, exact, payload, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model')
    parser.add_argument('corpus', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    env = {k: v for k, v in os.environ.items() if not k.startswith('CLEF_')}

    def run(requests, cache=False, settings=None):
        command = [str(root / 'clef'), '-m', args.model, '--logits', '--strict', '--no-truncate', '--time', '--batch', '1']
        if cache:
            command.append('--prefix-cache')
        result = subprocess.run(command, input=payload(requests), text=True, capture_output=True,
                                env={**env, **(settings or {})}, check=True, timeout=600)
        if len(rows(result.stdout)) != len(requests) or any('error' in r for r in rows(result.stdout)):
            raise AssertionError(result.stderr)
        return result.stdout, result.stderr

    corpus = rows(args.corpus.read_text())
    by_id = {r['id']: r for r in corpus}
    a, b = by_id['v001'], by_id['v009']
    # Empty state leaves the snapshot inside the large image's last partial 32-token block.
    partial = dict(b, state='')
    changed_q = dict(partial, questions=a['questions'])
    changed_state = dict(partial, state='A different question context after the same image.')
    # Same encoded placeholder lengths but different pixel content; order matters as well.
    encoded = a['images'][0].split(',', 1)[-1]
    decoded = Image.open(io.BytesIO(base64.b64decode(encoded))).convert('RGB')
    output = io.BytesIO()
    ImageOps.invert(decoded).save(output, format='PNG')
    other = dict(a, images=[base64.b64encode(output.getvalue()).decode()])
    pair = dict(a, images=[a['images'][0], other['images'][0]])
    swapped = dict(pair, images=list(reversed(pair['images'])))
    resized = dict(b, media_kwargs={'min_pixels': 65536, 'max_pixels': 262144})
    text = {k: v for k, v in a.items() if k not in ('images', 'media_kwargs')}
    text['state'] = 'A text-only cache replacement. ' * 90
    sequence = [a, a, other, other, a, partial, partial, changed_q, changed_state, pair, pair,
                swapped, swapped, resized, resized, b, text, text, b, b]
    expected = cold_reference(sequence, run)
    actual, log = run(sequence, True, {'CLEF_DEBUG_POISON': '1', 'CLEF_STAGE_TIME': '1'})
    count = exact(actual, expected)
    reused = [int(x) for x in re.findall(r'prefix cache reused (\d+) of', log)]
    if len(reused) != len(sequence):
        raise AssertionError(log)
    for i in (1, 3, 6, 7, 10, 12, 14, 17, 19):
        assert reused[i] > 0, (i, reused)
    for i in (0, 2, 4, 5, 9, 11, 13, 15, 16, 18):
        assert reused[i] == 0, (i, reused)
    assert len(re.findall(r'image cache reused \d+ features', log)) == 9, log
    print(f'PASS: {len(sequence)} image/text transitions, {count} exact logits; prefix rows {reused}', flush=True)

    # Every corpus image is filled and hit, including multiple images and unusual geometry.
    sequence = [r for r in corpus for _ in range(2)]
    expected = cold_reference(sequence, run)
    actual, _ = run(sequence, True, {'CLEF_DEBUG_POISON': '1'})
    print(f'PASS: complete vision corpus cache parity, {exact(actual, expected)} exact logits', flush=True)

    sequence = [a, a]
    expected, _ = run(sequence, settings={'CLEF_ACT_F16': '0'})
    actual, log = run(sequence, True, {'CLEF_DEBUG_F16_LIMIT': '1e-6', 'CLEF_DEBUG_POISON': '1', 'CLEF_STAGE_TIME': '1'})
    exact(actual, expected)
    assert not re.search(r'prefix cache reused [1-9]|image cache reused', log), log
    print('PASS: overflowing passes never populate image or prefix reuse; exact BF16 fallback', flush=True)


if __name__ == '__main__':
    main()
