"""Exact outputs across Flash short-GEMM tile boundaries and packed row counts.

Run under the shared GPU lock:
  .venv/bin/python -B tests/test_gemm_dispatch.py MODEL.gguf [--baseline /path/to/clef]
"""
from pathlib import Path
import argparse
import json
import os
import subprocess

from test_prefix_cache import exact, rows

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('model', type=Path)
parser.add_argument('--baseline', type=Path, default=ROOT / 'clef')
args = parser.parse_args()
env = {k: v for k, v in os.environ.items() if not k.startswith('CLEF_')}
lengths = (767, 768, 800, 801, 832, 864, 865, 960, 992, 993, 1023, 1024)
base = {'model': 'clef', 'questions': {
    'urgency': {'type': 'choice', 'instructions': 'How urgent is the situation?',
                'criteria': {'urgent': 'Immediate action needed', 'normal': 'Can wait',
                             'unknown': 'Insufficient evidence'}},
    'outage': {'type': 'noul', 'instructions': 'Is checkout unavailable?'}
}}


def payload(requests):
    return ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in requests)


def encode(requests):
    p = subprocess.run([str(ROOT / 'clef-tool'), 'encode-notrunc', str(args.model)],
                       input=payload(requests), text=True, capture_output=True, check=True, env=env)
    records = [json.loads(s) for s in p.stdout.splitlines()]
    if len(records) != len(requests):
        raise ValueError('incomplete encoded fixtures')
    return records


def sized(target, index):
    padding = target - 200
    # Alternating facts prevent agreement from depending on identical request content.
    fact = 'Checkout is unavailable.' if index % 2 else 'Checkout is healthy.'
    for _ in range(5):
        if padding < 0:
            raise ValueError('negative fixture padding')
        request = dict(base, state=fact + ' x' * padding)
        have = len(encode([request])[0]['input_ids'])
        if have == target:
            return request
        padding += target - have
    raise ValueError('could not construct exact token length ' + str(target))


def run(requests, flags=(), settings=None, baseline=False):
    binary = args.baseline if baseline else ROOT / 'clef'
    p = subprocess.run([str(binary), '-m', str(args.model), '--strict', '--no-truncate',
                        '--logits', *flags], input=payload(requests), text=True,
                       capture_output=True, env={**env, **(settings or {})})
    if p.returncode:
        raise ValueError('CLI failed: ' + p.stderr[-2000:])
    if len(rows(p.stdout)) != len(requests):
        raise ValueError('incomplete inference coverage')
    return p.stdout


requests = [sized(n, i) for i, n in enumerate(lengths)]
if tuple(len(r['input_ids']) for r in encode(requests)) != lengths:
    raise ValueError('fixture token counts changed')
reference = run(requests, baseline=True)
for label, flags in (('single', ()), ('packed', ('--batch', '4')),
                     ('template', ('--template-cache',))):
    checked = exact(run(requests, flags, {'CLEF_DEBUG_POISON': '1'}), reference)
    print(f'PASS: {label}, {len(requests)} full tile-boundary inputs, {checked} exact logits', flush=True)

# A forced BF16 rerun must remain independent of the short FP16 dispatch rule.
overflow_requests = [requests[i] for i in (1, 3, 9)]
fallback = run(overflow_requests, settings={'CLEF_DEBUG_F16_LIMIT': '1e-6'}, baseline=True)
checked = exact(run(overflow_requests, ('--batch', '4'),
                    {'CLEF_DEBUG_F16_LIMIT': '1e-6', 'CLEF_DEBUG_POISON': '1'}), fallback)
if rows(fallback) == [rows(reference)[i] for i in (1, 3, 9)]:
    raise ValueError('forced overflow did not exercise a changed result')
print(f'PASS: forced BF16 fallback, {checked} exact logits', flush=True)
