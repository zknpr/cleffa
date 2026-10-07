"""Exact cache parity, transitions and failure recovery on a public request corpus.

  .venv/bin/python -B tests/test_prefix_cache.py MODEL.gguf REQUESTS.jsonl

Run under the shared GPU lock. Cold references are deduplicated without reordering
questions or options. Failure tests use bounded requests; the transition sequence
still exercises the full longest context. No oracle or hosted API is called.
"""
import argparse
import io
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess


def rows(data):
    # Unicode line separators can occur inside a JSON string.
    return [json.loads(line) for line in io.StringIO(data)]


def payload(requests):
    return ''.join(json.dumps(r, ensure_ascii=True) + '\n' for r in requests)


def exact(actual, expected, errors=False):
    left, right = rows(actual), rows(expected)
    if len(left) != len(right):
        raise ValueError('incomplete output coverage')
    count = 0
    for i, (x, y) in enumerate(zip(left, right, strict=True)):
        if errors and 'error' in x:
            if set(x) != {'error'} or 'prefix cache entry' not in x['error']:
                raise ValueError(f'unexpected error at record {i}: {x}')
            continue
        if set(x) != set(y) or not x or 'error' in x:
            raise ValueError(f'question coverage differs at record {i}')
        for q in x:
            if len(x[q]) != len(y[q]) or not x[q] or not all(math.isfinite(v) for v in x[q]+y[q]):
                raise ValueError(f'nonfinite or incomplete logits: {i}/{q}')
            if struct.pack(f'{len(x[q])}f', *x[q]) != struct.pack(f'{len(y[q])}f', *y[q]):
                raise ValueError(f'logits differ: record {i}, question {q}')
            count += len(x[q])
    return count


def cold_reference(requests, run, settings=None):
    # Key order is input data: sorting it changes schema token IDs and option order.
    unique = list(dict.fromkeys(json.dumps(r) for r in requests))
    output, _ = run(list(map(json.loads, unique)), settings=settings)
    lines = list(io.StringIO(output))
    if len(lines) != len(unique):
        raise ValueError('incomplete cold reference')
    lookup = dict(zip(unique, lines, strict=True))
    return ''.join(lookup[json.dumps(r)].rstrip('\n')+'\n' for r in requests)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('corpus', type=Path)
    parser.add_argument('--binary', type=Path, default=Path(__file__).resolve().parents[1]/'clef')
    parser.add_argument('--lane', choices=('all', 'transitions', 'safety'), default='all')
    args = parser.parse_args()
    env = {k: v for k, v in os.environ.items() if not k.startswith('CLEF_')}

    def run(requests, cache=False, settings=None, rc=0):
        command = [str(args.binary.resolve()), '-m', str(args.model.resolve()), '--logits',
                   '--strict', '--no-truncate', '--time', '--batch', '1']
        if cache:
            command.append('--prefix-cache')
        result = subprocess.run(command, input=payload(requests), text=True, capture_output=True,
                                env={**env, **(settings or {})})
        if result.returncode != rc:
            raise ValueError(f'CLI returned {result.returncode}, expected {rc}: {result.stderr[-2000:]}')
        if len(rows(result.stdout)) != len(requests):
            raise ValueError('incomplete CLI responses')
        stats = [(int(a), int(b)) for a, b in re.findall(r'prefix cache reused (\d+) of (\d+)', result.stderr)]
        if cache and (len(stats) != len(requests) or any(n < 0 or n >= total or n % 32 for n, total in stats)):
            raise ValueError('missing or invalid cache statistics')
        return result.stdout, [n for n, _ in stats]

    corpus = rows(args.corpus.read_text())
    logs = sorted((r for r in corpus if isinstance(r['state'], str) and len(r['state']) > 2000),
                  key=lambda r: len(r['state']))
    if len(logs) < 3:
        raise ValueError('corpus needs at least three long log states')
    a, b, c = logs[0], logs[1], logs[-1]
    short = min(corpus, key=lambda r: len(json.dumps(r['state'])))
    other_q = [r['questions'] for r in corpus if r['questions'] != a['questions']][:2]
    if len(other_q) < 2:
        raise ValueError('corpus needs two alternative question sets')

    def cut(r, share):
        lines = r['state'].split('\n')
        return dict(r, state='\n'.join(lines[:max(1, int(len(lines)*share))]))

    if args.lane in ('all', 'transitions'):
        middle = len(b['state'])//2
        changed = dict(b, state=b['state'][:middle]+' CHANGED EVENT '+b['state'][middle:])
        steps = [
            ('fill', a, False), ('hit', a, True), ('changed questions', dict(a, questions=other_q[0]), True),
            ('replace', cut(b, .25), False), ('grow', cut(b, .6), None), ('grow full', b, True),
            ('changed questions on full', dict(b, questions=other_q[1]), True),
            ('short bypass', short, False), ('hit after bypass', b, True),
            ('changed middle', changed, True), ('changed middle hit', changed, True),
            ('longest fill', c, False), ('longest hit', dict(c, questions=other_q[0]), True),
            ('shrink', cut(b, .6), False), ('replace after shrink', a, False),
        ]
        requests = [r for _, r, _ in steps]
        expected = cold_reference(requests, run)
        actual, reused = run(requests, cache=True, settings={'CLEF_DEBUG_POISON': '1'})
        count = exact(actual, expected)
        for (label, _, must), n in zip(steps, reused, strict=True):
            if must is not None and bool(n) != must:
                raise ValueError(f'{label}: unexpected reused prefix {n}')
        print(f'PASS: {len(steps)} transitions, {count} exact logits, reused tokens {reused}', flush=True)

    if args.lane in ('all', 'safety'):
        requests = [a, a, short, a]
        expected = cold_reference(requests, run, {'CLEF_ACT_F16': '0'})
        bf16_first = rows(expected)[0]
        actual, reused = run(requests, cache=True,
                             settings={'CLEF_DEBUG_F16_LIMIT': '1e-6', 'CLEF_DEBUG_POISON': '1'})
        exact(actual, expected)
        if any(reused):
            raise ValueError('reused an overflowing discarded pass')
        print('PASS: overflow invalidates the entry and returns exact BF16 fallback', flush=True)
        actual, reused = run(requests, cache=True, settings={'CLEF_ACT_F16': '0'})
        exact(actual, expected)
        if any(reused):
            raise ValueError('unsupported activation mode reused an entry')
        print('PASS: unsupported activation mode bypasses the entry exactly', flush=True)

        requests = [a, a, c, a, a]
        expected = cold_reference(requests, run)
        if rows(expected)[0] == bf16_first:
            raise ValueError('overflow fixture does not distinguish FP16 from BF16')
        actual, reused = run(requests, cache=True, settings={'CLEF_DEBUG_PREFIX_FAIL_ABOVE': '4096'}, rc=1)
        if [i for i, r in enumerate(rows(actual)) if 'error' in r] != [2]:
            raise ValueError('allocation failure affected unexpected records')
        exact(actual, expected, errors=True)
        if reused[0] or not reused[1] or reused[3] or not reused[4]:
            raise ValueError('failed growth did not invalidate and recover the entry')
        print('PASS: failed capacity growth is explicit; the next fill and hit recover exactly', flush=True)


if __name__ == '__main__':
    main()
