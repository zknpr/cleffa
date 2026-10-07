"""Measure exact image reuse while alternating question schemas on the same image.

Warm CLI inference time, excluding decoding, encoding and model loading.
Run one GPU workload at a time. Fills are recorded separately from warm hits.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--models', nargs='+', default=['gguf/clef-flash.gguf', 'gguf/clef.gguf'])
    parser.add_argument('--requests', type=Path, default=ROOT / 'golden/clef-flash-vision-f32/requests.jsonl')
    parser.add_argument('--samples', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=4)
    parser.add_argument('--rounds', type=int, default=2)
    args = parser.parse_args()
    if min(args.samples, args.warmup, args.rounds) < 1:
        parser.error('sample, warmup and round counts must be positive')
    procs = subprocess.run(['ps', '-axo', 'state=,comm='], capture_output=True, text=True, check=True)
    if any('T' not in state and Path(cmd).name in ('clef', 'clef-server')
           for state, cmd in (line.strip().split(None, 1) for line in procs.stdout.splitlines())):
        parser.error('another engine is running; measure with one GPU workload at a time')
    corpus = {r['id']: r for r in map(json.loads, args.requests.read_text().splitlines())}
    env = {k: v for k, v in os.environ.items() if not k.startswith('CLEF_')}
    report = {'metric': 'warm CLI inference ms; excludes decode, encoding, loading and response serialization',
              'binary_sha256': hashlib.sha256((ROOT / 'clef').read_bytes()).hexdigest(), 'runs': [], 'summary': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for model in args.models:
        for rid in ('v001', 'v009'):
            original = corpus[rid]
            alternate = dict(original, questions=corpus['v000']['questions'])
            sequence = [original if i % 2 == 0 else alternate for i in range(args.samples + args.warmup)]
            payload = ''.join(json.dumps(r) + '\n' for r in sequence)
            expected = None
            for rep in range(args.rounds):
                for cached in ([False, True] if rep % 2 == 0 else [True, False]):
                    cmd = [str(ROOT / 'clef'), '-m', model, '--time', '--batch', '1', '--logits']
                    if cached:
                        cmd += ['--prefix-cache']
                    result = subprocess.run(cmd, input=payload, capture_output=True, text=True, env=env, check=True, timeout=600)
                    outputs = result.stdout.splitlines()
                    stats = [(int(n), float(t)) for n, t in re.findall(r'batch of 1 \((\d+) tokens\) in ([\d.]+) ms', result.stderr)]
                    if len(outputs) != len(sequence) or len(stats) != len(sequence) or any('error' in json.loads(r) for r in outputs):
                        raise RuntimeError(f'incomplete/error response: {result.stderr}')
                    if expected is None:
                        expected = outputs
                    if outputs != expected:
                        raise RuntimeError('cached logits differ from the same uncached request sequence')
                    reused = [int(n) for n in re.findall(r'prefix cache reused (\d+) of', result.stderr)]
                    if cached and (len(reused) != len(sequence) or reused[0] or not all(reused[1:])):
                        raise RuntimeError(f'expected one fill followed by hits: {reused}')
                    entry = {'model': model, 'id': rid, 'arm': 'cached' if cached else 'plain', 'round': rep,
                             'request_sha256': hashlib.sha256(payload.encode()).hexdigest(),
                             'fill_or_first_ms': stats[0][1], 'warmup_ms': [t for _, t in stats[:args.warmup]],
                             'ms': [t for _, t in stats[args.warmup:]], 'tokens': [n for n, _ in stats],
                             'reused_tokens': reused, 'stderr': result.stderr}
                    report['runs'].append(entry)
                    args.output.write_text(json.dumps(report, indent=2) + '\n')
                    print(f'{model} {rid} {entry["arm"]} round {rep}: {statistics.median(entry["ms"]):.1f} ms', flush=True)
            med = {arm: statistics.median(t for r in report['runs'] if r['model'] == model and r['id'] == rid
                                         and r['arm'] == arm for t in r['ms']) for arm in ('plain', 'cached')}
            report['summary'].append({'model': model, 'id': rid, 'median_ms': med,
                                      'speedup': med['plain'] / med['cached'], 'logits_byte_identical': True})
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
