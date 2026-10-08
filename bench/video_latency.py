"""Alternate native clip-to-response measurements on warm engines.

Includes clip snapshot, ffprobe/FFmpeg, lossless frame encoding, JSON transfer, host
processing and inference. Excludes model loading. Run with no competing GPU jobs.
--baseline-engine and --baseline-adapter must be saved before changing their sources.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import selectors
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--baseline-engine', type=Path, required=True)
    parser.add_argument('--baseline-adapter', type=Path, required=True)
    parser.add_argument('--clips', type=Path, nargs='+', required=True)
    parser.add_argument('--models', nargs='+', default=['gguf/clef-flash.gguf', 'gguf/clef.gguf'])
    parser.add_argument('--samples', type=int, default=5)
    parser.add_argument('--rounds', type=int, default=2)
    args = parser.parse_args()
    if min(args.samples, args.rounds) < 1:
        parser.error('samples and rounds must be positive')
    binaries = {'baseline': args.baseline_engine.resolve(), 'candidate': ROOT / 'clef'}
    adapters = {'baseline': args.baseline_adapter.resolve(), 'candidate': ROOT / 'tools/video_request.py'}
    procs = subprocess.check_output(['ps', '-axo', 'state=,comm='], text=True)
    names = {'clef', 'clef-server', *(p.name for p in binaries.values())}
    if any('T' not in state and Path(cmd).name in names
           for state, cmd in (line.strip().split(None, 1) for line in procs.splitlines())):
        parser.error('another engine is running; pause it before timing')
    converters = {arm: module(path, 'adapter_' + arm).convert for arm, path in adapters.items()}
    template = {'model': 'clef', 'state': 'Review the video in time order.',
                'questions': {'moving': {'type': 'noul', 'instructions': 'Does anything move?'}},
                'media_kwargs': {'videos_kwargs': {'size': {'shortest_edge': 4096, 'longest_edge': 2097152}}}}
    env = {k: v for k, v in os.environ.items() if not k.startswith('CLEF_')}
    report = {'metric': 'warm clip-to-response wall ms, including conversion and host encoding',
              'binaries': {k: {'path': str(v), 'sha256': digest(v)} for k, v in binaries.items()},
              'adapters': {k: {'path': str(v), 'sha256': digest(v)} for k, v in adapters.items()},
              'template': template, 'runs': [], 'summary': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for model in args.models:
        for clip in args.clips:
            expected = None
            processes, logs, selectors_by_arm = {}, {}, {}
            try:
                # Keep both models warm and alternate individual requests, reversing order.
                # This controls drift that otherwise spans a whole arm's process lifetime.
                for arm in binaries:
                    log = args.output.with_name(f'{Path(model).stem}-{clip.stem}-{arm}.log')
                    logs[arm] = log.open('w')
                    processes[arm] = subprocess.Popen(
                        [str(binaries[arm]), '-m', model, '--batch', '1', '--logits'],
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=logs[arm],
                        text=True, env=env, cwd=ROOT)
                    selector = selectors.DefaultSelector()
                    selectors_by_arm[arm] = selector
                    selector.register(processes[arm].stdout, selectors.EVENT_READ)

                def request(arm):
                    nonlocal expected
                    start = time.perf_counter()
                    payload = converters[arm](clip, template)
                    converted = time.perf_counter()
                    process = processes[arm]
                    process.stdin.write(payload); process.stdin.flush()
                    if not selectors_by_arm[arm].select(180):
                        raise RuntimeError(f'{arm} engine timeout; see {logs[arm].name}')
                    response = process.stdout.readline()
                    end = time.perf_counter()
                    if not response or 'error' in json.loads(response):
                        raise RuntimeError(f'engine error: {response}; see {logs[arm].name}')
                    if expected is None:
                        expected = response
                    if response != expected:
                        raise RuntimeError('clip arms differ in raw logits')
                    return {'convert_ms': (converted - start) * 1000,
                            'encode_infer_ms': (end - converted) * 1000,
                            'total_ms': (end - start) * 1000,
                            'request_bytes': len(payload.encode())}

                for _ in range(2):
                    for arm in binaries:
                        request(arm)
                for round_no in range(args.rounds):
                    samples = {arm: [] for arm in binaries}
                    for rep in range(args.samples):
                        order = ['baseline', 'candidate'] if (rep + round_no) % 2 == 0 else ['candidate', 'baseline']
                        for arm in order:
                            samples[arm].append(request(arm))
                    for arm in binaries:
                        report['runs'].append({'model': model, 'clip': str(clip), 'clip_sha256': digest(clip),
                                               'arm': arm, 'round': round_no, 'samples': samples[arm]})
                        print(model, clip.name, arm, round_no,
                              statistics.median(s['total_ms'] for s in samples[arm]), flush=True)
                    args.output.write_text(json.dumps(report, indent=2) + '\n')
            finally:
                for process in processes.values():
                    try:
                        process.stdin.close()
                    except BrokenPipeError:
                        pass  # The response/exit checks report the failed engine; still reap both.
                for process in processes.values():
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill(); process.wait()
                    process.stdout.close()
                for selector in selectors_by_arm.values():
                    selector.close()
                for log in logs.values():
                    log.close()
            if any(p.returncode for p in processes.values()):
                raise RuntimeError('engine exited unsuccessfully; see arm logs')
            medians = {arm: {key: statistics.median(s[key] for r in report['runs']
                                                  if r['model'] == model and r['clip'] == str(clip) and r['arm'] == arm
                                                  for s in r['samples'])
                              for key in ['convert_ms', 'encode_infer_ms', 'total_ms', 'request_bytes']}
                       for arm in binaries}
            report['summary'].append({'model': model, 'clip': str(clip), 'medians': medians,
                                      'speedup': medians['baseline']['total_ms'] / medians['candidate']['total_ms'],
                                      'raw_logits_identical': True})
            args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
