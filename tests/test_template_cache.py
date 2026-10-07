"""Exact template reuse across unrelated inputs, capacity boundaries and HTTP fallback.

Run under the shared GPU lock:
  .venv/bin/python -B tests/test_template_cache.py MODEL.gguf [--baseline /path/to/clef]

This uses public synthetic requests only. No oracle or hosted calls are needed:
the invariant is the exact ordinary-path result, with all input tokens retained.
"""
from pathlib import Path
import argparse
import contextlib
import http.client
import json
import os
import re
import socket
import subprocess
import tempfile
import time

from gguf import GGUFReader

from test_prefix_cache import exact, rows

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('model', type=Path)
parser.add_argument('--baseline', type=Path, default=ROOT / 'clef')
args = parser.parse_args()
env = {k: v for k, v in os.environ.items() if not k.startswith('CLEF_')}
base = {'model': 'clef', 'state': 'Checkout is healthy.', 'questions': {
    'urgency': {'type': 'choice', 'instructions': 'How urgent is the situation?',
                'criteria': {'urgent': 'Immediate action needed', 'normal': 'Can wait', 'unknown': 'Insufficient evidence'}},
    'outage': {'type': 'noul', 'instructions': 'Is checkout unavailable?'}
}}


def payload(requests):
    return ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in requests)


def encode(requests):
    p = subprocess.run([str(ROOT / 'clef-tool'), 'encode-notrunc', str(args.model)],
                       input=payload(requests), text=True, capture_output=True, check=True, env=env)
    if any(s.startswith('ERR ') for s in p.stdout.splitlines()):
        raise ValueError('fixture rejected: ' + p.stdout[:1000])
    out = [json.loads(s) for s in p.stdout.splitlines()]
    if len(out) != len(requests):
        raise ValueError('incomplete encoded fixtures')
    return out


def sized(target, label):
    n = target - 200
    for _ in range(5):
        if n < 0:
            raise ValueError('fixture padding became negative')
        req = dict(base, state=label + ' x' * n)
        have = len(encode([req])[0]['input_ids'])
        if have == target:
            return req
        n += target - have
    raise ValueError('could not make exact token length ' + str(target))


def run(requests, flags=(), settings=None, baseline=False):
    cli = args.baseline if baseline else ROOT / 'clef'
    p = subprocess.run([str(cli), '-m', str(args.model), '--strict', '--no-truncate',
                        '--logits', '--time', *flags], input=payload(requests),
                       text=True, capture_output=True, env={**env, **(settings or {})})
    if p.returncode:
        raise ValueError('CLI failed: ' + p.stderr[-2000:])
    if len(rows(p.stdout)) != len(requests):
        raise ValueError('incomplete CLI results')
    reuse = [(int(a), int(b)) for a, b in re.findall(r'prefix cache reused (\d+) of (\d+)', p.stderr)]
    return p.stdout, reuse


# Explicit outcomes on both sides of GEMM tile and padding boundaries. A bypass
# between hits must leave the existing template reusable, including after 2K.
# Columns: full tokens, Flash reuse, 27B reuse (zero on the initial fill).
cases = (
    (346, 0, 0), (594, 32, 32), (767, 32, 32), (768, 0, 32),
    (769, 32, 32), (800, 32, 32), (801, 0, 32), (992, 32, 32),
    (993, 0, 32), (1024, 0, 32), (1025, 32, 32), (1056, 32, 32),
    (1057, 0, 0), (1088, 0, 0), (1089, 32, 32), (1120, 32, 32),
    (1121, 0, 0), (2016, 32, 32), (2048, 0, 0), (2049, 0, 0),
    (346, 32, 32),
)
hidden = GGUFReader(str(args.model)).fields['clef.embedding_length'].contents()
if hidden not in (4096, 5120):
    raise ValueError('template expectations require a qualified Clef model')
requests = [sized(n, 'Checkout is unavailable.' if i % 2 else 'Checkout is healthy.')
            for i, (n, _, _) in enumerate(cases)]
requests += [dict(requests[1], questions=dict(reversed(list(base['questions'].items()))))]
encoded = encode(requests)
lengths = [len(r['input_ids']) for r in encoded]
if lengths != [c[0] for c in cases] + [594]:
    raise ValueError('fixture token counts changed')
if len({tuple(r['input_ids'][:36]) for r in encoded}) != 1:
    raise ValueError('fixtures do not share the fixed template')
want, _ = run(requests, baseline=True)
got, reused = run(requests, ['--template-cache'], {'CLEF_DEBUG_POISON': '1'})
count = exact(got, want)
expected = [(c[1 if hidden == 4096 else 2], c[0]) for c in cases] + [(32, 594)]
if reused != expected:
    raise ValueError(f'unexpected template reuse: {reused}; expected {expected}')
print(f'PASS: {len(requests)} unrelated/boundary requests, {count} exact poisoned logits; reuse {reused}', flush=True)

# Force the first cached pass to overflow. No invalid entry may be reused, and
# the result must equal the independent ordinary whole-record BF16 fallback.
small = requests[:2] + [requests[0]]
fallback, _ = run(small, settings={'CLEF_DEBUG_F16_LIMIT': '1e-6'}, baseline=True)
got, reused = run(small, ['--template-cache'], {'CLEF_DEBUG_F16_LIMIT': '1e-6', 'CLEF_DEBUG_POISON': '1'})
exact(got, fallback)
if len(reused) != len(small) or any(n for n, _ in reused) or rows(fallback) == [rows(want)[i] for i in (0, 1, 0)]:
    raise ValueError('overflow test did not discard a changed pass')
print('PASS: template overflow invalidates reuse and preserves the ordinary BF16 fallback', flush=True)


@contextlib.contextmanager
def server(tmp, name, flags=(), settings=None):
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]
    path = tmp / (name + '.log')
    with path.open('w') as log:
        proc = subprocess.Popen([str(ROOT / 'clef-server'), '-m', str(args.model), '--port', str(port),
                                 '--no-warmup', *flags], cwd=ROOT, stdout=log, stderr=log,
                                env={**env, **(settings or {})})
        try:
            deadline = time.monotonic() + 240
            while True:
                try:
                    c = http.client.HTTPConnection('127.0.0.1', port, timeout=2)
                    try:
                        c.request('GET', '/health')
                        r = c.getresponse()
                        ready = r.status == 200
                        r.read()
                    finally:
                        c.close()
                    if ready:
                        break
                except OSError:
                    pass
                if proc.poll() is not None or time.monotonic() > deadline:
                    raise ValueError('server startup failed: ' + path.read_text()[-2000:])
                time.sleep(0.1)
            yield port, path
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=20)


def post(port, req, key=None):
    c = http.client.HTTPConnection('127.0.0.1', port, timeout=120)
    headers = {'Content-Type': 'application/json'}
    if key is not None:
        headers['X-Clef-Prefix-Cache'] = key
    try:
        c.request('POST', '/v1/systemone', json.dumps(req).encode(), headers)
        r = c.getresponse()
        body = r.read()
        if r.status != 200:
            raise ValueError(f'HTTP {r.status}: {body[:200]}')
        return body
    finally:
        c.close()


with tempfile.TemporaryDirectory() as t:
    tmp = Path(t)
    # 1,025 remains eligible and grows the entry beyond the injected 1,024 cap;
    # 2,049 bypasses it. Testing only an ineligible large request would miss OOM.
    selected = [requests[i] for i in (0, 1, 10, 19, 20)]
    with server(tmp, 'plain') as (port, _):
        bodies = [post(port, req) for req in selected]
    with server(tmp, 'template', ['--template-cache', '--prefix-cache-mb', '1024'],
                {'CLEF_DEBUG_POISON': '1'}) as (port, log):
        for req, want_body in zip(selected, bodies, strict=True):
            if post(port, req) != want_body:
                raise ValueError('template HTTP response changed')
        # Keyed requests keep independent state while unkeyed template calls occur.
        for key in ('tenant-a', 'tenant-a', 'tenant-b'):
            if post(port, selected[1], key) != bodies[1] or post(port, selected[0]) != bodies[0]:
                raise ValueError('keyed/template interleave changed HTTP response')
        text = log.read_text()
        keyed = [int(x) for x in re.findall(r'prefix cache: reused (\d+) of', text)]
        if len(keyed) != 3 or keyed[0] or keyed[1] <= 32 or keyed[2]:
            raise ValueError('template call affected keyed isolation: ' + str(keyed))
        if 'template cache:' in text or 'keep-warm pass failed' in text:
            raise ValueError('normal template server silently used fallback: ' + text[-1500:])
    with server(tmp, 'allocation', ['--template-cache'],
                {'CLEF_DEBUG_PREFIX_FAIL_ABOVE': '1024'}) as (port, log):
        for i in (0, 2, 0, 0):
            if post(port, selected[i]) != bodies[i]:
                raise ValueError('allocation fallback/recovery changed HTTP response')
        text = log.read_text()
        failures = re.findall(r'^clef-server: template cache: (.+)$', text, re.MULTILINE)
        if len(failures) != 1 or 'cannot allocate the prefix cache entry' not in failures[0] or not failures[0].endswith('; serving uncached'):
            raise ValueError('allocation fallback was not explicit, or recovery failed: ' + text[-1500:])
    print('PASS: HTTP byte parity, keyed/template isolation, explicit allocation fallback and recovery', flush=True)
