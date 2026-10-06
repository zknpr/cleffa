"""All public requests: poisoned cache fills/hits must retain uncached logit bits.

  .venv/bin/python -B tests/test_prefix_public.py MODEL.gguf REQUESTS.jsonl

This supplements the transition/failure cases in test_prefix_cache.py. The FP32
oracle check remains separate: agreement between two engine paths is not by itself
evidence of oracle accuracy. Run under the shared GPU lock.
"""
import argparse
import os
from pathlib import Path
import re
import subprocess

from test_prefix_cache import exact, payload, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('corpus', type=Path)
    parser.add_argument('--binary', type=Path, default=Path(__file__).resolve().parents[1]/'clef')
    args = parser.parse_args()
    requests = rows(args.corpus.read_text())
    if not requests:
        raise ValueError('empty request corpus')
    env = {k: v for k, v in os.environ.items() if not k.startswith('CLEF_')}
    command = [str(args.binary.resolve()), '-m', str(args.model.resolve()), '--logits',
               '--strict', '--no-truncate', '--batch', '1', '--time']
    plain = subprocess.run(command, input=payload(requests), capture_output=True, text=True, env=env)
    if plain.returncode:
        raise ValueError('uncached CLI failed: '+plain.stderr[-2000:])
    reference = rows(plain.stdout)
    if len(reference) != len(requests):
        raise ValueError('incomplete uncached corpus')
    cached = subprocess.run(command+['--prefix-cache'],
        input=payload([r for r in requests for _ in range(2)]), capture_output=True, text=True,
        env={**env, 'CLEF_DEBUG_POISON': '1'})
    if cached.returncode:
        raise ValueError('cached CLI failed: '+cached.stderr[-2000:])
    count = exact(cached.stdout, payload([r for r in reference for _ in range(2)]))
    reused = [int(n) for n in re.findall(r'prefix cache reused (\d+) of', cached.stderr)]
    if len(reused) != 2*len(requests) or not any(reused) or any(n % 32 for n in reused):
        raise ValueError('missing or invalid reuse coverage')
    for i, request in enumerate(requests):
        if isinstance(request['state'], str) and len(request['state']) > 2000 and not reused[2*i+1]:
            raise ValueError(f'long request {i} did not hit its freshly populated entry')
    print(f'PASS: {len(requests)} public requests, {count} exact fill/hit logits with poisoned buffers')


if __name__ == '__main__':
    main()
