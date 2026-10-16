"""Collect bounded hosted Omni evidence, binding each payload to its FP32 fixture.

Video fixtures use local decoded frame arrays and require a separate transport
adapter. This collector covers text, embedded images and lossless WAV audio.
"""
import argparse
import datetime
import hashlib
import io
import json
from pathlib import Path
import subprocess

from cloudflare_checkout import ROOT, collect
from cloudflare_corpus import (check_budget, credentials, daily_usage, final_usage,
                               hosted_payload, PRICING, HEADROOM, FREE_NEURONS)

RATE = 13636  # Official Workers AI pricing, checked 2026-10-10.


def rows(path):
    return [json.loads(line) for line in io.StringIO(path.read_text()) if line.strip()]


def plan(inventory, passes, only=None):
    requests = []
    for modality in ('text', 'images', 'audio'):
        golden = ROOT / 'golden' / inventory[modality]['golden']
        manifest = json.loads((golden / 'manifest.json').read_text())
        if not manifest.get('complete') or manifest.get('precision') != 'float32-streamed':
            raise ValueError('a complete FP32 reference is required')
        originals = rows(golden / 'requests.jsonl')
        encodings = rows(golden / 'encoded.jsonl')
        payloads = [hosted_payload(request, 'clef-omni') for request in originals]
        result = subprocess.run([ROOT / 'clef-tool', 'encode-strict', ROOT / 'gguf/clef-omni.gguf'],
                                input=''.join(json.dumps(p) + '\n' for p in payloads),
                                text=True, capture_output=True, check=True)
        native = [json.loads(line) for line in io.StringIO(result.stdout)]
        for original, payload, expected, actual in zip(originals, payloads, encodings, native, strict=True):
            for key in ('input_ids', 'questions', 'images', 'position_ids', 'audio'):
                if actual.get(key, []) != expected.get(key, []):
                    raise ValueError(f'{original["id"]}: hosted adaptation changed {key}')
            requests.append(dict(model='clef-omni', modality=modality, id=original['id'], request=payload,
                                 golden=str(golden.relative_to(ROOT)),
                                 request_sha256=hashlib.sha256(json.dumps(payload).encode()).hexdigest(),
                                 full_input_tokens=len(expected['input_ids']), questions=expected['questions'],
                                 input_ids_sha256=hashlib.sha256(json.dumps(expected['input_ids']).encode()).hexdigest()))
    if only:
        wanted = set(only)
        requests = [row for row in requests if row['id'] in wanted]
        if {row['id'] for row in requests} != wanted:
            raise ValueError('unknown hosted fixture ID')
    estimate = (requests[0]['full_input_tokens'] + passes * sum(r['full_input_tokens'] for r in requests)) * RATE / 1e6
    check_budget(0, estimate)
    return dict(type='plan', timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                passes=passes, warmups_per_model=1, client_retries=0, planned_calls=1 + passes * len(requests),
                requests=requests, pricing_source=PRICING, neurons_per_million_input_tokens=RATE,
                full_token_neuron_estimate=estimate, reserved_headroom_neurons=HEADROOM,
                free_daily_neurons=FREE_NEURONS,
                adaptation='Explicit question instructions; exact local tokens, spans and media geometry checked')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--passes', type=int, choices=(1, 2, 3, 5), default=2)
    parser.add_argument('--only', nargs='+', help='restrict to these public fixture IDs')
    parser.add_argument('--account')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.run and not args.account:
        parser.error('--run requires an exact Wrangler account name')
    run_plan = plan(json.loads(args.inventory.read_text()), args.passes, args.only)
    run_plan['mode'] = 'collect' if args.run else 'plan-only'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as out:
        out.write(json.dumps(run_plan) + '\n'); out.flush()
        if args.run:
            account, token = credentials(args.account)
            usage = daily_usage(account, token)
            check_budget(usage['used_neurons'], run_plan['full_token_neuron_estimate'])
            out.write(json.dumps(usage) + '\n'); out.flush()
            collect(run_plan, out, account, token)
            out.write(json.dumps(dict(type='complete', timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat())) + '\n')
            out.flush()
            final_usage(out, account, token)
        else:
            print(f'{run_plan["planned_calls"]} calls, at most {run_plan["full_token_neuron_estimate"]:.2f} neurons; no inference sent')


if __name__ == '__main__':
    main()
