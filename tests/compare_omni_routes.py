"""Report expert-choice differences diagnostically; exact routing is not a numerical gate."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from safetensors.numpy import load_file
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]


def compare(rows, ref, first):
    layers = sorted(k for k in ref if k.startswith('routing.') and k.endswith('.ids'))
    if not layers:raise ValueError('empty routing reference')
    count = ref[layers[0]].shape[0]
    if count<1:raise ValueError('empty routing reference rows')
    rows = iter(rows)
    mismatch = set_mismatch = total = 0
    first_bad = None
    max_error = max_weight_error = 0.0
    for layer, key in enumerate(layers):
        prefix = key[:-3]
        logits = ref[prefix+'logits']
        weights = ref[prefix+'weights']
        ids = ref[key]
        margin = ref[prefix+'margin']
        if (logits.shape!=(count,128) or weights.shape!=(count,8) or ids.shape!=(count,8) or
                margin.shape!=(count,) or ids.dtype.kind not in 'iu' or
                np.any(ids<0) or np.any(ids>=128) or
                not all(np.isfinite(x).all() for x in (logits,weights,margin)) or
                np.any(weights<0) or np.any(abs(weights.sum(-1)-1)>1e-5) or
                any(len(set(row))!=8 for row in ids.tolist())):
            raise ValueError('invalid reference routing tensors')
        for t in range(count):
            row = next(rows, None)
            if row is None:raise ValueError('incomplete captured route rows')
            if (row['layer'], row['token']) != (layer, first+t):
                raise ValueError('incorrect captured row geometry')
            scores = np.array(row['logits'], dtype=np.float64)
            if scores.shape != (128,) or not np.isfinite(scores).all():
                raise ValueError('invalid captured router logits')
            native_ids=row['ids'];native_weights=np.asarray(row['weights'],dtype=np.float64)
            if (len(native_ids)!=8 or any(type(i) is not int or not 0<=i<128 for i in native_ids) or
                    len(set(native_ids))!=8 or native_weights.shape!=(8,) or
                    not np.isfinite(native_weights).all() or np.any(native_weights<0) or
                    abs(native_weights.sum()-1)>1e-5 or not np.isfinite(row['margin'])):
                raise ValueError('invalid captured expert IDs or weights')
            max_error = max(max_error, float(np.max(np.abs(scores-logits[t]))))
            # Match weights by expert ID so ordering-only differences remain visible
            # in the ID count without pretending they are a probability error.
            got = dict(zip(row['ids'], row['weights']))
            for expert, weight in zip(ids[t], weights[t]):
                max_weight_error = max(max_weight_error, abs(got.get(int(expert), 0)-float(weight)))
            if row['ids'] != ids[t].tolist():
                mismatch += 1
                set_mismatch += set(row['ids']) != set(ids[t].tolist())
                if first_bad is None:
                    first_bad = dict(layer=layer, token=first+t, native=row['ids'], reference=ids[t].tolist(),
                                     reference_margin=float(margin[t]), native_margin=row['margin'],
                                     row_logit_error=float(np.max(np.abs(scores-logits[t]))))
            total += 1
    if next(rows, None) is not None:raise ValueError('unexpected extra route rows')
    return dict(rows=total, mismatches=mismatch, mismatch_fraction=mismatch/total,
                selected_set_mismatches=set_mismatch, selected_set_mismatch_fraction=set_mismatch/total,
                diagnostic_only=True, max_logit_error=max_error,
                max_weight_error=max_weight_error, first_mismatch=first_bad)


def capture_rows(path, tokens, first, layers):
    """Stream a full native trace; retain only the suffix captured by the reference."""
    seen = 0
    with path.open() as source:
        for line in source:
            row = json.loads(line)
            if (row['layer'], row['token']) != (seen//tokens, seen%tokens):
                raise ValueError('full native trace has missing or reordered rows')
            seen += 1
            if row['token'] >= first:yield row
    if seen != tokens*layers:raise ValueError('incomplete full native trace')


def diagnose_corpus(golden, captures):
    manifest = json.loads((golden/'manifest.json').read_text())
    paired = json.loads((captures/'manifest.json').read_text())
    if not manifest.get('complete') or not paired.get('complete'):
        raise ValueError('routing diagnostics require complete captures')
    if paired['requests_sha256'] != manifest['corpus_sha256']:
        raise ValueError('routing capture corpus differs')
    records = []
    for record in manifest['records']:
        name, first = record['id'], record['dump_start']
        with safe_open(golden/'layers'/f'{name}.safetensors', framework='np') as source:
            # Exclude residual/tower tensors; full long-context dumps are much larger
            # than the selected router suffix. Native JSON rows stream one at a time.
            ref = {k:source.get_tensor(k) for k in source.keys() if k.startswith('routing.')}
        if len([k for k in ref if k.endswith('.ids')]) != 48:
            raise ValueError('incomplete routing reference')
        result = compare(capture_rows(captures/f'{name}.jsonl', record['tokens'], first, 48), ref, first)
        records.append(dict(id=name, first_token=first, **result))
    total = sum(r['rows'] for r in records)
    mismatches = sum(r['mismatches'] for r in records)
    return dict(complete=True, diagnostic_only=True, rows=total, mismatches=mismatches,
                mismatch_fraction=mismatches/total, records=records)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('model', type=Path)
    p.add_argument('golden', type=Path)
    p.add_argument('--record', default='r000')
    p.add_argument('--output', type=Path)
    p.add_argument('--captures', type=Path, help='diagnose every saved paired trace without GPU inference')
    p.add_argument('--require-exact', action='store_true', help='historical diagnostic: fail on any route difference')
    args = p.parse_args()
    if args.captures:
        if not args.output:p.error('--captures requires --output for the diagnostic report')
        result = diagnose_corpus(args.golden, args.captures)
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        print(f"Routing diagnostic: {result['mismatches']}/{result['rows']} rows differ")
        raise SystemExit(bool(args.require_exact and result['mismatches']))
    requests = [json.loads(s) for s in (args.golden/'requests.jsonl').read_text().splitlines()]
    req = next(r for r in requests if r['id'] == args.record)
    meta = json.loads((args.golden/'manifest.json').read_text())
    entry = next(r for r in meta['records'] if r['id'] == args.record)
    ref = load_file(str(args.golden/'layers'/f'{args.record}.safetensors'))
    with tempfile.TemporaryDirectory() as tmp:
        path = args.output or Path(tmp)/'routes.jsonl'
        cmd = [str(ROOT/'clef'), '-m', str(args.model), '--dump-routes', str(path)]
        if entry['dump_start']:
            cmd += ['--dump-last', str(entry['tokens']-entry['dump_start'])]
        run = subprocess.run(cmd, input=json.dumps(req)+'\n', text=True, capture_output=True, check=True)
        if 'error' in json.loads(run.stdout):
            raise RuntimeError(run.stdout)
        rows = [json.loads(s) for s in path.read_text().splitlines()]
        result = compare(rows, ref, entry['dump_start'])
    print(json.dumps(result, indent=2))
    raise SystemExit(bool(args.require_exact and result['mismatches']))


if __name__ == '__main__':
    main()
