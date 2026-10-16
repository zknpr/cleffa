"""Compare hosted/native rounded answers with audited FP32 Omni fixtures.

This measures numerical fidelity, not labeled semantic accuracy. A hosted token
count match does not expose or prove the hosted token sequence or model weights.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

import numpy as np
from safetensors.numpy import load

from cloudflare_checkout import ROOT
from cloudflare_corpus import hosted_payload
from compare_cloudflare import check_input_ids, decision, distribution, hosted_snapshot, metrics
from cloudflare_omni import rows

REVISION = '0db1cd2607d76a7bdb2a382f659e7b313079f84b'


def video_encoder():
    """Verify the pinned CPU encoder inputs without reading unused model weights."""
    sys.path[:0] = [str(ROOT / 'ref'), str(ROOT / 'tools')]
    from omni import REVISION
    from omni_staged import inventory
    from verify_snapshot import file_hash, load_manifest
    snapshot = ROOT / 'model-omni'
    manifest = load_manifest(REVISION)
    if set(inventory(snapshot)) != set(manifest['files']):
        raise ValueError('unexpected pinned encoder inventory')
    for name, expected in manifest['files'].items():
        if not name.endswith('.safetensors') and file_hash(snapshot / name, expected['hash']) != expected['hash']:
            raise ValueError('pinned encoder input changed')
    sys.path.insert(0, str(snapshot))
    from joint_schema_model import encode_record
    from transformers import AutoProcessor
    return encode_record, AutoProcessor.from_pretrained(snapshot)


def verify_video_transport(row, original, encoding, encoder):
    from cloudflare_omni_video import tensor_hashes
    from omni import REVISION
    from omni_media import encode_reference
    proof = row.get('transport', {})
    if proof.get('revision') != REVISION or proof.get('native_request_sha256') != hashlib.sha256(json.dumps(original).encode()).hexdigest():
        raise ValueError('video transport does not bind the source request')
    encode, processor = encoder
    before = encode_reference(encode, original, processor)
    after = encode_reference(encode, row['request'], processor)
    if list(before.input_ids) != encoding['input_ids'] or before.input_ids != after.input_ids or before.questions != after.questions:
        raise ValueError('video transport changed tokens or question spans')
    if tensor_hashes(before.media) != tensor_hashes(after.media) or tensor_hashes(before.media) != proof['media_tensors']:
        raise ValueError('video transport changed processor tensors')


def compare(journal, inventory, model=ROOT / 'gguf/clef-omni-full.gguf'):
    plan, calls, journal_hash = hosted_snapshot(journal)
    with Path(model).open('rb') as source:
        model_hash = hashlib.file_digest(source, 'sha256').hexdigest()
    result = dict(complete=False, hosted_sha256=journal_hash, comparisons=[], excluded=[], latency=[],
                  note=__doc__.strip(), source_hashes={}, model=str(Path(model).resolve()),
                  model_sha256=model_hash, reference_revision=REVISION)
    binary_hash = hashlib.sha256((ROOT / 'clef').read_bytes()).hexdigest()
    result['native_binary_sha256'] = binary_hash
    encoder = None
    for row in plan['requests']:
        if row['model'] != 'clef-omni':
            raise ValueError('expected an Omni-only hosted plan')
        name, modality = row['id'], row['modality']
        golden = ROOT / 'golden' / inventory[modality]['golden']
        captures = Path(inventory[modality]['captures'])
        manifest = json.loads((golden / 'manifest.json').read_text())
        paired = json.loads((captures / 'manifest.json').read_text())
        if not manifest.get('complete') or manifest.get('precision') != 'float32-streamed' or not paired.get('complete'):
            raise ValueError('incomplete native/reference evidence')
        if manifest.get('revision') != REVISION:
            raise ValueError('reference revision differs from the pinned snapshot')
        if paired['binary_sha256'] != binary_hash:
            raise ValueError('native response capture uses another CLI executable')
        if paired.get('model_sha256') != model_hash:
            raise ValueError('native response capture uses different model weights')
        corpus_hash = hashlib.sha256((golden / 'requests.jsonl').read_bytes()).hexdigest()
        if paired['requests_sha256'] != corpus_hash or manifest['corpus_sha256'] != corpus_hash:
            raise ValueError('native/reference request identity mismatch')
        originals = {r['id']: r for r in rows(golden / 'requests.jsonl')}
        enc = {r['id']: r for r in rows(golden / 'encoded.jsonl')}[name]
        original = hosted_payload(originals[name], 'clef-omni')
        if original != row['request']:
            if not row.get('transport'):
                raise ValueError('hosted request differs from the numerical fixture')
            if encoder is None:
                encoder = video_encoder()
            verify_video_transport(row, original, enc, encoder)
        check_input_ids(row, enc, False)
        if row['questions'] != enc['questions'] or row['full_input_tokens'] != len(enc['input_ids']):
            raise ValueError('hosted plan encoding differs from numerical fixture')
        samples = [calls['clef-omni', name, rep] for rep in range(plan['passes'])]
        if any(sample['full_input_reported'] is not True for sample in samples):
            result['excluded'].append(dict(id=name, reason='different or missing hosted token count',
                                           local_tokens=row['full_input_tokens'],
                                           hosted_tokens=[s['reported_input_tokens'] for s in samples]))
            continue
        local_bytes = (captures / (name + '.output.json')).read_bytes()
        local = json.loads(local_bytes)
        if local['usage']['input_tokens'] != row['full_input_tokens'] or set(local['answers']) != set(row['request']['questions']):
            raise ValueError('native response questions or tokens differ')
        result['source_hashes'][name] = hashlib.sha256(local_bytes).hexdigest()
        raw = (golden / 'logits.safetensors').read_bytes()
        result['source_hashes'][str(golden / 'logits.safetensors')] = hashlib.sha256(raw).hexdigest()
        logits = load(raw)
        milliseconds = [sample['ms'] for sample in samples]
        result['latency'].append(dict(id=name, modality=modality, tokens=row['full_input_tokens'],
                                     samples_ms=milliseconds, median_ms=statistics.median(milliseconds)))
        for question in row['questions']:
            qid, options = question['id'], question['option_ids']
            kind = row['request']['questions'][qid]['type']
            values = logits[name + '/' + qid].astype('f8')
            if values.shape != (len(options),) or not np.isfinite(values).all():
                raise ValueError('invalid FP32 logits')
            probabilities = np.exp(values - values.max()); probabilities /= probabilities.sum()
            reference = dict(zip(options, map(float, probabilities), strict=True))
            native = distribution(local['answers'][qid], options, kind)
            hosted = [distribution(s['answer']['answers'][qid], options, kind) for s in samples]
            result['comparisons'].append(dict(id=name, modality=modality, question=qid,
                reference_argmax=max(reference, key=reference.get), native_argmax=decision(local['answers'][qid], native),
                hosted_argmax=[decision(s['answer']['answers'][qid], p) for s, p in zip(samples, hosted, strict=True)],
                native_max_probability_error=max(abs(native[k] - reference[k]) for k in options),
                hosted_max_probability_error=[max(abs(p[k] - reference[k]) for k in options) for p in hosted],
                native_probabilities=native, reference_probabilities=reference, hosted_probabilities=hosted))
    compared = result['comparisons']
    if not compared:
        raise ValueError('no matching-token questions to compare')
    result.update(complete=True, records=len({r['id'] for r in compared}), decisions=len(compared),
                  native_argmax_mismatches=sum(r['reference_argmax'] != r['native_argmax'] for r in compared),
                  hosted_argmax_mismatches=sum(r['reference_argmax'] != v for r in compared for v in r['hosted_argmax']),
                  native_max_probability_error=max(r['native_max_probability_error'] for r in compared),
                  hosted_max_probability_error=max(max(r['hosted_max_probability_error']) for r in compared))
    # Keep hosted agreement separate from FP32 attribution. Repeated hosted
    # passes are paired observations of the same questions, never extra trials.
    direct=[]
    for row in compared:
        item=dict(id=row['id'],question=row['question'],modality=row['modality'],
                  C=row['native_probabilities'],C_decision=row['native_argmax'],
                  F=row['reference_probabilities'],F_decision=row['reference_argmax'])
        for index,(probs,choice) in enumerate(zip(row['hosted_probabilities'],row['hosted_argmax'],strict=True)):
            arm=f'H{index+1}';item[arm]=probs;item[arm+'_decision']=choice
        direct.append(item)
    pairs=[('C',f'H{i+1}') for i in range(plan['passes'])]+[('C','F')]
    if plan['passes']>=2:pairs.append(('H1','H2'))
    result['primary_metric']='C vs H1 decision agreement; FP32 distances are engineering attribution'
    result['direct_agreement']={a+'_vs_'+b:metrics(direct,a,b) for a,b in pairs}
    result['direct_by_modality']={modality:{a+'_vs_'+b:metrics([r for r in direct if r['modality']==modality],a,b)
        for a,b in pairs} for modality in dict.fromkeys(r['modality'] for r in direct)}
    result['native_arm']='historical CLI capture; not a fresh strict-server capture'
    result['hosted_capture_timestamp']=plan.get('timestamp')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('journal', type=Path)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=ROOT / 'gguf/clef-omni-full.gguf')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('refusing to replace comparison evidence')
    report = compare(args.journal, json.loads(args.inventory.read_text()), args.model)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f'{report["records"]} records, {report["decisions"]} questions, '
          f'{len(report["excluded"])} count-mismatch exclusions')


if __name__ == '__main__':
    main()
