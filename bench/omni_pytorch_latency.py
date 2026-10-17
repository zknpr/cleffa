"""BF16 measurements: explicit stock model-card and resident/eager profiles.

--stock uses the snapshot's loader and Transformers defaults, without allocator,
expert or attention overrides. The historical resident/eager profile remains
available and is labeled separately; it is not a stock PyTorch measurement.
"""
import argparse
import base64
from contextlib import ExitStack, contextmanager
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time

import torch
import transformers

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'ref'), str(ROOT / 'tools')]
from omni import REVISION
from omni_media import encode_reference
from omni_weights import load_paged
from verify_snapshot import load_manifest, verify_snapshot


@contextmanager
def stock_request(request, model):
    """Translate wire envelopes into the snapshot's supported input types only.

    Dense Transformers accepts encoded video paths; Omni's loader accepts bytes.
    The actual decoding, sampling and model execution remain stock.
    """
    with tempfile.TemporaryDirectory(prefix='clef-stock-media-') as folder:
        if model == 'clef-omni':
            result = dict(request)
            for key in ('images', 'audio', 'videos'):
                if key in request:
                    result[key] = [base64.b64decode(item['base64'], validate=True)
                                   if isinstance(item, dict) and 'base64' in item else item
                                   for item in request[key]]
        else:
            from oracle import with_pil_images
            videos = request.get('videos') or []
            encoded = [not (isinstance(item, dict) and 'frames' in item) for item in videos]
            if any(encoded):
                if not all(encoded):
                    raise ValueError('stock wire adapter cannot mix frames and encoded clips')
                result = with_pil_images({k: v for k, v in request.items() if k != 'videos'})
                result['videos'] = []
                for index, item in enumerate(videos):
                    raw = base64.b64decode(item['base64'] if isinstance(item, dict)
                                          else item.split(',', 1)[1], validate=True)
                    path = Path(folder) / f'{index}.mov'
                    path.write_bytes(raw)
                    result['videos'].append(str(path))
            else:
                result = with_pil_images(request)
        yield result


def stock(args):
    """Collect stock control answers and cold/three-warm latency from a frozen plan."""
    if args.plan is None:
        raise ValueError('--stock requires --plan')
    tuned = [key for key in os.environ if key.startswith('PYTORCH_MPS_') or
             key in ('PYTORCH_ENABLE_MPS_FALLBACK', 'PYTORCH_ENABLE_MPS_FALLBACK_RECURSIVE')]
    if tuned:
        raise ValueError('stock profile requires unset MPS tuning/fallback environment: ' + ', '.join(tuned))
    model_dir = ROOT / {'clef-flash': 'model-flash', 'clef': 'model', 'clef-omni': 'model-omni'}[args.model]
    raw = args.plan.read_bytes()
    plan = json.loads(raw)
    rows = [row for row in plan['requests'] if row['model'] == args.model]
    if args.only:
        rows = [row for row in rows if row['modality'] in args.only]
    if args.ids:
        rows = [row for row in rows if row['id'] in args.ids]
        if {row['id'] for row in rows} != set(args.ids):
            raise ValueError('stock plan does not contain every selected ID')
    if not rows:
        raise ValueError('stock plan has no selected requests')
    # Short warm-up/timing fixtures precede long contexts; a preceding 16k input
    # must not define the short-request timing state.
    rows.sort(key=lambda row: (row['full_input_tokens'] >= 8000, row['modality'], row['id']))
    report = dict(complete=False, profile='stock model-card loader and Transformers defaults',
                  model=args.model, precision='bfloat16', device='mps', records=[], phase='loading',
                  torch=str(torch.__version__), transformers=transformers.__version__,
                  flags={'safe_attention': False, 'paging': False, 'staging': False,
                         'expert_override': None, 'allocator_override': None},
                  git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  plan_sha256=hashlib.sha256(raw).hexdigest(),
                  loader_sha256=hashlib.sha256((model_dir/'joint_schema_model.py').read_bytes()).hexdigest(),
                  benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  safe_attn_sha256=hashlib.sha256((ROOT/'ref/safe_attn.py').read_bytes()).hexdigest(),
                  hardware=subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),
                  metric='JSON decode, snapshot processing, inference and typed response serialization; load excluded')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        temporary=args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        temporary.replace(args.output)
    save()
    try:
        sys.path.insert(0, str(model_dir))
        from joint_schema_model import load_release_model, encode_record, collate_records, systemone_answer
        start=time.perf_counter()
        model, processor=load_release_model(model_dir,device='mps',dtype=torch.bfloat16)
        torch.mps.synchronize()
        report['load_seconds']=time.perf_counter()-start
        report['implementations']={name: {key: str(getattr(module.config,key,None)) for key in
            ('_attn_implementation','_experts_implementation')} for name,module in model.named_modules()
            if hasattr(module,'config') and name in ('','thinker','thinker.model','language_model')}
        report['phase']='inference';save()
        timed=set()
        with torch.inference_mode():
            for row in rows:
                body=json.dumps(row['request']).encode()
                if hashlib.sha256(body).hexdigest()!=row['request_sha256']:
                    raise ValueError('stock request differs from frozen payload')
                report['active_request']=row['id'];save()
                def send():
                    torch.mps.synchronize();start=time.perf_counter()
                    with stock_request(json.loads(body), args.model) as request:
                        encoded=encode_record(processor.tokenizer,request,processor=processor)
                    ids=list(encoded.input_ids)
                    logits=model(collate_records([encoded],processor.tokenizer.pad_token_id,torch.device('mps')))[0]
                    values=[value.float().cpu() for value in logits]
                    answers={q.question_id:systemone_answer(request['questions'][q.question_id],
                        dict(zip(q.option_ids,value.softmax(-1).tolist(),strict=True)))
                        for q,value in zip(encoded.questions,values,strict=True)}
                    response=json.dumps(dict(model=args.model,answers=answers,
                                             usage=dict(input_tokens=len(ids),output_tokens=0)))
                    torch.mps.synchronize()
                    return (time.perf_counter()-start)*1000,response,[v.tolist() for v in values],ids
                try:
                    cold,response,values,ids=send()
                    item=dict(id=row['id'],modality=row['modality'],request_sha256=row['request_sha256'],
                              input_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                              tokens=len(ids),cold_ms=cold,response=json.loads(response),logits=values)
                    if row['modality'] not in timed:
                        samples=[];repeat_changes=0
                        for _ in range(3):
                            ms,again,_,_=send();samples.append(ms);repeat_changes+=again!=response
                        item['warm']=dict(samples_ms=samples,median_ms=statistics.median(samples),
                                          response_changes=repeat_changes)
                        timed.add(row['modality'])
                    report['records'].append(item);save()
                    print(f"stock {args.model}/{row['id']}: {cold:.2f} ms, {len(ids)} tokens",flush=True)
                except Exception as error:
                    report['records'].append(dict(id=row['id'], modality=row['modality'],
                        request_sha256=row['request_sha256'], failed=True,
                        error=f'{type(error).__name__}: {error}'))
                    save()
                    print(f"stock {args.model}/{row['id']}: FAILED {type(error).__name__}: {error}", flush=True)
        report['failed_requests']=sum(bool(row.get('failed')) for row in report['records'])
        report.update(complete=True,phase='complete');report.pop('active_request',None);save()
    except Exception as error:
        report['error']=f'{type(error).__name__}: {error}';save()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stock', action='store_true')
    parser.add_argument('--model', choices=('clef-flash','clef','clef-omni'),default='clef-omni')
    parser.add_argument('--plan',type=Path)
    parser.add_argument('--ids',nargs='+',help='stock control IDs to collect in a separate continuation artifact')
    parser.add_argument('--samples', type=int, default=5)
    parser.add_argument('--sparse-gap', type=float, default=2)
    parser.add_argument('--only', nargs='+', choices=('text', 'images', 'video', 'audio', 'audio_video'))
    parser.add_argument('--mps-limit-gib', type=int, choices=range(64, 77), default=72)
    args = parser.parse_args()
    if args.output.exists() or args.samples < 3 or not 0 < args.sparse_gap <= 60:
        parser.error('require a new output, >=3 samples, and sparse gap in (0,60]')
    if args.stock:
        stock(args)
        return
    if args.inventory is None or args.model!='clef-omni':
        parser.error('resident/eager profile requires --inventory and clef-omni')
    bad = verify_snapshot(ROOT / 'model-omni', load_manifest(REVISION))
    if bad:
        raise ValueError('snapshot verification failed: ' + '; '.join(bad))
    sys.path.insert(0, str(ROOT / 'model-omni'))
    from joint_schema_model import collate_records, encode_record, systemone_answer
    # Full BF16 weights need about 60 GiB. Run under the external 80 GiB guard
    # as well; MPS allocator accounting alone omits CPU tensors and graph caches.
    torch.mps.set_per_process_memory_fraction((args.mps_limit_gib << 30) / torch.mps.recommended_max_memory())
    report = dict(complete=False, profile='resident BF16 with sequential loader and eager experts',
                  precision='bfloat16', revision=REVISION, records=[],
                  metric='JSON decode, pinned processing, inference and response serialization; model load excluded',
                  torch=str(torch.__version__), transformers=transformers.__version__,
                  attention='stock sdpa', experts='eager', samples=args.samples,
                  loading='sequential BF16 module copies; every weight resident before timing',
                  mps_limit_gib=args.mps_limit_gib, selected_modalities=args.only,
                  sparse_gap_seconds=args.sparse_gap, platform=platform.platform(),
                  hardware=subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    start = time.perf_counter()
    model, processor, pager = load_paged(ROOT / 'model-omni', 'mps', max_weight_bytes=72 << 30)
    resident = ExitStack()
    modules = [(model.output_embeddings, 'thinker.lm_head'),
               (model.thinker.model.embed_tokens, 'thinker.model.embed_tokens'),
               (model.thinker.model.rotary_emb, 'thinker.model.rotary_emb'),
               (model.thinker.visual, 'thinker.visual'), (model.thinker.audio_tower, 'thinker.audio_tower')]
    modules += [(layer, f'thinker.model.layers.{i}') for i, layer in enumerate(model.thinker.model.layers)]
    modules.append((model.thinker.model.norm, 'thinker.model.norm'))
    for module, prefix in modules:
        resident.enter_context(pager.materialize(module, prefix, dtype=torch.bfloat16))
    resident.enter_context(pager.materialize(model.head, '', dtype=torch.bfloat16, head=True))
    if any(p.is_meta for p in model.parameters()):
        raise ValueError('resident baseline has unloaded parameters')
    report['resident_weight_bytes'] = pager.active_bytes
    torch.mps.synchronize()
    report['load_seconds'] = time.perf_counter() - start
    inventory = json.loads(args.inventory.read_text())
    with torch.inference_mode():
        for modality, index in [('text', 0), ('images', 0), ('video', 2), ('audio', 5), ('audio_video', 0)]:
            if args.only and modality not in args.only:
                continue
            golden = ROOT / 'golden' / inventory[modality]['golden']
            requests = [json.loads(s) for s in io.StringIO((golden / 'requests.jsonl').read_text())]
            encodings = [json.loads(s) for s in io.StringIO((golden / 'encoded.jsonl').read_text())]
            request = requests[index]; expected_ids = encodings[index]['input_ids']
            payload = json.dumps(request)
            def send():
                torch.mps.synchronize(); start = time.perf_counter()
                parsed = json.loads(payload)
                encoded = encode_reference(encode_record, parsed, processor)
                if list(encoded.input_ids) != expected_ids:
                    raise ValueError('PyTorch request tokens differ from the FP32 fixture')
                logits = model(collate_records([encoded], processor.tokenizer.pad_token_id, torch.device('mps')))[0]
                values = [value.float().cpu() for value in logits]
                answer = {q.question_id: systemone_answer(parsed['questions'][q.question_id],
                          dict(zip(q.option_ids, value.softmax(-1).tolist())))
                          for q, value in zip(encoded.questions, values, strict=True)}
                response = json.dumps(dict(model=parsed.get('model', 'clef'), answers=answer,
                                           usage=dict(input_tokens=len(encoded.input_ids), output_tokens=0)))
                torch.mps.synchronize()
                return (time.perf_counter() - start) * 1000, response, [v.tolist() for v in values]
            _, expected, logits = send(); send()
            record = dict(modality=modality, id=request['id'], tokens=len(expected_ids),
                          request_sha256=hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest(),
                          response=json.loads(expected), logits=logits)
            for mode in ('warm', 'sparse'):
                samples = []
                for _ in range(args.samples):
                    if mode == 'sparse':
                        time.sleep(args.sparse_gap)
                    ms, response, _ = send()
                    if response != expected:
                        raise ValueError('repeated PyTorch request changed its rounded response')
                    samples.append(ms)
                record[mode] = dict(samples_ms=samples, median_ms=statistics.median(samples),
                                    min_ms=min(samples), max_ms=max(samples))
            report['records'].append(record)
            args.output.write_text(json.dumps(report, indent=2) + '\n')
            print(f'{modality}: {len(expected_ids)} tokens, warm {record["warm"]["median_ms"]:.2f} ms, '
                  f'sparse {record["sparse"]["median_ms"]:.2f} ms', flush=True)
    report['complete'] = True
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    resident.close()


if __name__ == '__main__':
    main()
