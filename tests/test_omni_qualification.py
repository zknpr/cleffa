"""Run the complete Omni qualification matrix sequentially; missing evidence fails."""
import argparse
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np
from safetensors import safe_open

ROOT=Path(__file__).resolve().parents[1]
REVISION='0db1cd2607d76a7bdb2a382f659e7b313079f84b'
CORPORA={'text':('corpus','text'),'images':('corpus_omni_vision','omni-vision'),
         'video':('corpus_omni_video','omni-video'),'audio':('corpus_audio','audio'),
         'audio_video':('corpus_omni_av','omni-av'),'long':('corpus_omni_long','omni-long')}


def jsonl(path):return [json.loads(line) for line in io.StringIO(path.read_text()) if line.strip()]


def binary_hashes():
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ('clef','clef-server','clef-tool','clef-media')}


def validate_capture(path,model_hash,binary_hash):
    paired=json.loads((Path(path)/'manifest.json').read_text())
    if not paired.get('complete'):raise ValueError('incomplete native capture')
    if paired.get('model_sha256')!=model_hash:raise ValueError('capture weights differ from qualification model')
    if paired.get('binary_sha256')!=binary_hash:raise ValueError('capture executable differs from qualified CLI')


def validate_golden(path,expected,corpus,*,reference='fp32'):
    path=Path(path);manifest=json.loads((path/'manifest.json').read_text())
    if reference=='fp64':
        sys.path.insert(0,str(ROOT/'ref'))
        from oracle_fp64_bounded import validate_policy,verify_artifacts
        from compare_omni_layers import load_reference_tensor
        validate_policy(manifest)
    elif reference=='fp32':
        if not manifest.get('complete') or manifest.get('revision')!=REVISION or manifest.get('precision')!='float32-streamed':
            raise ValueError(f'{path}: incomplete or stale FP32 reference')
        attention=manifest.get('attention_reference') or {}
        if (manifest.get('independent_from_native') is not True or 'arbitration' in manifest or
                attention.get('value_reduction')!='blocked' or attention.get('first_layer')!=0):
            raise ValueError(f'{path}: qualification requires an independent blocked reference from layer zero')
    else:raise ValueError('unknown reference lane')
    requests=jsonl(path/'requests.jsonl');encoded=jsonl(path/'encoded.jsonl')
    if not expected or requests!=expected or manifest.get('corpus')!=corpus:
        raise ValueError(f'{path}: reference corpus differs from source')
    digest=hashlib.sha256((path/'requests.jsonl').read_bytes()).hexdigest()
    if digest!=manifest.get('corpus_sha256'):raise ValueError(f'{path}: corpus hash mismatch')
    ids=[r['id'] for r in expected];records=manifest.get('records',[])
    if len(set(ids))!=len(ids) or [r['id'] for r in records]!=ids or [r['id'] for r in encoded]!=ids:
        raise ValueError(f'{path}: incomplete record inventory')
    decoder={'embed','final_norm'}|{f'layer.{i:02d}' for i in range(48)}
    for request,record,enc in zip(requests,records,encoded):
        name=record['id'];tokens=record['tokens'];start=record['dump_start']
        if not re.fullmatch(r'[A-Za-z0-9_-]+',name) or not record.get('dumped') or not 0<=start<tokens or tokens!=len(enc['input_ids']):
            raise ValueError(f'{path}: invalid record geometry')
        if corpus=='omni-long' and tokens!=64000:raise ValueError('long-context fixture must contain 64000 tokens')
        media=manifest.get('media_shapes',{}).get(name,{})
        clips=len(request.get('audio') or [])+sum('audio' in v for v in request.get('videos') or [])
        required=set()
        if clips:
            required.update(['audio.position_table','audio.conv2d1','audio.conv2d2','audio.conv2d3'])
            for clip in range(clips):
                required.update(f'audio.{clip}.{key}' for key in ['features','conv','output']+[f'layer.{i:02d}' for i in range(32)])
        if request.get('images') or request.get('videos'):
            if not enc.get('images'):raise ValueError(f'{path}/{name}: missing visual layout')
            for image in range(len(enc['images'])):
                required.update([f'vision.{image}']+[f'vision.{image}.deepstack.{i}' for i in range(3)])
        if not required<=set(media):raise ValueError(f'{path}/{name}: incomplete media feature capture')
        with safe_open(path/'layers'/(name+'.safetensors'),framework='np') as tensors:
            keys=set(tensors.keys())
            if not decoder|set(media)<=keys:raise ValueError(f'{path}/{name}: missing layer or feature tensor')
            for key in decoder:
                shape=tensors.get_slice(key).get_shape()
                if len(shape)!=2 or shape[0]!=tokens-start or shape[1]<=0:raise ValueError('invalid decoder layer shape')
            for key,shape in media.items():
                if tensors.get_slice(key).get_shape()!=shape:raise ValueError('media capture shape mismatch')
            if reference=='fp64':
                routing={f'routing.{i:02d}.{key}' for i in range(48) for key in ('ids','weights','logits','margin')}
                if keys!=decoder|set(media)|routing:raise ValueError('FP64 stage inventory differs')
                for key in sorted(keys):
                    value=load_reference_tensor(path/'layers'/(name+'.safetensors'),key)
                    if re.fullmatch(r'audio\.\d+\.features',key):want='float32'
                    elif key.endswith('.ids'):want='int64'
                    else:want='float64'
                    if str(value.dtype)!=want:raise ValueError(f'FP64 capture dtype differs: {key}')
                    if not value.size or not np.isfinite(value).all():raise ValueError('nonfinite FP64 intermediate')
                    if key.startswith('routing.'):
                        part=key.rsplit('.',1)[1]
                        shape=(tokens-start,)+({'ids':(8,),'weights':(8,),'logits':(128,),'margin':()}[part])
                        if value.shape!=shape:raise ValueError('FP64 routing shape differs')
                        if part=='ids' and (np.any(value<0) or np.any(value>=128) or
                                           np.any(np.diff(np.sort(value,axis=-1),axis=-1)==0)):
                            raise ValueError('invalid FP64 routing IDs')
                        if part=='weights' and (np.any(value<0) or not np.allclose(value.sum(-1),1,rtol=0,atol=1e-14)):
                            raise ValueError('invalid FP64 routing weights')
    with safe_open(path/'logits.safetensors',framework='np') as logits:
        if not logits.keys():raise ValueError('empty logits reference')
        if reference=='fp64':
            questions={enc['id']+'/'+q['id']:len(q['option_ids']) for enc in encoded for q in enc.get('questions',[])}
            if not questions or set(logits.keys())!=set(questions):raise ValueError('FP64 question inventory differs')
        for key in logits.keys():
            value=logits.get_tensor(key)
            if not value.size or not np.isfinite(value).all():raise ValueError('nonfinite or empty logits reference')
            if reference=='fp64' and (str(value.dtype)!='float64' or value.shape!=(questions[key],)):
                raise ValueError('FP64 logits dtype or shape differs')
    if reference=='fp64':verify_artifacts(path,manifest)
    return dict(records=len(records),corpus_sha256=digest,revision=REVISION)


def run_gates(gates,folder,*,runner=subprocess.run):
    if not gates or len({name for name,_ in gates})!=len(gates):raise ValueError('qualification needs unique nonempty gates')
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    report=dict(complete=False,gates=[])
    (folder/'gates.json').write_text(json.dumps(report,indent=2)+'\n')
    env={k:v for k,v in os.environ.items() if not k.startswith('CLEF_')}
    for name,command in gates:
        if not re.fullmatch(r'[a-z0-9-]+',name):raise ValueError('invalid gate name')
        start=time.monotonic();entry=dict(name=name,command=list(map(str,command)),returncode=None)
        try:
            result=runner(entry['command'],cwd=ROOT,env=env,text=True,capture_output=True,timeout=7200)
            entry['returncode']=result.returncode
            log=folder/(name+'.log');log.write_text(result.stdout+result.stderr)
            entry['log']=str(log);entry['log_sha256']=hashlib.sha256(log.read_bytes()).hexdigest()
        except (OSError,subprocess.TimeoutExpired) as exc:entry['error']=str(exc)
        entry['seconds']=time.monotonic()-start;report['gates'].append(entry)
        print(f'{name}: {entry["returncode"]} ({entry["seconds"]:.1f}s)',flush=True)
        (folder/'gates.json').write_text(json.dumps(report,indent=2)+'\n')
    report['complete']=all(g['returncode']==0 for g in report['gates'])
    (folder/'gates.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def build_binaries(folder,*,runner=subprocess.run):
    # make test does not depend on the CLI. Build every production executable
    # before binding previously captured layers to the code under qualification.
    result=run_gates([('build',['make','-j1','all','media'])],folder,runner=runner)
    if not result['complete']:raise ValueError('production executable build failed')
    return result,binary_hashes()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--golden-root',type=Path,default=ROOT/'golden')
    ap.add_argument('--inventory',type=Path,help='Every modality needs golden/captures; short corpora also need an FP32 sdpa control directory')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--reference',choices=('fp32','fp64'),default='fp32',help='explicit FP64 inference lane; does not change FP32 qualification')
    args=ap.parse_args()
    args.model=args.model.resolve();args.golden_root=args.golden_root.resolve();args.output=args.output.resolve()
    if args.output.exists():ap.error('refusing to replace qualification evidence')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    report=dict(complete=False,reference=args.reference,model=str(args.model.resolve()),modalities={},gates=[])
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    try:
        artifacts=args.output.with_suffix('');artifacts.mkdir(parents=True,exist_ok=True)
        build,report['binaries_sha256']=build_binaries(artifacts/'build')
        report['gates']=build['gates']
        inventory=json.loads((args.inventory or args.golden_root/'omni-qualification-inventory.json').read_text())
        if set(inventory)!=set(CORPORA):raise ValueError('inventory must include text, images, video, audio, audio_video and long')
        for entry in inventory.values():entry['captures']=str(Path(entry['captures']).resolve())
        sys.path.insert(0,str(ROOT/'ref'));python=str(ROOT/'.venv/bin/python')
        def py(script,*options):return [python,'-B',str(ROOT/script),*map(str,options)]
        for name,(module,corpus) in CORPORA.items():
            entry=inventory[name];golden=args.golden_root/entry['golden'];captures=Path(entry['captures'])
            expected=importlib.import_module(module).build()
            report['modalities'][name]=validate_golden(golden,expected,corpus,reference=args.reference)
            if name!='long' and not entry.get('control'):
                raise ValueError(f'{name}: independent FP32 noise-floor control is missing')
            if args.reference=='fp64' and name!='long':
                if not entry.get('fp32'):raise ValueError(f'{name}: original FP32 evidence is missing')
                validate_golden(args.golden_root/entry['fp32'],expected,corpus)
        with args.model.open('rb') as source:report['model_sha256']=hashlib.file_digest(source,'sha256').hexdigest()
        for entry in inventory.values():
            validate_capture(entry['captures'],report['model_sha256'],report['binaries_sha256']['clef'])
        # Each command runs to completion before the next starts. CPU-only and
        # GPU gates share one ordered log so no passing modality hides a failure.
        gates=[('host',['make','test']),('encoded-video-host',['make','-j1','test-encoded-video']),
               ('encoded-video-server',py('tests/test_server_encoded_video.py')),
               ('gguf',py('tests/verify_gguf.py','model-omni',args.model)),
               ('cli-errors',['make','-j1','test-errors']),
               ('kernels',['make','-j1','test-omni-attention','test-omni-router','test-omni-moe','test-omni-deepstack',
                           'test-omni-audio','test-omni-buffers','test-vision-attention','test-vision-gemm']),
               ('host-vision',py('tests/test_omni_record.py',args.model,'model-omni')),
               ('host-audio',py('tests/test_omni_audio_record.py',args.model,args.golden_root/inventory['audio']['golden'])),
               ('host-av',py('tests/test_omni_av_record.py',args.model,args.golden_root/inventory['audio_video']['golden'])),
               ('context',py('tests/test_omni_context.py',args.model)),
               ('audio-helper',py('tests/test_audio_request.py')),('video-helper',py('tests/test_omni_video_request.py')),
               ('legacy-video-helper',py('tests/test_video_request.py'))]
        for name,entry in inventory.items():
            label=name.replace('_','-');golden=args.golden_root/entry['golden']
            # Captures were bound above to this exact CLI/model. The following
            # all-record layer gate checks them; --dump would repeat the first
            # prefill just to compare a subset of the same residual evidence.
            gates.append((label+'-parity',py('tests/test_parity.py',args.model,golden)))
            gates.append((label+'-layers',py('tests/compare_omni_layers.py',golden,entry['captures'],'--reference',args.reference,
                                           '--output',artifacts/(label+'-layers.json'))))
            # Corrupt or missing route evidence is an error; route differences are
            # reported without changing final-output or layer acceptance limits.
            gates.append((label+'-routing-diagnostic',py('tests/compare_omni_routes.py',args.model,golden,
                         '--captures',entry['captures'],'--output',artifacts/(label+'-routing.json'))))
            if name!='long':
                fp32=args.golden_root/entry['fp32'] if args.reference=='fp64' else golden
                gates.append((label+'-reference-spread',py('tests/compare_omni_layers.py',fp32,
                             args.golden_root/entry['control'],'--reference-control',
                             '--output',artifacts/(label+'-reference-spread.json'))))
        gates.extend([('audio-tower-build',['make','tests/audio-tower']),
                      ('audio-tower',py('tests/test_audio_tower.py',args.model,args.golden_root/inventory['audio']['golden'],
                                        artifacts/'audio-tower','--native-frontend','--poison','--reference',args.reference)),
                      ('av-tower',py('tests/test_audio_tower.py',args.model,args.golden_root/inventory['audio_video']['golden'],
                                     artifacts/'av-tower','--native-frontend','--poison','--reference',args.reference)),
                      ('audio-server',py('tests/test_omni_server_audio.py',args.model,args.golden_root/inventory['audio']['golden'],
                                        args.golden_root/inventory['audio_video']['golden'])),
                      ('vision-server',py('tests/test_omni_server_media.py',args.model,args.golden_root/inventory['images']['golden'])),
                      ('omni-runtime',py('tests/test_omni_runtime.py')),('omni-routes',py('tests/test_omni_routes.py')),
                      ('omni-text-server',py('tests/test_omni_server.py')),
                      ('omni-groups',py('tests/test_omni_groups.py',args.model)),
                      ('omni-bf16-vision-invariance',py('tests/test_omni_groups.py',args.model,'--bf16'))])
        for model in ('clef-flash','clef'):
            for corpus in ('','-vision','-video'):
                name=model+corpus
                gates.append((name+'-regression',py('tests/test_parity.py',ROOT/'gguf'/(model+'.gguf'),args.golden_root/(name+'-f32'),'--dump')))
        gates.extend([('legacy-image-server',py('tests/test_server_images.py',ROOT/'gguf/clef-flash.gguf',
                                               args.golden_root/'clef-flash-vision-f32/requests.jsonl')),
                      ('legacy-video-server',py('tests/test_server_video.py',ROOT/'gguf/clef-flash.gguf',
                                               args.golden_root/'clef-flash-video-f32/requests.jsonl'))])
        results=run_gates(gates,artifacts)
        report['gates'].extend(results['gates'])
        if binary_hashes()!=report['binaries_sha256']:raise ValueError('executables changed during qualification')
        report['complete']=results['complete']
        report['completed_modalities']=list(CORPORA) if report['complete'] else []
    except Exception as exc:
        report['error']=f'{type(exc).__name__}: {exc}'
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    raise SystemExit(0 if report['complete'] else 1)


if __name__=='__main__':main()
