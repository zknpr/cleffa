"""Native audio stages against the complete pinned FP32 model oracle.

The initial pass uses reference mel features to isolate the tower. End-to-end
request parity separately exercises the approved native frontend bounds.
"""
import argparse
import base64
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
from safetensors import safe_open
import torch
import av
from compare_omni_layers import residual_error

ROOT=Path(__file__).resolve().parents[1]


def native_features(request,folder,layout):
    from test_audio_features import Frontend
    inputs=list(request.get('audio') or [])
    inputs.extend(video['audio'] for video in request.get('videos') or [] if 'audio' in video)
    samples=[];paths=[]
    for i,value in enumerate(inputs):
        if isinstance(value,list):
            message=dict(samples=value);expected=np.asarray(value,dtype='<f4')
        else:
            payload=value['base64'] if isinstance(value,dict) else value
            message=dict(wav=payload)
            raw=base64.b64decode(payload.split(',',1)[-1])
            resampler=av.AudioResampler(format='flt',layout='mono',rate=16000)
            with av.open(io.BytesIO(raw)) as container:
                parts=[out.to_ndarray().reshape(-1) for frame in container.decode(audio=0) for out in resampler.resample(frame)]
                parts.extend(out.to_ndarray().reshape(-1) for out in resampler.resample(None))
            expected=np.concatenate(parts).astype('<f4')
        decoded=subprocess.run([ROOT/'clef-tool','audio'],input=json.dumps(message)+'\n',capture_output=True,text=True,check=True)
        native=bytes.fromhex(json.loads(decoded.stdout)['hex'])
        if native!=expected.tobytes():raise ValueError('native samples differ from independent PyAV decoding')
        path=folder/f'samples.{i}.f32';path.write_bytes(native);paths.append(path);samples.append(expected)
    Frontend.setUpClass()
    Frontend().check(samples)  # Independent FP64 checks of every FFT/mel/log stage.
    frontend=folder/'frontend';frontend.mkdir()
    result=subprocess.run([ROOT/'tests/audio-features',frontend,*paths],capture_output=True,text=True,check=True)
    meta=json.loads(result.stdout)
    want=[dict(frames=c['frames'],padded_frames=c['padded_frames'],tokens=len(c['token_rows'])) for c in layout]
    if meta!=want:raise ValueError('native frontend padding/mask/token geometry differs')
    # The frontend stores each clip's own frames; the tower reads nothing beyond them.
    return [np.fromfile(frontend/f'{i}-features.bin',dtype='<f4').reshape(128,c['frames']) for i,c in enumerate(layout)]


def compare(model,golden,output,*,poison=False,frontend=False,reference='fp32'):
    manifest=json.loads((golden/'manifest.json').read_text())
    if reference=='fp64':
        sys.path.insert(0,str(ROOT/'ref'))
        from omni_fp64_policy import validate_policy,verify_artifacts
        validate_policy(manifest)
        verify_artifacts(golden,manifest)
    elif reference!='fp32':raise ValueError('unknown reference lane')
    if not manifest.get('complete') or (reference=='fp32' and manifest.get('precision')!='float32-streamed'):
        raise ValueError('tower verification requires complete FP32 reference')
    if output.exists():raise ValueError('refusing to replace a tower capture')
    output.mkdir(parents=True)
    report=dict(complete=False,limit=.01,poison=poison,native_frontend=frontend,records=[])
    failures=0
    encoded={r['id']:r for r in map(json.loads,(golden/'encoded.jsonl').read_text().splitlines())}
    requests={r['id']:r for r in map(json.loads,(golden/'requests.jsonl').read_text().splitlines())}
    for record in manifest['records']:
        name=record['id']
        if not re.fullmatch(r'[A-Za-z0-9_-]+',name):raise ValueError('invalid record ID')
        layout=encoded[name]['audio']
        folder=output/name;folder.mkdir()
        features=native_features(requests[name],folder,layout) if frontend else None
        command=[str(ROOT/'tests/audio-tower'),str(model),str(folder),str(layout[0]['padded_frames'])]
        with safe_open(golden/'layers'/f'{name}.safetensors',framework='np') as reference:
            for i,clip in enumerate(layout):
                mel=reference.get_tensor(f'audio.{i}.features')
                if mel.shape!=(128,clip['padded_frames']) or not np.isfinite(mel).all():raise ValueError('invalid mel reference')
                if features is not None:
                    own=mel[:,:clip['frames']]
                    delta=features[i].astype('f8')-own
                    if abs(delta).max()>5e-5 or np.linalg.norm(delta)/max(1e-30,np.linalg.norm(own.astype('f8')))>5e-6:
                        raise ValueError('native frontend exceeds approved bounds against golden')
                    mel=mel.copy();mel[:,:clip['frames']]=features[i]
                path=folder/f'mel.{i}.f32';mel.astype('<f4').tofile(path)
                command.extend([str(path),str(clip['frames'])])
            env=dict(os.environ)
            env.pop('CLEF_DEBUG_POISON',None)
            if poison:env['CLEF_DEBUG_POISON']='1'
            result=subprocess.run(command,env=env,capture_output=True,text=True,timeout=600)
            (folder/'stderr.log').write_text(result.stderr)
            if result.returncode:raise RuntimeError(f'{name}: {result.stderr}')
            comparisons=[]
            def check(stage,native,expected):
                nonlocal failures
                if native.shape!=expected.shape or not np.isfinite(native).all() or not np.isfinite(expected).all():
                    raise ValueError(f'{name}/{stage}: invalid shape or nonfinite stage')
                relative,maximum=residual_error(native,expected)
                comparisons.append(dict(stage=stage,relative_l2=relative,max_absolute=maximum))
                failures+=int(relative>.01)
            for layer in range(1,4):
                raw=reference.get_tensor(f'audio.conv2d{layer}')
                expected=torch.nn.functional.gelu(torch.from_numpy(raw.copy())).numpy().transpose(0,2,3,1)
                parts=[]
                for first in range(0,len(expected),8):
                    size=min(8,len(expected)-first)
                    part=np.fromfile(folder/f'conv2d{layer}.gelu.{first}.f32',dtype='<f4').reshape(size,*expected.shape[1:])
                    parts.append(part)
                check(f'conv2d{layer}.gelu',np.concatenate(parts),expected)
            position=reference.get_tensor('audio.position_table')
            check('position',np.fromfile(folder/'position.0.f32',dtype='<f4').reshape(position.shape),position)
            for stage in ['conv']+[f'layer.{i:02d}' for i in range(32)]+['output']:
                expected=np.concatenate([reference.get_tensor(f'audio.{i}.{stage}') for i in range(len(layout))])
                native=np.fromfile(folder/f'{stage}.0.f32',dtype='<f4').reshape(expected.shape)
                offset=0
                for i,clip in enumerate(layout):
                    count=len(clip['token_rows'])
                    check(f'{i}.{stage}',native[offset:offset+count],expected[offset:offset+count]);offset+=count
                if offset!=len(expected):raise ValueError('audio row geometry differs')
            report['records'].append(dict(id=name,stages=comparisons))
            print(f'{name}: {len(comparisons)} audio stages, max relative L2 {max(v["relative_l2"] for v in comparisons):.9g}',flush=True)
    report.update(complete=failures==0,failures=failures)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('model',type=Path);ap.add_argument('golden',type=Path);ap.add_argument('output',type=Path)
    ap.add_argument('--poison',action='store_true')
    ap.add_argument('--native-frontend',action='store_true')
    ap.add_argument('--reference',choices=('fp32','fp64'),default='fp32')
    args=ap.parse_args()
    result=compare(args.model,args.golden,args.output,poison=args.poison,frontend=args.native_frontend,reference=args.reference)
    raise SystemExit(0 if result['complete'] else 1)


if __name__=='__main__':main()
