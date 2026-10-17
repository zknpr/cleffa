"""Check every captured Omni layer/suffix, not just the first corpus record."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from safetensors import safe_open


def load_reference_tensor(path, key):
    """Keep file-backed resident memory bounded by one tensor, not one record."""
    with safe_open(path, framework='np') as tensors:
        # Each handle touches one payload. Releasing its returned array allows
        # that mapping to be reclaimed instead of accumulating a whole record.
        return tensors.get_tensor(key)


def residual_error(actual,expected):
    """A nonfinite derived metric is invalid evidence, including finite FP64 inputs."""
    a,b=np.asarray(actual,dtype='f8'),np.asarray(expected,dtype='f8')
    if a.shape!=b.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('invalid shape or nonfinite residuals')
    with np.errstate(over='ignore',invalid='ignore'):
        delta=a-b
        relative=float(np.linalg.norm(delta)/max(1e-30,np.linalg.norm(b)))
        maximum=float(abs(delta).max())
    if not np.isfinite(relative) or not np.isfinite(maximum):
        raise ValueError('nonfinite residual comparison metric')
    return relative,maximum


def compare_references(golden, control, *, layers=48):
    """Measure two independent FP32 references; this never changes native limits."""
    golden,control=Path(golden),Path(control)
    left=json.loads((golden/'manifest.json').read_text())
    right=json.loads((control/'manifest.json').read_text())
    for path,meta,reduction in ((golden,left,'blocked'),(control,right,'sdpa')):
        if (not meta.get('complete') or meta.get('precision')!='float32-streamed' or
                meta.get('independent_from_native') is not True or 'arbitration' in meta or
                meta.get('attention_reference',{}).get('value_reduction')!=reduction or
                meta['attention_reference'].get('first_layer')!=0):
            raise ValueError('noise control requires independent FP32 references from layer zero')
        if hashlib.sha256((path/'requests.jsonl').read_bytes()).hexdigest()!=meta['corpus_sha256']:
            raise ValueError('noise control request hash differs')
    if (not left.get('revision') or left['revision']!=right.get('revision') or
            left['corpus_sha256']!=right['corpus_sha256'] or left['records']!=right['records'] or
            (golden/'encoded.jsonl').read_bytes()!=(control/'encoded.jsonl').read_bytes()):
        raise ValueError('noise control inputs or capture geometry differ')
    names=['embed']+[f'layer.{i:02d}' for i in range(layers)]+['final_norm']
    report=dict(complete=False,method='blocked FP32 versus query-bounded FP32 SDPA',
                native_limits_unchanged=True,records=[],questions=[])
    for record in left['records']:
        name=record['id'];values=[]
        with safe_open(golden/'layers'/(name+'.safetensors'),framework='np') as a, \
             safe_open(control/'layers'/(name+'.safetensors'),framework='np') as b:
            for key in names:
                x,y=a.get_tensor(key).astype('f8'),b.get_tensor(key).astype('f8')
                if x.shape!=y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
                    raise ValueError('invalid noise-control residuals')
                relative,maximum=residual_error(y,x)
                values.append(dict(layer=key,relative_l2=relative,max_absolute=maximum))
        report['records'].append(dict(id=name,first_token=record['dump_start'],layers=values))
    with safe_open(golden/'logits.safetensors',framework='np') as a, \
         safe_open(control/'logits.safetensors',framework='np') as b:
        if set(a.keys())!=set(b.keys()) or not a.keys():raise ValueError('noise-control questions differ')
        for key in a.keys():
            x,y=a.get_tensor(key).astype('f8'),b.get_tensor(key).astype('f8')
            if x.shape!=y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError('invalid noise-control logits')
            px=np.exp(x-x.max());px/=px.sum();py=np.exp(y-y.max());py/=py.sum()
            report['questions'].append(dict(id=key,max_logit_delta=float(abs(x-y).max()),
                max_probability_delta=float(abs(px-py).max()),argmax_equal=int(x.argmax())==int(y.argmax())))
    report['sources']={str(path):{name:hashlib.sha256((path/name).read_bytes()).hexdigest()
                                for name in ('manifest.json','requests.jsonl','encoded.jsonl','logits.safetensors')}
                       for path in (golden,control)}
    report['complete']=True
    return report


def compare(golden,captures,*,layers=48,reference='fp32'):
    golden,captures=Path(golden),Path(captures)
    manifest=json.loads((golden/'manifest.json').read_text())
    paired=json.loads((captures/'manifest.json').read_text())
    records=manifest['records']
    if reference=='fp64':
        import sys
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ref'))
        from oracle_fp64_bounded import validate_policy,verify_artifacts
        validate_policy(manifest)
        verify_artifacts(golden,manifest)
    elif reference!='fp32':
        raise ValueError('unknown reference lane')
    if (not manifest.get('complete') or not paired.get('complete') or
            (reference=='fp32' and manifest.get('precision')!='float32-streamed')):
        raise ValueError('layer comparison requires complete FP32 golden and paired captures')
    if not records or len({r['id'] for r in records})!=len(records) or {r['id'] for r in records}!={r['id'] for r in paired['records']}:
        raise ValueError('incomplete record inventory')
    if paired['requests_sha256']!=manifest['corpus_sha256']:
        raise ValueError('paired capture corpus differs from reference')
    names=['embed']+[f'layer.{i:02d}' for i in range(layers)]+['final_norm']
    report=dict(complete=False,limit=.01,records=[],failures=0)
    for record in records:
        name=record['id']
        if not isinstance(name,str) or not re.fullmatch(r'[A-Za-z0-9_-]+',name) or not record.get('dumped'):
            raise ValueError('invalid or missing layer record')
        T,start=record['tokens'],record['dump_start']
        if type(T) is not int or type(start) is not int or not 0<=start<T:
            raise ValueError('invalid layer suffix geometry')
        path=captures/(name+'.bin')
        with safe_open(golden/'layers'/(name+'.safetensors'),framework='np') as ref:
            expected_layers={k for k in ref.keys() if k.startswith('layer.')}
            if expected_layers!=set(names[1:-1]):raise ValueError('incomplete decoder layer inventory')
            shape=ref.get_slice('embed').get_shape()
            if len(shape)!=2 or shape[0]!=T-start or shape[1]<=0:raise ValueError('invalid reference embedding shape')
            H=shape[1]
            if path.stat().st_size!=(layers+2)*T*H*4:raise ValueError('invalid native residual file size')
            native=np.memmap(path,dtype='<f4',mode='r',shape=(layers+2,T,H))
            values=[]
            for i,key in enumerate(names):
                expected=load_reference_tensor(golden/'layers'/(name+'.safetensors'),key).astype('f8')
                if reference=='fp64' and ref.get_slice(key).get_dtype()!='F64':
                    raise ValueError('FP64 residual capture has a different dtype')
                actual=native[i,start:].astype('f8')
                if expected.shape!=actual.shape or not np.isfinite(expected).all() or not np.isfinite(actual).all():
                    raise ValueError(f'{name}/{key}: invalid shape or nonfinite residuals')
                relative,maximum=residual_error(actual,expected)
                values.append(dict(layer=key,relative_l2=relative,max_absolute=maximum))
                report['failures']+=int(relative>.01)
            del native
        hashes=manifest.get('arbitration',{}).get('native_activations_sha256',{})
        with path.open('rb') as source: digest=hashlib.file_digest(source,'sha256').hexdigest()
        if name in hashes and digest!=hashes[name]:raise ValueError('native activations differ from arbitration provenance')
        report['records'].append(dict(id=name,tokens=T,start=start,layers=values,sha256=digest))
        print(f'{name}: {len(names)} stages, {T-start} rows, max relative L2 {max(v["relative_l2"] for v in values):.9g}',flush=True)
    report['complete']=report['failures']==0
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('golden',type=Path);ap.add_argument('captures',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--reference-control',action='store_true',help='compare two independent FP32 references')
    ap.add_argument('--reference',choices=('fp32','fp64'),default='fp32',help='explicit independent FP64 inference lane; FP32 remains the default')
    args=ap.parse_args()
    report=(compare_references(args.golden,args.captures) if args.reference_control
            else compare(args.golden,args.captures,reference=args.reference))
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    raise SystemExit(0 if report['complete'] else 1)


if __name__=='__main__':main()
