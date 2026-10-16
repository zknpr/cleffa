"""Offline, independently checked local FP32 rounding evidence.

Every stage is recomputed in CPU FP64 from that stage's actual native input.
The probe shares production dispatch; it never supplies values to the oracle.
Only arbitrate() may accept a choice, after its separate one-ULP eligibility test.
"""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
STAGES = ('input_norm','qkv','q','k','v','attn','residual','norm')


def write_audit_report(folder, rows, report):
    # Long contexts can dispute hundreds of rows; spelling every row in the
    # filename exceeds NAME_MAX. The report itself retains the full row list.
    identity = json.dumps(rows.tolist(), separators=(',', ':')).encode()
    path = Path(folder) / ('audit-' + hashlib.sha256(identity).hexdigest() + '.json')
    path.write_text(json.dumps(report, indent=2) + '\n')
    return path


def digest(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source,'sha256').hexdigest()


def native_positions(positions,tokens):
    pos=positions.detach().cpu().numpy()
    # The pinned media processor returns float32 positions, including audio-only
    # records. Require exact integral values before the native int32 conversion.
    if pos.shape!=(3,1,tokens) or tokens<=0 or pos.dtype.kind not in 'ifu' or not np.isfinite(pos).all() or \
       pos.min()<0 or pos.max()>65536 or not np.equal(pos,np.trunc(pos)).all():
        raise ValueError('invalid native stage probe positions')
    return np.ascontiguousarray(pos[:,0].T,dtype='<i4')


def stage_error(name, actual, expected):
    if actual.shape != expected.shape or not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise ValueError(f'{name}: invalid stage geometry or nonfinite values')
    actual,expected = actual.double().reshape(-1,actual.shape[-1]),expected.double().reshape(-1,expected.shape[-1])
    delta = actual-expected
    metrics = dict(max_absolute=float(delta.abs().max()),
                   max_row_relative_l2=float((delta.norm(dim=-1)/expected.norm(dim=-1).clamp_min(1e-30)).max()),
                   max_row_scaled_absolute=float((delta.abs().amax(-1)/expected.abs().amax(-1).clamp_min(1)).max()))
    if name == 'router':
        metrics['absolute_limit'] = 1e-9
        bad = metrics['max_absolute'] > 1e-9
    else:
        metrics.update(relative_l2_limit=1e-5,scaled_absolute_limit=32*torch.finfo(torch.float32).eps)
        bad = metrics['max_row_relative_l2'] > metrics['relative_l2_limit'] or metrics['max_row_scaled_absolute'] > metrics['scaled_absolute_limit']
    if bad:
        raise ValueError(f'{name}: independent FP64 stage audit failed: {metrics}')
    return metrics


@torch.inference_mode()
def audit_stages(layer, incoming, pe, rows, stages, native_logits):
    """Audit every causal dependency of rows, including all preceding K/V rows."""
    if incoming.ndim != 2 or rows.ndim != 1 or not len(rows) or int(rows.min()) < 0 or int(rows.max()) >= len(incoming):
        raise ValueError('invalid stage audit rows or incoming geometry')
    a = layer.self_attn
    hd = a.head_dim
    nh,nkv = a.q_proj.weight.shape[0]//hd,a.k_proj.weight.shape[0]//hd
    T,H = incoming.shape
    end = int(rows.max())+1
    indices = rows.cpu().numpy()
    shapes = dict(input_norm=(T,H),qkv=(T,(nh+2*nkv)*hd),q=(nh,T,hd),k=(nkv,T,hd),v=(nkv,T,hd),
                  attn=(T,nh*hd),residual=(T,H),norm=(T,H))
    if set(stages) != set(shapes) or any(tuple(stages[k].shape) != shape for k,shape in shapes.items()):
        raise ValueError('incomplete or malformed stage capture')
    if any(p.bias is not None for p in (a.q_proj,a.k_proj,a.v_proj,a.o_proj)):
        raise ValueError('stage audit does not support attention biases')
    def cpu(x):
        return x.detach().cpu().double() if isinstance(x,torch.Tensor) else torch.from_numpy(np.array(x,copy=True)).double()
    def norm(x,module):
        return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+module.variance_epsilon)*cpu(module.weight)
    def linear(x,module):
        return x@cpu(module.weight).T
    def rope(x,c,s):
        return x*c+torch.cat((-x[...,hd//2:],x[...,:hd//2]),-1)*s
    metrics = {}
    def check(name,actual,expected):
        metrics[name] = stage_error(name,actual,expected)
    h = cpu(incoming[:end])
    x = cpu(stages['input_norm'][:end])
    check('input_norm',x,norm(h,layer.input_layernorm))
    packed = cpu(stages['qkv'][:end])
    q0 = packed[rows,:nh*hd].reshape(-1,nh,hd).transpose(0,1)
    k0 = packed[:,nh*hd:(nh+nkv)*hd].reshape(-1,nkv,hd).transpose(0,1)
    v0 = packed[:,(nh+nkv)*hd:].reshape(-1,nkv,hd).transpose(0,1)
    check('query_projection',packed[rows,:nh*hd],linear(x[rows],a.q_proj))
    check('key_projection',packed[:,nh*hd:(nh+nkv)*hd],linear(x,a.k_proj))
    check('value_projection',packed[:,(nh+nkv)*hd:],linear(x,a.v_proj))
    co,si = (cpu(p[0,:end]) for p in pe)
    q,k,v = cpu(stages['q'][:,indices]),cpu(stages['k'][:,:end]),cpu(stages['v'][:,:end])
    check('query_norm_rotary',q,rope(norm(q0,a.q_norm),co[rows],si[rows]))
    check('key_norm_rotary',k,rope(norm(k0,a.k_norm),co,si))
    if not torch.equal(v,v0):
        raise ValueError('value layout: native copy differs from projection')
    metrics['value_layout'] = dict(byte_exact=True)
    result = torch.empty(len(rows),nh,hd,dtype=torch.float64)
    keys = torch.arange(end)
    rep = nh//nkv
    for start in range(0,len(rows),16):
        stop = min(start+16,len(rows))
        mask = keys[None,:] > rows[start:stop,None]
        for head in range(nkv):
            scores = q[head*rep:(head+1)*rep,start:stop]@k[head].T*a.scaling
            scores.masked_fill_(mask,-torch.inf)
            result[start:stop,head*rep:(head+1)*rep] = (scores.softmax(-1)@v[head]).transpose(0,1)
    attn = cpu(stages['attn'][indices])
    check('attention',attn.reshape(-1,nh,hd),result)
    residual = cpu(stages['residual'][indices])
    check('output_projection_residual',residual,h[rows]+linear(attn,a.o_proj))
    ffn = cpu(stages['norm'][indices])
    check('ffn_norm',ffn,norm(residual,layer.post_attention_layernorm))
    logits = linear(ffn,layer.mlp.gate)
    check('router',cpu(native_logits),logits)
    return logits,metrics


class RoundingAuditor:
    """Lazily capture a disputed layer and keep reproducible, hashed artifacts."""
    def __init__(self, folder, activations, routes):
        self.folder,self.activations,self.routes = Path(folder),Path(activations),Path(routes)
        self.paired = json.loads((self.activations/'manifest.json').read_text())
        routing = json.loads((self.routes/'manifest.json').read_text())
        if not self.paired.get('complete') or not routing.get('complete') or any(
                self.paired[k] != routing[k] for k in ('model_sha256','binary_sha256','requests_sha256','records')):
            raise ValueError('rounding audits require paired route/residual provenance')
        self.model = Path(self.paired['model'])
        self.checked_model = False
        self.folder.mkdir(parents=True,exist_ok=True)

    def __call__(self, record, number, layer, incoming, pe, rows, positions):
        if not isinstance(record,str) or not re.fullmatch(r'[A-Za-z0-9_-]+',record) or record not in {r['id'] for r in self.paired['records']}:
            raise ValueError('invalid rounding record ID')
        if not self.checked_model:
            if digest(self.model) != self.paired['model_sha256']:
                raise ValueError('rounding model differs from paired capture')
            self.checked_model = True
        probe = ROOT/'tests/omni-layer-probe'
        native = np.asarray(incoming,dtype='<f4')
        if native.ndim != 2 or native.shape[1] != 2048 or not np.isfinite(native).all():
            raise ValueError('invalid native stage probe inputs')
        pos = native_positions(positions,len(native))
        identity = dict(version=1,record=record,layer=number,tokens=len(native),
                        input_sha256=hashlib.sha256(native.tobytes()).hexdigest(),
                        positions_sha256=hashlib.sha256(pos.tobytes()).hexdigest(),
                        probe_sha256=digest(probe),model_sha256=self.paired['model_sha256'],
                        paired_binary_sha256=self.paired['binary_sha256'])
        destination = self.folder/f'{record}.layer{number:02d}'
        if not destination.exists():
            print(f'{record} layer {number}: capturing local-rounding stage audit',flush=True)
            if any(p.device.type == 'mps' for p in layer.parameters()):
                torch.mps.synchronize()
            with tempfile.TemporaryDirectory(prefix=destination.name+'.incomplete-',dir=self.folder) as temp:
                temp = Path(temp)
                native.tofile(temp/'input.bin'); pos.tofile(temp/'positions.bin')
                shutil.copy2(probe,temp/'probe')
                if digest(temp/'probe') != identity['probe_sha256']:
                    raise ValueError('stage probe changed while preparing capture')
                completed = subprocess.run([str(temp/'probe'),str(self.model),str(temp/'input.bin'),str(temp/'positions.bin'),
                                            str(temp),str(number),str(len(native))],capture_output=True,text=True,timeout=3600)
                if completed.returncode:
                    raise ValueError(f'native stage probe failed: {completed.stderr}')
                manifest = dict(complete=True,identity=identity,sha256={name:digest(temp/(name+'.bin')) for name in STAGES})
                (temp/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
                temp.rename(destination)
        manifest = json.loads((destination/'manifest.json').read_text())
        if not manifest.get('complete') or manifest['identity'] != identity or set(manifest['sha256']) != set(STAGES):
            raise ValueError('stage capture provenance mismatch; use a fresh audit directory')
        if (destination/'probe').exists() and digest(destination/'probe') != identity['probe_sha256']:
            raise ValueError('archived stage probe hash mismatch')
        T = len(native)
        shapes = dict(input_norm=(T,2048),qkv=(T,5120),q=(32,T,128),k=(4,T,128),v=(4,T,128),
                      attn=(T,4096),residual=(T,2048),norm=(T,2048))
        stages = {}
        for name,shape in shapes.items():
            path = destination/(name+'.bin')
            if path.stat().st_size != int(np.prod(shape))*4 or digest(path) != manifest['sha256'][name]:
                raise ValueError(f'{name}: stage capture size or hash mismatch')
            stages[name] = np.memmap(path,dtype='<f4',mode='r',shape=shape)
        wanted = set(rows.tolist())
        traces = {}
        with (self.routes/(record+'.jsonl')).open() as source:
            for line in source:
                row = json.loads(line)
                if row['layer'] == number and row['token'] in wanted:
                    if row['token'] in traces:
                        raise ValueError('duplicate native trace row')
                    traces[row['token']] = row['logits']
                if row['layer'] > number:
                    break
        if set(traces) != wanted:
            raise ValueError('missing native trace rows for stage audit')
        logits = torch.tensor([traces[r] for r in rows.tolist()],dtype=torch.float64)
        try:
            high,metrics = audit_stages(layer,native,pe,rows,stages,logits)
        except ValueError as error:
            (destination/'failure.json').write_text(json.dumps(dict(rows=rows.tolist(),error=str(error)),indent=2)+'\n')
            raise
        report = dict(identity=identity,stages=metrics,rows=rows.tolist(),artifact=str(destination.resolve()),
                      manifest_sha256=digest(destination/'manifest.json'),
                      native_router_logits_sha256=hashlib.sha256(logits.numpy().astype('<f8').tobytes()).hexdigest())
        write_audit_report(destination, rows, report)
        return high,[report for _ in rows]
