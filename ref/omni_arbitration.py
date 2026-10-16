"""Independent FP64 arbitration of near-tied native/reference expert choices.

Only a disputed top-k boundary is eligible. Attention, its residual, FFN norm
and router are re-evaluated in CPU FP64. Input-sensitive ties additionally need
bounded incoming activation error and FP64 confirmation from the native input.
Reference expert execution and weights still use the reference's own activations.
"""
import json
from pathlib import Path

import torch


class ActivationCapture:
    """Map only a requested stage so completed layers release their file pages."""
    def __init__(self,path,shape):self.path,self.shape=path,shape

    def __len__(self):return self.shape[0]

    def __getitem__(self,index):
        import numpy as np
        if type(index)!=int or not 0<=index<len(self):raise IndexError('invalid activation layer')
        return np.memmap(self.path,dtype='<f4',mode='r',shape=self.shape[1:],
                         offset=index*self.shape[1]*self.shape[2]*4)


def load_activations(path, layers, tokens, hidden):
    path = Path(path)
    shape = (layers+2,tokens,hidden)
    if path.stat().st_size != (layers+2)*tokens*hidden*4:
        raise ValueError('native activation capture has incorrect size')
    return ActivationCapture(path,shape)


def load_routes(path, layers, tokens, experts, top):
    """Stream JSONL; keep only IDs, with complete, ordered capture geometry."""
    result = torch.empty(layers, tokens, top, dtype=torch.int64)
    count = 0
    with Path(path).open() as source:
        for count, line in enumerate(source, 1):
            row = json.loads(line)
            layer, token = divmod(count-1, tokens)
            ids = row['ids']
            if layer >= layers or (row['layer'], row['token']) != (layer, token):
                raise ValueError('native route capture must include every layer/token in order')
            if len(ids) != top or len(set(ids)) != top or any(type(e) is not int or not 0 <= e < experts for e in ids):
                raise ValueError('invalid native expert IDs')
            result[layer, token] = torch.tensor(ids)
    if count != layers*tokens:
        raise ValueError('incomplete native route capture')
    return result


@torch.inference_mode()
def fp64_router(layer, hidden, pe, rows):
    if hidden.ndim != 3 or hidden.shape[0] != 1:
        raise ValueError('FP64 arbitration requires one unpadded record')
    if rows.numel() == 0 or int(rows.min()) < 0 or int(rows.max()) >= hidden.shape[1]:
        raise ValueError('invalid arbitration query rows')
    end = int(rows.max())+1
    # MPS .to(device='cpu', dtype=float64) may cast before moving and fail;
    # copy first so every FP64 operation is unambiguously on the CPU.
    h = hidden[0, :end].detach().cpu().double()
    a = layer.self_attn
    hd = a.head_dim
    nh, nkv = a.q_proj.weight.shape[0]//hd, a.k_proj.weight.shape[0]//hd
    if any(p.bias is not None for p in (a.q_proj, a.k_proj, a.v_proj, a.o_proj)):
        raise ValueError('FP64 arbitration does not support attention biases')

    def norm(x, module):
        w = module.weight.detach().cpu().double()
        return x*torch.rsqrt(x.square().mean(-1, keepdim=True)+module.variance_epsilon)*w

    def linear(x, weight):
        return x@weight.detach().cpu().double().T

    x = norm(h, layer.input_layernorm)
    q = norm(linear(x[rows], a.q_proj.weight).view(-1,nh,hd), a.q_norm).transpose(0,1)
    k = norm(linear(x, a.k_proj.weight).view(-1,nkv,hd), a.k_norm).transpose(0,1)
    v = linear(x, a.v_proj.weight).view(-1,nkv,hd).transpose(0,1)
    # Position embeddings are inputs to the layer, already computed by the pinned
    # rotary implementation. Preserve those values when raising arithmetic precision.
    co, si = (p[0,:end].detach().cpu().double() for p in pe)
    def rotary(x, c, s):
        rotate = torch.cat((-x[...,hd//2:], x[...,:hd//2]), -1)
        return x*c+rotate*s
    q, k = rotary(q, co[rows], si[rows]), rotary(k, co, si)
    result = torch.empty(len(rows), nh, hd, dtype=torch.float64)
    rep = nh//nkv
    keys = torch.arange(end)
    # Bound score storage even when a long record has many ambiguous tokens.
    for start in range(0, len(rows), 16):
        stop = min(start+16, len(rows))
        mask = keys[None,:] > rows[start:stop,None]
        for head in range(nkv):
            scores = q[head*rep:(head+1)*rep,start:stop]@k[head].T*a.scaling
            scores.masked_fill_(mask, -torch.inf)
            values = scores.softmax(-1)@v[head]
            result[start:stop,head*rep:(head+1)*rep] = values.transpose(0,1)
    residual = h[rows]+linear(result.reshape(len(rows),-1), a.o_proj.weight)
    return linear(norm(residual, layer.post_attention_layernorm), layer.mlp.gate.weight)


@torch.inference_mode()
def fp64_native_router(layer, reference, pe, rows, native_input, *, max_relative_l2=.01):
    """Check the incoming states before using native inputs to resolve a tie."""
    native = torch.from_numpy(native_input.copy()).unsqueeze(0)
    source = reference.detach().cpu()
    if native.shape != source.shape or not torch.isfinite(native).all() or not torch.isfinite(source).all():
        raise ValueError('invalid native/reference incoming activation capture')
    if not 0 < max_relative_l2 <= .01:
        raise ValueError('invalid incoming activation tolerance')
    expected = source.double()
    delta = native.double()-expected
    relative = float(delta.norm()/expected.norm().clamp_min(1e-30))
    row_relative = delta.norm(dim=-1)/expected.norm(dim=-1).clamp_min(1e-30)
    worst = float(row_relative.max())
    if relative > max_relative_l2 or worst > max_relative_l2:
        raise ValueError(f'incoming activation error exceeds layer tolerance: relative_L2={relative}, max_row={worst}')
    metrics = dict(incoming_relative_l2=relative, incoming_max_row_relative_l2=worst,
                   incoming_max_absolute=float(delta.abs().max()), incoming_tolerance=max_relative_l2)
    return fp64_router(layer,native,pe,rows),metrics


def input_shift_eligibility(reference32, reference64, native64, native_ids, incoming):
    """Opt-in eligibility audit for independently measured incoming-state shifts.

    Bound each lost/gained pair's reference gap by the independently computed
    change in that pair of scores plus the existing
    FP32 allowance. This cannot authorize a choice rejected by native-input
    FP64, or relax any incoming-state or final-output limit.
    """
    vectors=[v.detach().cpu().double() for v in (reference32,reference64,native64)]
    if any(v.ndim!=1 or v.shape!=vectors[0].shape or not torch.isfinite(v).all() for v in vectors):
        raise ValueError('invalid input-shift score geometry or values')
    reference32,reference64,native64=vectors
    ids=native_ids.detach().cpu().tolist();top=len(ids)
    if not 0<top<len(reference64) or len(set(ids))!=top or any(type(i)!=int or not 0<=i<len(reference64) for i in ids):
        raise ValueError('invalid input-shift expert IDs')
    for key in ('incoming_relative_l2','incoming_max_row_relative_l2'):
        value=incoming.get(key)
        if not isinstance(value,(int,float)) or not 0<=value<=.01:
            raise ValueError('input-shift incoming state exceeds existing layer limit')
    if incoming.get('incoming_tolerance')!=.01:raise ValueError('input-shift requires the existing layer limit')
    native_order=native64.argsort(descending=True,stable=True)
    reference_order=reference64.argsort(descending=True,stable=True)
    if set(native_order[:top].tolist())!=set(ids):raise ValueError('native-input FP64 rejects input-shift choice')
    reference_ids=set(reference_order[:top].tolist());native_set=set(ids)
    lost=[i for i in reference_order[:top].tolist() if i not in native_set]
    gained=[i for i in native_order[:top].tolist() if i not in reference_ids]
    if not lost:raise ValueError('input-shift rule requires a change caused by the input')
    window=64*torch.finfo(torch.float32).eps*max(1,float(reference64.abs().max()))
    reference_error=float((reference32-reference64).abs().max())
    if reference_error>window:raise ValueError('reference FP32 error exceeds existing rounding window')
    shift=native64-reference64
    pairs=[]
    # A gained expert crosses each lost expert. The old kth expert can remain
    # selected (or leave with another expert), so its ordering need not change.
    # Audit every actual membership crossing, with no change to the window.
    for departing in lost:
        for entering in gained:
            gap=float(reference64[departing]-reference64[entering])
            bound=float(shift[departing].abs()+shift[entering].abs())+window
            if gap>bound:raise ValueError('reference gap exceeds independent input-shift bound')
            pairs.append(dict(lost=departing,gained=entering,gap=gap,bound=bound,
                              reference_lost=float(reference64[departing]),reference_gained=float(reference64[entering]),
                              native_lost=float(native64[departing]),native_gained=float(native64[entering]),
                              lost_shift=float(shift[departing]),gained_shift=float(shift[entering]),
                              native_margin=float(native64[entering]-native64[departing])))
    return dict(rounding_window=window,reference_fp32_error=reference_error,disputed_pairs=pairs)


@torch.inference_mode()
def arbitrate(output, native, calculate, events, *, layer, record, calculate_native=None,
              calculate_rounding=None, allow_input_shift=False):
    logits, weights, ids = output
    original = ids.detach().cpu()
    native = native.cpu()
    if native.shape != original.shape:
        raise ValueError('native/reference routing geometry differs')
    changed = torch.where((original.sort(-1).values != native.sort(-1).values).any(-1))[0]
    if not len(changed):
        return output
    values = logits.detach().cpu().double()
    top = ids.shape[-1]
    order = values.argsort(dim=-1, descending=True, stable=True)
    # This eligibility window is not an output tolerance. It prevents routing
    # replay from masking a clear disagreement; every eligible row still needs
    # an independent FP64 calculation that confirms the native expert set.
    window = 64*torch.finfo(torch.float32).eps*values.abs().amax(-1).clamp_min(1)
    outside = set()
    for row in changed.tolist():
        disputed = set(original[row].tolist()) ^ set(native[row].tolist())
        boundary = values[row,order[row,top-1]]
        if any(abs(float(values[row,e]-boundary)) > float(window[row]) for e in disputed):
            if not allow_input_shift or calculate_native is None:
                raise ValueError(f'{record} layer {layer} token {row}: routing difference is not a near tie')
            outside.add(row)
    high = calculate(changed).detach().cpu().double()
    if high.shape != (len(changed), logits.shape[1]) or not torch.isfinite(high).all():
        raise ValueError('invalid FP64 arbitration output')
    high_order = high.argsort(dim=-1, descending=True, stable=True)
    chosen = high_order[:,:top].clone()
    sensitive = [j for j,row in enumerate(changed.tolist()) if row in outside or set(chosen[j].tolist()) != set(native[row].tolist())]
    shifted = {}
    native_high, incoming = None, None
    if sensitive and calculate_native is not None:
        native_high,incoming = calculate_native(changed[sensitive])
        native_high = native_high.detach().cpu().double()
        if native_high.shape != (len(sensitive),logits.shape[1]) or not torch.isfinite(native_high).all():
            raise ValueError('invalid native-input FP64 arbitration output')
        native_order = native_high.argsort(dim=-1,descending=True,stable=True)
        for k,j in enumerate(sensitive):
            row = int(changed[j])
            disputed = set(chosen[j].tolist()) ^ set(native[row].tolist())
            boundary = high[j,high_order[j,top-1]]
            window64 = 64*torch.finfo(torch.float32).eps*max(1,float(high[j].abs().max()))
            if row in outside or any(abs(float(high[j,e]-boundary)) > window64 for e in disputed):
                if not allow_input_shift:
                    raise ValueError(f'{record} layer {layer} token {row}: FP64 disagreement is not input-sensitive near tie')
                # Widening eligibility requires native-input FP64 itself to
                # confirm the choice; the local-rounding exception cannot help.
                try:
                    shifted[j] = input_shift_eligibility(values[row],high[j],native_high[k],native[row],incoming)
                except ValueError as error:
                    # Retain the already computed evidence: a long reference
                    # need not repeat earlier layers merely to diagnose a gate.
                    evidence=dict(reference32=values[row].tolist(),reference64=high[j].tolist(),
                                  native64=native_high[k].tolist(),native_ids=native[row].tolist(),incoming=incoming)
                    raise ValueError(f'{record} layer {layer} token {row}: {error}; '
                                     f'input_shift_evidence={json.dumps(evidence)}') from error
            chosen[j] = native_order[k,:top]
    rounding = {}
    if native_high is not None and calculate_rounding is not None:
        pending = []
        for k,j in enumerate(sensitive):
            row = int(changed[j])
            if set(chosen[j].tolist()) == set(native[row].tolist()):
                continue
            disputed = sorted(set(chosen[j].tolist()) ^ set(native[row].tolist()))
            boundary = native_high[k,native_order[k,top-1]]
            b32 = boundary.float()
            # At a binade boundary use the smaller adjacent spacing. The whole
            # disputed range must fit, not just each expert's distance separately.
            ulp = min(float(torch.nextafter(b32,torch.tensor(float('inf'))).double()-b32.double()),
                      float(b32.double()-torch.nextafter(b32,torch.tensor(-float('inf'))).double()))
            scores = torch.cat((native_high[k,disputed],boundary[None]))
            gap = float(scores.max()-scores.min())
            if not 0 < ulp < float('inf') or gap > ulp:
                raise ValueError(f'{record} layer {layer} token {row}: local rounding gap {gap} exceeds one FP32 ULP {ulp}')
            rounding[j] = dict(rounding_gap=gap,rounding_ulp=ulp)
            pending.append(j)
        if pending:
            audited,reports = calculate_rounding(changed[pending])
            audited = audited.detach().cpu().double()
            if audited.shape != (len(pending),logits.shape[1]) or not torch.isfinite(audited).all() or len(reports) != len(pending):
                raise ValueError('invalid local-rounding FP64 audit output')
            for k,j in enumerate(pending):
                chosen[j] = audited[k].argsort(descending=True,stable=True)[:top]
                rounding[j]['rounding_audit'] = reports[k]
    for j, row in enumerate(changed.tolist()):
        if set(chosen[j].tolist()) != set(native[row].tolist()):
            raise ValueError(f'{record} layer {layer} token {row}: FP64 rejects native experts; '
                             f'native={native[row].tolist()}, FP64={chosen[j].tolist()}, '
                             f'original={original[row].tolist()}, '
                             f'FP32_margin={float(values[row,order[row,top-1]]-values[row,order[row,top]])}, '
                             f'FP64_margin={float(high[j,high_order[j,top-1]]-high[j,high_order[j,top]])}')
        events.append(dict(record=record, layer=layer, token=row,
                           method='local-rounding' if j in rounding else (
                               'input-sensitive' if j in sensitive and native_high is not None else 'reference-input'),
                           original_ids=original[row].tolist(), native_ids=native[row].tolist(),
                           fp64_ids=chosen[j].tolist(),
                           reference_fp64_ids=high_order[j,:top].tolist(),
                           original_margin=float(values[row,order[row,top-1]]-values[row,order[row,top]]),
                           fp64_margin=float(high[j,high_order[j,top-1]]-high[j,high_order[j,top]]),
                           max_fp32_fp64_logit_error=float((high[j]-values[row]).abs().max())))
        if j in sensitive and native_high is not None:
            k = sensitive.index(j)
            events[-1].update(incoming)
            events[-1]['native_input_fp64_ids'] = native_order[k,:top].tolist()
            events[-1]['native_input_fp64_margin'] = float(native_high[k,native_order[k,top-1]]-native_high[k,native_order[k,top]])
        if j in rounding:
            events[-1].update(rounding[j])
        events[-1]['eligibility'] = 'input-shift' if j in shifted else 'fixed-near-tie'
        if j in shifted:
            events[-1]['input_shift_audit'] = shifted[j]
    out_logits, out_weights, out_ids = logits.clone(), weights.clone(), ids.clone()
    dest = changed.to(ids.device)
    out_logits[dest] = high.to(logits.device, logits.dtype)
    out_ids[dest] = chosen.to(ids.device)
    out_weights[dest] = high.gather(1,chosen).softmax(-1).to(weights.device, weights.dtype)
    return out_logits, out_weights, out_ids
