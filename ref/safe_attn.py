"""Explicit dense and Omni attention controls for the reference oracles.

The dense path preserves committed head-only splitting. Omni opts into bounded
query slices and blocked FP32 value sums; neither changes dense --safe-attn.

Both work around a PyTorch MPS bug (ref/mps_sdpa_bug.py):
torch.nn.functional.scaled_dot_product_attention on MPS returns wrong output for every head h
whose score-matrix offset h * T^2 is >= 2^32 (torch 2.11, causal). For Clef 27B (24 heads) that
is every request of >= 13,666 tokens. Each call here covers few enough heads to stay below 2^32;
GQA groups are kept whole. Query slices carry explicit global causal positions;
using is_causal=True on a later slice would incorrectly restart its diagonal at 0.
"""

import torch
from transformers import AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward

LIMIT = 2**32
# Also bound temporary score storage while the BF16 model occupies unified memory.
SCORE_ELEMENTS = 2**26
VALUE_BLOCK_THRESHOLD = 16384
VALUE_BLOCK = 1024


def sdpa_blocked_values(query, key, value, mask, *, scaling=None):
    """Keep FP32 weighted-value reductions bounded for long reference inputs."""
    # Broadcasting one KV head avoids eight copies in the common GQA chunk.
    if key.shape[1] not in (1,query.shape[1]):
        repeat=query.shape[1]//key.shape[1]
        key=key.repeat_interleave(repeat,dim=1)
        value=value.repeat_interleave(repeat,dim=1)
    scores=query@key.transpose(-1,-2)
    scores*=query.shape[-1]**-.5 if scaling is None else scaling
    if mask is not None:
        if mask.dtype==torch.bool:scores.masked_fill_(~mask,-torch.inf)
        else:scores+=mask
    probabilities=scores.softmax(-1)
    # SDPA defines fully masked rows as zero. Preserve other nonfinite values
    # so the oracle's finite-output gates still reject them.
    probabilities=probabilities.masked_fill(torch.isneginf(scores).all(-1,keepdim=True),0)
    length=value.shape[-2]
    padding=(-length)%VALUE_BLOCK
    if padding:
        probabilities=torch.nn.functional.pad(probabilities,(0,padding))
        value=torch.nn.functional.pad(value,(0,0,0,padding))
    B,H,Q,_=probabilities.shape
    blocks=(length+padding)//VALUE_BLOCK
    weights=probabilities.reshape(B,H,Q,blocks,VALUE_BLOCK).transpose(2,3)
    values=value.reshape(value.shape[0],value.shape[1],blocks,VALUE_BLOCK,value.shape[-1])
    # One batched matmul evaluates all key blocks. A short sum of partials
    # avoids the accumulated P@V error measured over 64k keys in stock FP32.
    return (weights@values).sum(dim=2).transpose(1,2)


def sdpa_safe(module, query, key, value, attention_mask, **kwargs):
    B, Hq, T, _ = query.shape
    S = key.shape[2]
    rep = Hq // key.shape[1]
    per_call = max(rep, (LIMIT - 1) // max(1, T * S) // rep * rep)   # q heads per call, whole GQA groups
    if per_call >= Hq:
        return sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)
    outs = []
    for h0 in range(0, Hq, per_call):
        h1 = min(Hq, h0 + per_call)
        o, _ = sdpa_attention_forward(module, query[:, h0:h1], key[:, h0 // rep:h1 // rep],
                                      value[:, h0 // rep:h1 // rep], attention_mask, **kwargs)
        outs.append(o)                                   # [B, T, heads, D]
    return torch.cat(outs, dim=2), None


def sdpa_omni(module, query, key, value, attention_mask, **kwargs):
    B, Hq, T, _ = query.shape
    S = key.shape[2]
    rep = Hq // key.shape[1]
    budget = min(LIMIT - 1, SCORE_ELEMENTS)
    reduction = kwargs.pop('_value_reduction', 'auto')
    blocked_values = query.dtype == torch.float32 and (reduction == 'blocked' or
                          reduction == 'auto' and S >= VALUE_BLOCK_THRESHOLD)
    if blocked_values and kwargs.get('dropout',0):
        raise ValueError('blocked reference attention requires inference dropout=0')
    if B * Hq * T * S <= budget and not blocked_values:
        return sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)
    per_call = min(Hq, max(rep, budget // max(1, B * T * S) // rep * rep))
    rows = min(T, budget // (B * per_call * S))
    if not rows:
        raise ValueError("safe attention offset budget cannot hold one GQA query row")
    causal = kwargs.get("is_causal")
    if causal is None:
        causal = getattr(module, "is_causal", True)
    causal = causal and T > 1 and attention_mask is None
    outs = []
    for h0 in range(0, Hq, per_call):
        h1 = min(Hq, h0 + per_call)
        chunks = []
        for q0 in range(0, T, rows):
            q1 = min(T, q0 + rows)
            mask = attention_mask
            if mask is not None:
                if mask.shape[-2] != 1:
                    mask = mask[..., q0:q1, :]
                if mask.ndim >= 3 and mask.shape[-3] == Hq:
                    mask = mask[..., h0:h1, :, :]
            elif causal:
                mask = torch.arange(S, device=query.device)[None, :] <= torch.arange(q0, q1, device=query.device)[:, None]
            q,k,v=query[:,h0:h1,q0:q1],key[:,h0//rep:h1//rep],value[:,h0//rep:h1//rep]
            if blocked_values:
                o=sdpa_blocked_values(q,k,v,mask,scaling=kwargs.get('scaling'))
            else:
                o, _ = sdpa_attention_forward(module,q,k,v,mask,**{**kwargs,"is_causal":False})
            chunks.append(o)
        outs.append(torch.cat(chunks, dim=1))             # [B, T, heads, D]
    return torch.cat(outs, dim=2), None


def sdpa_omni_blocked(module, query, key, value, attention_mask, **kwargs):
    """Independent FP32 reference: use blocked P@V at every sequence length."""
    return sdpa_omni(module, query, key, value, attention_mask, **{**kwargs, '_value_reduction': 'blocked'})


def sdpa_omni_unblocked(module, query, key, value, attention_mask, **kwargs):
    """FP32 noise-floor control with the same bounded calls and SDPA's P@V."""
    return sdpa_omni(module, query, key, value, attention_mask, **{**kwargs, '_value_reduction': 'sdpa'})


def install(model, *, omni=False, value_reduction='auto') -> None:
    """model: a ClefModel (its HF backbone is model.language_model) or an HF model."""
    name, implementation = ("sdpa_omni", sdpa_omni) if omni else ("sdpa_safe", sdpa_safe)
    if value_reduction not in ('auto', 'blocked', 'sdpa') or not omni and value_reduction != 'auto':
        raise ValueError('explicit value reduction requires an Omni reference')
    if value_reduction != 'auto':
        name, implementation = (('sdpa_omni_blocked', sdpa_omni_blocked) if value_reduction == 'blocked'
                                else ('sdpa_omni_unblocked', sdpa_omni_unblocked))
    AttentionInterface.register(name, implementation)
    getattr(model, "thinker", getattr(model, "language_model", model)).set_attn_implementation(name)
