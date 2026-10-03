"""Head-chunked SDPA for the oracles, working around a PyTorch MPS bug (see ref/mps_sdpa_bug.py):
torch.nn.functional.scaled_dot_product_attention on MPS returns wrong output for every head h
whose score-matrix offset h * T^2 is >= 2^32 (torch 2.11, causal). For Clef 27B (24 heads) that
is every request of >= 13,666 tokens. Each call here covers few enough heads to stay below 2^32;
GQA groups are kept whole. Mathematically identical to sdpa (heads are independent).
"""

import torch
from transformers import AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward

LIMIT = 2**32


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


def install(model) -> None:
    """model: a ClefModel (its HF backbone is model.language_model) or an HF model."""
    AttentionInterface.register("sdpa_safe", sdpa_safe)
    getattr(model, "language_model", model).set_attn_implementation("sdpa_safe")
