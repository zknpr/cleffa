"""Minimal reproduction: torch.nn.functional.scaled_dot_product_attention on MPS returns wrong
results for every head h whose score-matrix offset h * T^2 is >= 2^32 elements (causal, f32):
a 32-bit offset overflow.

Random q/k/v, no model. Prints per-head relative error of MPS vs CPU for a few sizes.
Observed with torch 2.11.0 on an M5 Max / macOS 27.
"""
import torch
import torch.nn.functional as F

torch.manual_seed(0)
for heads, T in ((16, 16347), (17, 16347), (24, 13400), (24, 13660), (24, 13670), (24, 16347)):
    q = torch.randn(1, heads, T, 64)
    k = torch.randn(1, heads, T, 64)
    v = torch.randn(1, heads, T, 64)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    out = F.scaled_dot_product_attention(q.to("mps"), k.to("mps"), v.to("mps"), is_causal=True).cpu()
    err = ((out - ref).flatten(2).norm(dim=-1) / ref.flatten(2).norm(dim=-1))[0]
    bad = [h for h in range(heads) if err[h] > 1e-3]
    predicted = [h for h in range(heads) if h * T * T >= 2**32]
    del q, k, v, ref, out
    torch.mps.empty_cache()
    print(f"heads={heads:2d} T={T:5d}: wrong heads {bad or 'none'}; predicted (h*T^2 >= 2^32) {predicted or 'none'}; "
          f"{'MATCH' if bad == predicted else 'MISMATCH'}")
