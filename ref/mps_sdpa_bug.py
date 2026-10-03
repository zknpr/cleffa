"""Minimal reproduction: torch.nn.functional.scaled_dot_product_attention on MPS returns wrong
results for every head h whose score-matrix offset h * T^2 is >= 2^32 elements (causal, f32):
a 32-bit offset overflow.

Random q/k/v, no model. Prints per-head relative error of MPS vs CPU for a few sizes. The MPS
call runs on all heads at once (that is what overflows); the CPU reference runs one head at a
time so its memory stays near T^2 floats. Each size runs in its own process: within one process
the earlier sizes' MPS memory was not returned (empty_cache did not help) and the largest size
then ran out of memory.
Observed with torch 2.11.0 on an M5 Max / macOS 27.0.1. Known upstream as pytorch/pytorch#179352,
closed as an Apple MPSGraph bug (Apple FB22437937); this script pins the per-head boundary.

Usage: mps_sdpa_bug.py            (all sizes)
       mps_sdpa_bug.py HEADS T    (one size)
"""
import subprocess
import sys

CASES = ((16, 16347), (17, 16347), (24, 13400), (24, 13660), (24, 13670), (24, 16347))


def run(heads: int, T: int) -> None:
    import torch
    import torch.nn.functional as F

    torch.manual_seed(0)
    q = torch.randn(1, heads, T, 64)
    k = torch.randn(1, heads, T, 64)
    v = torch.randn(1, heads, T, 64)
    out = F.scaled_dot_product_attention(q.to("mps"), k.to("mps"), v.to("mps"), is_causal=True).cpu()
    err = []
    for h in range(heads):
        ref = F.scaled_dot_product_attention(q[:, h:h + 1], k[:, h:h + 1], v[:, h:h + 1], is_causal=True)
        err.append(float((out[:, h:h + 1] - ref).norm() / ref.norm()))
    bad = [h for h in range(heads) if err[h] > 1e-3]
    predicted = [h for h in range(heads) if h * T * T >= 2**32]
    worst = max(err[h] for h in bad) if bad else max(err)
    print(f"heads={heads:2d} T={T:5d}: wrong heads {bad or 'none'}; predicted (h*T^2 >= 2^32) {predicted or 'none'}; "
          f"{'MATCH' if bad == predicted else 'MISMATCH'}; max rel err {worst:.2e}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        run(int(sys.argv[1]), int(sys.argv[2]))
    else:
        rc = 0
        for heads, T in CASES:
            rc |= subprocess.run([sys.executable, __file__, str(heads), str(T)]).returncode
        sys.exit(rc)
