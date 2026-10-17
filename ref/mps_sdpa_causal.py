"""Model-free causal SDPA audit. Run under the GPU lock and memory guard.

The independent CPU FP64 oracle uses the same rounded operands as each backend.
Changing future values must leave earlier outputs unchanged. This distinguishes
causal-mask leakage from ordinary BF16 rounding without loading model weights.
An incorrect backend produces a complete report and exit status 1.
"""
import argparse
import datetime
import hashlib
import json
import platform
from pathlib import Path

import torch
import torch.nn.functional as F


def reference(q, k, v, causal):
    q, k, v = (x.cpu().double() for x in (q, k, v))
    scores = (q @ k.transpose(-1, -2)) * q.shape[-1] ** -0.5
    if causal:
        mask = torch.ones(scores.shape[-2:], dtype=torch.bool).tril()
        scores.masked_fill_(~mask, -torch.inf)
    return scores.softmax(-1) @ v


def relative(actual, expected):
    actual = actual.cpu().double()
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise ValueError("non-finite attention output")
    error = float((actual - expected).norm() / expected.norm())
    if not torch.isfinite(torch.tensor(error, dtype=torch.float64)):
        raise ValueError("non-finite attention error")
    return error


@torch.inference_mode()
def run_case(device, dtype, head_dim, tokens=512, heads=8):
    # Generate on CPU for reproducibility across PyTorch/MPS versions. Round on
    # CPU before transfer so the FP64 oracle sees the exact backend operands.
    gen = torch.Generator().manual_seed(0)
    cpu = [torch.randn(1, heads, tokens, head_dim, generator=gen).to(dtype)
           for _ in range(3)]
    causal, bidirectional = reference(*cpu, True), reference(*cpu, False)
    q, k, v = (x.to(device) for x in cpu)
    mask = torch.ones(tokens, tokens, dtype=torch.bool, device=device).tril()
    implicit = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    explicit = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)
    unmasked = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    # A tile-aligned midpoint alone can miss leakage within an early tile.
    # Include the first future key and several unaligned prefix boundaries.
    changes = []
    for cut in sorted({c for c in (1, 2, 7, 17, tokens // 2) if c < tokens}):
        changed = v.clone()
        changed[:, :, cut:] += 4
        implicit_changed = F.scaled_dot_product_attention(q, k, changed, is_causal=True)
        explicit_changed = F.scaled_dot_product_attention(q, k, changed, attn_mask=mask, is_causal=False)
        changes.append({"cut": cut,
            "implicit_max_change": float((implicit[:, :, :cut].float() -
                                           implicit_changed[:, :, :cut].float()).abs().max()),
            "explicit_max_change": float((explicit[:, :, :cut].float() -
                                           explicit_changed[:, :, :cut].float()).abs().max())})
    leak = max(c["implicit_max_change"] for c in changes)
    control_leak = max(c["explicit_max_change"] for c in changes)
    errors = {"implicit_causal": relative(implicit, causal),
              "explicit_causal": relative(explicit, causal),
              "unmasked": relative(unmasked, bidirectional)}
    # Fixed diagnostic limits, not the model's qualification policy. These allow
    # ordinary output rounding while rejecting the observed ~0.4 relative error.
    limit = {torch.float32: 1e-5, torch.float16: .002, torch.bfloat16: .02}[dtype]
    row_error = ((implicit.cpu().double() - causal).norm(dim=-1) /
                 causal.norm(dim=-1)).mean(dim=(0, 1))
    return {"dtype": str(dtype), "head_dim": head_dim, "tokens": tokens, "heads": heads,
            "relative_l2_limit": limit, "relative_l2": errors,
            "implicit_prefix_max_change": leak, "explicit_prefix_max_change": control_leak,
            "prefix_changes": changes,
            "prefix_change_limit": 0.0,
            "implicit_row_relative_l2": {str(i): float(row_error[i])
                                         for i in (0, 1, 2, 10, 100, tokens - 1) if i < tokens},
            "controls_pass": errors["explicit_causal"] <= limit and
                             errors["unmasked"] <= limit and control_leak == 0,
            "implicit_pass": errors["implicit_causal"] <= limit and leak == 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; no audit was run")
    cases = [run_case(args.device, dtype, dim)
             for dim in (64, 128, 256)
             for dtype in (torch.float32, torch.float16, torch.bfloat16)]
    valid = all(c["implicit_pass"] and c["controls_pass"] for c in cases)
    report = {"complete": True, "causal_backend_valid": valid, "device": args.device,
              "torch": torch.__version__, "torch_git_version": torch.version.git_version,
              "platform": platform.platform(), "python": platform.python_version(),
              "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "seed": 0, "reference": "CPU FP64 on dtype-rounded inputs", "cases": cases}
    with args.output.open("x") as out:
        out.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for case in cases:
        print(case["dtype"], case["head_dim"], case["relative_l2"],
              "future-value leakage", case["implicit_prefix_max_change"])
    raise SystemExit(0 if valid else 1)


if __name__ == "__main__":
    main()
