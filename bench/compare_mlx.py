"""Compare Cleffa's GEMM algorithm with MLX, using identical 16-bit operands.

Optional dependencies: numpy and mlx==0.32.3, installed in a venv. This does not
load a model or measure complete requests. All paths return FP32 outputs.

PYTHONPATH=golden/mlx-env .venv/bin/python -B bench/compare_mlx.py \
    4510 5120 34816 --output golden/mlx-comparison.json
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import time

import mlx.core as mx
import numpy as np


def positive_dimension(value):
    value = int(value)
    if not 1 <= value <= 65536:
        raise argparse.ArgumentTypeError("dimensions must be between 1 and 65536")
    return value


def values(count, offset=0):
    """The same deterministic inputs as bench/gemm_tiles.m."""
    x = np.arange(offset, offset + count, dtype=np.uint32)
    x ^= x >> 16
    x *= np.uint32(0x7FEB352D)
    x ^= x >> 15
    x *= np.uint32(0x846CA68B)
    x ^= x >> 16
    return (x & 65535).astype(np.float32) / 32768 - 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tokens", type=positive_dimension)
    parser.add_argument("input_width", type=positive_dimension)
    parser.add_argument("output_width", type=positive_dimension)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    t, k, n = args.tokens, args.input_width, args.output_width

    x_host = values(t * k)
    x_host[::67] *= 8
    x_host = x_host.astype(np.float16).reshape(t, k)
    w_host = values(n * k, 123) * np.float32(0.02)
    bits = w_host.view(np.uint32)
    bits += 32767 + ((bits >> 16) & 1)  # BF16 round-to-nearest, ties-to-even
    bits &= np.uint32(0xFFFF0000)
    w_host = bits.view(np.float32).reshape(n, k)

    mx.set_cache_limit(1024 * 1024 * 1024)
    x = mx.array(x_host)
    weight = mx.array(w_host).astype(mx.bfloat16)
    xf, wf = x.astype(mx.float32), weight.astype(mx.float32)
    shape = mx.array([t, n, k], dtype=mx.int32)
    mx.eval(x, weight, xf, wf, shape)

    # Match gemm() in clef_metal.m. Using a custom MLX kernel puts the MPP and
    # native MLX paths behind the same Python/lazy-evaluation timing boundary.
    tm = 64 if t >= 1024 else 32
    tn = 256 if t < 1024 and k >= 5120 and n >= 5120 else 128
    header = """
    #include <metal_tensor>
    #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
    using namespace metal;
    using namespace mpp::tensor_ops;
    """
    source = f"""
    auto tg = threadgroup_position_in_grid;
    int t = shape[0], n = shape[1], k = shape[2];
    auto tx = tensor<device half, dextents<int32_t, 2>, tensor_inline>(
        (device half *)X, dextents<int32_t, 2>(k, t));
    auto tw = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(
        (device bfloat *)W, dextents<int32_t, 2>(k, n));
    auto ty = tensor<device float, dextents<int32_t, 2>, tensor_inline>(
        Y, dextents<int32_t, 2>(n, t));
    matmul2d<matmul2d_descriptor({tm}, {tn}, dynamic_length_v<int>, false, true, false),
             execution_simdgroups<4>> mm;
    auto ax = tx.slice(0, (int)tg.y * {tm});
    auto aw = tw.slice(0, (int)tg.x * {tn});
    auto ay = ty.slice((int)tg.x * {tn}, (int)tg.y * {tm});
    mm.run(ax, aw, ay);
    """
    kernel = mx.fast.metal_kernel(
        name=f"clef_gemm_{tm}_{tn}", input_names=["X", "W", "shape"],
        output_names=["Y"], source=source, header=header,
        compile_options={"math_mode": "safe"},
    )

    def mpp():
        return kernel(
            inputs=[x, weight, shape],
            grid=(((n + tn - 1) // tn) * 128, (t + tm - 1) // tm, 1),
            threadgroup=(128, 1, 1), output_shapes=[(t, n)],
            output_dtypes=[mx.float32],
        )[0]

    methods = {
        "MPP": mpp,
        "MLX_mixed": lambda: mx.matmul(x, weight.T),
        "MLX_precast": lambda: mx.matmul(xf, wf.T),
    }
    samples = {name: [] for name in methods}
    outputs = {}
    order = list(methods)
    # Four warm-ups per method, eight timed calls. Reverse order each round to
    # reduce ordering bias; conversions for MLX_precast are outside this loop.
    for rep in range(12):
        for name in order if rep % 2 == 0 else order[::-1]:
            mx.synchronize()
            start = time.perf_counter()
            z = methods[name]()
            mx.eval(z)
            mx.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            if rep >= 4:
                samples[name].append(elapsed)
            outputs[name] = z

    report = {
        "T": t, "K": k, "N": n, "tile": [tm, tn],
        "mlx_version": importlib.metadata.version("mlx"),
        "device": mx.device_info(),
        "tf32": os.environ.get("MLX_ENABLE_TF32", "default"),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "samples_ms": samples,
        "median_ms": {name: statistics.median(ms) for name, ms in samples.items()},
        "dtype": {name: str(z.dtype) for name, z in outputs.items()},
        "accuracy": {},
    }
    baseline = np.array(outputs["MPP"])
    for name, z in outputs.items():
        if z.dtype != mx.float32:
            raise RuntimeError(f"{name} returned {z.dtype}, expected float32")
        actual = np.array(z)
        if not np.isfinite(actual).all():
            raise RuntimeError(f"{name} returned nonfinite output")
        worst = 0
        for i in range(64):
            row = t - 1 if i == 0 else i * 7919 % t
            col = n - 1 if i == 0 else i * 104729 % n
            terms = x_host[row].astype(np.float64) * w_host[col].astype(np.float64)
            error = abs(float(actual[row, col]) - terms.sum()) / (abs(terms).sum() + 1e-30)
            worst = max(worst, error)
        report["accuracy"][name] = {
            "max_sampled_error_over_sum_abs": worst,
            "max_abs_vs_MPP": float(np.max(np.abs(actual - baseline))),
            "different_elements": int(np.count_nonzero(
                actual.view(np.uint32) != baseline.view(np.uint32))),
        }
        if worst > 2e-6:
            raise RuntimeError(f"{name} exceeded the sampled float64 error bound: {worst}")

    # TF32 can represent these already-rounded FP16/BF16 operands. This test
    # does not establish its accuracy for arbitrary, unrounded FP32 inputs.
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "samples_ms"}))


if __name__ == "__main__":
    main()
