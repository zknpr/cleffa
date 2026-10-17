// Omni expert GEMM operand schemes at the decoder's shapes: 128 experts, each one ragged run of
// R rows (a 32-row tile) against its own BF16 weights, as omni_expert_gemm dispatches them at short
// lengths. Compares FP32 operands (the engine), plain FP16, and two compensated FP16 schemes
// (hi = FP16(x), lo = residual): two products into cooperative tensors combined as hi + lo/2048
// (vis_gemm_comp's form), and two direct runs into Y (lo unscaled, accumulate mode). Reports time,
// effective TFLOPS, weight GB/s and error against an f64 dot of the unrounded activations.
//
// build: clang -O2 -fobjc-arc bench/expert_bench.m -framework Metal -framework Foundation -o expert-bench
// usage: expert-bench [R]   (rows per expert, default 19: 298 tokens * 8 / 128)
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static NSString *kSource = @
"#include <metal_stdlib>\n"
"#include <metal_tensor>\n"
"#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
"using namespace metal;\n"
"using namespace mpp::tensor_ops;\n"
"struct args { int R, N, K, E; };\n"
"template <typename XT>\n"
"kernel void plain(constant args &a [[buffer(0)]], device XT *X [[buffer(1)]], device bfloat *W [[buffer(2)]],\n"
"                  device float *Y [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]]) {\n"
"    const int e = tg.y;\n"
"    auto tX = tensor<device XT, dextents<int32_t, 2>, tensor_inline>(X + (long)e*32*a.K, dextents<int32_t, 2>(a.K, a.R));\n"
"    auto tW = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(W + (long)e*a.N*a.K, dextents<int32_t, 2>(a.K, a.N));\n"
"    auto tY = tensor<device float, dextents<int32_t, 2>, tensor_inline>(Y + (long)e*32*a.N, dextents<int32_t, 2>(a.N, a.R));\n"
"    matmul2d<matmul2d_descriptor(32, 128, dynamic_length_v<int>, false, true, false), execution_simdgroups<4>> mm;\n"
"    auto mX = tX.slice(0, 0); auto mW = tW.slice(0, (int)tg.x*128); auto mY = tY.slice((int)tg.x*128, 0);\n"
"    mm.run(mX, mW, mY);\n"
"}\n"
"template [[host_name(\"plain_float\")]] kernel void plain<float>(constant args &, device float *, device bfloat *, device float *, uint2);\n"
"template [[host_name(\"plain_half\")]] kernel void plain<half>(constant args &, device half *, device bfloat *, device float *, uint2);\n"
"kernel void stream(constant args &a [[buffer(0)]], device const uint4 *W [[buffer(2)]], device uint *Y [[buffer(3)]],\n"
"                   uint gid [[thread_position_in_grid]], uint n [[threads_per_grid]]) {\n"
"    const ulong total = (ulong)a.E*a.N*a.K*2/16; uint4 acc = 0;\n"
"    for (ulong i = gid; i < total; i += n) acc ^= W[i];\n"
"    if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x9e3779b9u) Y[gid] = 1;\n"
"}\n"
"kernel void plain_float64(constant args &a [[buffer(0)]], device float *X [[buffer(1)]], device bfloat *W [[buffer(2)]],\n"
"                  device float *Y [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]]) {\n"
"    const int e = tg.y;\n"
"    auto tX = tensor<device float, dextents<int32_t, 2>, tensor_inline>(X + (long)e*32*a.K, dextents<int32_t, 2>(a.K, a.R));\n"
"    auto tW = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(W + (long)e*a.N*a.K, dextents<int32_t, 2>(a.K, a.N));\n"
"    auto tY = tensor<device float, dextents<int32_t, 2>, tensor_inline>(Y + (long)e*32*a.N, dextents<int32_t, 2>(a.N, a.R));\n"
"    matmul2d<matmul2d_descriptor(32, 64, dynamic_length_v<int>, false, true, false), execution_simdgroups<4>> mm;\n"
"    auto mX = tX.slice(0, 0); auto mW = tW.slice(0, (int)tg.x*64); auto mY = tY.slice((int)tg.x*64, 0);\n"
"    mm.run(mX, mW, mY);\n"
"}\n"
"kernel void splitk(constant args &a [[buffer(0)]], device float *X [[buffer(1)]], device bfloat *W [[buffer(2)]],\n"
"                   device float *Y [[buffer(3)]], constant int &S [[buffer(4)]], uint3 tg [[threadgroup_position_in_grid]]) {\n"
"    const int e = tg.y, s = tg.z, ks = a.K / S;\n"
"    auto tX = tensor<device float, dextents<int32_t, 2>, tensor_inline>(X + (long)e*32*a.K + s*ks, dextents<int32_t, 2>(ks, a.R), array<int32_t, 2>{1, a.K});\n"
"    auto tW = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(W + (long)e*a.N*a.K + s*ks, dextents<int32_t, 2>(ks, a.N), array<int32_t, 2>{1, a.K});\n"
"    auto tY = tensor<device float, dextents<int32_t, 2>, tensor_inline>(Y + ((long)s*a.E + e)*32*a.N, dextents<int32_t, 2>(a.N, a.R));\n"
"    matmul2d<matmul2d_descriptor(32, 128, dynamic_length_v<int>, false, true, false), execution_simdgroups<4>> mm;\n"
"    auto mX = tX.slice(0, 0); auto mW = tW.slice(0, (int)tg.x*128); auto mY = tY.slice((int)tg.x*128, 0);\n"
"    mm.run(mX, mW, mY);\n"
"}\n"
"kernel void comp_coop(constant args &a [[buffer(0)]], device half *X [[buffer(1)]], device bfloat *W [[buffer(2)]],\n"
"                      device float *Y [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]]) {\n"
"    const int e = tg.y; const long lo = (long)a.E*32*a.K;\n"
"    auto tX = tensor<device half, dextents<int32_t, 2>, tensor_inline>(X + (long)e*32*a.K, dextents<int32_t, 2>(a.K, a.R));\n"
"    auto tL = tensor<device half, dextents<int32_t, 2>, tensor_inline>(X + lo + (long)e*32*a.K, dextents<int32_t, 2>(a.K, a.R));\n"
"    auto tW = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(W + (long)e*a.N*a.K, dextents<int32_t, 2>(a.K, a.N));\n"
"    auto tY = tensor<device float, dextents<int32_t, 2>, tensor_inline>(Y + (long)e*32*a.N, dextents<int32_t, 2>(a.N, a.R));\n"
"    matmul2d<matmul2d_descriptor(32, 128, dynamic_length_v<int>, false, true, false), execution_simdgroups<4>> mm;\n"
"    auto mX = tX.slice(0, 0), mL = tL.slice(0, 0); auto mW = tW.slice(0, (int)tg.x*128); auto mY = tY.slice((int)tg.x*128, 0);\n"
"    auto hi = mm.template get_destination_cooperative_tensor<decltype(mX), decltype(mW), float>();\n"
"    auto lw = mm.template get_destination_cooperative_tensor<decltype(mX), decltype(mW), float>();\n"
"    mm.run(mX, mW, hi); mm.run(mL, mW, lw);\n"
"    for (uint16_t i = 0; i < hi.get_capacity(); i++) if (hi.is_valid_element(i)) hi[i] += lw[i] * (1.0f/2048.0f);\n"
"    hi.store(mY);\n"
"}\n"
"kernel void comp_two(constant args &a [[buffer(0)]], device half *X [[buffer(1)]], device bfloat *W [[buffer(2)]],\n"
"                     device float *Y [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]]) {\n"
"    const int e = tg.y; const long lo = (long)a.E*32*a.K;\n"
"    auto tX = tensor<device half, dextents<int32_t, 2>, tensor_inline>(X + (long)e*32*a.K, dextents<int32_t, 2>(a.K, a.R));\n"
"    auto tL = tensor<device half, dextents<int32_t, 2>, tensor_inline>(X + lo + (long)e*32*a.K, dextents<int32_t, 2>(a.K, a.R));\n"
"    auto tW = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(W + (long)e*a.N*a.K, dextents<int32_t, 2>(a.K, a.N));\n"
"    auto tY = tensor<device float, dextents<int32_t, 2>, tensor_inline>(Y + (long)e*32*a.N, dextents<int32_t, 2>(a.N, a.R));\n"
"    matmul2d<matmul2d_descriptor(32, 128, dynamic_length_v<int>, false, true, false), execution_simdgroups<4>> mm;\n"
"    matmul2d<matmul2d_descriptor(32, 128, dynamic_length_v<int>, false, true, false, matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroups<4>> ma;\n"
"    auto mW = tW.slice(0, (int)tg.x*128); auto mY = tY.slice((int)tg.x*128, 0);\n"
"    auto mX = tX.slice(0, 0), mL = tL.slice(0, 0);\n"
"    mm.run(mX, mW, mY); ma.run(mL, mW, mY);\n"
"}\n";

static uint16_t f2bf(float f) { uint32_t u; memcpy(&u, &f, 4); u += 0x7fff + ((u >> 16) & 1); return (uint16_t)(u >> 16); }
static float bf2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }
static double gauss(void) {
    double u = ((double)rand() + 1) / ((double)RAND_MAX + 2), v = (double)rand() / RAND_MAX;
    return sqrt(-2 * log(u)) * cos(2 * M_PI * v);
}
typedef struct { int R, N, K, E; } args;

int main(int argc, char **argv) {
    @autoreleasepool {
        const int E = 128, R = argc > 1 ? atoi(argv[1]) : 19;
        if (R < 1 || R > 32) { fprintf(stderr, "R must be 1..32\n"); return 2; }
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        NSError *err = nil;
        MTLCompileOptions *opt = [MTLCompileOptions new]; opt.mathMode = MTLMathModeSafe;
        id<MTLLibrary> lib = [dev newLibraryWithSource:kSource options:opt error:&err];
        if (!lib) { fprintf(stderr, "compile: %s\n", err.localizedDescription.UTF8String); return 1; }
        id<MTLCommandQueue> q = [dev newCommandQueue];
        const int shapes[][2] = { {2048, 1536}, {768, 2048} };   // gate_up, down
        const char *names[] = { "plain_float", "plain_half", "comp_coop", "comp_two", "plain_float64", "stream", "splitk", "splitk" };
        const char *labels[] = { "plain_float", "plain_half", "comp_coop", "comp_two", "float_32x64", "stream", "splitk2", "splitk4" };
        printf("R=%d rows per expert, %d experts\n%-12s %-12s %9s %8s %8s %11s %11s\n", R, E, "K->N", "scheme", "ms", "TFLOPS", "GB/s", "rms err", "max err");
        for (int si = 0; si < 2; si++) {
            const int K = shapes[si][0], N = shapes[si][1];
            srand(11);
            float *xf = malloc((size_t)E * 32 * K * 4);
            for (size_t i = 0; i < (size_t)E * 32 * K; i++) xf[i] = (float)(gauss() * (rand() % 64 == 0 ? 8.0 : 1.0));
            id<MTLBuffer> bw = [dev newBufferWithLength:(size_t)E * N * K * 2 options:MTLResourceStorageModeShared];
            uint16_t *w = bw.contents;
            for (size_t i = 0; i < (size_t)E * N * K; i++) w[i] = f2bf((float)(gauss() * 0.02));
            id<MTLBuffer> by = [dev newBufferWithLength:(size_t)4 * E * 32 * N * 4 options:MTLResourceStorageModeShared];
            enum { NS = 256 };
            int se_[NS], sr[NS], sn[NS]; double ref[NS], nrm[NS];
            for (int s = 0; s < NS; s++) {
                se_[s] = (s * 37) % E; sr[s] = (s * 7) % R; sn[s] = (s * 104729) % N;
                double r = 0, m = 0;
                for (int k = 0; k < K; k++) {
                    const double p = (double)xf[((size_t)se_[s] * 32 + sr[s]) * K + k] * bf2f(w[((size_t)se_[s] * N + sn[s]) * K + k]);
                    r += p; m += fabs(p);
                }
                ref[s] = r; nrm[s] = m;
            }
            for (int v = 0; v < 8; v++) {
                const bool comp = v == 2 || v == 3;
                const size_t elems = (size_t)E * 32 * K;
                id<MTLBuffer> bx = [dev newBufferWithLength:elems * (v == 0 || v >= 4 ? 4 : comp ? 4 : 2) options:MTLResourceStorageModeShared];
                for (size_t i = 0; i < elems; i++) {
                    if (v == 0 || v >= 4) ((float *)bx.contents)[i] = xf[i];
                    else if (v == 1) ((__fp16 *)bx.contents)[i] = (__fp16)xf[i];
                    else {
                        const __fp16 h = (__fp16)xf[i];
                        ((__fp16 *)bx.contents)[i] = h;
                        ((__fp16 *)bx.contents)[elems + i] = (__fp16)((xf[i] - (float)h) * (v == 2 ? 2048.0f : 1.0f));
                    }
                }
                id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@(names[v])] error:&err];
                if (!ps) { fprintf(stderr, "%s: %s\n", names[v], err.localizedDescription.UTF8String); return 1; }
                args a = { R, N, K, E };
                const int reps = 30, warm = 10;
                double total = 0;
                for (int r = 0; r < reps; r++) {
                    id<MTLCommandBuffer> cb = [q commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    [enc setComputePipelineState:ps];
                    [enc setBytes:&a length:sizeof(a) atIndex:0];
                    [enc setBuffer:bx offset:0 atIndex:1]; [enc setBuffer:bw offset:0 atIndex:2]; [enc setBuffer:by offset:0 atIndex:3];
                    const int S = v == 6 ? 2 : 4;
                    if (v >= 6) { [enc setBytes:&S length:4 atIndex:4]; [enc dispatchThreadgroups:MTLSizeMake(N / 128, E, S) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)]; }
                    else if (v == 5) [enc dispatchThreads:MTLSizeMake(40 * 2048, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    else [enc dispatchThreadgroups:MTLSizeMake(N / (v == 4 ? 64 : 128), E, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
                    if (cb.error) { fprintf(stderr, "%s: %s\n", names[v], cb.error.localizedDescription.UTF8String); return 1; }
                    if (r >= warm) total += cb.GPUEndTime - cb.GPUStartTime;
                }
                const double sec = total / (reps - warm);
                const float *y = by.contents;
                double s2 = 0, mx = 0;
                for (int s = 0; s < NS; s++) {
                    double got = 0; const int parts = v == 6 ? 2 : v == 7 ? 4 : 1;
                    for (int p = 0; p < parts; p++) got += y[(size_t)p * E * 32 * N + ((size_t)se_[s] * 32 + sr[s]) * N + sn[s]];
                    const double e = fabs(got - ref[s]) / nrm[s];
                    s2 += e * e; if (e > mx) mx = e;
                }
                printf("%5d->%-6d %-12s %9.3f %8.1f %8.0f %11.2e %11.2e\n", K, N, labels[v], sec * 1e3,
                       2.0 * E * R * N * K / sec / 1e12, (double)E * N * K * 2 / sec / 1e9, sqrt(s2 / NS), mx);
            }
            free(xf);
        }
    }
    return 0;
}
