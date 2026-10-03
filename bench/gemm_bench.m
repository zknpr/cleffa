// BF16 x BF16 -> F32 GEMM throughput on Metal 4 tensor ops (matmul2d), Y[T,N] = X[T,K] . W[N,K]^T.
// Purpose: find the achievable prefill matmul rate on this GPU and the best tile shape,
// and check numerical correctness against a CPU f64 reference on sampled outputs.
//
// build: clang -O2 -fobjc-arc bench/gemm_bench.m -framework Metal -framework Foundation -o gemm-bench

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static NSString *kSource = @
"#include <metal_stdlib>\n"
"#include <metal_tensor>\n"
"#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
"using namespace metal;\n"
"using namespace mpp::tensor_ops;\n"
"struct gemm_args { int T, N, K; };\n"
"\n"
"// Direct: both operands read from device memory, MPP iterates the full K.\n"
"template <int TM, int TN, int NSG>\n"
"kernel void gemm_direct(constant gemm_args &a [[buffer(0)]],\n"
"                        device bfloat *X [[buffer(1)]],\n"
"                        device bfloat *W [[buffer(2)]],\n"
"                        device float *Y [[buffer(3)]],\n"
"                        uint2 tg [[threadgroup_position_in_grid]]) {\n"
"    auto tX = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(X, dextents<int32_t, 2>(a.K, a.T));\n"
"    auto tW = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(W, dextents<int32_t, 2>(a.K, a.N));\n"
"    auto tY = tensor<device float, dextents<int32_t, 2>, tensor_inline>(Y, dextents<int32_t, 2>(a.N, a.T));\n"
"    const int t0 = tg.y * TM, n0 = tg.x * TN;\n"
"    matmul2d<matmul2d_descriptor(TM, TN, dynamic_length_v<int>, false, true, false), execution_simdgroups<NSG>> mm;\n"
"    auto mX = tX.slice(0, t0);\n"
"    auto mW = tW.slice(0, n0);\n"
"    auto mY = tY.slice(n0, t0);\n"
"    mm.run(mX, mW, mY);\n"
"}\n"
"\n"
"// K-loop with a cooperative accumulator, static K step.\n"
"template <int TM, int TN, int TK, int NSG>\n"
"kernel void gemm_kloop(constant gemm_args &a [[buffer(0)]],\n"
"                       device bfloat *X [[buffer(1)]],\n"
"                       device bfloat *W [[buffer(2)]],\n"
"                       device float *Y [[buffer(3)]],\n"
"                       uint2 tg [[threadgroup_position_in_grid]]) {\n"
"    auto tX = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(X, dextents<int32_t, 2>(a.K, a.T));\n"
"    auto tW = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(W, dextents<int32_t, 2>(a.K, a.N));\n"
"    auto tY = tensor<device float, dextents<int32_t, 2>, tensor_inline>(Y, dextents<int32_t, 2>(a.N, a.T));\n"
"    const int t0 = tg.y * TM, n0 = tg.x * TN;\n"
"    matmul2d<matmul2d_descriptor(TM, TN, TK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroups<NSG>> mm;\n"
"    auto c = mm.template get_destination_cooperative_tensor<decltype(tX), decltype(tW), float>();\n"
"    for (uint16_t i = 0; i < c.get_capacity(); ++i) if (c.is_valid_element(i)) c[i] = 0.0f;\n"
"    for (int k = 0; k < a.K; k += TK) {\n"
"        auto sX = tX.slice(k, t0); auto sW = tW.slice(k, n0);\n"
"        mm.run(sX, sW, c);\n"
"    }\n"
"    auto sY = tY.slice(n0, t0);\n"
"    c.store(sY);\n"
"}\n"
"\n"
"#define DIRECT(TM, TN, NSG) template [[host_name(\"direct_\" #TM \"_\" #TN \"_\" #NSG)]] kernel void gemm_direct<TM, TN, NSG>(constant gemm_args &, device bfloat *, device bfloat *, device float *, uint2);\n"
"#define KLOOP(TM, TN, TK, NSG) template [[host_name(\"kloop_\" #TM \"_\" #TN \"_\" #TK \"_\" #NSG)]] kernel void gemm_kloop<TM, TN, TK, NSG>(constant gemm_args &, device bfloat *, device bfloat *, device float *, uint2);\n"
"DIRECT(64, 32, 4) DIRECT(64, 64, 4) DIRECT(128, 64, 4) DIRECT(64, 128, 4) DIRECT(128, 128, 4) DIRECT(128, 128, 8)\n"
"KLOOP(64, 64, 32, 4) KLOOP(64, 64, 64, 4) KLOOP(128, 64, 32, 4) KLOOP(64, 128, 32, 4) KLOOP(128, 128, 32, 4) KLOOP(128, 128, 64, 8)\n";

static uint16_t f2bf(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    u += 0x7fff + ((u >> 16) & 1);  // round to nearest even
    return (uint16_t)(u >> 16);
}

static float bf2f(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

typedef struct { int T, N, K; } gemm_args;

int main(int argc, char **argv) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        printf("device: %s\n", dev.name.UTF8String);
        NSError *err = nil;
        MTLCompileOptions *opt = [MTLCompileOptions new];
        id<MTLLibrary> lib = [dev newLibraryWithSource:kSource options:opt error:&err];
        if (!lib) { fprintf(stderr, "compile failed: %s\n", err.localizedDescription.UTF8String); return 1; }
        id<MTLCommandQueue> q = [dev newCommandQueue];

        // Clef-Flash shapes (K x N): qkv 4096->8192, mlp up 4096->12288, mlp down 12288->4096;
        // Clef 27B: mlp up 5120->17408, down 17408->5120.
        int shapes[][2] = { {4096, 12288}, {12288, 4096}, {4096, 8192}, {5120, 17408}, {17408, 5120} };
        int Ts[] = { 128, 512, 2048 };
        const char *kernels[] = {
            "direct_64_32_4", "direct_64_64_4", "direct_128_64_4", "direct_64_128_4", "direct_128_128_4", "direct_128_128_8",
            "kloop_64_64_32_4", "kloop_64_64_64_4", "kloop_128_64_32_4", "kloop_64_128_32_4", "kloop_128_128_32_4", "kloop_128_128_64_8",
        };
        int tile[][2] = { {64,32},{64,64},{128,64},{64,128},{128,128},{128,128},{64,64},{64,64},{128,64},{64,128},{128,128},{128,128} };
        int nsg[] = { 4,4,4,4,4,8, 4,4,4,4,4,8 };
        int nk = sizeof(kernels) / sizeof(kernels[0]);

        for (size_t si = 0; si < sizeof(shapes) / sizeof(shapes[0]); si++) {
            int K = shapes[si][0], N = shapes[si][1];
            for (size_t ti = 0; ti < sizeof(Ts) / sizeof(Ts[0]); ti++) {
                int T = Ts[ti];
                id<MTLBuffer> bx = [dev newBufferWithLength:(size_t)T * K * 2 options:MTLResourceStorageModeShared];
                id<MTLBuffer> bw = [dev newBufferWithLength:(size_t)N * K * 2 options:MTLResourceStorageModeShared];
                id<MTLBuffer> by = [dev newBufferWithLength:(size_t)T * N * 4 options:MTLResourceStorageModeShared];
                uint16_t *x = bx.contents, *w = bw.contents;
                srand(42);
                for (size_t i = 0; i < (size_t)T * K; i++) x[i] = f2bf((float)((double)rand() / RAND_MAX) - 0.5f);
                for (size_t i = 0; i < (size_t)N * K; i++) w[i] = f2bf(((float)((double)rand() / RAND_MAX) - 0.5f) * 0.05f);
                double best = 0;
                const char *best_name = "";
                for (int k = 0; k < nk; k++) {
                    if (T % tile[k][0] || N % tile[k][1]) continue;
                    id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:kernels[k]]];
                    id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:fn error:&err];
                    if (!ps) { fprintf(stderr, "%s: %s\n", kernels[k], err.localizedDescription.UTF8String); continue; }
                    gemm_args args = { T, N, K };
                    double total = 0;
                    int reps = 12;
                    for (int r = 0; r < reps; r++) {
                        id<MTLCommandBuffer> cb = [q commandBuffer];
                        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                        [enc setComputePipelineState:ps];
                        [enc setBytes:&args length:sizeof(args) atIndex:0];
                        [enc setBuffer:bx offset:0 atIndex:1];
                        [enc setBuffer:bw offset:0 atIndex:2];
                        [enc setBuffer:by offset:0 atIndex:3];
                        [enc dispatchThreadgroups:MTLSizeMake(N / tile[k][1], T / tile[k][0], 1)
                            threadsPerThreadgroup:MTLSizeMake(32 * nsg[k], 1, 1)];
                        [enc endEncoding];
                        [cb commit];
                        [cb waitUntilCompleted];
                        if (cb.error) { fprintf(stderr, "%s: %s\n", kernels[k], cb.error.localizedDescription.UTF8String); break; }
                        if (r >= 2) total += cb.GPUEndTime - cb.GPUStartTime;
                    }
                    double s = total / (reps - 2);
                    double tflops = 2.0 * T * N * K / s / 1e12;
                    // correctness: 64 sampled outputs vs f64 reference
                    float *y = by.contents;
                    double max_rel = 0;
                    for (int smp = 0; smp < 64; smp++) {
                        int ti2 = (smp * 7919) % T, ni = (smp * 104729) % N;
                        double ref = 0, mag = 0;
                        for (int kk = 0; kk < K; kk++) {
                            double p = (double)bf2f(x[(size_t)ti2 * K + kk]) * bf2f(w[(size_t)ni * K + kk]);
                            ref += p; mag += fabs(p);
                        }
                        double rel = fabs(y[(size_t)ti2 * N + ni] - ref) / (mag + 1e-30);
                        if (rel > max_rel) max_rel = rel;
                    }
                    printf("K=%5d N=%5d T=%4d %-20s %7.3f ms %6.1f TFLOPS  max_err/sum|p| %.2e\n",
                           K, N, T, kernels[k], s * 1e3, tflops, max_rel);
                    if (tflops > best) { best = tflops; best_name = kernels[k]; }
                }
                printf("  best: %s %.1f TFLOPS\n", best_name, best);
            }
        }
    }
    return 0;
}
