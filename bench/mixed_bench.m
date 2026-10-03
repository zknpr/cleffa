// Activation precision vs speed for the engine's GEMM: Y[T,N] = X[T,K] . W[N,K]^T with W BF16
// (exact model weights) and X in bfloat, half or float (MPP matmul2d accepts all three against a
// bfloat right operand, accumulating in float). Same kernel shape as the engine (32x128 tile,
// 4 simdgroups, MPP iterates K).
//
// Precision is measured against an f64 reference that uses the *unrounded* f32 activations, so
// it shows the full error of each path: activation rounding plus whatever the hardware does
// internally. If "float" were silently converted to bf16/half inside the unit, its error would
// match that type's.
//
// build: clang -O2 -fobjc-arc bench/mixed_bench.m -framework Metal -framework Foundation -o mixed-bench
// usage: mixed-bench [T ...]   (token counts; default 260 2048)

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
"template <typename XT>\n"
"kernel void gemm(constant gemm_args &a [[buffer(0)]], device XT *X [[buffer(1)]], device bfloat *W [[buffer(2)]],\n"
"                 device float *Y [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]]) {\n"
"    auto tX = tensor<device XT, dextents<int32_t, 2>, tensor_inline>(X, dextents<int32_t, 2>(a.K, a.T));\n"
"    auto tW = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(W, dextents<int32_t, 2>(a.K, a.N));\n"
"    auto tY = tensor<device float, dextents<int32_t, 2>, tensor_inline>(Y, dextents<int32_t, 2>(a.N, a.T));\n"
"    matmul2d<matmul2d_descriptor(32, 128, dynamic_length_v<int>, false, true, false), execution_simdgroups<4>> mm;\n"
"    auto mX = tX.slice(0, (int)tg.y * 32); auto mW = tW.slice(0, (int)tg.x * 128); auto mY = tY.slice((int)tg.x * 128, (int)tg.y * 32);\n"
"    mm.run(mX, mW, mY);\n"
"}\n"
"template [[host_name(\"gemm_bfloat\")]] kernel void gemm<bfloat>(constant gemm_args &, device bfloat *, device bfloat *, device float *, uint2);\n"
"template [[host_name(\"gemm_half\")]] kernel void gemm<half>(constant gemm_args &, device half *, device bfloat *, device float *, uint2);\n"
"template [[host_name(\"gemm_float\")]] kernel void gemm<float>(constant gemm_args &, device float *, device bfloat *, device float *, uint2);\n";

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
static double gauss(void) {   // Box-Muller
    double u = ((double)rand() + 1) / ((double)RAND_MAX + 2), v = (double)rand() / RAND_MAX;
    return sqrt(-2 * log(u)) * cos(2 * M_PI * v);
}

typedef struct { int T, N, K; } gemm_args;

int main(int argc, char **argv) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        NSError *err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:kSource options:[MTLCompileOptions new] error:&err];
        if (!lib) { fprintf(stderr, "compile failed: %s\n", err.localizedDescription.UTF8String); return 1; }
        id<MTLCommandQueue> q = [dev newCommandQueue];
        const char *types[] = { "bfloat", "half", "float" };
        const int esz[] = { 2, 2, 4 };
        // Clef 27B: qkv-ish 5120->12288, gate_up 5120->34816, down 17408->5120
        int shapes[][2] = { {5120, 12288}, {5120, 34816}, {17408, 5120} };
        int Ts[16] = { 260, 2048 }, nT = 2;
        if (argc > 1) { nT = 0; for (int i = 1; i < argc && nT < 16; i++) Ts[nT++] = atoi(argv[i]); }
        printf("%-14s %5s %-7s %8s %12s %12s\n", "K->N", "T", "X type", "TFLOPS", "rel err", "max rel");
        for (size_t si = 0; si < 3; si++) {
            const int K = shapes[si][0], N = shapes[si][1];
            for (int ti = 0; ti < nT; ti++) {
                const int T = Ts[ti];
                srand(7);
                float *xf = malloc((size_t)T * K * 4);
                for (size_t i = 0; i < (size_t)T * K; i++) xf[i] = (float)(gauss() * (rand() % 64 == 0 ? 8.0 : 1.0));   // some outliers
                id<MTLBuffer> bw = [dev newBufferWithLength:(size_t)N * K * 2 options:MTLResourceStorageModeShared];
                uint16_t *w = bw.contents;
                for (size_t i = 0; i < (size_t)N * K; i++) w[i] = f2bf((float)(gauss() * 0.02));
                id<MTLBuffer> by = [dev newBufferWithLength:(size_t)T * N * 4 options:MTLResourceStorageModeShared];
                // f64 reference on sampled outputs, from the unrounded activations
                enum { NS = 256 };
                int st[NS], sn[NS];
                double ref[NS], nrm[NS];
                for (int s = 0; s < NS; s++) {
                    st[s] = (s * 7919) % T; sn[s] = (s * 104729) % N;
                    double r = 0, m = 0;
                    for (int k = 0; k < K; k++) {
                        const double p = (double)xf[(size_t)st[s] * K + k] * bf2f(w[(size_t)sn[s] * K + k]);
                        r += p; m += fabs(p);
                    }
                    ref[s] = r; nrm[s] = m;
                }
                for (int ty = 0; ty < 3; ty++) {
                    id<MTLBuffer> bx = [dev newBufferWithLength:(size_t)T * K * esz[ty] options:MTLResourceStorageModeShared];
                    for (size_t i = 0; i < (size_t)T * K; i++) {
                        if (ty == 0) ((uint16_t *)bx.contents)[i] = f2bf(xf[i]);
                        else if (ty == 1) ((__fp16 *)bx.contents)[i] = (__fp16)xf[i];
                        else ((float *)bx.contents)[i] = xf[i];
                    }
                    id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithFormat:@"gemm_%s", types[ty]]];
                    id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:fn error:&err];
                    if (!ps) { fprintf(stderr, "%s: %s\n", types[ty], err.localizedDescription.UTF8String); return 1; }
                    gemm_args a = { T, N, K };
                    double total = 0;
                    // 10 untimed warm-up dispatches: generating inputs on the CPU idles the GPU, its clocks
                    // drop, and the first type timed after that used to read ~40% slow at T=260 (review #3)
                    const int reps = 20, warm = 10;
                    for (int r = 0; r < reps; r++) {
                        id<MTLCommandBuffer> cb = [q commandBuffer];
                        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                        [enc setComputePipelineState:ps];
                        [enc setBytes:&a length:sizeof(a) atIndex:0];
                        [enc setBuffer:bx offset:0 atIndex:1];
                        [enc setBuffer:bw offset:0 atIndex:2];
                        [enc setBuffer:by offset:0 atIndex:3];
                        [enc dispatchThreadgroups:MTLSizeMake((N + 127) / 128, (T + 31) / 32, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                        [enc endEncoding];
                        [cb commit];
                        [cb waitUntilCompleted];
                        if (cb.error) { fprintf(stderr, "%s: %s\n", types[ty], cb.error.localizedDescription.UTF8String); return 1; }
                        if (r >= warm) total += cb.GPUEndTime - cb.GPUStartTime;
                    }
                    const double tflops = 2.0 * T * N * K / (total / (reps - warm)) / 1e12;
                    // error relative to the sum of |terms| (scale of the dot product), RMS and max
                    const float *y = by.contents;
                    double se = 0, mx = 0;
                    for (int s = 0; s < NS; s++) {
                        const double e = fabs((double)y[(size_t)st[s] * N + sn[s]] - ref[s]) / nrm[s];
                        se += e * e; if (e > mx) mx = e;
                    }
                    printf("%5d->%-7d %5d %-7s %8.1f %12.2e %12.2e\n", K, N, T, types[ty], tflops, sqrt(se / NS), mx);
                }
                free(xf);
            }
        }
    }
    return 0;
}
