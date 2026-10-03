// Does MPP matmul2d on device tensors mask out-of-range rows/cols at the edges?
// Runs the direct kernel with ragged T/N and checks every output plus a guard band
// after Y that must stay untouched.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdio.h>

static NSString *kSource = @
"#include <metal_stdlib>\n"
"#include <metal_tensor>\n"
"#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
"using namespace metal;\n"
"using namespace mpp::tensor_ops;\n"
"struct gemm_args { int T, N, K; };\n"
"kernel void gemm(constant gemm_args &a [[buffer(0)]], device bfloat *X [[buffer(1)]], device bfloat *W [[buffer(2)]],\n"
"                 device float *Y [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]]) {\n"
"    auto tX = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(X, dextents<int32_t, 2>(a.K, a.T));\n"
"    auto tW = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(W, dextents<int32_t, 2>(a.K, a.N));\n"
"    auto tY = tensor<device float, dextents<int32_t, 2>, tensor_inline>(Y, dextents<int32_t, 2>(a.N, a.T));\n"
"    matmul2d<matmul2d_descriptor(128, 64, dynamic_length_v<int>, false, true, false), execution_simdgroups<4>> mm;\n"
"    auto mX = tX.slice(0, (int)tg.y * 128); auto mW = tW.slice(0, (int)tg.x * 64); auto mY = tY.slice((int)tg.x * 64, (int)tg.y * 128);\n"
"    mm.run(mX, mW, mY);\n"
"}\n"
"kernel void gemm_acc(constant gemm_args &a [[buffer(0)]], device bfloat *X [[buffer(1)]], device bfloat *W [[buffer(2)]],\n"
"                 device float *Y [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]]) {\n"
"    auto tX = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(X, dextents<int32_t, 2>(a.K, a.T));\n"
"    auto tW = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(W, dextents<int32_t, 2>(a.K, a.N));\n"
"    auto tY = tensor<device float, dextents<int32_t, 2>, tensor_inline>(Y, dextents<int32_t, 2>(a.N, a.T));\n"
"    matmul2d<matmul2d_descriptor(128, 64, dynamic_length_v<int>, false, true, false, matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroups<4>> mm;\n"
"    auto mX = tX.slice(0, (int)tg.y * 128); auto mW = tW.slice(0, (int)tg.x * 64); auto mY = tY.slice((int)tg.x * 64, (int)tg.y * 128);\n"
"    mm.run(mX, mW, mY);\n"
"}\n";

static uint16_t f2bf(float f) { uint32_t u; memcpy(&u, &f, 4); u += 0x7fff + ((u >> 16) & 1); return (uint16_t)(u >> 16); }
static float bf2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }

int main(void) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        NSError *err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:kSource options:[MTLCompileOptions new] error:&err];
        if (!lib) { fprintf(stderr, "%s\n", err.localizedDescription.UTF8String); return 1; }
        id<MTLComputePipelineState> ps_mul = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"gemm"] error:&err];
        id<MTLComputePipelineState> ps_acc = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"gemm_acc"] error:&err];
        id<MTLCommandQueue> q = [dev newCommandQueue];
        printf("maxBufferLength = %.1f GB\n", dev.maxBufferLength / 1e9);
        int cases[][3] = { {100, 48, 4096}, {1, 32, 4096}, {129, 4160, 4096}, {37, 6240, 5120}, {513, 64, 1000} };
        int fails = 0;
        for (int c = 0; c < 10; c++) {
            const int acc = c >= 5;  // second pass: Y += X.W^T over a pre-filled Y (residual add)
            id<MTLComputePipelineState> ps = acc ? ps_acc : ps_mul;
            int T = cases[c % 5][0], N = cases[c % 5][1], K = cases[c % 5][2];
            const int guard = 4096;
            id<MTLBuffer> bx = [dev newBufferWithLength:(size_t)T * K * 2 options:MTLResourceStorageModeShared];
            id<MTLBuffer> bw = [dev newBufferWithLength:(size_t)N * K * 2 options:MTLResourceStorageModeShared];
            id<MTLBuffer> by = [dev newBufferWithLength:((size_t)T * N + guard) * 4 options:MTLResourceStorageModeShared];
            uint16_t *x = bx.contents, *w = bw.contents;
            float *y = by.contents;
            for (size_t i = 0; i < (size_t)T * K; i++) x[i] = f2bf(sinf((float)i * 0.37f));
            for (size_t i = 0; i < (size_t)N * K; i++) w[i] = f2bf(cosf((float)i * 0.11f) * 0.02f);
            for (size_t i = 0; i < (size_t)T * N + guard; i++) y[i] = 12345.0f;
            if (acc) for (size_t i = 0; i < (size_t)T * N; i++) y[i] = 0.25f * (float)(i % 13);
            struct { int T, N, K; } args = { T, N, K };
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:ps];
            [enc setBytes:&args length:sizeof(args) atIndex:0];
            [enc setBuffer:bx offset:0 atIndex:1];
            [enc setBuffer:bw offset:0 atIndex:2];
            [enc setBuffer:by offset:0 atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake((N + 63) / 64, (T + 127) / 128, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            double max_rel = 0;
            for (int t = 0; t < T; t++) for (int n = 0; n < N; n++) {
                double ref = acc ? 0.25 * (double)(((size_t)t * N + n) % 13) : 0, mag = fabs(ref);
                for (int k = 0; k < K; k++) { double p = (double)bf2f(x[(size_t)t * K + k]) * bf2f(w[(size_t)n * K + k]); ref += p; mag += fabs(p); }
                double rel = fabs(y[(size_t)t * N + n] - ref) / (mag + 1e-30);
                if (rel > max_rel) max_rel = rel;
            }
            int guard_hit = 0;
            for (int i = 0; i < guard; i++) if (y[(size_t)T * N + i] != 12345.0f) guard_hit++;
            int ok = max_rel < 1e-5 && !guard_hit;
            fails += !ok;
            printf("%s T=%4d N=%5d K=%5d max_rel=%.2e guard_writes=%d %s\n", acc ? "acc" : "mul", T, N, K, max_rel, guard_hit, ok ? "OK" : "FAIL");
        }
        return fails != 0;
    }
}
