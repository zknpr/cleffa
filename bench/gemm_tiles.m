// Production FP16 x BF16 GEMMs: exact tile/packing parity, sampled float64 oracle,
// and NaN guards at ragged edges. Optional T K N arguments also time all tiles.
// Build/run: make test-gemm; ./gemm-tiles 10387 17408 5120
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { int T, N, K; } gemm_args;

static void require(bool ok, const char *why) {
    if (!ok) { fprintf(stderr, "gemm tiles: %s\n", why); exit(1); }
}

static id<MTLBuffer> buffer(id<MTLDevice> dev, size_t bytes) {
    id<MTLBuffer> b = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    require(b != nil, "buffer allocation failed");
    return b;
}

static float value(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352d; x ^= x >> 15; x *= 0x846ca68b; x ^= x >> 16;
    return (float)(x & 0xffff) / 32768.0f - 1.0f;
}

static uint16_t bf16(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    u += 0x7fff + ((u >> 16) & 1);
    return u >> 16;
}

static float unbf16(uint16_t h) {
    uint32_t u = (uint32_t)h << 16;
    float f; memcpy(&f, &u, 4); return f;
}

static float initial(size_t i) { return value((uint32_t)i + 373) * 0.1f; }

static double run(id<MTLCommandQueue> queue, id<MTLComputePipelineState> ps,
                  gemm_args a, NSArray<id<MTLBuffer>> *buffers, int tm, int tn) {
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    require(cb != nil && enc != nil, "command creation failed");
    [enc setComputePipelineState:ps];
    [enc setBytes:&a length:sizeof(a) atIndex:0];
    for (int i = 0; i < 3; i++) [enc setBuffer:buffers[i] offset:0 atIndex:i + 1];
    [enc dispatchThreadgroups:MTLSizeMake((a.N + tn - 1) / tn, (a.T + tm - 1) / tm, 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
    require(cb.error == nil, cb.error.localizedDescription.UTF8String);
    return (cb.GPUEndTime - cb.GPUStartTime) * 1e3;
}

static void check(gemm_args a, NSArray<id<MTLBuffer>> *b, bool acc) {
    const __fp16 *x = b[0].contents;
    const uint16_t *w = b[1].contents;
    const float *y = b[2].contents;
    size_t ny = (size_t)a.T * a.N;
    for (size_t i = 0; i < ny; i++) require(isfinite(y[i]), "non-finite/unwritten output");
    for (size_t i = ny; i < ny + 128; i++) require(isnan(y[i]), "output overrun");
    for (int s = 0; s < 32; s++) {
        int t = s == 0 ? a.T - 1 : (s * 7919) % a.T;
        int n = s == 0 ? a.N - 1 : (s * 104729) % a.N;
        size_t at = (size_t)t * a.N + n;
        double ref = acc ? initial(at) : 0, mag = fabs(ref);
        for (int k = 0; k < a.K; k++) {
            double p = (double)x[(size_t)t * a.K + k] * unbf16(w[(size_t)n * a.K + k]);
            ref += p; mag += fabs(p);
        }
        require(fabs(y[at] - ref) <= 2e-6 * (mag + 1e-30), "float64 oracle mismatch");
    }
}

static void fixture(id<MTLDevice> dev, id<MTLCommandQueue> queue, NSArray *ps,
                    gemm_args a, int reps) {
    @autoreleasepool {
        size_t nx = (size_t)a.T * a.K, nw = (size_t)a.N * a.K, ny = (size_t)a.T * a.N;
        // 31 changes a record's row alignment in both tile sizes. Poison after each logical
        // operand so a masked tail that reads NaNs into valid outputs fails explicitly.
        const int shift = 31;
        size_t xp = (size_t)(a.T + shift + 64) * a.K, yp = (size_t)(a.T + shift) * a.N + 128;
        NSArray *b = @[buffer(dev, xp * 2), buffer(dev, (nw + 128 * a.K) * 2), buffer(dev, yp * 4)];
        __fp16 *x = [b[0] contents]; uint16_t *w = [b[1] contents]; float *y = [b[2] contents];
        for (size_t i = 0; i < xp; i++) x[i] = NAN;
        for (size_t i = 0; i < nx; i++) x[i] = (__fp16)(value((uint32_t)i) * (i % 67 == 0 ? 8 : 1));
        for (size_t i = 0; i < nw + 128 * a.K; i++) w[i] = 0x7fc1;
        for (size_t i = 0; i < nw; i++) w[i] = bf16(value((uint32_t)i + 123) * 0.02f);
        for (int acc = 0; acc < 2; acc++) {
            NSData *reference = nil;
            double ms[4] = { 0, 0, 0, 0 };
            const int tm[] = { 32, 64, 32, 64 }, tn[] = { 128, 128, 256, 128 };
            const int order[8][4] = {{0, 1, 2, 3}, {1, 2, 3, 0}, {2, 3, 0, 1}, {3, 0, 1, 2},
                                     {3, 2, 1, 0}, {2, 1, 0, 3}, {1, 0, 3, 2}, {0, 3, 2, 1}};
            // Alternate order on timed runs; reset the residual before every accumulate.
            for (int rep = 0; rep < reps; rep++) for (int j = 0; j < 4; j++) {
                int tile = order[rep % 8][j];
                for (size_t i = 0; i < yp; i++) y[i] = i < ny && acc ? initial(i) : NAN;
                double elapsed = run(queue, ps[tile * 2 + acc], a, b, tm[tile], tn[tile]);
                if (rep >= 2) ms[tile] += elapsed;
                check(a, b, acc);
                if (!reference) reference = [NSData dataWithBytes:y length:ny * 4];
                else require(memcmp(reference.bytes, y, ny * 4) == 0, "tile outputs differ");
            }
            // A packed record must match standalone even across the dispatch threshold.
            memmove(x + (size_t)shift * a.K, x, nx * 2);
            for (size_t i = 0; i < (size_t)shift * a.K; i++) x[i] = (__fp16)value((uint32_t)i + 29);
            size_t skip = (size_t)shift * a.N, packed = ny + skip;
            gemm_args p = { a.T + shift, a.N, a.K };
            for (int tile = 1; tile < 4; tile++) {
                for (size_t i = 0; i < yp; i++) y[i] = i < packed && acc ? initial(i < skip ? i : i - skip) : NAN;
                run(queue, ps[tile * 2 + acc], p, b, tm[tile], tn[tile]);
                require(memcmp(reference.bytes, y + skip, ny * 4) == 0, "packed row alignment changes output");
                for (size_t i = packed; i < yp; i++) require(isnan(y[i]), "packed output overrun");
            }
            memmove(x, x + (size_t)shift * a.K, nx * 2);
            for (size_t i = nx; i < xp; i++) x[i] = NAN;
            printf("T=%d K=%d N=%d acc=%d: exact tile/packing parity, float64 and NaN guards OK",
                   a.T, a.K, a.N, acc);
            if (reps > 2) printf("; 32x128 %.3f ms, 64x128 %.3f ms (%.3fx), 32x256 %.3f ms (%.3fx)",
                                ms[0] / (reps - 2), ms[1] / (reps - 2), ms[0] / ms[1], ms[2] / (reps - 2), ms[0] / ms[2]);
            if (reps > 2) printf("; 64x128 group4 %.3f ms (%.3fx)", ms[3] / (reps - 2), ms[0] / ms[3]);
            putchar('\n'); fflush(stdout);
        }
    }
}

int main(int argc, char **argv) {
    @autoreleasepool {
        require(argc == 1 || argc == 4, "usage: gemm-tiles [T K N]");
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice(); require(dev != nil, "no Metal device");
        id<MTLCommandQueue> queue = [dev newCommandQueue]; require(queue != nil, "no command queue");
        NSError *error = nil;
        NSString *src = [NSString stringWithContentsOfFile:@"metal/clef.metal" encoding:NSUTF8StringEncoding error:&error];
        require(src != nil, error.localizedDescription.UTF8String);
        MTLCompileOptions *options = [MTLCompileOptions new]; options.mathMode = MTLMathModeSafe;
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:options error:&error];
        require(lib != nil, error.localizedDescription.UTF8String);
        NSMutableArray *ps = [NSMutableArray new];
        for (NSString *name in @[@"gemm_f16_32x128", @"gemm_f16_acc_32x128", @"gemm_f16_64x128", @"gemm_f16_acc_64x128",
                                @"gemm_f16_32x256", @"gemm_f16_acc_32x256", @"gemm_f16_g4_64x128", @"gemm_f16_g4_acc_64x128"]) {
            id<MTLFunction> fn = [lib newFunctionWithName:name]; require(fn != nil, "missing GEMM kernel");
            id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:fn error:&error];
            require(p != nil, error.localizedDescription.UTF8String); [ps addObject:p];
        }
        if (argc == 4) {
            gemm_args a = { atoi(argv[1]), atoi(argv[3]), atoi(argv[2]) };
            require(a.T > 0 && a.T <= 32768 && a.K > 0 && a.K <= 34816 && a.N > 0 && a.N <= 34816, "invalid shape");
            fixture(dev, queue, ps, a, 8);
        } else {
            for (int t = 1; t <= 65; t++) fixture(dev, queue, ps, (gemm_args){t, 131, 129}, 1);
            // Partial final row groups have heights 1, 2 or 3; exercise all of them,
            // plus the transition to the next full four-row group, with poisoned tails.
            const int group_edges[] = {127, 128, 129, 191, 192, 193, 255, 256, 257};
            for (size_t i = 0; i < sizeof(group_edges) / sizeof(group_edges[0]); i++)
                fixture(dev, queue, ps, (gemm_args){group_edges[i], 131, 129}, 1);
            int shapes[][2] = {{4096, 24576}, {12288, 4096}, {5120, 34816}, {17408, 5120}, {5120, 16480}};
            for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); s++)
                for (int t = 1023; t <= 1025; t++) fixture(dev, queue, ps, (gemm_args){t, shapes[s][1], shapes[s][0]}, 1);
            // The long 27B down-projection switches back to 32x256 here. Packing
            // shifts each record by 31 rows, including across this dispatch boundary.
            for (int t = 4095; t <= 4097; t++) fixture(dev, queue, ps, (gemm_args){t, 5120, 17408}, 1);
            // The grouped 27B expansion dispatch starts at the same token boundary.
            for (int t = 4095; t <= 4097; t++) fixture(dev, queue, ps, (gemm_args){t, 34816, 5120}, 1);
        }
    }
    return 0;
}
