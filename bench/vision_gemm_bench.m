// Vision GEMMs: FP32 tile parity, compensated tile/packing parity, float64 samples,
// poisoned ragged tails, and split overflow isolation. No model required.
// make test-vision-gemm; ./vision-gemm-bench T K N (also times production kernels)
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { int T, N, K; } gemm_args;
typedef struct { int f16; float lim; } act_args;

static void require(bool ok, const char *why) {
    if (!ok) { fprintf(stderr, "vision GEMM: %s\n", why); exit(1); }
}
static id<MTLBuffer> buffer(id<MTLDevice> d, size_t n) {
    id<MTLBuffer> b = [d newBufferWithLength:n options:MTLResourceStorageModeShared];
    require(b != nil, "allocation failed"); return b;
}
static float value(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352d; x ^= x >> 15; x *= 0x846ca68b; x ^= x >> 16;
    return (float)(x & 0xffffff) / 8388608.0f - 1.0f;
}
static uint16_t bf16(float x) {
    uint32_t u; memcpy(&u, &x, 4); u += 0x7fff + ((u >> 16) & 1); return u >> 16;
}
static float unbf16(uint16_t h) {
    uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f;
}
static float initial(size_t i) { return value((uint32_t)i + 373) * 0.1f; }

static void split(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> ps,
                  id<MTLBuffer> x, id<MTLBuffer> halves, id<MTLBuffer> flags, long n, float limit) {
    act_args ac = { 1, limit };
    [enc setComputePipelineState:ps];
    [enc setBuffer:x offset:0 atIndex:0]; [enc setBuffer:halves offset:0 atIndex:1];
    [enc setBytes:&n length:sizeof(n) atIndex:2]; [enc setBytes:&ac length:sizeof(ac) atIndex:3];
    [enc setBuffer:flags offset:3 * sizeof(int) atIndex:4];
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

static double run(id<MTLCommandQueue> q, NSArray *ps, gemm_args a, NSArray *b, int variant, int acc) {
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    require(cb && enc, "command creation failed");
    const bool comp = variant >= 2;
    const int tm[] = { 32, 16, 32, 64 };
    if (comp) split(enc, ps[6], b[0], b[3], b[4], (long)a.T * a.K, 65504);
    [enc setComputePipelineState:ps[comp ? variant + 2 : variant * 2 + acc]];
    [enc setBytes:&a length:sizeof(a) atIndex:0];
    [enc setBuffer:b[comp ? 3 : 0] offset:0 atIndex:1];
    [enc setBuffer:b[1] offset:0 atIndex:2]; [enc setBuffer:b[2] offset:0 atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake((a.N + 127) / 128, (a.T + tm[variant] - 1) / tm[variant], 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
    require(!cb.error, cb.error.localizedDescription.UTF8String);
    return (cb.GPUEndTime - cb.GPUStartTime) * 1e3;
}

static double check(gemm_args a, NSArray *b, bool acc) {
    const float *x = [b[0] contents], *y = [b[2] contents]; const uint16_t *w = [b[1] contents];
    const size_t ny = (size_t)a.T * a.N;
    for (size_t i = 0; i < ny; i++) require(isfinite(y[i]), "unwritten/nonfinite result");
    for (size_t i = ny; i < ny + 128; i++) require(isnan(y[i]), "output overrun");
    double se = 0, sr = 0;
    for (int s = 0; s < 64; s++) {
        const int t = s == 0 ? a.T - 1 : s * 7919 % a.T, n = s == 0 ? a.N - 1 : s * 104729 % a.N;
        const size_t at = (size_t)t * a.N + n;
        double ref = acc ? initial(at) : 0, mag = fabs(ref);
        for (int k = 0; k < a.K; k++) {
            double term = (double)x[(size_t)t * a.K + k] * unbf16(w[(size_t)n * a.K + k]);
            ref += term; mag += fabs(term);
        }
        const double err = y[at] - ref;
        require(fabs(err) <= 2e-6 * (mag + 1e-30), "float64 dot-product bound");
        se += err * err; sr += ref * ref;
    }
    require(se <= 1e-10 * (sr + 1e-30), "sampled relative L2 exceeds 1e-5");
    return sqrt(se / (sr + 1e-30));
}

static void fixture(id<MTLDevice> d, id<MTLCommandQueue> q, NSArray *ps, gemm_args a, int mode, int reps) {
    @autoreleasepool {
        const int shift = 31;
        size_t nx = (size_t)a.T * a.K, nw = (size_t)a.N * a.K, ny = (size_t)a.T * a.N;
        size_t xp = (size_t)(a.T + shift + 64) * a.K, yp = (size_t)(a.T + shift) * a.N + 128;
        NSArray *b = @[buffer(d, xp * 4), buffer(d, (nw + 128 * a.K) * 2), buffer(d, yp * 4),
                       buffer(d, xp * 4), buffer(d, 8 * sizeof(int))];
        float *x = [b[0] contents], *y = [b[2] contents]; uint16_t *w = [b[1] contents];
        __fp16 *halves = [b[3] contents]; int *flags = [b[4] contents];
        for (size_t i = 0; i < xp; i++) x[i] = NAN;
        for (size_t i = 0; i < nx; i++) {
            // Full FP32 mantissas, including tiny values and outliers. Never pre-round to half.
            x[i] = value((uint32_t)i + 19) * (mode ? ldexpf(1.0f, (int)(i % 36) - 20) : i % 31 == 0 ? 8 : 1);
        }
        for (size_t i = 0; i < nw + 128 * a.K; i++) w[i] = 0x7fc1;
        for (size_t i = 0; i < nw; i++) w[i] = bf16(value((uint32_t)i + 123) * 0.02f);
        for (int acc = 0; acc < 2; acc++) {
            NSData *ref32 = nil, *refcomp = nil;
            double ms[4] = {0}, error[4] = {0}; const int count = acc ? 2 : 4;
            for (int rep = 0; rep < reps; rep++) for (int j = 0; j < count; j++) {
                const int v = (j + rep) % count;
                for (size_t i = 0; i < yp; i++) y[i] = i < ny && acc ? initial(i) : NAN;
                for (size_t i = 0; i < xp * 2; i++) halves[i] = NAN;
                memset(flags, 0, 8 * sizeof(int));
                double elapsed = run(q, ps, a, b, v, acc);
                error[v] = check(a, b, acc);
                if (rep >= 2) ms[v] += elapsed;
                if (v < 2) {
                    if (!ref32) ref32 = [NSData dataWithBytes:y length:ny * 4];
                    else require(!memcmp(ref32.bytes, y, ny * 4), "FP32 tile outputs differ");
                } else {
                    if (!refcomp) refcomp = [NSData dataWithBytes:y length:ny * 4];
                    else require(!memcmp(refcomp.bytes, y, ny * 4), "compensated tile outputs differ");
                    for (size_t i = nx * 2; i < xp * 2; i++) require(isnan(halves[i]), "split overrun");
                    for (int i = 0; i < 8; i++) require(flags[i] == 0, "unexpected overflow");
                }
            }
            // Shift the record, change its tile alignment and cross the dispatch threshold.
            memmove(x + (size_t)shift * a.K, x, nx * 4);
            for (size_t i = 0; i < (size_t)shift * a.K; i++) x[i] = value((uint32_t)i + 29);
            const size_t skip = (size_t)shift * a.N, packed = ny + skip;
            for (int v = 0; v < count; v++) {
                for (size_t i = 0; i < yp; i++) y[i] = i < packed && acc ? initial(i < skip ? i : i - skip) : NAN;
                for (size_t i = 0; i < xp * 2; i++) halves[i] = NAN;
                run(q, ps, (gemm_args){a.T + shift, a.N, a.K}, b, v, acc);
                require(!memcmp((v < 2 ? ref32 : refcomp).bytes, y + skip, ny * 4), "packed row alignment changes output");
                for (size_t i = packed; i < yp; i++) require(isnan(y[i]), "packed output overrun");
            }
            memmove(x, x + (size_t)shift * a.K, nx * 4);
            for (size_t i = nx; i < xp; i++) x[i] = NAN;
            printf("T=%d K=%d N=%d mode=%d acc=%d: oracle/packing/NaN OK; sampled relL2", a.T, a.K, a.N, mode, acc);
            for (int v = 0; v < count; v++) printf(" %.2e", error[v]);
            if (reps > 2) { printf("; ms (f32-32,f32-16,comp-32,comp-64)"); for (int v = 0; v < count; v++) printf(" %.4f", ms[v] / (reps - 2)); }
            puts(""); fflush(stdout);
        }
    }
}

static void overflow(id<MTLDevice> d, id<MTLCommandQueue> q, id<MTLComputePipelineState> ps) {
    const float inputs[] = { 0, -0.0f, 65504, -65504, 65505, -65505, INFINITY, -INFINITY, NAN, 1e-20f, -1e-20f };
    const long n = sizeof(inputs) / sizeof(inputs[0]);
    id<MTLBuffer> x = buffer(d, sizeof(inputs)), h = buffer(d, n * 4 + 32), f = buffer(d, 8 * sizeof(int));
    memcpy(x.contents, inputs, sizeof(inputs));
    for (int trial = 0; trial < 2; trial++) {
        const float limit = trial ? 0.01f : 65504;
        memset(f.contents, 0, f.length);
        __fp16 *hp = h.contents; for (size_t i = 0; i < h.length / 2; i++) hp[i] = NAN;
        id<MTLCommandBuffer> cb = [q commandBuffer]; id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        require(cb && enc, "command creation failed");
        split(enc, ps, x, h, f, n, limit);
        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted]; require(!cb.error, cb.error.localizedDescription.UTF8String);
        for (int i = 0; i < 8; i++) require(((int *)f.contents)[i] == (i == 3), "overflow flag crossed record boundary");
        for (long i = 0; i < n; i++) {
            const bool bad = !(fabsf(inputs[i]) <= limit);
            const float hi = bad ? copysignf(65504, inputs[i]) : (float)(__fp16)inputs[i];
            const float lo = bad ? 0 : (float)(__fp16)((inputs[i] - hi) * 2048);
            require(isfinite(hp[i]) && isfinite(hp[n + i]), "overflow wrote nonfinite half");
            require((float)hp[i] == hi && (float)hp[n + i] == lo, "overflow saturation/split mismatch");
        }
        for (size_t i = n * 2; i < h.length / 2; i++) require(isnan(hp[i]), "split tail overwritten");
    }
    puts("split: finite saturation, NaN/Inf, lowered limit and record-local flags OK");
}

// Compare fusion with the original two-dispatch arithmetic, including the residual writeback.
static void norm_fusion(id<MTLDevice> d, id<MTLCommandQueue> q, NSArray *ps) {
    typedef struct { int H; float eps; int rows; } ln_args;
    const int widths[] = { 37, 1152 };
    for (int width = 0; width < 2; width++) for (int mode = 0; mode < 3; mode++) {
        const int H = widths[width], rows = 17, out32 = mode == 0;
        const size_t n = (size_t)rows * H, bytes = n * (out32 ? 4 : 2);
        const ln_args la = { H, 1e-6f, rows }; const act_args ac = { mode != 2, 65504 };
        id<MTLBuffer> x = buffer(d, n * 4), w = buffer(d, H * 4), b = buffer(d, H * 4), bias = buffer(d, H * 4);
        id<MTLBuffer> y = buffer(d, bytes + 16), flags = buffer(d, 4);
        for (int i = 0; i < H; i++) {
            ((float *)w.contents)[i] = value(i + 100);
            ((float *)b.contents)[i] = value(i + 200);
            ((float *)bias.contents)[i] = value(i + 300);
        }
        NSData *rx = nil, *ry = nil;
        for (int fused = 0; fused < 2; fused++) {
            for (size_t i = 0; i < n; i++) ((float *)x.contents)[i] = value((uint32_t)i) * 13;
            memset(y.contents, 0xa5, y.length); memset(flags.contents, 0, flags.length);
            id<MTLCommandBuffer> cb = [q commandBuffer]; id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            require(cb && enc, "command creation failed");
            if (!fused) {
                [enc setComputePipelineState:ps[7]]; [enc setBytes:&H length:sizeof(H) atIndex:0];
                [enc setBuffer:x offset:0 atIndex:1]; [enc setBuffer:bias offset:0 atIndex:2];
                [enc dispatchThreads:MTLSizeMake(H, rows, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
            [enc setComputePipelineState:ps[fused ? 9 : 8]];
            [enc setBytes:&la length:sizeof(la) atIndex:0];
            [enc setBuffer:x offset:0 atIndex:1]; [enc setBuffer:w offset:0 atIndex:2]; [enc setBuffer:b offset:0 atIndex:3];
            [enc setBuffer:y offset:0 atIndex:4]; [enc setBytes:&ac length:sizeof(ac) atIndex:5];
            [enc setBuffer:flags offset:0 atIndex:6]; [enc setBuffer:y offset:0 atIndex:7];
            [enc setBytes:&out32 length:sizeof(out32) atIndex:8]; [enc setBuffer:bias offset:0 atIndex:9];
            [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(H >= 1024 ? 1024 : (H + 31) / 32 * 32, 1, 1)];
            [enc endEncoding]; [cb commit]; [cb waitUntilCompleted]; require(!cb.error, cb.error.localizedDescription.UTF8String);
            if (!fused) { rx = [NSData dataWithBytes:x.contents length:x.length]; ry = [NSData dataWithBytes:y.contents length:y.length]; }
            else {
                require(!memcmp(rx.bytes, x.contents, x.length), "fused norm changed residual writeback");
                require(!memcmp(ry.bytes, y.contents, y.length), "fused norm changed output");
            }
            for (size_t i = bytes; i < y.length; i++) require(((unsigned char *)y.contents)[i] == 0xa5, "norm overrun");
            require(*(int *)flags.contents == 0, "unexpected norm overflow");
        }
    }
    puts("bias + LayerNorm fusion: residual and FP32/FP16/BF16 outputs exactly match separate passes");
}

// A producer writing the half planes must match the old FP32 store + split exactly,
// including overflow flags and the residual bias writeback. Alternate image shapes
// and limits so stale scratch, wrong plane strides and missing flags are observable.
static void producer_split(id<MTLDevice> d, id<MTLCommandQueue> q, NSArray *ps) {
    typedef struct { int H; float eps; int rows; } ln_args;
    typedef struct { int N, mode, rows; } gelu_args;
    for (int producer = 0; producer < 4; producer++) for (int shape = 0; shape < 2; shape++)
    for (int low_limit = 0; low_limit < 2; low_limit++) {
        const int H = shape ? (producer < 2 ? 1152 : 4608) : 37, rows = shape ? 17 : 1;
        const size_t n = (size_t)rows * H, bytes = n * 4;
        const ln_args la = { H, 1e-6f, rows }; const gelu_args ga = { H, producer - 2, rows };
        const act_args ac = { 1, low_limit ? 0.01f : 65504 };
        id<MTLBuffer> x = buffer(d, bytes), w = buffer(d, H * 4), b = buffer(d, H * 4), bias = buffer(d, H * 4);
        id<MTLBuffer> temp = buffer(d, bytes), y = buffer(d, bytes + 64), flags = buffer(d, 8 * sizeof(int));
        for (int i = 0; i < H; i++) {
            ((float *)w.contents)[i] = value(i + 100) * (i % 19 == 0 ? 100000 : 1);
            ((float *)b.contents)[i] = value(i + 200);
            ((float *)bias.contents)[i] = value(i + 300);
        }
        NSData *rx = nil, *ry = nil, *rf = nil;
        for (int fused = 0; fused < 2; fused++) {
            const int output = fused ? 2 : 1;
            for (size_t i = 0; i < n; i++) ((float *)x.contents)[i] = value((uint32_t)i) * (producer >= 2 && i % 23 == 0 ? 100000 : 13);
            memset(y.contents, 0xa5, y.length); memset(temp.contents, 0xff, temp.length); memset(flags.contents, 0, flags.length);
            id<MTLBuffer> out = fused ? y : temp;
            id<MTLCommandBuffer> cb = [q commandBuffer]; id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            require(cb && enc, "command creation failed");
            if (producer < 2) {
                [enc setComputePipelineState:ps[producer ? 9 : 8]];
                [enc setBytes:&la length:sizeof(la) atIndex:0]; [enc setBuffer:x offset:0 atIndex:1];
                [enc setBuffer:w offset:0 atIndex:2]; [enc setBuffer:b offset:0 atIndex:3]; [enc setBuffer:out offset:0 atIndex:4];
                [enc setBytes:&ac length:sizeof(ac) atIndex:5]; [enc setBuffer:flags offset:3 * sizeof(int) atIndex:6];
                [enc setBuffer:out offset:0 atIndex:7]; [enc setBytes:&output length:sizeof(output) atIndex:8];
                [enc setBuffer:bias offset:0 atIndex:9];
                [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(H >= 1024 ? 1024 : (H + 31) / 32 * 32, 1, 1)];
            } else {
                [enc setComputePipelineState:ps[10]];
                [enc setBytes:&ga length:sizeof(ga) atIndex:0]; [enc setBuffer:x offset:0 atIndex:1];
                [enc setBuffer:bias offset:0 atIndex:2]; [enc setBuffer:out offset:0 atIndex:3];
                [enc setBytes:&ac length:sizeof(ac) atIndex:4]; [enc setBuffer:flags offset:3 * sizeof(int) atIndex:5];
                [enc setBuffer:out offset:0 atIndex:6]; [enc setBytes:&output length:sizeof(output) atIndex:7];
                [enc dispatchThreads:MTLSizeMake(H, rows, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
            if (!fused) split(enc, ps[6], temp, y, flags, n, ac.lim);
            [enc endEncoding]; [cb commit]; [cb waitUntilCompleted]; require(!cb.error, cb.error.localizedDescription.UTF8String);
            if (!fused) {
                rx = [NSData dataWithBytes:x.contents length:x.length]; ry = [NSData dataWithBytes:y.contents length:y.length];
                rf = [NSData dataWithBytes:flags.contents length:flags.length];
            } else {
                require(!memcmp(rx.bytes, x.contents, x.length), "producer split changed input/writeback");
                require(!memcmp(ry.bytes, y.contents, y.length), "producer split changed half planes");
                require(!memcmp(rf.bytes, flags.contents, flags.length), "producer split changed overflow flags");
            }
            for (size_t i = 0; i < n * 2; i++) require(isfinite(((__fp16 *)y.contents)[i]), "producer wrote nonfinite half");
            for (size_t i = bytes; i < y.length; i++) require(((unsigned char *)y.contents)[i] == 0xa5, "producer split overrun");
            for (int i = 0; i < 8; i++) if (i != 3) require(((int *)flags.contents)[i] == 0, "producer split crossed record flags");
            if (low_limit) require(((int *)flags.contents)[3] == 1, "lowered limit did not exercise overflow");
        }
    }
    puts("norm/GELU split fusion: bit-exact half planes, residual writes, output guards and record-local overflow flags");
}

int main(int argc, char **argv) {
    @autoreleasepool {
        require(argc == 1 || argc == 4, "usage: vision-gemm-bench [T K N]");
        id<MTLDevice> d = MTLCreateSystemDefaultDevice(); require(d != nil, "no Metal device");
        id<MTLCommandQueue> q = [d newCommandQueue]; require(q != nil, "no command queue"); NSError *e = nil;
        NSString *src = [NSString stringWithContentsOfFile:@"metal/clef.metal" encoding:NSUTF8StringEncoding error:&e];
        require(src != nil, e.localizedDescription.UTF8String);
        MTLCompileOptions *o = [MTLCompileOptions new]; o.mathMode = MTLMathModeSafe;
        id<MTLLibrary> lib = [d newLibraryWithSource:src options:o error:&e]; require(lib != nil, e.localizedDescription.UTF8String);
        NSMutableArray *ps = [NSMutableArray new];
        for (NSString *name in @[@"gemm_f32_32x128", @"gemm_f32_acc_32x128", @"gemm_f32_16x128", @"gemm_f32_acc_16x128",
                                 @"vis_gemm_comp_32x128", @"vis_gemm_comp_64x128", @"vis_split_gemm",
                                 @"add_bias", @"layernorm_act", @"layernorm_bias_act", @"vis_bias_gelu"]) {
            id<MTLFunction> fn = [lib newFunctionWithName:name]; require(fn != nil, "missing kernel");
            id<MTLComputePipelineState> p = [d newComputePipelineStateWithFunction:fn error:&e];
            require(p != nil, e.localizedDescription.UTF8String); [ps addObject:p];
        }
        overflow(d, q, ps[6]);
        norm_fusion(d, q, ps);
        producer_split(d, q, ps);
        if (argc == 4) {
            gemm_args a = { atoi(argv[1]), atoi(argv[3]), atoi(argv[2]) };
            require(a.T > 0 && a.T <= 8192 && a.K > 0 && a.K <= 5120 && a.N > 0 && a.N <= 5120, "invalid shape");
            fixture(d, q, ps, a, 0, 10);
        } else {
            for (int t = 1; t <= 65; t++) fixture(d, q, ps, (gemm_args){t, 131, 129}, t % 2, 1);
            const int ts[] = { 352, 1023, 1024, 1025, 4095, 4096, 4097 };
            const int shapes[][2] = { {1536,1152}, {1152,3456}, {1152,1152}, {1152,4304}, {4304,1152}, {4608,4608}, {4608,4096}, {4608,5120} };
            for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); s++)
                for (size_t t = 0; t < sizeof(ts) / sizeof(ts[0]); t++)
                    fixture(d, q, ps, (gemm_args){s >= 5 ? ts[t] / 4 : ts[t], shapes[s][1], shapes[s][0]}, t % 2, 1);
        }
    }
    return 0;
}
