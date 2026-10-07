// FP32 chunked DeltaNet: float64 recurrence, every 32-row offset/tail, NaN guards,
// strong decay and overflowing prefix sums. Compiles the authored production shader.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { int Hk, Hv, dk, dv, C; } gdn_args;
static void require(bool ok, const char *why) {
    if (!ok) { fprintf(stderr, "gdn: %s\n", why); exit(1); }
}
static id<MTLBuffer> buffer(id<MTLDevice> d, size_t floats) {
    id<MTLBuffer> b = [d newBufferWithLength:floats * 4 options:MTLResourceStorageModeShared];
    require(b != nil, "allocation failed");
    return b;
}
static float value(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352d; x ^= x >> 15; x *= 0x846ca68b; x ^= x >> 16;
    return (float)(x & 65535) / 32768 - 1;
}
static void poison(id<MTLBuffer> b) {
    const float nan = NAN;
    memset_pattern4(b.contents, &nan, b.length);
}

static void run(id<MTLCommandQueue> q, NSArray *ps, NSArray *bs, gdn_args a, int T, int offset, int mode) {
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
    require(cb && enc, "command creation failed");
    [enc setBytes:&a length:sizeof(a) atIndex:0];
    for (NSUInteger i = 0; i < bs.count; i++) [enc setBuffer:bs[i] offset:0 atIndex:i + 1];
    if (mode == 0) {
        [enc setComputePipelineState:ps[0]];
        [enc dispatchThreadgroups:MTLSizeMake(a.Hv, 32, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    } else {
        [enc setBuffer:bs[0] offset:(NSUInteger)offset * a.C * 4 atIndex:1];
        [enc setBuffer:bs[1] offset:(NSUInteger)offset * a.Hv * 4 atIndex:2];
        [enc setBuffer:bs[2] offset:(NSUInteger)offset * a.Hv * 4 atIndex:3];
        [enc setBuffer:bs[4] offset:(NSUInteger)offset * a.Hv * 128 * 4 atIndex:5];
        [enc setBytes:&T length:sizeof(T) atIndex:11];
        [enc setComputePipelineState:ps[1]];
        [enc dispatchThreadgroups:MTLSizeMake((T + 31) / 32, a.Hv, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [enc setComputePipelineState:ps[mode + 1]];
        [enc dispatchThreadgroups:MTLSizeMake(a.Hv, mode == 1 ? 8 : 4, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    }
    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
    require(cb.error == nil, cb.error.localizedDescription.UTF8String);
}

static double oracle(gdn_args a, const float *x, const float *beta, const float *gate,
                     const float *out, int T) {
    double worst = 0;
    for (int sample = 0; sample < 24; sample++) {
        const int h = sample * 7 % a.Hv, col = sample * 37 % 128, kh = h / (a.Hv / a.Hk);
        double state[128] = {0};
        for (int t = 0; t < T; t++) {
            const float *xt = x + (size_t)t * a.C;
            double decay = exp((double)gate[(size_t)t * a.Hv + h]), kv = 0, result = 0;
            for (int j = 0; j < 128; j++) {
                state[j] *= decay;
                kv += state[j] * xt[a.Hk * 128 + kh * 128 + j];
            }
            double delta = (xt[2 * a.Hk * 128 + h * 128 + col] - kv) * beta[(size_t)t * a.Hv + h];
            for (int j = 0; j < 128; j++) {
                state[j] += xt[a.Hk * 128 + kh * 128 + j] * delta;
                result += state[j] * xt[kh * 128 + j];
            }
            if (t == 0 || t == T / 2 || t == T - 1) {
                double error = fabs(out[((size_t)t * a.Hv + h) * 128 + col] - result);
                if (!(error <= 2e-5 * (1 + fabs(result)))) {
                    fprintf(stderr, "T=%d Hv=%d t=%d h=%d error=%.9g reference=%.9g\n", T, a.Hv, t, h, error, result);
                    require(false, "float64 recurrence bound");
                }
                if (error > worst) worst = error;
            }
        }
    }
    return worst;
}

static void check(id<MTLDevice> d, id<MTLCommandQueue> q, NSArray *ps, int T, int Hv) {
    gdn_args a = {16, Hv, 128, 128, 4096 + Hv * 128};
    const size_t nx = (size_t)T * a.C, ng = (size_t)T * Hv, ny = ng * 128;
    const size_t nr = (size_t)((T + 31) / 32 * 32) * Hv;
    NSMutableData *data = [NSMutableData dataWithLength:(nx + 2 * ng) * 4];
    require(data != nil, "host allocation failed");
    float *x = data.mutableBytes, *beta = x + nx, *gate = beta + ng;
    for (size_t i = 0; i < nx; i++) x[i] = value((uint32_t)i);
    for (int t = 0; t < T; t++) for (int h = 0; h < a.Hk; h++) {
        double qnorm = 1e-6, knorm = 1e-6;
        for (int j = 0; j < 128; j++) {
            const float qv = x[(size_t)t * a.C + h * 128 + j], kv = x[(size_t)t * a.C + 2048 + h * 128 + j];
            qnorm += (double)qv * qv; knorm += (double)kv * kv;
        }
        for (int j = 0; j < 128; j++) {
            x[(size_t)t * a.C + h * 128 + j] *= 1 / sqrt(qnorm * 128);
            x[(size_t)t * a.C + 2048 + h * 128 + j] *= 1 / sqrt(knorm);
        }
    }
    for (size_t i = 0; i < ng; i++) {
        int h = i % Hv, t = (int)(i / Hv);
        beta[i] = h % 5 == 0 ? 0 : h % 5 == 1 ? 1 : (value((uint32_t)i + 67) + 1) * .5f;
        gate[i] = h % 5 == 0 ? 0 : h % 5 == 1 ? -1e-5f : -fabsf(value((uint32_t)i + 97)) * 5;
        if (h % 5 == 3) gate[i] = t % 32 == 4 ? -1e8f : -.02f;
        if (h % 5 == 4) gate[i] = t % 32 == 4 || t % 32 == 7 ? -FLT_MAX : -.02f;
    }
    NSArray *bs = @[buffer(d, (size_t)(T + 64) * a.C), buffer(d, (size_t)(T + 64) * Hv),
                    buffer(d, (size_t)(T + 64) * Hv), buffer(d, 2), buffer(d, (size_t)(T + 64) * Hv * 128),
                    buffer(d, nr * 128 + 128), buffer(d, nr * 128 + 128), buffer(d, nr * 128 + 128),
                    buffer(d, nr * 32 + 128), buffer(d, nr + 128)];
    NSData *reference[3] = {nil};
    double errors[3] = {0};
    for (int offset = 0; offset < 32; offset++) {
        for (int i = 0; i < 3; i++) poison(bs[i]);
        memcpy((float *)[bs[0] contents] + (size_t)offset * a.C, x, nx * 4);
        memcpy((float *)[bs[1] contents] + (size_t)offset * Hv, beta, ng * 4);
        memcpy((float *)[bs[2] contents] + (size_t)offset * Hv, gate, ng * 4);
        ((int *)[bs[3] contents])[0] = offset; ((int *)[bs[3] contents])[1] = offset + T;
        for (int mode = 0; mode < 3; mode++) {
            for (int i = 4; i < 10; i++) poison(bs[i]);
            run(q, ps, bs, a, T, offset, mode);
            float *y = [bs[4] contents], *active = y + (size_t)offset * Hv * 128;
            for (size_t i = 0; i < (size_t)offset * Hv * 128; i++) require(isnan(y[i]), "prefix overwrite");
            for (size_t i = 0; i < ny; i++) require(isfinite(active[i]), "unwritten/non-finite output");
            for (size_t i = (size_t)(offset + T) * Hv * 128; i < [bs[4] length] / 4; i++) require(isnan(y[i]), "tail overwrite");
            if (mode) for (int b = 5; b < 10; b++) {
                float *v = [bs[b] contents]; size_t count = [bs[b] length] / 4 - 128;
                for (size_t i = 0; i < count; i++) require(isfinite(v[i]), "unwritten/non-finite preprocessing");
                for (size_t i = count; i < count + 128; i++) require(isnan(v[i]), "preprocessing overrun");
            }
            if (!offset) {
                reference[mode] = [NSData dataWithBytes:active length:ny * 4];
                require(reference[mode] != nil, "reference allocation failed");
                errors[mode] = oracle(a, x, beta, gate, active, T);
            } else require(memcmp(reference[mode].bytes, active, ny * 4) == 0, "record-offset dependence");
        }
    }
    require([reference[1] isEqualToData:reference[2]], "chunk value tiles differ");
    printf("T=%d Hv=%d: 32 offsets, strong/overflow decay, NaN guards PASS; f64 %.3g / %.3g / %.3g\n",
           T, Hv, errors[0], errors[1], errors[2]);
    fflush(stdout);
}

int main(int argc, char **argv) {
    @autoreleasepool {
        int first = 1;
        NSString *source = @"metal/clef.metal";
        if (argc > 2 && !strcmp(argv[1], "--source")) { source = @(argv[2]); first = 3; }
        id<MTLDevice> d = MTLCreateSystemDefaultDevice(); require(d != nil, "no GPU");
        id<MTLCommandQueue> q = [d newCommandQueue]; require(q != nil, "no queue");
        NSError *err = nil;
        NSString *s = [NSString stringWithContentsOfFile:source encoding:NSUTF8StringEncoding error:&err];
        require(s != nil, err.localizedDescription.UTF8String);
        MTLCompileOptions *opt = [MTLCompileOptions new]; opt.mathMode = MTLMathModeSafe;
        id<MTLLibrary> lib = [d newLibraryWithSource:s options:opt error:&err];
        require(lib != nil, err.localizedDescription.UTF8String);
        NSMutableArray *ps = [NSMutableArray new];
        for (NSString *name in @[@"gdn_scan_8", @"gdn_chunk_prep_32", @"gdn_chunk_scan_32_16", @"gdn_chunk_scan_32_32"]) {
            id<MTLFunction> fn = [lib newFunctionWithName:name]; require(fn != nil, "missing kernel");
            id<MTLComputePipelineState> pipeline = [d newComputePipelineStateWithFunction:fn error:&err];
            require(pipeline != nil, err.localizedDescription.UTF8String);
            require(pipeline.staticThreadgroupMemoryLength <= d.maxThreadgroupMemoryLength, "scratch limit");
            [ps addObject:pipeline];
        }
        if (first == argc) {
            for (int T = 1; T <= 65; T++) for (int Hv = 32; Hv <= 48; Hv += 16)
                @autoreleasepool { check(d, q, ps, T, Hv); }
        } else for (int i = first; i < argc; i++) {
            int T = atoi(argv[i]); require(T > 0 && T <= 32768, "invalid length");
            for (int Hv = 32; Hv <= 48; Hv += 16) @autoreleasepool { check(d, q, ps, T, Hv); }
        }
    }
    return 0;
}
