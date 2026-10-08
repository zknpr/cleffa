// Vision attention against float64, with poisoned padding, FP16 overflow and output guards.
// Uses the production source; no model or Python dependency. Optional source enables tile probes.
// Build/run: make test-vision-attention
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { int P, E, heads, hd, grid_w, merge; float eps, scale; } vis_args;
typedef struct { int f16; float lim; } act_args;

static void require(bool ok, const char *why) {
    if (!ok) { fprintf(stderr, "vision attention: %s\n", why); exit(1); }
}

static id<MTLBuffer> buffer(id<MTLDevice> dev, size_t n) {
    id<MTLBuffer> b = [dev newBufferWithLength:n options:MTLResourceStorageModeShared];
    require(b != nil, "allocation failed");
    const float nan = NAN;
    memset_pattern4(b.contents, &nan, n);
    return b;
}

static float value(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352d; x ^= x >> 15; x *= 0x846ca68b; x ^= x >> 16;
    return ((float)(x & 0xffffff) / 8388608.0f - 1.0f) * 2.0f;
}

static double finish(id<MTLCommandBuffer> cb, id<MTLComputeCommandEncoder> enc) {
    require(cb != nil && enc != nil, "command creation failed");
    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
    require(cb.error == nil, cb.error.localizedDescription.UTF8String);
    return (cb.GPUEndTime - cb.GPUStartTime) * 1e3;
}

static void prep(id<MTLCommandQueue> queue, id<MTLComputePipelineState> ps, vis_args a,
                 NSArray<id<MTLBuffer>> *in, NSArray<id<MTLBuffer>> *out) {
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    require(cb && enc, "command creation failed");
    [enc setComputePipelineState:ps]; [enc setBytes:&a length:sizeof(a) atIndex:0];
    for (int i = 0; i < 3; i++) {
        [enc setBuffer:in[i] offset:0 atIndex:1 + i];
        [enc setBuffer:out[i] offset:0 atIndex:4 + i];
    }
    const int tail_rows = a.P >= 2048 ? 128 : 32;
    [enc setBytes:&tail_rows length:sizeof(tail_rows) atIndex:7];
    [enc dispatchThreads:MTLSizeMake(36, a.heads, a.P)
         threadsPerThreadgroup:MTLSizeMake(36, 1, 1)];
    finish(cb, enc);
}

static double attend(id<MTLCommandQueue> queue, id<MTLComputePipelineState> ps, vis_args a,
                     NSArray<id<MTLBuffer>> *in, id<MTLBuffer> out, int f32, int f16, float limit) {
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    require(cb && enc, "command creation failed");
    act_args ac = {f16, limit};
    [enc setComputePipelineState:ps]; [enc setBytes:&a length:sizeof(a) atIndex:0];
    for (int i = 0; i < 3; i++) [enc setBuffer:in[i] offset:0 atIndex:1 + i];
    [enc setBuffer:out offset:0 atIndex:4];
    [enc setBytes:&ac length:sizeof(ac) atIndex:5];
    [enc setBuffer:in[3] offset:4 atIndex:6];
    [enc setBuffer:out offset:0 atIndex:7];
    [enc setBytes:&f32 length:sizeof(f32) atIndex:8];
    [enc dispatchThreadgroups:MTLSizeMake((a.P + 31) / 32, a.heads, 1)
         threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    return finish(cb, enc);
}

static void guards(id<MTLBuffer> b, size_t used) {
    const uint32_t *p = (const uint32_t *)((const char *)b.contents + used);
    const float nan = NAN; uint32_t bits; memcpy(&bits, &nan, 4);
    for (size_t i = 0; i < (b.length - used) / 4; i++) require(p[i] == bits, "output guard overwritten");
}

static void check(id<MTLDevice> dev, id<MTLCommandQueue> queue, NSDictionary *ps, int patches) {
    const int heads = 16, hd = 72, E = heads * hd;
    vis_args a = {patches, E, heads, hd, 32, 2, 1e-6f, 1.0f / sqrtf(hd)};
    const size_t n = (size_t)patches * E;
    const size_t guard = 512;
    id<MTLBuffer> qkv = buffer(dev, n * 3 * 4), bias = buffer(dev, 3 * E * 4), freq = buffer(dev, 18 * 4);
    for (size_t i = 0; i < n * 3; i++) {
        // Flat and sharp softmax rows, with full FP32 mantissas in every operand.
        const int part = (i % (3 * E)) / E, h = (i % E) / hd;
        const float scale = part == 2 ? 1.0f : h % 4 == 1 ? 0.01f : h % 4 == 2 ? 4.0f : 1.0f;
        ((float *)qkv.contents)[i] = value((uint32_t)i + 13) * scale;
    }
    for (int i = 0; i < E * 3; i++) ((float *)bias.contents)[i] = value(i + 897) * 0.1f;
    for (int i = 0; i < 18; i++) ((float *)freq.contents)[i] = 1.0f / powf(10000.0f, (float)(2 * i) / 36);
    NSArray *input = @[qkv, bias, freq];
    NSArray *fp = @[buffer(dev, (n + 128 * hd) * 4 + guard), buffer(dev, (n + 128 * hd) * 4 + guard),
                    buffer(dev, (n + 128 * hd) * 4 + guard), buffer(dev, 12)];
    ((int *)((id<MTLBuffer>)fp[3]).contents)[1] = 0;
    prep(queue, ps[@"vis_qkv_rope"], a, input, fp);
    const size_t tail = (patches >= 2048 ? 128 : 32) * hd;
    for (int k = 0; k < 3; k++) {
        const float *p = ((id<MTLBuffer>)fp[k]).contents;
        for (size_t i = 0; i < n; i++) require(isfinite(p[i]), "QKV preparation is non-finite");
        for (size_t i = n; i < n + tail; i++) require(p[i] == 0, "QKV preparation did not clear attention tail");
        guards(fp[k], (n + tail) * 4);
    }
    // Exactly the readable tail is finite; everything beyond it stays poisoned.
    // Small shapes also exercise MPP here, whose wider tail production only needs at P >= 2048.
    for (int k = 0; k < 3; k++) memset((float *)((id<MTLBuffer>)fp[k]).contents + n, 0, 128 * hd * 4);
    id<MTLBuffer> base = buffer(dev, n * 4 + guard), out = buffer(dev, n * 4 + guard);
    double ft = 0, tt = 0;
    for (int rep = 0; rep < 4; rep++) {
        // Alternating order, discard the first pair for shader/data warmup.
        double f, t;
        if (rep & 1) {
            t = attend(queue, ps[@"vis_attention_mpp"], a, fp, out, 1, 1, 65504);
            f = attend(queue, ps[@"vis_attention_mma"], a, fp, base, 1, 1, 65504);
        } else {
            f = attend(queue, ps[@"vis_attention_mma"], a, fp, base, 1, 1, 65504);
            t = attend(queue, ps[@"vis_attention_mpp"], a, fp, out, 1, 1, 65504);
        }
        if (rep) { ft += f / 3; tt += t / 3; }
    }
    guards(base, n * 4); guards(out, n * 4);
    float *got = out.contents, *baseline = base.contents;
    double max_err = 0, base_err = 0, sq = 0, bsq = 0, refsq = 0;
    double *scores = malloc(patches * sizeof(double)); require(scores != NULL, "oracle allocation failed");
    for (int h = 0; h < heads; h++) for (int sample = 0; sample < 5; sample++) {
        const int row = sample * (patches - 1) / 4;
        const float *Q = (const float *)((id<MTLBuffer>)fp[0]).contents + (size_t)h * patches * hd;
        const float *K = (const float *)((id<MTLBuffer>)fp[1]).contents + (size_t)h * patches * hd;
        const float *V = (const float *)((id<MTLBuffer>)fp[2]).contents + (size_t)h * patches * hd;
        double mx = -INFINITY, sum = 0;
        for (int k = 0; k < patches; k++) {
            double s = 0;
            for (int d = 0; d < hd; d++) s += (double)Q[(size_t)row * hd + d] * K[(size_t)k * hd + d];
            scores[k] = s * a.scale; mx = fmax(mx, scores[k]);
        }
        for (int k = 0; k < patches; k++) { scores[k] = exp(scores[k] - mx); sum += scores[k]; }
        for (int d = 0; d < hd; d++) {
            double ref = 0;
            for (int k = 0; k < patches; k++) ref += scores[k] * V[(size_t)k * hd + d];
            ref /= sum;
            size_t i = (size_t)row * E + h * hd + d;
            double err = fabs(got[i] - ref);
            if (!isfinite(got[i]) || err >= 5e-5) {
                fprintf(stderr, "P=%d h=%d row=%d d=%d oracle=%.9g mpp=%.9g mma=%.9g mpp_error=%.3g mma_error=%.3g\n",
                        patches, h, row, d, ref, got[i], baseline[i], err, fabs(baseline[i] - ref));
                require(false, "float64 output tolerance exceeded");
            }
            max_err = fmax(max_err, err); base_err = fmax(base_err, fabs(baseline[i] - ref));
            sq += err * err; bsq += (baseline[i] - ref) * (baseline[i] - ref); refsq += ref * ref;
        }
    }
    for (size_t i = 0; i < n; i++) require(isfinite(got[i]), "non-finite output");
    free(scores);
    // Sharp rows already put the original FP32 kernel near 3e-5 absolute error. Keep an
    // absolute guard and bound aggregate error relative to that control, using the same oracle.
    require(sq <= bsq * 1.05 + 1e-12, "aggregate float64 error exceeds FP32 control by over 5%");
    // The FP16/BF16 epilogues must round the FP32 output, and stop at its logical length.
    for (int f16 = 0; f16 <= 1; f16++) {
        id<MTLBuffer> narrow = buffer(dev, n * 2 + guard);
        attend(queue, ps[@"vis_attention_mpp"], a, fp, narrow, 0, f16, 65504);
        for (size_t i = 0; i < n; i++) {
            uint16_t want;
            if (f16) { _Float16 v = (_Float16)got[i]; memcpy(&want, &v, 2); }
            else { uint32_t u; memcpy(&u, got + i, 4); u += 0x7fff + ((u >> 16) & 1); want = u >> 16; }
            require(((uint16_t *)narrow.contents)[i] == want, "16-bit output differs from rounded FP32");
        }
        guards(narrow, n * 2);
    }
    require(((int *)((id<MTLBuffer>)fp[3]).contents)[1] == 0, "spurious overflow");
    id<MTLBuffer> narrow = buffer(dev, n * 2 + guard);
    attend(queue, ps[@"vis_attention_mpp"], a, fp, narrow, 0, 1, 1e-6f);
    for (size_t i = 0; i < n; i++) require(isfinite((float)((_Float16 *)narrow.contents)[i]), "overflow output is not finite");
    guards(narrow, n * 2);
    const int *flags = ((id<MTLBuffer>)fp[3]).contents;
    const float nan = NAN; uint32_t poison; memcpy(&poison, &nan, 4);
    require(flags[1] == 1 && (uint32_t)flags[0] == poison && (uint32_t)flags[2] == poison, "overflow flag isolation");
    for (int k = 0; k < 3; k++) guards(fp[k], (n + 128 * hd) * 4);
    printf("P=%d fp32=%.3f ms mpp=%.3f ms speedup=%.2fx max_error=%.2e fp32_error=%.2e rel_l2=%.2e PASS\n",
           patches, ft, tt, ft / tt, max_err, base_err, sqrt(sq / refsq));
}

int main(int argc, char **argv) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice(); require(dev != nil, "no device");
        id<MTLCommandQueue> queue = [dev newCommandQueue]; require(queue != nil, "no queue");
        int arg = 1; const char *path = "metal/clef.metal";
        if (arg < argc && !strcmp(argv[arg], "--source")) { require(arg + 1 < argc, "missing source"); path = argv[arg + 1]; arg += 2; }
        NSError *err = nil;
        NSString *source = [NSString stringWithContentsOfFile:@(path) encoding:NSUTF8StringEncoding error:&err];
        require(source != nil, err.localizedDescription.UTF8String);
        MTLCompileOptions *options = [MTLCompileOptions new]; options.mathMode = MTLMathModeSafe;
        id<MTLLibrary> lib = [dev newLibraryWithSource:source options:options error:&err];
        require(lib != nil, err.localizedDescription.UTF8String);
        NSMutableDictionary *ps = [NSMutableDictionary new];
        for (NSString *name in @[@"vis_qkv_rope", @"vis_attention_mma", @"vis_attention_mpp"]) {
            id<MTLFunction> fn = [lib newFunctionWithName:name]; require(fn != nil, "missing kernel");
            id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:fn error:&err];
            require(p != nil, err.localizedDescription.UTF8String);
            require(p.threadExecutionWidth == 32 && p.maxTotalThreadsPerThreadgroup >= 128 && p.staticThreadgroupMemoryLength <= dev.maxThreadgroupMemoryLength, "unsupported pipeline geometry");
            ps[name] = p;
        }
        if (arg == argc) { int sizes[] = {4, 28, 32, 36, 124, 128, 132, 256, 352, 1024, 2044, 2048, 2052, 4092, 4096, 4100};
            for (size_t i = 0; i < sizeof(sizes) / sizeof(*sizes); i++) @autoreleasepool { check(dev, queue, ps, sizes[i]); }
        } else for (; arg < argc; arg++) @autoreleasepool {
            int n = atoi(argv[arg]); require(n > 0 && n <= 65536 && n % 4 == 0, "patch count must be a positive multiple of four up to 65536");
            check(dev, queue, ps, n);
        }
    }
    return 0;
}
