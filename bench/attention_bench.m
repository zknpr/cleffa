// Time the production attention kernel and check it against a float64 CPU oracle.
// An optional older .metal source is run on the same inputs, alternating dispatch order.
// --tu KERNEL checks a tensor-unit kernel (attention_tu...) instead: float64 error, packed
// records bitwise equal to the same records alone, overflow isolation, attn_prep's hi/lo split,
// and the prefix-cache layout (K/V planes indexed by the record's tokens, only the last rows
// computed). It is timed against the FP32 kernel the engine would otherwise select.
// Build: make attention-bench
// Usage: ./attention-bench [--source FILE.metal] [--baseline OLD.metal] [--reuse 2|4 | --prefetch 64 | --tu KERNEL]
//                          [--values grid|fp32] [T ...]
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { int nh, nkv, hd, T; float scale; } attn_args;
typedef struct { int f16; float lim; } act_args;
typedef struct { int nh, nkv, hd, n_rot, row; float eps; } attn_prep_args;
typedef struct { int nh, nkv, hd, n_rot, row; float eps, lim; int kvT, koff; } attn_split_args;
typedef struct { int nh, nkv, T, kvT; float scale; } attn_tu_args;
typedef struct { int query_blocks, key_block; } attn_variant;
static const attn_variant original = {0, 32};
static bool full_fp32_values = false;

static void require(bool ok, const char *what) {
    if (!ok) { fprintf(stderr, "attention: %s\n", what); exit(1); }
}

static id<MTLComputePipelineState> kernel_pipeline(id<MTLDevice> dev, const char *path, NSString *name) {
    NSError *err = nil;
    NSString *source = [NSString stringWithContentsOfFile:@(path) encoding:NSUTF8StringEncoding error:&err];
    require(source != nil, err.localizedDescription.UTF8String);
    MTLCompileOptions *options = [MTLCompileOptions new];
    options.mathMode = MTLMathModeSafe;
    id<MTLLibrary> lib = [dev newLibraryWithSource:source options:options error:&err];
    require(lib != nil, err.localizedDescription.UTF8String);
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    require(fn != nil, "missing attention kernel");
    id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:fn error:&err];
    require(ps != nil, err.localizedDescription.UTF8String);
    return ps;
}

static id<MTLComputePipelineState> pipeline(id<MTLDevice> dev, const char *path, attn_variant variant) {
    return kernel_pipeline(dev, path, variant.key_block == 64 ? @"attention_prefetch_64" :
        variant.query_blocks ? [NSString stringWithFormat:@"attention_reuse_%d", variant.query_blocks] : @"attention_fa");
}

static id<MTLBuffer> buffer(id<MTLDevice> dev, size_t size) {
    id<MTLBuffer> b = [dev newBufferWithLength:size options:MTLResourceStorageModeShared];
    require(b != nil, "buffer allocation failed");
    memset(b.contents, 0, size);
    return b;
}

// n_blk > 0 dispatches a tensor-unit kernel: b comes from tu_inputs, with the query-block list
// in the seq_start slot.
static double run_blocks(id<MTLCommandQueue> queue, id<MTLComputePipelineState> ps, attn_args a,
                  NSArray<id<MTLBuffer>> *b, int f16, attn_variant variant, int n_blk) {
    act_args ac = { f16, 65504.0f };
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    require(cb != nil && enc != nil, "command creation failed");
    [enc setComputePipelineState:ps];
    [enc setBytes:&a length:sizeof(a) atIndex:0];
    for (int i = 0; i < 6; i++) [enc setBuffer:b[i] offset:0 atIndex:i + 1];
    [enc setBytes:&ac length:sizeof(ac) atIndex:7];
    [enc setBuffer:b[6] offset:0 atIndex:8];
    if (n_blk) {
        attn_tu_args ta = { a.nh, a.nkv, a.T, a.T, a.scale };   // K/V planes laid out like Q's
        [enc setBytes:&ta length:sizeof(ta) atIndex:0];
        require(128 <= ps.maxTotalThreadsPerThreadgroup, "attention threadgroup exceeds pipeline limit");
        require(ps.staticThreadgroupMemoryLength <= queue.device.maxThreadgroupMemoryLength, "attention scratch exceeds device limit");
        [enc dispatchThreadgroups:MTLSizeMake(n_blk, a.nh, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    } else {
        int grp = variant.query_blocks ? variant.query_blocks : a.nh / a.nkv;
        require((NSUInteger)32 * grp <= ps.maxTotalThreadsPerThreadgroup, "attention threadgroup exceeds pipeline limit");
        require(grp * (8 * variant.key_block + 64) * sizeof(float) + ps.staticThreadgroupMemoryLength <= queue.device.maxThreadgroupMemoryLength,
                "attention scratch exceeds device limit");
        [enc setThreadgroupMemoryLength:grp * (8 * variant.key_block + 64) * sizeof(float) atIndex:0];
        int rows = variant.query_blocks ? 8 * variant.query_blocks : 8;
        [enc dispatchThreadgroups:MTLSizeMake((a.T + rows - 1) / rows, variant.query_blocks ? a.nh : a.nkv, 1)
            threadsPerThreadgroup:MTLSizeMake(32 * grp, 1, 1)];
    }
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    require(cb.error == nil, cb.error.localizedDescription.UTF8String);
    return (cb.GPUEndTime - cb.GPUStartTime) * 1e3;
}

static float value(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352d; x ^= x >> 15; x *= 0x846ca68b; x ^= x >> 16;
    if (full_fp32_values) {
        // The original 16-bit grid is unusually easy to represent as two FP16
        // planes. Vary all mantissa bits and magnitudes, including near zero.
        uint32_t bits = (x & 0x807fffffu) | ((110u + (x >> 24) % 18u) << 23);
        float result;
        memcpy(&result, &bits, sizeof(result));
        return result;
    }
    return (float)(x & 0xffff) / 32768.0f - 1.0f;
}

static NSArray<id<MTLBuffer>> *inputs(id<MTLDevice> dev, attn_args a, const int *lengths, int n, int key_block) {
    size_t nq = (size_t)a.nh * a.T * a.hd, nk = (size_t)a.nkv * a.T * a.hd;
    NSArray *b = @[buffer(dev, (nq + 128 * a.hd) * 4), buffer(dev, (nk + 128 * a.hd) * 4),
                   buffer(dev, (nk + 128 * a.hd) * 4), buffer(dev, nq * 4),
                   buffer(dev, a.T * 4), buffer(dev, (nq + 16) * 2), buffer(dev, a.T * 4)];
    float *q = [b[0] contents], *k = [b[1] contents], *v = [b[2] contents], *g = [b[3] contents];
    int *seq = [b[4] contents];
    int start = 0;
    for (int r = 0; r < n; r++) {
        for (int t = 0; t < lengths[r]; t++) {
            seq[start + t] = start;
            for (int h = 0; h < a.nh; h++) for (int d = 0; d < a.hd; d++) {
                uint32_t seed = t * 7907u + h * 104729u + d * 1543u;
                // Alternate diffuse and sharp distributions, with varied gates and signed V.
                q[((size_t)h * a.T + start + t) * a.hd + d] = value(seed) * (h % 2 ? 1 : 12);
                g[((size_t)(start + t) * a.nh + h) * a.hd + d] = value(seed + 1) * 6;
            }
            for (int h = 0; h < a.nkv; h++) for (int d = 0; d < a.hd; d++) {
                uint32_t seed = t * 3137u + h * 65537u + d * 2311u;
                k[((size_t)h * a.T + start + t) * a.hd + d] = value(seed);
                v[((size_t)h * a.T + start + t) * a.hd + d] = value(seed + 2) * 3;
            }
        }
        start += lengths[r];
    }
    require(start == a.T, "bad fixture length");
    uint16_t *out = [b[5] contents];
    // NaN in both FP16 and BF16, so either mode detects unwritten output slots.
    for (size_t i = 0; i < nq + 16; i++) out[i] = 0x7fc1;
    // Poison beyond the variant's required tail, including past production's
    // allocation of 64 spare rows. Prefetch must not read a third score block.
    for (size_t i = nq + 8 * a.hd; i < nq + 128 * a.hd; i++) q[i] = NAN;
    for (size_t i = nk + key_block * a.hd; i < nk + 128 * a.hd; i++) k[i] = v[i] = NAN;
    return b;
}

// split16 of metal/clef.metal on the CPU; prep_check requires the kernel to match it bit for bit.
static void split16(float x, __fp16 *hi, __fp16 *lo, float lim, int *flag) {
    if (!(fabsf(x) <= lim)) { *flag = 1; *hi = (__fp16)copysignf(65504.0f, x); *lo = 0; return; }
    *hi = (__fp16)x;
    *lo = (__fp16)(x - (float)*hi);
}

// x [heads][T][hd] -> hi plane, then lo plane
static void split_plane(const float *x, __fp16 *planes, int heads, attn_args a, float lim, const int *seq, int *flags) {
    const size_t n = (size_t)heads * a.T * a.hd;
    for (int h = 0; h < heads; h++) for (int t = 0; t < a.T; t++) for (int d = 0; d < a.hd; d++) {
        const size_t i = ((size_t)h * a.T + t) * a.hd + d;
        split16(x[i], planes + i, planes + n + i, lim, flags + seq[t]);
    }
}

// What attn_prep with split hands attention_tu for the FP32 fixture `b`: hi/lo half planes of
// Q, K and V, the overflow flags the split raises, and the query-block list. Output and flag
// buffers are shared with `b`, so check_rows() works on either.
static NSArray<id<MTLBuffer>> *tu_inputs(id<MTLDevice> dev, attn_args a, NSArray<id<MTLBuffer>> *b,
                                         const int *lengths, int n, int *n_blk) {
    const size_t nq = (size_t)a.nh * a.T * a.hd, nk = (size_t)a.nkv * a.T * a.hd;
    NSArray *t = @[buffer(dev, (nq + 128 * a.hd) * 4), buffer(dev, (nk + 128 * a.hd) * 4), buffer(dev, (nk + 128 * a.hd) * 4),
                   b[3], buffer(dev, (size_t)a.T * 32), b[5], b[6]];
    const int *seq = b[4].contents;
    int *flags = b[6].contents;
    split_plane(b[0].contents, [t[0] contents], a.nh, a, 65504.0f, seq, flags);
    split_plane(b[1].contents, [t[1] contents], a.nkv, a, 65504.0f, seq, flags);
    split_plane(b[2].contents, [t[2] contents], a.nkv, a, 65504.0f, seq, flags);
    // Production zeroes 16 float rows after Q's planes and all 64 it allocates after K/V's, of
    // which the tiles read 31 and 127 half rows. Poison everything past those reads, into
    // rows production does not allocate.
    float *q = [t[0] contents];
    for (size_t i = nq + 16 * a.hd; i < nq + 128 * a.hd; i++) q[i] = NAN;
    for (int i = 1; i <= 2; i++) {
        uint16_t *kv = [t[i] contents];
        for (size_t j = 2 * nk + 127 * a.hd; j < 2 * nk + 256 * a.hd; j++) kv[j] = 0x7fc1;
    }
    int *blk = [t[4] contents], start = 0;
    *n_blk = 0;
    for (int r = 0; r < n; r++) {
        for (int i0 = (lengths[r] - 1) / 32 * 32; i0 >= 0; i0 -= 32, (*n_blk)++) {
            int *e = blk + 8 * *n_blk;   // qbase, record length, first query token, kbase, flag row
            e[0] = start; e[1] = lengths[r]; e[2] = i0; e[3] = start; e[4] = start;
        }
        start += lengths[r];
    }
    return t;
}

static float decode(uint16_t x, int f16) {
    if (f16) { __fp16 h; memcpy(&h, &x, 2); return h; }
    uint32_t bits = (uint32_t)x << 16;
    float f; memcpy(&f, &bits, 4); return f;
}

// tu: the split saturates a flagged record's operands, so its (discarded) output is finite
// but not the oracle's; those samples are skipped.
static void check_rows(attn_args a, NSArray<id<MTLBuffer>> *b, int f16, bool overflow_case, bool tu) {
    const float *q = b[0].contents, *k = b[1].contents, *v = b[2].contents, *g = b[3].contents;
    const int *seq = b[4].contents, *ovf = b[6].contents;
    const uint16_t *out = b[5].contents;
    size_t nq = (size_t)a.T * a.nh * a.hd;
    for (size_t i = 0; i < nq; i++) require(isfinite(decode(out[i], f16)), "non-finite/unwritten output");
    for (size_t i = nq; i < nq + 16; i++) require(out[i] == 0x7fc1, "output overrun");
    int record = -1;
    for (int i = 0; i < a.T; i++) {
        if (seq[i] == i) record++;
        int expected = overflow_case && f16 && record % 2 && seq[i] == i;
        require(ovf[i] == expected, "incorrect per-record overflow flag");
    }
    double *scores = malloc((size_t)a.T * sizeof(double));
    require(scores != NULL, "oracle allocation failed");
    double worst = 0;
    for (int sample = 0; sample < 24; sample++) {
        int t = sample == 0 ? a.T - 1 : (sample * 7919) % a.T;
        int h = sample % a.nh, kh = h / (a.nh / a.nkv), d = (sample * 137) % a.hd;
        if (tu && ovf[seq[t]]) continue;
        double mx = -INFINITY;
        for (int s = seq[t]; s <= t; s++) {
            double dot = 0;
            for (int j = 0; j < a.hd; j++) dot += (double)q[((size_t)h * a.T + t) * a.hd + j] * k[((size_t)kh * a.T + s) * a.hd + j];
            scores[s] = dot * a.scale;
            if (scores[s] > mx) mx = scores[s];
        }
        double sum = 0, acc = 0;
        for (int s = seq[t]; s <= t; s++) {
            double p = exp(scores[s] - mx);
            sum += p; acc += p * v[((size_t)kh * a.T + s) * a.hd + d];
        }
        size_t o = ((size_t)t * a.nh + h) * a.hd + d;
        double want = acc / sum / (1 + exp(-(double)g[o]));
        if (f16 && fabs(want) > 65504) want = copysign(65504, want);
        double error = fabs(decode(out[o], f16) - want);
        double tol = fabs(want) * (f16 ? 0.0005 : 0.004) + 2e-6;
        if (error > worst) worst = error;
        if (!(error <= tol)) {
            fprintf(stderr, "T=%d h=%d t=%d d=%d got=%.9g f64=%.9g error=%.3g limit=%.3g\n", a.T, h, t, d, decode(out[o], f16), want, error, tol);
            exit(1);
        }
    }
    free(scores);
    printf(" f64_max_abs=%.3g", worst);
}

// main's signatures (tests/test_prefix_attention.m includes this file)
__attribute__((unused)) static double run(id<MTLCommandQueue> queue, id<MTLComputePipelineState> ps, attn_args a,
                  NSArray<id<MTLBuffer>> *b, int f16, attn_variant variant) {
    return run_blocks(queue, ps, a, b, f16, variant, 0);
}
__attribute__((unused)) static void check(attn_args a, NSArray<id<MTLBuffer>> *b, int f16, bool overflow_case) {
    check_rows(a, b, f16, overflow_case, false);
}

static void scale_values(attn_args a, NSArray<id<MTLBuffer>> *b, int start, int length) {
    float *v = b[2].contents;
    for (int h = 0; h < a.nkv; h++) for (int t = start; t < start + length; t++) {
        for (int d = 0; d < a.hd; d++) v[((size_t)h * a.T + t) * a.hd + d] *= 100000;
    }
}

static void packed_check(id<MTLDevice> dev, id<MTLCommandQueue> queue, id<MTLComputePipelineState> ps,
                         id<MTLComputePipelineState> baseline, int nh, int f16, attn_variant variant, bool overflow_case, bool tu) {
    // All 32 query-row offsets, key-block edges, and several records in one tile.
    const int lengths[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                            2, 3, 4, 5, 6, 7, 8, 9, 31, 32, 33, 63, 65, 263 };
    int n = sizeof(lengths) / sizeof(lengths[0]), T = 0;
    for (int i = 0; i < n; i++) T += lengths[i];
    attn_args a = { nh, 4, 256, T, 1.0f / 16 };
    NSArray *packed = inputs(dev, a, lengths, n, variant.key_block);
    int offset = 0;
    for (int i = 0; i < n; i++) {
        if (overflow_case && i % 2) scale_values(a, packed, offset, lengths[i]);
        offset += lengths[i];
    }
    NSData *want = nil, *flags = nil;
    if (baseline && !tu) {
        run_blocks(queue, baseline, a, packed, f16, original, 0);
        want = [NSData dataWithBytes:[packed[5] contents] length:(size_t)T * nh * 256 * 2];
        flags = [NSData dataWithBytes:[packed[6] contents] length:T * 4];
        memset([packed[6] contents], 0, T * 4);
        uint16_t *out = [packed[5] contents];
        for (size_t i = 0; i < (size_t)T * nh * 256 + 16; i++) out[i] = 0x7fc1;
    }
    int n_blk = 0;
    run_blocks(queue, ps, a, tu ? tu_inputs(dev, a, packed, lengths, n, &n_blk) : packed, f16, variant, n_blk);
    if (want) {
        require(memcmp(want.bytes, [packed[5] contents], want.length) == 0, "packed candidate changed baseline bits");
        require(memcmp(flags.bytes, [packed[6] contents], flags.length) == 0, "candidate changed overflow flags");
    }
    printf("packed nh=%d f16=%d overflow=%d", nh, f16, overflow_case); check_rows(a, packed, f16, overflow_case, tu);
    offset = 0;
    for (int i = 0; i < n; i++) {
        a.T = lengths[i];
        NSArray *single = inputs(dev, a, lengths + i, 1, variant.key_block);
        if (overflow_case && i % 2) scale_values(a, single, 0, a.T);
        run_blocks(queue, ps, a, tu ? tu_inputs(dev, a, single, lengths + i, 1, &n_blk) : single, f16, variant, n_blk);
        require(memcmp((uint16_t *)[packed[5] contents] + (size_t)offset * nh * 256, [single[5] contents], (size_t)a.T * nh * 256 * 2) == 0,
                "packing changed attention bits");
        const int *single_flags = [single[6] contents], *packed_flags = [packed[6] contents];
        require(memcmp(single_flags, packed_flags + offset, (size_t)a.T * sizeof(int)) == 0,
                "packing changed overflow flags");
        offset += lengths[i];
    }
    puts(" batch_bits=identical");
}

static void long_packed_check(id<MTLDevice> dev, id<MTLCommandQueue> queue, id<MTLComputePipelineState> ps,
                              id<MTLComputePipelineState> baseline, int nh, int f16, attn_variant variant, bool tu) {
    // Long records cross many key blocks; ragged boundaries put two records in
    // one reused query tile. Compare against the original kernel after poisoning
    // every candidate output, so skipped stores cannot inherit a passing value.
    // A tensor-unit kernel does not reproduce those bits: each record must instead
    // equal its own run alone.
    const int lengths[][8] = {{512, 512, 512, 512, 512, 512, 512, 512},
                              {1382, 1382, 1382, 1382, 1382, 1382, 1382, 1382},
                              {4510, 346}, {8072, 8072}};
    const int counts[] = {8, 8, 2, 2};
    for (int c = 0; c < 4; c++) @autoreleasepool {
        int T = 0;
        for (int i = 0; i < counts[c]; i++) T += lengths[c][i];
        attn_args a = {nh, 4, 256, T, 1.0f / 16};
        NSArray<id<MTLBuffer>> *b = inputs(dev, a, lengths[c], counts[c], variant.key_block);
        if (tu) {
            int n_blk = 0, single_length = 0;
            run_blocks(queue, ps, a, tu_inputs(dev, a, b, lengths[c], counts[c], &n_blk), f16, variant, n_blk);
            NSArray<id<MTLBuffer>> *single = nil;
            for (int i = 0, offset = 0; i < counts[c]; offset += lengths[c][i++]) {
                attn_args s = a;
                s.T = lengths[c][i];
                if (s.T != single_length) {   // equal lengths are equal fixtures
                    single = inputs(dev, s, lengths[c] + i, 1, variant.key_block);
                    run_blocks(queue, ps, s, tu_inputs(dev, s, single, lengths[c] + i, 1, &n_blk), f16, variant, n_blk);
                    single_length = s.T;
                }
                require(memcmp((uint16_t *)b[5].contents + (size_t)offset * nh * 256, single[5].contents, (size_t)s.T * nh * 256 * 2) == 0,
                        "long packing changed attention bits");
            }
            printf("long-packed case=%d nh=%d f16=%d T=%d", c, nh, f16, T);
            check_rows(a, b, f16, false, true);
            puts(" batch_bits=identical"); fflush(stdout);
            continue;
        }
        run_blocks(queue, baseline, a, b, f16, original, 0);
        const size_t elements = (size_t)T * nh * 256;
        NSData *want = [NSData dataWithBytes:b[5].contents length:elements * 2];
        NSData *flags = [NSData dataWithBytes:b[6].contents length:T * 4];
        uint16_t *output = b[5].contents;
        for (size_t i = 0; i < elements + 16; i++) output[i] = 0x7fc1;
        memset(b[6].contents, 0, T * 4);
        run_blocks(queue, ps, a, b, f16, variant, 0);
        require(memcmp(want.bytes, output, want.length) == 0, "long packed candidate changed baseline bits");
        require(memcmp(flags.bytes, b[6].contents, flags.length) == 0, "long packed candidate changed overflow flags");
        printf("long-packed case=%d nh=%d f16=%d T=%d", c, nh, f16, T);
        check_rows(a, b, f16, false, false);
        puts(" baseline_bits=identical"); fflush(stdout);
    }
}

// Prefix-cache layout: the K/V planes hold the whole record by token index in a buffer with more
// rows than the record (as a cache entry does), while Q, G and the output hold only tokens L..T-1.
// Those rows must equal, bit for bit, the rows of an ordinary pass over the whole record.
static void cached_layout_check(id<MTLDevice> dev, id<MTLCommandQueue> queue, id<MTLComputePipelineState> ps, int nh) {
    enum { T = 733, L = 416, CAP = 1024, NKV = 4, HD = 256, TW = T - L };   // L: a query block boundary inside a key tile
    const int len = T;
    attn_args a = { nh, NKV, HD, T, 1.0f / 16 };
    NSArray<id<MTLBuffer>> *full = inputs(dev, a, &len, 1, 32);
    int n_blk = 0;
    NSArray<id<MTLBuffer>> *ft = tu_inputs(dev, a, full, &len, 1, &n_blk);
    run_blocks(queue, ps, a, ft, 1, original, n_blk);

    const size_t row = HD * 2;   // bytes of one half row
    id<MTLBuffer> q = buffer(dev, (size_t)2 * nh * TW * row + 128 * row), g = buffer(dev, (size_t)TW * nh * HD * 4);
    id<MTLBuffer> k = buffer(dev, (size_t)2 * NKV * CAP * row + 128 * row), v = buffer(dev, (size_t)2 * NKV * CAP * row + 128 * row);
    id<MTLBuffer> out = buffer(dev, (size_t)TW * nh * HD * 2), blocks = buffer(dev, (size_t)TW * 32), ovf = buffer(dev, TW * 4);
    for (int pl = 0; pl < 2; pl++) {
        for (int h = 0; h < nh; h++)   // Q planes: rows L.. of the full planes
            memcpy((char *)q.contents + ((size_t)pl * nh + h) * TW * row,
                   (char *)ft[0].contents + (((size_t)pl * nh + h) * T + L) * row, TW * row);
        for (int h = 0; h < NKV; h++) {   // K/V planes: all rows, at the entry's row stride
            memcpy((char *)k.contents + ((size_t)pl * NKV + h) * CAP * row, (char *)ft[1].contents + ((size_t)pl * NKV + h) * T * row, T * row);
            memcpy((char *)v.contents + ((size_t)pl * NKV + h) * CAP * row, (char *)ft[2].contents + ((size_t)pl * NKV + h) * T * row, T * row);
        }
    }
    memcpy(g.contents, (float *)full[3].contents + (size_t)L * nh * HD, (size_t)TW * nh * HD * 4);
    uint16_t *o = out.contents;
    for (size_t i = 0; i < (size_t)TW * nh * HD; i++) o[i] = 0x7fc1;
    int *e = blocks.contents, nb = 0;
    for (int i0 = (T - 1) / 32 * 32; i0 >= L; i0 -= 32, nb++, e += 8) { e[0] = -L; e[1] = T; e[2] = i0; e[3] = 0; e[4] = 0; }
    attn_tu_args ta = { nh, NKV, TW, CAP, a.scale };
    act_args ac = { 1, 65504.0f };
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    require(cb != nil && enc != nil, "command creation failed");
    [enc setComputePipelineState:ps];
    [enc setBytes:&ta length:sizeof(ta) atIndex:0];
    NSArray<id<MTLBuffer>> *bind = @[q, k, v, g, blocks, out];
    for (int i = 0; i < 6; i++) [enc setBuffer:bind[i] offset:0 atIndex:i + 1];
    [enc setBytes:&ac length:sizeof(ac) atIndex:7];
    [enc setBuffer:ovf offset:0 atIndex:8];
    [enc dispatchThreadgroups:MTLSizeMake(nb, nh, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    require(cb.error == nil, cb.error.localizedDescription.UTF8String);
    require(memcmp(o, (uint16_t *)full[5].contents + (size_t)L * nh * HD, (size_t)TW * nh * HD * 2) == 0,
            "cached K/V layout changed attention bits");
    printf("cached-layout nh=%d T=%d from=%d rows=identical\n", nh, T, L);
}

// attn_prep_split against the CPU split of attn_prep's FP32 output: planes, layout, saturation
// and per-record flags. Only the middle record's V overflows FP16; a lowered limit flags all.
static void prep_check(id<MTLDevice> dev, id<MTLCommandQueue> queue, const char *source, int nh) {
    enum { T = 77, NKV = 4, HD = 256 };
    const int lengths[] = { 5, 40, 32 }, row = nh * 2 * HD + 2 * NKV * HD;
    id<MTLComputePipelineState> ps32 = kernel_pipeline(dev, source, @"attn_prep");
    id<MTLComputePipelineState> ps16 = kernel_pipeline(dev, source, @"attn_prep_split");
    // pos is [T][3]: interleaved M-RoPE, one value per axis; text tokens repeat the same position
    id<MTLBuffer> qkv = buffer(dev, (size_t)T * row * 4), w = buffer(dev, HD * 4), pos = buffer(dev, T * 12);
    id<MTLBuffer> freq = buffer(dev, 32 * 4), seq = buffer(dev, T * 4);
    float *x = qkv.contents, *wv = w.contents, *fr = freq.contents;
    int *p = pos.contents, *s = seq.contents;
    for (size_t i = 0; i < (size_t)T * row; i++) x[i] = value((uint32_t)i * 2654435761u) * 3;
    for (int d = 0; d < HD; d++) wv[d] = 2 + value(d * 97u);
    for (int j = 0; j < 32; j++) fr[j] = powf(10000.0f, -(float)j / 32);
    for (int r = 0, start = 0; r < 3; start += lengths[r++]) {
        for (int t = 0; t < lengths[r]; t++) { s[start + t] = start; for (int k = 0; k < 3; k++) p[3 * (start + t) + k] = t; }
    }
    for (int t = 5; t < 45; t++) for (int i = nh * 2 * HD + NKV * HD; i < row; i++) x[(size_t)t * row + i] *= 100000;
    const attn_args a = { nh, NKV, HD, T, 0 };
    const size_t nq = (size_t)nh * T * HD, nk = (size_t)NKV * T * HD;
    for (int lowered = 0; lowered <= 1; lowered++) {
        const float lim = lowered ? 0.5f : 65504.0f;
        NSMutableArray<NSArray<id<MTLBuffer>> *> *out = [NSMutableArray new];
        for (int split = 0; split <= 1; split++) {
            NSArray<id<MTLBuffer>> *o = @[buffer(dev, (nq + 64 * HD) * 4), buffer(dev, (nk + 64 * HD) * 4),
                                          buffer(dev, (nk + 64 * HD) * 4), buffer(dev, nq * 4), buffer(dev, T * 4)];
            attn_prep_args pa = { nh, NKV, HD, 64, row, 1e-6f };
            attn_split_args sa = { nh, NKV, HD, 64, row, 1e-6f, lim, T, 0 };
            int t = T;
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            require(cb != nil && enc != nil, "command creation failed");
            [enc setComputePipelineState:split ? ps16 : ps32];
            if (split) [enc setBytes:&sa length:sizeof(sa) atIndex:0];
            else [enc setBytes:&pa length:sizeof(pa) atIndex:0];
            [enc setBuffer:qkv offset:0 atIndex:1];
            [enc setBuffer:w offset:0 atIndex:2];
            [enc setBuffer:w offset:0 atIndex:3];
            [enc setBuffer:pos offset:0 atIndex:4];
            [enc setBuffer:freq offset:0 atIndex:5];
            for (int i = 0; i < 4; i++) [enc setBuffer:o[i] offset:0 atIndex:6 + i];
            [enc setBytes:&t length:sizeof(t) atIndex:10];
            [enc setBuffer:seq offset:0 atIndex:11];
            [enc setBuffer:o[4] offset:0 atIndex:12];
            [enc dispatchThreadgroups:MTLSizeMake(T, nh + NKV, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            require(cb.error == nil, cb.error.localizedDescription.UTF8String);
            [out addObject:o];
        }
        int flags[T] = { 0 };
        const int heads[3] = { nh, NKV, NKV };
        for (int i = 0; i < 3; i++) {
            const size_t n = (size_t)heads[i] * T * HD;
            __fp16 *want = calloc(2 * n, 2);
            require(want != NULL, "prep allocation failed");
            split_plane(out[0][i].contents, want, heads[i], a, lim, s, flags);
            require(memcmp(want, out[1][i].contents, 2 * n * 2) == 0, "attn_prep split differs from the CPU split");
            free(want);
        }
        require(memcmp(out[0][3].contents, out[1][3].contents, nq * 4) == 0, "split changed the gate");
        require(memcmp(flags, out[1][4].contents, T * 4) == 0, "attn_prep split raised different overflow flags");
        for (int t = 0; t < T; t++) {
            require(((const int *)out[0][4].contents)[t] == 0, "FP32 attn_prep raised an overflow flag");
            require(flags[t] == ((t == 5 || (lowered && s[t] == t)) ? 1 : 0), "overflow flag outside the overflowing record");
        }
    }
    printf("prep nh=%d split=cpu_bits flags=per-record\n", nh);
}

int main(int argc, char **argv) {
    @autoreleasepool {
        const char *old = NULL, *source = "metal/clef.metal", *tu = NULL;
        int first = 1;
        attn_variant variant = original;
        while (first + 1 < argc && argv[first][0] == '-') {
            if (!strcmp(argv[first], "--baseline")) old = argv[first + 1];
            else if (!strcmp(argv[first], "--source")) source = argv[first + 1];
            else if (!strcmp(argv[first], "--values")) {
                require(!strcmp(argv[first + 1], "grid") || !strcmp(argv[first + 1], "fp32"), "values must be grid or fp32");
                full_fp32_values = !strcmp(argv[first + 1], "fp32");
            }
            else if (!strcmp(argv[first], "--reuse")) {
                require(!strcmp(argv[first + 1], "2") || !strcmp(argv[first + 1], "4"), "reuse must be 2 or 4");
                variant = (attn_variant){atoi(argv[first + 1]), 32};
            }
            else if (!strcmp(argv[first], "--prefetch")) {
                require(!strcmp(argv[first + 1], "64"), "prefetch must be 64");
                variant = (attn_variant){4, 64};
            }
            else if (!strcmp(argv[first], "--tu")) tu = argv[first + 1];
            else require(false, "unknown option");
            first += 2;
        }
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        require(dev != nil, "no Metal device");
        id<MTLCommandQueue> queue = [dev newCommandQueue];
        require(queue != nil, "no Metal queue");
        require(!tu || !variant.query_blocks, "--tu excludes --reuse and --prefetch");
        id<MTLComputePipelineState> ps = tu ? kernel_pipeline(dev, source, @(tu)) : pipeline(dev, source, variant);
        id<MTLComputePipelineState> baseline = old ? pipeline(dev, old, original) : nil;
        // the engine's FP32 kernel for one record of at least 1,024 tokens
        const attn_variant prefetch = {4, 64};
        id<MTLComputePipelineState> long_base = tu && old ? pipeline(dev, old, prefetch) : nil;
        if (tu) for (int nh = 16; nh <= 24; nh += 8) { prep_check(dev, queue, source, nh); cached_layout_check(dev, queue, ps, nh); }
        // The engine runs tensor-unit attention only with FP16 output.
        for (int nh = 16; nh <= 24; nh += 8) for (int f16 = tu ? 1 : 0; f16 <= 1; f16++) {
            for (int overflow = 0; overflow <= 1; overflow++) {
                packed_check(dev, queue, ps, baseline, nh, f16, variant, overflow, tu != NULL);
            }
            if (tu || (variant.query_blocks && baseline)) long_packed_check(dev, queue, ps, baseline, nh, f16, variant, tu != NULL);
        }
        for (int arg = first; arg < argc; arg++) {
            char *end;
            long parsed = strtol(argv[arg], &end, 10);
            require(*argv[arg] && !*end && parsed > 0 && parsed <= 32768, "T must be in 1..32768");
            int T = (int)parsed;
            for (int nh = 16; nh <= 24; nh += 8) {
                attn_args a = { nh, 4, 256, T, 1.0f / 16 };
                const bool long_run = long_base && T >= 1024;
                id<MTLComputePipelineState> base = long_run ? long_base : baseline;
                const attn_variant base_variant = long_run ? prefetch : original;
                NSArray *b = inputs(dev, a, &T, 1, tu ? base_variant.key_block : variant.key_block);
                int n_blk = 0;
                NSArray *cand = tu ? tu_inputs(dev, a, b, &T, 1, &n_blk) : b;
                NSData *want = nil;
                if (base && !tu) { run_blocks(queue, base, a, b, 1, original, 0); want = [NSData dataWithBytes:[b[5] contents] length:(size_t)T * nh * 256 * 2]; }
                double ms = 0, old_ms = 0;
                for (int i = 0; i < 8; i++) {
                    double x, y = 0;
                    if (base && i % 2) y = run_blocks(queue, base, a, b, 1, base_variant, 0);
                    x = run_blocks(queue, ps, a, cand, 1, variant, n_blk);
                    if (want) require(memcmp(want.bytes, [b[5] contents], want.length) == 0, "candidate changed baseline bits");
                    if (base && !(i % 2)) y = run_blocks(queue, base, a, b, 1, base_variant, 0);
                    if (i >= 2) { ms += x; old_ms += y; }
                }
                run_blocks(queue, ps, a, cand, 1, variant, n_blk);
                printf("T=%d nh=%d current=%.3f ms", T, nh, ms / 6);
                if (base) printf(" baseline=%.3f ms speedup=%.3fx%s", old_ms / 6, old_ms / ms, tu ? "" : " bits=identical");
                check_rows(a, b, 1, false, tu != NULL); puts(""); fflush(stdout);
            }
        }
    }
    return 0;
}
