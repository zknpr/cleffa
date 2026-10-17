// Omni decoder attention: candidate kernels against omni_attention on identical Q/K/V.
// Every candidate must reproduce the baseline's output bytes exactly (single and packed
// records); timing is ABBA over GPU command-buffer intervals, reported as the median of
// per-quartet mean(candidate)/mean(baseline) ratios.
//
//   clang -O2 -fobjc-arc bench/omni_attention_bench.m -framework Metal -framework Foundation -o omni-attention-bench
//   ./omni-attention-bench [-q QUARTETS] KERNEL [T ...]
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { int nh, nkv, hd, T; float scale; } attn_args;
enum { NH = 32, NKV = 4, HD = 128, GRP = NH / NKV };

static void die(const char *m) { fprintf(stderr, "omni-attention-bench: %s\n", m); exit(1); }

static uint32_t rng = 12345;
static float uni(void) { rng = rng * 1664525u + 1013904223u; return (float)((rng >> 8) & 0xffffff) / 8388608.0f - 1.0f; }

typedef struct { id<MTLComputePipelineState> ps; NSUInteger smem, threads; } kern;   // 32 lanes per head

static kern make(id<MTLDevice> dev, id<MTLLibrary> lib, const char *name) {
    NSError *err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
    if (!fn) die(name);
    kern k = { [dev newComputePipelineStateWithFunction:fn error:&err], (NSUInteger)GRP * (8 * 32 + 64) * 4, 32 * GRP };
    if (!k.ps) die(err.localizedDescription.UTF8String);
    if (k.ps.maxTotalThreadsPerThreadgroup < k.threads) die("pipeline cannot run the required threadgroup size");
    k.smem += 32 * 132 * 4;   // omni_attention's staged K/V block; candidates may use the same space
    return k;
}

// One dispatch; returns GPU milliseconds and leaves the output in o.
static double run(id<MTLCommandQueue> qu, kern k, id<MTLBuffer> q, id<MTLBuffer> kk, id<MTLBuffer> v,
                  id<MTLBuffer> ss, id<MTLBuffer> o, int T) {
    attn_args a = {NH, NKV, HD, T, 1.0f / sqrtf((float)HD)};
    id<MTLCommandBuffer> cb = [qu commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:k.ps];
    [e setBytes:&a length:sizeof a atIndex:0];
    [e setBuffer:q offset:0 atIndex:1]; [e setBuffer:kk offset:0 atIndex:2];
    [e setBuffer:v offset:0 atIndex:3]; [e setBuffer:ss offset:0 atIndex:4];
    [e setBuffer:o offset:0 atIndex:5];
    [e setThreadgroupMemoryLength:k.smem atIndex:0];
    [e dispatchThreadgroups:MTLSizeMake((T + 7) / 8, NKV, 1) threadsPerThreadgroup:MTLSizeMake(k.threads, 1, 1)];
    [e endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) die("command buffer failed");
    return (cb.GPUEndTime - cb.GPUStartTime) * 1e3;
}

static int cmpd(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }

int main(int argc, char **argv) {
    @autoreleasepool {
        int quartets = 4, ai = 1;
        if (argc > 2 && !strcmp(argv[1], "-q")) { quartets = atoi(argv[2]); ai = 3; }
        if (argc <= ai) die("usage: omni-attention-bench [-q N] KERNEL [T ...]");
        const char *cand_name = argv[ai++];
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        NSError *err = nil;
        NSString *src = [NSString stringWithContentsOfFile:@"metal/clef.metal" encoding:NSUTF8StringEncoding error:&err];
        if (!src) die("read metal/clef.metal");
        MTLCompileOptions *opt = [MTLCompileOptions new];
        opt.mathMode = MTLMathModeSafe;   // as clef_metal.m compiles the engine
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opt error:&err];
        if (!lib) die(err.localizedDescription.UTF8String);
        kern base = make(dev, lib, "omni_attention"), cand = make(dev, lib, cand_name);
        id<MTLCommandQueue> qu = [dev newCommandQueue];
        int defaults[] = {298, 2231, 8069, 16344}, nT = argc - ai;
        for (int ti = 0; ti < (nT ? nT : 4); ti++) {
            const int T = nT ? atoi(argv[ai + ti]) : defaults[ti];
            const size_t qn = ((size_t)NH * T + 32) * HD, kn = ((size_t)NKV * T + 32) * HD;
            id<MTLBuffer> q = [dev newBufferWithLength:qn * 4 options:MTLResourceStorageModeShared];
            id<MTLBuffer> k = [dev newBufferWithLength:kn * 4 options:MTLResourceStorageModeShared];
            id<MTLBuffer> v = [dev newBufferWithLength:kn * 4 options:MTLResourceStorageModeShared];
            id<MTLBuffer> ss = [dev newBufferWithLength:(size_t)T * 4 options:MTLResourceStorageModeShared];
            id<MTLBuffer> o1 = [dev newBufferWithLength:(size_t)T * NH * HD * 4 options:MTLResourceStorageModeShared];
            id<MTLBuffer> o2 = [dev newBufferWithLength:(size_t)T * NH * HD * 4 options:MTLResourceStorageModeShared];
            float *qp = q.contents, *kp = k.contents, *vp = v.contents;
            // Per-head magnitudes vary so some rows are sharp and others flat; the slack is zero.
            for (size_t i = 0; i < qn; i++) qp[i] = i < (size_t)NH * T * HD ? uni() * (1.0f + (float)((i / ((size_t)T * HD)) % 4)) : 0.0f;
            for (size_t i = 0; i < kn; i++) { kp[i] = i < (size_t)NKV * T * HD ? uni() * 2.0f : 0.0f; vp[i] = i < (size_t)NKV * T * HD ? uni() : 0.0f; }
            // Exactness: one record, then three packed records with unaligned starts.
            int *sp = ss.contents;
            for (int pass = 0; pass < 2; pass++) {
                for (int t = 0; t < T; t++) sp[t] = pass == 0 ? 0 : t < T / 3 ? 0 : t < T / 3 + 37 ? T / 3 : T / 3 + 37;
                memset(o1.contents, 0xff, o1.length); memset(o2.contents, 0x7f, o2.length);
                run(qu, base, q, k, v, ss, o1, T);
                run(qu, cand, q, k, v, ss, o2, T);
                if (memcmp(o1.contents, o2.contents, o1.length)) {
                    const uint32_t *x = o1.contents, *y = o2.contents;
                    size_t bad = 0, first = 0;
                    for (size_t i = 0; i < o1.length / 4; i++) if (x[i] != y[i]) { if (!bad++) first = i; }
                    printf("T=%d %s: %s differs from omni_attention in %zu of %zu words (first %zu)\n",
                           T, pass ? "packed" : "single", cand_name, bad, o1.length / 4, first);
                    return 1;
                }
            }
            for (int t = 0; t < T; t++) sp[t] = 0;
            run(qu, base, q, k, v, ss, o1, T); run(qu, cand, q, k, v, ss, o2, T);   // warm
            double *ratio = calloc(quartets, sizeof *ratio), *bt = calloc(quartets, sizeof *bt), *ct = calloc(quartets, sizeof *ct);
            for (int i = 0; i < quartets; i++) {
                const double b1 = run(qu, base, q, k, v, ss, o1, T), c1 = run(qu, cand, q, k, v, ss, o2, T);
                const double c2 = run(qu, cand, q, k, v, ss, o2, T), b2 = run(qu, base, q, k, v, ss, o1, T);
                bt[i] = (b1 + b2) / 2; ct[i] = (c1 + c2) / 2; ratio[i] = ct[i] / bt[i];
            }
            qsort(ratio, quartets, sizeof *ratio, cmpd); qsort(bt, quartets, sizeof *bt, cmpd); qsort(ct, quartets, sizeof *ct, cmpd);
            printf("T=%5d exact (single, packed); omni_attention %.3f ms, %s %.3f ms, median ratio %.4f [%.4f, %.4f]\n",
                   T, bt[quartets / 2], cand_name, ct[quartets / 2], ratio[quartets / 2], ratio[0], ratio[quartets - 1]);
            free(ratio); free(bt); free(ct);
        }
    }
    return 0;
}
