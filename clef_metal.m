// Metal backend: zero-copy weights, one command buffer per packed batch.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "clef_engine.h"
#include "clef_metal_src.inc"   // clef_metal_src / clef_metal_src_len, generated from metal/clef.metal

struct clef_gpu {
    id<MTLDevice> dev;
    id<MTLCommandQueue> queue;
    id<MTLBuffer> weights;            // the whole mmap'd GGUF, no copy
    NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *ps;
    id<MTLBuffer> inv_freq;
    int cap;                          // token capacity of the activation buffers
    id<MTLBuffer> ids, pos, seq_start, seq_bounds;
    id<MTLBuffer> x, xn, P, Q, K, V, G, A, Xc, beta, gate, O, hfin, nh32, nh16, mem, mem16, mn16, mn32;
    id<MTLBuffer> ovf;                // FP16 overflow flags, int per token, set at each record's first token
    id<MTLBuffer> kv[16];
    int n_kv;
    int act_f16;                      // FP16 GEMM inputs per producer class (ACT_*), else BF16
    float f16_lim;                    // magnitude past which an FP16 operand counts as overflow
    bool head_f32;                    // head inputs unrounded, head GEMMs on f32 activations
    int scan_lpc;                     // lanes per value column in gdn_scan (CLEF_SCAN_LPC)
    bool attn_ref;                    // CLEF_ATTN_REF=1: the simple reference attention kernel
    bool debug_poison;                // CLEF_DEBUG_POISON=1 (tests): fill every activation buffer with NaN before each forward
    bool profile;                     // CLEF_PROFILE=1: GPU time per kernel category (serializes!)
    double prof_ms[8];
    bool cb_failed;                   // a command buffer or encoder could not be created this pass
    int nil_cb_at, cb_count;          // test hook CLEF_DEBUG_NIL_CMDBUF: creation #nil_cb_at fails
};

enum { P_GEMM, P_ATTN, P_SCAN, P_OTHER, P_N };
// GEMM input classes, by the kernel that produces them
enum { ACT_NORM = 1, ACT_ATTN = 2, ACT_GDN = 4, ACT_MLP = 8 };
enum { X_BF16, X_F16, X_F32 };
static const char *prof_name[P_N] = { "gemm", "attention", "gdn_scan", "other" };

// Command buffers and encoders are nullable (device loss, resource exhaustion). Messages to nil
// are no-ops and a nil buffer has no .error, so a missing one let the pass "succeed" without
// running and the head read the previous pass's activations, another request's (review #4,
// tests/test_gpu_fail.sh). Every creation goes through here; a failure sticks for the pass and
// clef_gpu_forward reports it.
static bool new_cb(clef_gpu *g, id<MTLCommandBuffer> __strong *cb, id<MTLComputeCommandEncoder> __strong *enc) {
    *cb = ++g->cb_count == g->nil_cb_at ? nil : [g->queue commandBuffer];
    *enc = *cb ? [*cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial] : nil;
    if (!*cb || !*enc) g->cb_failed = true;
    return !g->cb_failed;
}

// Profiling: end the current command buffer, attribute its GPU time to `cat`, start a new one.
static bool prof(clef_gpu *g, id<MTLCommandBuffer> __strong *cb, id<MTLComputeCommandEncoder> __strong *enc,
                 int cat, char *err, size_t errlen) {
    if (!g->profile) return true;
    [*enc endEncoding];
    [*cb commit];
    [*cb waitUntilCompleted];
    // Check before replacing the buffer: later stages must never consume a failed
    // stage's incomplete activations, even if the final command buffer succeeds.
    NSError *error = (*cb).error;
    if (error) { snprintf(err, errlen, "metal: %s", error.localizedDescription.UTF8String); return false; }
    g->prof_ms[cat] += ((*cb).GPUEndTime - (*cb).GPUStartTime) * 1e3;
    if (!new_cb(g, cb, enc)) {
        snprintf(err, errlen, "metal: cannot create a command buffer (mid-pass)");
        return false;
    }
    return true;
}

static bool gerr(char *err, size_t errlen, const char *msg) {
    snprintf(err, errlen, "metal: %s", msg);
    return false;
}

static id<MTLComputePipelineState> pipeline(clef_gpu *g, id<MTLLibrary> lib, const char *name, char *err, size_t errlen) {
    NSString *n = [NSString stringWithUTF8String:name];
    id<MTLFunction> fn = [lib newFunctionWithName:n];
    if (!fn) { snprintf(err, errlen, "metal: kernel %s missing", name); return nil; }
    NSError *e = nil;
    id<MTLComputePipelineState> p = [g->dev newComputePipelineStateWithFunction:fn error:&e];
    if (!p) { snprintf(err, errlen, "metal: pipeline %s: %s", name, e.localizedDescription.UTF8String); return nil; }
    g->ps[n] = p;
    return p;
}

clef_gpu *clef_gpu_open(const clef_engine *e, char *err, size_t errlen) {
    @autoreleasepool {
        clef_gpu *g = (clef_gpu *)calloc(1, sizeof(*g));
        if (!g) { gerr(err, errlen, "out of memory"); return NULL; }
        g->dev = MTLCreateSystemDefaultDevice();
        if (!g->dev) { gerr(err, errlen, "no Metal device"); free(g); return NULL; }
        g->queue = [g->dev newCommandQueue];
        if (!g->queue) { gerr(err, errlen, "cannot create command queue"); free(g); return NULL; }

        const size_t page = (size_t)getpagesize();
        const size_t len = (e->gguf.size + page - 1) / page * page;  // within the mapping's last page
        if (len > g->dev.maxBufferLength) {
            snprintf(err, errlen, "metal: model (%zu bytes) exceeds maxBufferLength (%lu)", len, (unsigned long)g->dev.maxBufferLength);
            free(g);
            return NULL;
        }
        g->weights = [g->dev newBufferWithBytesNoCopy:(void *)e->gguf.base length:len
                                              options:MTLResourceStorageModeShared deallocator:nil];
        if (!g->weights) { gerr(err, errlen, "cannot wrap the model mapping in a Metal buffer"); free(g); return NULL; }

        NSString *src = [[NSString alloc] initWithBytes:clef_metal_src length:clef_metal_src_len encoding:NSUTF8StringEncoding];
        MTLCompileOptions *opt = [MTLCompileOptions new];
        // Safe math: the parity target is the reference, not peak ALU throughput, and the
        // GEMMs (where the time goes) run on the tensor units either way.
        opt.mathMode = MTLMathModeSafe;
        NSError *ce = nil;
        id<MTLLibrary> lib = [g->dev newLibraryWithSource:src options:opt error:&ce];
        if (!lib) { snprintf(err, errlen, "metal: shader compile failed: %s", ce.localizedDescription.UTF8String); free(g); return NULL; }
        g->ps = [NSMutableDictionary new];
        const char *names[] = { "gemm_bf16_32x128", "gemm_bf16_acc_32x128", "gemm_f16_32x128", "gemm_f16_acc_32x128",
                                "gemm_f32_32x128", "gemm_f32_acc_32x128", "embed", "rmsnorm_act", "rmsnorm_f32", "layernorm",
                                "f32_to_bf16", "fill_zero", "attn_prep", "attention", "attention_fa", "ssm_conv", "gdn_prep", "gdn_scan_2", "gdn_scan_4", "gdn_scan_8", "gdn_out", "swiglu" };
        for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); i++) {
            if (!pipeline(g, lib, names[i], err, errlen)) { free(g); return NULL; }
        }

        const clef_config *c = &e->cfg;
        float inv[64];
        for (int j = 0; j < c->n_rot / 2; j++) {
            // HF: 1 / base ** (arange(0, dim, 2).float() / dim), all f32
            inv[j] = 1.0f / powf(c->rope_theta, (float)(2 * j) / (float)c->n_rot);
        }
        g->inv_freq = [g->dev newBufferWithBytes:inv length:sizeof(inv) options:MTLResourceStorageModeShared];
        if (!g->inv_freq) { gerr(err, errlen, "cannot allocate rope table"); free(g); return NULL; }
        g->n_kv = c->routing_layers + c->head_layers;
        g->profile = getenv("CLEF_PROFILE") != NULL;
        g->attn_ref = getenv("CLEF_ATTN_REF") != NULL;
        g->debug_poison = getenv("CLEF_DEBUG_POISON") != NULL;
        g->scan_lpc = getenv("CLEF_SCAN_LPC") ? atoi(getenv("CLEF_SCAN_LPC")) : 8;
        if (g->scan_lpc != 2 && g->scan_lpc != 4 && g->scan_lpc != 8) g->scan_lpc = 8;
        // Precision (README "Accuracy"): the target is the FP32 reference. GEMM activations are
        // FP16 for every producer class by default (7x closer to FP32 than BF16 on the 27B, same
        // GEMM rate); CLEF_ACT_F16=<mask of ACT_*> narrows that, 0 = BF16 everywhere as in the
        // HF BF16 path. The head takes the f32 hidden state and runs its GEMMs on f32 activations
        // (tiny GEMMs); CLEF_HEAD_BF16=1 restores the BF16 rounding the HF BF16 path applies there.
        g->act_f16 = getenv("CLEF_ACT_F16") ? atoi(getenv("CLEF_ACT_F16")) & 15 : ACT_NORM | ACT_ATTN | ACT_GDN | ACT_MLP;
        g->head_f32 = getenv("CLEF_HEAD_BF16") == NULL;
        g->nil_cb_at = getenv("CLEF_DEBUG_NIL_CMDBUF") ? atoi(getenv("CLEF_DEBUG_NIL_CMDBUF")) : 0;
        // test hook (tests/test_f16_overflow.sh): a lower limit forces the BF16 rerun on real inputs
        g->f16_lim = getenv("CLEF_DEBUG_F16_LIMIT") ? strtof(getenv("CLEF_DEBUG_F16_LIMIT"), NULL) : 65504.0f;
        if (!(g->f16_lim > 0.0f) || g->f16_lim > 65504.0f) g->f16_lim = 65504.0f;
        return g;
    }
}

void clef_gpu_close(clef_gpu *g) {
    if (!g) return;
    @autoreleasepool {
        g->ps = nil; g->weights = nil; g->queue = nil; g->dev = nil;
        g->ids = g->pos = g->seq_start = g->seq_bounds = nil;
        g->x = g->xn = g->P = g->Q = g->K = g->V = g->G = g->A = g->Xc = g->beta = g->gate = g->O = nil;
        g->hfin = g->nh32 = g->nh16 = g->mem = g->mem16 = g->mn16 = g->mn32 = g->ovf = g->inv_freq = nil;
        for (int i = 0; i < 16; i++) g->kv[i] = nil;
    }
    free(g);
}

static id<MTLBuffer> buf(clef_gpu *g, size_t bytes) {
    return [g->dev newBufferWithLength:(bytes ? bytes : 16) options:MTLResourceStorageModeShared];
}

static bool ensure_capacity(clef_gpu *g, const clef_config *c, int T, int n_seq, char *err, size_t errlen) {
    if (T <= g->cap && n_seq + 1 <= g->cap) return true;
    int cap = g->cap ? g->cap : 512;
    while (cap < T || cap < n_seq + 1) cap *= 2;
    const size_t Tc = (size_t)cap;
    const int C = 2 * c->Hk * c->dk + c->Hv * c->dv;
    const size_t n_ssm = (size_t)C + (size_t)c->Hv * c->dv + 2 * (size_t)c->Hv;
    const size_t n_attn = (size_t)c->nh * 2 * c->hd + 2 * (size_t)c->nkv * c->hd;
    size_t maxN = 2 * (size_t)c->ffn;
    if (n_ssm > maxN) maxN = n_ssm;
    if (n_attn > maxN) maxN = n_attn;
    size_t maxA = (size_t)c->ffn;
    if ((size_t)c->nh * c->hd > maxA) maxA = (size_t)c->nh * c->hd;
    if ((size_t)c->Hv * c->dv > maxA) maxA = (size_t)c->Hv * c->dv;
    // Allocate the whole new set first and commit it only if every allocation succeeded:
    // replacing live buffers in place left nil buffers behind a stale g->cap on failure, and
    // later requests that fit the old capacity then ran against nil bindings and returned
    // silently wrong logits (tests/test_grow_fail.sh).
    id<MTLBuffer> ids, pos, seq_start, seq_bounds, x, xn, P, Q, K, V, G, A, Xc, beta, gate, O, hfin, nh32, nh16, mem, mem16, mn16, mn32, ovf;
    id<MTLBuffer> kv[16] = { nil };
    @autoreleasepool {
        ids = buf(g, Tc * 4); pos = buf(g, Tc * 4); seq_start = buf(g, Tc * 4); seq_bounds = buf(g, (Tc + 1) * 4); ovf = buf(g, Tc * 4);
        x = buf(g, Tc * c->H * 4); xn = buf(g, Tc * c->H * 2);
        P = buf(g, Tc * maxN * 4);
        // test hook (tests/test_grow_fail.sh): simulate the largest allocation failing when
        // capacity would grow past CLEF_DEBUG_GROW_FAIL_ABOVE tokens
        const char *gf = getenv("CLEF_DEBUG_GROW_FAIL_ABOVE");
        if (gf && cap > atoi(gf)) P = nil;
        // +64 rows of slack: attention tiles load up to 32 rows past T in the last head; those
        // rows are zeroed on every forward (see fill_zero in the attention block).
        const size_t slack = 64 * (size_t)c->hd * 4;
        Q = buf(g, Tc * c->nh * c->hd * 4 + slack); K = buf(g, Tc * c->nkv * c->hd * 4 + slack);
        V = buf(g, Tc * c->nkv * c->hd * 4 + slack);
        G = buf(g, Tc * c->nh * c->hd * 4);
        A = buf(g, Tc * maxA * 2);
        Xc = buf(g, Tc * C * 4); beta = buf(g, Tc * c->Hv * 4); gate = buf(g, Tc * c->Hv * 4);
        O = buf(g, Tc * c->Hv * c->dv * 4);
        // the head uses either the 16-bit operands (CLEF_HEAD_BF16) or mn32; the other set is a stub
        const size_t h16 = g->head_f32 ? 0 : 1, h32 = g->head_f32 ? 1 : 0;
        hfin = buf(g, Tc * c->H * 4); nh32 = buf(g, Tc * c->H * 4); nh16 = buf(g, h16 * Tc * c->H * 2);
        mem = buf(g, Tc * c->W * 4); mem16 = buf(g, h16 * Tc * c->W * 2); mn16 = buf(g, h16 * Tc * c->W * 2);
        mn32 = buf(g, h32 * Tc * c->W * 4);
        for (int i = 0; i < g->n_kv; i++) kv[i] = buf(g, Tc * 2 * c->W * 4);
    }
    id<MTLBuffer> all[] = { ids, pos, seq_start, seq_bounds, x, xn, P, Q, K, V, G, A, Xc, beta, gate, O, hfin, nh32, nh16, mem, mem16, mn16, mn32, ovf };
    for (size_t i = 0; i < sizeof(all) / sizeof(all[0]); i++) {
        if (!all[i]) return gerr(err, errlen, "cannot allocate activation buffers (previous buffers kept)");
    }
    for (int i = 0; i < g->n_kv; i++) if (!kv[i]) return gerr(err, errlen, "cannot allocate activation buffers (previous buffers kept)");
    memset(Q.contents, 0, Q.length); memset(K.contents, 0, K.length); memset(V.contents, 0, V.length);
    g->ids = ids; g->pos = pos; g->seq_start = seq_start; g->seq_bounds = seq_bounds; g->x = x; g->xn = xn; g->P = P;
    g->Q = Q; g->K = K; g->V = V; g->G = G; g->A = A; g->Xc = Xc; g->beta = beta; g->gate = gate; g->O = O;
    g->hfin = hfin; g->nh32 = nh32; g->nh16 = nh16; g->mem = mem; g->mem16 = mem16; g->mn16 = mn16; g->mn32 = mn32; g->ovf = ovf;
    for (int i = 0; i < g->n_kv; i++) g->kv[i] = kv[i];
    g->cap = cap;
    return true;
}

// ---- encoding helpers ----

typedef struct { int T, N, K; } gemm_args;

static NSUInteger woff(const gguf_tensor *t) { return (NSUInteger)t->offset; }

static void gemm(clef_gpu *g, id<MTLComputeCommandEncoder> enc, int xt, bool acc, id<MTLBuffer> X, NSUInteger xoff,
                 NSUInteger w_off, int N, int K, id<MTLBuffer> Y, NSUInteger yoff, int T) {
    gemm_args a = { T, N, K };
    // One tile shape for every T, so a record's GEMM never depends on the packed batch size
    // (whether MPP's per-element reduction order varies with tile shape is unverified; a
    // single shape removes the question). 32x128 (bench/tile_bench.m) is best or within 3%
    // for T in 146..594, up to 28% faster than 128x64 on ragged T, within 15% at 1k-2k and
    // faster at 8k.
    const int tm = 32, tn = 128;
    static NSString *const names[3][2] = { { @"gemm_bf16_32x128", @"gemm_bf16_acc_32x128" },
                                           { @"gemm_f16_32x128", @"gemm_f16_acc_32x128" },
                                           { @"gemm_f32_32x128", @"gemm_f32_acc_32x128" } };
    NSString *name = names[xt][acc];
    [enc setComputePipelineState:g->ps[name]];
    [enc setBytes:&a length:sizeof(a) atIndex:0];
    [enc setBuffer:X offset:xoff atIndex:1];
    [enc setBuffer:g->weights offset:w_off atIndex:2];
    [enc setBuffer:Y offset:yoff atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake((N + tn - 1) / tn, (T + tm - 1) / tm, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
}

static void rows_kernel(clef_gpu *g, id<MTLComputeCommandEncoder> enc, NSString *name, int rows, int width) {
    id<MTLComputePipelineState> p = g->ps[name];
    NSUInteger tpg = width >= 1024 ? 1024 : (NSUInteger)((width + 31) / 32 * 32);
    if (tpg > p.maxTotalThreadsPerThreadgroup) tpg = p.maxTotalThreadsPerThreadgroup;
    [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
}

typedef struct { int H; float eps; } norm_args;
typedef struct { int f16; float lim; } act_args;
typedef struct { int H; float eps; int round_input, write32, write16; } ln_args;
typedef struct { int nh, nkv, hd, n_rot, row; float eps; } attn_prep_args;
typedef struct { int nh, nkv, hd, T; float scale; } attn_args;
typedef struct { int C, row, ksize; } conv_args;
typedef struct { int Hk, Hv, dk, dv, row, C; } gdn_prep_args;
typedef struct { int Hk, Hv, dk, dv, C; } gdn_args;
typedef struct { int Hv, dv, row, zoff; float eps; } gdn_out_args;
typedef struct { int I; } swiglu_args;

static const gguf_tensor *head_t(const clef_engine *e, const char *name) { return gguf_find_tensor(&e->gguf, name); }

bool clef_gpu_forward(clef_gpu *g, const clef_engine *e, const int32_t *ids, const int32_t *pos,
                      const int32_t *seq_start, const int32_t *seq_bounds, int n_seq, int T, bool bf16_only,
                      bool *overflow, clef_head_inputs *in, float *dump_layers, int dump_rows, char *err, size_t errlen) {
    const int R = dump_rows > 0 && dump_rows < T ? dump_rows : T;   /* dump the last R token rows */
    const clef_config *c = &e->cfg;
    // FP16 operands are only safe with the overflow flags read back (and acted on) by the caller
    if (!bf16_only && g->act_f16 && !overflow) return gerr(err, errlen, "FP16 activations need the overflow flags (pass overflow or bf16_only)");
    if (!ensure_capacity(g, c, T, n_seq, err, errlen)) return false;
    @autoreleasepool {
        memset(g->prof_ms, 0, sizeof(g->prof_ms));   // a prior failed pass may have partial timings
        if (g->debug_poison) {
            // Test hook: rows a kernel must not depend on hold NaN. Every row it does need is
            // rewritten by attn_prep, so correct code gives identical results (tests/test_poison.sh).
            const float nan = NAN;   // also a NaN pattern in BF16 halves (0x7fc0)
            id<MTLBuffer> bufs[] = { g->x, g->xn, g->P, g->Q, g->K, g->V, g->G, g->A, g->Xc, g->beta, g->gate,
                                     g->O, g->hfin, g->nh32, g->nh16, g->mem, g->mem16, g->mn16, g->mn32 };
            for (size_t i = 0; i < sizeof(bufs) / sizeof(bufs[0]); i++) memset_pattern4(bufs[i].contents, &nan, bufs[i].length);
            for (int i = 0; i < g->n_kv; i++) memset_pattern4(g->kv[i].contents, &nan, g->kv[i].length);
        }
        memcpy(g->ids.contents, ids, (size_t)T * 4);
        memcpy(g->pos.contents, pos, (size_t)T * 4);
        memcpy(g->seq_start.contents, seq_start, (size_t)T * 4);
        memcpy(g->seq_bounds.contents, seq_bounds, (size_t)(n_seq + 1) * 4);
        memset(g->ovf.contents, 0, (size_t)T * 4);

        const int H = c->H, W = c->W;
        const int C = 2 * c->Hk * c->dk + c->Hv * c->dv;
        const int f16 = bf16_only ? 0 : g->act_f16;
        const int f_norm = !!(f16 & ACT_NORM), f_attn = !!(f16 & ACT_ATTN);
        const int f_gdn = !!(f16 & ACT_GDN), f_mlp = !!(f16 & ACT_MLP);
        const act_args ac_norm = { f_norm, g->f16_lim }, ac_attn = { f_attn, g->f16_lim };
        const act_args ac_gdn = { f_gdn, g->f16_lim }, ac_mlp = { f_mlp, g->f16_lim };
        const int x_norm = f_norm ? X_F16 : X_BF16, x_attn = f_attn ? X_F16 : X_BF16;
        const int x_gdn = f_gdn ? X_F16 : X_BF16, x_mlp = f_mlp ? X_F16 : X_BF16;
        const int n_ssm = C + c->Hv * c->dv + 2 * c->Hv;
        const int n_attn = c->nh * 2 * c->hd + 2 * c->nkv * c->hd;
        id<MTLCommandBuffer> cb;
        id<MTLComputeCommandEncoder> enc;
        g->cb_failed = false;
        if (!new_cb(g, &cb, &enc)) return gerr(err, errlen, "cannot create a command buffer");

        // embedding
        {
            [enc setComputePipelineState:g->ps[@"embed"]];
            int h = H;
            [enc setBytes:&h length:sizeof(h) atIndex:0];
            [enc setBuffer:g->ids offset:0 atIndex:1];
            [enc setBuffer:g->weights offset:woff(e->w.token_embd) atIndex:2];
            [enc setBuffer:g->x offset:0 atIndex:3];
            [enc dispatchThreads:MTLSizeMake(H, T, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
        if (dump_layers) {
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            if (cb.error) { snprintf(err, errlen, "metal: %s", cb.error.localizedDescription.UTF8String); return false; }
            memcpy(dump_layers, (float *)g->x.contents + (size_t)(T - R) * H, (size_t)R * H * 4);
            if (!new_cb(g, &cb, &enc)) return gerr(err, errlen, "cannot create a command buffer");
        }

        for (int l = 0; l < c->n_layer; l++) {
            const clef_layer_w *lw = &e->w.layer[l];
            norm_args na = { H, c->eps };
            [enc setComputePipelineState:g->ps[@"rmsnorm_act"]];
            [enc setBytes:&na length:sizeof(na) atIndex:0];
            [enc setBuffer:g->x offset:0 atIndex:1];
            [enc setBuffer:g->weights offset:woff(lw->attn_norm) atIndex:2];
            [enc setBuffer:g->xn offset:0 atIndex:3];
            [enc setBytes:&ac_norm length:sizeof(ac_norm) atIndex:4];
            [enc setBuffer:g->seq_start offset:0 atIndex:5];
            [enc setBuffer:g->ovf offset:0 atIndex:6];
            rows_kernel(g, enc, @"rmsnorm_act", T, H);

            if (c->layer_full[l]) {
                if (!prof(g, &cb, &enc, P_OTHER, err, errlen)) return false;
                gemm(g, enc, x_norm, false, g->xn, 0, woff(lw->attn_qkv), n_attn, H, g->P, 0, T);
                if (!prof(g, &cb, &enc, P_GEMM, err, errlen)) return false;
                attn_prep_args pa = { c->nh, c->nkv, c->hd, c->n_rot, n_attn, c->eps };
                [enc setComputePipelineState:g->ps[@"attn_prep"]];
                [enc setBytes:&pa length:sizeof(pa) atIndex:0];
                [enc setBuffer:g->P offset:0 atIndex:1];
                [enc setBuffer:g->weights offset:woff(lw->attn_q_norm) atIndex:2];
                [enc setBuffer:g->weights offset:woff(lw->attn_k_norm) atIndex:3];
                [enc setBuffer:g->pos offset:0 atIndex:4];
                [enc setBuffer:g->inv_freq offset:0 atIndex:5];
                [enc setBuffer:g->Q offset:0 atIndex:6];
                [enc setBuffer:g->K offset:0 atIndex:7];
                [enc setBuffer:g->V offset:0 atIndex:8];
                [enc setBuffer:g->G offset:0 atIndex:9];
                [enc setBytes:&T length:sizeof(T) atIndex:10];
                [enc dispatchThreadgroups:MTLSizeMake(T, c->nh + c->nkv, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

                // zero the rows attention tiles read past T in the last head (Q: 8, K/V: 32)
                {
                    struct { id<MTLBuffer> b; NSUInteger off; long n; } z[3] = {
                        { g->Q, (NSUInteger)c->nh * T * c->hd * 4, 8L * c->hd },
                        { g->K, (NSUInteger)c->nkv * T * c->hd * 4, 32L * c->hd },
                        { g->V, (NSUInteger)c->nkv * T * c->hd * 4, 32L * c->hd },
                    };
                    [enc setComputePipelineState:g->ps[@"fill_zero"]];
                    for (int i = 0; i < 3; i++) {
                        [enc setBuffer:z[i].b offset:z[i].off atIndex:0];
                        [enc setBytes:&z[i].n length:sizeof(long) atIndex:1];
                        [enc dispatchThreads:MTLSizeMake((NSUInteger)z[i].n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    }
                }
                attn_args aa = { c->nh, c->nkv, c->hd, T, 1.0f / sqrtf((float)c->hd) };
                [enc setComputePipelineState:g->ps[g->attn_ref ? @"attention" : @"attention_fa"]];
                [enc setBytes:&aa length:sizeof(aa) atIndex:0];
                [enc setBuffer:g->Q offset:0 atIndex:1];
                [enc setBuffer:g->K offset:0 atIndex:2];
                [enc setBuffer:g->V offset:0 atIndex:3];
                [enc setBuffer:g->G offset:0 atIndex:4];
                [enc setBuffer:g->seq_start offset:0 atIndex:5];
                [enc setBuffer:g->A offset:0 atIndex:6];
                [enc setBytes:&ac_attn length:sizeof(ac_attn) atIndex:7];
                [enc setBuffer:g->ovf offset:0 atIndex:8];
                if (g->attn_ref) {
                    [enc dispatchThreadgroups:MTLSizeMake(T, c->nh, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                } else {
                    const int grp = c->nh / c->nkv;
                    [enc setThreadgroupMemoryLength:(NSUInteger)grp * (8 * 32 + 64) * 4 atIndex:0];
                    [enc dispatchThreadgroups:MTLSizeMake((T + 7) / 8, c->nkv, 1) threadsPerThreadgroup:MTLSizeMake(32 * grp, 1, 1)];
                }
                if (!prof(g, &cb, &enc, P_ATTN, err, errlen)) return false;

                gemm(g, enc, x_attn, true, g->A, 0, woff(lw->attn_output), H, c->nh * c->hd, g->x, 0, T);
                if (!prof(g, &cb, &enc, P_GEMM, err, errlen)) return false;
            } else {
                if (!prof(g, &cb, &enc, P_OTHER, err, errlen)) return false;
                gemm(g, enc, x_norm, false, g->xn, 0, woff(lw->ssm_in), n_ssm, H, g->P, 0, T);
                if (!prof(g, &cb, &enc, P_GEMM, err, errlen)) return false;
                conv_args ca = { C, n_ssm, c->ssm_kernel };
                [enc setComputePipelineState:g->ps[@"ssm_conv"]];
                [enc setBytes:&ca length:sizeof(ca) atIndex:0];
                [enc setBuffer:g->P offset:0 atIndex:1];
                [enc setBuffer:g->weights offset:woff(lw->ssm_conv1d) atIndex:2];
                [enc setBuffer:g->seq_start offset:0 atIndex:3];
                [enc setBuffer:g->Xc offset:0 atIndex:4];
                [enc dispatchThreads:MTLSizeMake(C, T, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                gdn_prep_args gp = { c->Hk, c->Hv, c->dk, c->dv, n_ssm, C };
                [enc setComputePipelineState:g->ps[@"gdn_prep"]];
                [enc setBytes:&gp length:sizeof(gp) atIndex:0];
                [enc setBuffer:g->Xc offset:0 atIndex:1];
                [enc setBuffer:g->P offset:0 atIndex:2];
                [enc setBuffer:g->weights offset:woff(lw->ssm_a) atIndex:3];
                [enc setBuffer:g->weights offset:woff(lw->ssm_dt) atIndex:4];
                [enc setBuffer:g->beta offset:0 atIndex:5];
                [enc setBuffer:g->gate offset:0 atIndex:6];
                [enc dispatchThreadgroups:MTLSizeMake(T, 2 * c->Hk + c->Hv, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

                if (!prof(g, &cb, &enc, P_OTHER, err, errlen)) return false;   // boundary must precede the encoder setup below
                gdn_args ga = { c->Hk, c->Hv, c->dk, c->dv, C };
                [enc setComputePipelineState:g->ps[[NSString stringWithFormat:@"gdn_scan_%d", g->scan_lpc]]];
                [enc setBytes:&ga length:sizeof(ga) atIndex:0];
                [enc setBuffer:g->Xc offset:0 atIndex:1];
                [enc setBuffer:g->beta offset:0 atIndex:2];
                [enc setBuffer:g->gate offset:0 atIndex:3];
                [enc setBuffer:g->seq_bounds offset:0 atIndex:4];
                [enc setBuffer:g->O offset:0 atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake(c->Hv, c->dv / (32 / g->scan_lpc), n_seq) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                if (!prof(g, &cb, &enc, P_SCAN, err, errlen)) return false;

                gdn_out_args go = { c->Hv, c->dv, n_ssm, C, c->eps };
                [enc setComputePipelineState:g->ps[@"gdn_out"]];
                [enc setBytes:&go length:sizeof(go) atIndex:0];
                [enc setBuffer:g->O offset:0 atIndex:1];
                [enc setBuffer:g->P offset:0 atIndex:2];
                [enc setBuffer:g->weights offset:woff(lw->ssm_norm) atIndex:3];
                [enc setBuffer:g->A offset:0 atIndex:4];
                [enc setBytes:&ac_gdn length:sizeof(ac_gdn) atIndex:5];
                [enc setBuffer:g->seq_start offset:0 atIndex:6];
                [enc setBuffer:g->ovf offset:0 atIndex:7];
                [enc dispatchThreadgroups:MTLSizeMake(T, c->Hv, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

                if (!prof(g, &cb, &enc, P_OTHER, err, errlen)) return false;
                gemm(g, enc, x_gdn, true, g->A, 0, woff(lw->ssm_out), H, c->Hv * c->dv, g->x, 0, T);
                if (!prof(g, &cb, &enc, P_GEMM, err, errlen)) return false;
            }

            [enc setComputePipelineState:g->ps[@"rmsnorm_act"]];
            [enc setBytes:&na length:sizeof(na) atIndex:0];
            [enc setBuffer:g->x offset:0 atIndex:1];
            [enc setBuffer:g->weights offset:woff(lw->ffn_norm) atIndex:2];
            [enc setBuffer:g->xn offset:0 atIndex:3];
            [enc setBytes:&ac_norm length:sizeof(ac_norm) atIndex:4];
            [enc setBuffer:g->seq_start offset:0 atIndex:5];
            [enc setBuffer:g->ovf offset:0 atIndex:6];
            rows_kernel(g, enc, @"rmsnorm_act", T, H);
            if (!prof(g, &cb, &enc, P_OTHER, err, errlen)) return false;
            gemm(g, enc, x_norm, false, g->xn, 0, woff(lw->ffn_gate_up), 2 * c->ffn, H, g->P, 0, T);
            if (!prof(g, &cb, &enc, P_GEMM, err, errlen)) return false;
            swiglu_args sa = { c->ffn };
            [enc setComputePipelineState:g->ps[@"swiglu"]];
            [enc setBytes:&sa length:sizeof(sa) atIndex:0];
            [enc setBuffer:g->P offset:0 atIndex:1];
            [enc setBuffer:g->A offset:0 atIndex:2];
            [enc setBytes:&ac_mlp length:sizeof(ac_mlp) atIndex:3];
            [enc setBuffer:g->seq_start offset:0 atIndex:4];
            [enc setBuffer:g->ovf offset:0 atIndex:5];
            [enc dispatchThreads:MTLSizeMake(c->ffn, T, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            if (!prof(g, &cb, &enc, P_OTHER, err, errlen)) return false;
            gemm(g, enc, x_mlp, true, g->A, 0, woff(lw->ffn_down), H, c->ffn, g->x, 0, T);
            if (!prof(g, &cb, &enc, P_GEMM, err, errlen)) return false;

            if (dump_layers) {
                // Debug path: finish this command buffer and copy the residual out.
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                if (cb.error) { snprintf(err, errlen, "metal: %s", cb.error.localizedDescription.UTF8String); return false; }
                memcpy(dump_layers + (size_t)(l + 1) * R * H, (float *)g->x.contents + (size_t)(T - R) * H, (size_t)R * H * 4);
                if (!new_cb(g, &cb, &enc)) return gerr(err, errlen, "cannot create a command buffer");
            }
        }

        // final norm, then the GPU half of the head
        norm_args nf = { H, c->eps };
        [enc setComputePipelineState:g->ps[@"rmsnorm_f32"]];
        [enc setBytes:&nf length:sizeof(nf) atIndex:0];
        [enc setBuffer:g->x offset:0 atIndex:1];
        [enc setBuffer:g->weights offset:woff(e->w.output_norm) atIndex:2];
        [enc setBuffer:g->hfin offset:0 atIndex:3];
        rows_kernel(g, enc, @"rmsnorm_f32", T, H);

        // hidden_norm over the f32 last hidden state (BF16-rounded with CLEF_HEAD_BF16, as HF's BF16 path)
        ln_args la = { H, 1e-5f, !g->head_f32, 1, !g->head_f32 };
        [enc setComputePipelineState:g->ps[@"layernorm"]];
        [enc setBytes:&la length:sizeof(la) atIndex:0];
        [enc setBuffer:g->hfin offset:0 atIndex:1];
        [enc setBuffer:g->weights offset:woff(head_t(e, "head.hidden_norm.weight")) atIndex:2];
        [enc setBuffer:g->weights offset:woff(head_t(e, "head.hidden_norm.bias")) atIndex:3];
        [enc setBuffer:g->nh32 offset:0 atIndex:4];
        [enc setBuffer:g->nh16 offset:0 atIndex:5];
        rows_kernel(g, enc, @"layernorm", T, H);
        if (g->head_f32) gemm(g, enc, X_F32, false, g->nh32, 0, woff(head_t(e, "head.memory_projection.weight")), W, H, g->mem, 0, T);
        else gemm(g, enc, X_BF16, false, g->nh16, 0, woff(head_t(e, "head.memory_projection.weight")), W, H, g->mem, 0, T);
        if (!g->head_f32) {
            long n = (long)T * W;
            [enc setComputePipelineState:g->ps[@"f32_to_bf16"]];
            [enc setBuffer:g->mem offset:0 atIndex:0];
            [enc setBuffer:g->mem16 offset:0 atIndex:1];
            [enc setBytes:&n length:sizeof(n) atIndex:2];
            [enc dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
        char name[128];
        for (int i = 0; i < c->routing_layers; i++) {
            // evidence layer: K|V = memory_norm(memory) . in_proj[E:3E]^T (memory is BF16 in the reference)
            ln_args lm = { W, 1e-5f, !g->head_f32, g->head_f32, !g->head_f32 };
            snprintf(name, sizeof(name), "head.evidence_layers.%d.memory_norm.weight", i);
            const gguf_tensor *mw = head_t(e, name);
            snprintf(name, sizeof(name), "head.evidence_layers.%d.memory_norm.bias", i);
            const gguf_tensor *mb = head_t(e, name);
            snprintf(name, sizeof(name), "head.evidence_layers.%d.attention.in_proj_weight", i);
            const gguf_tensor *ip = head_t(e, name);
            if (!mw || !mb || !ip) return gerr(err, errlen, "head tensors missing");
            [enc setComputePipelineState:g->ps[@"layernorm"]];
            [enc setBytes:&lm length:sizeof(lm) atIndex:0];
            [enc setBuffer:g->mem offset:0 atIndex:1];
            [enc setBuffer:g->weights offset:woff(mw) atIndex:2];
            [enc setBuffer:g->weights offset:woff(mb) atIndex:3];
            [enc setBuffer:g->mn32 offset:0 atIndex:4];
            [enc setBuffer:g->mn16 offset:0 atIndex:5];
            rows_kernel(g, enc, @"layernorm", T, W);
            if (g->head_f32) gemm(g, enc, X_F32, false, g->mn32, 0, woff(ip) + (NSUInteger)W * W * 2, 2 * W, W, g->kv[i], 0, T);
            else gemm(g, enc, X_BF16, false, g->mn16, 0, woff(ip) + (NSUInteger)W * W * 2, 2 * W, W, g->kv[i], 0, T);
        }
        for (int i = 0; i < c->head_layers; i++) {
            snprintf(name, sizeof(name), "head.layers.%d.multihead_attn.in_proj_weight", i);
            const gguf_tensor *ip = head_t(e, name);
            if (!ip) return gerr(err, errlen, "head tensors missing");
            if (g->head_f32) gemm(g, enc, X_F32, false, g->mem, 0, woff(ip) + (NSUInteger)W * W * 2, 2 * W, W, g->kv[c->routing_layers + i], 0, T);
            else gemm(g, enc, X_BF16, false, g->mem16, 0, woff(ip) + (NSUInteger)W * W * 2, 2 * W, W, g->kv[c->routing_layers + i], 0, T);
        }
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error) { snprintf(err, errlen, "metal: %s", cb.error.localizedDescription.UTF8String); return false; }
        if (g->cb_failed) return gerr(err, errlen, "cannot create a command buffer (mid-pass)");   // see new_cb
        if (dump_layers) memcpy(dump_layers + (size_t)(c->n_layer + 1) * R * H, (float *)g->hfin.contents + (size_t)(T - R) * H, (size_t)R * H * 4);
        if (g->profile) {
            g->prof_ms[P_OTHER] += (cb.GPUEndTime - cb.GPUStartTime) * 1e3;
            fprintf(stderr, "clef: profile T=%d:", T);
            for (int i = 0; i < P_N; i++) { fprintf(stderr, " %s %.1f ms", prof_name[i], g->prof_ms[i]); g->prof_ms[i] = 0; }
            fprintf(stderr, "\n");
        }

        if (overflow) {
            const int32_t *flag = (const int32_t *)g->ovf.contents;
            for (int r = 0; r < n_seq; r++) overflow[r] = flag[seq_bounds[r]] != 0;
        }
        in->nh = (const float *)g->nh32.contents;
        in->n_kv = g->n_kv;
        for (int i = 0; i < g->n_kv; i++) in->kv[i] = (const float *)g->kv[i].contents;
    }
    return true;
}
