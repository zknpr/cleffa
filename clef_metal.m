// Metal backend: zero-copy weights, one command buffer per packed batch.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "clef_engine.h"
#include "clef_metal_src.inc"   // clef_metal_src / clef_metal_src_len, generated from metal/clef.metal

struct clef_gpu {
    id<MTLDevice> dev;
    id<MTLCommandQueue> queue;
    id<MTLBuffer> weights;            // the whole mmap'd GGUF, no copy
    NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *ps;
    id<MTLBuffer> inv_freq;
    id<MTLBuffer> keep;               // keep-warm sink (clef_gpu_keepalive)
    int cap;                          // token capacity of the activation buffers
    int gdn_cap;                      // longest record capacity, rounded to 32 tokens
    id<MTLBuffer> gdn_w, gdn_u, gdn_ke, gdn_a, gdn_e;
    id<MTLBuffer> ids, pos, seq_start, seq_bounds;
    id<MTLBuffer> x, xn, P, Q, K, V, G, A, Xc, beta, gate, O, hfin, nh32, nh16, mem, mem16, mn16, mn32;
    id<MTLBuffer> ovf;                // FP16 overflow flags, int per token, set at each record's first token
    id<MTLBuffer> attn_blk;           // attention_tu query blocks: 8 ints each, at most one block per token
    id<MTLBuffer> kv[16];
    int n_kv;
    int act_f16;                      // FP16 GEMM inputs per producer class (ACT_*), else BF16
    float f16_lim;                    // magnitude past which an FP16 operand counts as overflow
    bool head_f32;                    // head inputs unrounded, head GEMMs on f32 activations
    int scan_lpc;                     // lanes per value column in gdn_scan (CLEF_SCAN_LPC)
    bool attn_ref;                    // CLEF_ATTN_REF=1: the simple reference attention kernel
    bool attn_tu;                     // attention_tu in FP16 passes; false = FP32 kernels everywhere (CLEF_ATTN_TU=0)
    bool debug_poison;                // CLEF_DEBUG_POISON=1 (tests): fill every activation buffer with NaN before each forward
    bool flash_gemm;                  // Flash matrix shapes qualify for selected short 64-row tiles
    bool profile;                     // CLEF_PROFILE=1: GPU time per kernel category (serializes!)
    double prof_ms[16];
    bool stage_time;                  // CLEF_STAGE_TIME=1: host wall time per forward stage (does not serialize)
    bool cb_failed;                   // a command buffer or encoder could not be created this pass
    int nil_cb_at, cb_count;          // test hook CLEF_DEBUG_NIL_CMDBUF: creation #nil_cb_at fails
    int ck_fail_at, ck_count;         // test hook CLEF_DEBUG_PREFIX_CKPT_FAIL: checkpoint allocation #ck_fail_at fails
};

// Cached backbone state of one record's leading tokens (clef_gpu_forward_prefix). The rows of every
// buffer are the record's own token indices, so a later request that starts with the same tokens
// computes only its remaining rows and reads the rest here.
struct clef_gpu_prefix {
    int cap;                                              // token rows of capacity, a multiple of 1024
    id<MTLBuffer> K[CLEF_MAX_LAYERS], V[CLEF_MAX_LAYERS]; // attention layers: FP32 [nkv][cap][hd], then 64 zero rows
    // Checkpoints: the DeltaNet layers' state [n][Hv][dk][dv] and conv tail [n][ksize-1][C] at one
    // row of the record per slot. A slot's buffers are allocated when a pass first stores into it.
    id<MTLBuffer> ck_state[CLEF_PREFIX_CKPT], ck_tail[CLEF_PREFIX_CKPT];
    id<MTLBuffer> kv[16];                                 // the head's memory K|V rows [cap][2W]
};

enum { P_GEMM, P_ATTN, P_SCAN, P_NORM, P_ATTNPREP, P_CONV, P_GDNOUT, P_SWIGLU, P_HEAD, P_N };
// GEMM input classes, by the kernel that produces them
enum { ACT_NORM = 1, ACT_ATTN = 2, ACT_GDN = 4, ACT_MLP = 8 };
enum { X_BF16, X_F16, X_F32 };
// norm includes the embedding; conv_prep is ssm_conv + gdn_prep; head is the final norm and the GPU half of the head
static const char *prof_name[P_N] = { "gemm", "attention", "gdn_scan", "norm", "attn_prep", "conv_prep", "gdn_out", "swiglu", "head" };

// Command buffers and encoders are nullable (device loss, resource exhaustion). Messages to nil
// are no-ops and a nil buffer has no .error, so a missing one let the pass "succeed" without
// running and the head read the previous pass's activations, another request's (review #4,
// tests/test_gpu_fail.sh). Every forward-pass creation goes through here; a failure sticks for the pass and
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

static double wall_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
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
    // Direct attention fragment rescaling accesses the two local float elements.
    if ((!strcmp(name, "attention_reuse_4") || !strcmp(name, "attention_prefetch_64")) &&
        p.threadExecutionWidth != 32) {
        snprintf(err, errlen, "metal: %s requires a 32-lane SIMDgroup", name);
        return nil;
    }
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
                                "gemm_f16_64x128", "gemm_f16_acc_64x128", "gemm_f16_g4_64x128", "gemm_f16_g4_acc_64x128",
                                "gemm_f16_32x256", "gemm_f16_acc_32x256",
                                "gemm_f32_32x128", "gemm_f32_acc_32x128", "embed", "rmsnorm_act", "rmsnorm_f32", "layernorm",
                                "f32_to_bf16", "fill_zero", "keepalive", "attn_prep", "attention", "attention_fa", "attention_reuse_4", "attention_prefetch_64",
                                "attn_prep_prefix", "attention_prefix_64", "attn_prep_split", "attention_tu", "ssm_conv", "ssm_conv_tail", "ssm_tail_save", "gdn_prep", "gdn_scan_2", "gdn_scan_4", "gdn_scan_8",
                                "gdn_scan_st_2", "gdn_scan_st_4", "gdn_scan_st_8",
                                "gdn_chunk_prep_32", "gdn_chunk_scan_32_16", "gdn_chunk_scan_32_32",
                                "gdn_chunk_scan_st_32_16", "gdn_chunk_scan_st_32_32", "gdn_out", "swiglu" };
        for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); i++) {
            if (!pipeline(g, lib, names[i], err, errlen)) { free(g); return NULL; }
        }

        const clef_config *c = &e->cfg;
        g->flash_gemm = c->H == 4096;
        float inv[64];
        for (int j = 0; j < c->n_rot / 2; j++) {
            // HF: 1 / base ** (arange(0, dim, 2).float() / dim), all f32
            inv[j] = 1.0f / powf(c->rope_theta, (float)(2 * j) / (float)c->n_rot);
        }
        g->inv_freq = [g->dev newBufferWithBytes:inv length:sizeof(inv) options:MTLResourceStorageModeShared];
        if (!g->inv_freq) { gerr(err, errlen, "cannot allocate rope table"); free(g); return NULL; }
        g->keep = [g->dev newBufferWithLength:16 options:MTLResourceStorageModeShared];
        if (!g->keep) { gerr(err, errlen, "cannot allocate keep-warm buffer"); free(g); return NULL; }
        g->n_kv = c->routing_layers + c->head_layers;
        g->profile = getenv("CLEF_PROFILE") != NULL;
        g->stage_time = getenv("CLEF_STAGE_TIME") != NULL;
        g->attn_ref = getenv("CLEF_ATTN_REF") != NULL;
        // Compensated half operands use tensor units with FP32 accumulation. Numerical and
        // task-level validation are in docs/attention.md; CLEF_ATTN_TU=0
        // keeps the previous FP32 attention kernels.
        const char *tu = getenv("CLEF_ATTN_TU");
        g->attn_tu = !tu || strcmp(tu, "0") != 0;
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
        g->ck_fail_at = getenv("CLEF_DEBUG_PREFIX_CKPT_FAIL") ? atoi(getenv("CLEF_DEBUG_PREFIX_CKPT_FAIL")) : 0;
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
        g->hfin = g->nh32 = g->nh16 = g->mem = g->mem16 = g->mn16 = g->mn32 = g->ovf = g->attn_blk = g->inv_freq = g->keep = nil;
        g->gdn_w = g->gdn_u = g->gdn_ke = g->gdn_a = g->gdn_e = nil;
        for (int i = 0; i < 16; i++) g->kv[i] = nil;
    }
    free(g);
}

// One-thread passes that each read one element of the weights or of an activation buffer.
// A pass that follows 1 to 1.5 s without a command buffer using the weights waits ~205 ms (27B,
// 54 GB) between commit and GPU start; the GPU's execution interval itself is unchanged. This is
// consistent with the buffer losing its GPU residency. What matters is referencing the weights:
// the same pass bound to another buffer does not help, and a residency set attached to the queue
// only moves the threshold to 2-3 s. A pass every second or less removes the delay at every gap
// measured (to 10 s). It only reads, and writes a private sink, so it cannot change a result.
// The caller must not run it concurrently with clef_gpu_forward (the server calls it from the
// worker thread, between batches).
bool clef_gpu_keepalive(clef_gpu *g, char *err, size_t errlen) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [g->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = cb ? [cb computeCommandEncoder] : nil;
        if (!cb || !enc) return gerr(err, errlen, "cannot create a command buffer (keep-warm)");
        [enc setComputePipelineState:g->ps[@"keepalive"]];
        [enc setBuffer:g->keep offset:0 atIndex:1];
        // Activation allocations also showed an idle-start penalty (~20 ms at a 4,096-token
        // capacity on the 27B), so each gets its own one-thread dispatch too. They are nil before the first
        // forward. CLEF_DEBUG_KEEPWARM_NOWEIGHTS leaves the weights out: the control that shows
        // the weights reference is what removes the large delay.
        id<MTLBuffer> bufs[64];
        int n = 0;
        if (!getenv("CLEF_DEBUG_KEEPWARM_NOWEIGHTS")) bufs[n++] = g->weights;
        id<MTLBuffer> act[] = { g->ids, g->pos, g->seq_start, g->seq_bounds, g->x, g->xn, g->P, g->Q, g->K, g->V, g->G,
                                g->A, g->Xc, g->beta, g->gate, g->O, g->hfin, g->nh32, g->nh16, g->mem, g->mem16,
                                g->mn16, g->mn32, g->ovf, g->attn_blk, g->gdn_w, g->gdn_u, g->gdn_ke, g->gdn_a, g->gdn_e };
        for (size_t i = 0; i < sizeof(act) / sizeof(act[0]); i++) if (act[i]) bufs[n++] = act[i];
        for (int i = 0; i < g->n_kv; i++) if (g->kv[i]) bufs[n++] = g->kv[i];
        for (int i = 0; i < n; i++) {
            [enc setBuffer:bufs[i] offset:0 atIndex:0];
            [enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        }
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error) { snprintf(err, errlen, "metal: %s", cb.error.localizedDescription.UTF8String); return false; }
    }
    return true;
}

static id<MTLBuffer> buf(clef_gpu *g, size_t bytes) {
    return [g->dev newBufferWithLength:(bytes ? bytes : 16) options:MTLResourceStorageModeShared];
}

static bool gdn_chunked(const clef_config *c, int length) {
    // Choose per record, never by the packed token count: these FP32 algorithms
    // have different reduction orders. Restrict the change to measured long 27B work.
    // gdn_chunk_prep/scan address the V and O heads at a fixed 128 columns (dk is already
    // 128 by load_config), while load_config admits any dv divisible by 16: a crafted GGUF
    // with Hv=48 and another dv must stay on the sequential scan, which reads a.dv (review #12).
    return c->Hv == 48 && c->dv == 128 && length >= 4096;
}

static bool ensure_gdn_capacity(clef_gpu *g, const clef_config *c, int T, char *err, size_t errlen) {
    if (T <= g->gdn_cap) return true;
    const size_t rows = ((size_t)T + 31) / 32 * 32;
    if (rows > INT_MAX || c->Hv <= 0 || rows > g->dev.maxBufferLength / (128 * 4) / (size_t)c->Hv)
        return gerr(err, errlen, "DeltaNet scratch exceeds buffer limit");
    const size_t nr = rows * (size_t)c->Hv;
    // Commit all buffers together so allocation failure leaves the previous capacity usable.
    id<MTLBuffer> w = buf(g, nr * 128 * 4), u = buf(g, nr * 128 * 4), ke = buf(g, nr * 128 * 4);
    id<MTLBuffer> a = buf(g, nr * 32 * 4), e = buf(g, nr * 4);
    if (!w || !u || !ke || !a || !e)
        return gerr(err, errlen, "cannot allocate DeltaNet scratch (previous buffers kept)");
    g->gdn_w = w; g->gdn_u = u; g->gdn_ke = ke; g->gdn_a = a; g->gdn_e = e;
    g->gdn_cap = (int)rows;
    return true;
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
    id<MTLBuffer> ids, pos, seq_start, seq_bounds, x, xn, P, Q, K, V, G, A, Xc, beta, gate, O, hfin, nh32, nh16, mem, mem16, mn16, mn32, ovf, attn_blk;
    id<MTLBuffer> kv[16] = { nil };
    @autoreleasepool {
        ids = buf(g, Tc * 4); pos = buf(g, Tc * 4); seq_start = buf(g, Tc * 4); seq_bounds = buf(g, (Tc + 1) * 4); ovf = buf(g, Tc * 4);
        attn_blk = buf(g, Tc * 32);
        x = buf(g, Tc * c->H * 4); xn = buf(g, Tc * c->H * 2);
        P = buf(g, Tc * maxN * 4);
        // test hook (tests/test_grow_fail.sh): simulate the largest allocation failing when
        // capacity would grow past CLEF_DEBUG_GROW_FAIL_ABOVE tokens
        const char *gf = getenv("CLEF_DEBUG_GROW_FAIL_ABOVE");
        if (gf && cap > atoi(gf)) P = nil;
        // +64 FP32 rows cover prefetch loads beyond the final head.
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
    id<MTLBuffer> all[] = { ids, pos, seq_start, seq_bounds, x, xn, P, Q, K, V, G, A, Xc, beta, gate, O, hfin, nh32, nh16, mem, mem16, mn16, mn32, ovf, attn_blk };
    for (size_t i = 0; i < sizeof(all) / sizeof(all[0]); i++) {
        if (!all[i]) return gerr(err, errlen, "cannot allocate activation buffers (previous buffers kept)");
    }
    for (int i = 0; i < g->n_kv; i++) if (!kv[i]) return gerr(err, errlen, "cannot allocate activation buffers (previous buffers kept)");
    memset(Q.contents, 0, Q.length); memset(K.contents, 0, K.length); memset(V.contents, 0, V.length);
    g->ids = ids; g->pos = pos; g->seq_start = seq_start; g->seq_bounds = seq_bounds; g->x = x; g->xn = xn; g->P = P;
    g->Q = Q; g->K = K; g->V = V; g->G = G; g->A = A; g->Xc = Xc; g->beta = beta; g->gate = gate; g->O = O;
    g->hfin = hfin; g->nh32 = nh32; g->nh16 = nh16; g->mem = mem; g->mem16 = mem16; g->mn16 = mn16; g->mn32 = mn32; g->ovf = ovf;
    g->attn_blk = attn_blk;
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
    // Larger FP16 tiles improve long-prefill GEMMs without changing the K reduction.
    // Short requests keep 32 rows to avoid wasting work on ragged tiles; the wider
    // 27B matrices benefit from 256 output columns. Long 27B down projections also
    // benefit from that width; keep their measured selection separate from Flash.
    // Exact parity across tile shapes,
    // residual accumulation and packed row offsets is checked by
    // bench/gemm_tiles.m and the model batch tests; BF16 fallback/head shapes stay fixed.
    const bool wide_down = xt == X_F16 && T >= 4096 && N >= 5120 && K >= 2 * N;
    // Below 1K, Flash gains when 64-row tiles add no padding beyond 32-row tiles.
    // The 27B short matrices keep their separately measured 32x256 selection.
    const bool short_flash = g->flash_gemm && T >= 768 && T < 1024 &&
                             ((T & 63) == 0 || (T & 63) > 32);
    const int tm = xt == X_F16 && (T >= 1024 || short_flash) && !wide_down ? 64 : 32;
    const int tn = wide_down || (xt == X_F16 && T < 1024 && !short_flash && K >= 5120 && N >= 5120) ? 256 : 128;
    static NSString *const names[3][2] = { { @"gemm_bf16_32x128", @"gemm_bf16_acc_32x128" },
                                           { @"gemm_f16_32x128", @"gemm_f16_acc_32x128" },
                                           { @"gemm_f32_32x128", @"gemm_f32_acc_32x128" } };
    NSString *name = names[xt][acc];
    if (tn == 256) name = acc ? @"gemm_f16_acc_32x256" : @"gemm_f16_32x256";
    if (tm == 64) name = acc ? @"gemm_f16_acc_64x128" : @"gemm_f16_64x128";
    // Group four tile rows for long 27B expansion projections; keep down projections
    // and Flash on their separately measured dispatch. Per-element reductions are unchanged.
    if (xt == X_F16 && T >= 4096 && K >= 5120 && (long)N >= 4L * K)
        name = acc ? @"gemm_f16_g4_acc_64x128" : @"gemm_f16_g4_64x128";

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
typedef struct { int nh, nkv, hd, n_rot, row; float eps; int kvT, koff; } attn_prefix_prep_args;
typedef struct { int nh, nkv, hd, T; float scale; } attn_args;
typedef struct { int nh, nkv, hd, n_rot, row; float eps, lim; int kvT, koff; } attn_split_args;
typedef struct { int nh, nkv, T, kvT; float scale; } attn_tu_args;
typedef struct { int nh, nkv, hd, T; float scale; int kvT, qoff; } attn_prefix_args;
typedef struct { int C, row, ksize; } conv_args;
typedef struct { int C, row, first; } tail_args;
typedef struct { int Hk, Hv, dk, dv, row, C; } gdn_prep_args;
typedef struct { int Hk, Hv, dk, dv, C; } gdn_args;
typedef struct { int Hv, dv, row, zoff; float eps; } gdn_out_args;
typedef struct { int I; } swiglu_args;

static void encode_gdn_scan(clef_gpu *g, id<MTLComputeCommandEncoder> enc, const clef_config *c,
                            int C, const int32_t *bounds, int first, int count) {
    gdn_args ga = { c->Hk, c->Hv, c->dk, c->dv, C };
    const int start = bounds[first], length = bounds[first + 1] - start;
    [enc setBytes:&ga length:sizeof(ga) atIndex:0];
    if (count == 1 && gdn_chunked(c, length)) {
        [enc setBuffer:g->Xc offset:(NSUInteger)start * C * 4 atIndex:1];
        [enc setBuffer:g->beta offset:(NSUInteger)start * c->Hv * 4 atIndex:2];
        [enc setBuffer:g->gate offset:(NSUInteger)start * c->Hv * 4 atIndex:3];
        [enc setBuffer:g->O offset:(NSUInteger)start * c->Hv * c->dv * 4 atIndex:5];
        [enc setBuffer:g->gdn_w offset:0 atIndex:6];
        [enc setBuffer:g->gdn_u offset:0 atIndex:7];
        [enc setBuffer:g->gdn_ke offset:0 atIndex:8];
        [enc setBuffer:g->gdn_a offset:0 atIndex:9];
        [enc setBuffer:g->gdn_e offset:0 atIndex:10];
        [enc setBytes:&length length:sizeof(length) atIndex:11];
        [enc setComputePipelineState:g->ps[@"gdn_chunk_prep_32"]];
        [enc dispatchThreadgroups:MTLSizeMake((length + 31) / 32, c->Hv, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        const int vc = c->Hv == 48 && length < 8192 ? 16 : 32;
        [enc setComputePipelineState:g->ps[vc == 16 ? @"gdn_chunk_scan_32_16" : @"gdn_chunk_scan_32_32"]];
        [enc dispatchThreadgroups:MTLSizeMake(c->Hv, 128 / vc, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    } else {
        [enc setComputePipelineState:g->ps[[NSString stringWithFormat:@"gdn_scan_%d", g->scan_lpc]]];
        [enc setBuffer:g->Xc offset:0 atIndex:1];
        [enc setBuffer:g->beta offset:0 atIndex:2];
        [enc setBuffer:g->gate offset:0 atIndex:3];
        [enc setBuffer:g->seq_bounds offset:(NSUInteger)first * 4 atIndex:4];
        [enc setBuffer:g->O offset:0 atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(c->Hv, c->dv / (32 / g->scan_lpc), count) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    }
}

// One record whose first L tokens come from a prefix-cache entry. The scan is split at the
// plan's checkpoint rows: each segment starts from the state of the checkpoint at its first row
// (the one the pass resumes from, or the one the previous segment stored; zero at the record's
// row 0) and stores the checkpoint at its end. L and the rows are multiples of 32, so the
// chunked path forms the 32-token blocks of an uncached pass over the whole record, and both
// paths apply the same updates in the same order however many segments there are.
static void encode_gdn_scan_px(clef_gpu *g, id<MTLComputeCommandEncoder> enc, const clef_config *c, int C,
                               clef_gpu_prefix *px, int gi, int T, int L, const clef_prefix_plan *plan) {
    gdn_args ga = { c->Hk, c->Hv, c->dk, c->dv, C };
    const NSUInteger soff = (NSUInteger)gi * c->Hv * c->dk * c->dv * 4;
    const bool chunked = gdn_chunked(c, L + T);   // by the record's length, as without an entry
    for (int seg = 0; seg <= plan->n; seg++) {
        const int start = seg ? plan->row[seg - 1] - L : 0;                // pass rows
        const int length = (seg < plan->n ? plan->row[seg] - L : T) - start;
        const int from = seg ? plan->slot[seg - 1] : L > 0 ? plan->load : -1;
        const int to = seg < plan->n ? plan->slot[seg] : -1;
        const int32_t st[2] = { from >= 0, to >= 0 };   // load, store
        // an unused side still gets a buffer; clef_gpu_forward_prefix rules out a pass with neither
        id<MTLBuffer> sin = px->ck_state[from >= 0 ? from : to], sout = px->ck_state[to >= 0 ? to : from];
        [enc setBytes:&ga length:sizeof(ga) atIndex:0];
        if (chunked) {
            [enc setBuffer:g->Xc offset:(NSUInteger)start * C * 4 atIndex:1];
            [enc setBuffer:g->beta offset:(NSUInteger)start * c->Hv * 4 atIndex:2];
            [enc setBuffer:g->gate offset:(NSUInteger)start * c->Hv * 4 atIndex:3];
            [enc setBuffer:g->O offset:(NSUInteger)start * c->Hv * c->dv * 4 atIndex:5];
            [enc setBuffer:g->gdn_w offset:0 atIndex:6];
            [enc setBuffer:g->gdn_u offset:0 atIndex:7];
            [enc setBuffer:g->gdn_ke offset:0 atIndex:8];
            [enc setBuffer:g->gdn_a offset:0 atIndex:9];
            [enc setBuffer:g->gdn_e offset:0 atIndex:10];
            [enc setBytes:&length length:sizeof(length) atIndex:11];
            [enc setBuffer:sin offset:soff atIndex:12];
            [enc setBytes:st length:sizeof(st) atIndex:13];
            [enc setBuffer:sout offset:soff atIndex:14];
            [enc setComputePipelineState:g->ps[@"gdn_chunk_prep_32"]];
            [enc dispatchThreadgroups:MTLSizeMake((length + 31) / 32, c->Hv, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            const int vc = c->Hv == 48 && L + T < 8192 ? 16 : 32;   // the two value tiles return identical bits
            [enc setComputePipelineState:g->ps[vc == 16 ? @"gdn_chunk_scan_st_32_16" : @"gdn_chunk_scan_st_32_32"]];
            [enc dispatchThreadgroups:MTLSizeMake(c->Hv, 128 / vc, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        } else {
            const int32_t bounds[2] = { start, start + length };
            [enc setComputePipelineState:g->ps[[NSString stringWithFormat:@"gdn_scan_st_%d", g->scan_lpc]]];
            [enc setBuffer:g->Xc offset:0 atIndex:1];
            [enc setBuffer:g->beta offset:0 atIndex:2];
            [enc setBuffer:g->gate offset:0 atIndex:3];
            [enc setBytes:bounds length:sizeof(bounds) atIndex:4];
            [enc setBuffer:g->O offset:0 atIndex:5];
            [enc setBuffer:sin offset:soff atIndex:6];
            [enc setBytes:st length:sizeof(st) atIndex:7];
            [enc setBuffer:sout offset:soff atIndex:8];
            [enc dispatchThreadgroups:MTLSizeMake(c->Hv, c->dv / (32 / g->scan_lpc), 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        }
    }
}

clef_gpu_prefix *clef_gpu_prefix_new(void) { return (clef_gpu_prefix *)calloc(1, sizeof(clef_gpu_prefix)); }

void clef_gpu_prefix_free(clef_gpu_prefix *px) {
    if (!px) return;
    @autoreleasepool {
        for (int i = 0; i < CLEF_MAX_LAYERS; i++) px->K[i] = px->V[i] = nil;
        for (int i = 0; i < 16; i++) px->kv[i] = nil;
        for (int i = 0; i < CLEF_PREFIX_CKPT; i++) px->ck_state[i] = px->ck_tail[i] = nil;
    }
    free(px);
}

size_t clef_gpu_prefix_bytes(const clef_gpu_prefix *px) {
    size_t n = 0;
    for (int i = 0; i < CLEF_PREFIX_CKPT; i++) n += px->ck_state[i].length + px->ck_tail[i].length;
    for (int i = 0; i < CLEF_MAX_LAYERS; i++) n += px->K[i].length + px->V[i].length;
    for (int i = 0; i < 16; i++) n += px->kv[i].length;
    return n;
}

// Keep-warm for an entry's buffers, as clef_gpu_keepalive does for the engine's: without it a hit
// that follows an idle gap started 20 to 30 ms late (27B, 3.1 GB entry). Same rules: it only
// reads, and must not run concurrently with a forward.
bool clef_gpu_prefix_keepalive(clef_gpu *g, const clef_gpu_prefix *px, char *err, size_t errlen) {
    if (!px->cap) return true;
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [g->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = cb ? [cb computeCommandEncoder] : nil;
        if (!cb || !enc) return gerr(err, errlen, "cannot create a command buffer (prefix keep-warm)");
        [enc setComputePipelineState:g->ps[@"keepalive"]];
        [enc setBuffer:g->keep offset:0 atIndex:1];
        for (int i = 0; i < 2 * CLEF_MAX_LAYERS + 16 + 2 * CLEF_PREFIX_CKPT; i++) {
            const int k = i - 2 * CLEF_MAX_LAYERS - 16;   // the checkpoint buffers follow the K, V and memory planes
            id<MTLBuffer> b = i < CLEF_MAX_LAYERS ? px->K[i] : i < 2 * CLEF_MAX_LAYERS ? px->V[i - CLEF_MAX_LAYERS]
                            : k < 0 ? px->kv[i - 2 * CLEF_MAX_LAYERS]
                            : k < CLEF_PREFIX_CKPT ? px->ck_state[k] : px->ck_tail[k - CLEF_PREFIX_CKPT];
            if (!b) continue;
            [enc setBuffer:b offset:0 atIndex:0];
            [enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        }
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error) { snprintf(err, errlen, "metal: %s", cb.error.localizedDescription.UTF8String); return false; }
    }
    return true;
}

// The entry holds attention K/V in the engine's fixed layout and state from an FP16-output pass.
bool clef_gpu_prefix_supported(const clef_gpu *g) { return !g->attn_ref && (g->act_f16 & ACT_ATTN); }

// The DeltaNet scan of a record depends on its length, so an entry serves only records of its class.
int clef_gpu_prefix_class(const clef_engine *e, int length) { return gdn_chunked(&e->cfg, length); }

// Touch every page from the CPU. Left to the pass, the first GPU write to each fresh page takes
// the fault instead, which made the pass that fills a new 16k-token entry 1.6 s (flash) and
// 4.4 s (27B) slower than an uncached one.
static void prefault(id<MTLBuffer> b) {
    volatile char *p = (volatile char *)b.contents;
    const size_t page = (size_t)getpagesize();
    for (size_t i = 0; i < b.length; i += page) p[i] = 0;
}

// Capacity for `rows` tokens, keeping rows [0, keep). Every new buffer is allocated before any is
// replaced, so a failure leaves the entry as it was. New buffers are zero-filled by Metal: the
// slack after the planes must be zero, and rows not written yet must be finite. An entry that
// has to grow grows by at least a quarter, so a state that keeps growing copies its entry a
// handful of times and not at every 1,024 tokens.
static bool prefix_reserve(clef_gpu *g, const clef_config *c, clef_gpu_prefix *px, int rows, int keep, char *err, size_t errlen) {
    if (rows <= px->cap) return true;
    size_t want = (size_t)rows;
    if (want < (size_t)px->cap + (size_t)px->cap / 4) want = (size_t)px->cap + (size_t)px->cap / 4;
    const size_t cap = (want + 1023) / 1024 * 1024;
    const size_t kvb = ((size_t)c->nkv * cap + 64) * c->hd * 4; // FP32 heads plus final-tile slack
    const size_t memb = cap * 2 * c->W * 4;
    if (cap > INT_MAX || kvb > g->dev.maxBufferLength || memb > g->dev.maxBufferLength)
        return gerr(err, errlen, "prefix cache entry exceeds buffer limit");
    id<MTLBuffer> K[CLEF_MAX_LAYERS] = { nil }, V[CLEF_MAX_LAYERS] = { nil }, kv[16] = { nil };
    bool ok = true;
    @autoreleasepool {
        for (int l = 0; ok && l < c->n_layer; l++) {
            if (!c->layer_full[l]) continue;
            K[l] = buf(g, kvb); V[l] = buf(g, kvb);
            ok = K[l] && V[l];
        }
        for (int i = 0; ok && i < g->n_kv; i++) ok = (kv[i] = buf(g, memb)) != nil;
        // test hook (tests/test_prefix_cache.sh): the entry cannot grow past N tokens
        const char *lim = getenv("CLEF_DEBUG_PREFIX_FAIL_ABOVE");
        if (lim && cap > (size_t)atoi(lim)) ok = false;
    }
    if (!ok) return gerr(err, errlen, "cannot allocate the prefix cache entry (previous entry kept)");
    for (int l = 0; l < c->n_layer; l++) if (c->layer_full[l]) { prefault(K[l]); prefault(V[l]); }
    for (int i = 0; i < g->n_kv; i++) prefault(kv[i]);
    if (keep > 0) {
        // FP32 heads, or with attention_tu hi and lo half planes of each head: the same bytes in
        // all, at another stride. The kernel is fixed for the engine's life, so an entry never mixes.
        const bool halves = g->attn_tu && !g->attn_ref;
        const size_t row = (size_t)c->hd * (halves ? 2 : 4);
        for (int l = 0; l < c->n_layer; l++) {
            if (!c->layer_full[l]) continue;
            for (int pl = 0; pl < (halves ? 2 : 1) * c->nkv; pl++) {
                memcpy((char *)K[l].contents + pl * cap * row, (char *)px->K[l].contents + (size_t)pl * px->cap * row, keep * row);
                memcpy((char *)V[l].contents + pl * cap * row, (char *)px->V[l].contents + (size_t)pl * px->cap * row, keep * row);
            }
        }
        for (int i = 0; i < g->n_kv; i++) memcpy(kv[i].contents, px->kv[i].contents, (size_t)keep * 2 * c->W * 4);
    }
    for (int l = 0; l < c->n_layer; l++) { px->K[l] = K[l]; px->V[l] = V[l]; }
    for (int i = 0; i < g->n_kv; i++) px->kv[i] = kv[i];
    px->cap = (int)cap;
    return true;
}

// Buffers for one checkpoint slot, allocated the first time a pass stores into it and kept after
// that. Pre-faulted like the planes above, for the same reason.
static bool ck_reserve(clef_gpu *g, const clef_config *c, clef_gpu_prefix *px, int slot, char *err, size_t errlen) {
    if (px->ck_state[slot] && px->ck_tail[slot]) return true;
    const int C = 2 * c->Hk * c->dk + c->Hv * c->dv;
    int n_gdn = 0;
    for (int l = 0; l < c->n_layer; l++) n_gdn += !c->layer_full[l];
    @autoreleasepool {
        id<MTLBuffer> state = buf(g, (size_t)n_gdn * c->Hv * c->dk * c->dv * 4);
        id<MTLBuffer> tail = buf(g, (size_t)n_gdn * (c->ssm_kernel - 1) * C * 4);
        // test hook (tests/test_prefix_checkpoints.py): the Nth checkpoint allocation of the engine fails
        if (++g->ck_count == g->ck_fail_at) state = nil;
        if (!state || !tail) return gerr(err, errlen, "cannot allocate the prefix cache entry (checkpoint)");
        prefault(state); prefault(tail);
        px->ck_state[slot] = state; px->ck_tail[slot] = tail;
    }
    return true;
}

// One attention layer on the tensor units: attn_prep_split writes Q, K and V as hi/lo half planes,
// K and V into the prefix-cache entry when px is set, then attention_tu reads them. Only in
// passes whose attention output is FP16, where the caller acts on the overflow flags.
static bool encode_attention_tu(clef_gpu *g, id<MTLCommandBuffer> __strong *cb, id<MTLComputeCommandEncoder> __strong *encp,
                                const clef_config *c, const clef_layer_w *lw, int l, int T, int n_attn, int n_blk,
                                const act_args *ac, clef_gpu_prefix *px, int L, char *err, size_t errlen) {
    id<MTLComputeCommandEncoder> enc = *encp;
    id<MTLBuffer> Kb = px ? px->K[l] : g->K, Vb = px ? px->V[l] : g->V;
    const int kvT = px ? px->cap : T;
    attn_split_args pa = { c->nh, c->nkv, c->hd, c->n_rot, n_attn, c->eps, g->f16_lim, kvT, L };
    [enc setComputePipelineState:g->ps[@"attn_prep_split"]];
    [enc setBytes:&pa length:sizeof(pa) atIndex:0];
    [enc setBuffer:g->P offset:0 atIndex:1];
    [enc setBuffer:g->weights offset:woff(lw->attn_q_norm) atIndex:2];
    [enc setBuffer:g->weights offset:woff(lw->attn_k_norm) atIndex:3];
    [enc setBuffer:g->pos offset:0 atIndex:4];
    [enc setBuffer:g->inv_freq offset:0 atIndex:5];
    [enc setBuffer:g->Q offset:0 atIndex:6];
    [enc setBuffer:Kb offset:0 atIndex:7];
    [enc setBuffer:Vb offset:0 atIndex:8];
    [enc setBuffer:g->G offset:0 atIndex:9];
    [enc setBytes:&T length:sizeof(T) atIndex:10];
    [enc setBuffer:g->seq_start offset:0 atIndex:11];
    [enc setBuffer:g->ovf offset:0 atIndex:12];
    [enc dispatchThreadgroups:MTLSizeMake(T, c->nh + c->nkv, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    // Zero the rows the last head's tiles read past the lo planes, which end where the float
    // layout does: 31 half rows for Q, 127 for K and V. An entry's buffers end in rows that stay
    // zero, and its rows past this record are finite (split16), which is all a masked tile needs.
    {
        struct { id<MTLBuffer> b; NSUInteger off; long n; } z[3] = {
            { g->Q, (NSUInteger)c->nh * T * c->hd * 4, 16L * c->hd },
            { g->K, (NSUInteger)c->nkv * T * c->hd * 4, 64L * c->hd },
            { g->V, (NSUInteger)c->nkv * T * c->hd * 4, 64L * c->hd },
        };
        [enc setComputePipelineState:g->ps[@"fill_zero"]];
        for (int i = 0; i < (px ? 1 : 3); i++) {
            [enc setBuffer:z[i].b offset:z[i].off atIndex:0];
            [enc setBytes:&z[i].n length:sizeof(long) atIndex:1];
            [enc dispatchThreads:MTLSizeMake((NSUInteger)z[i].n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
    }
    if (!prof(g, cb, encp, P_ATTNPREP, err, errlen)) return false;
    enc = *encp;
    attn_tu_args ta = { c->nh, c->nkv, T, kvT, 1.0f / sqrtf((float)c->hd) };
    [enc setComputePipelineState:g->ps[@"attention_tu"]];
    [enc setBytes:&ta length:sizeof(ta) atIndex:0];
    [enc setBuffer:g->Q offset:0 atIndex:1];
    [enc setBuffer:Kb offset:0 atIndex:2];
    [enc setBuffer:Vb offset:0 atIndex:3];
    [enc setBuffer:g->G offset:0 atIndex:4];
    [enc setBuffer:g->attn_blk offset:0 atIndex:5];
    [enc setBuffer:g->A offset:0 atIndex:6];
    [enc setBytes:ac length:sizeof(*ac) atIndex:7];
    [enc setBuffer:g->ovf offset:0 atIndex:8];
    [enc dispatchThreadgroups:MTLSizeMake(n_blk, c->nh, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    return prof(g, cb, encp, P_ATTN, err, errlen);
}

static const gguf_tensor *head_t(const clef_engine *e, const char *name) { return gguf_find_tensor(&e->gguf, name); }

// px (with L and plan, else NULL, 0, NULL): the pass is one record whose first L tokens are in
// the prefix-cache entry. T, ids and pos cover its remaining tokens, pos continuing from L. The
// pass writes its own rows into the entry, resumes the DeltaNet layers from the checkpoint
// plan->load and stores their state and conv tail at each of the plan's rows. Everything it
// computes is what a pass over the whole record computes for these rows.
static bool forward(clef_gpu *g, const clef_engine *e, const int32_t *ids, const int32_t *pos,
                    const int32_t *seq_start, const int32_t *seq_bounds, int n_seq, int T, bool bf16_only,
                    bool *overflow, clef_head_inputs *in, float *dump_layers, int dump_rows,
                    clef_gpu_prefix *px, int L, const clef_prefix_plan *plan, char *err, size_t errlen) {
    const int R = dump_rows > 0 && dump_rows < T ? dump_rows : T;   /* dump the last R token rows */
    const clef_config *c = &e->cfg;
    const double t_enter = wall_ms();
    // FP16 operands are only safe with the overflow flags read back (and acted on) by the caller
    if (!bf16_only && g->act_f16 && !overflow) return gerr(err, errlen, "FP16 activations need the overflow flags (pass overflow or bf16_only)");
    if (!ensure_capacity(g, c, T, n_seq, err, errlen)) return false;
    int gdn_longest = 0;
    for (int i = 0; i < n_seq; i++) {
        const int length = seq_bounds[i + 1] - seq_bounds[i];
        if (gdn_chunked(c, length + L) && length > gdn_longest) gdn_longest = length;   // L = 0 without an entry
    }
    if (!ensure_gdn_capacity(g, c, gdn_longest, err, errlen)) return false;
    @autoreleasepool {
        memset(g->prof_ms, 0, sizeof(g->prof_ms));   // a prior failed pass may have partial timings
        if (g->debug_poison) {
            // Test hook: rows a kernel must not depend on hold NaN. Every row it does need is
            // rewritten by attn_prep, so correct code gives identical results (tests/test_poison.sh).
            const float nan = NAN;   // also a NaN pattern in BF16 halves (0x7fc0)
            id<MTLBuffer> bufs[] = { g->x, g->xn, g->P, g->Q, g->K, g->V, g->G, g->A, g->Xc, g->beta, g->gate,
                                     g->O, g->hfin, g->nh32, g->nh16, g->mem, g->mem16, g->mn16, g->mn32 };
            for (size_t i = 0; i < sizeof(bufs) / sizeof(bufs[0]); i++) memset_pattern4(bufs[i].contents, &nan, bufs[i].length);
            if (g->gdn_cap) {
                id<MTLBuffer> scratch[] = { g->gdn_w, g->gdn_u, g->gdn_ke, g->gdn_a, g->gdn_e };
                for (size_t i = 0; i < sizeof(scratch) / sizeof(scratch[0]); i++)
                    memset_pattern4(scratch[i].contents, &nan, scratch[i].length);
            }
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
        // Tensor-unit attention takes half operands, so it runs only where the pass acts on the
        // overflow flags: FP16 attention output. A flagged record's BF16 rerun (and CLEF_ACT_F16
        // without the attention class) keeps the FP32 kernels. Every record of the pass uses the
        // same kernel, so the choice never depends on what a record is packed with.
        const bool tu = f_attn && !g->attn_ref && g->attn_tu;
        int n_blk = 0;
        if (tu) {
            // One entry per 32 query rows of a record, longest causal rows first within a record:
            // qbase, record length, first query token, kbase, overflow flag row (see attention_tu).
            // With a prefix-cache entry the record's token i is pass row i - L and K/V row i.
            int32_t *blk = (int32_t *)g->attn_blk.contents;
            for (int i = 0; i < n_seq; i++) {
                const int start = seq_bounds[i], length = seq_bounds[i + 1] - start + L;
                for (int i0 = (length - 1) / 32 * 32; i0 >= L; i0 -= 32, n_blk++) {
                    int32_t *b = blk + 8 * n_blk;
                    b[0] = start - L; b[1] = length; b[2] = i0; b[3] = px ? 0 : start; b[4] = start;
                    b[5] = b[6] = b[7] = 0;
                }
            }
        }
        int gdn_i = 0;   // DeltaNet layers seen so far
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
                if (!prof(g, &cb, &enc, P_NORM, err, errlen)) return false;
                gemm(g, enc, x_norm, false, g->xn, 0, woff(lw->attn_qkv), n_attn, H, g->P, 0, T);
                if (!prof(g, &cb, &enc, P_GEMM, err, errlen)) return false;
                if (tu) {
                    if (!encode_attention_tu(g, &cb, &enc, c, lw, l, T, n_attn, n_blk, &ac_attn, px, L, err, errlen)) return false;
                } else {
                    attn_prep_args pa = { c->nh, c->nkv, c->hd, c->n_rot, n_attn, c->eps };
                    id<MTLBuffer> Kb = px ? px->K[l] : g->K, Vb = px ? px->V[l] : g->V;
                    [enc setComputePipelineState:g->ps[px ? @"attn_prep_prefix" : @"attn_prep"]];
                    if (px) {
                        attn_prefix_prep_args pp = { c->nh, c->nkv, c->hd, c->n_rot, n_attn, c->eps, px->cap, L };
                        [enc setBytes:&pp length:sizeof(pp) atIndex:0];
                    } else [enc setBytes:&pa length:sizeof(pa) atIndex:0];
                    [enc setBuffer:g->P offset:0 atIndex:1];
                    [enc setBuffer:g->weights offset:woff(lw->attn_q_norm) atIndex:2];
                    [enc setBuffer:g->weights offset:woff(lw->attn_k_norm) atIndex:3];
                    [enc setBuffer:g->pos offset:0 atIndex:4];
                    [enc setBuffer:g->inv_freq offset:0 atIndex:5];
                    [enc setBuffer:g->Q offset:0 atIndex:6];
                    [enc setBuffer:Kb offset:0 atIndex:7];
                    [enc setBuffer:Vb offset:0 atIndex:8];
                    [enc setBuffer:g->G offset:0 atIndex:9];
                    [enc setBytes:&T length:sizeof(T) atIndex:10];
                    [enc dispatchThreadgroups:MTLSizeMake(T, c->nh + c->nkv, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

                    // Zero temporary Q slack and uncached K/V slack before masked tile reads.
                    {
                        struct { id<MTLBuffer> b; NSUInteger off; long n; } z[3] = {
                            { g->Q, (NSUInteger)c->nh * T * c->hd * 4, 8L * c->hd },
                            { g->K, (NSUInteger)c->nkv * T * c->hd * 4, 64L * c->hd },
                            { g->V, (NSUInteger)c->nkv * T * c->hd * 4, 64L * c->hd },
                        };
                        [enc setComputePipelineState:g->ps[@"fill_zero"]];
                        for (int i = 0; i < (px ? 1 : 3); i++) {
                            [enc setBuffer:z[i].b offset:z[i].off atIndex:0];
                            [enc setBytes:&z[i].n length:sizeof(long) atIndex:1];
                            [enc dispatchThreads:MTLSizeMake((NSUInteger)z[i].n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                        }
                    }
                    if (px) {
                        // A replaced or shorter entry may leave old rows after this record. Zero only
                        // unused rows within each head; never overwrite the next head's live prefix.
                        [enc setComputePipelineState:g->ps[@"fill_zero"]];
                        for (int kh = 0; kh < c->nkv; kh++) {
                            const long rows = kh == c->nkv - 1 ? 64 : MIN(64, px->cap - (L + T));
                            const long n = rows * c->hd;
                            if (!n) continue;
                            const NSUInteger off = ((NSUInteger)kh * px->cap + L + T) * c->hd * 4;
                            for (id<MTLBuffer> buffer in @[Kb, Vb]) {
                                [enc setBuffer:buffer offset:off atIndex:0];
                                [enc setBytes:&n length:sizeof(n) atIndex:1];
                                [enc dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                            }
                        }
                    }
                    if (!prof(g, &cb, &enc, P_ATTNPREP, err, errlen)) return false;
                    attn_args aa = { c->nh, c->nkv, c->hd, T, 1.0f / sqrtf((float)c->hd) };
                    // Prefetch improves single-request timing; packed measurements are mixed.
                    // Retain the previously qualified packed dispatch rather than extrapolating
                    // isolated-kernel gains to multi-request throughput.
                    const bool prefetch = T >= 1024 && n_seq == 1;
                    const bool reuse = T >= 4096 && n_seq <= 8;
                    [enc setComputePipelineState:g->ps[px ? @"attention_prefix_64" : g->attn_ref ? @"attention" : prefetch ? @"attention_prefetch_64" : reuse ? @"attention_reuse_4" : @"attention_fa"]];
                    [enc setBytes:&aa length:sizeof(aa) atIndex:0];
                    if (px) {
                        attn_prefix_args ca = { c->nh, c->nkv, c->hd, T, aa.scale, px->cap, L };
                        [enc setBytes:&ca length:sizeof(ca) atIndex:0];
                    }
                    [enc setBuffer:g->Q offset:0 atIndex:1];
                    [enc setBuffer:Kb offset:0 atIndex:2];
                    [enc setBuffer:Vb offset:0 atIndex:3];
                    [enc setBuffer:g->G offset:0 atIndex:4];
                    [enc setBuffer:g->seq_start offset:0 atIndex:5];
                    [enc setBuffer:g->A offset:0 atIndex:6];
                    [enc setBytes:&ac_attn length:sizeof(ac_attn) atIndex:7];
                    [enc setBuffer:g->ovf offset:0 atIndex:8];
                    if (g->attn_ref) {
                        [enc dispatchThreadgroups:MTLSizeMake(T, c->nh, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                    } else if (px || prefetch || reuse) {
                        const int key_block = px || prefetch ? 64 : 32;
                        [enc setThreadgroupMemoryLength:4 * (8 * key_block + 64) * 4 atIndex:0];
                        [enc dispatchThreadgroups:MTLSizeMake((T + 31) / 32, c->nh, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    } else {
                        const int grp = c->nh / c->nkv;
                        [enc setThreadgroupMemoryLength:(NSUInteger)grp * (8 * 32 + 64) * 4 atIndex:0];
                        [enc dispatchThreadgroups:MTLSizeMake((T + 7) / 8, c->nkv, 1) threadsPerThreadgroup:MTLSizeMake(32 * grp, 1, 1)];
                    }
                    if (!prof(g, &cb, &enc, P_ATTN, err, errlen)) return false;
                }

                gemm(g, enc, x_attn, true, g->A, 0, woff(lw->attn_output), H, c->nh * c->hd, g->x, 0, T);
                if (!prof(g, &cb, &enc, P_GEMM, err, errlen)) return false;
            } else {
                if (!prof(g, &cb, &enc, P_NORM, err, errlen)) return false;
                gemm(g, enc, x_norm, false, g->xn, 0, woff(lw->ssm_in), n_ssm, H, g->P, 0, T);
                if (!prof(g, &cb, &enc, P_GEMM, err, errlen)) return false;
                conv_args ca = { C, n_ssm, c->ssm_kernel };
                [enc setComputePipelineState:g->ps[@"ssm_conv"]];
                [enc setBytes:&ca length:sizeof(ca) atIndex:0];
                [enc setBuffer:g->P offset:0 atIndex:1];
                [enc setBuffer:g->weights offset:woff(lw->ssm_conv1d) atIndex:2];
                [enc setBuffer:g->seq_start offset:0 atIndex:3];
                [enc setBuffer:g->Xc offset:0 atIndex:4];
                const NSUInteger tail_off = (NSUInteger)gdn_i * (c->ssm_kernel - 1) * C * 4;
                if (px && L > 0) {
                    // the conv's taps before row 0 are the entry's projection rows L-3..L-1
                    [enc setComputePipelineState:g->ps[@"ssm_conv_tail"]];
                    [enc setBuffer:px->ck_tail[plan->load] offset:tail_off atIndex:3];
                }
                [enc dispatchThreads:MTLSizeMake(C, T, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                for (int i = 0; px && i < plan->n; i++) {
                    // a checkpoint's conv tail is the projection rows just before its row
                    tail_args ta = { C, n_ssm, plan->row[i] - L - (c->ssm_kernel - 1) };
                    [enc setComputePipelineState:g->ps[@"ssm_tail_save"]];
                    [enc setBytes:&ta length:sizeof(ta) atIndex:0];
                    [enc setBuffer:g->P offset:0 atIndex:1];
                    [enc setBuffer:px->ck_tail[plan->slot[i]] offset:tail_off atIndex:2];
                    [enc dispatchThreads:MTLSizeMake(C, c->ssm_kernel - 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                }

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

                if (!prof(g, &cb, &enc, P_CONV, err, errlen)) return false;   // boundary must precede the encoder setup below
                if (px) {
                    encode_gdn_scan_px(g, enc, c, C, px, gdn_i, T, L, plan);
                } else if (gdn_longest) {
                    // The serial encoder finishes each record's scan before reusing scratch.
                    for (int i = 0; i < n_seq; i++) encode_gdn_scan(g, enc, c, C, seq_bounds, i, 1);
                } else encode_gdn_scan(g, enc, c, C, seq_bounds, 0, n_seq);
                gdn_i++;
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

                if (!prof(g, &cb, &enc, P_GDNOUT, err, errlen)) return false;
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
            if (!prof(g, &cb, &enc, P_NORM, err, errlen)) return false;
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
            if (!prof(g, &cb, &enc, P_SWIGLU, err, errlen)) return false;
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
        // with an entry the head's memory rows live in it, the pass's rows after the cached ones
        id<MTLBuffer> kvb[16] = { nil };
        for (int i = 0; i < g->n_kv; i++) kvb[i] = px ? px->kv[i] : g->kv[i];
        const NSUInteger kvo = (NSUInteger)L * 2 * W * 4;
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
            if (g->head_f32) gemm(g, enc, X_F32, false, g->mn32, 0, woff(ip) + (NSUInteger)W * W * 2, 2 * W, W, kvb[i], kvo, T);
            else gemm(g, enc, X_BF16, false, g->mn16, 0, woff(ip) + (NSUInteger)W * W * 2, 2 * W, W, kvb[i], kvo, T);
        }
        for (int i = 0; i < c->head_layers; i++) {
            snprintf(name, sizeof(name), "head.layers.%d.multihead_attn.in_proj_weight", i);
            const gguf_tensor *ip = head_t(e, name);
            if (!ip) return gerr(err, errlen, "head tensors missing");
            if (g->head_f32) gemm(g, enc, X_F32, false, g->mem, 0, woff(ip) + (NSUInteger)W * W * 2, 2 * W, W, kvb[c->routing_layers + i], kvo, T);
            else gemm(g, enc, X_BF16, false, g->mem16, 0, woff(ip) + (NSUInteger)W * W * 2, 2 * W, W, kvb[c->routing_layers + i], kvo, T);
        }
        [enc endEncoding];
        const double t_commit = wall_ms();
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error) { snprintf(err, errlen, "metal: %s", cb.error.localizedDescription.UTF8String); return false; }
        if (g->cb_failed) return gerr(err, errlen, "cannot create a command buffer (mid-pass)");   // see new_cb
        if (g->stage_time) {
            // encode: host time to build the pass; wait: commit to completion; exec: the GPU's own
            // interval for the (last) command buffer. wait - exec is how late the GPU started.
            fprintf(stderr, "clef: stage T=%d n=%d: encode %.2f ms, wait %.2f ms, exec %.2f ms\n", T, n_seq,
                    t_commit - t_enter, wall_ms() - t_commit, (cb.GPUEndTime - cb.GPUStartTime) * 1e3);
        }
        if (dump_layers) memcpy(dump_layers + (size_t)(c->n_layer + 1) * R * H, (float *)g->hfin.contents + (size_t)(T - R) * H, (size_t)R * H * 4);
        if (g->profile) {
            g->prof_ms[P_HEAD] += (cb.GPUEndTime - cb.GPUStartTime) * 1e3;
            fprintf(stderr, "clef: profile T=%d:", T);
            for (int i = 0; i < P_N; i++) { fprintf(stderr, " %s %.1f ms", prof_name[i], g->prof_ms[i]); g->prof_ms[i] = 0; }
            fprintf(stderr, "\n");
        }

        if (overflow) {
            const int32_t *flag = (const int32_t *)g->ovf.contents;
            for (int r = 0; r < n_seq; r++) overflow[r] = flag[seq_bounds[r]] != 0;
        }
        in->nh = (const float *)g->nh32.contents;
        in->nh_skip = L;
        in->n_kv = g->n_kv;
        for (int i = 0; i < g->n_kv; i++) in->kv[i] = (const float *)kvb[i].contents;
    }
    return true;
}

bool clef_gpu_forward(clef_gpu *g, const clef_engine *e, const int32_t *ids, const int32_t *pos,
                      const int32_t *seq_start, const int32_t *seq_bounds, int n_seq, int T, bool bf16_only,
                      bool *overflow, clef_head_inputs *in, float *dump_layers, int dump_rows, char *err, size_t errlen) {
    return forward(g, e, ids, pos, seq_start, seq_bounds, n_seq, T, bf16_only, overflow, in, dump_layers, dump_rows,
                   NULL, 0, NULL, err, errlen);
}

bool clef_gpu_forward_prefix(clef_gpu *g, const clef_engine *e, clef_gpu_prefix *px, const int32_t *ids, int T,
                             int L, const clef_prefix_plan *plan, bool *overflow, clef_head_inputs *in, char *err, size_t errlen) {
    if (!clef_gpu_prefix_supported(g)) return gerr(err, errlen, "the prefix cache needs tiled attention with FP16 output");
    // 32 aligns both attention paths' query tiles and the DeltaNet block: resuming there
    // reproduces the tiles and blocks of a pass over the whole record
    if (L < 0 || L % 32 || L >= T || L > px->cap) return gerr(err, errlen, "bad prefix cache bounds");
    // The plan: a checkpoint to resume from unless the pass starts at row 0, and rows to store that
    // ascend inside (L, T) on 32-token boundaries, in distinct slots other than the one being read.
    // A pass that neither loads nor stores has no use for an entry.
    bool plan_ok = plan->n >= 0 && plan->n <= CLEF_PREFIX_CKPT && (L > 0 || plan->n > 0);
    if (plan_ok && L > 0)
        plan_ok = plan->load >= 0 && plan->load < CLEF_PREFIX_CKPT && px->ck_state[plan->load] && px->ck_tail[plan->load];
    for (int i = 0; plan_ok && i < plan->n; i++) {
        const int row = plan->row[i], slot = plan->slot[i];
        plan_ok = row % 32 == 0 && row > (i ? plan->row[i - 1] : L) && row < T && slot >= 0 && slot < CLEF_PREFIX_CKPT &&
                  (L == 0 || slot != plan->load);
        for (int j = 0; plan_ok && j < i; j++) plan_ok = plan->slot[j] != slot;
    }
    if (!plan_ok) return gerr(err, errlen, "bad prefix cache checkpoints");
    if (!prefix_reserve(g, &e->cfg, px, T, L, err, errlen)) return false;
    for (int i = 0; i < plan->n; i++) if (!ck_reserve(g, &e->cfg, px, plan->slot[i], err, errlen)) return false;
    const int n = T - L;
    int32_t *pos = malloc((size_t)n * 4), *ss = calloc((size_t)n, 4);
    if (!pos || !ss) { free(pos); free(ss); return gerr(err, errlen, "out of memory (prefix pass)"); }
    for (int i = 0; i < n; i++) pos[i] = L + i;
    const int32_t bounds[2] = { 0, n };
    const bool ok = forward(g, e, ids + L, pos, ss, bounds, 1, n, false, overflow, in, NULL, 0, px, L, plan, err, errlen);
    free(pos); free(ss);
    return ok;
}
