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
    // vision tower (encode_vision): per-pass patch uploads sized by the pass's patches, activations
    // by its largest image; feat holds every image's features, img_row maps tokens to them
    id<MTLBuffer> vis_inv_freq;
    int vcap, vcap_total;             // patch capacity of the per-image activations and of the uploads
    id<MTLBuffer> vpatch, vpatch16, vposidx, vposw;
    id<MTLBuffer> vx, vxn, vqkv, vq, vk, vv, va, vff, vff16, vm, vm16, vsplit;
    id<MTLBuffer> img_row, feat;
    int act_f16;                      // FP16 GEMM inputs per producer class (ACT_*), else BF16
    float f16_lim;                    // magnitude past which an FP16 operand counts as overflow
    bool head_f32;                    // head inputs unrounded, head GEMMs on f32 activations
    int scan_lpc;                     // lanes per value column in gdn_scan (CLEF_SCAN_LPC)
    bool attn_ref;                    // CLEF_ATTN_REF=1: the simple reference attention kernel
    bool attn_tu;                     // attention_tu in FP16 passes; false = FP32 kernels everywhere (CLEF_ATTN_TU=0)
    bool vis_mpp;                     // CLEF_VIS_MPP=0 keeps simdgroup vision attention
    bool vis_comp;                    // CLEF_VIS_COMP=0 keeps direct FP32 vision GEMMs
    bool vis_f32;                     // retain FP32 vision operands; non-residual GEMMs may compensate (vis_comp)
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
    id<MTLBuffer> vision;                                // exact merged features, all images in request order
    int cap;                                              // token rows of capacity, a multiple of 1024
    id<MTLBuffer> K[CLEF_MAX_LAYERS], V[CLEF_MAX_LAYERS]; // attention layers: FP32 [nkv][cap][hd], then 64 zero rows
    // Checkpoints: the DeltaNet layers' state [n][Hv][dk][dv] and conv tail [n][ksize-1][C] at one
    // row of the record per slot. A slot's buffers are allocated when a pass first stores into it.
    id<MTLBuffer> ck_state[CLEF_PREFIX_CKPT], ck_tail[CLEF_PREFIX_CKPT];
    id<MTLBuffer> kv[16];                                 // the head's memory K|V rows [cap][2W]
};

enum { P_GEMM, P_ATTN, P_SCAN, P_NORM, P_ATTNPREP, P_CONV, P_GDNOUT, P_SWIGLU, P_HEAD, P_VISION, P_N };
// GEMM input classes, by the kernel that produces them; ACT_VIS covers every vision-tower operand
enum { ACT_NORM = 1, ACT_ATTN = 2, ACT_GDN = 4, ACT_MLP = 8, ACT_VIS = 16 };
enum { X_BF16, X_F16, X_F32 };
// norm includes the embedding; conv_prep is ssm_conv + gdn_prep; head is the final norm and the GPU half of the head;
// vision is the whole tower, every image
static const char *prof_name[P_N] = { "gemm", "attention", "gdn_scan", "norm", "attn_prep", "conv_prep", "gdn_out", "swiglu", "head", "vision" };

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
    // test hook (tests/test_cli_errors.py): CLEF_DEBUG_SIMD_WIDTH_FOR=NAME reports that pipeline as 16 lanes wide
    const char *fake = getenv("CLEF_DEBUG_SIMD_WIDTH_FOR");
    const NSUInteger width = fake && !strcmp(fake, name) ? 16 : p.threadExecutionWidth;
    // attention_prefix_64 shares that rescaling (review #38). Vision MPP softmax partitions
    // each 32-lane SIMDgroup into four-thread rows, so it requires the same width.
    if ((!strcmp(name, "attention_reuse_4") || !strcmp(name, "attention_prefetch_64") || !strcmp(name, "attention_prefix_64") ||
         !strcmp(name, "vis_attention_mpp")) &&
        width != 32) {
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
                                "gemm_f32_32x128", "gemm_f32_acc_32x128", "gemm_f32_16x128", "gemm_f32_acc_16x128", "embed", "rmsnorm_act", "rmsnorm_f32", "layernorm",
                                "f32_to_bf16", "fill_zero", "keepalive", "attn_prep", "attention", "attention_fa", "attention_reuse_4", "attention_prefetch_64",
                                "attn_prep_prefix", "attention_prefix_64", "attn_prep_split", "attention_tu", "ssm_conv", "ssm_conv_tail", "ssm_tail_save", "gdn_prep", "gdn_scan_2", "gdn_scan_4", "gdn_scan_8",
                                "gdn_scan_st_2", "gdn_scan_st_4", "gdn_scan_st_8",
                                "gdn_chunk_prep_32", "gdn_chunk_scan_32_16", "gdn_chunk_scan_32_32",
                                "gdn_chunk_scan_st_32_16", "gdn_chunk_scan_st_32_32", "gdn_out", "swiglu",
                                "vis_split_gemm", "vis_gemm_comp_32x128", "vis_gemm_comp_64x128",
                                "vis_act", "vis_embed", "layernorm_act", "layernorm_bias_act", "vis_qkv_rope", "vis_attention", "vis_attention_mma", "vis_attention_mpp", "add_bias", "vis_bias_gelu" };
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
        if (c->has_vision) {
            // Qwen3_5VisionRotaryEmbedding(hd / 2): 1 / 10000 ** (arange(0, hd/2, 2) / (hd/2)), f32
            float vinv[64];
            const int dim = c->v_hd / 2;
            for (int j = 0; j < dim / 2; j++) vinv[j] = 1.0f / powf(10000.0f, (float)(2 * j) / (float)dim);
            g->vis_inv_freq = [g->dev newBufferWithBytes:vinv length:sizeof(vinv) options:MTLResourceStorageModeShared];
            if (!g->vis_inv_freq) { gerr(err, errlen, "cannot allocate vision rope table"); free(g); return NULL; }
        }
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
        // The vision tower's 27 layers run before any text is read, so operand rounding there
        // compounds: FP16 operands put the image features 2.9e-3 (relative L2) from the FP32
        // oracle and two of 41 vision questions past the parity limits; f32 operands put them at
        // about 1e-5 and every question within, for ~15 ms more on a 336-pixel image (docs/vision.md).
        const char *vf = getenv("CLEF_VIS_F32");
        g->vis_f32 = !vf || strcmp(vf, "0") != 0;
        // Compensate non-residual projections only; residual GEMMs retain FP32 operands.
        // BF16 retries and ACT_VIS=0 also retain direct FP32 GEMMs throughout the tower.
        const char *vc = getenv("CLEF_VIS_COMP");
        g->vis_comp = !vc || strcmp(vc, "0") != 0;
        const char *vm = getenv("CLEF_VIS_MPP");
        g->vis_mpp = !vm || strcmp(vm, "0") != 0;
        g->scan_lpc = getenv("CLEF_SCAN_LPC") ? atoi(getenv("CLEF_SCAN_LPC")) : 8;
        if (g->scan_lpc != 2 && g->scan_lpc != 4 && g->scan_lpc != 8) g->scan_lpc = 8;
        // Precision (README "Accuracy"): the target is the FP32 reference. GEMM activations are
        // FP16 for every producer class by default (7x closer to FP32 than BF16 on the 27B, same
        // GEMM rate); CLEF_ACT_F16=<mask of ACT_*> narrows that, 0 = BF16 everywhere as in the
        // HF BF16 path. The head takes the f32 hidden state and runs its GEMMs on f32 activations
        // (tiny GEMMs); CLEF_HEAD_BF16=1 restores the BF16 rounding the HF BF16 path applies there.
        g->act_f16 = getenv("CLEF_ACT_F16") ? atoi(getenv("CLEF_ACT_F16")) & 31 : ACT_NORM | ACT_ATTN | ACT_GDN | ACT_MLP | ACT_VIS;
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
        g->vis_inv_freq = g->vpatch = g->vpatch16 = g->vposidx = g->vposw = g->vx = g->vxn = g->vqkv = nil;
        g->vq = g->vk = g->vv = g->va = g->vff = g->vff16 = g->vm = g->vm16 = g->img_row = g->feat = g->vsplit = nil;
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
        id<MTLBuffer> act[] = { g->ids, g->pos, g->seq_start, g->seq_bounds, g->x, g->xn, g->P, g->Q, g->K, g->V, g->G,
                                g->A, g->Xc, g->beta, g->gate, g->O, g->hfin, g->nh32, g->nh16, g->mem, g->mem16,
                                g->mn16, g->mn32, g->ovf, g->attn_blk, g->gdn_w, g->gdn_u, g->gdn_ke, g->gdn_a, g->gdn_e,
                                g->img_row, g->feat, g->vpatch, g->vpatch16, g->vposidx, g->vposw, g->vx, g->vxn, g->vqkv,
                                g->vq, g->vk, g->vv, g->va, g->vff, g->vff16, g->vm, g->vm16, g->vsplit };
        // Include every possible activation and KV buffer, plus the weights. Adding vision
        // scratch must not overflow this list when all 16 head KV slots are populated.
        id<MTLBuffer> bufs[1 + sizeof(act) / sizeof(act[0]) + sizeof(g->kv) / sizeof(g->kv[0])];
        int n = 0;
        if (!getenv("CLEF_DEBUG_KEEPWARM_NOWEIGHTS")) bufs[n++] = g->weights;
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
    id<MTLBuffer> img_row;
    id<MTLBuffer> kv[16] = { nil };
    @autoreleasepool {
        ids = buf(g, Tc * 4); pos = buf(g, Tc * 12); seq_start = buf(g, Tc * 4); seq_bounds = buf(g, (Tc + 1) * 4); ovf = buf(g, Tc * 4);
        // image features live in the vision scratch, sized by the pass's image rows: a
        // [capacity][H] FP32 buffer here cost every text-only workload 256 MiB on Flash at
        // 16,384 tokens (review #3); embed binds img_row in its place when there are no images
        img_row = buf(g, Tc * 4);
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
    id<MTLBuffer> all[] = { ids, pos, seq_start, seq_bounds, x, xn, P, Q, K, V, G, A, Xc, beta, gate, O, hfin, nh32, nh16, mem, mem16, mn16, mn32, ovf, attn_blk, img_row };
    for (size_t i = 0; i < sizeof(all) / sizeof(all[0]); i++) {
        if (!all[i]) return gerr(err, errlen, "cannot allocate activation buffers (previous buffers kept)");
    }
    for (int i = 0; i < g->n_kv; i++) if (!kv[i]) return gerr(err, errlen, "cannot allocate activation buffers (previous buffers kept)");
    memset(Q.contents, 0, Q.length); memset(K.contents, 0, K.length); memset(V.contents, 0, V.length);
    g->ids = ids; g->pos = pos; g->seq_start = seq_start; g->seq_bounds = seq_bounds; g->x = x; g->xn = xn; g->P = P;
    g->Q = Q; g->K = K; g->V = V; g->G = G; g->A = A; g->Xc = Xc; g->beta = beta; g->gate = gate; g->O = O;
    g->hfin = hfin; g->nh32 = nh32; g->nh16 = nh16; g->mem = mem; g->mem16 = mem16; g->mn16 = mn16; g->mn32 = mn32; g->ovf = ovf;
    g->attn_blk = attn_blk;
    g->img_row = img_row;
    for (int i = 0; i < g->n_kv; i++) g->kv[i] = kv[i];
    g->cap = cap;
    return true;
}

// Vision scratch: the pass's patch uploads (every image, each [P][in] f32 plus the 16-bit operand
// and the position interpolation), the merged features of every image (one row per merge window,
// total patches / merge^2) and one image's activations at the largest patch count. Grown by
// doubling and committed only when every allocation succeeded, like ensure_capacity.
static bool ensure_vision_capacity(clef_gpu *g, const clef_config *c, int max_patches, int total_patches, char *err, size_t errlen) {
    if (max_patches <= g->vcap && total_patches <= g->vcap_total) return true;
    int cap = g->vcap ? g->vcap : 1024, total = g->vcap_total ? g->vcap_total : 1024;
    while (cap < max_patches) cap *= 2;
    while (total < total_patches) total *= 2;
    const size_t Pc = (size_t)cap, Tp = (size_t)total, E = (size_t)c->v_E, F = (size_t)c->v_ff, In = (size_t)c->v_in;
    const size_t M = E * (size_t)c->v_merge * c->v_merge;   // merger width, four patches per row
    const size_t ob = g->vis_f32 ? 4 : 2;                     // bytes per GEMM operand element
    id<MTLBuffer> vpatch, vpatch16, vposidx, vposw, vx, vxn, vqkv, vq, vk, vv, va, vff, vff16, vm, vm16, vsplit, feat;
    @autoreleasepool {
        feat = buf(g, Tp / ((size_t)c->v_merge * c->v_merge) * c->H * 4);
        vpatch = buf(g, Tp * In * 4); vpatch16 = buf(g, g->vis_f32 ? 0 : Tp * In * 2); vposidx = buf(g, Tp * 16); vposw = buf(g, Tp * 16);
        vx = buf(g, Pc * E * 4); vxn = buf(g, Pc * E * ob); vqkv = buf(g, Pc * 3 * E * 4);
        vq = buf(g, Pc * E * 4 + 128 * 72 * 4); vk = buf(g, Pc * E * 4 + 128 * 72 * 4); vv = buf(g, Pc * E * 4 + 128 * 72 * 4); va = buf(g, Pc * E * ob);
        vff = buf(g, Pc * F * 4); vff16 = buf(g, Pc * F * ob);
        vm = buf(g, Pc / 4 * M * 4 + 16); vm16 = buf(g, Pc / 4 * M * ob + 16);
        // Only patch embedding needs a separate split. Norm/GELU producers write the
        // same high/residual planes directly into their existing four-byte buffers.
        vsplit = buf(g, g->vis_f32 && g->vis_comp ? Pc * In * 4 : 0);
    }
    id<MTLBuffer> all[] = { vpatch, vpatch16, vposidx, vposw, vx, vxn, vqkv, vq, vk, vv, va, vff, vff16, vm, vm16, vsplit, feat };
    for (size_t i = 0; i < sizeof(all) / sizeof(all[0]); i++) {
        if (!all[i]) return gerr(err, errlen, "cannot allocate vision buffers (previous buffers kept)");
    }
    g->feat = feat;
    g->vpatch = vpatch; g->vpatch16 = vpatch16; g->vposidx = vposidx; g->vposw = vposw;
    g->vx = vx; g->vxn = vxn; g->vqkv = vqkv; g->vq = vq; g->vk = vk; g->vv = vv; g->va = va;
    g->vff = vff; g->vff16 = vff16; g->vm = vm; g->vm16 = vm16; g->vsplit = vsplit;
    g->vcap = cap;
    g->vcap_total = total;
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
typedef struct { int P, E, heads, hd, grid_w, merge; float eps, scale; } vis_args;
typedef struct { int H; float eps; int rows; } ln_act_args;
typedef struct { int N, mode, rows; } bias_act_args;

static void vision_gemm(clef_gpu *g, id<MTLComputeCommandEncoder> enc, int xt, bool acc,
                        id<MTLBuffer> X, NSUInteger xoff, NSUInteger w_off, int N, int K,
                        id<MTLBuffer> Y, NSUInteger yoff, int T, const act_args *ac, NSUInteger ovf_off, bool presplit) {
    if (!g->vis_comp || acc || xt != X_F32 || !ac->f16) {
        if (xt != X_F32) {
            gemm(g, enc, xt, acc, X, xoff, w_off, N, K, Y, yoff, T);
            return;
        }
        // The 16-row FP32 tile preserves reductions and saves work at short/ragged shapes.
        const gemm_args a = { T, N, K };
        [enc setComputePipelineState:g->ps[acc ? @"gemm_f32_acc_16x128" : @"gemm_f32_16x128"]];
        [enc setBytes:&a length:sizeof(a) atIndex:0];
        [enc setBuffer:X offset:xoff atIndex:1];
        [enc setBuffer:g->weights offset:w_off atIndex:2];
        [enc setBuffer:Y offset:yoff atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake((N + 127) / 128, (T + 15) / 16, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        return;
    }
    if (!presplit) {
        const long n = (long)T * K;
        [enc setComputePipelineState:g->ps[@"vis_split_gemm"]];
        [enc setBuffer:X offset:xoff atIndex:0];
        [enc setBuffer:g->vsplit offset:0 atIndex:1];
        [enc setBytes:&n length:sizeof(n) atIndex:2];
        [enc setBytes:ac length:sizeof(*ac) atIndex:3];
        [enc setBuffer:g->ovf offset:ovf_off atIndex:4];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        X = g->vsplit;
        xoff = 0;
    }
    // Selection uses this image's rows, never the packed batch's length.
    const int tm = T >= 1024 ? 64 : 32;
    NSString *name = tm == 64 ? @"vis_gemm_comp_64x128" : @"vis_gemm_comp_32x128";
    const gemm_args a = { T, N, K };
    [enc setComputePipelineState:g->ps[name]];
    [enc setBytes:&a length:sizeof(a) atIndex:0];
    [enc setBuffer:X offset:xoff atIndex:1];
    [enc setBuffer:g->weights offset:w_off atIndex:2];
    [enc setBuffer:Y offset:yoff atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake((N + 127) / 128, (T + tm - 1) / tm, 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
}

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
        px->vision = nil;
        for (int i = 0; i < CLEF_MAX_LAYERS; i++) px->K[i] = px->V[i] = nil;
        for (int i = 0; i < 16; i++) px->kv[i] = nil;
        for (int i = 0; i < CLEF_PREFIX_CKPT; i++) px->ck_state[i] = px->ck_tail[i] = nil;
    }
    free(px);
}

size_t clef_gpu_prefix_bytes(const clef_gpu_prefix *px) {
    size_t n = px->vision.length;
    for (int i = 0; i < CLEF_PREFIX_CKPT; i++) n += px->ck_state[i].length + px->ck_tail[i].length;
    for (int i = 0; i < CLEF_MAX_LAYERS; i++) n += px->K[i].length + px->V[i].length;
    for (int i = 0; i < 16; i++) n += px->kv[i].length;
    return n;
}

/* What the entry would hold after a pass over `rows` tokens that stores `new_ckpts` checkpoints:
 * the K/V and memory planes at the capacity prefix_reserve would choose, the checkpoint slots
 * already allocated, and one state plus tail per checkpoint that needs a new slot (ck_reserve
 * returns early for an allocated one; the caller counts those with
 * clef_gpu_prefix_slot_allocated, review #44). The server compares it
 * with its budget before the pass allocates anything (review #27). Capacity growth copies the
 * planes, so a growing entry transiently needs its old planes as well. */
bool clef_gpu_prefix_slot_allocated(const clef_gpu_prefix *px, int slot) {
    return slot >= 0 && slot < CLEF_PREFIX_CKPT && px->ck_state[slot] && px->ck_tail[slot];
}

size_t clef_gpu_prefix_estimate(const clef_gpu *g, const clef_config *c, const clef_gpu_prefix *px, int rows, int new_ckpts, int image_rows) {
    size_t cap = (size_t)px->cap;
    if ((size_t)rows > cap) {
        size_t want = (size_t)rows;
        if (want < cap + cap / 4) want = cap + cap / 4;
        cap = (want + 1023) / 1024 * 1024;
    }
    const size_t kvb = ((size_t)c->nkv * cap + 64) * c->hd * 4, memb = cap * 2 * c->W * 4;
    int n_attn = 0, n_gdn = 0;
    for (int l = 0; l < c->n_layer; l++) { if (c->layer_full[l]) n_attn++; else n_gdn++; }
    const size_t C = 2 * (size_t)c->Hk * c->dk + (size_t)c->Hv * c->dv;
    const size_t ckpt = (size_t)n_gdn * c->Hv * c->dk * c->dv * 4 + (size_t)n_gdn * (c->ssm_kernel - 1) * C * 4;
    size_t n = (size_t)n_attn * 2 * kvb + (size_t)g->n_kv * memb;
    for (int i = 0; i < CLEF_PREFIX_CKPT; i++) n += px->ck_state[i].length + px->ck_tail[i].length;
    return n + (size_t)new_ckpts * ckpt + (size_t)image_rows * c->H * sizeof(float);
}

// Allocate before replacing; a failed growth preserves the old allocation for cleanup.
static bool prefix_vision_reserve(clef_gpu *g, clef_gpu_prefix *px, size_t bytes, char *err, size_t errlen) {
    if (px->vision.length == bytes) return true;
    id<MTLBuffer> next = bytes ? buf(g, bytes) : nil;
    if (bytes && !next) return gerr(err, errlen, "cannot allocate prefix image features");
    px->vision = next;
    return true;
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
        for (int i = 0; i <= 2 * CLEF_MAX_LAYERS + 16 + 2 * CLEF_PREFIX_CKPT; i++) {
            const int k = i - 2 * CLEF_MAX_LAYERS - 16;   // the checkpoint buffers follow the K, V and memory planes
            id<MTLBuffer> b = i < CLEF_MAX_LAYERS ? px->K[i] : i < 2 * CLEF_MAX_LAYERS ? px->V[i - CLEF_MAX_LAYERS]
                            : k < 0 ? px->kv[i - 2 * CLEF_MAX_LAYERS]
                            : k < CLEF_PREFIX_CKPT ? px->ck_state[k] : k < 2 * CLEF_PREFIX_CKPT ? px->ck_tail[k - CLEF_PREFIX_CKPT] : px->vision;
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

// One image through the vision tower (metal/clef.metal "vision tower"), its features into the
// feat rows [img->feat_row, + tokens). patch_off is the image's first row in the pass's patch
// uploads. The tower's GEMM operands are the ACT_VIS class: f32 by default (CLEF_VIS_F32=1), else x_vis/ac select FP16 or BF16, and an
// overflow flags the image's record (ovf_row) like any backbone operand.
static bool encode_vision(clef_gpu *g, const clef_engine *e, id<MTLCommandBuffer> __strong *cb, id<MTLComputeCommandEncoder> __strong *encp,
                          const clef_gpu_image *img, int patch_off, int x_vis, const act_args *ac, char *err, size_t errlen) {
    id<MTLComputeCommandEncoder> enc = *encp;
    const clef_config *c = &e->cfg;
    const clef_weights *w = &e->w;
    const clef_image_patches *pt = img->pt;
    const int P = pt->n_patch, E = c->v_E, F = c->v_ff, In = c->v_in, H = c->H;
    // CLEF_PROFILE: the tower's GEMMs and attention in their own categories, the rest as "vision"
#define VPROF(cat) do { if (!prof(g, cb, encp, (cat), err, errlen)) return false; enc = *encp; } while (0)
    const int window = c->v_merge * c->v_merge, M = E * window, Pm = P / window;
    const NSUInteger ovf_off = (NSUInteger)img->ovf_row * 4;
    const vis_args va = { P, E, c->v_heads, c->v_hd, pt->grid_w, c->v_merge, c->v_eps, 1.0f / sqrtf((float)c->v_hd) };
    const ln_act_args la = { E, c->v_eps, P };
    // CLEF_VIS_F32: the operand producers write f32 into the same buffers (allocated at f32 size)
    // and the GEMMs take float; the patches are then read as uploaded.
    const int f32out = g->vis_f32;
    // Mode 2 stores compensated halves with the same rounding as FP32 output followed
    // by vis_split_gemm. Residual projections still consume mode 1 FP32 operands.
    const int compout = g->vis_comp && x_vis == X_F32 && ac->f16 ? 2 : f32out;
    // Small images favor the simdgroup kernel; selection depends only on this image.
    const bool mpp = g->vis_mpp && !g->attn_ref && P >= 2048;
    if (x_vis == X_F32) {
        VPROF(P_VISION);
        vision_gemm(g, enc, x_vis, false, g->vpatch, (NSUInteger)patch_off * In * 4, woff(w->v_patch_w), E, In, g->vx, 0, P, ac, ovf_off, false);
        VPROF(P_GEMM);
    } else {
        const long n = (long)P * In;
        [enc setComputePipelineState:g->ps[@"vis_act"]];
        [enc setBuffer:g->vpatch offset:(NSUInteger)patch_off * In * 4 atIndex:0];
        [enc setBuffer:g->vpatch16 offset:(NSUInteger)patch_off * In * 2 atIndex:1];
        [enc setBytes:&n length:sizeof(n) atIndex:2];
        [enc setBytes:ac length:sizeof(*ac) atIndex:3];
        [enc setBuffer:g->ovf offset:ovf_off atIndex:4];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        // patch embedding: Conv3d as one GEMM over the flattened patch, then bias and position
        VPROF(P_VISION);
        vision_gemm(g, enc, x_vis, false, g->vpatch16, (NSUInteger)patch_off * In * 2, woff(w->v_patch_w), E, In, g->vx, 0, P, ac, ovf_off, false);
        VPROF(P_GEMM);
    }
    [enc setComputePipelineState:g->ps[@"vis_embed"]];
    [enc setBytes:&va length:sizeof(va) atIndex:0];
    [enc setBuffer:g->vx offset:0 atIndex:1];
    [enc setBuffer:g->weights offset:woff(w->v_patch_b) atIndex:2];
    [enc setBuffer:g->weights offset:woff(w->v_pos) atIndex:3];
    [enc setBuffer:g->vposidx offset:(NSUInteger)patch_off * 16 atIndex:4];
    [enc setBuffer:g->vposw offset:(NSUInteger)patch_off * 16 atIndex:5];
    [enc dispatchThreads:MTLSizeMake(E, P, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    for (int l = 0; l < c->v_layers; l++) {
        const clef_vlayer_w *lw = &w->vlayer[l];
        // Each residual bias is applied by the next norm, which writes the rounded sum
        // back to vx before it is reused. This removes two standalone passes per block.
        // x + attn(norm1(x))
        [enc setComputePipelineState:g->ps[l ? @"layernorm_bias_act" : @"layernorm_act"]];
        [enc setBytes:&la length:sizeof(la) atIndex:0];
        [enc setBuffer:g->vx offset:0 atIndex:1];
        [enc setBuffer:g->weights offset:woff(lw->ln1_w) atIndex:2];
        [enc setBuffer:g->weights offset:woff(lw->ln1_b) atIndex:3];
        [enc setBuffer:g->vxn offset:0 atIndex:4];
        [enc setBytes:ac length:sizeof(*ac) atIndex:5];
        [enc setBuffer:g->ovf offset:ovf_off atIndex:6];
        [enc setBuffer:g->vxn offset:0 atIndex:7];
        [enc setBytes:&compout length:sizeof(compout) atIndex:8];
        if (l) [enc setBuffer:g->weights offset:woff(w->vlayer[l - 1].down_b) atIndex:9];
        rows_kernel(g, enc, l ? @"layernorm_bias_act" : @"layernorm_act", P, E);
        VPROF(P_VISION);
        vision_gemm(g, enc, x_vis, false, g->vxn, 0, woff(lw->qkv_w), 3 * E, E, g->vqkv, 0, P, ac, ovf_off, true);
        VPROF(P_GEMM);
        [enc setComputePipelineState:g->ps[@"vis_qkv_rope"]];
        [enc setBytes:&va length:sizeof(va) atIndex:0];
        [enc setBuffer:g->vqkv offset:0 atIndex:1];
        [enc setBuffer:g->weights offset:woff(lw->qkv_b) atIndex:2];
        [enc setBuffer:g->vis_inv_freq offset:0 atIndex:3];
        [enc setBuffer:g->vq offset:0 atIndex:4];
        [enc setBuffer:g->vk offset:0 atIndex:5];
        [enc setBuffer:g->vv offset:0 atIndex:6];
        const int tail_rows = mpp ? 128 : 32;
        [enc setBytes:&tail_rows length:sizeof(tail_rows) atIndex:7];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)c->v_hd / 2, (NSUInteger)c->v_heads, (NSUInteger)P)
             threadsPerThreadgroup:MTLSizeMake((NSUInteger)c->v_hd / 2, 1, 1)];
        VPROF(P_VISION);   // a boundary must precede the pipeline and buffer setup of the next dispatch
        [enc setComputePipelineState:g->ps[g->attn_ref ? @"vis_attention" : mpp ? @"vis_attention_mpp" : @"vis_attention_mma"]];
        [enc setBytes:&va length:sizeof(va) atIndex:0];
        [enc setBuffer:g->vq offset:0 atIndex:1];
        [enc setBuffer:g->vk offset:0 atIndex:2];
        [enc setBuffer:g->vv offset:0 atIndex:3];
        [enc setBuffer:g->va offset:0 atIndex:4];
        [enc setBytes:ac length:sizeof(*ac) atIndex:5];
        [enc setBuffer:g->ovf offset:ovf_off atIndex:6];
        [enc setBuffer:g->va offset:0 atIndex:7];
        [enc setBytes:&f32out length:sizeof(f32out) atIndex:8];
        if (g->attn_ref) [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)(P + 7) / 8, (NSUInteger)c->v_heads, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        else [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)(P + 31) / 32, (NSUInteger)c->v_heads, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        VPROF(P_ATTN);
        vision_gemm(g, enc, x_vis, true, g->va, 0, woff(lw->out_w), E, E, g->vx, 0, P, ac, ovf_off, false);
        VPROF(P_GEMM);
        // x + mlp(norm2(x))
        [enc setComputePipelineState:g->ps[@"layernorm_bias_act"]];
        [enc setBytes:&la length:sizeof(la) atIndex:0];
        [enc setBuffer:g->vx offset:0 atIndex:1];
        [enc setBuffer:g->weights offset:woff(lw->ln2_w) atIndex:2];
        [enc setBuffer:g->weights offset:woff(lw->ln2_b) atIndex:3];
        [enc setBuffer:g->vxn offset:0 atIndex:4];
        [enc setBytes:ac length:sizeof(*ac) atIndex:5];
        [enc setBuffer:g->ovf offset:ovf_off atIndex:6];
        [enc setBuffer:g->vxn offset:0 atIndex:7];
        [enc setBytes:&compout length:sizeof(compout) atIndex:8];
        [enc setBuffer:g->weights offset:woff(lw->out_b) atIndex:9];
        rows_kernel(g, enc, @"layernorm_bias_act", P, E);
        VPROF(P_VISION);
        vision_gemm(g, enc, x_vis, false, g->vxn, 0, woff(lw->up_w), F, E, g->vff, 0, P, ac, ovf_off, true);
        VPROF(P_GEMM);
        const bias_act_args ga = { F, 0, P };
        [enc setComputePipelineState:g->ps[@"vis_bias_gelu"]];
        [enc setBytes:&ga length:sizeof(ga) atIndex:0];
        [enc setBuffer:g->vff offset:0 atIndex:1];
        [enc setBuffer:g->weights offset:woff(lw->up_b) atIndex:2];
        [enc setBuffer:g->vff16 offset:0 atIndex:3];
        [enc setBytes:ac length:sizeof(*ac) atIndex:4];
        [enc setBuffer:g->ovf offset:ovf_off atIndex:5];
        [enc setBuffer:g->vff16 offset:0 atIndex:6];
        [enc setBytes:&f32out length:sizeof(f32out) atIndex:7];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)F, (NSUInteger)P, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        VPROF(P_VISION);
        vision_gemm(g, enc, x_vis, true, g->vff16, 0, woff(lw->down_w), E, F, g->vx, 0, P, ac, ovf_off, false);
        VPROF(P_GEMM);
    }
    // merger: LayerNorm per patch, then each merge window's four rows as one row of 4E
    [enc setComputePipelineState:g->ps[@"layernorm_bias_act"]];
    [enc setBytes:&la length:sizeof(la) atIndex:0];
    [enc setBuffer:g->vx offset:0 atIndex:1];
    [enc setBuffer:g->weights offset:woff(w->v_post_ln_w) atIndex:2];
    [enc setBuffer:g->weights offset:woff(w->v_post_ln_b) atIndex:3];
    [enc setBuffer:g->vxn offset:0 atIndex:4];
    [enc setBytes:ac length:sizeof(*ac) atIndex:5];
    [enc setBuffer:g->ovf offset:ovf_off atIndex:6];
    [enc setBuffer:g->vxn offset:0 atIndex:7];
    [enc setBytes:&compout length:sizeof(compout) atIndex:8];
    [enc setBuffer:g->weights offset:woff(w->vlayer[c->v_layers - 1].down_b) atIndex:9];
    rows_kernel(g, enc, @"layernorm_bias_act", P, E);
    VPROF(P_VISION);
    vision_gemm(g, enc, x_vis, false, g->vxn, 0, woff(w->v_mm0_w), M, M, g->vm, 0, Pm, ac, ovf_off, true);
    VPROF(P_GEMM);
    const bias_act_args ma = { M, 1, Pm };
    [enc setComputePipelineState:g->ps[@"vis_bias_gelu"]];
    [enc setBytes:&ma length:sizeof(ma) atIndex:0];
    [enc setBuffer:g->vm offset:0 atIndex:1];
    [enc setBuffer:g->weights offset:woff(w->v_mm0_b) atIndex:2];
    [enc setBuffer:g->vm16 offset:0 atIndex:3];
    [enc setBytes:ac length:sizeof(*ac) atIndex:4];
    [enc setBuffer:g->ovf offset:ovf_off atIndex:5];
    [enc setBuffer:g->vm16 offset:0 atIndex:6];
    [enc setBytes:&compout length:sizeof(compout) atIndex:7];
    [enc dispatchThreads:MTLSizeMake((NSUInteger)M, (NSUInteger)Pm, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    const NSUInteger feat_off = (NSUInteger)img->feat_row * H * 4;
    VPROF(P_VISION);
    vision_gemm(g, enc, x_vis, false, g->vm16, 0, woff(w->v_mm2_w), H, M, g->feat, feat_off, Pm, ac, ovf_off, true);
    VPROF(P_GEMM);
    [enc setComputePipelineState:g->ps[@"add_bias"]];
    [enc setBytes:&H length:sizeof(H) atIndex:0];
    [enc setBuffer:g->feat offset:feat_off atIndex:1];
    [enc setBuffer:g->weights offset:woff(w->v_mm2_b) atIndex:2];
    [enc dispatchThreads:MTLSizeMake((NSUInteger)H, (NSUInteger)Pm, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    VPROF(P_VISION);
#undef VPROF
    return true;
}

static const gguf_tensor *head_t(const clef_engine *e, const char *name) { return gguf_find_tensor(&e->gguf, name); }

// px (with L and plan, else NULL, 0, NULL): the pass is one record whose first L tokens are in
// the prefix-cache entry. T, ids and pos cover its remaining tokens, pos continuing from L. The
// pass writes its own rows into the entry, resumes the DeltaNet layers from the checkpoint
// plan->load and stores their state and conv tail at each of the plan's rows. Everything it
// computes is what a pass over the whole record computes for these rows.
static bool forward(clef_gpu *g, const clef_engine *e, const int32_t *ids, const int32_t *pos,
                    const int32_t *seq_start, const int32_t *seq_bounds, int n_seq, int T, bool bf16_only,
                    bool *overflow, clef_head_inputs *in, float *dump_layers, int dump_rows, const clef_gpu_images *imgs,
                    clef_gpu_prefix *px, int L, const clef_prefix_plan *plan, char *err, size_t errlen) {
    const int R = dump_rows > 0 && dump_rows < T ? dump_rows : T;   /* dump the last R token rows */
    const clef_config *c = &e->cfg;
    const double t_enter = wall_ms();
    // FP16 operands are only safe with the overflow flags read back (and acted on) by the caller
    if (!bf16_only && g->act_f16 && !overflow) return gerr(err, errlen, "FP16 activations need the overflow flags (pass overflow or bf16_only)");
    if (!ensure_capacity(g, c, T, n_seq, err, errlen)) return false;
    const int n_img = imgs ? imgs->n : 0;
    if (n_img && !c->has_vision) return gerr(err, errlen, "images with a model file without the vision tower");
    const bool cache_images = px && n_img && imgs->cache;
    const bool reuse_images = cache_images && imgs->reuse;
    if (imgs && imgs->reuse && !cache_images) return gerr(err, errlen, "image reuse requires a keyed prefix entry");
    int max_patches = 0, total_patches = 0, image_rows = 0;
    for (int i = 0; i < n_img; i++) {
        const clef_image_patches *pt = imgs->img[i].pt;
        if (pt->n_patch <= 0 || pt->patch_dim != c->v_in || pt->grid_h % c->v_merge || pt->grid_w % c->v_merge ||
            pt->n_patch > INT_MAX - total_patches || pt->n_tokens <= 0 ||
            imgs->img[i].feat_row != image_rows || pt->n_tokens > (reuse_images ? T + L : T) - image_rows)
            return gerr(err, errlen, "invalid image patches");
        if (pt->n_patch > max_patches) max_patches = pt->n_patch;
        total_patches += pt->n_patch;
        image_rows += pt->n_tokens;
    }
    if (n_img) for (int t = 0; t < T; t++)
        if (imgs->img_row[t] < -1 || imgs->img_row[t] >= image_rows) return gerr(err, errlen, "invalid image feature row");
    const size_t image_bytes = (size_t)image_rows * c->H * sizeof(float);
    if (reuse_images && px->vision.length != image_bytes) return gerr(err, errlen, "cached image feature size mismatch");
    if (px && !reuse_images && !prefix_vision_reserve(g, px, cache_images ? image_bytes : 0, err, errlen)) return false;
    if (n_img && !reuse_images && !ensure_vision_capacity(g, c, max_patches, total_patches, err, errlen)) return false;
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
            if (g->vcap) {
                id<MTLBuffer> vis[] = { g->vpatch, g->vpatch16, g->vposidx, g->vposw, g->vx, g->vxn, g->vqkv, g->vq, g->vk, g->vv,
                                        g->va, g->vff, g->vff16, g->vm, g->vm16, g->vsplit, g->feat };
                for (size_t i = 0; i < sizeof(vis) / sizeof(vis[0]); i++) memset_pattern4(vis[i].contents, &nan, vis[i].length);
            }
        }
        memcpy(g->ids.contents, ids, (size_t)T * 4);
        memcpy(g->pos.contents, pos, (size_t)T * 12);
        memcpy(g->seq_start.contents, seq_start, (size_t)T * 4);
        memcpy(g->seq_bounds.contents, seq_bounds, (size_t)(n_seq + 1) * 4);
        memset(g->ovf.contents, 0, (size_t)T * 4);
        if (n_img) memcpy(g->img_row.contents, imgs->img_row, (size_t)T * 4);
        else memset(g->img_row.contents, 0xff, (size_t)T * 4);   // -1: every token is text
        // the images' patches and position interpolation, each image at its own rows
        for (int i = 0, off = 0; !reuse_images && i < n_img; i++) {
            const clef_image_patches *pt = imgs->img[i].pt;
            memcpy((float *)g->vpatch.contents + (size_t)off * c->v_in, pt->patches, (size_t)pt->n_patch * c->v_in * 4);
            clef_image_pos_interp(pt->grid_h, pt->grid_w, c->v_pos_side, c->v_merge,
                                  (int32_t *)g->vposidx.contents + (size_t)off * 4, (float *)g->vposw.contents + (size_t)off * 4);
            off += pt->n_patch;
        }

        const int H = c->H, W = c->W;
        const int C = 2 * c->Hk * c->dk + c->Hv * c->dv;
        const int f16 = bf16_only ? 0 : g->act_f16;
        const int f_norm = !!(f16 & ACT_NORM), f_attn = !!(f16 & ACT_ATTN);
        const int f_gdn = !!(f16 & ACT_GDN), f_mlp = !!(f16 & ACT_MLP);
        const act_args ac_norm = { f_norm, g->f16_lim }, ac_attn = { f_attn, g->f16_lim };
        const act_args ac_gdn = { f_gdn, g->f16_lim }, ac_mlp = { f_mlp, g->f16_lim };
        const int x_norm = f_norm ? X_F16 : X_BF16, x_attn = f_attn ? X_F16 : X_BF16;
        const int x_gdn = f_gdn ? X_F16 : X_BF16, x_mlp = f_mlp ? X_F16 : X_BF16;
        const int f_vis = !!(f16 & ACT_VIS), x_vis = g->vis_f32 ? X_F32 : f_vis ? X_F16 : X_BF16;
        const act_args ac_vis = { f_vis, g->f16_lim };
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

        // vision tower, one image after another, before the embedding reads its features
        for (int i = 0, off = 0; !reuse_images && i < n_img; i++) {
            if (!encode_vision(g, e, &cb, &enc, &imgs->img[i], off, x_vis, &ac_vis, err, errlen)) return false;
            off += imgs->img[i].pt->n_patch;
        }

        // embedding, with image features in place of their placeholder tokens
        {
            [enc setComputePipelineState:g->ps[@"embed"]];
            int h = H;
            [enc setBytes:&h length:sizeof(h) atIndex:0];
            [enc setBuffer:g->ids offset:0 atIndex:1];
            [enc setBuffer:g->weights offset:woff(e->w.token_embd) atIndex:2];
            [enc setBuffer:g->x offset:0 atIndex:3];
            [enc setBuffer:g->img_row offset:0 atIndex:4];
            // without images every img_row is -1 and embed never reads this binding; img_row
            // stands in so a text-only engine needs no feature buffer at all
            [enc setBuffer:reuse_images ? px->vision : n_img ? g->feat : g->img_row offset:0 atIndex:5];
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
            if (reuse_images) fprintf(stderr, "clef: image cache reused %d features\n", image_rows);
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
        // A flagged or failed pass never becomes reusable. The host commits its image
        // identity and prefix only after the head succeeds; no other key can see this buffer.
        if (cache_images && !reuse_images && overflow && !overflow[0])
            memcpy(px->vision.contents, g->feat.contents, image_bytes);
        in->nh = (const float *)g->nh32.contents;
        in->nh_skip = L;
        in->n_kv = g->n_kv;
        for (int i = 0; i < g->n_kv; i++) in->kv[i] = (const float *)kvb[i].contents;
    }
    return true;
}

bool clef_gpu_forward(clef_gpu *g, const clef_engine *e, const int32_t *ids, const int32_t *pos3,
                      const int32_t *seq_start, const int32_t *seq_bounds, int n_seq, int T, bool bf16_only,
                      bool *overflow, clef_head_inputs *in, float *dump_layers, int dump_rows,
                      const clef_gpu_images *imgs, char *err, size_t errlen) {
    return forward(g, e, ids, pos3, seq_start, seq_bounds, n_seq, T, bf16_only, overflow, in, dump_layers, dump_rows,
                   imgs, NULL, 0, NULL, err, errlen);
}

bool clef_gpu_forward_prefix(clef_gpu *g, const clef_engine *e, clef_gpu_prefix *px, const int32_t *ids, const int32_t *pos3,
                             int T, int L, const clef_prefix_plan *plan, bool *overflow, clef_head_inputs *in,
                             const clef_gpu_images *imgs, char *err, size_t errlen) {
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
    int32_t *ss = calloc((size_t)n, 4);
    if (!ss) return gerr(err, errlen, "out of memory (prefix pass)");
    const int32_t bounds[2] = { 0, n };
    // the record's own positions from row L on: text continues its count, image rows keep their grid
    const bool ok = forward(g, e, ids + L, pos3 + 3 * (size_t)L, ss, bounds, 1, n, false, overflow, in, NULL, 0, imgs, px, L, plan, err, errlen);
    free(ss);
    return ok;
}
