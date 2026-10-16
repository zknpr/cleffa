// Qwen3-Omni decoder dispatch and scratch. Shared lifecycle stays in clef_metal.m.
#include "clef_metal_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Larger record-relative chunks reuse expert weights across more rows without
// changing a token's reduction order. The arena remains independent of context.
enum { OMNI_CHUNK = 1024, OMNI_EXPERTS = 128, OMNI_TOP = 8, OMNI_PAD = OMNI_CHUNK*OMNI_TOP+OMNI_EXPERTS*31 };
typedef struct { int rows, H, F, padded; } omni_moe_args;
typedef struct { int nh, nkv, T; float eps; int rope_exact; } omni_prep_args;
typedef struct { int nh, nkv, hd, T; float scale; } omni_attention_args;
typedef struct { int T, N, K; } omni_gemm_args;
typedef struct { float logits[OMNI_EXPERTS], low[OMNI_EXPERTS]; int32_t ids[OMNI_TOP]; float weights[OMNI_TOP]; } omni_route_row;
typedef struct { int rows, start, first, captured, layer; } omni_trace_args;

struct clef_omni_gpu {
    id<MTLBuffer> scores, scores_low, ids, weights, status, counts, offsets, rowmap, slots;
    id<MTLBuffer> gathered, gate_up, activated, down;
    id<MTLBuffer> trace;
    int trace_first, trace_rows, trace_layers;
};

static void free_state(clef_omni_gpu *s) {
    if (!s) return;
    s->scores = s->ids = s->weights = s->status = s->counts = s->offsets = s->rowmap = s->slots = nil;
    s->scores_low = nil;
    s->gathered = s->gate_up = s->activated = s->down = nil;
    s->trace = nil;
    free(s);
}

static id<MTLBuffer> allocate(clef_gpu *g, size_t rows, size_t columns, size_t width) {
    if (!columns || !width || rows > SIZE_MAX/columns || rows*columns > SIZE_MAX/width ||
        rows*columns*width > g->dev.maxBufferLength) return nil;
    return [g->dev newBufferWithLength:rows*columns*width options:MTLResourceStorageModeShared];
}

bool clef_omni_reserve(clef_gpu *g, const clef_config *c, int T, char *err, size_t errlen) {
    if (T <= 0 || c->H != 2048 || c->ffn != 768 || c->n_expert != OMNI_EXPERTS || c->n_expert_used != OMNI_TOP) {
        snprintf(err, errlen, "metal: invalid Omni scratch dimensions");
        return false;
    }
    if (g->omni) return true;
    clef_omni_gpu *s = calloc(1, sizeof(*s));
    if (!s) { snprintf(err, errlen, "metal: cannot allocate Omni state"); return false; }
    s->scores = allocate(g, OMNI_CHUNK, OMNI_EXPERTS, 4);
    s->scores_low = allocate(g, OMNI_CHUNK, OMNI_EXPERTS, 4);
    s->ids = allocate(g, OMNI_CHUNK, OMNI_TOP, 4);
    s->weights = allocate(g, OMNI_CHUNK, OMNI_TOP, 4);
    s->status = allocate(g, 1, 1, 4);
    s->counts = allocate(g, OMNI_EXPERTS, 1, 4);
    s->offsets = allocate(g, OMNI_EXPERTS+1, 1, 4);
    s->rowmap = allocate(g, OMNI_PAD, 1, 4);
    s->slots = allocate(g, OMNI_CHUNK, OMNI_TOP, 4);
    s->gathered = allocate(g, OMNI_PAD, c->H, 4);
    s->gate_up = allocate(g, OMNI_PAD, 2*c->ffn, 4);
    s->activated = allocate(g, OMNI_PAD, c->ffn, 4);
    s->down = allocate(g, OMNI_PAD, c->H, 4);
    if (!s->scores || !s->scores_low || !s->ids || !s->weights || !s->status || !s->counts || !s->offsets ||
        !s->rowmap || !s->slots || !s->gathered || !s->gate_up || !s->activated || !s->down) {
        free_state(s);
        snprintf(err, errlen, "metal: cannot allocate Omni expert arena (previous buffers kept)");
        return false;
    }
    g->omni = s;
    return true;
}

void clef_omni_free(clef_gpu *g) {
    free_state(g->omni);
    g->omni = NULL;
}

bool clef_omni_trace_reserve(clef_gpu *g, int T, int layers, int rows, bool enabled, char *err, size_t errlen) {
    clef_omni_gpu *s = g->omni;
    if (!enabled) {
        s->trace = nil;
        s->trace_rows = s->trace_layers = 0;
        return true;
    }
    if (T <= 0 || layers <= 0 || layers > CLEF_MAX_LAYERS || rows < 0) {
        snprintf(err, errlen, "metal: invalid Omni trace dimensions");
        return false;
    }
    const int R = rows > 0 && rows < T ? rows : T;
    id<MTLBuffer> next = allocate(g, (size_t)layers, R, sizeof(omni_route_row));
    if (!next) {
        snprintf(err, errlen, "metal: cannot allocate Omni route capture (previous buffers kept)");
        return false;
    }
    s->trace = next;
    s->trace_first = T-R; s->trace_rows = R; s->trace_layers = layers;
    return true;
}

bool clef_gpu_dump_routes(clef_gpu *g, const char *path, char *err, size_t errlen) {
    const clef_omni_gpu *s = g->omni;
    if (!s || !s->trace || !s->trace_rows) {
        snprintf(err, errlen, "no captured Omni routes");
        return false;
    }
    const omni_route_row *rows = s->trace.contents;
    // Validate before creating output; diagnostics never serialize NaN as invalid JSON.
    for (size_t i = 0; i < (size_t)s->trace_layers*s->trace_rows; i++) {
        for (int j = 0; j < OMNI_EXPERTS; j++) if (!isfinite(rows[i].logits[j]) || !isfinite(rows[i].low[j])) {
            snprintf(err, errlen, "nonfinite captured Omni router logits"); return false;
        }
        for (int j = 0; j < OMNI_TOP; j++) if (rows[i].ids[j] < 0 || rows[i].ids[j] >= OMNI_EXPERTS || !isfinite(rows[i].weights[j])) {
            snprintf(err, errlen, "invalid captured Omni routes"); return false;
        }
    }
    FILE *f = fopen(path, "w");
    if (!f) { snprintf(err, errlen, "cannot open Omni route dump: %s", path); return false; }
    for (int l = 0; l < s->trace_layers; l++) for (int t = 0; t < s->trace_rows; t++) {
        const omni_route_row *r = rows+(size_t)l*s->trace_rows+t;
        double top[OMNI_TOP+1];
        for (int j = 0; j <= OMNI_TOP; j++) top[j] = -INFINITY;
        for (int e = 0; e < OMNI_EXPERTS; e++) {
            double v = (double)r->logits[e]+r->low[e];
            for (int j = 0; j <= OMNI_TOP; j++) if (v > top[j]) { double old = top[j]; top[j] = v; v = old; }
        }
        fprintf(f, "{\"layer\":%d,\"token\":%d,\"logits\":[", l, s->trace_first+t);
        for (int j = 0; j < OMNI_EXPERTS; j++) fprintf(f, "%s%.17g", j ? "," : "", (double)r->logits[j]+r->low[j]);
        fputs("],\"ids\":[", f);
        for (int j = 0; j < OMNI_TOP; j++) fprintf(f, "%s%d", j ? "," : "", r->ids[j]);
        fputs("],\"weights\":[", f);
        for (int j = 0; j < OMNI_TOP; j++) fprintf(f, "%s%.9g", j ? "," : "", r->weights[j]);
        fprintf(f, "],\"margin\":%.17g}\n", top[OMNI_TOP-1]-top[OMNI_TOP]);
    }
    bool ok = !ferror(f);
    if (fclose(f)) ok = false;
    if (!ok) snprintf(err, errlen, "cannot write Omni route dump: %s", path);
    return ok;
}

void clef_omni_keepalive(clef_gpu *g, id<MTLComputeCommandEncoder> enc) {
    const clef_omni_gpu *s = g->omni;
    if (!s) return;
    id<MTLBuffer> buffers[] = {s->scores, s->scores_low, s->ids, s->weights, s->status, s->counts, s->offsets,
        s->rowmap, s->slots, s->gathered, s->gate_up, s->activated, s->down, s->trace};
    for (size_t i = 0; i < sizeof(buffers)/sizeof(buffers[0]); i++) if (buffers[i]) {
        [enc setBuffer:buffers[i] offset:0 atIndex:0];
        [enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    }
}

void clef_omni_begin(clef_gpu *g) {
    clef_omni_gpu *s = g->omni;
    *(uint32_t *)s->status.contents = 0;
    if (g->debug_poison) {
        id<MTLBuffer> buffers[] = {s->scores, s->scores_low, s->weights, s->gathered, s->gate_up, s->activated, s->down, s->trace};
        for (size_t i = 0; i < sizeof(buffers)/sizeof(buffers[0]); i++) {
            float *p = buffers[i].contents;
            for (NSUInteger j = 0; j < buffers[i].length/4; j++) p[j] = NAN;
        }
    }
}

bool clef_omni_complete(clef_gpu *g, char *err, size_t errlen) {
    if (*(uint32_t *)g->omni->status.contents) {
        snprintf(err, errlen, "metal: nonfinite Omni attention or router values");
        return false;
    }
    return true;
}

static void norm(clef_gpu *g, id<MTLComputeCommandEncoder> enc, const clef_config *c,
                  const gguf_tensor *weight, int T) {
    struct { int H; float eps; } args = {c->H, c->eps};
    id<MTLComputePipelineState> ps = g->ps[@"rmsnorm_f32"];
    [enc setComputePipelineState:ps];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:g->x offset:0 atIndex:1];
    [enc setBuffer:g->weights offset:weight->offset atIndex:2];
    [enc setBuffer:g->xn offset:0 atIndex:3];
    NSUInteger threads = ps.maxTotalThreadsPerThreadgroup < 1024 ? ps.maxTotalThreadsPerThreadgroup : 1024;
    [enc dispatchThreadgroups:MTLSizeMake(T, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
}

static void expert_gemm(clef_gpu *g, id<MTLComputeCommandEncoder> enc, id<MTLBuffer> input,
                         const gguf_tensor *weight, id<MTLBuffer> output, int rows, int padded, int N, int K) {
    clef_omni_gpu *s = g->omni;
    omni_gemm_args args = {rows, N, K};
    [enc setComputePipelineState:g->ps[@"omni_expert_gemm"]];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:input offset:0 atIndex:1];
    [enc setBuffer:g->weights offset:weight->offset atIndex:2];
    [enc setBuffer:output offset:0 atIndex:3];
    [enc setBuffer:s->counts offset:0 atIndex:4];
    [enc setBuffer:s->offsets offset:0 atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake((N+127)/128, (padded+31)/32, 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
}

// Profile boundary: books the preceding dispatches to `cat` and continues on a new encoder.
#define OPROF(cat) do { if (!clef_metal_prof(g, cb, encp, (cat), err, errlen)) return false; enc = *encp; } while (0)

bool clef_omni_encode_attention(clef_gpu *g, const clef_engine *e, id<MTLCommandBuffer> __strong *cb,
                               id<MTLComputeCommandEncoder> __strong *encp, int layer, int T, char *err, size_t errlen) {
    id<MTLComputeCommandEncoder> enc = encp ? *encp : nil;
    if (!enc || !g->omni || layer < 0 || layer >= e->cfg.n_layer) {
        snprintf(err, errlen, "metal: invalid Omni layer dispatch");
        return false;
    }
    const clef_config *c = &e->cfg;
    const clef_layer_w *w = &e->w.layer[layer];
    clef_omni_gpu *s = g->omni;
    norm(g, enc, c, w->attn_norm, T);
    OPROF(P_NORM);
    clef_metal_gemm(g, enc, X_F32, false, g->xn, 0, w->attn_qkv->offset,
                    (c->nh+2*c->nkv)*c->hd, c->H, g->P, 0, T);
    OPROF(P_GEMM);
    omni_prep_args pa = {c->nh, c->nkv, T, c->eps, g->omni_rope_exact};
    [enc setComputePipelineState:g->ps[@"omni_attn_prep"]];
    [enc setBytes:&pa length:sizeof(pa) atIndex:0];
    [enc setBuffer:g->P offset:0 atIndex:1];
    [enc setBuffer:g->weights offset:w->attn_q_norm->offset atIndex:2];
    [enc setBuffer:g->weights offset:w->attn_k_norm->offset atIndex:3];
    [enc setBuffer:g->pos offset:0 atIndex:4];
    [enc setBuffer:g->inv_freq offset:0 atIndex:5];
    [enc setBuffer:g->Q offset:0 atIndex:6];
    [enc setBuffer:g->K offset:0 atIndex:7];
    [enc setBuffer:g->V offset:0 atIndex:8];
    [enc setBuffer:s->status offset:0 atIndex:9];
    [enc dispatchThreadgroups:MTLSizeMake(T, c->nh+c->nkv, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    OPROF(P_ATTNPREP);
    omni_attention_args aa = {c->nh, c->nkv, c->hd, T, 1.f/sqrtf((float)c->hd)};
    [enc setComputePipelineState:g->ps[@"omni_attention"]];
    [enc setBytes:&aa length:sizeof(aa) atIndex:0];
    [enc setBuffer:g->Q offset:0 atIndex:1]; [enc setBuffer:g->K offset:0 atIndex:2];
    [enc setBuffer:g->V offset:0 atIndex:3]; [enc setBuffer:g->seq_start offset:0 atIndex:4];
    // There is no query gate; its otherwise unused FP32 buffer holds attention output.
    [enc setBuffer:g->G offset:0 atIndex:5];
    [enc setThreadgroupMemoryLength:(c->nh/c->nkv)*(8*32+64)*4 atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake((T+7)/8, c->nkv, 1)
        threadsPerThreadgroup:MTLSizeMake(32*(c->nh/c->nkv), 1, 1)];
    OPROF(P_ATTN);
    clef_metal_gemm(g, enc, X_F32, true, g->G, 0, w->attn_output->offset, c->H, c->nh*c->hd, g->x, 0, T);
    OPROF(P_GEMM);
    norm(g, enc, c, w->ffn_norm, T);
    OPROF(P_NORM);
    return true;
}

bool clef_omni_encode_layer(clef_gpu *g, const clef_engine *e, id<MTLCommandBuffer> __strong *cb,
                            id<MTLComputeCommandEncoder> __strong *encp, int layer, int T, int n_seq, bool bf16_only,
                            char *err, size_t errlen) {
    if (!clef_omni_encode_attention(g, e, cb, encp, layer, T, err, errlen)) return false;
    id<MTLComputeCommandEncoder> enc = *encp;
    const clef_config *c = &e->cfg;
    const clef_layer_w *w = &e->w.layer[layer];
    clef_omni_gpu *s = g->omni;
    const int32_t *bounds = g->seq_bounds.contents;
    for (int record = 0; record < n_seq; record++) {
        for (int start = bounds[record]; start < bounds[record+1]; start += OMNI_CHUNK) {
            int rows = bounds[record+1]-start;
            if (rows > OMNI_CHUNK) rows = OMNI_CHUNK;
            const NSUInteger input_offset = (NSUInteger)start*c->H*4;
            omni_gemm_args ra = {rows, OMNI_EXPERTS, c->H};
            [enc setComputePipelineState:g->ps[@"omni_router"]];
            [enc setBytes:&ra length:sizeof(ra) atIndex:0];
            [enc setBuffer:g->xn offset:input_offset atIndex:1];
            [enc setBuffer:g->weights offset:w->ffn_router->offset atIndex:2];
            [enc setBuffer:s->scores offset:0 atIndex:3];
            [enc setBuffer:s->scores_low offset:0 atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(rows, OMNI_EXPERTS, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            [enc setComputePipelineState:g->ps[@"omni_route"]];
            [enc setBuffer:s->scores offset:0 atIndex:0]; [enc setBuffer:s->ids offset:0 atIndex:1];
            [enc setBuffer:s->weights offset:0 atIndex:2]; [enc setBuffer:s->status offset:0 atIndex:3];
            [enc setBuffer:s->scores_low offset:0 atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            if (s->trace && start+rows > s->trace_first) {
                omni_trace_args ta = {rows, start, s->trace_first, s->trace_rows, layer};
                [enc setComputePipelineState:g->ps[@"omni_trace"]];
                [enc setBytes:&ta length:sizeof(ta) atIndex:0];
                [enc setBuffer:s->scores offset:0 atIndex:1];
                [enc setBuffer:s->ids offset:0 atIndex:2];
                [enc setBuffer:s->weights offset:0 atIndex:3];
                [enc setBuffer:s->trace offset:0 atIndex:4];
                [enc setBuffer:s->scores_low offset:0 atIndex:5];
                [enc dispatchThreads:MTLSizeMake(OMNI_EXPERTS, rows, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            }
            OPROF(P_OMNI_ROUTER);
            // Each live assignment contributes one row; each occupied expert
            // adds at most 31 padding rows. Avoid clearing the full arena for
            // short requests or the final partial chunk.
            const int assignments = rows*OMNI_TOP;
            const int padded = assignments+(assignments < OMNI_EXPERTS ? assignments : OMNI_EXPERTS)*31;
            omni_moe_args ma = {rows, c->H, c->ffn, padded};
            // The serial encoder orders count -> offsets -> stable scatter.
            // All three kernels use the same bindings and existing scratch.
            [enc setComputePipelineState:g->ps[@"omni_assign_count"]];
            [enc setBytes:&ma length:sizeof(ma) atIndex:0];
            [enc setBuffer:s->ids offset:0 atIndex:1]; [enc setBuffer:s->counts offset:0 atIndex:2];
            [enc setBuffer:s->offsets offset:0 atIndex:3]; [enc setBuffer:s->rowmap offset:0 atIndex:4];
            [enc setBuffer:s->slots offset:0 atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(OMNI_EXPERTS, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            [enc setComputePipelineState:g->ps[@"omni_assign_offsets"]];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            [enc setComputePipelineState:g->ps[@"omni_assign_scatter"]];
            [enc dispatchThreadgroups:MTLSizeMake(OMNI_EXPERTS, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            [enc setComputePipelineState:g->ps[@"omni_gather"]];
            [enc setBytes:&ma length:sizeof(ma) atIndex:0];
            [enc setBuffer:g->xn offset:input_offset atIndex:1]; [enc setBuffer:s->rowmap offset:0 atIndex:2];
            [enc setBuffer:s->gathered offset:0 atIndex:3];
            [enc dispatchThreads:MTLSizeMake(c->H, padded, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            OPROF(P_OMNI_MOE);
            expert_gemm(g, enc, s->gathered, w->ffn_gate_up_exps, s->gate_up, rows, padded, 2*c->ffn, c->H);
            OPROF(P_OMNI_EXPERTS);
            [enc setComputePipelineState:g->ps[@"omni_swiglu"]];
            [enc setBytes:&ma length:sizeof(ma) atIndex:0];
            [enc setBuffer:s->gate_up offset:0 atIndex:1]; [enc setBuffer:s->activated offset:0 atIndex:2];
            [enc setBuffer:s->rowmap offset:0 atIndex:3];
            [enc dispatchThreads:MTLSizeMake(c->ffn, padded, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            OPROF(P_OMNI_MOE);
            expert_gemm(g, enc, s->activated, w->ffn_down_exps, s->down, rows, padded, c->H, c->ffn);
            OPROF(P_OMNI_EXPERTS);
            [enc setComputePipelineState:g->ps[@"omni_combine"]];
            [enc setBytes:&ma length:sizeof(ma) atIndex:0];
            [enc setBuffer:s->down offset:0 atIndex:1]; [enc setBuffer:s->ids offset:0 atIndex:2];
            [enc setBuffer:s->weights offset:0 atIndex:3]; [enc setBuffer:s->slots offset:0 atIndex:4];
            [enc setBuffer:g->x offset:input_offset atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            OPROF(P_OMNI_MOE);
        }
    }
    return true;
}
