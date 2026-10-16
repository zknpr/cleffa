// Offline diagnostic: execute the production attention half with captured inputs.
// Including the backend gives the probe access without adding a public debug API.
#include "../clef_metal.m"
#include "../clef_metal_omni.m"
#include <errno.h>

static bool probe_read(const char *path, void *out, size_t bytes) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    bool ok = fread(out, 1, bytes, f) == bytes && fgetc(f) == EOF && !ferror(f);
    if (fclose(f)) ok = false;
    return ok;
}

static bool probe_save_part(const char *folder, const char *name, id<MTLBuffer> b,
                            size_t first, size_t elements) {
    char path[4096];
    int n = snprintf(path, sizeof(path), "%s/%s.bin", folder, name);
    if (n < 0 || (size_t)n >= sizeof(path) || first > b.length/4 || elements > b.length/4-first) return false;
    FILE *f = fopen(path, "wb");
    if (!f) return false;
    bool ok = fwrite((const char *)b.contents+first*4, 4, elements, f) == elements;
    if (fclose(f)) ok = false;
    return ok;
}

static bool probe_save(const char *folder, const char *name, id<MTLBuffer> b, size_t elements) {
    return probe_save_part(folder, name, b, 0, elements);
}

static bool probe_save_experts(const char *folder, const char *name, id<MTLBuffer> b,
                               int width, const int32_t *slots) {
    char path[4096];
    int n = snprintf(path, sizeof(path), "%s/%s.bin", folder, name);
    if (n < 0 || (size_t)n >= sizeof(path) || width <= 0) return false;
    for (int k = 0; k < OMNI_TOP; k++) {
        if (slots[k] < 0 || slots[k] >= OMNI_PAD || (size_t)(slots[k]+1)*width > b.length/4) return false;
    }
    FILE *f = fopen(path, "wb");
    if (!f) return false;
    bool ok = true;
    for (int k = 0; k < OMNI_TOP && ok; k++) {
        ok = fwrite((const float *)b.contents+(size_t)slots[k]*width, 4, width, f) == (size_t)width;
    }
    if (fclose(f)) ok = false;
    return ok;
}

static bool probe_finish(id<MTLCommandBuffer> cb, id<MTLComputeCommandEncoder> enc) {
    if (!cb || !enc) return false;
    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
    return cb.status == MTLCommandBufferStatusCompleted && !cb.error;
}

static int probe_integer(const char *arg, int minimum, int maximum) {
    char *end;
    errno = 0;
    long value = strtol(arg, &end, 10);
    return errno || !*arg || *end || value < minimum || value > maximum ? -1 : (int)value;
}

int main(int argc, char **argv) { @autoreleasepool {
    if (argc != 7 && argc != 8) {
        fprintf(stderr, "usage: omni-layer-probe MODEL INPUT.f32 POS.i32 OUTPUT LAYER TOKENS [MOE_TOKEN]\n");
        return 2;
    }
    const int layer = probe_integer(argv[5], 0, CLEF_MAX_LAYERS-1);
    const int T = probe_integer(argv[6], 1, 65536);
    if (layer < 0 || T < 0) { fprintf(stderr, "invalid probe geometry\n"); return 2; }
    const int token = argc == 8 ? probe_integer(argv[7], 0, T-1) : -1;
    if (argc == 8 && token < 0) { fprintf(stderr, "invalid MoE probe token\n"); return 2; }
    char err[512] = "probe input, output, or command failure";
    clef_engine *e = clef_open(argv[1], err, sizeof(err));
    if (!e) { fprintf(stderr, "%s\n", err); return 1; }
    clef_gpu *g = e->gpu;
    bool ok = false;
    if (e->cfg.H != 2048 || layer >= e->cfg.n_layer ||
        !ensure_capacity(g, &e->cfg, T, 1, err, sizeof(err)) ||
        !clef_omni_reserve(g, &e->cfg, T, err, sizeof(err))) goto done;
    if (!probe_read(argv[2], g->x.contents, (size_t)T*2048*4) ||
        !probe_read(argv[3], g->pos.contents, (size_t)T*3*4)) goto done;
    for (size_t i = 0; i < (size_t)T*2048; i++) if (!isfinite(((float *)g->x.contents)[i])) goto done;
    for (size_t i = 0; i < (size_t)T*3; i++) {
        int32_t p = ((int32_t *)g->pos.contents)[i];
        if (p < 0 || p > 65536) goto done;
    }
    memset(g->seq_start.contents, 0, T*4);
    clef_omni_begin(g);
    {
        // xn is reused for FFN norm; preserve the input norm before that overwrite.
        id<MTLCommandBuffer> cb = [g->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (!enc) goto done;
        norm(g, enc, &e->cfg, e->w.layer[layer].attn_norm, T);
        if (!probe_finish(cb, enc) || !probe_save(argv[4], "input_norm", g->xn, (size_t)T*2048)) goto done;
        cb = [g->queue commandBuffer]; enc = [cb computeCommandEncoder];
        if (!clef_omni_encode_attention(g, e, enc, layer, T, err, sizeof(err))) goto done;
        if (!probe_finish(cb, enc) || !clef_omni_complete(g, err, sizeof(err))) goto done;
    }
    ok = probe_save(argv[4], "qkv", g->P, (size_t)T*5120) &&
         probe_save(argv[4], "q", g->Q, (size_t)T*4096) &&
         probe_save(argv[4], "k", g->K, (size_t)T*512) &&
         probe_save(argv[4], "v", g->V, (size_t)T*512) &&
         probe_save(argv[4], "attn", g->G, (size_t)T*4096) &&
         probe_save(argv[4], "residual", g->x, (size_t)T*2048) &&
         probe_save(argv[4], "norm", g->xn, (size_t)T*2048);
    if (ok && token >= 0) {
        ok = false;
        // Replay production dispatch for the token's original expert chunk.
        // Attention still sees the whole causal prefix; limiting only the MoE
        // loop leaves this chunk's scratch available for a same-input audit.
        if (!probe_read(argv[2], g->x.contents, (size_t)T*2048*4)) goto done;
        const int start = token/OMNI_CHUNK*OMNI_CHUNK, local = token-start;
        int32_t *bounds = g->seq_bounds.contents;
        bounds[0] = start; bounds[1] = start+OMNI_CHUNK < T ? start+OMNI_CHUNK : T;
        clef_omni_begin(g);
        id<MTLCommandBuffer> cb = [g->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (!clef_omni_encode_layer(g, e, enc, layer, T, 1, false, err, sizeof(err)) ||
            !probe_finish(cb, enc) || !clef_omni_complete(g, err, sizeof(err))) goto done;
        clef_omni_gpu *s = g->omni;
        const int32_t *slots = (const int32_t *)s->slots.contents+local*OMNI_TOP;
        ok = probe_save(argv[4], "inv_freq", g->inv_freq, 64) &&
             probe_save_part(argv[4], "router_hi", s->scores, local*OMNI_EXPERTS, OMNI_EXPERTS) &&
             probe_save_part(argv[4], "router_lo", s->scores_low, local*OMNI_EXPERTS, OMNI_EXPERTS) &&
             probe_save_part(argv[4], "moe_ids", s->ids, local*OMNI_TOP, OMNI_TOP) &&
             probe_save_part(argv[4], "moe_weights", s->weights, local*OMNI_TOP, OMNI_TOP) &&
             probe_save_experts(argv[4], "expert_gathered", s->gathered, e->cfg.H, slots) &&
             probe_save_experts(argv[4], "expert_gate_up", s->gate_up, 2*e->cfg.ffn, slots) &&
             probe_save_experts(argv[4], "expert_activated", s->activated, e->cfg.ffn, slots) &&
             probe_save_experts(argv[4], "expert_down", s->down, e->cfg.H, slots) &&
             probe_save(argv[4], "output", g->x, (size_t)T*e->cfg.H);
    }
done:
    if (!ok) fprintf(stderr, "%s\n", err);
    clef_close(e);
    return ok ? 0 : 1;
} }
