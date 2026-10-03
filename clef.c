#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "clef_engine.h"

static bool cfg_u32(const gguf_file *f, const char *key, int *out, char *err, size_t errlen) {
    uint32_t v;
    if (!gguf_get_u32(f, key, &v) || v == 0 || v > (1u << 24)) {
        snprintf(err, errlen, "model: missing or invalid %s", key);
        return false;
    }
    *out = (int)v;
    return true;
}

static bool load_config(const gguf_file *f, clef_config *c, char *err, size_t errlen) {
    gguf_str arch;
    if (!gguf_get_str(f, "general.architecture", &arch) || arch.len != 4 || memcmp(arch.ptr, "clef", 4)) {
        snprintf(err, errlen, "model: not a clef GGUF (convert with tools/convert.py)");
        return false;
    }
    if (!cfg_u32(f, "clef.block_count", &c->n_layer, err, errlen) ||
        !cfg_u32(f, "clef.embedding_length", &c->H, err, errlen) ||
        !cfg_u32(f, "clef.feed_forward_length", &c->ffn, err, errlen) ||
        !cfg_u32(f, "clef.attention.head_count", &c->nh, err, errlen) ||
        !cfg_u32(f, "clef.attention.head_count_kv", &c->nkv, err, errlen) ||
        !cfg_u32(f, "clef.attention.key_length", &c->hd, err, errlen) ||
        !cfg_u32(f, "clef.rope.dimension_count", &c->n_rot, err, errlen) ||
        !cfg_u32(f, "clef.vocab_size", &c->vocab, err, errlen) ||
        !cfg_u32(f, "clef.ssm.conv_kernel", &c->ssm_kernel, err, errlen) ||
        !cfg_u32(f, "clef.ssm.k_heads", &c->Hk, err, errlen) ||
        !cfg_u32(f, "clef.ssm.v_heads", &c->Hv, err, errlen) ||
        !cfg_u32(f, "clef.ssm.k_head_dim", &c->dk, err, errlen) ||
        !cfg_u32(f, "clef.ssm.v_head_dim", &c->dv, err, errlen) ||
        !cfg_u32(f, "clef.head.width", &c->W, err, errlen) ||
        !cfg_u32(f, "clef.head.routing_layers", &c->routing_layers, err, errlen) ||
        !cfg_u32(f, "clef.head.layers", &c->head_layers, err, errlen) ||
        !cfg_u32(f, "clef.head.heads", &c->head_heads, err, errlen) ||
        !cfg_u32(f, "clef.head.feedforward", &c->head_ff, err, errlen)) {
        return false;
    }
    if (!gguf_get_f32(f, "clef.attention.layer_norm_rms_epsilon", &c->eps) ||
        !gguf_get_f32(f, "clef.rope.freq_base", &c->rope_theta)) {
        snprintf(err, errlen, "model: missing rms eps / rope base");
        return false;
    }
    uint64_t n = 0;
    int32_t types[CLEF_MAX_LAYERS];
    if (c->n_layer > CLEF_MAX_LAYERS ||
        !gguf_read_i32_array(f, "clef.layer_types", types, CLEF_MAX_LAYERS, &n) || (int)n != c->n_layer) {
        snprintf(err, errlen, "model: bad layer_types");
        return false;
    }
    for (int i = 0; i < c->n_layer; i++) {
        if (types[i] != 0 && types[i] != 1) {
            snprintf(err, errlen, "model: layer %d has unknown type %d", i, types[i]);
            return false;
        }
        c->layer_full[i] = types[i] == 1;
    }
    /* Shape assumptions baked into the kernels; refuse anything else rather than misbehave. */
    if (c->hd != 256 || c->n_rot != 64 || c->nh % c->nkv || c->dk != 128 || c->dv % 16 ||
        c->Hv % c->Hk || c->ssm_kernel > 8 || c->routing_layers + c->head_layers > 16) {
        snprintf(err, errlen, "model: unsupported shape (hd=%d n_rot=%d dk=%d)", c->hd, c->n_rot, c->dk);
        return false;
    }
    return true;
}

static const gguf_tensor *bind(const gguf_file *f, const char *name, uint32_t type, uint64_t ne0, uint64_t ne1,
                               char *err, size_t errlen) {
    const gguf_tensor *t = gguf_find_tensor(f, name);
    if (!t) { snprintf(err, errlen, "model: missing tensor %s", name); return NULL; }
    if (t->type != type || t->ne[0] != ne0 || t->ne[1] != ne1) {
        snprintf(err, errlen, "model: tensor %s has type %u shape [%llu, %llu], expected type %u [%llu, %llu]",
                 name, t->type, (unsigned long long)t->ne[0], (unsigned long long)t->ne[1], type,
                 (unsigned long long)ne0, (unsigned long long)ne1);
        return NULL;
    }
    return t;
}

#define BIND(dst, type, ne0, ne1) do { if (!((dst) = bind(f, name, (type), (ne0), (ne1), err, errlen))) return false; } while (0)

static bool bind_weights(const gguf_file *f, const clef_config *c, clef_weights *w, char *err, size_t errlen) {
    const char *name;
    name = "token_embd.weight";  BIND(w->token_embd, GGML_BF16, c->H, c->vocab);
    name = "output.weight";      BIND(w->output, GGML_BF16, c->H, c->vocab);
    name = "output_norm.weight"; BIND(w->output_norm, GGML_F32, c->H, 1);
    const uint64_t C = 2ull * c->Hk * c->dk + (uint64_t)c->Hv * c->dv;
    const uint64_t n_ssm = C + (uint64_t)c->Hv * c->dv + 2ull * c->Hv;
    const uint64_t n_attn = (uint64_t)c->nh * 2 * c->hd + 2ull * c->nkv * c->hd;
    for (uint64_t i = 0; i < f->n_tensors; i++) {
        const gguf_tensor *t = &f->tensors[i];
        if (t->name.len > 5 && !memcmp(t->name.ptr, "head.", 5) && t->type != GGML_BF16) {
            snprintf(err, errlen, "model: head tensor %.*s is not BF16", (int)t->name.len, t->name.ptr);
            return false;
        }
    }
    char buf[128];
    name = buf;
    for (int l = 0; l < c->n_layer; l++) {
        clef_layer_w *L = &w->layer[l];
        snprintf(buf, sizeof(buf), "blk.%d.attn_norm.weight", l);   BIND(L->attn_norm, GGML_F32, c->H, 1);
        snprintf(buf, sizeof(buf), "blk.%d.ffn_norm.weight", l);    BIND(L->ffn_norm, GGML_F32, c->H, 1);
        snprintf(buf, sizeof(buf), "blk.%d.ffn_gate_up.weight", l); BIND(L->ffn_gate_up, GGML_BF16, c->H, 2ull * c->ffn);
        snprintf(buf, sizeof(buf), "blk.%d.ffn_down.weight", l);    BIND(L->ffn_down, GGML_BF16, c->ffn, c->H);
        if (c->layer_full[l]) {
            snprintf(buf, sizeof(buf), "blk.%d.attn_qkv.weight", l);    BIND(L->attn_qkv, GGML_BF16, c->H, n_attn);
            snprintf(buf, sizeof(buf), "blk.%d.attn_output.weight", l); BIND(L->attn_output, GGML_BF16, (uint64_t)c->nh * c->hd, c->H);
            snprintf(buf, sizeof(buf), "blk.%d.attn_q_norm.weight", l); BIND(L->attn_q_norm, GGML_F32, c->hd, 1);
            snprintf(buf, sizeof(buf), "blk.%d.attn_k_norm.weight", l); BIND(L->attn_k_norm, GGML_F32, c->hd, 1);
        } else {
            snprintf(buf, sizeof(buf), "blk.%d.ssm_in.weight", l);      BIND(L->ssm_in, GGML_BF16, c->H, n_ssm);
            snprintf(buf, sizeof(buf), "blk.%d.ssm_out.weight", l);     BIND(L->ssm_out, GGML_BF16, (uint64_t)c->Hv * c->dv, c->H);
            snprintf(buf, sizeof(buf), "blk.%d.ssm_conv1d.weight", l);  BIND(L->ssm_conv1d, GGML_F32, c->ssm_kernel, C);
            snprintf(buf, sizeof(buf), "blk.%d.ssm_dt.bias", l);        BIND(L->ssm_dt, GGML_F32, c->Hv, 1);
            snprintf(buf, sizeof(buf), "blk.%d.ssm_a", l);              BIND(L->ssm_a, GGML_F32, c->Hv, 1);
            snprintf(buf, sizeof(buf), "blk.%d.ssm_norm.weight", l);    BIND(L->ssm_norm, GGML_F32, c->dv, 1);
        }
    }
    return true;
}

clef_engine *clef_open(const char *path, char *err, size_t errlen) {
    clef_engine *e = calloc(1, sizeof(*e));
    if (!e) { snprintf(err, errlen, "out of memory"); return NULL; }
    if (!gguf_open(&e->gguf, path, err, errlen)) { free(e); return NULL; }
    if (!load_config(&e->gguf, &e->cfg, err, errlen) || !bind_weights(&e->gguf, &e->cfg, &e->w, err, errlen)) {
        clef_close(e);
        return NULL;
    }
    if (!(e->tok = clef_tok_load(&e->gguf, err, errlen)) ||
        !(e->head = clef_head_load(&e->gguf, &e->cfg, err, errlen)) ||
        !(e->gpu = clef_gpu_open(e, err, errlen))) {
        clef_close(e);
        return NULL;
    }
    return e;
}

void clef_close(clef_engine *e) {
    if (!e) return;
    clef_gpu_close(e->gpu);
    clef_head_free(e->head);
    clef_tok_free(e->tok);
    gguf_close(&e->gguf);
    free(e);
}

void clef_free_probs(const clef_record *recs, int n, float ***probs) {
    if (!probs) return;
    for (int r = 0; r < n; r++) {
        if (!probs[r]) continue;
        for (int q = 0; q < recs[r].nq; q++) free(probs[r][q]);
        free(probs[r]);
    }
    free(probs);
}

static void softmax_f32(float *x, int n) {
    float m = -INFINITY;
    for (int i = 0; i < n; i++) if (x[i] > m) m = x[i];
    float s = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - m); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}

/* A non-finite logit would leave as a NaN decision in an HTTP 200; fail explicitly instead (the
 * server then retries each record alone). A correct forward never produces one: FP16 operands are
 * saturated when they overflow (act16) and everything else is f32. */
static bool logits_finite(const clef_record *rec, float **lg) {
    for (int q = 0; q < rec->nq; q++)
        for (int k = 0; k < rec->q[q].n_opt; k++)
            if (!isfinite(lg[q][k])) return false;
    return true;
}

/* Shared by clef_run and the parity tool: logits (not probabilities) when raw is set. */
bool clef_run_ex(clef_engine *e, const clef_record *recs, int n, float ****out, bool raw, float *dump,
                 int dump_rows, char *err, size_t errlen) {
    if (errlen) err[0] = '\0';   /* every failure path below writes a message */
    if (n > 1 && getenv("CLEF_DEBUG_FAIL_MULTI")) {
        /* test hook (tests/test_server_retry.py): fail every multi-record forward so the
         * server's per-record retry after a batch failure is exercised */
        snprintf(err, errlen, "injected batch failure (CLEF_DEBUG_FAIL_MULTI)");
        return false;
    }
    int T = 0;
    for (int r = 0; r < n; r++) {
        if (recs[r].ids.len == 0 || recs[r].ids.len > (size_t)(1 << 20)) { snprintf(err, errlen, "record %d: bad length", r); return false; }
        T += (int)recs[r].ids.len;
    }
    int32_t *ids = malloc((size_t)T * 4), *pos = malloc((size_t)T * 4), *ss = malloc((size_t)T * 4);
    int32_t *bounds = malloc((size_t)(n + 1) * 4);
    float ***probs = calloc((size_t)n, sizeof(*probs));
    bool ok = ids && pos && ss && bounds && probs;
    if (!ok) snprintf(err, errlen, "out of memory (batch of %d, %d tokens)", n, T);
    int t = 0;
    for (int r = 0; ok && r < n; r++) {
        bounds[r] = t;
        for (size_t i = 0; i < recs[r].ids.len; i++, t++) {
            const int32_t id = recs[r].ids.ids[i];
            if (id < 0 || id >= e->cfg.vocab) { snprintf(err, errlen, "token id %d out of range", id); ok = false; break; }
            ids[t] = id;
            pos[t] = (int32_t)i;
            ss[t] = bounds[r];
        }
    }
    if (ok) bounds[n] = T;
    clef_head_inputs in = {0};
    bool *ovf = calloc((size_t)n, sizeof(bool));
    if (ok && !ovf) { snprintf(err, errlen, "out of memory (batch of %d)", n); ok = false; }
    if (ok) ok = clef_gpu_forward(e->gpu, e, ids, pos, ss, bounds, n, T, false, ovf, &in, dump, dump_rows, err, errlen);
    for (int r = 0; ok && r < n; r++) {
        probs[r] = calloc((size_t)recs[r].nq, sizeof(float *));
        ok = probs[r] != NULL;
        for (int q = 0; ok && q < recs[r].nq; q++) ok = (probs[r][q] = calloc((size_t)recs[r].q[q].n_opt, sizeof(float))) != NULL;
        if (!ok) { snprintf(err, errlen, "out of memory (logits)"); break; }
        if (ovf[r]) continue;   /* recomputed below; the head inputs must be read before the rerun overwrites them */
        if (!clef_head_run(e->head, &e->cfg, e->w.output, &in, bounds[r], &recs[r], probs[r])) {
            snprintf(err, errlen, "head failed (out of memory)");
            ok = false;
        } else if (!logits_finite(&recs[r], probs[r])) {
            snprintf(err, errlen, "record %d: non-finite logits", r);
            ok = false;
        }
    }
    /* A record whose FP16 GEMM operands left FP16's range is recomputed alone with BF16
     * activations (the pre-FP16 precision, and the reference's). Overflow depends only on the
     * record's own rows, so its result is the same in any batch. */
    for (int r = 0; ok && r < n; r++) {
        if (!ovf[r]) continue;
        const int len = (int)recs[r].ids.len;
        int32_t *ss0 = calloc((size_t)len, sizeof(int32_t));   /* alone: every token's record starts at 0 */
        const int32_t b1[2] = { 0, len };
        if (!ss0) { snprintf(err, errlen, "out of memory (rerun of %d tokens)", len); ok = false; break; }
        ok = clef_gpu_forward(e->gpu, e, ids + bounds[r], pos + bounds[r], ss0, b1, 1, len, true, NULL, &in,
                              r == 0 ? dump : NULL, dump_rows, err, errlen);
        free(ss0);
        if (ok && !clef_head_run(e->head, &e->cfg, e->w.output, &in, 0, &recs[r], probs[r])) {
            snprintf(err, errlen, "head failed (out of memory)");
            ok = false;
        } else if (ok && !logits_finite(&recs[r], probs[r])) {
            snprintf(err, errlen, "record %d: non-finite logits", r);
            ok = false;
        }
    }
    for (int r = 0; ok && !raw && r < n; r++)
        for (int q = 0; q < recs[r].nq; q++) softmax_f32(probs[r][q], recs[r].q[q].n_opt);
    free(ovf);
    free(ids); free(pos); free(ss); free(bounds);
    if (!ok) { clef_free_probs(recs, n, probs); return false; }
    *out = probs;
    return true;
}

bool clef_run(clef_engine *e, const clef_record *recs, int n, float ****probs, char *err, size_t errlen) {
    return clef_run_ex(e, recs, n, probs, false, NULL, 0, err, errlen);
}
