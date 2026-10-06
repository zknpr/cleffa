#include <math.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "clef_engine.h"

static double wall_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

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

/* An engine address may be recycled after close. A monotonic identity prevents an old
 * cache entry from being accepted by a newly opened model at that same address. */
static atomic_uint_fast64_t next_engine_id = 1;

static bool claim_engine_id(uint64_t *id) {
    uint_fast64_t current = atomic_load_explicit(&next_engine_id, memory_order_relaxed);
    do {
        if (current == UINT64_MAX) return false;  /* never wrap into an earlier identity */
    } while (!atomic_compare_exchange_weak_explicit(&next_engine_id, &current, current + 1,
                                                    memory_order_relaxed, memory_order_relaxed));
    *id = (uint64_t)current;
    return true;
}

clef_engine *clef_open(const char *path, char *err, size_t errlen) {
    clef_engine *e = calloc(1, sizeof(*e));
    if (!e) { snprintf(err, errlen, "out of memory"); return NULL; }
    if (!claim_engine_id(&e->instance_id)) { snprintf(err, errlen, "engine identity exhausted"); free(e); return NULL; }
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
    const double t_head = wall_ms();
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
    /* CLEF_STAGE_TIME=1 (with the GPU stage line in clef_metal.m): CPU head time for the batch */
    if (ok && getenv("CLEF_STAGE_TIME")) fprintf(stderr, "clef: stage head n=%d: %.2f ms\n", n, wall_ms() - t_head);
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

/* ---- prefix cache ---- */

struct clef_prefix {
    clef_gpu_prefix *gpu;
    uint64_t engine_id;  /* bound before any model-dependent buffer can be allocated */
    int32_t *ids;   /* the tokens the entry holds state for */
    int len;        /* how many: the snapshot row, a multiple of 32; 0 = nothing usable */
    int cls;        /* clef_gpu_prefix_class of the record that produced the state */
    /* Checkpoints: the rows of the record whose DeltaNet state the entry holds, by slot of the GPU
     * entry; 0 = free. While the entry is usable, `len` is one of them. ck_use orders them by the
     * last pass that stored or resumed from each, for eviction. */
    int ck_row[CLEF_PREFIX_CKPT];
    unsigned ck_use[CLEF_PREFIX_CKPT], use;
};

/* The attention K/V and head memory rows of an entry are per token, so any leading part of them
 * serves a record that starts with the same tokens. The DeltaNet layers are recurrent: their
 * state exists only at the rows where a pass stored it. So an entry keeps it at several rows,
 * and a record resumes from the last one inside the tokens it shares with the entry:
 *   - the snapshot before the schema, as always;
 *   - every CLEF_PREFIX_PERIOD tokens, so that a state which differs from the entry's somewhere
 *     recomputes fewer than that many of the tokens they share;
 *   - the block where a record left the entry's tokens, so that the next record leaving them
 *     there, a fixed preamble with another tail, recomputes only its own part.
 * Each is one copy of the state: 50 MB on clef-flash, 151 MB on the 27B. Below CLEF_PREFIX_MIN
 * tokens nothing is cached: every request shares the template tokens, and a checkpoint there
 * would cost a state copy to save a few milliseconds. */
#define CLEF_PREFIX_PERIOD 2048
#define CLEF_PREFIX_MIN 128

/* A longer state usually re-tokenizes the last tokens of a shorter one, so the snapshot stays
 * this far before the schema. Reuse is decided by token equality; the margin only makes a match
 * likely when a state grows. */
#define CLEF_PREFIX_MARGIN 8

clef_prefix *clef_prefix_new(void) {
    clef_prefix *p = calloc(1, sizeof(*p));
    if (p && !(p->gpu = clef_gpu_prefix_new())) { free(p); p = NULL; }
    return p;
}

void clef_prefix_free(clef_prefix *p) {
    if (!p) return;
    clef_gpu_prefix_free(p->gpu);
    free(p->ids);
    free(p);
}

size_t clef_prefix_bytes(const clef_prefix *p) { return p ? clef_gpu_prefix_bytes(p->gpu) : 0; }

static bool prefix_owner_ok(const clef_engine *e, const clef_prefix *p, char *err, size_t errlen) {
    if (!e || !p || !e->instance_id) {
        snprintf(err, errlen, "invalid prefix cache owner");
        return false;
    }
    if (p->engine_id && p->engine_id != e->instance_id) {
        snprintf(err, errlen, "prefix cache belongs to another engine");
        return false;
    }
    return true;
}

bool clef_prefix_keep_warm(clef_engine *e, const clef_prefix *p, char *err, size_t errlen) {
    if (!prefix_owner_ok(e, p, err, errlen)) return false;
    return clef_gpu_prefix_keepalive(e->gpu, p->gpu, err, errlen);
}

/* Template reuse (clef_run_template). Every request begins with the same 36 template tokens: the
 * system prompt and the header of the user turn (clef_record.c). An entry pinned to their first
 * 32, one attention query block, can reuse only public template state across requests. Its
 * buffers also retain request-dependent suffix rows, which each pass overwrites before
 * reading. Like any entry it keeps K/V rows for the whole record, so CLEF_TEMPLATE_MAX bounds
 * its capacity. Eligibility also accounts for GEMM tile padding below. All eligible records
 * are in one DeltaNet class on both models. */
#define CLEF_TEMPLATE_TOKENS 32
#define CLEF_TEMPLATE_MAX 2048

/* One record through an entry whose snapshot row for this record is Ls (0: do not use it).
 * `multi` adds the periodic and divergence checkpoints; without it the entry has its snapshot only. */
static bool run_entry(clef_engine *e, clef_prefix *p, const clef_record *rec, int Ls, bool multi, float ****out, bool raw,
                      int *reused, char *err, size_t errlen) {
    if (errlen) err[0] = '\0';
    if (reused) *reused = 0;
    if (!prefix_owner_ok(e, p, err, errlen)) return false;
    p->engine_id = e->instance_id;
    const int T = rec->ids.len <= (size_t)(1 << 20) ? (int)rec->ids.len : 0;
    /* Every row the head reads from the pass (schema spans, the last token) must lie past the snapshot. */
    bool spans_ok = Ls > 0 && Ls < T;
    for (int q = 0; spans_ok && q < rec->nq; q++) {
        spans_ok = rec->q[q].span[0] >= Ls;
        for (int k = 0; spans_ok && k < rec->q[q].n_opt; k++) spans_ok = rec->q[q].opt_span[k][0] >= Ls;
    }
    /* nothing to keep, or a configuration the entry cannot serve: the plain path, entry untouched */
    if (!spans_ok || !clef_gpu_prefix_supported(e->gpu)) return clef_run_ex(e, rec, 1, out, raw, NULL, 0, err, errlen);
    for (int i = 0; i < T; i++) {
        const int32_t id = rec->ids.ids[i];
        if (id < 0 || id >= e->cfg.vocab) { snprintf(err, errlen, "token id %d out of range", id); return false; }
    }
    const int cls = clef_gpu_prefix_class(e, T);
    /* The leading tokens this record shares with the entry, up to both snapshots. */
    int same = 0;
    if (p->len > 0 && p->cls == cls) {
        const int n = p->len < Ls ? p->len : Ls;
        while (same < n && p->ids[same] == rec->ids.ids[same]) same++;
    }
    const bool left = p->len > 0 && p->cls == cls && same < p->len && same < Ls;   /* it leaves the entry's tokens midway */
    /* Resume from the last checkpoint inside the shared tokens. A later one describes tokens this
     * record does not have, and the pass overwrites the rows it belongs to. */
    clef_prefix_plan plan = { .load = -1 };
    int L = 0;
    for (int i = 0; i < CLEF_PREFIX_CKPT; i++) {
        if (p->ck_row[i] > same) p->ck_row[i] = 0;
        else if (p->ck_row[i] > L) { L = p->ck_row[i]; plan.load = i; }
    }
    /* The rows this pass stores, ascending: the periodic ones it crosses, the block where the record
     * left the entry's tokens, and its snapshot. At most CLEF_PREFIX_CKPT - 1, so a slot is left for
     * the checkpoint being read: the period doubles for a record too long for that. */
    if (multi) {
        int period = CLEF_PREFIX_PERIOD;
        while ((Ls - 1) / period > CLEF_PREFIX_CKPT - 3) period *= 2;
        const int at = same / 32 * 32;
        const bool anchor = left && at > L && at >= CLEF_PREFIX_MIN && at % period;
        for (int r = (L / period + 1) * period; r < Ls; r += period) {
            if (anchor && at < r && (plan.n == 0 || plan.row[plan.n - 1] < at)) plan.row[plan.n++] = at;
            plan.row[plan.n++] = r;
        }
        if (anchor && at < Ls && (plan.n == 0 || plan.row[plan.n - 1] < at)) plan.row[plan.n++] = at;
    }
    if (Ls > L) plan.row[plan.n++] = Ls;
    /* The pass rewrites the entry in place from row L. It holds nothing usable until it succeeds. */
    p->len = 0;
    /* A slot for each stored row: a free one, else the least recently used of the checkpoints that
     * stay valid. The counts above leave a slot for every row, so a miss is a bug here, not input. */
    bool taken[CLEF_PREFIX_CKPT] = { false };
    for (int i = 0; i < plan.n; i++) {
        int s = -1;
        for (int j = 0; j < CLEF_PREFIX_CKPT; j++) {
            if (j == plan.load || taken[j]) continue;
            if (p->ck_row[j] == 0) { s = j; break; }
            if (s < 0 || p->ck_use[j] < p->ck_use[s]) s = j;
        }
        if (s < 0) { snprintf(err, errlen, "prefix cache entry: no checkpoint slot for row %d", plan.row[i]); return false; }
        taken[s] = true;
        p->ck_row[s] = 0;
        plan.slot[i] = s;
    }
    int32_t *ids = realloc(p->ids, (size_t)Ls * sizeof(int32_t));
    if (!ids) { snprintf(err, errlen, "out of memory (prefix cache tokens)"); return false; }
    p->ids = ids;
    clef_head_inputs in = {0};
    bool ovf = false;
    if (!clef_gpu_forward_prefix(e->gpu, e, p->gpu, rec->ids.ids, T, L, &plan, &ovf, &in, err, errlen)) return false;
    /* FP16 overflow: the entry now holds a discarded pass. The plain path reruns the record in BF16. */
    if (ovf) return clef_run_ex(e, rec, 1, out, raw, NULL, 0, err, errlen);
    const double t_head = wall_ms();
    float ***probs = calloc(1, sizeof(*probs));
    bool ok = probs && (probs[0] = calloc((size_t)rec->nq, sizeof(float *))) != NULL;
    for (int q = 0; ok && q < rec->nq; q++) ok = (probs[0][q] = calloc((size_t)rec->q[q].n_opt, sizeof(float))) != NULL;
    if (!ok) snprintf(err, errlen, "out of memory (logits)");
    if (ok && !clef_head_run(e->head, &e->cfg, e->w.output, &in, 0, rec, probs[0])) {
        snprintf(err, errlen, "head failed (out of memory)");
        ok = false;
    } else if (ok && !logits_finite(rec, probs[0])) {
        snprintf(err, errlen, "record 0: non-finite logits");
        ok = false;
    }
    if (!ok) { clef_free_probs(rec, 1, probs); return false; }
    if (getenv("CLEF_STAGE_TIME")) fprintf(stderr, "clef: stage head n=1: %.2f ms\n", wall_ms() - t_head);
    for (int q = 0; !raw && q < rec->nq; q++) softmax_f32(probs[0][q], rec->q[q].n_opt);
    memcpy(p->ids, rec->ids.ids, (size_t)Ls * sizeof(int32_t));
    p->len = Ls;
    p->cls = cls;
    p->use++;
    if (plan.load >= 0) p->ck_use[plan.load] = p->use;
    for (int i = 0; i < plan.n; i++) {
        p->ck_row[plan.slot[i]] = plan.row[i];
        p->ck_use[plan.slot[i]] = p->use;
    }
    if (reused) *reused = L;
    *out = probs;
    return true;
}

/* The snapshot row clef_run_prefix uses for a record, 0 when it takes the plain path. */
static int snapshot_row(const clef_record *rec) {
    const int Ls = rec->schema_start > CLEF_PREFIX_MARGIN ? (rec->schema_start - CLEF_PREFIX_MARGIN) / 32 * 32 : 0;
    return Ls >= CLEF_PREFIX_MIN ? Ls : 0;
}

size_t clef_prefix_estimate(const clef_engine *e, const clef_prefix *p, const clef_record *rec) {
    if (!e || !p || !rec || !e->gpu) return 0;
    const int Ls = snapshot_row(rec);
    const int T = rec->ids.len <= (size_t)(1 << 20) ? (int)rec->ids.len : 0;
    /* The same plain-path conditions as run_entry: nothing would be allocated. */
    bool spans_ok = Ls > 0 && Ls < T;
    for (int q = 0; spans_ok && q < rec->nq; q++) {
        spans_ok = rec->q[q].span[0] >= Ls;
        for (int k = 0; spans_ok && k < rec->q[q].n_opt; k++) spans_ok = rec->q[q].opt_span[k][0] >= Ls;
    }
    if (!spans_ok || !clef_gpu_prefix_supported(e->gpu)) return 0;
    /* run_entry's resume point, read-only: checkpoints past the shared tokens would be invalidated. */
    const int cls = clef_gpu_prefix_class(e, T);
    int same = 0;
    if (p->len > 0 && p->cls == cls) {
        const int n = p->len < Ls ? p->len : Ls;
        while (same < n && p->ids[same] == rec->ids.ids[same]) same++;
    }
    const bool left = p->len > 0 && p->cls == cls && same < p->len && same < Ls;
    int L = 0;
    for (int i = 0; i < CLEF_PREFIX_CKPT; i++)
        if (p->ck_row[i] <= same && p->ck_row[i] > L) L = p->ck_row[i];
    /* The rows the pass would store, counted as run_entry enumerates them. */
    int period = CLEF_PREFIX_PERIOD, n = 0, last = 0;
    while ((Ls - 1) / period > CLEF_PREFIX_CKPT - 3) period *= 2;
    const int at = same / 32 * 32;
    const bool anchor = left && at > L && at >= CLEF_PREFIX_MIN && at % period;
    for (int r = (L / period + 1) * period; r < Ls; r += period) {
        if (anchor && at < r && (n == 0 || last < at)) { n++; last = at; }
        n++; last = r;
    }
    if (anchor && at < Ls && (n == 0 || last < at)) n++;
    if (Ls > L) n++;
    return clef_gpu_prefix_estimate(e->gpu, &e->cfg, p->gpu, T, n);
}

bool clef_run_prefix(clef_engine *e, clef_prefix *p, const clef_record *rec, float ****out, bool raw,
                     int *reused, char *err, size_t errlen) {
    /* The snapshot sits where the schema begins, on attention's 32-query-row boundary. A request
     * with fewer than 128 cacheable tokens takes the plain path and leaves the entry alone: every
     * request shares the template tokens, so it would otherwise "match" and then overwrite an
     * entry that took seconds to fill. clef_run_template is the entry for those tokens. */
    return run_entry(e, p, rec, snapshot_row(rec), true, out, raw, reused, err, errlen);
}

bool clef_run_template(clef_engine *e, clef_prefix *p, const clef_record *rec, float ****out, bool raw,
                       int *reused, char *err, size_t errlen) {
    const size_t T = rec->ids.len, rem = T % 64;
    bool fits = T <= CLEF_TEMPLATE_MAX && rec->schema_start >= CLEF_TEMPLATE_TOKENS;
    /* Flash's selected short GEMM uses 64-row tiles at these lengths; removing 32 rows
     * switches to its slower 32-row tile. Bypass preserves the ordinary dispatch and entry. */
    if (e && e->cfg.H == 4096 && T >= 768 && T <= 1024 && (rem == 0 || rem > 32)) fits = false;
    /* Above 1,056, both full and suffix GEMMs use 64-row tiles. Reuse pays off when it
     * removes a padded tile; otherwise the measured bookkeeping cost cancels the saving. */
    if (T > 1056 && (rem == 0 || rem > 32)) fits = false;
    return run_entry(e, p, rec, fits ? CLEF_TEMPLATE_TOKENS : 0, false, out, raw, reused, err, errlen);
}

bool clef_keep_warm(clef_engine *e, char *err, size_t errlen) {
    if (!e) { snprintf(err, errlen, "invalid engine"); return false; }
    return clef_gpu_keepalive(e->gpu, err, errlen);
}

bool clef_run(clef_engine *e, const clef_record *recs, int n, float ****probs, char *err, size_t errlen) {
    return clef_run_ex(e, recs, n, probs, false, NULL, 0, err, errlen);
}
