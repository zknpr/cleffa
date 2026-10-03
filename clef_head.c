/* Joint schema head (JointSchemaHead in joint_schema_model.py), on the CPU in f32.
 *
 * Everything here is small (a handful of option/question rows) except attention over
 * the T memory rows; the memory-side K/V projections (T x W x 2W per module) are done
 * on the GPU and passed in without bias:
 *   - the K bias adds q.b_k to every score of a query: constant across keys, cancelled
 *     by the softmax, so it is skipped exactly;
 *   - the V bias passes through unchanged because attention weights sum to 1.
 * Matmuls go through Accelerate's cblas_sgemm. */

#include <Accelerate/Accelerate.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "clef_engine.h"

typedef struct { float *w, *b; int out, in; } linear;
typedef struct { float *w, *b; int n; } layernorm;
typedef struct { linear in_q; float *in_bv; linear out; int E, heads; } mha;  /* in_q: rows [0,E) */

typedef struct {
    layernorm query_norm, memory_norm, ff_norm;
    mha attn;
    linear ff1, ff2;
} evidence_layer;

typedef struct {
    layernorm norm1, norm2, norm3;
    mha self_attn;          /* full in_proj (q, k, v) for self-attention among fields */
    linear self_in;         /* [3E][E] */
    mha cross;
    linear lin1, lin2;
} decoder_layer;

struct clef_head {
    int H, W;
    layernorm hidden_norm, option_summary_norm, field_norm, option_norm;
    linear memory_projection, question_projection, option_question_projection, global_projection;
    linear option_context_projection, option_lexical_projection;
    float *type_embedding;  /* [3][W] */
    evidence_layer ev[8];
    int n_ev;
    decoder_layer dec[16];
    int n_dec;
    linear scorer0, scorer3;
    float prior_logit_scale, joint_logit_scale, residual_gate;
};

static float bf16_to_f32(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

static float *load_f32(const gguf_file *f, const char *name, int64_t expect, char *err, size_t errlen) {
    const gguf_tensor *t = gguf_find_tensor(f, name);
    if (!t) { snprintf(err, errlen, "head: missing tensor %s", name); return NULL; }
    int64_t n = 1;
    for (uint32_t d = 0; d < t->n_dims; d++) n *= (int64_t)t->ne[d];
    if (expect >= 0 && n != expect) { snprintf(err, errlen, "head: %s has %lld elements, expected %lld", name, (long long)n, (long long)expect); return NULL; }
    float *out = malloc((size_t)n * sizeof(float));
    if (!out) { snprintf(err, errlen, "head: out of memory"); return NULL; }
    if (t->type == GGML_BF16) {
        const uint16_t *s = t->data;
        for (int64_t i = 0; i < n; i++) out[i] = bf16_to_f32(s[i]);
    } else if (t->type == GGML_F32) {
        memcpy(out, t->data, (size_t)n * sizeof(float));
    } else {
        free(out);
        snprintf(err, errlen, "head: %s has unsupported type", name);
        return NULL;
    }
    return out;
}

#define LOAD(dst, name, n) do { if (!((dst) = load_f32(f, (name), (n), err, errlen))) return false; } while (0)

static bool load_linear(const gguf_file *f, const char *prefix, int out, int in, bool bias, linear *l, char *err, size_t errlen) {
    char name[160];
    l->out = out; l->in = in; l->b = NULL;
    snprintf(name, sizeof(name), "%s.weight", prefix);
    LOAD(l->w, name, (int64_t)out * in);
    if (bias) { snprintf(name, sizeof(name), "%s.bias", prefix); LOAD(l->b, name, out); }
    return true;
}

static bool load_ln(const gguf_file *f, const char *prefix, int n, layernorm *l, char *err, size_t errlen) {
    char name[160];
    l->n = n;
    snprintf(name, sizeof(name), "%s.weight", prefix);
    LOAD(l->w, name, n);
    snprintf(name, sizeof(name), "%s.bias", prefix);
    LOAD(l->b, name, n);
    return true;
}

static bool load_mha(const gguf_file *f, const char *prefix, int E, int heads, mha *m, char *err, size_t errlen) {
    char name[160];
    float *w, *b;
    m->E = E; m->heads = heads;
    snprintf(name, sizeof(name), "%s.in_proj_weight", prefix);
    LOAD(w, name, (int64_t)3 * E * E);
    snprintf(name, sizeof(name), "%s.in_proj_bias", prefix);
    LOAD(b, name, 3 * E);
    m->in_q = (linear){ w, b, E, E };   /* rows [0,E): query projection; K/V rows used on the GPU */
    m->in_bv = b + 2 * E;
    snprintf(name, sizeof(name), "%s.out_proj", prefix);
    return load_linear(f, name, E, E, true, &m->out, err, errlen);
}

static float scalar(const gguf_file *f, const char *name, bool *ok, char *err, size_t errlen) {
    float *p = load_f32(f, name, 1, err, errlen);
    if (!p) { *ok = false; return 0; }
    float v = *p;
    free(p);
    return v;
}

clef_head *clef_head_load(const gguf_file *f, const clef_config *cfg, char *err, size_t errlen) {
    clef_head *h = calloc(1, sizeof(*h));
    if (!h) { snprintf(err, errlen, "head: out of memory"); return NULL; }
    const int H = cfg->H, W = cfg->W, FF = cfg->head_ff;
    h->H = H; h->W = W;
    h->n_ev = cfg->routing_layers;
    h->n_dec = cfg->head_layers;
    if (h->n_ev > 8 || h->n_dec > 16 || W % cfg->head_heads) { snprintf(err, errlen, "head: unsupported shape"); free(h); return NULL; }
    bool ok = load_ln(f, "head.hidden_norm", H, &h->hidden_norm, err, errlen)
        && load_ln(f, "head.option_summary_norm", W, &h->option_summary_norm, err, errlen)
        && load_ln(f, "head.field_norm", W, &h->field_norm, err, errlen)
        && load_ln(f, "head.option_norm", W, &h->option_norm, err, errlen)
        && load_linear(f, "head.memory_projection", W, H, false, &h->memory_projection, err, errlen)
        && load_linear(f, "head.question_projection", W, H, false, &h->question_projection, err, errlen)
        && load_linear(f, "head.option_question_projection", W, H, false, &h->option_question_projection, err, errlen)
        && load_linear(f, "head.global_projection", W, H, false, &h->global_projection, err, errlen)
        && load_linear(f, "head.option_context_projection", W, H, false, &h->option_context_projection, err, errlen)
        && load_linear(f, "head.option_lexical_projection", W, H, false, &h->option_lexical_projection, err, errlen)
        && load_linear(f, "head.residual_scorer.0", W, 4 * W, true, &h->scorer0, err, errlen)
        && load_linear(f, "head.residual_scorer.3", 1, W, true, &h->scorer3, err, errlen);
    if (ok) ok = (h->type_embedding = load_f32(f, "head.type_embedding.weight", 3 * W, err, errlen)) != NULL;
    char p[128];
    for (int i = 0; ok && i < h->n_ev; i++) {
        evidence_layer *e = &h->ev[i];
        snprintf(p, sizeof(p), "head.evidence_layers.%d.query_norm", i); ok = load_ln(f, p, W, &e->query_norm, err, errlen);
        snprintf(p, sizeof(p), "head.evidence_layers.%d.memory_norm", i); ok = ok && load_ln(f, p, W, &e->memory_norm, err, errlen);
        snprintf(p, sizeof(p), "head.evidence_layers.%d.feedforward_norm", i); ok = ok && load_ln(f, p, W, &e->ff_norm, err, errlen);
        snprintf(p, sizeof(p), "head.evidence_layers.%d.attention", i); ok = ok && load_mha(f, p, W, cfg->head_heads, &e->attn, err, errlen);
        snprintf(p, sizeof(p), "head.evidence_layers.%d.feedforward.0", i); ok = ok && load_linear(f, p, FF, W, true, &e->ff1, err, errlen);
        snprintf(p, sizeof(p), "head.evidence_layers.%d.feedforward.3", i); ok = ok && load_linear(f, p, W, FF, true, &e->ff2, err, errlen);
    }
    for (int i = 0; ok && i < h->n_dec; i++) {
        decoder_layer *d = &h->dec[i];
        snprintf(p, sizeof(p), "head.layers.%d.norm1", i); ok = load_ln(f, p, W, &d->norm1, err, errlen);
        snprintf(p, sizeof(p), "head.layers.%d.norm2", i); ok = ok && load_ln(f, p, W, &d->norm2, err, errlen);
        snprintf(p, sizeof(p), "head.layers.%d.norm3", i); ok = ok && load_ln(f, p, W, &d->norm3, err, errlen);
        snprintf(p, sizeof(p), "head.layers.%d.self_attn", i); ok = ok && load_mha(f, p, W, cfg->head_heads, &d->self_attn, err, errlen);
        if (ok) d->self_in = (linear){ d->self_attn.in_q.w, d->self_attn.in_q.b, 3 * W, W };
        snprintf(p, sizeof(p), "head.layers.%d.multihead_attn", i); ok = ok && load_mha(f, p, W, cfg->head_heads, &d->cross, err, errlen);
        snprintf(p, sizeof(p), "head.layers.%d.linear1", i); ok = ok && load_linear(f, p, FF, W, true, &d->lin1, err, errlen);
        snprintf(p, sizeof(p), "head.layers.%d.linear2", i); ok = ok && load_linear(f, p, W, FF, true, &d->lin2, err, errlen);
    }
    if (ok) h->prior_logit_scale = scalar(f, "head.prior_logit_scale", &ok, err, errlen);
    if (ok) h->joint_logit_scale = scalar(f, "head.joint_logit_scale", &ok, err, errlen);
    if (ok) h->residual_gate = scalar(f, "head.residual_gate", &ok, err, errlen);
    if (!ok) { clef_head_free(h); return NULL; }
    return h;
}

static void free_linear(linear *l) { free(l->w); free(l->b); }
static void free_ln(layernorm *l) { free(l->w); free(l->b); }

void clef_head_free(clef_head *h) {
    if (!h) return;
    free_ln(&h->hidden_norm); free_ln(&h->option_summary_norm); free_ln(&h->field_norm); free_ln(&h->option_norm);
    free_linear(&h->memory_projection); free_linear(&h->question_projection);
    free_linear(&h->option_question_projection); free_linear(&h->global_projection);
    free_linear(&h->option_context_projection); free_linear(&h->option_lexical_projection);
    free_linear(&h->scorer0); free_linear(&h->scorer3);
    free(h->type_embedding);
    for (int i = 0; i < h->n_ev; i++) {
        evidence_layer *e = &h->ev[i];
        free_ln(&e->query_norm); free_ln(&e->memory_norm); free_ln(&e->ff_norm);
        free(e->attn.in_q.w); free(e->attn.in_q.b); free_linear(&e->attn.out);
        free_linear(&e->ff1); free_linear(&e->ff2);
    }
    for (int i = 0; i < h->n_dec; i++) {
        decoder_layer *d = &h->dec[i];
        free_ln(&d->norm1); free_ln(&d->norm2); free_ln(&d->norm3);
        free(d->self_attn.in_q.w); free(d->self_attn.in_q.b); free_linear(&d->self_attn.out);
        free(d->cross.in_q.w); free(d->cross.in_q.b); free_linear(&d->cross.out);
        free_linear(&d->lin1); free_linear(&d->lin2);
    }
    free(h);
}

/* ---- primitives -------------------------------------------------------------- */

/* Y[n][out] = X[n][in] . W^T + b */
static void lin(const linear *l, const float *X, int n, float *Y) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, l->out, l->in, 1.0f, X, l->in, l->w, l->in, 0.0f, Y, l->out);
    if (l->b) for (int r = 0; r < n; r++) for (int j = 0; j < l->out; j++) Y[(size_t)r * l->out + j] += l->b[j];
}

static void ln_rows(const layernorm *l, const float *X, int n, float *Y) {
    for (int r = 0; r < n; r++) {
        const float *x = X + (size_t)r * l->n;
        float *y = Y + (size_t)r * l->n;
        double mean = 0, var = 0;
        for (int i = 0; i < l->n; i++) mean += x[i];
        mean /= l->n;
        for (int i = 0; i < l->n; i++) { double d = x[i] - mean; var += d * d; }
        var /= l->n;
        const float inv = (float)(1.0 / sqrt(var + 1e-5));
        for (int i = 0; i < l->n; i++) y[i] = (float)(x[i] - mean) * inv * l->w[i] + l->b[i];
    }
}

static float gelu(float x) { return 0.5f * x * (1.0f + erff(x * 0.70710678118654752f)); }

static void softmax_inplace(float *x, int n) {
    float m = -INFINITY;
    for (int i = 0; i < n; i++) if (x[i] > m) m = x[i];
    double s = 0;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - m); s += x[i]; }
    const float inv = (float)(1.0 / s);
    for (int i = 0; i < n; i++) x[i] *= inv;
}

/* Multi-head attention of nq query rows (already projected, bias included) over L
 * key/value rows. Q row stride ldq, K/V row stride ld; bv (may be NULL when V already carries its bias)
 * is added after the weighted sum, which is exact because the weights sum to 1. Out: [nq][E]. */
static bool attend(const float *Qp, int ldq, int nq, const float *K, const float *V, int ld, int L,
                   const float *bv, int E, int heads, float *out) {
    const int hd = E / heads;
    float *S = malloc((size_t)nq * L * sizeof(float));
    if (!S) return false;
    const float scale = 1.0f / sqrtf((float)hd);
    for (int h = 0; h < heads; h++) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, nq, L, hd, scale,
                    Qp + h * hd, ldq, K + h * hd, ld, 0.0f, S, L);
        for (int r = 0; r < nq; r++) softmax_inplace(S + (size_t)r * L, L);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nq, hd, L, 1.0f,
                    S, L, V + h * hd, ld, 0.0f, out + h * hd, E);
    }
    if (bv) for (int r = 0; r < nq; r++) for (int j = 0; j < E; j++) out[(size_t)r * E + j] += bv[j];
    free(S);
    return true;
}

static void add_rows(float *y, const float *x, size_t n) { for (size_t i = 0; i < n; i++) y[i] += x[i]; }

static void l2_normalize(const float *x, int n, float *y) {
    double s = 0;
    for (int i = 0; i < n; i++) s += (double)x[i] * x[i];
    const double nrm = sqrt(s);
    const float inv = (float)(1.0 / (nrm > 1e-12 ? nrm : 1e-12));  /* F.normalize eps */
    for (int i = 0; i < n; i++) y[i] = x[i] * inv;
}

/* ---- forward ---------------------------------------------------------------- */

bool clef_head_run(const clef_head *h, const clef_config *cfg, const gguf_tensor *lm_head,
                   const clef_head_inputs *in, int base, const clef_record *rec, float **logits) {
    const int H = h->H, W = h->W, heads = cfg->head_heads, nq = rec->nq;
    const int L = (int)rec->ids.len;
    const float *nh = in->nh + (size_t)base * H;
    int n_opt = 0;
    for (int q = 0; q < nq; q++) n_opt += rec->q[q].n_opt;

    size_t floats = (size_t)nq * H * 2 + (size_t)n_opt * H * 2 + (size_t)n_opt * W * 8 + (size_t)nq * W * 16
                    + (size_t)(n_opt > nq ? n_opt : nq) * (cfg->head_ff + 4 * W) + (size_t)H + 64;
    float *mem = calloc(floats, sizeof(float));
    if (!mem) return false;
    float *p = mem;
    float *qv = p; p += (size_t)nq * H;          /* question vectors */
    float *ctx = p; p += (size_t)n_opt * H;      /* option context vectors */
    float *lex = p; p += (size_t)n_opt * H;      /* option lexical vectors */
    float *tmp = p; p += (size_t)n_opt * W;
    float *routed = p; p += (size_t)n_opt * W;
    float *nrm = p; p += (size_t)n_opt * W;
    float *qp = p; p += (size_t)n_opt * W;
    float *att = p; p += (size_t)n_opt * W;
    float *opt_n = p; p += (size_t)n_opt * W;
    float *opt_q = p; p += (size_t)n_opt * W;
    p += (size_t)n_opt * W;
    float *fields = p; p += (size_t)nq * W;
    float *base_f = p; p += (size_t)nq * W;
    float *summ = p; p += (size_t)nq * W;
    float *fq = p; p += (size_t)nq * W * 3;
    float *fatt = p; p += (size_t)nq * W;
    float *fn = p; p += (size_t)nq * W;
    float *gp = p; p += (size_t)W;
    p += (size_t)nq * W * 6;
    float *ffbuf = p; p += (size_t)(n_opt > nq ? n_opt : nq) * cfg->head_ff;
    float *feat = p;
    const float *global = nh + (size_t)(L - 1) * H;

    /* span means and lexical means */
    int o = 0;
    const uint16_t *lm = lm_head->data;
    for (int q = 0; q < nq; q++) {
        const clef_question *cq = &rec->q[q];
        for (int t = cq->span[0]; t < cq->span[1]; t++) add_rows(qv + (size_t)q * H, nh + (size_t)t * H, H);
        const float inv = 1.0f / (float)(cq->span[1] - cq->span[0]);
        for (int i = 0; i < H; i++) qv[(size_t)q * H + i] *= inv;
        for (int k = 0; k < cq->n_opt; k++, o++) {
            const int s0 = cq->opt_span[k][0], s1 = cq->opt_span[k][1];
            float *c = ctx + (size_t)o * H, *x = lex + (size_t)o * H;
            for (int t = s0; t < s1; t++) {
                add_rows(c, nh + (size_t)t * H, H);
                const uint16_t *row = lm + (size_t)rec->ids.ids[t] * H;
                for (int i = 0; i < H; i++) x[i] += bf16_to_f32(row[i]);
            }
            const float iv = 1.0f / (float)(s1 - s0);
            for (int i = 0; i < H; i++) { c[i] *= iv; x[i] *= iv; }
        }
    }

    /* option queries */
    lin(&h->option_context_projection, ctx, n_opt, routed);
    lin(&h->option_lexical_projection, lex, n_opt, tmp);
    add_rows(routed, tmp, (size_t)n_opt * W);
    o = 0;
    float *qq = malloc((size_t)nq * W * sizeof(float));
    if (!qq) { free(mem); return false; }
    lin(&h->option_question_projection, qv, nq, qq);
    for (int q = 0; q < nq; q++)
        for (int k = 0; k < rec->q[q].n_opt; k++, o++) add_rows(routed + (size_t)o * W, qq + (size_t)q * W, W);
    free(qq);

    /* evidence routing layers: options attend over memory */
    for (int l = 0; l < h->n_ev; l++) {
        const evidence_layer *e = &h->ev[l];
        const float *kv = in->kv[l] + (size_t)base * 2 * W;
        ln_rows(&e->query_norm, routed, n_opt, nrm);
        lin(&e->attn.in_q, nrm, n_opt, qp);
        if (!attend(qp, W, n_opt, kv, kv + W, 2 * W, L, e->attn.in_bv, W, heads, att)) { free(mem); return false; }
        lin(&e->attn.out, att, n_opt, tmp);
        add_rows(routed, tmp, (size_t)n_opt * W);
        ln_rows(&e->ff_norm, routed, n_opt, nrm);
        lin(&e->ff1, nrm, n_opt, ffbuf);
        for (size_t i = 0; i < (size_t)n_opt * cfg->head_ff; i++) ffbuf[i] = gelu(ffbuf[i]);
        lin(&e->ff2, ffbuf, n_opt, tmp);
        add_rows(routed, tmp, (size_t)n_opt * W);
    }

    /* fields */
    lin(&h->question_projection, qv, nq, base_f);
    lin(&h->global_projection, global, 1, gp);
    o = 0;
    for (int q = 0; q < nq; q++) {
        const int n = rec->q[q].n_opt;
        const float *opts = routed + (size_t)o * W, *field = base_f + (size_t)q * W;
        float *wts = malloc((size_t)n * sizeof(float));
        if (!wts) { free(mem); return false; }
        for (int k = 0; k < n; k++) {
            double d = 0;
            for (int i = 0; i < W; i++) d += (double)opts[(size_t)k * W + i] * field[i];
            wts[k] = (float)(d / sqrt((double)W));
        }
        softmax_inplace(wts, n);
        for (int k = 0; k < n; k++) for (int i = 0; i < W; i++) summ[(size_t)q * W + i] += wts[k] * opts[(size_t)k * W + i];
        free(wts);
        o += n;
    }
    ln_rows(&h->option_summary_norm, summ, nq, fn);
    for (int q = 0; q < nq; q++) {
        float *f = fields + (size_t)q * W;
        const float *te = h->type_embedding + (size_t)rec->q[q].type * W;
        for (int i = 0; i < W; i++) f[i] = base_f[(size_t)q * W + i] + fn[(size_t)q * W + i] + gp[i] + te[i];
    }

    /* transformer decoder layers (norm_first) over the fields */
    for (int l = 0; l < h->n_dec; l++) {
        const decoder_layer *d = &h->dec[l];
        ln_rows(&d->norm1, fields, nq, fn);
        lin(&d->self_in, fn, nq, fq);   /* [nq][3W]: q | k | v with biases */
        /* self-attention: Q, K and V rows interleave in the fused [nq][3W] projection (biases included) */
        if (!attend(fq, 3 * W, nq, fq + W, fq + 2 * W, 3 * W, nq, NULL, W, heads, fatt)) { free(mem); return false; }
        lin(&d->self_attn.out, fatt, nq, fn);
        add_rows(fields, fn, (size_t)nq * W);

        ln_rows(&d->norm2, fields, nq, fn);
        lin(&d->cross.in_q, fn, nq, fq);
        const float *kv = in->kv[h->n_ev + l] + (size_t)base * 2 * W;
        if (!attend(fq, W, nq, kv, kv + W, 2 * W, L, d->cross.in_bv, W, heads, fatt)) { free(mem); return false; }
        lin(&d->cross.out, fatt, nq, fn);
        add_rows(fields, fn, (size_t)nq * W);

        ln_rows(&d->norm3, fields, nq, fn);
        lin(&d->lin1, fn, nq, ffbuf);
        for (size_t i = 0; i < (size_t)nq * cfg->head_ff; i++) ffbuf[i] = gelu(ffbuf[i]);
        lin(&d->lin2, ffbuf, nq, fn);
        add_rows(fields, fn, (size_t)nq * W);
    }
    ln_rows(&h->field_norm, fields, nq, fn);

    /* scoring */
    const float prior_scale = expf(fminf(h->prior_logit_scale, logf(100.0f)));
    const float joint_scale = expf(fminf(h->joint_logit_scale, logf(100.0f)));
    const float gate = 1.0f / (1.0f + expf(-h->residual_gate));
    ln_rows(&h->option_norm, routed, n_opt, opt_n);
    o = 0;
    for (int q = 0; q < nq; q++) {
        const int n = rec->q[q].n_opt;
        const float *field = fn + (size_t)q * W;
        float *anchor = malloc((size_t)H * 2 * sizeof(float));
        if (!anchor) { free(mem); return false; }
        float *lexn = anchor + H;
        for (int i = 0; i < H; i++) anchor[i] = qv[(size_t)q * H + i] + global[i];
        l2_normalize(anchor, H, anchor);
        double fnorm = 0;
        for (int i = 0; i < W; i++) fnorm += (double)field[i] * field[i];
        fnorm = sqrt(fnorm);
        for (int k = 0; k < n; k++, o++) {
            const float *opt = opt_n + (size_t)o * W;
            l2_normalize(lex + (size_t)o * H, H, lexn);
            double prior = 0;
            for (int i = 0; i < H; i++) prior += (double)lexn[i] * anchor[i];
            double dot = 0, onorm = 0;
            for (int i = 0; i < W; i++) { dot += (double)field[i] * opt[i]; onorm += (double)opt[i] * opt[i]; }
            const double denom = fnorm * sqrt(onorm);
            const double cosine = dot / (denom > 1e-8 ? denom : 1e-8);
            for (int i = 0; i < W; i++) {
                feat[i] = field[i];
                feat[W + i] = opt[i];
                feat[2 * W + i] = field[i] * opt[i];
                feat[3 * W + i] = fabsf(field[i] - opt[i]);
            }
            lin(&h->scorer0, feat, 1, opt_q);
            for (int i = 0; i < W; i++) opt_q[i] = gelu(opt_q[i]);
            float residual;
            lin(&h->scorer3, opt_q, 1, &residual);
            const double joint = joint_scale * cosine + residual;
            logits[q][k] = (float)(prior_scale * prior + gate * joint);
        }
        free(anchor);
    }
    free(mem);
    return true;
}
