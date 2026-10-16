/* CPU-only loader regression: retain every live weight bit, omit GPU-only copies,
 * and release all partially loaded tensors. Include the implementation to inspect
 * ownership; no Metal engine is opened. Optional GGUF arguments benchmark loading. */
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static struct { void *p; size_t n; } allocations[1024];
static size_t calls, fail_at, live, peak;

static void *tracked_alloc(size_t n, int zero) {
    if (++calls == fail_at) return NULL;
    void *p = zero ? calloc(1, n) : malloc(n);
    if (!p) return NULL;
    for (size_t i = 0; i < 1024; i++) if (!allocations[i].p) {
        allocations[i].p = p; allocations[i].n = n;
        live += n; if (live > peak) peak = live;
        return p;
    }
    abort();
}
static void *tracked_malloc(size_t n) { return tracked_alloc(n, 0); }
static void *tracked_calloc(size_t n, size_t size) { return tracked_alloc(n * size, 1); }
static void tracked_free(void *p) {
    if (!p) return;
    for (size_t i = 0; i < 1024; i++) if (allocations[i].p == p) {
        live -= allocations[i].n; allocations[i].p = NULL; free(p); return;
    }
    abort();
}
static size_t allocation_size(const void *p) {
    for (size_t i = 0; i < 1024; i++) if (allocations[i].p == p) return allocations[i].n;
    abort();
}

#define malloc tracked_malloc
#define calloc tracked_calloc
#define free tracked_free
#ifndef HEAD_SOURCE
#define HEAD_SOURCE "../clef_head.c"
#endif
#include HEAD_SOURCE
#undef malloc
#undef calloc
#undef free

static float sample(size_t i) { return ((int)(i * 37 % 101) - 50) / 256.0f; }
static void add_tensor(gguf_file *f, const char *name, size_t n, uint32_t type) {
    gguf_tensor *t = &f->tensors[f->n_tensors++];
    t->name = (gguf_str){strdup(name), strlen(name)};
    t->n_dims = 1; t->ne[0] = n; t->type = type;
    t->nbytes = n * (type == GGML_BF16 ? 2 : 4);
    void *data = malloc(t->nbytes); assert(data); t->data = data;
    for (size_t i = 0; i < n; i++) {
        /* F32 fixtures must contain bits beyond BF16 precision. */
        float v = sample(i) + 0.000001f; uint32_t bits; memcpy(&bits, &v, 4);
        if (type == GGML_BF16) ((uint16_t *)data)[i] = (uint16_t)(bits >> 16);
        else ((float *)data)[i] = v;
    }
}
static void add_pair(gguf_file *f, const char *prefix, int out, int in, int bias, uint32_t type) {
    char name[160]; snprintf(name, sizeof(name), "%s.weight", prefix);
    add_tensor(f, name, (size_t)out * in, type);
    if (bias) { snprintf(name, sizeof(name), "%s.bias", prefix); add_tensor(f, name, out, type); }
}
static void add_attn(gguf_file *f, const char *prefix, int W, uint32_t type) {
    char name[160]; snprintf(name, sizeof(name), "%s.in_proj_weight", prefix);
    add_tensor(f, name, (size_t)3 * W * W, type);
    snprintf(name, sizeof(name), "%s.in_proj_bias", prefix); add_tensor(f, name, 3 * W, type);
    snprintf(name, sizeof(name), "%s.out_proj", prefix); add_pair(f, name, W, W, 1, type);
}
static gguf_file fixture(const clef_config *c, uint32_t type) {
    gguf_file f = {.tensors = calloc(256, sizeof(gguf_tensor))}; assert(f.tensors);
    const char *norms[] = {"hidden_norm", "option_summary_norm", "field_norm", "option_norm"};
    const char *projs[] = {"memory_projection", "question_projection", "option_question_projection",
                          "global_projection", "option_context_projection", "option_lexical_projection"};
    char p[160];
    for (int i = 0; i < 4; i++) {
        snprintf(p, sizeof(p), "head.%s", norms[i]); add_pair(&f, p, i ? c->W : c->H, 1, 1, type);
    }
    for (int i = 0; i < 6; i++) {
        snprintf(p, sizeof(p), "head.%s", projs[i]); add_pair(&f, p, c->W, c->H, 0, type);
    }
    add_pair(&f, "head.residual_scorer.0", c->W, 4 * c->W, 1, type);
    add_pair(&f, "head.residual_scorer.3", 1, c->W, 1, type);
    add_tensor(&f, "head.type_embedding.weight", 3 * c->W, type);
    for (int i = 0; i < c->routing_layers; i++) {
        const char *ln[] = {"query_norm", "memory_norm", "feedforward_norm"};
        for (int j = 0; j < 3; j++) {
            snprintf(p, sizeof(p), "head.evidence_layers.%d.%s", i, ln[j]); add_pair(&f, p, c->W, 1, 1, type);
        }
        snprintf(p, sizeof(p), "head.evidence_layers.%d.attention", i); add_attn(&f, p, c->W, type);
        snprintf(p, sizeof(p), "head.evidence_layers.%d.feedforward.0", i); add_pair(&f, p, c->head_ff, c->W, 1, type);
        snprintf(p, sizeof(p), "head.evidence_layers.%d.feedforward.3", i); add_pair(&f, p, c->W, c->head_ff, 1, type);
    }
    for (int i = 0; i < c->head_layers; i++) {
        for (int j = 1; j <= 3; j++) {
            snprintf(p, sizeof(p), "head.layers.%d.norm%d", i, j); add_pair(&f, p, c->W, 1, 1, type);
        }
        snprintf(p, sizeof(p), "head.layers.%d.self_attn", i); add_attn(&f, p, c->W, type);
        snprintf(p, sizeof(p), "head.layers.%d.multihead_attn", i); add_attn(&f, p, c->W, type);
        snprintf(p, sizeof(p), "head.layers.%d.linear1", i); add_pair(&f, p, c->head_ff, c->W, 1, type);
        snprintf(p, sizeof(p), "head.layers.%d.linear2", i); add_pair(&f, p, c->W, c->head_ff, 1, type);
    }
    add_tensor(&f, "head.prior_logit_scale", 1, type);
    add_tensor(&f, "head.joint_logit_scale", 1, type);
    add_tensor(&f, "head.residual_gate", 1, type);
    return f;
}

static uint64_t hash = 1469598103934665603ull;
static size_t checked_bytes;
static void hash_float(float v) {
    uint32_t bits; memcpy(&bits, &v, 4); hash ^= bits; hash *= 1099511628211ull;
}
static void check_data(const gguf_file *f, const char *name, const float *data, size_t n) {
    const gguf_tensor *t = gguf_find_tensor(f, name); assert(t);
    for (size_t i = 0; i < 1024; i++) if (allocations[i].p == data) checked_bytes += allocations[i].n;
    for (size_t i = 0; i < n; i++) {
        float want;
        if (t->type == GGML_F32) memcpy(&want, (const float *)t->data + i, 4);
        else { uint32_t bits = (uint32_t)((const uint16_t *)t->data)[i] << 16; memcpy(&want, &bits, 4); }
        assert(memcmp(&want, data + i, 4) == 0); hash_float(data[i]);
    }
}
static void check_packed(const linear *l) {
    if (l->packed) {
        checked_bytes += allocation_size(l->packed);
        for (int i = 0; i < l->in; i++) for (int j = 0; j < l->packed_ld; j++) {
            float want = j < l->out ? l->w[(size_t)j * l->in + i] : 0;
            assert(memcmp(&want, l->packed + (size_t)i * l->packed_ld + j, 4) == 0);
        }
    }
}
static void check_linear(const gguf_file *f, const char *prefix, const linear *l) {
    char name[160]; snprintf(name, sizeof(name), "%s.weight", prefix);
    check_data(f, name, l->w, (size_t)l->in * l->out);
    if (l->b) { snprintf(name, sizeof(name), "%s.bias", prefix); check_data(f, name, l->b, l->out); }
    check_packed(l);
}
static void check_ln(const gguf_file *f, const char *prefix, const layernorm *l) {
    char name[160]; snprintf(name, sizeof(name), "%s.weight", prefix); check_data(f, name, l->w, l->n);
    snprintf(name, sizeof(name), "%s.bias", prefix); check_data(f, name, l->b, l->n);
}
static void check_attn(const gguf_file *f, const char *prefix, const mha *m, int self, int compact) {
    char name[160]; size_t n = (size_t)(self ? 3 : 1) * m->E * m->E;
    snprintf(name, sizeof(name), "%s.in_proj_weight", prefix); check_data(f, name, m->in_q.w, n);
    snprintf(name, sizeof(name), "%s.in_proj_bias", prefix); check_data(f, name, m->in_q.b, 3 * m->E);
    assert(m->in_bv == m->in_q.b + 2 * m->E);
    if (compact) assert(allocation_size(m->in_q.w) == n * sizeof(float));
    check_packed(&m->in_q);
    snprintf(name, sizeof(name), "%s.out_proj", prefix); check_linear(f, name, &m->out);
}
static void check_head(const gguf_file *f, const clef_head *h, int compact) {
    checked_bytes = sizeof(*h);
    const linear *ls[] = {&h->question_projection, &h->option_question_projection, &h->global_projection,
                         &h->option_context_projection, &h->option_lexical_projection, &h->scorer0, &h->scorer3};
    const char *names[] = {"question_projection", "option_question_projection", "global_projection",
                          "option_context_projection", "option_lexical_projection", "residual_scorer.0", "residual_scorer.3"};
    char p[160];
    for (int i = 0; i < 7; i++) { snprintf(p, sizeof(p), "head.%s", names[i]); check_linear(f, p, ls[i]); }
    check_ln(f, "head.option_summary_norm", &h->option_summary_norm);
    check_ln(f, "head.field_norm", &h->field_norm); check_ln(f, "head.option_norm", &h->option_norm);
    check_data(f, "head.type_embedding.weight", h->type_embedding, 3 * h->W);
    check_data(f, "head.prior_logit_scale", &h->prior_logit_scale, 1);
    check_data(f, "head.joint_logit_scale", &h->joint_logit_scale, 1);
    check_data(f, "head.residual_gate", &h->residual_gate, 1);
    for (int i = 0; i < h->n_ev; i++) {
        const evidence_layer *e = &h->ev[i];
        snprintf(p, sizeof(p), "head.evidence_layers.%d.query_norm", i); check_ln(f, p, &e->query_norm);
        snprintf(p, sizeof(p), "head.evidence_layers.%d.feedforward_norm", i); check_ln(f, p, &e->ff_norm);
        snprintf(p, sizeof(p), "head.evidence_layers.%d.attention", i); check_attn(f, p, &e->attn, 0, compact);
        snprintf(p, sizeof(p), "head.evidence_layers.%d.feedforward.0", i); check_linear(f, p, &e->ff1);
        snprintf(p, sizeof(p), "head.evidence_layers.%d.feedforward.3", i); check_linear(f, p, &e->ff2);
    }
    for (int i = 0; i < h->n_dec; i++) {
        const decoder_layer *d = &h->dec[i];
        const layernorm *norms[] = {&d->norm1, &d->norm2, &d->norm3};
        for (int j = 0; j < 3; j++) { snprintf(p, sizeof(p), "head.layers.%d.norm%d", i, j + 1); check_ln(f, p, norms[j]); }
        snprintf(p, sizeof(p), "head.layers.%d.self_attn", i); check_attn(f, p, &d->self_attn, 1, compact);
        assert(d->self_in.w == d->self_attn.in_q.w && d->self_in.b == d->self_attn.in_q.b && d->self_in.out == 3 * h->W);
        check_packed(&d->self_in);
        snprintf(p, sizeof(p), "head.layers.%d.multihead_attn", i); check_attn(f, p, &d->cross, 0, compact);
        snprintf(p, sizeof(p), "head.layers.%d.linear1", i); check_linear(f, p, &d->lin1);
        snprintf(p, sizeof(p), "head.layers.%d.linear2", i); check_linear(f, p, &d->lin2);
    }
    /* Every retained allocation belongs to a checked, CPU-live tensor or the head.
     * This catches accidental restoration of unused memory projection/norm copies. */
    if (compact) assert(checked_bytes == live);
}

/* The same loader serves Omni. Its projection policy must remain two workers,
 * including decoder self_in, whose weights alias self_attn.in_q. */
static void check_policy(const clef_head *h, int workers) {
    const linear *projections[] = {&h->scorer0, &h->question_projection, &h->option_question_projection,
                                   &h->option_context_projection, &h->option_lexical_projection};
    for (size_t i = 0; i < sizeof(projections) / sizeof(projections[0]); i++)
        assert(projections[i]->max_workers == workers);
    assert(h->global_projection.max_workers == 0 && h->scorer3.max_workers == 0);
    for (int i = 0; i < h->n_ev; i++) {
        const evidence_layer *e = &h->ev[i];
        assert(e->attn.in_q.max_workers == workers && e->attn.out.max_workers == workers);
        assert(e->ff1.max_workers == workers && e->ff2.max_workers == workers);
    }
    for (int i = 0; i < h->n_dec; i++) {
        const decoder_layer *d = &h->dec[i];
        assert(d->self_in.max_workers == workers && d->self_attn.out.max_workers == workers);
        assert(d->cross.in_q.max_workers == workers && d->cross.out.max_workers == workers);
        assert(d->lin1.max_workers == workers && d->lin2.max_workers == workers);
    }
}

static void synthetic(int baseline) {
    clef_config c = {.H = 64, .W = 64, .head_ff = 128, .routing_layers = 2, .head_layers = 2, .head_heads = 4};
    for (int kind = 0; kind < 2; kind++) {
      c.model.backbone = kind ? CLEF_OMNI : CLEF_QWEN35;
      for (int format = 0; format < 2; format++) {
        gguf_file f = fixture(&c, format ? GGML_F32 : GGML_BF16);
        char err[256]; calls = peak = 0;
        clef_head *h = clef_head_load(&f, &c, err, sizeof(err)); assert(h);
        const size_t count = calls; hash = 1469598103934665603ull;
        check_head(&f, h, !baseline);
        check_policy(h, kind ? 2 : 4);
        printf("backbone=%s format=%s arrays=%016llx live=%zu peak=%zu calls=%zu\n", kind ? "omni" : "dense", format ? "f32" : "bf16", (unsigned long long)hash, live, peak, count);
        clef_head_free(h); assert(live == 0);
        if (!baseline) {
            for (fail_at = 1; fail_at <= count; fail_at++) {
                calls = 0; err[0] = 0; h = clef_head_load(&f, &c, err, sizeof(err));
                assert(!h && err[0] && live == 0);
            }
            fail_at = 0;
            for (size_t i = 0; i < f.n_tensors; i++) for (int bad = 0; bad < 3; bad++) {
                gguf_tensor save = f.tensors[i];
                if (bad == 0) f.tensors[i].name.len = 0;
                if (bad == 1) f.tensors[i].ne[0]++;
                if (bad == 2) f.tensors[i].type = GGML_F16;
                err[0] = 0; h = clef_head_load(&f, &c, err, sizeof(err));
                assert(!h && err[0] && live == 0); f.tensors[i] = save;
            }
            printf("backbone=%s format=%s allocation failures=%zu malformed tensors=%llu PASS\n", kind ? "omni" : "dense", format ? "f32" : "bf16", count, (unsigned long long)f.n_tensors * 3);
        }
        for (size_t i = 0; i < f.n_tensors; i++) { free((void *)f.tensors[i].name.ptr); free((void *)f.tensors[i].data); }
        free(f.tensors);
      }
    }
}

static double ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
int main(int argc, char **argv) {
    int baseline = argc > 1 && !strcmp(argv[1], "--baseline");
    if (argc <= 1 + baseline) { synthetic(baseline); return 0; }
    gguf_file f; char err[256]; const char *path = argv[1 + baseline];
    if (!gguf_open(&f, path, err, sizeof(err))) { fprintf(stderr, "%s\n", err); return 1; }
    clef_config c = {0};
    int *fields[] = {&c.H, &c.W, &c.routing_layers, &c.head_layers, &c.head_heads, &c.head_ff};
    const char *keys[] = {"clef.embedding_length", "clef.head.width", "clef.head.routing_layers", "clef.head.layers", "clef.head.heads", "clef.head.feedforward"};
    for (int i = 0; i < 6; i++) { uint32_t value; assert(gguf_get_u32(&f, keys[i], &value)); *fields[i] = (int)value; }
    for (int run = 0; run < 6; run++) {
        calls = peak = 0; double start = ms();
        clef_head *h = clef_head_load(&f, &c, err, sizeof(err)); double elapsed = ms() - start;
        if (!h) { fprintf(stderr, "%s\n", err); return 1; }
        printf("model=%s run=%d ms=%.3f live=%zu peak=%zu calls=%zu\n", path, run, elapsed, live, peak, calls);
        if (!run) { hash = 1469598103934665603ull; check_head(&f, h, !baseline); printf("arrays=%016llx\n", (unsigned long long)hash); }
        clef_head_free(h); assert(live == 0);
    }
    gguf_close(&f); return 0;
}
