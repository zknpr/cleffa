#include "clef_record.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char SYSTEM_PROMPT[] =
    "Read the complete state and schema. Decide every field jointly. Each answer "
    "must be exactly one of that field's allowed options.";

static bool fail(char *err, size_t errlen, const char *fmt, const char *arg, size_t arg_len) {
    if (arg) snprintf(err, errlen, fmt, (int)(arg_len > 200 ? 200 : arg_len), arg);
    else snprintf(err, errlen, "%s", fmt);
    return false;
}

/* Python truthiness of a JSON value. */
static bool truthy(const jval *v) {
    if (!v) return false;
    switch (v->type) {
    case J_NULL: case J_FALSE: return false;
    case J_TRUE: return true;
    case J_INT: return !(v->len == 1 && v->s[0] == '0');
    case J_FLOAT: return v->f != 0.0;  /* NaN is truthy in Python, and NaN != 0 */
    case J_STRING: return v->len > 0;
    case J_ARRAY: case J_OBJECT: return v->n > 0;
    }
    return false;
}

/* render(): strings verbatim, everything else json.dumps(sort_keys=True, compact, ensure_ascii=False). */
static void render(jbuf *b, const jval *v) {
    if (v->type == J_STRING) jbuf_put(b, v->s, v->len);
    else json_dump(b, v, true);
}

/* split: strict mode for request-derived text (see clef_encode_opts.strict) */
static bool tok_z(const clef_tokenizer *tok, const char *z, bool split, clef_tokens *out) {
    return clef_tok_encode_ex(tok, z, strlen(z), SIZE_MAX, split, out);
}

static bool tok_jbuf(const clef_tokenizer *tok, jbuf *b, bool split, clef_tokens *out) {
    if (b->oom) return false;
    bool ok = clef_tok_encode_ex(tok, b->p ? b->p : "", b->len, SIZE_MAX, split, out);
    b->len = 0;
    return ok;
}

typedef struct { const char *id; size_t id_len; const jval *desc; } option;

static int option_cmp(const void *a, const void *b) {
    const option *x = a, *y = b;
    size_t n = x->id_len < y->id_len ? x->id_len : y->id_len;
    int c = memcmp(x->id, y->id, n);  /* UTF-8 byte order == Python str order */
    if (c) return c;
    return (x->id_len > y->id_len) - (x->id_len < y->id_len);
}

void clef_record_free(clef_record *r) {
    clef_tokens_free(&r->ids);
    for (int i = 0; i < r->q_alloc; i++) {
        free(r->q[i].opt_span);
        free(r->q[i].opt_id);
        free(r->q[i].opt_id_len);
    }
    free(r->q);
    free(r->owned);
    memset(r, 0, sizeof(*r));
}

static bool encode_request(const clef_tokenizer *tok, const jval *req, clef_encode_opts opts,
                           clef_record *out, char *err, size_t errlen) {
    if (!req || req->type != J_OBJECT) return fail(err, errlen, "request must be a JSON object", NULL, 0);
    const jval *model = json_get(req, "model");
    const jval *state = json_get(req, "state");
    const jval *questions = json_get(req, "questions");
    if (!model || model->type != J_STRING || !state) return fail(err, errlen, "model and state are required", NULL, 0);
    if (!questions || questions->type != J_OBJECT || questions->n == 0) {
        return fail(err, errlen, "at least one question is required", NULL, 0);
    }
    const jval *images = json_get(req, "images"), *videos = json_get(req, "videos");
    if (truthy(images) || truthy(videos)) {
        return fail(err, errlen, "images and videos are not supported by this build (text-only)", NULL, 0);
    }

    size_t nq = questions->n;
    out->q = calloc(nq, sizeof(*out->q));
    if (!out->q) return fail(err, errlen, "out of memory", NULL, 0);
    out->q_alloc = (int)nq;

    /* Validation pass (systemone), and count the generated option ids. */
    size_t owned_bytes = 0;
    for (size_t i = 0; i < nq; i++) {
        const jmember *m = &questions->members[i];
        const jval *q = m->val;
        if (q->type != J_OBJECT) return fail(err, errlen, "%.*s: question must be an object", m->key, m->klen);
        const jval *type = json_get(q, "type");
        int t;
        if (json_str_eq(type, "noul")) t = CLEF_Q_NOUL;
        else if (json_str_eq(type, "choice")) t = CLEF_Q_CHOICE;
        else if (json_str_eq(type, "score")) t = CLEF_Q_SCORE;
        else return fail(err, errlen, "%.*s: type must be noul, choice, or score", m->key, m->klen);
        const jval *criteria = json_get(q, "criteria");
        if (t != CLEF_Q_NOUL && !truthy(criteria)) {
            return fail(err, errlen, "%.*s: criteria must not be empty", m->key, m->klen);
        }
        if (t == CLEF_Q_CHOICE && criteria->type != J_OBJECT) {
            return fail(err, errlen, "%.*s: choice criteria must be an object", m->key, m->klen);
        }
        /* Deliberate divergence: the reference enumerates a dict's keys here; the
         * documented input format is a list, so anything else is rejected. */
        if (t == CLEF_Q_SCORE && criteria->type != J_ARRAY) {
            return fail(err, errlen, "%.*s: score criteria must be a list", m->key, m->klen);
        }
        if (t == CLEF_Q_NOUL && truthy(criteria) && criteria->type != J_OBJECT) {
            return fail(err, errlen, "%.*s: noul criteria must be an object (deliberate: lists of pairs are not accepted)", m->key, m->klen);
        }
        out->q[i].type = t;
        if (t == CLEF_Q_SCORE) owned_bytes += criteria->n * 21;
    }
    out->owned = malloc(owned_bytes + 16);
    if (!out->owned) return fail(err, errlen, "out of memory", NULL, 0);
    char *owned = out->owned;

    clef_tokens schema = {0}, prefix = {0}, suffix = {0}, state_ids = {0};
    jbuf b = {0};
    bool ok = true;
    char head[96];

    ok = tok_z(tok, "\n\nSCHEMA FIELDS:\n", opts.strict, &schema);
    for (size_t i = 0; ok && i < nq; i++) {
        const jmember *m = &questions->members[i];
        const jval *q = m->val;
        clef_question *cq = &out->q[i];
        cq->id = m->key;
        cq->id_len = m->klen;
        cq->question = q;

        const jval *type = json_get(q, "type");
        snprintf(head, sizeof(head), "\nFIELD %zu\nID: ", i + 1);
        jbuf_puts(&b, head);
        jbuf_put(&b, m->key, m->klen);
        jbuf_puts(&b, "\nTYPE: ");
        jbuf_put(&b, type->s, type->len);
        jbuf_puts(&b, "\nINSTRUCTION: ");
        ok = tok_jbuf(tok, &b, opts.strict, &schema);

        cq->span[0] = (int32_t)schema.len;
        const jval *ins = json_get(q, "instructions");
        if (!ins || ins->type == J_NULL || (ins->type == J_STRING && ins->len == 0)) {
            jbuf_put(&b, m->key, m->klen);  /* str(question_id) */
        } else {
            render(&b, ins);
        }
        ok = ok && tok_jbuf(tok, &b, opts.strict, &schema);
        cq->span[1] = (int32_t)schema.len;
        if (cq->span[1] == cq->span[0]) {
            /* deliberate divergence: the reference averages an empty span (NaN logits) */
            jbuf_free(&b);
            clef_tokens_free(&schema);
            return fail(err, errlen, "question with an empty id needs non-empty instructions", NULL, 0);
        }
        ok = ok && tok_z(tok, "\nALLOWED OPTIONS:\n", opts.strict, &schema);

        /* question_options() */
        const jval *criteria = json_get(q, "criteria");
        option *opts_ = NULL;
        int n_opt = 0;
        if (cq->type == CLEF_Q_NOUL) {
            static jval def_true = { .type = J_STRING, .s = "The proposition is true or the answer is yes.", .len = 45 };
            static jval def_false = { .type = J_STRING, .s = "The proposition is false or the answer is no.", .len = 45 };
            const jval *dt = &def_true, *df = &def_false;
            if (truthy(criteria)) {
                const jval *ct = json_get(criteria, "true"), *cf = json_get(criteria, "false");
                if (ct) dt = ct;
                if (cf) df = cf;
            }
            opts_ = malloc(2 * sizeof(*opts_));
            if (!opts_) { ok = false; break; }
            opts_[0] = (option){ "true", 4, dt };
            opts_[1] = (option){ "false", 5, df };
            n_opt = 2;
        } else if (cq->type == CLEF_Q_CHOICE) {
            n_opt = (int)criteria->n;
            opts_ = malloc((size_t)n_opt * sizeof(*opts_));
            if (!opts_) { ok = false; break; }
            for (int k = 0; k < n_opt; k++) {
                opts_[k] = (option){ criteria->members[k].key, criteria->members[k].klen, criteria->members[k].val };
            }
            qsort(opts_, (size_t)n_opt, sizeof(*opts_), option_cmp);
        } else {
            n_opt = (int)criteria->n;
            opts_ = malloc((size_t)n_opt * sizeof(*opts_));
            if (!opts_) { ok = false; break; }
            for (int k = 0; k < n_opt; k++) {
                int l = snprintf(owned, 21, "%d", k);
                opts_[k] = (option){ owned, (size_t)l, criteria->items[k] };
                owned += l + 1;
            }
        }

        cq->n_opt = n_opt;
        cq->opt_span = malloc((size_t)n_opt * sizeof(*cq->opt_span));
        cq->opt_id = malloc((size_t)n_opt * sizeof(*cq->opt_id));
        cq->opt_id_len = malloc((size_t)n_opt * sizeof(*cq->opt_id_len));
        if (!cq->opt_span || !cq->opt_id || !cq->opt_id_len) { free(opts_); ok = false; break; }
        for (int k = 0; ok && k < n_opt; k++) {
            snprintf(head, sizeof(head), "OPTION %d: ", k + 1);
            ok = tok_z(tok, head, opts.strict, &schema);
            cq->opt_span[k][0] = (int32_t)schema.len;
            /* render({"option_id": id, "description": desc}) with sorted keys */
            jbuf_put(&b, "{", 1);
            if (opts_[k].desc->type != J_NULL) {
                jbuf_puts(&b, "\"description\":");
                json_dump(&b, opts_[k].desc, true);
                jbuf_put(&b, ",", 1);
            }
            jbuf_puts(&b, "\"option_id\":");
            json_put_string(&b, opts_[k].id, opts_[k].id_len);
            jbuf_put(&b, "}", 1);
            ok = ok && tok_jbuf(tok, &b, opts.strict, &schema);
            cq->opt_span[k][1] = (int32_t)schema.len;
            cq->opt_id[k] = opts_[k].id;
            cq->opt_id_len[k] = opts_[k].id_len;
            ok = ok && tok_z(tok, "\n", opts.strict, &schema);
        }
        free(opts_);
        ok = ok && tok_z(tok, "END FIELD\n", opts.strict, &schema);
        out->nq = (int)i + 1;
        if (ok && schema.len > (size_t)opts.max_length) {
            /* the reference builds the whole schema and then rejects it; stop early instead */
            jbuf_free(&b);
            clef_tokens_free(&schema);
            snprintf(err, errlen, "schema requires more than %d tokens before state; maximum is %d",
                     opts.max_length, opts.max_length);
            return false;
        }
    }

    if (ok) {
        jbuf_puts(&b, "<|im_start|>system\n");
        jbuf_puts(&b, SYSTEM_PROMPT);
        jbuf_puts(&b, "<|im_end|>\n<|im_start|>user\nSTATE:\n");
        ok = tok_jbuf(tok, &b, false, &prefix);   /* template: real control tokens */
        ok = ok && tok_z(tok, "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:", false, &suffix);
        /* only max_length - fixed state tokens can survive truncation (and max_state_tokens) */
        const size_t fixed0 = prefix.len + schema.len + suffix.len;
        size_t keep = fixed0 < (size_t)opts.max_length ? (size_t)opts.max_length - fixed0 : 0;
        if (opts.max_state_tokens >= 0 && (size_t)opts.max_state_tokens < keep) keep = (size_t)opts.max_state_tokens;
        render(&b, state);
        /* one token past the budget is enough to know whether truncation would happen */
        const size_t probe = opts.reject_truncation ? keep + 1 : keep;
        ok = ok && !b.oom && clef_tok_encode_ex(tok, b.p ? b.p : "", b.len, probe, opts.strict, &state_ids);
        b.len = 0;
        if (ok && opts.reject_truncation && state_ids.len > keep) {
            jbuf_free(&b);
            clef_tokens_free(&schema); clef_tokens_free(&prefix); clef_tokens_free(&suffix); clef_tokens_free(&state_ids);
            snprintf(err, errlen, "state needs more than %zu tokens but only %zu fit (max %d tokens including "
                     "%zu for the schema and template); the reference would silently drop the rest",
                     keep, keep, opts.max_length, fixed0);
            return false;
        }
    }
    jbuf_free(&b);
    if (!ok) {
        clef_tokens_free(&schema); clef_tokens_free(&prefix); clef_tokens_free(&suffix); clef_tokens_free(&state_ids);
        return fail(err, errlen, "out of memory while encoding", NULL, 0);
    }

    size_t n_state = state_ids.len;
    if (opts.max_state_tokens >= 0 && n_state > (size_t)opts.max_state_tokens) n_state = (size_t)opts.max_state_tokens;
    size_t fixed = prefix.len + schema.len + suffix.len;
    if (fixed > (size_t)opts.max_length) {
        snprintf(err, errlen, "schema requires %zu tokens before state; maximum is %d", fixed, opts.max_length);
        clef_tokens_free(&schema); clef_tokens_free(&prefix); clef_tokens_free(&suffix); clef_tokens_free(&state_ids);
        return false;
    }
    if (n_state > (size_t)opts.max_length - fixed) n_state = (size_t)opts.max_length - fixed;
    int32_t offset = (int32_t)(prefix.len + n_state);

    for (size_t i = 0; ok && i < prefix.len; i++) ok = clef_tokens_push(&out->ids, prefix.ids[i]);
    for (size_t i = 0; ok && i < n_state; i++) ok = clef_tokens_push(&out->ids, state_ids.ids[i]);
    for (size_t i = 0; ok && i < schema.len; i++) ok = clef_tokens_push(&out->ids, schema.ids[i]);
    for (size_t i = 0; ok && i < suffix.len; i++) ok = clef_tokens_push(&out->ids, suffix.ids[i]);
    clef_tokens_free(&schema); clef_tokens_free(&prefix); clef_tokens_free(&suffix); clef_tokens_free(&state_ids);
    if (!ok) return fail(err, errlen, "out of memory while encoding", NULL, 0);

    for (int i = 0; i < out->nq; i++) {
        out->q[i].span[0] += offset;
        out->q[i].span[1] += offset;
        for (int k = 0; k < out->q[i].n_opt; k++) {
            out->q[i].opt_span[k][0] += offset;
            out->q[i].opt_span[k][1] += offset;
        }
    }
    return true;
}

bool clef_encode_request(const clef_tokenizer *tok, const jval *req, clef_encode_opts opts,
                         clef_record *out, char *err, size_t errlen) {
    memset(out, 0, sizeof(*out));
    if (encode_request(tok, req, opts, out, err, errlen)) return true;
    clef_record_free(out);  /* every failure path, including ones after partial allocation */
    return false;
}

static double p_of(const clef_question *q, const float *probs, const char *id, size_t len) {
    for (int k = 0; k < q->n_opt; k++) {
        if (q->opt_id_len[k] == len && !memcmp(q->opt_id[k], id, len)) return (double)probs[k];
    }
    return 0.0;
}

bool clef_build_response(const jval *req, const clef_record *rec, float *const *probs, jbuf *b) {
    const jval *model = json_get(req, "model");
    jbuf_puts(b, "{\"model\":");
    json_put_string(b, model->s, model->len);
    jbuf_puts(b, ",\"answers\":{");
    for (int i = 0; i < rec->nq; i++) {
        const clef_question *q = &rec->q[i];
        const float *p = probs[i];
        if (i) jbuf_put(b, ",", 1);
        json_put_string(b, q->id, q->id_len);
        jbuf_put(b, ":", 1);
        if (q->type == CLEF_Q_NOUL) {
            jbuf_puts(b, "{\"type\":\"noul\",\"noul\":");
            json_put_float(b, py_round(p_of(q, p, "true", 4), 4));
            jbuf_put(b, "}", 1);
        } else if (q->type == CLEF_Q_CHOICE) {
            /* options in the request's insertion order; max() keeps the first maximum */
            const jval *criteria = json_get(q->question, "criteria");
            size_t best = 0;
            double best_p = -1.0;
            for (size_t k = 0; k < criteria->n; k++) {
                double pk = p_of(q, p, criteria->members[k].key, criteria->members[k].klen);
                if (pk > best_p) { best_p = pk; best = k; }
            }
            jbuf_puts(b, "{\"type\":\"choice\",\"choice\":");
            json_put_string(b, criteria->members[best].key, criteria->members[best].klen);
            jbuf_puts(b, ",\"confidence\":");
            json_put_float(b, py_round(best_p, 4));
            jbuf_puts(b, ",\"probabilities\":{");
            for (size_t k = 0; k < criteria->n; k++) {
                if (k) jbuf_put(b, ",", 1);
                json_put_string(b, criteria->members[k].key, criteria->members[k].klen);
                jbuf_put(b, ":", 1);
                json_put_float(b, py_round(p_of(q, p, criteria->members[k].key, criteria->members[k].klen), 4));
            }
            jbuf_puts(b, "}}");
        } else {
            const jval *criteria = json_get(q->question, "criteria");
            /* sum(index * p) exactly as CPython >= 3.12 computes it: Neumaier-compensated
             * float summation (Python/bltinmodule.c builtin_sum_impl). */
            double score = 0.0, comp = 0.0, conf = -1.0;
            for (int k = 0; k < q->n_opt; k++) {
                double x = (double)k * (double)p[k];
                double t = score + x;
                if (fabs(score) >= fabs(x)) comp += (score - t) + x;
                else comp += (x - t) + score;
                score = t;
                if ((double)p[k] > conf) conf = (double)p[k];
            }
            if (comp != 0.0 && isfinite(comp)) score += comp;
            jbuf_puts(b, "{\"type\":\"score\",\"score\":");
            json_put_float(b, py_round(score, 4));
            jbuf_puts(b, ",\"confidence\":");
            json_put_float(b, py_round(conf, 4));
            jbuf_puts(b, ",\"legend\":{");
            char num[24];
            for (int k = 0; k < q->n_opt; k++) {
                if (k) jbuf_put(b, ",", 1);
                snprintf(num, sizeof(num), "\"%d\":", k);
                jbuf_puts(b, num);
                json_dump(b, criteria->items[k], false);
            }
            jbuf_puts(b, "},\"probabilities\":{");
            for (int k = 0; k < q->n_opt; k++) {
                if (k) jbuf_put(b, ",", 1);
                snprintf(num, sizeof(num), "\"%d\":", k);
                jbuf_puts(b, num);
                json_put_float(b, py_round((double)p[k], 4));
            }
            jbuf_puts(b, "}}");
        }
    }
    char usage[96];
    snprintf(usage, sizeof(usage), "},\"usage\":{\"input_tokens\":%zu,\"output_tokens\":0}}", rec->ids.len);
    jbuf_puts(b, usage);
    return !b->oom;
}
