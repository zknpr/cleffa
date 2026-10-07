#include "clef_record.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <strings.h>

static const char SYSTEM_PROMPT[] =
    "Read the complete state and schema. Decide every field jointly. Each answer "
    "must be exactly one of that field's allowed options.";
/* The fixed prompt around the request's content (template prefix, then the closing suffix). */
static const char PROMPT_HEAD[] = "<|im_start|>system\n";
static const char PROMPT_USER[] = "<|im_end|>\n<|im_start|>user\nSTATE:\n";
static const char PROMPT_TAIL[] = "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:";

static bool fail(char *err, size_t errlen, const char *fmt, const char *arg, size_t arg_len) {
    if (arg) {
        size_t n = arg_len > 200 ? 200 : arg_len;
        if (errlen && n >= errlen) n = errlen - 1;
        /* IDs are validated UTF-8. Keep a whole-character prefix, also when the
         * caller's buffer is small; the remaining format text is ASCII. */
        while (n && n < arg_len && ((unsigned char)arg[n] & 0xc0) == 0x80) n--;
        snprintf(err, errlen, fmt, (int)n, arg);
    } else snprintf(err, errlen, "%s", fmt);
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

/* One entry of a request's images list, decoded to the encoded file bytes. The hosted API's two
 * forms (its input schema): a data URL string, or an object {"content_type": "image/png" |
 * "image/jpeg" | "image/webp", "base64": "..."}. A bare base64 string is taken as well, an
 * extension for the CLI and tests. The declared type is checked against the file's signature
 * after decoding, so a mislabeled image is an error rather than a silently different format. */
/* The declared type against the file's signature, for both wire forms: a mislabeled image is an
 * error rather than a silently different format (the reference's PIL decode ignores the label).
 * Frees the bytes on a mismatch. */
static bool signature_matches(const char *declared, const char *label, uint8_t **bytes, size_t n, size_t index, char *err, size_t errlen) {
    const bool png = n >= 8 && !memcmp(*bytes, "\x89PNG\r\n\x1a\n", 8);
    const bool jpeg = n >= 2 && (*bytes)[0] == 0xff && (*bytes)[1] == 0xd8;
    if ((declared[0] == 'P') == png && (declared[0] == 'J') == jpeg) return true;
    snprintf(err, errlen, "images[%zu]: %s says %s but the data is %s", index, label, declared,
             png ? "PNG" : jpeg ? "JPEG" : "neither PNG nor JPEG");
    free(*bytes);
    *bytes = NULL;
    return false;
}

static bool image_bytes(const jval *im, size_t index, uint8_t **bytes, size_t *n, char *err, size_t errlen) {
    char ierr[256];
    const char *declared = NULL;
    if (im->type == J_STRING) {
        if (im->len >= 5 && !strncasecmp(im->s, "data:", 5)) {
            /* data:[<mediatype>][;base64],<payload>. A media type, when present, must name a format
             * the decoders take and must match the bytes, exactly as content_type must below; an
             * absent type leaves the signature to decide. MIME types compare case-insensitively.
             * (review #3: the string form skipped this check while the object form enforced it) */
            const char *p = im->s + 5, *end = im->s + im->len, *q = p;
            while (q < end && *q != ';' && *q != ',') q++;
            const size_t tlen = (size_t)(q - p);
            if (tlen == 9 && !strncasecmp(p, "image/png", 9)) declared = "PNG";
            else if (tlen == 10 && !strncasecmp(p, "image/jpeg", 10)) declared = "JPEG";
            else if (tlen == 10 && !strncasecmp(p, "image/webp", 10)) { snprintf(err, errlen, "images[%zu]: WebP is not supported (PNG or JPEG only)", index); return false; }
            else if (tlen) { snprintf(err, errlen, "images[%zu]: data URL media type must be image/png or image/jpeg, not %.*s", index, (int)(tlen > 40 ? 40 : tlen), p); return false; }
        }
        if (!clef_base64_decode(im->s, im->len, bytes, n, ierr, sizeof(ierr))) { snprintf(err, errlen, "images[%zu]: %s", index, ierr); return false; }
        return !declared || signature_matches(declared, "data URL media type", bytes, *n, index, err, errlen);
    }
    if (im->type != J_OBJECT) { snprintf(err, errlen, "images[%zu]: must be a data URL string or {\"content_type\", \"base64\"}", index); return false; }
    const jval *ct = json_get(im, "content_type"), *b64 = json_get(im, "base64");
    if (!ct || ct->type != J_STRING || !b64 || b64->type != J_STRING) {
        snprintf(err, errlen, "images[%zu]: an image object needs content_type and base64 strings", index);
        return false;
    }
    if (json_str_eq(ct, "image/png")) declared = "PNG";
    else if (json_str_eq(ct, "image/jpeg")) declared = "JPEG";
    else if (json_str_eq(ct, "image/webp")) { snprintf(err, errlen, "images[%zu]: WebP is not supported (PNG or JPEG only)", index); return false; }
    else { snprintf(err, errlen, "images[%zu]: content_type must be image/png, image/jpeg or image/webp", index); return false; }
    if (b64->len >= 5 && !strncasecmp(b64->s, "data:", 5)) { snprintf(err, errlen, "images[%zu]: base64 must not be a data URL", index); return false; }
    if (!clef_base64_decode(b64->s, b64->len, bytes, n, ierr, sizeof(ierr))) { snprintf(err, errlen, "images[%zu]: %s", index, ierr); return false; }
    return signature_matches(declared, "content_type", bytes, *n, index, err, errlen);
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
    for (int i = 0; i < r->n_images; i++) clef_image_patches_free(&r->images[i].pt);
    free(r->images);
    for (int i = 0; i < r->q_alloc; i++) {
        free(r->q[i].opt_span);
        free(r->q[i].opt_id);
        free(r->q[i].opt_id_len);
    }
    free(r->q);
    free(r->owned);
    memset(r, 0, sizeof(*r));
}

/* Decodes and preprocesses the request's images into out->images, after the schema is built so
 * that its length counts toward each image's budget (see the reserve below). */
static bool encode_images(const clef_tokenizer *tok, const jval *req, const jval *images, size_t n_images,
                          clef_encode_opts opts, size_t schema_tokens, clef_record *out, char *err, size_t errlen) {
    /* media_kwargs: the reference forwards them to the processor; only its pixel bounds are
     * taken here (deliberate divergence: other processor arguments are rejected, not ignored).
     * The processor applies the bounds only when both are given and silently ignores a lone
     * one (Qwen2VLImageProcessor._standardize_kwargs); a lone bound is rejected here instead. */
    clef_image_params prm = opts.vision.image;
    const jval *mk = json_get(req, "media_kwargs");
    if (mk && mk->type != J_NULL) {
        if (mk->type != J_OBJECT) return fail(err, errlen, "media_kwargs must be an object", NULL, 0);
        /* n counts distinct keys: the DOM merges a repeated key as json.loads does (clef_json.h), so
         * {"min_pixels": a, "min_pixels": b} is one member and refused here like any lone bound. */
        if (mk->n == 1) return fail(err, errlen, "media_kwargs: give both min_pixels and max_pixels (the reference ignores one alone)", NULL, 0);
        for (size_t i = 0; i < mk->n; i++) {
            const jmember *m = &mk->members[i];
            const bool is_min = m->klen == 10 && !memcmp(m->key, "min_pixels", 10);
            const bool is_max = m->klen == 10 && !memcmp(m->key, "max_pixels", 10);
            if (!is_min && !is_max) return fail(err, errlen, "media_kwargs: only min_pixels and max_pixels are supported, not %.*s", m->key, m->klen);
            /* strtol over the token's own bytes: atol on an out-of-range number is undefined
             * behavior, and only a saturating libc made it look safe (review #3). */
            char num[24];
            long value = 0;
            bool is_int = m->val->type == J_INT && m->val->len > 0 && m->val->len < sizeof(num);
            if (is_int) {
                memcpy(num, m->val->s, m->val->len);
                num[m->val->len] = 0;
                char *endp;
                errno = 0;
                value = strtol(num, &endp, 10);
                is_int = errno == 0 && endp == num + m->val->len;
            }
            if (!is_int || value <= 0 || value > INT32_MAX) {
                return fail(err, errlen, "media_kwargs: %.*s must be a positive integer", m->key, m->klen);
            }
            if (is_min) prm.min_pixels = value; else prm.max_pixels = value;
        }
        if (prm.min_pixels > prm.max_pixels) return fail(err, errlen, "media_kwargs: min_pixels exceeds max_pixels", NULL, 0);
    }
    /* Images are never truncated, and the fixed prompt, one start/end pair per image, the
     * newline after them and the schema always come with them: an image is refused before
     * preprocessing when it, the images before it and all of those cannot fit the context. That
     * is exactly the final length check's sum less the later images, so it refuses nothing the
     * reference accepts, only earlier. A per-image comparison with the whole context let an image
     * of exactly 16,384 tokens allocate 384 MiB of patches, and several large images each pass
     * it, before the length check refused the request (review #3); without the schema, a request
     * whose schema could never fit still preprocessed its images (review #3, Codex on 600ddfe). */
    size_t reserve = 1 + 2 * n_images + schema_tokens;
    {
        jbuf pb = {0};
        clef_tokens pt = {0};
        jbuf_puts(&pb, PROMPT_HEAD);
        jbuf_puts(&pb, SYSTEM_PROMPT);
        jbuf_puts(&pb, PROMPT_USER);
        const bool tok_ok = tok_jbuf(tok, &pb, false, &pt) && tok_z(tok, PROMPT_TAIL, false, &pt);
        reserve += pt.len;
        jbuf_free(&pb);
        clef_tokens_free(&pt);
        if (!tok_ok) return fail(err, errlen, "out of memory", NULL, 0);
    }
    out->images = calloc(n_images, sizeof(*out->images));
    if (!out->images) return fail(err, errlen, "out of memory", NULL, 0);
    for (size_t i = 0; i < n_images; i++) {
        const jval *im = images->items[i];
        char ierr[256];
        uint8_t *bytes = NULL;
        size_t n = 0;
        clef_rgb rgb = {0};
        out->n_images = (int)i + 1;   /* freed by clef_record_free even when this one fails */
        if (!image_bytes(im, i, &bytes, &n, err, errlen)) return false;
        bool ok = clef_image_decode_limited(bytes, n, opts.vision.max_image_pixels, &rgb, ierr, sizeof(ierr));
        free(bytes);
        const int width = rgb.width, height = rgb.height;
        /* Bound the work before doing it. The resize buffer and the f32 patches scale with the
         * resized area, which media_kwargs can push to 268 Mpx (16384x16384, about 6 GB of
         * patches) from a 40x40 file; the per-image limit used to be checked only after that
         * allocation (review #3). Compute the geometry first and refuse what the per-image
         * limit forbids or what no request could hold, then resize and patch. */
        const int factor = prm.patch * prm.merge;
        int rh = 0, rw = 0;
        ok = ok && clef_smart_resize(height, width, factor, prm.min_pixels, prm.max_pixels, &rh, &rw, ierr, sizeof(ierr));
        if (!ok) { clef_rgb_free(&rgb); snprintf(err, errlen, "images[%zu]: %s", i, ierr); return false; }
        const long tokens = (long)(rh / factor) * (rw / factor);
        if (opts.vision.max_image_tokens > 0 && tokens > opts.vision.max_image_tokens) {
            clef_rgb_free(&rgb);
            /* one token per merge window of patch*merge pixels on each side */
            const long px_per_token = (long)prm.patch * prm.patch * prm.merge * prm.merge;
            snprintf(err, errlen, "images[%zu]: %dx%d resizes to %ld tokens, above the limit of %ld per image; "
                     "downscale it or pass media_kwargs.max_pixels <= %ld",
                     i, width, height, tokens, opts.vision.max_image_tokens,
                     opts.vision.max_image_tokens * px_per_token);
            return false;
        }
        if ((size_t)out->n_image_tokens + (size_t)tokens + reserve > (size_t)opts.max_length) {
            clef_rgb_free(&rgb);
            snprintf(err, errlen, "images[%zu]: %dx%d resizes to %ld tokens; with %d image tokens before it and the "
                     "%zu-token prompt the request cannot fit %d tokens",
                     i, width, height, tokens, out->n_image_tokens, reserve, opts.max_length);
            return false;
        }
        ok = clef_image_preprocess(&rgb, &prm, &out->images[i].pt, ierr, sizeof(ierr));
        clef_rgb_free(&rgb);
        if (!ok) { snprintf(err, errlen, "images[%zu]: %s", i, ierr); return false; }
        if (out->images[i].pt.n_tokens != tokens) return fail(err, errlen, "internal: image token count changed after preprocessing", NULL, 0);
        if (out->n_image_tokens > INT32_MAX - tokens) return fail(err, errlen, "too many image tokens", NULL, 0);
        out->n_image_tokens += (int32_t)tokens;
    }
    return true;
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
    /* Deliberate divergence: the reference also takes videos (frame arrays). */
    if (truthy(videos)) return fail(err, errlen, "videos are not supported", NULL, 0);
    /* The reference iterates record.get("images") or []: any JSON array of images; the engine
     * takes encoded images in the hosted API's forms (image_bytes). */
    if (truthy(images) && images->type != J_ARRAY) {
        return fail(err, errlen, "images must be a list of data URLs or {\"content_type\", \"base64\"} objects (PNG or JPEG)", NULL, 0);
    }
    const size_t n_images = images && images->type == J_ARRAY ? images->n : 0;
    if (n_images) {
        const clef_vision_opts *v = &opts.vision;
        if (!v->image_token_id) return fail(err, errlen, "images are not supported by this model file (no vision tower)", NULL, 0);
        if (v->max_images > 0 && n_images > (size_t)v->max_images) {
            snprintf(err, errlen, "too many images: %zu, at most %d per request", n_images, v->max_images);
            return false;
        }
        if (n_images > INT32_MAX / 4) return fail(err, errlen, "too many images", NULL, 0);
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

    /* Images after the schema, so that its length is part of each image's budget. */
    if (ok && n_images && !encode_images(tok, req, images, n_images, opts, schema.len, out, err, errlen)) {
        jbuf_free(&b);
        clef_tokens_free(&schema);
        return false;
    }

    if (ok) {
        jbuf_puts(&b, PROMPT_HEAD);
        jbuf_puts(&b, SYSTEM_PROMPT);
        jbuf_puts(&b, PROMPT_USER);
        ok = tok_jbuf(tok, &b, false, &prefix);   /* template: real control tokens */
        if (ok && n_images) {
            /* _encode_media: "<|vision_start|><|image_pad|><|vision_end|>" per image, then "\n",
             * tokenized by the processor, which expands each <|image_pad|> to the image's tokens.
             * These are engine-made control tokens, never request text, in both modes. */
            for (size_t i = 0; ok && i < n_images; i++) {
                clef_image_ref *ir = &out->images[i];
                ok = clef_tokens_push(&prefix, opts.vision.start_token_id);
                ir->tok_start = (int32_t)prefix.len;
                for (int k = 0; ok && k < ir->pt.n_tokens; k++) ok = clef_tokens_push(&prefix, opts.vision.image_token_id);
                ok = ok && clef_tokens_push(&prefix, opts.vision.end_token_id);
            }
            clef_tokens nl = {0};
            ok = ok && tok_z(tok, "\n", false, &nl);
            if (ok && nl.len != 1) ok = false;   /* the tokenizer's newline is one token */
            for (size_t i = 0; ok && i < nl.len; i++) ok = clef_tokens_push(&prefix, nl.ids[i]);
            clef_tokens_free(&nl);
        }
        ok = ok && tok_z(tok, PROMPT_TAIL, false, &suffix);
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
    out->schema_start = offset;

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
    if (out->n_images) {
        /* In parity mode a literal "<|image_pad|>" in request text becomes the placeholder token;
         * the reference then fails scattering the image features ("Image features and image tokens
         * do not match"). Fail the same way instead of guessing which tokens are the images. */
        int32_t pads = 0;
        for (size_t i = 0; i < out->ids.len; i++) pads += out->ids.ids[i] == opts.vision.image_token_id;
        if (pads != out->n_image_tokens) {
            snprintf(err, errlen, "image placeholder tokens in request content: %d image tokens for %d image features",
                     pads, out->n_image_tokens);
            return false;
        }
    }
    return true;
}

bool clef_vision_opts_load(const gguf_file *f, clef_vision_opts *v, char *err, size_t errlen) {
    uint32_t patch, merge, temporal, min_px, max_px, image_id, start_id, end_id, video_id, vocab;
    if (!gguf_find_kv(f, "clef.vision.image_token_id")) {
        /* a model file converted without the vision tower: text only */
        v->image_token_id = 0;
        return true;
    }
    if (!gguf_get_u32(f, "clef.vision.patch_size", &patch) || !gguf_get_u32(f, "clef.vision.spatial_merge_size", &merge) ||
        !gguf_get_u32(f, "clef.vision.temporal_patch_size", &temporal) ||
        !gguf_get_u32(f, "clef.vision.image.min_pixels", &min_px) || !gguf_get_u32(f, "clef.vision.image.max_pixels", &max_px) ||
        !gguf_get_u32(f, "clef.vision.image_token_id", &image_id) || !gguf_get_u32(f, "clef.vision.start_token_id", &start_id) ||
        !gguf_get_u32(f, "clef.vision.end_token_id", &end_id) || !gguf_get_u32(f, "clef.vision.video_token_id", &video_id) ||
        !gguf_get_u32(f, "clef.vocab_size", &vocab)) {
        snprintf(err, errlen, "model: incomplete clef.vision.* keys");
        return false;
    }
    /* Geometry the preprocessing and kernels assume (clef_image.c, the vision tower). */
    if (patch != 16 || merge != 2 || temporal != 2 || min_px == 0 || max_px < min_px || max_px > INT32_MAX ||
        !image_id || !start_id || !end_id || !video_id || image_id > INT32_MAX || start_id > INT32_MAX || end_id > INT32_MAX) {
        snprintf(err, errlen, "model: unsupported vision geometry (patch %u, merge %u, temporal %u)", patch, merge, temporal);
        return false;
    }
    /* Every id the encoder emits indexes the embedding table. An id at or past the vocabulary
     * loaded fine and failed each image request only after decoding it (review #3). */
    if (vocab > INT32_MAX || image_id >= vocab || start_id >= vocab || end_id >= vocab || video_id >= vocab) {
        snprintf(err, errlen, "model: vision token ids must be below the vocabulary size %u", vocab);
        return false;
    }
    v->image = (clef_image_params){ (long)min_px, (long)max_px, (int)patch, (int)merge, (int)temporal };
    v->image_token_id = (int32_t)image_id;
    v->start_token_id = (int32_t)start_id;
    v->end_token_id = (int32_t)end_id;
    v->video_token_id = (int32_t)video_id;
    return true;
}

void clef_record_positions(const clef_record *r, int32_t *pos3) {
    const int32_t T = (int32_t)r->ids.len;
    int32_t cur = 0, t = 0;
    int next = 0;   /* images are in token order */
    while (t < T) {
        if (next < r->n_images && r->images[next].tok_start == t) {
            const clef_image_ref *ir = &r->images[next++];
            const int h = ir->pt.grid_h / 2, w = ir->pt.grid_w / 2;   /* merged grid */
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++, t++) {
                    pos3[3 * t] = cur;
                    pos3[3 * t + 1] = cur + y;
                    pos3[3 * t + 2] = cur + x;
                }
            cur += h > w ? h : w;
        } else {
            pos3[3 * t] = pos3[3 * t + 1] = pos3[3 * t + 2] = cur;
            cur++;
            t++;
        }
    }
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
