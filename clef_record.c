#include "clef_record.h"
#include "clef_video.h"

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

static bool nonnegative_int(const jval *v, long *out) {
    if (!v || v->type != J_INT || !v->len || v->len > 10) return false;
    char s[12]; memcpy(s, v->s, v->len); s[v->len] = 0;
    char *end; errno = 0;
    long n = strtol(s, &end, 10);
    if (errno || *end || n < 0 || n > INT32_MAX) return false;
    *out = n; return true;
}

static bool positive_int(const jval *v, long *out) {
    return nonnegative_int(v, out) && *out > 0;
}

static bool positive_number(const jval *v, double *out) {
    if (!v || (v->type != J_INT && v->type != J_FLOAT)) return false;
    double value = v->f;
    if (v->type == J_INT) {
        long n;
        if (!positive_int(v, &n)) return false;
        value = (double)n;
    }
    if (!isfinite(value) || value <= 0 || value > 1000) return false;
    *out = value; return true;
}

static bool decode_frame(const jval *im, size_t index, long limit, clef_rgb *rgb, char *err, size_t errlen) {
    uint8_t *bytes = NULL; size_t size = 0;
    if (!image_bytes(im, index, &bytes, &size, err, errlen)) return false;
    bool ok = clef_image_decode_limited(bytes, size, limit, rgb, err, errlen);
    free(bytes);
    return ok;
}

static bool encode_video(const clef_tokenizer *tok, const jval *input, const jval *kwargs, int group,
                         clef_encode_opts opts, size_t *reserve, clef_record *out, char *err, size_t errlen) {
    long min_px = CLEF_VIDEO_MIN_PIXELS, max_px = CLEF_VIDEO_MAX_PIXELS, requested = 0;
    double sample_fps = 2;
    bool sample = true, explicit_fps = false;
    if (kwargs) {
        if (kwargs->type != J_OBJECT) return fail(err, errlen, "videos_kwargs must be an object", NULL, 0);
        for (size_t i = 0; i < kwargs->n; i++) {
            const jmember *m = &kwargs->members[i];
            if (m->klen == 16 && !memcmp(m->key, "do_sample_frames", 16)) {
                if (m->val->type != J_TRUE && m->val->type != J_FALSE) return fail(err, errlen, "do_sample_frames must be boolean", NULL, 0);
                sample = m->val->type == J_TRUE;
            } else if (m->klen == 10 && !memcmp(m->key, "num_frames", 10)) {
                if (!positive_int(m->val, &requested)) return fail(err, errlen, "num_frames must be a positive integer", NULL, 0);
            } else if (m->klen == 3 && !memcmp(m->key, "fps", 3)) {
                explicit_fps = m->val->type != J_NULL;
                if (explicit_fps && !positive_number(m->val, &sample_fps)) return fail(err, errlen, "sampling fps must be finite and in (0,1000]", NULL, 0);
            } else if (m->klen == 4 && !memcmp(m->key, "size", 4)) {
                const jval *s = m->val;
                if (s->type != J_OBJECT || s->n != 2 || !positive_int(json_get(s, "shortest_edge"), &min_px) ||
                    !positive_int(json_get(s, "longest_edge"), &max_px) || min_px > max_px)
                    return fail(err, errlen, "video size needs positive shortest_edge <= longest_edge", NULL, 0);
            } else return fail(err, errlen, "unsupported videos_kwargs: %.*s", m->key, m->klen);
        }
    }
    if ((requested && explicit_fps) || (!sample && (requested || explicit_fps)))
        return fail(err, errlen, "choose sampling fps or num_frames; neither with do_sample_frames=false", NULL, 0);

    const jval *frames = input->type == J_OBJECT ? json_get(input, "frames") : NULL;
    if (!frames) return fail(err, errlen, "video must be {frames,fps}; convert MP4/MOV with tools/video_request.py", NULL, 0);
    const jval *source_indices = json_get(input, "frame_indices"), *total = json_get(input, "total_num_frames");
    struct { int frames, width, height; double fps; } info = {0};
    clef_rgb first = {0}, second = {0};
    bool ok = false;
    if (frames->type != J_ARRAY || frames->n < 1 || frames->n > CLEF_VIDEO_MAX_SOURCE_FRAMES ||
        !positive_number(json_get(input, "fps"), &info.fps) || frames->n / info.fps > 600)
        return fail(err, errlen, "frames must contain 1..18000 images with positive fps and duration <= 600 seconds", NULL, 0);
    if (input->n != (source_indices ? 4u : 2u) || !!source_indices != !!total)
        return fail(err, errlen, "frame video accepts frames, fps and optional frame_indices with total_num_frames", NULL, 0);
    int source_map[CLEF_VIDEO_MAX_FRAMES];
    if (source_indices) {
        long source_count;
        if (sample || frames->n > CLEF_VIDEO_MAX_FRAMES || source_indices->type != J_ARRAY || source_indices->n != frames->n ||
            !positive_int(total, &source_count) || source_count > CLEF_VIDEO_MAX_SOURCE_FRAMES || source_count / info.fps > 600)
            return fail(err, errlen, "frame_indices requires do_sample_frames=false, one index per frame and a bounded total_num_frames", NULL, 0);
        long previous = -1;
        for (size_t i = 0; i < source_indices->n; i++) {
            const jval *v = source_indices->items[i];
            long index;
            if (!nonnegative_int(v, &index) || index <= previous || index >= source_count)
                return fail(err, errlen, "frame_indices must be strictly increasing integers in [0,total_num_frames)", NULL, 0);
            source_map[i] = (int)index;
            previous = index;
        }
    }
    info.frames = (int)frames->n;
    if (!decode_frame(frames->items[0], 0, opts.vision.max_image_pixels, &first, err, errlen)) goto done;
    info.width = first.width; info.height = first.height;
    int count = info.frames;
    if (sample) {
        if (requested) count = requested > CLEF_VIDEO_MAX_FRAMES ? CLEF_VIDEO_MAX_FRAMES + 1 : (int)requested;
        else {
            double wanted = trunc(info.frames / info.fps * sample_fps);
            count = (int)fmin(fmax(wanted, 4), CLEF_VIDEO_MAX_FRAMES);
            if (count > info.frames) count = info.frames;
        }
    }
    if (count < 1 || count > info.frames || count > CLEF_VIDEO_MAX_FRAMES ||
        (opts.vision.max_video_frames > 0 && count > opts.vision.max_video_frames)) {
        snprintf(err, errlen, "video has %d sampled frames, above the limit of %d or its source count; lower videos_kwargs.fps/num_frames",
                 count, opts.vision.max_video_frames > 0 ? opts.vision.max_video_frames : CLEF_VIDEO_MAX_FRAMES);
        goto done;
    }
    int rh, rw;
    if (!clef_video_resize(count, info.height, info.width, min_px, max_px, &rh, &rw, err, errlen)) goto done;
    int groups = (count + 1) / 2;
    long tokens = (long)groups * (rh / 32) * (rw / 32);
    if (opts.vision.max_video_tokens > 0 && tokens > opts.vision.max_video_tokens) {
        snprintf(err, errlen, "video resizes to %ld tokens, above the limit of %ld per video; lower videos_kwargs.size.longest_edge", tokens, opts.vision.max_video_tokens);
        goto done;
    }
    int indices[CLEF_VIDEO_MAX_FRAMES];
    double timestamps[CLEF_VIDEO_MAX_FRAMES / 2];
    for (int i = 0; i < count; i++) indices[i] = sample && count > 1 ? (int)rint(i * ((info.frames - 1.0) / (count - 1))) : sample ? 0 : i;
    /* NumPy linspace explicitly writes its endpoint after multiplication. */
    if (sample && count > 1) indices[count - 1] = info.frames - 1;
    size_t extra = 2 + 2 * (size_t)groups;  /* reference retains the original outer video wrapper */
    for (int i = 0; i < groups; i++) {
        int a = indices[i * 2], b = indices[i * 2 + 1 < count ? i * 2 + 1 : i * 2];
        /* An adapter may have sampled already. Keep the original source indices and FPS,
         * otherwise timestamps shrink to the duration of the supplied frame array. */
        int source_a = source_indices ? source_map[a] : a;
        int source_b = source_indices ? source_map[b] : b;
        timestamps[i] = (source_a / info.fps + source_b / info.fps) / 2;
        char stamp[64]; snprintf(stamp, sizeof(stamp), "<%.1f seconds>", timestamps[i]);
        clef_tokens t = {0};
        bool encoded = tok_z(tok, stamp, false, &t);
        extra += t.len; clef_tokens_free(&t);
        if (!encoded) { fail(err, errlen, "out of memory (timestamp)", NULL, 0); goto done; }
    }
    if ((size_t)tokens + out->n_image_tokens + *reserve + extra > (size_t)opts.max_length) {
        fail(err, errlen, "video tokens and prompt cannot fit the context", NULL, 0); goto done;
    }
    clef_image_ref *refs = realloc(out->images, ((size_t)out->n_images + groups) * sizeof(*refs));
    if (!refs) { fail(err, errlen, "out of memory (video groups)", NULL, 0); goto done; }
    out->images = refs;
    for (int i = 0; i < groups; i++) {
        int a = indices[2 * i], b = indices[2 * i + 1 < count ? 2 * i + 1 : 2 * i];
        /* Uniform sampling starts at frame zero, already decoded for geometry and limits. */
        if (i && !decode_frame(frames->items[a], a, opts.vision.max_image_pixels, &first, err, errlen)) goto done;
        if (a != b && !decode_frame(frames->items[b], b, opts.vision.max_image_pixels, &second, err, errlen)) goto done;
        if (first.width != info.width || first.height != info.height ||
            (a != b && (second.width != info.width || second.height != info.height))) {
            fail(err, errlen, "video frame dimensions differ", NULL, 0); goto done;
        }
        clef_image_ref *r = &out->images[out->n_images];
        memset(r, 0, sizeof(*r));
        if (!clef_video_pair_preprocess(&first, a == b ? &first : &second, rh, rw, &r->pt, err, errlen)) goto done;
        r->video_group = group; r->timestamp = timestamps[i];
        out->n_images++; out->n_image_tokens += r->pt.n_tokens;
        clef_rgb_free(&first); clef_rgb_free(&second);
    }
    *reserve += extra;
    ok = true;
done:
    clef_rgb_free(&first); clef_rgb_free(&second);
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

/* Decodes and preprocesses images and video frame pairs into out->images, after the schema
 * and any untruncated state have been tokenized. Their tokens count toward every media budget. */
static bool encode_media(const clef_tokenizer *tok, const jval *req, const jval *images, size_t n_images,
                         const jval *videos, size_t n_videos,
                         clef_encode_opts opts, size_t reserved_tokens, clef_record *out, char *err, size_t errlen) {
    /* media_kwargs: the reference forwards them to the processor; only its pixel bounds are
     * taken here (deliberate divergence: other processor arguments are rejected, not ignored).
     * The processor applies the bounds only when both are given and silently ignores a lone
     * one (Qwen2VLImageProcessor._standardize_kwargs); a lone bound is rejected here instead. */
    clef_image_params prm = opts.vision.image;
    const jval *mk = json_get(req, "media_kwargs");
    if (mk && mk->type != J_NULL) {
        if (mk->type != J_OBJECT) return fail(err, errlen, "media_kwargs must be an object", NULL, 0);
        /* The DOM merges repeated keys as json.loads does. Test bound presence explicitly,
         * since videos_kwargs is independent and does not supply a missing image bound. */
        if (!!json_get(mk, "min_pixels") != !!json_get(mk, "max_pixels")) return fail(err, errlen, "media_kwargs: give both min_pixels and max_pixels (the reference ignores one alone)", NULL, 0);
        for (size_t i = 0; i < mk->n; i++) {
            const jmember *m = &mk->members[i];
            if (n_videos && m->klen == 13 && !memcmp(m->key, "videos_kwargs", 13)) continue;
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
     * newline after them, the schema and, when truncation is refused, the state always come with
     * them (reserved_tokens holds the last two): an image is refused before
     * preprocessing when it, the images before it and all of those cannot fit the context. That
     * is exactly the final length check's sum less the later images, so it refuses nothing the
     * reference accepts, only earlier. A per-image comparison with the whole context let an image
     * of exactly 16,384 tokens allocate 384 MiB of patches, and several large images each pass
     * it, before the length check refused the request (review #3); without the schema, a request
     * whose schema could never fit still preprocessed its images (review #3, Codex on 600ddfe),
     * and so did one whose state could not (Codex on a99a8a9). */
    size_t reserve = 1 + 2 * n_images + reserved_tokens;
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
    if (n_images) {
        out->images = calloc(n_images, sizeof(*out->images));
        if (!out->images) return fail(err, errlen, "out of memory", NULL, 0);
    }
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
            snprintf(err, errlen, "images[%zu]: %dx%d resizes to %ld tokens; with %d image tokens before it and "
                     "%zu tokens of prompt, schema and untruncated state the request cannot fit %d tokens",
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
    const jval *vk = mk && mk->type == J_OBJECT ? json_get(mk, "videos_kwargs") : NULL;
    for (size_t i = 0; i < n_videos; i++) {
        char verr[256];
        if (!encode_video(tok, videos->items[i], vk, (int)i + 1, opts, &reserve, out, verr, sizeof(verr))) {
            snprintf(err, errlen, "videos[%zu]: %s", i, verr); return false;
        }
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
    if (videos && videos->type != J_NULL && videos->type != J_ARRAY)
        return fail(err, errlen, "videos must be a list", NULL, 0);
    const size_t n_videos = videos && videos->type == J_ARRAY ? videos->n : 0;
    if (n_videos && (!opts.vision.image_token_id || !opts.vision.video_token_id))
        return fail(err, errlen, "videos are not supported by this model file (no vision tower)", NULL, 0);
    if (n_videos > INT32_MAX || (opts.vision.max_videos > 0 && n_videos > (size_t)opts.vision.max_videos))
        return fail(err, errlen, "too many videos per request", NULL, 0);
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

    /* With truncation refused the state is never cut, so like the schema it is part of each media
     * budget: tokenize it before the media, at most max_length + 1 tokens (enough to know it cannot
     * fit; a limited encode is a prefix of the full one). The state check below then refuses nothing
     * new, but before any patches exist rather than after (Codex on a99a8a9). With truncation the
     * state yields to the media and is tokenized later, as before. */
    bool state_done = false;
    size_t state_reserve = 0;
    if (ok && (n_images || n_videos) && opts.reject_truncation) {
        size_t cap = (size_t)opts.max_length + 1;
        if (opts.max_state_tokens >= 0 && (size_t)opts.max_state_tokens + 1 < cap) cap = (size_t)opts.max_state_tokens + 1;
        b.len = 0;
        render(&b, state);
        ok = !b.oom && clef_tok_encode_ex(tok, b.p ? b.p : "", b.len, cap, opts.strict, &state_ids);
        b.len = 0;
        state_done = ok;
        state_reserve = state_ids.len;
    }

    /* Media follows the schema and that state, which count toward every allocation budget. */
    if (ok && (n_images || n_videos) && !encode_media(tok, req, images, n_images, videos, n_videos,
                                                   opts, schema.len + state_reserve, out, err, errlen)) {
        jbuf_free(&b);
        clef_tokens_free(&schema);
        clef_tokens_free(&state_ids);
        return false;
    }

    if (ok) {
        jbuf_puts(&b, PROMPT_HEAD);
        jbuf_puts(&b, SYSTEM_PROMPT);
        jbuf_puts(&b, PROMPT_USER);
        ok = tok_jbuf(tok, &b, false, &prefix);   /* template: real control tokens */
        if (ok && out->n_images) {
            /* _encode_media: "<|vision_start|><|image_pad|><|vision_end|>" per image, then "\n",
             * tokenized by the processor, which expands each <|image_pad|> to the image's tokens.
             * These are engine-made control tokens, never request text, in both modes. */
            int video_group = 0;
            for (int i = 0; ok && i < out->n_images; i++) {
                clef_image_ref *ir = &out->images[i];
                if (ir->video_group != video_group) {
                    if (video_group) ok = clef_tokens_push(&prefix, opts.vision.end_token_id);
                    video_group = ir->video_group;
                    if (video_group) ok = ok && clef_tokens_push(&prefix, opts.vision.start_token_id);
                }
                if (video_group) {
                    char stamp[64]; snprintf(stamp, sizeof(stamp), "<%.1f seconds>", ir->timestamp);
                    ok = ok && tok_z(tok, stamp, false, &prefix);
                }
                ok = ok && clef_tokens_push(&prefix, opts.vision.start_token_id);
                ir->tok_start = (int32_t)prefix.len;
                for (int k = 0; ok && k < ir->pt.n_tokens; k++) ok = clef_tokens_push(&prefix, video_group ? opts.vision.video_token_id : opts.vision.image_token_id);
                ok = ok && clef_tokens_push(&prefix, opts.vision.end_token_id);
            }
            if (video_group) ok = ok && clef_tokens_push(&prefix, opts.vision.end_token_id);
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
        if (!state_done) {
            render(&b, state);
            /* one token past the budget is enough to know whether truncation would happen */
            const size_t probe = opts.reject_truncation ? keep + 1 : keep;
            ok = ok && !b.oom && clef_tok_encode_ex(tok, b.p ? b.p : "", b.len, probe, opts.strict, &state_ids);
            b.len = 0;
        }
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
        int32_t pads = 0, video_pads = 0, video_features = 0;
        for (size_t i = 0; i < out->ids.len; i++) {
            pads += out->ids.ids[i] == opts.vision.image_token_id;
            video_pads += out->ids.ids[i] == opts.vision.video_token_id;
        }
        for (int i = 0; i < out->n_images; i++) if (out->images[i].video_group) video_features += out->images[i].pt.n_tokens;
        if (pads != out->n_image_tokens - video_features || video_pads != video_features) {
            snprintf(err, errlen, "%s placeholder tokens in request content do not match features",
                     pads != out->n_image_tokens - video_features ? "image" : "video");
            return false;
        }
    }
    return true;
}

/* An integer JSON value equal to want (J_INT keeps its canonical decimal text). */
static bool json_int_is(const jval *v, long long want) {
    char buf[24];
    const int n = snprintf(buf, sizeof(buf), "%lld", want);
    return v && v->type == J_INT && v->len == (size_t)n && !memcmp(v->s, buf, (size_t)n);
}

static bool json_halves(const jval *v) {   /* [0.5, 0.5, 0.5] */
    if (!v || v->type != J_ARRAY || v->n != 3) return false;
    for (size_t i = 0; i < 3; i++) if (v->items[i]->type != J_FLOAT || v->items[i]->f != 0.5) return false;
    return true;
}

/* clef_image.c hard-codes the reference's Qwen2VLImageProcessor: bicubic resample, rescale by
 * 1/255, normalize with mean and std 0.5, RGB conversion. tools/convert.py refuses any other
 * processor and records the one it accepted as clef.vision.image_processor, but the loader never
 * read it, so a GGUF from elsewhere describing other preprocessing loaded and was served with this
 * one (Codex on a82292d). Each field the converter checks is checked again here, and the recorded
 * geometry and pixel bounds must equal the numeric keys the engine uses. */
static bool processor_supported(const gguf_file *f, uint32_t patch, uint32_t merge, uint32_t temporal,
                                uint32_t min_px, uint32_t max_px, char *err, size_t errlen) {
    gguf_str text;
    if (!gguf_get_str(f, "clef.vision.image_processor", &text)) {
        snprintf(err, errlen, "model: incomplete clef.vision.* keys (no image_processor)");
        return false;
    }
    jarena *a = jarena_new();
    if (!a) { snprintf(err, errlen, "out of memory"); return false; }
    char jerr[128];
    const jval *ip = text.len <= (1u << 20) ? json_parse(a, text.ptr, (size_t)text.len, jerr, sizeof(jerr)) : NULL;
    const jval *size = ip && ip->type == J_OBJECT ? json_get(ip, "size") : NULL;
    const jval *rescale = ip && ip->type == J_OBJECT ? json_get(ip, "rescale_factor") : NULL;
    const bool ok = ip && ip->type == J_OBJECT &&
        json_str_eq(json_get(ip, "image_processor_type"), "Qwen2VLImageProcessor") &&
        json_int_is(json_get(ip, "resample"), 3) &&
        json_halves(json_get(ip, "image_mean")) && json_halves(json_get(ip, "image_std")) &&
        rescale && rescale->type == J_FLOAT && fabs(rescale->f - 1.0 / 255) <= 1e-12 &&
        json_get(ip, "do_convert_rgb") && json_get(ip, "do_convert_rgb")->type == J_TRUE &&
        json_get(ip, "do_resize") && json_get(ip, "do_resize")->type == J_TRUE &&
        json_get(ip, "do_rescale") && json_get(ip, "do_rescale")->type == J_TRUE &&
        json_get(ip, "do_normalize") && json_get(ip, "do_normalize")->type == J_TRUE &&
        json_int_is(json_get(ip, "patch_size"), patch) && json_int_is(json_get(ip, "merge_size"), merge) &&
        json_int_is(json_get(ip, "temporal_patch_size"), temporal) &&
        size && size->type == J_OBJECT && json_int_is(json_get(size, "shortest_edge"), min_px) &&
        json_int_is(json_get(size, "longest_edge"), max_px);
    jarena_free(a);
    if (!ok) snprintf(err, errlen, "model: clef.vision.image_processor describes preprocessing this engine does not implement");
    return ok;
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
    /* The encoder emits start, the image's placeholders, end; the reference finds images by its
     * start token and counts placeholders by the image token. A start or end id equal to the image
     * id made every image request fail its placeholder count after decoding and preprocessing
     * (Codex on e4f4e1d), so the four ids must differ, as in the released models. */
    if (image_id == start_id || image_id == end_id || image_id == video_id || start_id == end_id ||
        start_id == video_id || end_id == video_id) {
        snprintf(err, errlen, "model: vision token ids must be distinct (image %u, start %u, end %u, video %u)",
                 image_id, start_id, end_id, video_id);
        return false;
    }
    if (!processor_supported(f, patch, merge, temporal, min_px, max_px, err, errlen)) return false;
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
