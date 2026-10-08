/* Test driver for the host-side pieces that must match Python byte for byte.
 *
 *   clef-tool json  < lines      each line: JSON text -> json.dumps(sort_keys) or "ERR <msg>"
 *   clef-tool float < lines      each line: 16 hex digits (IEEE-754 bits) -> repr(float)
 *   clef-tool round < lines      each line: "<hexbits> <ndigits>" -> repr(round(x, ndigits))
 *   clef-tool tok MODEL.gguf < lines   each line: a JSON string -> space-separated token ids
 *   clef-tool encode MODEL.gguf < lines   each line: a SystemOne request -> encoded record JSON
 *   clef-tool encode-strict MODEL.gguf    same, strict tokenization of request content
 *   clef-tool encode-notrunc MODEL.gguf   same, but reject a state that would be truncated
 *   clef-tool respond MODEL.gguf < lines  each line: {"request":..., "probs":[[...],...]} -> response JSON
 *   clef-tool image < lines      each line: {"image": base64 or data URL, "out": PREFIX, "min_pixels": N,
 *                                "max_pixels": N} -> {"width","height","grid_h","grid_w","n_tokens"} or "ERR <msg>";
 *                                writes PREFIX.rgb (decoded), PREFIX.resized (resized RGB), PREFIX.patches (f32
 *                                [n_patch][1536]) and PREFIX.pos (int32 [n_patch][4] indices, f32 [n_patch][4] weights)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../clef_gguf.h"
#include "../clef_image.h"
#include "../clef_json.h"
#include "../clef_record.h"
#include "../clef_tok.h"

static char *read_line(FILE *f, size_t *len) {
    char *line = NULL;
    size_t cap = 0;
    ssize_t n = getline(&line, &cap, f);
    if (n < 0) { free(line); return NULL; }
    if (n && line[n - 1] == '\n') line[--n] = '\0';
    *len = (size_t)n;
    return line;
}

static double bits_to_double(const char *hex) {
    unsigned long long u = strtoull(hex, NULL, 16);
    double d;
    memcpy(&d, &u, 8);
    return d;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: clef-tool json|float|round|tok [model]\n"); return 2; }
    const char *mode = argv[1];
    char *line;
    size_t len;
    char err[256];

    if (!strcmp(mode, "json")) {
        while ((line = read_line(stdin, &len))) {
            jarena *a = jarena_new();
            jval *v = json_parse(a, line, len, err, sizeof(err));
            if (!v) {
                printf("ERR %s\n", err);
            } else {
                jbuf b = {0};
                json_dump(&b, v, true);
                printf("%s\n", b.oom ? "ERR oom" : b.p);
                jbuf_free(&b);
            }
            jarena_free(a);
            free(line);
        }
        return 0;
    }
    if (!strcmp(mode, "float")) {
        while ((line = read_line(stdin, &len))) {
            jbuf b = {0};
            json_put_float(&b, bits_to_double(line));
            printf("%s\n", b.p);
            jbuf_free(&b);
            free(line);
        }
        return 0;
    }
    if (!strcmp(mode, "round")) {
        while ((line = read_line(stdin, &len))) {
            char *sp = strchr(line, ' ');
            if (!sp) { free(line); return 1; }
            *sp = '\0';
            jbuf b = {0};
            json_put_float(&b, py_round(bits_to_double(line), atoi(sp + 1)));
            printf("%s\n", b.p);
            jbuf_free(&b);
            free(line);
        }
        return 0;
    }
    if (!strcmp(mode, "image")) {
        while ((line = read_line(stdin, &len))) {
            jarena *a = jarena_new();
            jval *v = json_parse(a, line, len, err, sizeof(err));
            const jval *img = v ? json_get(v, "image") : NULL, *out = v ? json_get(v, "out") : NULL;
            const jval *mn = v ? json_get(v, "min_pixels") : NULL, *mx = v ? json_get(v, "max_pixels") : NULL;
            if (!v || !img || img->type != J_STRING || !out || out->type != J_STRING) {
                printf("ERR %s\n", v ? "need image and out strings" : err);
            } else {
                clef_image_params prm = { 65536, 16777216, 16, 2, 2 };
                if (mn && mn->type == J_INT) prm.min_pixels = atol(mn->s);
                if (mx && mx->type == J_INT) prm.max_pixels = atol(mx->s);
                uint8_t *bytes = NULL;
                size_t n = 0;
                clef_rgb rgb = {0};
                clef_image_patches pt = {0};
                bool ok = clef_base64_decode(img->s, img->len, &bytes, &n, err, sizeof(err)) &&
                          clef_image_decode(bytes, n, &rgb, err, sizeof(err)) &&
                          clef_image_preprocess(&rgb, &prm, &pt, err, sizeof(err));
                if (ok) {
                    char path[1024];
                    uint8_t *resized = malloc((size_t)pt.width * pt.height * 3);
                    int32_t *idx = malloc((size_t)pt.n_patch * 4 * sizeof(int32_t));
                    float *w = malloc((size_t)pt.n_patch * 4 * sizeof(float));
                    ok = resized && idx && w &&
                         clef_resize_bicubic_aa(rgb.rgb, rgb.width, rgb.height, resized, pt.width, pt.height, err, sizeof(err));
                    if (ok) clef_image_pos_interp(pt.grid_h, pt.grid_w, 48, 2, idx, w);
                    const struct { const char *suffix; const void *data; size_t bytes; } files[] = {
                        { ".rgb", rgb.rgb, (size_t)rgb.width * rgb.height * 3 },
                        { ".resized", resized, (size_t)pt.width * pt.height * 3 },
                        { ".patches", pt.patches, (size_t)pt.n_patch * pt.patch_dim * sizeof(float) },
                        { ".pos", idx, (size_t)pt.n_patch * 4 * sizeof(int32_t) },
                    };
                    for (size_t i = 0; ok && i < 4; i++) {
                        snprintf(path, sizeof(path), "%.*s%s", (int)out->len, out->s, files[i].suffix);
                        FILE *f = fopen(path, i == 3 ? "wb" : "wb");
                        ok = f && fwrite(files[i].data, 1, files[i].bytes, f) == files[i].bytes;
                        if (ok && i == 3) ok = fwrite(w, sizeof(float), (size_t)pt.n_patch * 4, f) == (size_t)pt.n_patch * 4;
                        if (f && fclose(f)) ok = false;
                        if (!ok) snprintf(err, sizeof(err), "cannot write %s", path);
                    }
                    if (ok) printf("{\"width\":%d,\"height\":%d,\"grid_h\":%d,\"grid_w\":%d,\"n_tokens\":%d,\"decoded\":[%d,%d]}\n",
                                   pt.width, pt.height, pt.grid_h, pt.grid_w, pt.n_tokens, rgb.width, rgb.height);
                    free(resized); free(idx); free(w);
                }
                if (!ok) printf("ERR %s\n", err);
                clef_image_patches_free(&pt);
                clef_rgb_free(&rgb);
                free(bytes);
            }
            jarena_free(a);
            free(line);
        }
        return 0;
    }
    if (!strcmp(mode, "tok")) {
        if (argc < 3) { fprintf(stderr, "tok needs a model path\n"); return 2; }
        gguf_file f;
        if (!gguf_open(&f, argv[2], err, sizeof(err))) { fprintf(stderr, "%s\n", err); return 1; }
        clef_tokenizer *t = clef_tok_load(&f, err, sizeof(err));
        if (!t) { fprintf(stderr, "%s\n", err); return 1; }
        while ((line = read_line(stdin, &len))) {
            jarena *a = jarena_new();
            jval *v = json_parse(a, line, len, err, sizeof(err));
            if (!v || v->type != J_STRING) {
                printf("ERR %s\n", v ? "not a string" : err);
            } else {
                clef_tokens toks = {0};
                if (!clef_tok_encode(t, v->s, v->len, &toks)) { printf("ERR oom\n"); }
                else {
                    for (size_t i = 0; i < toks.len; i++) printf(i ? " %d" : "%d", toks.ids[i]);
                    printf("\n");
                }
                clef_tokens_free(&toks);
            }
            jarena_free(a);
            free(line);
        }
        clef_tok_free(t);
        gguf_close(&f);
        return 0;
    }
    if (!strcmp(mode, "encode") || !strcmp(mode, "encode-strict") || !strcmp(mode, "encode-notrunc") || !strcmp(mode, "respond")) {
        if (argc < 3) { fprintf(stderr, "%s needs a model path\n", mode); return 2; }
        gguf_file f;
        if (!gguf_open(&f, argv[2], err, sizeof(err))) { fprintf(stderr, "%s\n", err); return 1; }
        clef_tokenizer *t = clef_tok_load(&f, err, sizeof(err));
        if (!t) { fprintf(stderr, "%s\n", err); return 1; }
        bool respond = !strcmp(mode, "respond");
        clef_vision_opts vision = {0};
        if (!clef_vision_opts_load(&f, &vision, err, sizeof(err))) { fprintf(stderr, "%s\n", err); return 1; }
        while ((line = read_line(stdin, &len))) {
            jarena *a = jarena_new();
            jval *doc = json_parse(a, line, len, err, sizeof(err));
            jval *req = respond ? json_get(doc, "request") : doc;
            clef_record rec;
            clef_encode_opts opts = CLEF_ENCODE_DEFAULTS;
            opts.strict = !strcmp(mode, "encode-strict");
            opts.reject_truncation = !strcmp(mode, "encode-notrunc");
            opts.vision = vision;
            if (!doc) {
                printf("ERR %s\n", err);
            } else if (!clef_encode_request(t, req, opts, &rec, err, sizeof(err))) {
                printf("ERR %s\n", err);
            } else if (respond) {
                const jval *pj = json_get(doc, "probs");
                float **probs = calloc((size_t)rec.nq, sizeof(*probs));
                for (int i = 0; i < rec.nq; i++) {
                    probs[i] = calloc((size_t)rec.q[i].n_opt, sizeof(float));
                    for (int k = 0; k < rec.q[i].n_opt; k++) probs[i][k] = (float)pj->items[i]->items[k]->f;
                }
                jbuf b = {0};
                clef_build_response(req, &rec, probs, &b);
                printf("%s\n", b.p);
                jbuf_free(&b);
                for (int i = 0; i < rec.nq; i++) free(probs[i]);
                free(probs);
                clef_record_free(&rec);
            } else {
                jbuf b = {0};
                jbuf_puts(&b, "{\"input_ids\":[");
                char num[32];
                for (size_t i = 0; i < rec.ids.len; i++) {
                    snprintf(num, sizeof(num), i ? ",%d" : "%d", rec.ids.ids[i]);
                    jbuf_puts(&b, num);
                }
                jbuf_puts(&b, "],\"questions\":[");
                for (int i = 0; i < rec.nq; i++) {
                    const clef_question *q = &rec.q[i];
                    if (i) jbuf_put(&b, ",", 1);
                    jbuf_puts(&b, "{\"id\":");
                    json_put_string(&b, q->id, q->id_len);
                    snprintf(num, sizeof(num), ",\"type\":%d", q->type);
                    jbuf_puts(&b, num);
                    snprintf(num, sizeof(num), ",\"span\":[%d,%d]", q->span[0], q->span[1]);
                    jbuf_puts(&b, num);
                    jbuf_puts(&b, ",\"option_spans\":[");
                    for (int k = 0; k < q->n_opt; k++) {
                        snprintf(num, sizeof(num), k ? ",[%d,%d]" : "[%d,%d]", q->opt_span[k][0], q->opt_span[k][1]);
                        jbuf_puts(&b, num);
                    }
                    jbuf_puts(&b, "],\"option_ids\":[");
                    for (int k = 0; k < q->n_opt; k++) {
                        if (k) jbuf_put(&b, ",", 1);
                        json_put_string(&b, q->opt_id[k], q->opt_id_len[k]);
                    }
                    jbuf_puts(&b, "]}");
                }
                jbuf_puts(&b, "]");
                if (rec.n_images) {
                    /* images: token offset and merged grid of each; position_ids as the reference's
                     * get_rope_index lays them out, [3][T] */
                    jbuf_puts(&b, ",\"images\":[");
                    for (int i = 0; i < rec.n_images; i++) {
                        snprintf(num, sizeof(num), i ? ",[%d,%d,%d]" : "[%d,%d,%d]", rec.images[i].tok_start,
                                 rec.images[i].pt.grid_h, rec.images[i].pt.grid_w);
                        jbuf_puts(&b, num);
                    }
                    int32_t *pos3 = malloc(rec.ids.len * 3 * sizeof(int32_t));
                    if (!pos3) { fprintf(stderr, "out of memory\n"); return 1; }
                    clef_record_positions(&rec, pos3);
                    jbuf_puts(&b, "],\"position_ids\":[");
                    for (int axis = 0; axis < 3; axis++) {
                        jbuf_puts(&b, axis ? ",[" : "[");
                        for (size_t t = 0; t < rec.ids.len; t++) {
                            snprintf(num, sizeof(num), t ? ",%d" : "%d", pos3[3 * t + axis]);
                            jbuf_puts(&b, num);
                        }
                        jbuf_puts(&b, "]");
                    }
                    jbuf_puts(&b, "]");
                    free(pos3);
                }
                jbuf_puts(&b, "}");
                printf("%s\n", b.p);
                jbuf_free(&b);
                clef_record_free(&rec);
            }
            jarena_free(a);
            free(line);
        }
        clef_tok_free(t);
        gguf_close(&f);
        return 0;
    }
    fprintf(stderr, "unknown mode %s\n", mode);
    return 2;
}
