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
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../clef_gguf.h"
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
        while ((line = read_line(stdin, &len))) {
            jarena *a = jarena_new();
            jval *doc = json_parse(a, line, len, err, sizeof(err));
            jval *req = respond ? json_get(doc, "request") : doc;
            clef_record rec;
            clef_encode_opts opts = CLEF_ENCODE_DEFAULTS;
            opts.strict = !strcmp(mode, "encode-strict");
            opts.reject_truncation = !strcmp(mode, "encode-notrunc");
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
                jbuf_puts(&b, "]}");
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
