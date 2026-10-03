/* clef: run SystemOne requests through a Clef GGUF.
 *
 *   clef -m MODEL.gguf [requests.jsonl]     one request per line (stdin if no file);
 *                                            one SystemOne response (or {"error":...}) per line
 *   --logits      print raw per-question logits instead of responses (parity testing)
 *   --dump FILE   write [n_layer+2][R][H] f32 residual dumps of the first request
 *   --dump-last N dump only the last N token rows of each layer (R = N; default all T)
 *   --batch N     pack up to N requests per forward pass (default 1)
 *   --time        print per-batch latency to stderr
 *   --strict      tokenize request content without special-token recognition (see README)
 *   --no-truncate reject requests whose state does not fit, instead of silently cutting it
 */

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "clef_engine.h"

bool clef_run_ex(clef_engine *e, const clef_record *recs, int n, float ****out, bool raw, float *dump,
                 int dump_rows, char *err, size_t errlen);

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

/* Each item owns its input line: the JSON DOM points into it (unescaped strings, integer
 * text, question and option ids), and responses are built after the whole batch is read. */
typedef struct { char *line; jarena *arena; jval *req; clef_record rec; bool ok, encoded; char err[256]; } item;

static void free_items(item *items, int n) {
    for (int i = 0; i < n; i++) {
        if (items[i].encoded) clef_record_free(&items[i].rec);
        jarena_free(items[i].arena);
        free(items[i].line);
    }
}

static bool emit(item *it, float **p, bool logits) {
    if (!it->ok) {
        jbuf b = {0};
        jbuf_puts(&b, "{\"error\":");
        json_put_string(&b, it->err, strlen(it->err));
        jbuf_puts(&b, "}");
        bool ok = printf("%s\n", b.oom ? "{\"error\":\"out of memory\"}" : b.p) >= 0;
        jbuf_free(&b);
        return ok;
    }
    jbuf b = {0};
    if (logits) {
        jbuf_puts(&b, "{");
        for (int q = 0; q < it->rec.nq; q++) {
            if (q) jbuf_put(&b, ",", 1);
            json_put_string(&b, it->rec.q[q].id, it->rec.q[q].id_len);
            jbuf_puts(&b, ":[");
            for (int k = 0; k < it->rec.q[q].n_opt; k++) {
                if (k) jbuf_put(&b, ",", 1);
                json_put_float(&b, (double)p[q][k]);
            }
            jbuf_puts(&b, "]");
        }
        jbuf_puts(&b, "}");
    } else {
        clef_build_response(it->req, &it->rec, p, &b);
    }
    bool ok = printf("%s\n", b.oom ? "{\"error\":\"out of memory\"}" : b.p) >= 0;
    jbuf_free(&b);
    return ok;
}

int main(int argc, char **argv) {
    const char *model = NULL, *input = NULL, *dump_path = NULL;
    bool logits = false, timing = false, strict = false, no_truncate = false;
    int batch = 1;
    int dump_last = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "--logits")) logits = true;
        else if (!strcmp(argv[i], "--time")) timing = true;
        else if (!strcmp(argv[i], "--strict")) strict = true;
        else if (!strcmp(argv[i], "--no-truncate")) no_truncate = true;
        else if (!strcmp(argv[i], "--dump") && i + 1 < argc) dump_path = argv[++i];
        else if (!strcmp(argv[i], "--dump-last") && i + 1 < argc) dump_last = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--batch") && i + 1 < argc) batch = atoi(argv[++i]);
        else if (argv[i][0] != '-' && !input) input = argv[i];
        else { fprintf(stderr, "usage: clef -m MODEL.gguf [--logits] [--time] [--strict] [--no-truncate] [--batch N] [--dump FILE [--dump-last N]] [requests.jsonl]\n"); return 2; }
    }
    if (!model || batch < 1) { fprintf(stderr, "clef: -m MODEL.gguf is required\n"); return 2; }
    FILE *in = input ? fopen(input, "r") : stdin;
    if (!in) { fprintf(stderr, "clef: cannot open %s\n", input); return 1; }

    char err[512];
    double t0 = now_ms();
    clef_engine *e = clef_open(model, err, sizeof(err));
    if (!e) { fprintf(stderr, "clef: %s\n", err); if (in != stdin) fclose(in); return 1; }
    if (timing) fprintf(stderr, "clef: loaded %s in %.0f ms\n", model, now_ms() - t0);

    item *items = calloc((size_t)batch, sizeof(*items));
    clef_record *recs = calloc((size_t)batch, sizeof(*recs));
    int *map = calloc((size_t)batch, sizeof(int));
    if (!items || !recs || !map) {
        fprintf(stderr, "clef: out of memory (batch of %d)\n", batch);
        free(items); free(recs); free(map);
        if (in != stdin) fclose(in);
        clef_close(e);
        return 1;
    }
    char *line = NULL;
    size_t cap = 0;
    bool dumped = false;
    int rc = 0;
    int read_err = 0;
    int write_err = 0;
    for (;;) {
        int n = 0;
        ssize_t len = 0;
        while (n < batch && (len = getline(&line, &cap, in)) >= 0) {
            while (len && (line[len - 1] == '\n' || line[len - 1] == '\r')) line[--len] = '\0';
            if (!len) continue;
            item *it = &items[n++];
            memset(it, 0, sizeof(*it));
            it->line = line;   /* take ownership; getline allocates a fresh buffer next time */
            line = NULL;
            cap = 0;
            it->arena = jarena_new();
            if (!it->arena) {
                snprintf(it->err, sizeof(it->err), "out of memory");
                read_err = ENOMEM;
                break;
            }
            it->req = json_parse(it->arena, it->line, (size_t)len, it->err, sizeof(it->err));
            clef_encode_opts opts = CLEF_ENCODE_DEFAULTS;
            opts.strict = strict;
            opts.reject_truncation = no_truncate;
            it->ok = it->encoded = it->req && clef_encode_request(e->tok, it->req, opts, &it->rec, it->err, sizeof(it->err));
        }
        /* getline() returns -1 for EOF and for a read error (EISDIR, EIO, ENOMEM) alike; an error
         * must not pass for a complete input (review #4). Lines read before it are still answered. */
        if (len < 0 && ferror(in) && !read_err) read_err = errno ? errno : EIO;
        if (!n) break;
        int m = 0;
        for (int i = 0; i < n; i++) if (items[i].ok) { recs[m] = items[i].rec; map[m++] = i; }
        float ***probs = NULL;
        float *dump = NULL;
        size_t dump_rows = 0;
        if (dump_path && !dumped && m) {
            dump_rows = recs[0].ids.len;
            if (dump_last > 0 && (size_t)dump_last < dump_rows) dump_rows = (size_t)dump_last;
            dump = calloc((size_t)(e->cfg.n_layer + 2) * dump_rows * e->cfg.H, sizeof(float));
            if (!dump) {
                fprintf(stderr, "clef: cannot allocate dump\n");
                free_items(items, n);
                rc = 1;
                break;
            }
        }
        double t1 = now_ms();
        if (m && !clef_run_ex(e, recs, dump ? 1 : m, &probs, logits, dump, (int)dump_rows, err, sizeof(err))) {
            for (int i = 0; i < m; i++) { items[map[i]].ok = false; snprintf(items[map[i]].err, sizeof(items[map[i]].err), "%s", err); }
            rc = 1;
        } else if (dump) {
            FILE *df = fopen(dump_path, "wb");
            const size_t want = (size_t)(e->cfg.n_layer + 2) * dump_rows * e->cfg.H;
            bool wrote = df && fwrite(dump, sizeof(float), want, df) == want;   /* short write = failure */
            if (df && fclose(df) != 0) wrote = false;                           /* buffered data can fail here */
            if (!wrote) {
                fprintf(stderr, "clef: cannot write %s (incomplete dump)\n", dump_path);
                rc = 1;
            }
            dumped = true;
            /* the dump run covered only the first record; run the rest normally */
            if (m > 1) {
                float ***rest = NULL;
                float ***all = calloc((size_t)m, sizeof(*all));
                if (!all || !clef_run_ex(e, recs + 1, m - 1, &rest, logits, NULL, 0, err, sizeof(err))) {
                    fprintf(stderr, "clef: %s\n", all ? err : "out of memory (combined results)");
                    /* Until combined, probs owns one record, not m records. */
                    clef_free_probs(recs, 1, probs);
                    free(all); free(dump);
                    free_items(items, n);
                    rc = 1;
                    break;
                }
                all[0] = probs[0];
                for (int i = 1; i < m; i++) all[i] = rest[i - 1];
                free(probs); free(rest);
                probs = all;
            }
        }
        if (timing && m) {
            size_t toks = 0;
            for (int i = 0; i < m; i++) toks += recs[i].ids.len;
            fprintf(stderr, "clef: batch of %d (%zu tokens) in %.1f ms\n", m, toks, now_ms() - t1);
        }
        free(dump);
        int k = 0;
        for (int i = 0; i < n; i++) {
            errno = 0;
            if (!emit(&items[i], items[i].ok && probs ? probs[k] : NULL, logits)) {
                write_err = errno ? errno : EIO;
                break;
            }
            if (items[i].ok) k++;
        }
        clef_free_probs(recs, m, probs);
        free_items(items, n);
        /* Buffered output may fail only on flush. Stop after freeing the batch;
         * continuing inference cannot recover an output stream that lost results. */
        errno = 0;
        if (!write_err && fflush(stdout) == EOF) write_err = errno ? errno : EIO;
        if (write_err || read_err) break;
    }
    if (read_err) {
        fprintf(stderr, "clef: cannot read %s: %s\n", input ? input : "stdin", strerror(read_err));
        rc = 1;
    }
    if (write_err) {
        fprintf(stderr, "clef: cannot write stdout: %s\n", strerror(write_err));
        rc = 1;
    }
    free(line); free(items); free(recs); free(map);
    if (in != stdin) fclose(in);
    clef_close(e);
    return rc;
}
