/* clef-server: Jev/SystemOne-compatible HTTP endpoint.
 *
 *   clef-server -m MODEL.gguf [--host 127.0.0.1] [--port 8080] [--batch 8] [--batch-tokens 4096]
 *               [--max-body 8388608] [--max-conn 256]
 *
 *   POST /v1/systemone   SystemOne request body -> SystemOne response body
 *   GET  /health         {"status":"ok","model":...}
 *
 * One thread per connection parses HTTP/1.1 (Content-Length bodies only, keep-alive
 * supported). Requests are encoded on the connection thread (CPU) and queued; a single GPU
 * worker drains the queue in packed batches of up to --batch requests and --batch-tokens tokens. Engine results do
 * not depend on batch composition (tests/test_batch.sh), so batching is invisible to clients.
 *
 * Exposure: binds 127.0.0.1 by default. Every input is bounded: request line + headers
 * (16 KiB), body (--max-body), connections (--max-conn), and a per-socket I/O timeout.
 * Strict mode (default): request content is tokenized without special-token recognition, so a
 * literal "<|im_end|>" cannot close the template's user turn. --no-strict reproduces the
 * reference exactly, including that injection (README, Security notes).
 * Over-long state is rejected by default (HTTP 400 with token counts); --truncate keeps the
 * reference behaviour of silently dropping the end of the state.
 */

#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#include "clef_engine.h"

#define MAX_HEADER 16384
#define IO_TIMEOUT_S 30

typedef struct job {
    clef_record rec;
    const jval *req;
    float **probs;           /* filled by the worker */
    char err[256];
    bool ok, done;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    struct job *next;
} job;

static struct {
    clef_engine *e;
    int batch;
    size_t batch_tokens;
    bool truncate;           /* default off: over-long state is rejected (--truncate = reference) */
    bool strict;             /* default on: content cannot inject chat-control tokens (--no-strict = reference) */     /* token budget per forward pass (the head job is always admitted) */
    size_t max_body;
    int max_conn;
    double io_timeout;       /* seconds; request deadline and per-write timeout (IO_TIMEOUT_S) */
    atomic_int conns;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    job *head, *tail;
    const char *model_name;
} S;

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

/* ---- GPU worker -------------------------------------------------------------- */

static void *worker(void *arg) {
    (void)arg;
    job **batch = calloc((size_t)S.batch, sizeof(*batch));
    clef_record *recs = calloc((size_t)S.batch, sizeof(*recs));
    if (!batch || !recs) { fprintf(stderr, "clef-server: out of memory\n"); exit(1); }
    for (;;) {
        pthread_mutex_lock(&S.mu);
        while (!S.head) pthread_cond_wait(&S.cv, &S.mu);
        /* FIFO; take jobs from the head while the batch stays within the token budget. A
         * request larger than the budget runs alone, so small requests are never packed into
         * a long request's forward pass (review #2, I1). The head is always admitted, so
         * nothing starves; a job queued behind a long one still waits for it (one GPU, no
         * preemption). */
        int n = 0;
        size_t toks = 0;
        while (S.head && n < S.batch &&
               (n == 0 || toks + S.head->rec.ids.len <= S.batch_tokens)) {
            toks += S.head->rec.ids.len;
            batch[n++] = S.head;
            S.head = S.head->next;
        }
        if (!S.head) S.tail = NULL;
        pthread_mutex_unlock(&S.mu);

        for (int i = 0; i < n; i++) recs[i] = batch[i]->rec;
        float ***probs = NULL;
        char err[256] = "";   /* clef_run only writes it on failure */
        double t0 = now_ms();
        bool ok = clef_run(S.e, recs, n, &probs, err, sizeof(err));
        fprintf(stderr, "clef-server: batch %d (%zu tokens) %.1f ms%s%s\n", n, toks, now_ms() - t0,
                ok ? "" : " error: ", ok ? "" : err);
        for (int i = 0; i < n; i++) {
            job *j = batch[i];
            float ***one = NULL;
            bool jok = ok;
            char jerr[256] = "";
            if (!ok) snprintf(jerr, sizeof(jerr), "%s", err);
            if (!ok && n > 1) {
                /* a batch-level failure (e.g. allocation for the combined length) must not
                 * fail unrelated co-batched requests: retry each record on its own (M2) */
                jok = clef_run(S.e, &recs[i], 1, &one, jerr, sizeof(jerr));
            }
            pthread_mutex_lock(&j->mu);
            j->ok = jok;
            if (jok && ok) { j->probs = probs[i]; probs[i] = NULL; }
            else if (jok) { j->probs = one[0]; free(one); }
            else snprintf(j->err, sizeof(j->err), "%s", jerr);
            j->done = true;
            pthread_cond_signal(&j->cv);
            pthread_mutex_unlock(&j->mu);
        }
        free(probs);   /* per-record arrays now belong to the jobs */
    }
    return NULL;
}

/* ---- HTTP -------------------------------------------------------------------- */

static bool write_all(int fd, const char *p, size_t n) {
    while (n) {
        ssize_t w = write(fd, p, n);
        if (w <= 0) { if (w < 0 && errno == EINTR) continue; return false; }
        p += w;
        n -= (size_t)w;
    }
    return true;
}

static bool respond(int fd, int status, const char *body, size_t len, bool keep_alive) {
    const char *reason = status == 200 ? "OK" : status == 400 ? "Bad Request" : status == 404 ? "Not Found"
                       : status == 405 ? "Method Not Allowed" : status == 413 ? "Payload Too Large" : status == 431 ? "Request Header Fields Too Large"
                       : status == 503 ? "Service Unavailable" : "Internal Server Error";
    char head[256];
    int hn = snprintf(head, sizeof(head),
                      "HTTP/1.1 %d %s\r\nContent-Type: application/json\r\nContent-Length: %zu\r\n"
                      "Connection: %s\r\n\r\n", status, reason, len, keep_alive ? "keep-alive" : "close");
    if (hn < 0 || (size_t)hn >= sizeof(head)) return false;   /* never send a truncated header */
    return write_all(fd, head, (size_t)hn) && write_all(fd, body, len);
}

static bool respond_error(int fd, int status, const char *msg, bool keep_alive) {
    jbuf b = {0};
    jbuf_puts(&b, "{\"error\":");
    json_put_string(&b, msg, strlen(msg));
    jbuf_puts(&b, "}");
    bool ok = !b.oom && respond(fd, status, b.p, b.len, keep_alive);
    jbuf_free(&b);
    return ok;
}

/* SO_RCVTIMEO bounds each read only, so a client that keeps trickling a byte at a time could hold
 * a connection (and one of --max-conn slots) for as long as it liked: slowloris (review #4,
 * tests/test_server_slow.py). Reads therefore run against a monotonic deadline: before each read
 * the socket timeout is set to the time left, and none left means give up. */
static double now_s(void) { return now_ms() / 1e3; }

static bool read_deadline(int fd, double deadline) {
    const double left = deadline - now_s();
    if (left <= 0) return false;
    struct timeval tv = { (time_t)left, (suseconds_t)((left - (double)(time_t)left) * 1e6) };
    if (tv.tv_sec == 0 && tv.tv_usec == 0) tv.tv_usec = 1;   /* zero would mean no timeout */
    return setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) == 0;
}

/* Reads one request. Returns 0 on success, -1 on EOF/IO error/deadline (close silently), or an
 * HTTP status for a protocol error to report. *buf keeps any bytes read past the body. The whole
 * request, headers and body, must arrive within S.io_timeout of the call (on a keep-alive
 * connection that includes waiting for it). */
typedef struct { char *p; size_t len, cap; } rbuf;

static int read_request(int fd, rbuf *in, char **method, char **path, char **body, size_t *body_len, bool *keep_alive) {
    const double deadline = now_s() + S.io_timeout;
    size_t hdr_end = 0;
    for (;;) {
        char *e = in->len >= 4 ? memmem(in->p, in->len, "\r\n\r\n", 4) : NULL;
        if (e) {
            hdr_end = (size_t)(e - in->p) + 4;
            /* the limit applies to the header itself, whatever a single read delivered */
            if (hdr_end > MAX_HEADER) return 431;
            break;
        }
        if (in->len >= MAX_HEADER) return 431;
        if (in->cap - in->len < 4096) {
            size_t cap = in->cap ? in->cap * 2 : 8192;
            char *p = realloc(in->p, cap);
            if (!p) return 500;
            in->p = p;
            in->cap = cap;
        }
        /* bound header-stage reads by the header limit, not by capacity retained from an
         * earlier body on this keep-alive connection */
        size_t want = in->cap - in->len;
        if (want > MAX_HEADER + 4 - in->len) want = MAX_HEADER + 4 - in->len;
        if (!read_deadline(fd, deadline)) return -1;
        ssize_t r = read(fd, in->p + in->len, want);
        if (r <= 0) { if (r < 0 && errno == EINTR) continue; return -1; }
        in->len += (size_t)r;
    }
    /* request line */
    char *line_end = memmem(in->p, hdr_end, "\r\n", 2);
    char *sp1 = memchr(in->p, ' ', (size_t)(line_end - in->p));
    char *sp2 = sp1 ? memchr(sp1 + 1, ' ', (size_t)(line_end - sp1 - 1)) : NULL;
    if (!sp1 || !sp2) return 400;
    *sp1 = '\0';
    *sp2 = '\0';
    *method = in->p;
    *path = sp1 + 1;
    bool http11 = (size_t)(line_end - sp2 - 1) == 8 && !memcmp(sp2 + 1, "HTTP/1.1", 8);
    *keep_alive = http11;
    /* headers */
    size_t clen = 0;
    bool have_len = false;
    for (char *h = line_end + 2; h < in->p + hdr_end - 2;) {
        char *eol = memmem(h, (size_t)(in->p + hdr_end - h), "\r\n", 2);
        if (!eol) return 400;
        char *colon = memchr(h, ':', (size_t)(eol - h));
        if (colon) {
            size_t nl = (size_t)(colon - h);
            char *v = colon + 1;
            while (v < eol && (*v == ' ' || *v == '\t')) v++;
            size_t vl = (size_t)(eol - v);
            while (vl && (v[vl - 1] == ' ' || v[vl - 1] == '\t')) vl--;   /* trailing OWS */
            if (nl == 14 && !strncasecmp(h, "Content-Length", 14)) {
                if (have_len || vl == 0 || vl > 12) return 400;   /* duplicate or absurd length */
                size_t x = 0;
                for (size_t i = 0; i < vl; i++) {
                    if (v[i] < '0' || v[i] > '9') return 400;
                    x = x * 10 + (size_t)(v[i] - '0');
                }
                clen = x;
                have_len = true;
            } else if (nl == 17 && !strncasecmp(h, "Transfer-Encoding", 17)) {
                return 400;   /* chunked bodies are not supported; refuse rather than misframe */
            } else if (nl == 10 && !strncasecmp(h, "Connection", 10)) {
                if (vl == 5 && !strncasecmp(v, "close", 5)) *keep_alive = false;
                if (vl == 10 && !strncasecmp(v, "keep-alive", 10)) *keep_alive = true;
            }
        }
        h = eol + 2;
    }
    if (clen > S.max_body) return 413;
    size_t need = hdr_end + clen;
    if (need > in->cap) {
        char *p = realloc(in->p, need);
        if (!p) return 500;
        /* method/path point into the old buffer: rebase them */
        *method = p + (*method - in->p);
        *path = p + (*path - in->p);
        in->p = p;
        in->cap = need;
    }
    while (in->len < need) {
        if (!read_deadline(fd, deadline)) return -1;
        ssize_t r = read(fd, in->p + in->len, need - in->len);
        if (r <= 0) { if (r < 0 && errno == EINTR) continue; return -1; }
        in->len += (size_t)r;
    }
    *body = in->p + hdr_end;
    *body_len = clen;
    return 0;
}

static bool handle_systemone(int fd, const char *body, size_t len, bool keep_alive) {
    char err[256];
    jarena *a = jarena_new();
    if (!a) { respond_error(fd, 500, "out of memory", false); return false; }
    jval *req = json_parse(a, body, len, err, sizeof(err));
    job j;
    memset(&j, 0, sizeof(j));
    clef_encode_opts opts = CLEF_ENCODE_DEFAULTS;
    opts.strict = S.strict;
    opts.reject_truncation = !S.truncate;
    if (!req || !clef_encode_request(S.e->tok, req, opts, &j.rec, err, sizeof(err))) {
        bool sent = respond_error(fd, 400, err, keep_alive);
        jarena_free(a);
        return sent;
    }
    j.req = req;
    pthread_mutex_init(&j.mu, NULL);
    pthread_cond_init(&j.cv, NULL);
    pthread_mutex_lock(&S.mu);
    if (S.tail) S.tail->next = &j; else S.head = &j;
    S.tail = &j;
    pthread_cond_signal(&S.cv);
    pthread_mutex_unlock(&S.mu);

    pthread_mutex_lock(&j.mu);
    while (!j.done) pthread_cond_wait(&j.cv, &j.mu);
    pthread_mutex_unlock(&j.mu);

    bool sent;
    if (!j.ok) {
        sent = respond_error(fd, 500, j.err, keep_alive);
    } else {
        jbuf b = {0};
        if (!clef_build_response(req, &j.rec, j.probs, &b)) sent = respond_error(fd, 500, "out of memory", keep_alive);
        else sent = respond(fd, 200, b.p, b.len, keep_alive);
        jbuf_free(&b);
        for (int q = 0; q < j.rec.nq; q++) free(j.probs[q]);
        free(j.probs);
    }
    clef_record_free(&j.rec);
    pthread_mutex_destroy(&j.mu);
    pthread_cond_destroy(&j.cv);
    jarena_free(a);
    return sent;
}

/* Error close with unread input: closing a TCP socket while bytes are pending sends RST,
 * which can destroy the error response before the client reads it. Half-close, then drain
 * briefly so the response is delivered: at most 1 MiB and 1 s in total (a per-read timeout
 * alone let a trickling client stretch the drain indefinitely, review #4). */
static void lingering_close(int fd) {
    shutdown(fd, SHUT_WR);
    const double deadline = now_s() + (S.io_timeout < 1.0 ? S.io_timeout : 1.0);
    char sink[4096];
    size_t drained = 0;
    while (drained < (1u << 20) && read_deadline(fd, deadline)) {
        ssize_t r = read(fd, sink, sizeof(sink));
        if (r <= 0) break;
        drained += (size_t)r;
    }
}

/* One forward pass before accepting connections: faults in the weight pages, wires the
 * Metal buffers and allocates activations, so the first client does not pay for it
 * (measured: first request 208-223 ms vs 95 ms steady, with a warm OS page cache). */
static bool warmup(char *err, size_t errlen) {
    static const char req_text[] =
        "{\"model\":\"warmup\",\"state\":\"warm-up request\",\"questions\":"
        "{\"q\":{\"type\":\"choice\",\"criteria\":{\"a\":\"x\",\"b\":\"y\"}}}}";
    jarena *a = jarena_new();
    if (!a) { snprintf(err, errlen, "out of memory"); return false; }
    jval *req = json_parse(a, req_text, sizeof(req_text) - 1, err, errlen);
    clef_record rec;
    clef_encode_opts opts = CLEF_ENCODE_DEFAULTS;
    bool ok = req && clef_encode_request(S.e->tok, req, opts, &rec, err, errlen);
    if (ok) {
        float ***probs = NULL;
        ok = clef_run(S.e, &rec, 1, &probs, err, errlen);
        if (ok) clef_free_probs(&rec, 1, probs);
        clef_record_free(&rec);
    }
    jarena_free(a);
    return ok;
}

static void *connection(void *arg) {
    int fd = (int)(intptr_t)arg;
    rbuf in = {0};
    for (;;) {
        char *method, *path, *body;
        size_t body_len;
        bool keep_alive;
        int st = read_request(fd, &in, &method, &path, &body, &body_len, &keep_alive);
        if (st < 0) break;
        if (st > 0) {
            respond_error(fd, st, st == 413 ? "request body too large" : st == 431 ? "headers too large"
                                : st == 500 ? "out of memory" : "malformed request", false);
            lingering_close(fd);
            break;
        }
        size_t consumed = (size_t)(body - in.p) + body_len;
        bool sent;
        if (!strcmp(path, "/v1/systemone")) {
            if (strcmp(method, "POST")) sent = respond_error(fd, 405, "use POST", keep_alive);
            else sent = handle_systemone(fd, body, body_len, keep_alive);
        } else if (!strcmp(path, "/health")) {
            jbuf b = {0};
            jbuf_puts(&b, "{\"status\":\"ok\",\"model\":");
            json_put_string(&b, S.model_name, strlen(S.model_name));
            jbuf_puts(&b, "}");
            sent = !b.oom && respond(fd, 200, b.p, b.len, keep_alive);
            jbuf_free(&b);
        } else {
            sent = respond_error(fd, 404, "not found", keep_alive);
        }
        /* A partial response cannot be repaired by sending another response on
         * this stream. Drop pipelined requests after any failed write. */
        if (!sent || !keep_alive) break;
        memmove(in.p, in.p + consumed, in.len - consumed);   /* pipelined bytes, if any */
        in.len -= consumed;
    }
    free(in.p);
    close(fd);
    atomic_fetch_sub(&S.conns, 1);
    return NULL;
}

int main(int argc, char **argv) {
    const char *model = NULL, *host = "127.0.0.1";
    int port = 8080;
    bool do_warmup = true;
    S.batch = 8;
    S.batch_tokens = 4096;
    S.strict = true;
    S.max_body = 8u << 20;
    S.max_conn = 256;
    S.io_timeout = IO_TIMEOUT_S;
    /* test hook (tests/test_server_slow.py): a short deadline instead of 30 s */
    if (getenv("CLEF_DEBUG_IO_TIMEOUT")) S.io_timeout = atof(getenv("CLEF_DEBUG_IO_TIMEOUT"));
    if (!(S.io_timeout > 0)) S.io_timeout = IO_TIMEOUT_S;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "--host") && i + 1 < argc) host = argv[++i];
        else if (!strcmp(argv[i], "--port") && i + 1 < argc) port = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--batch") && i + 1 < argc) S.batch = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--max-body") && i + 1 < argc) S.max_body = (size_t)strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--no-strict")) S.strict = false;
        else if (!strcmp(argv[i], "--truncate")) S.truncate = true;
        else if (!strcmp(argv[i], "--no-warmup")) do_warmup = false;
        else if (!strcmp(argv[i], "--batch-tokens") && i + 1 < argc) S.batch_tokens = (size_t)strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--max-conn") && i + 1 < argc) S.max_conn = atoi(argv[++i]);
        else {
            fprintf(stderr, "usage: clef-server -m MODEL.gguf [--host 127.0.0.1] [--port 8080] [--batch 8] "
                            "[--batch-tokens 4096] [--no-strict] [--truncate] [--no-warmup] [--max-body BYTES] [--max-conn N]\n");
            return 2;
        }
    }
    if (!model || S.batch < 1 || port <= 0 || port > 65535 || S.max_conn < 1) { fprintf(stderr, "clef-server: bad arguments\n"); return 2; }
    signal(SIGPIPE, SIG_IGN);

    char err[512];
    S.e = clef_open(model, err, sizeof(err));
    if (!S.e) { fprintf(stderr, "clef-server: %s\n", err); return 1; }
    if (do_warmup) {
        double t0 = now_ms();
        if (!warmup(err, sizeof(err))) { fprintf(stderr, "clef-server: warm-up failed: %s\n", err); return 1; }
        fprintf(stderr, "clef-server: warm-up %.0f ms\n", now_ms() - t0);
    }
    gguf_str nm;
    S.model_name = gguf_get_str(&S.e->gguf, "general.name", &nm) ? strndup(nm.ptr, nm.len) : "clef";
    pthread_mutex_init(&S.mu, NULL);
    pthread_cond_init(&S.cv, NULL);
    pthread_t wt;
    if (pthread_create(&wt, NULL, worker, NULL)) { fprintf(stderr, "clef-server: cannot start worker\n"); return 1; }

    int ls = socket(AF_INET, SOCK_STREAM, 0);
    int one = 1;
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    struct sockaddr_in sa = { .sin_family = AF_INET, .sin_port = htons((uint16_t)port) };
    if (inet_pton(AF_INET, host, &sa.sin_addr) != 1) { fprintf(stderr, "clef-server: bad host %s\n", host); return 1; }
    if (bind(ls, (struct sockaddr *)&sa, sizeof(sa)) || listen(ls, 128)) { perror("clef-server: bind/listen"); return 1; }
    fprintf(stderr, "clef-server: %s on http://%s:%d (batch %d, %s)\n", model, host, port, S.batch,
            S.strict ? "strict: content cannot emit control tokens" : "no-strict: reference tokenization, injectable");
    fprintf(stderr, "clef-server: over-long state is %s\n", S.truncate ? "truncated silently (reference behaviour)" : "rejected");

    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
    pthread_attr_setstacksize(&attr, 8u << 20);   /* JSON recursion depth 512 + encoder */
    for (;;) {
        int fd = accept(ls, NULL, NULL);
        if (fd < 0) { if (errno == EINTR) continue; perror("clef-server: accept"); continue; }
        if (atomic_load(&S.conns) >= S.max_conn) {
            respond_error(fd, 503, "too many connections", false);
            close(fd);
            continue;
        }
        struct timeval tv = { (time_t)S.io_timeout, (suseconds_t)((S.io_timeout - (double)(time_t)S.io_timeout) * 1e6) };
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));   /* replaced per read by read_deadline */
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
        atomic_fetch_add(&S.conns, 1);
        pthread_t t;
        if (pthread_create(&t, &attr, connection, (void *)(intptr_t)fd)) {
            atomic_fetch_sub(&S.conns, 1);
            respond_error(fd, 503, "cannot start connection thread", false);
            close(fd);
        }
    }
}
