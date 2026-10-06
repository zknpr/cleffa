/* clef-server: Jev/SystemOne-compatible HTTP endpoint.
 *
 *   clef-server -m MODEL.gguf [--host 127.0.0.1] [--port 8080] [--batch 8] [--batch-tokens 4096]
 *               [--max-body 8388608] [--max-conn 256] [--no-keep-warm] [--prefix-cache-mb 0]
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
 * (16 KiB), body (--max-body), connections (--max-conn), and a deadline per request. Slots are
 * not protected from clients that keep sending valid requests (see read_deadline).
 * Strict mode (default): request content is tokenized without special-token recognition, so a
 * literal "<|im_end|>" cannot close the template's user turn. --no-strict reproduces the
 * reference exactly, including that injection (README, Security notes).
 * Over-long state is rejected by default (HTTP 400 with token counts); --truncate keeps the
 * reference behaviour of silently dropping the end of the state.
 * Keep-warm (default; --no-keep-warm turns it off): while idle, the worker submits one-thread
 * GPU passes over the weights and activation buffers every KEEP_WARM_MS. Without them, a request
 * that arrives more than about a second after the previous one starts 205-290 ms late on the 27B
 * (clef_gpu_keepalive). They only read model and activation buffers; their writes target a
 * separate sink. See docs/performance-history.md (keep-warm) for measurements and limitations.
 * Prefix cache (--prefix-cache-mb N, default off): a request with the header
 * X-Clef-Prefix-Cache: KEY reuses the backbone state of the tokens it shares with the previous
 * request under that key, and its answer is bitwise the uncached one. The key is the isolation
 * boundary, because a hit is visible in the latency (see the cache table below).
 */

#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
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
#define KEEP_WARM_MS 500   /* keep-warm idle period; the idle start delay appears between 1 and 1.5 s on the 27B */

#define CACHE_KEY_MAX 64
#define CACHE_ENTRIES 32

typedef struct job {
    clef_record rec;
    const jval *req;
    char cache_key[CACHE_KEY_MAX + 1];   /* X-Clef-Prefix-Cache; "" = none */
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
    int keep_warm_ms;        /* idle period between keep-warm GPU passes (KEEP_WARM_MS); 0 = off (--no-keep-warm) */
    size_t cache_bytes;      /* prefix cache budget (--prefix-cache-mb); 0 = off, the header is ignored */
    bool template_cache;     /* --template-cache: single unkeyed requests reuse the template tokens' state */
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

/* Prefix cache (clef_run_prefix). One entry per client-chosen key, used only by the worker thread.
 * A request names its entry with the X-Clef-Prefix-Cache header and runs alone through it. The
 * key is the isolation boundary: whether a request hits an entry shows in its latency, so two
 * parties that must not learn about each other's states must not share a key. The deployment in
 * front of this server assigns keys (per tenant, per conversation); the server never shares an
 * entry across keys and never uses one for a request without the header. Entries are dropped
 * least recently used first when their GPU memory exceeds the budget. */
static struct { char key[CACHE_KEY_MAX + 1]; clef_prefix *p; double used; } cache[CACHE_ENTRIES];

/* A size flag is a whole decimal number that fits after scaling: strtoull alone accepts "-1"
 * (wrapping to ULLONG_MAX), trailing junk, and values whose MiB shift overflows (review #22). */
static bool parse_size(const char *s, size_t max, size_t *out) {
    if (!*s || strspn(s, "0123456789") != strlen(s)) return false;
    errno = 0;
    unsigned long long v = strtoull(s, NULL, 10);
    if (errno || v > max) return false;
    *out = (size_t)v;
    return true;
}

static size_t cache_total(void) {
    size_t n = 0;
    for (int i = 0; i < CACHE_ENTRIES; i++) if (cache[i].p) n += clef_prefix_bytes(cache[i].p);
    return n;
}

static void cache_drop(int i, const char *why) {
    fprintf(stderr, "clef-server: prefix cache: dropped an entry of %.0f MB (%s)\n", clef_prefix_bytes(cache[i].p) / 1e6, why);
    clef_prefix_free(cache[i].p);
    cache[i].p = NULL;
    cache[i].key[0] = '\0';
}

/* The populated entry for `key`, or -1. */
static int cache_find(const char *key) {
    for (int i = 0; i < CACHE_ENTRIES; i++)
        if (cache[i].p && !strcmp(cache[i].key, key)) return i;
    return -1;
}

/* Store an entry that a pass has just populated, evicting the least recently used entry when
 * the table is full. Only populated entries go in: a keyed request too short to cache, or an
 * unsupported one, bypasses its entry and must not displace another key's state (review #15). */
static int cache_insert(const char *key, clef_prefix *p) {
    int slot = -1, lru = -1;
    for (int i = 0; i < CACHE_ENTRIES; i++) {
        if (!cache[i].p) { if (slot < 0) slot = i; }
        else if (lru < 0 || cache[i].used < cache[lru].used) lru = i;
    }
    if (slot < 0) { cache_drop(lru, "table full"); slot = lru; }
    cache[slot].p = p;
    snprintf(cache[slot].key, sizeof(cache[slot].key), "%s", key);
    return slot;
}

/* Template reuse (clef_run_template): one entry for the 32 tokens every request starts with. It
 * reuses only the fixed public prompt, so it needs no key. Its scratch suffix rows are
 * request-dependent and overwritten before use; this entry is extra to the keyed budget.
 * A failed pass falls back to the plain path like a keyed one. */
static clef_prefix *template_entry;

static bool run_template(job *j, float ****probs, char *err, size_t errlen) {
    if (!template_entry && !(template_entry = clef_prefix_new())) {
        fprintf(stderr, "clef-server: template cache: cannot allocate an entry, serving uncached\n");
        return clef_run(S.e, &j->rec, 1, probs, err, errlen);
    }
    if (clef_run_template(S.e, template_entry, &j->rec, probs, false, NULL, err, errlen)) return true;
    fprintf(stderr, "clef-server: template cache: %s; serving uncached\n", err);
    clef_prefix_free(template_entry);
    template_entry = NULL;
    return clef_run(S.e, &j->rec, 1, probs, err, errlen);
}

/* A failed cached pass is recomputed uncached and its entry is dropped. */
static bool run_keyed(job *j, float ****probs, char *err, size_t errlen) {
    int i = cache_find(j->cache_key);
    clef_prefix *p = i >= 0 ? cache[i].p : clef_prefix_new();
    if (!p) {
        fprintf(stderr, "clef-server: prefix cache: cannot allocate an entry, serving uncached\n");
        return clef_run(S.e, &j->rec, 1, probs, err, errlen);
    }
    int reused = 0;
    bool ok = clef_run_prefix(S.e, p, &j->rec, probs, false, &reused, err, errlen);
    if (!ok) {
        fprintf(stderr, "clef-server: prefix cache: %s; serving uncached\n", err);
        if (i >= 0) cache_drop(i, "failed pass"); else clef_prefix_free(p);
        return clef_run(S.e, &j->rec, 1, probs, err, errlen);
    }
    fprintf(stderr, "clef-server: prefix cache: reused %d of %zu tokens\n", reused, j->rec.ids.len);
    if (i < 0) {
        /* Nothing cached (bypassed) or more than the whole budget: the entry is never retained, so
           it must not enter the table and displace another key's entry (reviews #15, #19). */
        if (clef_prefix_bytes(p) > S.cache_bytes)
            fprintf(stderr, "clef-server: prefix cache: dropped an entry of %.0f MB (larger than budget)\n", clef_prefix_bytes(p) / 1e6);
        if (clef_prefix_bytes(p) == 0 || clef_prefix_bytes(p) > S.cache_bytes) { clef_prefix_free(p); return true; }
        i = cache_insert(j->cache_key, p);
    }
    cache[i].used = now_ms();
    /* An existing entry that grew past the whole budget can never be retained either: drop it
       alone, before the LRU pass below would evict every other key's entry on its behalf (review #6). */
    if (clef_prefix_bytes(cache[i].p) > S.cache_bytes) {
        cache_drop(i, "larger than budget");
        return true;
    }
    /* over budget: least recently used first; the newest entry goes last */
    while (cache_total() > S.cache_bytes) {
        int lru = -1;
        for (int k = 0; k < CACHE_ENTRIES; k++)
            if (cache[k].p && k != i && (lru < 0 || cache[k].used < cache[lru].used)) lru = k;
        cache_drop(lru < 0 ? i : lru, "over budget");
        if (lru < 0) break;
    }
    return true;
}

static bool queue_pending(void) {
    pthread_mutex_lock(&S.mu);
    const bool pending = S.head != NULL;
    pthread_mutex_unlock(&S.mu);
    return pending;
}

/* One idle keep-warm round: the engine's buffers, then each cache entry's, then the template
 * entry's. Cache buffers need their own idle touches; only the worker accesses this table.
 * The queue is rechecked between entries, so a request that arrives mid-round waits for at
 * most one entry's pass rather than the whole table's (review #24); the round resumes at the
 * next idle period. */
static bool keep_warm_round(char *err, size_t errlen) {
    if (!clef_keep_warm(S.e, err, errlen)) return false;
    for (int i = 0; i < CACHE_ENTRIES; i++) {
        if (queue_pending()) return true;
        if (cache[i].p && !clef_prefix_keep_warm(S.e, cache[i].p, err, errlen)) return false;
    }
    if (queue_pending()) return true;
    return !template_entry || clef_prefix_keep_warm(S.e, template_entry, err, errlen);
}

static void *worker(void *arg) {
    (void)arg;
    job **batch = calloc((size_t)S.batch, sizeof(*batch));
    clef_record *recs = calloc((size_t)S.batch, sizeof(*recs));
    if (!batch || !recs) { fprintf(stderr, "clef-server: out of memory\n"); exit(1); }
    bool warned_keep_warm = false;
    for (;;) {
        pthread_mutex_lock(&S.mu);
        while (!S.head) {
            if (S.keep_warm_ms <= 0) { pthread_cond_wait(&S.cv, &S.mu); continue; }
            /* Keep-warm: while idle, submit the keep-warm passes every keep_warm_ms so the next
             * request does not start late (clef_gpu_keepalive). They run on this thread, so they
             * never overlap a forward; a request that arrives meanwhile is seen by the re-check
             * of S.head and waits at most for those one-thread passes. */
            struct timespec rel = { S.keep_warm_ms / 1000, (long)(S.keep_warm_ms % 1000) * 1000000L };
            if (pthread_cond_timedwait_relative_np(&S.cv, &S.mu, &rel) != ETIMEDOUT || S.head) continue;
            pthread_mutex_unlock(&S.mu);
            char kerr[256] = "";
            bool warm = keep_warm_round(kerr, sizeof(kerr));
            if (!warm && !warned_keep_warm) {
                fprintf(stderr, "clef-server: keep-warm pass failed: %s\n", kerr);   /* once: latency only */
                warned_keep_warm = true;
            }
            pthread_mutex_lock(&S.mu);
        }
        /* FIFO; take jobs from the head while the batch stays within the token budget. A
         * request larger than the budget runs alone, so small requests are never packed into
         * a long request's forward pass (review #2, I1). The head is always admitted, so
         * nothing starves; a job queued behind a long one still waits for it (one GPU, no
         * preemption). */
        int n = 0;
        size_t toks = 0;
        /* a request with a prefix cache key runs alone, through its entry */
        const bool keyed = S.cache_bytes && S.head->cache_key[0];
        while (S.head && n < S.batch &&
               (n == 0 || (!keyed && !(S.cache_bytes && S.head->cache_key[0]) && toks + S.head->rec.ids.len <= S.batch_tokens))) {
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
        bool ok = keyed ? run_keyed(batch[0], &probs, err, sizeof(err))
                : n == 1 && S.template_cache ? run_template(batch[0], &probs, err, sizeof(err))
                : clef_run(S.e, recs, n, &probs, err, sizeof(err));
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
 * the socket timeout is set to the time left, and none left means give up.
 * This bounds a request, not a slot. A client that sends a complete request on a keep-alive
 * connection more often than the deadline (a 24-byte GET /health will do) keeps its slot, and
 * --max-conn such connections get everyone else 503 (review #5). Nothing here can tell that
 * client from a legitimate one; per-client limits belong in a proxy in front (README, Security). */
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

static int read_request(int fd, rbuf *in, char **method, char **path, char **body, size_t *body_len, bool *keep_alive,
                        char *cache_key) {
    const double deadline = now_s() + S.io_timeout;
    cache_key[0] = '\0';
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
            } else if (nl == 19 && !strncasecmp(h, "X-Clef-Prefix-Cache", 19)) {
                /* an opaque token; anything else is refused, not silently served uncached */
                if (vl == 0 || vl > CACHE_KEY_MAX || cache_key[0]) return 400;
                for (size_t i = 0; i < vl; i++) {
                    const char ch = v[i];
                    if (!((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') || ch == '-' || ch == '_' || ch == '.'))
                        return 400;
                }
                memcpy(cache_key, v, vl);
                cache_key[vl] = '\0';
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

static bool handle_systemone(int fd, const char *body, size_t len, bool keep_alive, const char *cache_key) {
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
    snprintf(j.cache_key, sizeof(j.cache_key), "%s", cache_key);
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
 * alone let a trickling client stretch the drain indefinitely, review #4).
 * The wait is poll(), not read_deadline(): when the client has half-closed too, the socket is
 * shut both ways after our shutdown, macOS then fails setsockopt with EINVAL, and the drain was
 * skipped with the body unread (review #5, tests/test_lingering_close.sh). After POLLIN a read
 * returns at once, with data or EOF. */
static void lingering_close(int fd) {
    shutdown(fd, SHUT_WR);
    const double deadline = now_s() + (S.io_timeout < 1.0 ? S.io_timeout : 1.0);
    char sink[4096];
    size_t drained = 0;
    while (drained < (1u << 20)) {
        const double left = deadline - now_s();
        if (left <= 0) break;
        struct pollfd pfd = { .fd = fd, .events = POLLIN };
        int pr = poll(&pfd, 1, (int)(left * 1e3) + 1);
        if (pr < 0 && errno == EINTR) continue;
        if (pr <= 0) break;
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
        char cache_key[CACHE_KEY_MAX + 1];
        int st = read_request(fd, &in, &method, &path, &body, &body_len, &keep_alive, cache_key);
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
            else sent = handle_systemone(fd, body, body_len, keep_alive, cache_key);
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
    S.keep_warm_ms = KEEP_WARM_MS;
    /* test hook (tests/test_server_slow.py): a short deadline instead of 30 s */
    if (getenv("CLEF_DEBUG_IO_TIMEOUT")) S.io_timeout = atof(getenv("CLEF_DEBUG_IO_TIMEOUT"));
    if (!(S.io_timeout > 0)) S.io_timeout = IO_TIMEOUT_S;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "--host") && i + 1 < argc) host = argv[++i];
        else if (!strcmp(argv[i], "--port") && i + 1 < argc) port = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--batch") && i + 1 < argc) S.batch = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--max-body") && i + 1 < argc) {
            if (!parse_size(argv[++i], SIZE_MAX, &S.max_body)) { fprintf(stderr, "clef-server: --max-body must be a whole number of bytes\n"); return 2; }
        }
        else if (!strcmp(argv[i], "--no-strict")) S.strict = false;
        else if (!strcmp(argv[i], "--truncate")) S.truncate = true;
        else if (!strcmp(argv[i], "--no-warmup")) do_warmup = false;
        else if (!strcmp(argv[i], "--keep-warm")) S.keep_warm_ms = KEEP_WARM_MS;   /* the default, accepted explicitly */
        else if (!strcmp(argv[i], "--no-keep-warm")) S.keep_warm_ms = 0;
        else if (!strcmp(argv[i], "--batch-tokens") && i + 1 < argc) {
            if (!parse_size(argv[++i], SIZE_MAX, &S.batch_tokens)) { fprintf(stderr, "clef-server: --batch-tokens must be a whole number of tokens\n"); return 2; }
        }
        else if (!strcmp(argv[i], "--prefix-cache-mb") && i + 1 < argc) {
            if (!parse_size(argv[++i], SIZE_MAX >> 20, &S.cache_bytes)) { fprintf(stderr, "clef-server: --prefix-cache-mb must be a whole number of MiB below %zu\n", (size_t)(SIZE_MAX >> 20)); return 2; }
            S.cache_bytes <<= 20;
        }
        else if (!strcmp(argv[i], "--template-cache")) S.template_cache = true;
        else if (!strcmp(argv[i], "--max-conn") && i + 1 < argc) S.max_conn = atoi(argv[++i]);
        else {
            fprintf(stderr, "usage: clef-server -m MODEL.gguf [--host 127.0.0.1] [--port 8080] [--batch 8] "
                            "[--batch-tokens 4096] [--no-strict] [--truncate] [--no-warmup] [--no-keep-warm] [--prefix-cache-mb N] [--template-cache] [--max-body BYTES] [--max-conn N]\n");
            return 2;
        }
    }
    /* experiment hook: another keep-warm period in ms (0 turns it off) */
    if (getenv("CLEF_DEBUG_KEEPWARM_MS")) S.keep_warm_ms = atoi(getenv("CLEF_DEBUG_KEEPWARM_MS"));
    if (S.keep_warm_ms < 0) S.keep_warm_ms = 0;
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
    if (S.keep_warm_ms) fprintf(stderr, "clef-server: keep-warm pass every %d ms while idle\n", S.keep_warm_ms);
    else fprintf(stderr, "clef-server: keep-warm off\n");
    if (S.cache_bytes) fprintf(stderr, "clef-server: prefix cache of %zu MB for requests with X-Clef-Prefix-Cache\n", S.cache_bytes >> 20);

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
