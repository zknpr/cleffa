/* clef-server: Jev/SystemOne-compatible HTTP endpoint.
 *
 *   clef-server -m MODEL.gguf [--host 127.0.0.1] [--port 8080] [--batch 8] [--batch-tokens 4096]
 *               [--max-body 8388608] [--max-conn 256] [--no-keep-warm] [--prefix-cache-mb 0]
 *               [--max-images 4] [--max-image-tokens 1024]
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
#include <limits.h>
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
    /* Images per request and tokens per image after the model's own resizing (--max-images,
     * --max-image-tokens; 0 = no limit). Over the limit a request is rejected with the count and
     * the media_kwargs.max_pixels that would fit, never downscaled silently: the reference keeps up
     * to 16,384 tokens per image, and image tokens cost the same prefill as text. */
    int max_images;
    int max_videos, max_video_frames;
    long max_video_tokens;
    long max_image_tokens;
    /* Image memory is bounded at two more points (review #3). Decoding allocates several times a
     * source image's RGB size before any token limit can apply, and a compressible 8192x8192 PNG
     * is about 1.4 MB as base64: max_image_pixels refuses large sources at the header, before the
     * decoders allocate. And each connection thread encodes its own request, so per-image bounds
     * still multiply by --max-conn; an encoded 1,024-token image holds 24 MiB of f32 patches
     * from a request of about 1 KB. max_image_requests bounds the requests with images that are
     * decoding or holding patches at once; the rest wait, holding only their body. */
    long max_image_pixels;   /* per source image; default 16,777,216 (the processor's max_pixels) */
    int max_image_requests;  /* image requests decoded or queued at once; 0 = unlimited */
    int image_inflight;
    pthread_mutex_t image_mu;
    pthread_cond_t image_cv;
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
    /* The budget bounds what the table retains, and the pass allocates the entry: project its
       size first and serve a request whose entry would exceed the budget uncached, so no entry
       larger than the budget is ever allocated. An existing entry that would grow past it is
       dropped the same way (review #27). */
    const size_t projected = clef_prefix_estimate(S.e, p, &j->rec);
    if (projected > S.cache_bytes) {
        fprintf(stderr, "clef-server: prefix cache: an entry for this request would hold %.0f MB and would exceed the budget; serving uncached\n", projected / 1e6);
        if (i >= 0) cache_drop(i, "would exceed the budget"); else clef_prefix_free(p);
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
    fprintf(stderr, "clef-server: prefix cache: projected %.0f MB, holds %.0f MB\n", projected / 1e6, clef_prefix_bytes(p) / 1e6);
    if (i < 0) {
        /* An entry with no usable state (a bypassed request, or an FP16 overflow answered by the
           BF16 rerun, which leaves allocated buffers and no checkpoint) or one larger than the
           whole budget is never retained: it must not enter the table and displace another
           key's entry (reviews #15, #19, #33). */
        if (!clef_prefix_usable(p)) { clef_prefix_free(p); return true; }
        if (clef_prefix_bytes(p) > S.cache_bytes) {
            fprintf(stderr, "clef-server: prefix cache: dropped an entry of %.0f MB (larger than budget)\n", clef_prefix_bytes(p) / 1e6);
            clef_prefix_free(p);
            return true;
        }
        i = cache_insert(j->cache_key, p);
    } else if (!clef_prefix_usable(p)) {
        cache_drop(i, "no usable state after the pass");   /* an overflowing pass: the buffers would only cost budget */
        return true;
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
/* An idle round resumes where an arriving request interrupted the previous one: starting at
 * entry zero every time would let intermittent traffic starve the later entries and the
 * template entry of their warming pass (review #93). Position CACHE_ENTRIES is the template. */
static int keep_warm_cursor;

static bool keep_warm_round(char *err, size_t errlen) {
    if (!clef_keep_warm(S.e, err, errlen)) return false;
    for (int k = 0; k <= CACHE_ENTRIES; k++) {
        const int i = (keep_warm_cursor + k) % (CACHE_ENTRIES + 1);
        if (queue_pending()) { keep_warm_cursor = i; return true; }
        const clef_prefix *p = i == CACHE_ENTRIES ? template_entry : cache[i].p;
        if (p && !clef_prefix_keep_warm(S.e, p, err, errlen)) return false;
    }
    return true;
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

static void image_slot_acquire(void) {
    pthread_mutex_lock(&S.image_mu);
    while (S.image_inflight >= S.max_image_requests) pthread_cond_wait(&S.image_cv, &S.image_mu);
    S.image_inflight++;
    pthread_mutex_unlock(&S.image_mu);
}

static void image_slot_release(void) {
    pthread_mutex_lock(&S.image_mu);
    S.image_inflight--;
    pthread_cond_signal(&S.image_cv);
    pthread_mutex_unlock(&S.image_mu);
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
    opts.vision = S.e->vision;
    opts.vision.max_images = S.max_images;
    opts.vision.max_image_tokens = S.max_image_tokens;
    opts.vision.max_image_pixels = S.max_image_pixels;
    opts.vision.max_videos = S.max_videos;
    opts.vision.max_video_frames = S.max_video_frames;
    opts.vision.max_video_tokens = S.max_video_tokens;
    /* Any request naming images or videos takes a slot before decoding and keeps it until
     * its patches are freed; a malformed list is refused by the encoder before any decoding. */
    const jval *ims = req ? json_get(req, "images") : NULL;
    const jval *vids = req ? json_get(req, "videos") : NULL;
    const bool slot = S.max_image_requests > 0 && ((ims && ims->type == J_ARRAY && ims->n > 0) ||
                                                 (vids && vids->type == J_ARRAY && vids->n > 0));
    if (slot) image_slot_acquire();
    if (!req || !clef_encode_request(S.e->tok, req, opts, &j.rec, err, sizeof(err))) {
        if (slot) image_slot_release();
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

    /* Build the response, then free the record (its image patches) and the slot before writing:
     * a client that reads slowly must not hold image memory or keep other image requests out. */
    jbuf b = {0};
    bool built = false;
    if (j.ok) {
        built = clef_build_response(req, &j.rec, j.probs, &b);
        for (int q = 0; q < j.rec.nq; q++) free(j.probs[q]);
        free(j.probs);
    }
    clef_record_free(&j.rec);
    if (slot) image_slot_release();
    bool sent;
    if (!j.ok) sent = respond_error(fd, 500, j.err, keep_alive);
    else if (!built) sent = respond_error(fd, 500, "out of memory", keep_alive);
    else sent = respond(fd, 200, b.p, b.len, keep_alive);
    jbuf_free(&b);
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
 * (measured: first request 208-223 ms vs 95 ms steady, with a warm OS page cache).
 * When the model has a vision tower, a second pass carries one image sized to the per-image
 * limit (a 1x1 PNG the processor upscales through media_kwargs), so the vision scratch is
 * allocated at the size the first real image needs and the tower's weight pages are faulted
 * in too; a text-only warm-up left both to the first image request (review #3). */
static bool run_warm_request(const char *req_text, size_t len, char *err, size_t errlen) {
    jarena *a = jarena_new();
    if (!a) { snprintf(err, errlen, "out of memory"); return false; }
    jval *req = json_parse(a, req_text, len, err, errlen);
    clef_record rec;
    clef_encode_opts opts = CLEF_ENCODE_DEFAULTS;
    opts.vision = S.e->vision;
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

static bool warmup(char *err, size_t errlen) {
    static const char req_text[] =
        "{\"model\":\"warmup\",\"state\":\"warm-up request\",\"questions\":"
        "{\"q\":{\"type\":\"choice\",\"criteria\":{\"a\":\"x\",\"b\":\"y\"}}}}";
    if (!run_warm_request(req_text, sizeof(req_text) - 1, err, errlen)) return false;
    if (!S.e->vision.image_token_id) return true;
    /* side = floor(sqrt(limit)) merge windows, so the upscaled square stays within the limit;
     * 1,024 tokens when the limit is off, 4,096 at most (about 3 s on Flash at startup). */
    long limit = S.max_image_tokens > 0 ? S.max_image_tokens : 1024;
    if (limit > 4096) limit = 4096;
    long side = 1;
    while ((side + 1) * (side + 1) <= limit) side++;
    const long window = (long)S.e->vision.image.patch * S.e->vision.image.merge;
    const long px = side * window * side * window;
    char req_image[512];
    const int n = snprintf(req_image, sizeof(req_image),
        "{\"model\":\"warmup\",\"state\":\"warm-up request\",\"images\":[\"data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR42mNoaGgAAAMEAYF1LgG8AAAAAElFTkSuQmCC\"],"
        "\"media_kwargs\":{\"min_pixels\":%ld,\"max_pixels\":%ld},"
        "\"questions\":{\"q\":{\"type\":\"choice\",\"criteria\":{\"a\":\"x\",\"b\":\"y\"}}}}", px, px);
    if (n < 0 || (size_t)n >= sizeof(req_image)) { snprintf(err, errlen, "warm-up request too long"); return false; }
    const double t0 = now_ms();
    if (!run_warm_request(req_image, (size_t)n, err, errlen)) return false;
    fprintf(stderr, "clef-server: warm-up image pass: %ld image tokens in %.0f ms\n", side * side, now_ms() - t0);
    return true;
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
    S.max_images = 4;
    S.max_videos = 1;
    S.max_video_frames = 32;
    S.max_video_tokens = 1024;
    S.max_image_tokens = 1024;
    S.max_image_pixels = 16777216;
    S.max_image_requests = 8;
    pthread_mutex_init(&S.image_mu, NULL);
    pthread_cond_init(&S.image_cv, NULL);
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
        else if ((!strcmp(argv[i], "--max-videos") || !strcmp(argv[i], "--max-video-frames") || !strcmp(argv[i], "--max-video-tokens")) && i + 1 < argc) {
            const char *flag = argv[i]; size_t v;
            if (!parse_size(argv[++i], INT_MAX, &v)) { fprintf(stderr, "clef-server: %s must be a whole number (0 = processor/context limit)\n", flag); return 2; }
            if (!strcmp(flag, "--max-videos")) S.max_videos = (int)v;
            else if (!strcmp(flag, "--max-video-frames")) S.max_video_frames = (int)v;
            else S.max_video_tokens = (long)v;
        }
        else if (!strcmp(argv[i], "--max-images") && i + 1 < argc) {
            /* whole numbers only: atoi turned a typo into 0, which means unlimited (review #3) */
            size_t v;
            if (!parse_size(argv[++i], INT_MAX, &v)) { fprintf(stderr, "clef-server: --max-images must be a whole number (0 = unlimited)\n"); return 2; }
            S.max_images = (int)v;
        }
        else if (!strcmp(argv[i], "--max-image-tokens") && i + 1 < argc) {
            size_t v;
            if (!parse_size(argv[++i], LONG_MAX, &v)) { fprintf(stderr, "clef-server: --max-image-tokens must be a whole number (0 = unlimited)\n"); return 2; }
            S.max_image_tokens = (long)v;
        }
        else if (!strcmp(argv[i], "--max-image-pixels") && i + 1 < argc) {
            size_t v;
            if (!parse_size(argv[++i], LONG_MAX, &v)) { fprintf(stderr, "clef-server: --max-image-pixels must be a whole number (0 = the decoders' 64 Mpx cap)\n"); return 2; }
            S.max_image_pixels = (long)v;
        }
        else if (!strcmp(argv[i], "--max-image-requests") && i + 1 < argc) {
            size_t v;
            if (!parse_size(argv[++i], INT_MAX, &v)) { fprintf(stderr, "clef-server: --max-image-requests must be a whole number (0 = unlimited)\n"); return 2; }
            S.max_image_requests = (int)v;
        }
        else {
            fprintf(stderr, "usage: clef-server -m MODEL.gguf [--host 127.0.0.1] [--port 8080] [--batch 8] "
                            "[--batch-tokens 4096] [--no-strict] [--truncate] [--no-warmup] [--no-keep-warm] [--prefix-cache-mb N] [--template-cache] [--max-body BYTES] [--max-conn N] "
                            "[--max-images N] [--max-image-tokens N] [--max-image-pixels N] [--max-image-requests N] "
                            "[--max-videos N] [--max-video-frames N] [--max-video-tokens N]\n");
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
    if (S.e->vision.image_token_id)
        fprintf(stderr, "clef-server: images: at most %d per request, %ld tokens each after resizing, %ld source pixels each; "
                        "%d image/video requests decoded or queued at once (0 = unlimited)\n",
                S.max_images, S.max_image_tokens, S.max_image_pixels, S.max_image_requests);
    else fprintf(stderr, "clef-server: this model file has no vision tower; requests with images are rejected\n");
    if (S.e->vision.image_token_id)
        fprintf(stderr, "clef-server: videos: at most %d per request, %d sampled frames and %ld tokens each (0 = processor/context limit)\n",
                S.max_videos, S.max_video_frames, S.max_video_tokens);
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
