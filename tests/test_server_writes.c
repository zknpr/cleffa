/* Inject transport failures locally; a partial response must end the connection
 * before a second, otherwise valid, pipelined request can be processed. */
#include <assert.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>

static int test_fd, write_count, fail_at, writes_after_failure, zero_write, responses;
static ssize_t test_write(int fd, const void *data, size_t size) {
    if (fd != test_fd) return write(fd, data, size);
    write_count++;
    if (size >= 8 && !memcmp(data, "HTTP/1.1", 8)) responses++;
    if (fail_at && write_count == fail_at) {
        errno = EAGAIN;
        return zero_write ? 0 : -1;
    }
    if (fail_at && write_count > fail_at) writes_after_failure++;
    if (fail_at == 3 && write_count == 1) return (ssize_t)size; /* complete header */
    if (fail_at == 3 && write_count == 2) return 1;            /* partial body */
    return size > 7 ? 7 : (ssize_t)size;
}

/* Keep-warm mocks: the engine pass is free; each entry pass is counted, and the first one
 * enqueues a job, as a request arriving mid-round would. */
#define clef_keep_warm test_keep_warm
#define clef_prefix_keep_warm test_prefix_keep_warm
/* Keyed-request mocks: the pass succeeds (an FP16 overflow is answered by the BF16 rerun) but
 * leaves the entry with allocated buffers and no usable state. */
#define clef_run_prefix test_run_prefix
#define clef_run test_run
#define clef_prefix_new test_prefix_new
#define clef_prefix_free test_prefix_free
#define clef_prefix_bytes test_prefix_bytes
#define clef_prefix_usable test_prefix_usable
#define clef_prefix_estimate test_prefix_estimate
#define main clef_server_main
#define write test_write
#include "../clef_server.c"
#undef write
#undef main
static int warm_calls;
static char fresh[4];                /* the entry a new key gets */
static int uncached_runs, freed_fresh;
clef_prefix *test_prefix_new(void) { return (clef_prefix *)fresh; }
void test_prefix_free(clef_prefix *p) { if (p == (clef_prefix *)fresh) freed_fresh++; }
size_t test_prefix_bytes(const clef_prefix *p) { (void)p; return 100; }   /* buffers were allocated */
bool test_prefix_usable(const clef_prefix *p) { return p != (clef_prefix *)fresh; }   /* the overflowing pass stored nothing */
size_t test_prefix_estimate(const clef_engine *e, const clef_prefix *p, const clef_record *rec) { (void)e; (void)p; (void)rec; return 100; }
bool test_run_prefix(clef_engine *e, clef_prefix *p, const clef_record *rec, float ****out, bool raw, int *reused, char *err, size_t errlen) {
    (void)e; (void)p; (void)rec; (void)raw; (void)err; (void)errlen; if (reused) *reused = 0; *out = NULL; return true;
}
bool test_run(clef_engine *e, const clef_record *recs, int n, float ****probs, char *err, size_t errlen) {
    (void)e; (void)recs; (void)n; (void)err; (void)errlen; uncached_runs++; *probs = NULL; return true;
}
static job arriving;
static const clef_prefix *warmed[8];   /* the entries of a round, in order */
static int warmed_n;
bool test_keep_warm(clef_engine *e, char *err, size_t errlen) { (void)e; (void)err; (void)errlen; return true; }
bool test_prefix_keep_warm(clef_engine *e, const clef_prefix *p, char *err, size_t errlen) {
    (void)e; (void)err; (void)errlen;
    if (warmed_n < 8) warmed[warmed_n++] = p;
    if (++warm_calls == 1) { pthread_mutex_lock(&S.mu); S.head = S.tail = &arriving; pthread_mutex_unlock(&S.mu); }
    return true;
}

int main(void) {
    const char *requests[] = {
        "GET /health HTTP/1.1\r\n\r\n",
        "GET /missing HTTP/1.1\r\n\r\n",
        "GET /v1/systemone HTTP/1.1\r\n\r\n",
        "POST /v1/systemone HTTP/1.1\r\nContent-Length: 2\r\n\r\n{}"
    };
    clef_engine engine = {0};
    S.e = &engine;
    S.model_name = "test";
    S.io_timeout = 1;
    S.max_body = 1024;
    int cases = 0;
    for (size_t r = 0; r < sizeof(requests) / sizeof(requests[0]); r++) {
        for (int zero = 0; zero < 2; zero++) {
            for (int failure = 0; failure < 3; failure++) {
                int fds[2];
                assert(socketpair(AF_UNIX, SOCK_STREAM, 0, fds) == 0);
                test_fd = fds[1];
                write_count = writes_after_failure = responses = 0;
                fail_at = failure == 0 ? 0 : failure == 1 ? 2 : 3;
                zero_write = zero;
                assert(write(fds[0], requests[r], strlen(requests[r])) == (ssize_t)strlen(requests[r]));
                assert(write(fds[0], requests[r], strlen(requests[r])) == (ssize_t)strlen(requests[r]));
                shutdown(fds[0], SHUT_WR);
                atomic_store(&S.conns, 1);
                connection((void *)(intptr_t)fds[1]);
                assert(atomic_load(&S.conns) == 0);
                assert(writes_after_failure == 0);
                assert(fail_at ? write_count == fail_at : responses == 2);
                assert(responses == (fail_at ? 1 : 2));
                close(fds[0]);
                cases++;
            }
        }
    }
    printf("server writes: %d partial-write, timeout, zero-write and keep-alive cases passed\n", cases);

    /* A request that arrives during the idle round must wait for at most one more entry pass,
       not for every populated entry and the template entry (review #24). */
    static char fake[4];
    for (int i = 0; i < 3; i++) cache[i].p = (clef_prefix *)(fake + i);
    template_entry = (clef_prefix *)(fake + 3);
    char kerr[64] = "";
    warm_calls = 0;
    assert(keep_warm_round(kerr, sizeof(kerr)));
    assert(S.head == &arriving);
    if (warm_calls != 1) { fprintf(stderr, "keep-warm round touched %d entries after a request arrived (expected 1)\n", warm_calls); return 1; }
    /* The next idle round resumes at the entry the arrival skipped, not at entry zero, so
       intermittent traffic cannot starve the later entries and the template (review #93).
       The round still covers every position: entries 1 and 2, the template, then entry 0. */
    S.head = S.tail = NULL;
    warmed_n = 0;
    assert(keep_warm_round(kerr, sizeof(kerr)));
    const clef_prefix *expected[4] = { (clef_prefix *)(fake + 1), (clef_prefix *)(fake + 2), (clef_prefix *)(fake + 3), (clef_prefix *)(fake + 0) };
    for (int i = 0; i < 4; i++) {
        if (warmed_n != 4 || warmed[i] != expected[i]) {
            fprintf(stderr, "keep-warm round after an interruption warmed %d entries, position %d was entry %ld (expected %ld)\n",
                    warmed_n, i, warmed_n > i ? (long)((const char *)warmed[i] - fake) : -1L, (long)((const char *)expected[i] - fake));
            return 1;
        }
    }
    for (int i = 0; i < 3; i++) cache[i].p = NULL;
    template_entry = NULL;
    S.head = S.tail = NULL;
    printf("server keep-warm: an arriving request stops the idle round after one entry; the next round resumes there\n");

    /* A full table of populated entries; a new key whose pass overflowed must not take a slot
       (it would evict a populated entry for nothing), and an existing key's overflowing pass
       must drop its now-useless buffers (review #33). */
    static char populated[CACHE_ENTRIES];
    for (int i = 0; i < CACHE_ENTRIES; i++) { cache[i].p = (clef_prefix *)(populated + i); snprintf(cache[i].key, sizeof(cache[i].key), "k%d", i); cache[i].used = i; }
    S.cache_bytes = (size_t)CACHE_ENTRIES * 100 + 100;
    job keyed = {0};
    snprintf(keyed.cache_key, sizeof(keyed.cache_key), "new-key");
    float ***probs = NULL;
    assert(run_keyed(&keyed, &probs, kerr, sizeof(kerr)));
    int intact = 0;
    for (int i = 0; i < CACHE_ENTRIES; i++) intact += cache[i].p == (clef_prefix *)(populated + i);
    if (intact != CACHE_ENTRIES || !freed_fresh) { fprintf(stderr, "an overflowing pass under a new key took a slot: %d of %d populated entries intact, fresh entry freed %d times\n", intact, CACHE_ENTRIES, freed_fresh); return 1; }
    for (int i = 0; i < CACHE_ENTRIES; i++) { cache[i].p = NULL; cache[i].key[0] = '\0'; }
    printf("server prefix cache: an overflowing pass under a new key takes no slot\n");
    return 0;
}
