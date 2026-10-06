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
#define main clef_server_main
#define write test_write
#include "../clef_server.c"
#undef write
#undef main
static int warm_calls;
static job arriving;
bool test_keep_warm(clef_engine *e, char *err, size_t errlen) { (void)e; (void)err; (void)errlen; return true; }
bool test_prefix_keep_warm(clef_engine *e, const clef_prefix *p, char *err, size_t errlen) {
    (void)e; (void)p; (void)err; (void)errlen;
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
    for (int i = 0; i < 3; i++) cache[i].p = NULL;
    template_entry = NULL;
    S.head = S.tail = NULL;
    printf("server keep-warm: an arriving request stops the idle round after one entry\n");
    return 0;
}
