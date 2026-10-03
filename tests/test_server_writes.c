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

#define main clef_server_main
#define write test_write
#include "../clef_server.c"
#undef write
#undef main

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
    return 0;
}
