/* close() interposer for tests/test_lingering_close.sh: logs how many unread bytes each TCP socket
 * still holds when the server closes it. Closing with unread input sends RST instead of FIN, which
 * a client on loopback never notices (the response and FIN arrive first), so the test reads the
 * server side instead. Loaded with DYLD_INSERT_LIBRARIES; it changes nothing but the log. */
#include <stdio.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>

static int traced_close(int fd) {
    struct sockaddr_storage ss;
    socklen_t sl = sizeof(ss);
    int type = 0, unread = 0;
    socklen_t tl = sizeof(type);
    if (getsockname(fd, (struct sockaddr *)&ss, &sl) == 0 && ss.ss_family == AF_INET &&
        getsockopt(fd, SOL_SOCKET, SO_TYPE, &type, &tl) == 0 && type == SOCK_STREAM &&
        ioctl(fd, FIONREAD, &unread) == 0)
        fprintf(stderr, "close_trace: unread %d\n", unread);
    return close(fd);
}

__attribute__((used)) static const struct { const void *replacement, *original; } interpose_close
    __attribute__((section("__DATA,__interpose"))) = { (const void *)traced_close, (const void *)close };
