"""Regression: the 16 KiB header limit must hold however the header arrives.

Usage: test_server_headers.py PORT
  (a) fresh connection, ~30 KiB header sent in two parts (the buffer grows to 32 KiB while
      reading headers, so a whole oversized header could arrive before the size check)
  (b) keep-alive: a request with a 200 KB body enlarges the buffer, then oversized headers
      (20-60 KiB, small enough to arrive in a single read) follow on the same connection
Both must get 431. Before the fix both were accepted.
"""

import socket
import sys
import time


def status(sock) -> int:
    sock.settimeout(60)
    data = b""
    while b"\r\n" not in data:
        chunk = sock.recv(4096)
        if not chunk:
            break
        data += chunk
    return int(data[9:12]) if data.startswith(b"HTTP/1.1") else 0


def read_response(sock) -> int:
    sock.settimeout(120)
    data = b""
    while b"\r\n\r\n" not in data:
        data += sock.recv(65536)
    head, rest = data.split(b"\r\n\r\n", 1)
    n = int([l for l in head.split(b"\r\n") if l.lower().startswith(b"content-length")][0].split(b":")[1])
    while len(rest) < n:
        rest += sock.recv(65536)
    return int(head[9:12])


def main() -> None:
    port = int(sys.argv[1])
    body = b'{"model":"m","state":"s","questions":{"q":{"type":"noul"}}}'
    fails = 0
    # (a)
    pad = b"X-Pad: " + b"a" * 30000 + b"\r\n"
    req = b"POST /v1/systemone HTTP/1.1\r\nHost: x\r\n" + pad + b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
    s = socket.create_connection(("127.0.0.1", port))
    s.sendall(req[:12300]); time.sleep(0.3); s.sendall(req[12300:])
    st = status(s); s.close()
    print(f"(a) 30 KiB header in two parts: {st} {'OK' if st == 431 else 'FAIL'}"); fails += st != 431
    # (b)
    big = ('{"model":"m","state":"' + "w " * 100000 + '","questions":{"q":{"type":"noul"}}}').encode()
    first = b"POST /v1/systemone HTTP/1.1\r\nHost: x\r\nContent-Length: " + str(len(big)).encode() + b"\r\n\r\n" + big
    for kib in (20, 40, 60):
        second = (b"POST /v1/systemone HTTP/1.1\r\nHost: x\r\nX-Pad: " + b"b" * (kib * 1024) + b"\r\nContent-Length: "
                  + str(len(body)).encode() + b"\r\n\r\n" + body)
        s = socket.create_connection(("127.0.0.1", port))
        s.sendall(first)
        st1 = read_response(s)
        s.sendall(second)
        st = status(s); s.close()
        print(f"(b) keep-alive, {kib} KiB header after a 200 KB body (first: {st1}): {st} {'OK' if st == 431 else 'FAIL'}")
        fails += st != 431
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
