"""clef-server checks against a running server.

Usage: test_server.py PORT MODEL.gguf REQUESTS.jsonl
  - every corpus request over HTTP == the CLI response for the same request (byte-equal), with the
    CLI run in the server's default modes (--strict --no-truncate). The corpus fits the window, so
    one extra request (the longest one with its state doubled) must be rejected with a 400 and the
    CLI's error body instead of being silently truncated
  - 32 concurrent clients: correct, isolated responses (exercises micro-batching)
  - protocol and limit handling: bad JSON 400, bad request 400, wrong method 405, unknown
    path 404, oversize body 413, oversize headers 431, chunked body refused, keep-alive
"""

from __future__ import annotations

import concurrent.futures
import http.client
import json
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def post(port: int, body: bytes, path: str = "/v1/systemone", method: str = "POST", headers=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
    c.request(method, path, body=body, headers=headers or {"Content-Type": "application/json"})
    r = c.getresponse()
    data = r.read()
    c.close()
    return r.status, data


def raw(port: int, payload: bytes) -> bytes:
    s = socket.create_connection(("127.0.0.1", port), timeout=30)
    s.sendall(payload)
    out = b""
    try:
        while True:
            chunk = s.recv(65536)
            if not chunk:
                break
            out += chunk
    except socket.timeout:
        pass
    s.close()
    return out


def main() -> None:
    port, model, requests = int(sys.argv[1]), sys.argv[2], sys.argv[3]
    fails = 0
    lines = [l.rstrip("\n") for l in open(requests) if l.strip()]
    over = json.loads(max(lines, key=len))
    over["state"] += over["state"]                       # ~2x the 16k window: the reference would truncate it
    lines.append(json.dumps(over, ensure_ascii=False))
    cli = subprocess.run([ROOT / "clef", "-m", model, "--strict", "--no-truncate"], input="\n".join(lines) + "\n",
                         capture_output=True, text=True).stdout.splitlines()

    same = rejected = 0
    for line, want in zip(lines, cli):
        st, got = post(port, line.encode())
        is_err = want.startswith('{"error"')
        same += st == (400 if is_err else 200) and got.decode() == want
        rejected += is_err
    print(f"http == cli: {same}/{len(lines)} ({rejected} over-long state rejected with 400, expected 1)")
    fails += same != len(lines) or rejected != 1

    # concurrent clients: shuffle-free mapping, each response must equal its own CLI output
    jobs = [(i % len(lines)) for i in range(32)]
    with concurrent.futures.ThreadPoolExecutor(32) as ex:
        res = list(ex.map(lambda i: post(port, lines[i].encode()), jobs))
    ok = sum(got.decode() == cli[i] for i, (st, got) in zip(jobs, res))
    print(f"32 concurrent clients: {ok}/32 correct")
    fails += ok != 32

    checks = [
        ("bad json", post(port, b"{not json")[0], 400),
        ("missing questions", post(port, b'{"model":"m","state":"s"}')[0], 400),
        ("wrong method", post(port, b"", method="GET")[0], 405),
        ("unknown path", post(port, b"{}", path="/nope")[0], 404),
        ("health", post(port, b"", path="/health", method="GET")[0], 200),
    ]
    # the server rejects on the declared length without reading the body (no upload needed)
    over = b"POST /v1/systemone HTTP/1.1\r\nHost: x\r\nContent-Length: 8388609\r\n\r\n"
    checks.append(("oversize body", int(raw(port, over)[9:12] or 0), 413))
    big_hdr = b"POST /v1/systemone HTTP/1.1\r\nHost: x\r\nX-Pad: " + b"a" * 20000 + b"\r\n\r\n"
    checks.append(("oversize headers", int(raw(port, big_hdr)[9:12] or 0), 431))
    chunked = b"POST /v1/systemone HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n0\r\n\r\n"
    checks.append(("chunked refused", int(raw(port, chunked)[9:12] or 0), 400))
    dup = b"POST /v1/systemone HTTP/1.1\r\nHost: x\r\nContent-Length: 2\r\nContent-Length: 5\r\n\r\n{}"
    checks.append(("duplicate content-length", int(raw(port, dup)[9:12] or 0), 400))
    # optional whitespace after a header value is legal (RFC 9110); review #2, M4
    ows_body = lines[0].encode()
    ows = (b"POST /v1/systemone HTTP/1.1\r\nHost: x\r\nConnection: close\r\nContent-Length: "
           + str(len(ows_body)).encode() + b" \t\r\n\r\n" + ows_body)
    checks.append(("content-length trailing OWS", int(raw(port, ows)[9:12] or 0), 200))
    # keep-alive: two requests pipelined on one connection, both answered
    body = lines[0].encode()
    req = b"POST /v1/systemone HTTP/1.1\r\nHost: x\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
    out = raw(port, req + req)  # second request sent right after the first, same connection
    checks.append(("pipelined keep-alive", out.count(b"HTTP/1.1 200 OK"), 2))
    for name, got, want in checks:
        print(f"{name:26s} {'OK' if got == want else 'FAIL'} ({got})")
        fails += got != want
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
