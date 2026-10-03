"""Regression (review #4): a client that keeps trickling bytes must not hold a connection past the
request deadline (slowloris). SO_RCVTIMEO bounds each read only; the server now keeps a monotonic
deadline per request and per error-drain.

Usage: test_server_slow.py PORT
The server must run with CLEF_DEBUG_IO_TIMEOUT=2 (request deadline 2 s instead of 30 s). Each arm
sends one byte every 0.4 s, which never lets a single read time out, and measures when the server
drops the connection:
  - headers that never finish;
  - a body that never finishes;
  - the drain after an error response (413), which used to read until 1 MiB had arrived.
Pass: every connection is dropped within the deadline plus slack, well before the 10 s cap.
"""

from __future__ import annotations

import socket
import sys
import time

TIMEOUT, SLACK, CAP, TICK = 2.0, 2.0, 10.0, 0.4


def trickle(port: int, prefix: bytes, expect_response: bool) -> tuple[float, bytes]:
    """Send prefix, then one byte per TICK; return (seconds until the server dropped us, response)."""
    s = socket.create_connection(("127.0.0.1", port), timeout=5)
    s.sendall(prefix)
    t0 = time.monotonic()
    got = b""
    s.setblocking(False)
    while time.monotonic() - t0 < CAP:
        try:
            chunk = s.recv(65536)
            if chunk:
                got += chunk
            elif not expect_response or got:
                # FIN. After an error response the server half-closes first and drains, so for
                # that arm only a failed send (below) means the socket is gone.
                if not expect_response:
                    return time.monotonic() - t0, got
        except BlockingIOError:
            pass
        except (ConnectionResetError, BrokenPipeError):
            return time.monotonic() - t0, got
        try:
            s.send(b"x")
        except (ConnectionResetError, BrokenPipeError, OSError):
            return time.monotonic() - t0, got
        time.sleep(TICK)
    s.close()
    return float("inf"), got


def main() -> None:
    port = int(sys.argv[1])
    arms = [
        ("headers never finish", b"POST /v1/systemone HTTP/1.1\r\nHost: x\r\nX-Slow: ", False),
        ("body never finishes", b"POST /v1/systemone HTTP/1.1\r\nHost: x\r\nContent-Length: 100000\r\n\r\n{", False),
        ("drain after 413", b"POST /v1/systemone HTTP/1.1\r\nHost: x\r\nContent-Length: 999999999999\r\n\r\n", True),
    ]
    fails = 0
    for name, prefix, expect_response in arms:
        dt, got = trickle(port, prefix, expect_response)
        ok = dt <= TIMEOUT + SLACK and (not expect_response or got.startswith(b"HTTP/1.1 413"))
        fails += not ok
        shown = "still open at the 10 s cap" if dt == float("inf") else f"dropped after {dt:.1f} s"
        print(f"{name:22s} {shown:28s} {'OK' if ok else 'FAIL'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
