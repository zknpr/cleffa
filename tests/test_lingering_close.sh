#!/bin/sh
# Regression (review #5): after an error response the server half-closes and drains the client's
# unread input, so its close sends FIN, not an RST that can destroy the response in flight. When the
# client had half-closed as well, the socket was shut both ways, macOS refused the drain's
# setsockopt (EINVAL), and the server closed with the body unread. A loopback client cannot see that
# RST, so the server runs with tests/close_trace.c interposed on close() and the test checks the
# unread byte count at close.
#   1. 64 KiB and 256 KiB of body after an over-limit Content-Length, then half-close: the 413
#      arrives and the server closes with 0 bytes unread.
#   2. Control: 3 MiB of body passes the drain's 1 MiB cap, so the server closes with bytes unread
#      by design. This shows the probe sees unread input (the test cannot pass vacuously).
# Usage: tests/test_lingering_close.sh MODEL.gguf [PORT]   (starts its own ./clef-server)
set -eu
model=$1; port=${2:-18433}
tmp=$(mktemp -d "${TMPDIR:-/tmp}/clef-linger.XXXXXX")
pid=
trap '[ -n "$pid" ] && kill "$pid" 2>/dev/null; rm -rf "$tmp"' EXIT
cc -dynamiclib -o "$tmp/close_trace.dylib" tests/close_trace.c
DYLD_INSERT_LIBRARIES="$tmp/close_trace.dylib" ./clef-server -m "$model" --port "$port" 2> "$tmp/log" &
pid=$!
i=0
until curl -s -o /dev/null "http://127.0.0.1:$port/health"; do
    i=$((i + 1)); [ $i -gt 240 ] && { echo "server did not start"; cat "$tmp/log"; exit 1; }
    sleep 0.5
done
cat > "$tmp/client.py" <<'EOF'
import socket, sys
port, body = int(sys.argv[1]), int(sys.argv[2])
got = b""
s = socket.create_connection(("127.0.0.1", port), timeout=5)
try:
    s.sendall(b"POST /v1/systemone HTTP/1.1\r\nHost: x\r\nContent-Length: 999999999999\r\n\r\n" + b"x" * body)
    s.shutdown(socket.SHUT_WR)
    while chunk := s.recv(65536):
        got += chunk
except OSError:
    pass   # the control arm is reset mid-send by design
print("413" if got.startswith(b"HTTP/1.1 413") else "no-413")
EOF

# arm BODY: run the client, wait for the server's close, print "RESPONSE UNREAD".
arm() {
    before=$(grep -c '^close_trace:' "$tmp/log" || true)
    resp=$(.venv/bin/python "$tmp/client.py" "$port" "$1")
    i=0
    while [ "$(grep -c '^close_trace:' "$tmp/log" || true)" -le "$before" ]; do
        i=$((i + 1)); [ $i -gt 50 ] && { echo "$resp none"; return; }
        sleep 0.1
    done
    echo "$resp $(grep '^close_trace:' "$tmp/log" | sed -n "$((before + 1))p" | awk '{print $3}')"
}

fail=0
for body in 65536 262144; do
    set -- $(arm $body)
    if [ "$1" = 413 ] && [ "$2" = 0 ]; then r=OK; else r=FAIL; fail=1; fi
    printf '%-38s %-6s closed with %s bytes unread  %s\n' "half-closed, $body B of body:" "$1," "$2" "$r"
done
set -- $(arm 3145728)
case $2 in none|0) r=FAIL; fail=1 ;; *) r=OK ;; esac
printf '%-38s closed with %s bytes unread (expected > 0)  %s\n' "control, 3 MiB (past the drain cap):" "$2" "$r"
exit $fail
