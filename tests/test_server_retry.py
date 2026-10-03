"""Regression (review #2, M2): a batch-level failure must not fail co-batched requests.

Usage: test_server_retry.py PORT REQUESTS.jsonl MODEL.gguf
Run against a server started with CLEF_DEBUG_FAIL_MULTI=1, which makes every multi-record
forward fail. Eight concurrent requests are queued behind a warm-up request so they form one
batch; the server must retry each record alone and answer all eight correctly (equal to the
CLI response). Without the retry, all eight get HTTP 500.
"""

from __future__ import annotations

import http.client
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def post(port, body, out, key):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=300)
    c.request("POST", "/v1/systemone", body=body, headers={"Content-Type": "application/json"})
    r = c.getresponse()
    out[key] = (r.status, r.read().decode())
    c.close()


def main() -> None:
    port, requests, model = int(sys.argv[1]), sys.argv[2], sys.argv[3]
    lines = [l.rstrip("\n") for l in open(requests)]
    cli = subprocess.run([ROOT / "clef", "-m", model, "--strict", "--no-truncate", requests], capture_output=True, text=True).stdout.splitlines()
    res: dict = {}
    warm = threading.Thread(target=post, args=(port, lines[19].encode(), res, "warm"))
    warm.start()
    time.sleep(0.3)                              # warm-up (~2k tokens) holds the GPU
    ts = [threading.Thread(target=post, args=(port, lines[i].encode(), res, i)) for i in range(1, 9)]
    for t in ts:
        t.start()
    for t in ts + [warm]:
        t.join()
    ok = sum(res[i][0] == 200 and res[i][1] == cli[i] for i in range(1, 9))
    codes = sorted({res[i][0] for i in range(1, 9)})
    print(f"co-batched requests answered correctly after injected batch failure: {ok}/8 (status codes {codes})")
    sys.exit(0 if ok == 8 else 1)


if __name__ == "__main__":
    main()
