"""Regression (review #2, I1): small requests must not be co-batched with a large one.

Usage: test_server_hol.py PORT REQUESTS.jsonl  (corpus: r001..r006 small, r020 ~8k, r021 ~16k tokens)
The server must run with --truncate: r021's state exceeds the window and is rejected by default.
A warm-up request (r020) holds the GPU; meanwhile six small requests and then one large request
(r021) queue up. When the GPU frees, the old worker packed all seven into one forward pass, so
the small requests waited for the 16k-token request (~13 s). Pass: the small requests finish
well before the large one.
"""

from __future__ import annotations

import http.client
import sys
import threading
import time


def post(port: int, body: bytes, out: dict, key: str) -> None:
    t0 = time.perf_counter()
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=300)
    c.request("POST", "/v1/systemone", body=body, headers={"Content-Type": "application/json"})
    r = c.getresponse()
    r.read()
    out[key] = (r.status, time.perf_counter() - t0, time.perf_counter())
    c.close()


def main() -> None:
    port, requests = int(sys.argv[1]), sys.argv[2]
    lines = [l.rstrip("\n").encode() for l in open(requests)]
    res: dict = {}
    threads = [threading.Thread(target=post, args=(port, lines[20], res, "warmup"))]
    threads[0].start()
    time.sleep(0.5)                                   # warm-up is now on the GPU (~4 s)
    for i in range(1, 7):
        t = threading.Thread(target=post, args=(port, lines[i], res, f"small{i}"))
        t.start(); threads.append(t)
    time.sleep(0.3)                                   # small requests queued first
    t = threading.Thread(target=post, args=(port, lines[21], res, "large"))
    t.start(); threads.append(t)
    for t in threads:
        t.join()
    small_done = max(res[f"small{i}"][2] for i in range(1, 7))
    large_done = res["large"][2]
    warm_done = res["warmup"][2]
    print(f"warm-up finished at +0.0 s; small requests done +{small_done - warm_done:.2f} s; "
          f"large done +{large_done - warm_done:.2f} s")
    ok = all(v[0] == 200 for v in res.values()) and (small_done - warm_done) < 0.25 * (large_done - warm_done)
    print("head-of-line isolation:", "OK" if ok else "FAIL (small requests waited for the large one)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
