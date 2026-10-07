"""The server's prefix cache returns the bytes of an uncached answer, isolates entries by key,
stays within its budget and refuses a malformed key.

Usage: test_server_prefix.py MODEL.gguf REQUESTS.jsonl   (starts its own ./clef-server three times)

REQUESTS.jsonl is the corpus; its long log requests supply the states.

  1. Reference server without the option: every body, that a valid key changes nothing there, and
     that a malformed key is still HTTP 400 (the header is validated whether or not caching is on).
  2. Server with --prefix-cache-mb: the same requests under a key return the same bytes, and the
     log shows which reused an entry. A second key never reuses the first key's entry, although
     its requests are identical. A request without the header never touches an entry.
  3. A budget smaller than one entry never allocates one: the request is served uncached and the
     projected size is logged; answers do not change. Every cached pass logs its projected and
     retained size, and the retained size never exceeds the projection.
     A budget that holds a short entry but not a long one drops only the long entry: the other
     key's entry survives and keeps reusing.
  3b. Keyed requests too short to cache never take a table slot: 33 of them under new keys leave
     the populated entries in place.
  3c. The projection charges only checkpoints that need new buffers: after an 8K entry has
     allocated its slots, a different request under the same key reuses them, so it is cached under
     a budget that holds the entry but not an over-charged projection.
  4. A key with a character outside [A-Za-z0-9._-], or longer than 64, is HTTP 400.
"""

from __future__ import annotations

import http.client
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
model, corpus_path = sys.argv[1], sys.argv[2]
corpus = [json.loads(line) for line in open(corpus_path)]
logs = sorted((r for r in corpus if isinstance(r["state"], str) and len(r["state"]) > 2000), key=lambda r: len(r["state"]))
A, B, LONG = logs[0], logs[1], logs[-1]
SHORT = min(corpus, key=lambda r: len(json.dumps(r)))   # under the 128-token minimum prefix: bypasses the entry
other_q = next(r["questions"] for r in corpus if r["questions"] != A["questions"])
REUSED = re.compile(r"prefix cache: reused (\d+) of (\d+) tokens")
SIZED = re.compile(r"prefix cache: projected (\d+) MB, holds (\d+) MB")


def fail(msg: str) -> None:
    sys.exit("FAIL: " + msg)


class Server:
    def __init__(self, flags: list[str], tmp: Path, name: str):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]
        self.log_path = tmp / (name + ".log")
        self.log = self.log_path.open("w")
        env = {k: v for k, v in os.environ.items() if not k.startswith("CLEF_")}
        self.proc = subprocess.Popen([str(ROOT / "clef-server"), "-m", model, "--port", str(self.port), *flags],
                                     cwd=ROOT, stdout=self.log, stderr=self.log, env=env)
        deadline = time.monotonic() + 240
        while True:
            try:
                c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
                c.request("GET", "/health")
                if c.getresponse().status == 200:
                    break
            except OSError:
                pass
            if self.proc.poll() is not None or time.monotonic() > deadline:
                self.stop()
                fail(f"server did not start ({name})")
            time.sleep(0.5)

    def post(self, req: dict, key: str | None = None) -> tuple[int, bytes]:
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=600)
        headers = {"Content-Type": "application/json"}
        if key is not None:
            headers["X-Clef-Prefix-Cache"] = key
        c.request("POST", "/v1/systemone", json.dumps(req).encode(), headers)
        r = c.getresponse()
        body = r.read()
        c.close()
        return r.status, body

    def reused(self) -> list[int]:
        self.log.flush()
        return [int(m.group(1)) for m in REUSED.finditer(self.log_path.read_text())]

    def dropped(self) -> int:
        self.log.flush()
        return self.log_path.read_text().count("prefix cache: dropped")

    def bypassed(self) -> int:
        self.log.flush()
        return self.log_path.read_text().count("would exceed the budget")

    def sizes(self) -> list[tuple[int, int]]:
        self.log.flush()
        return [(int(m.group(1)), int(m.group(2))) for m in SIZED.finditer(self.log_path.read_text())]

    def stop(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.log.close()


requests = [A, dict(A, questions=other_q), B, dict(B, questions=other_q)]

with tempfile.TemporaryDirectory() as t:
    tmp = Path(t)
    ref = Server([], tmp, "plain")
    try:
        want = [ref.post(r) for r in requests]
        want_long = ref.post(LONG)
        want_short = ref.post(SHORT)
        if any(st != 200 for st, _ in want + [want_long, want_short]):
            fail("reference server returned an error")
        if ref.post(A, "tenant-1") != want[0] or ref.reused():
            fail("the header changed an answer on a server without --prefix-cache-mb")
        if ref.post(A, "has space")[0] != 400:
            fail("a malformed key was accepted on a server without --prefix-cache-mb")
    finally:
        ref.stop()

    srv = Server(["--prefix-cache-mb", "20000"], tmp, "cached")
    try:
        # key tenant-1: A (new entry), A with other questions (hit), B (replaces), B other questions (hit)
        for r, w in zip(requests, want):
            if srv.post(r, "tenant-1") != w:
                fail("a cached answer differs from the uncached bytes")
        got = srv.reused()
        if len(got) != 4 or got[0] != 0 or got[1] == 0 or got[2] != 0 or got[3] == 0:
            fail(f"unexpected reuse under one key: {got}")
        # another key sends the request the first key just cached: it must not see that entry
        if srv.post(requests[3], "tenant-2") != want[3]:
            fail("a second key's answer differs")
        if srv.reused()[-1] != 0:
            fail("a second key reused the first key's entry")
        # no header: no entry is read or written
        n = len(srv.reused())
        if srv.post(requests[3]) != want[3] or len(srv.reused()) != n:
            fail("a request without the header used the cache")
        # both keys still hit their own entries
        if srv.post(requests[2], "tenant-1") != want[2] or srv.reused()[-1] == 0:
            fail("the first key lost its entry")
        if srv.post(requests[2], "tenant-2") != want[2] or srv.reused()[-1] == 0:
            fail("the second key lost its entry")
        # 33 new keys with a request too short to cache: no slot is taken, so the two populated
        # entries survive and keep reusing (before the fix, the table filled and evicted them)
        drops = srv.dropped()
        for k in range(33):
            if srv.post(SHORT, f"flush-{k}") != want_short:
                fail("a short keyed request's answer differs")
        if srv.reused()[-1] != 0:
            fail("the short request was cached; pick a shorter one for this scenario")
        if srv.dropped() != drops:
            fail(f"short keyed requests evicted populated entries: {srv.dropped() - drops} drops")
        if srv.post(requests[2], "tenant-1") != want[2] or srv.reused()[-1] == 0:
            fail("the first key lost its entry to uncacheable keyed requests")
        if srv.post(requests[2], "tenant-2") != want[2] or srv.reused()[-1] == 0:
            fail("the second key lost its entry to uncacheable keyed requests")
        # the projection is an upper bound on the retained size, and not a loose one
        sizes = srv.sizes()
        if not sizes or any(held > projected for projected, held in sizes):
            fail(f"a retained entry exceeded its projection: {sizes}")
        if any(projected > 2 * held for projected, held in sizes if held):
            fail(f"a projection was more than twice the retained size: {sizes}")
        for bad in ("has space", "semi;colon", "x" * 65, ""):
            st, _ = srv.post(A, bad)
            if st != 400:
                fail(f"key {bad[:20]!r} was accepted with status {st}")
        print(f"server prefix cache: answers byte-identical; reuse per keyed request {srv.reused()}; keys isolated; bad keys refused")
    finally:
        srv.stop()

    # a budget of 1 MB holds no entry: every keyed request is served uncached without allocating one
    small = Server(["--prefix-cache-mb", "1"], tmp, "small")
    try:
        for r, w in zip(requests[:2], want[:2]):
            if small.post(r, "tenant-1") != w:
                fail("an answer changed under a budget smaller than one entry")
        if small.bypassed() != 2 or small.dropped() or any(small.reused()):
            fail(f"budget not enforced before allocation: {small.bypassed()} bypasses, {small.dropped()} drops, reuse {small.reused()}")
        print(f"server prefix cache over budget: {small.bypassed()} requests served uncached, nothing allocated, answers unchanged")
    finally:
        small.stop()

    # 700 MiB holds the entry for A (2,235 tokens, under 0.5 GB with its checkpoints) but not the
    # one for LONG (16,347 tokens, about 1.9 GB). The oversized entry must be the only one dropped.
    mid = Server(["--prefix-cache-mb", "700"], tmp, "mid")
    try:
        if mid.post(requests[0], "tenant-1") != want[0] or mid.post(requests[1], "tenant-1") != want[1]:
            fail("an answer changed under the 700 MiB budget")
        if mid.reused()[-1] == 0:
            fail("the short entry did not fit the 700 MiB budget; the scenario needs a larger budget")
        if mid.post(LONG, "tenant-2") != want_long:
            fail("the oversized request's answer changed")
        if mid.bypassed() != 1 or mid.dropped():
            fail(f"an oversized request was not bypassed before allocation: {mid.bypassed()} bypasses, {mid.dropped()} drops")
        if mid.post(requests[1], "tenant-1") != want[1] or mid.reused()[-1] == 0:
            fail("the short entry was evicted by an oversized entry under another key")
        print(f"server prefix cache oversized entry: served uncached without allocation, other key kept reusing ({mid.reused()})")
    finally:
        mid.stop()

    # 1,120 MiB (1,174 MB) holds the 8,072-token entry (1,151 MB with its five checkpoints) but not a
    # projection that charges one more checkpoint (1,204 MB). A then replaces it under the same key: its
    # stores land in already-allocated slots, so the entry does not grow and the request must be cached.
    tight = Server(["--prefix-cache-mb", "1120"], tmp, "tight")
    try:
        if tight.post(requests[2], "tenant-1") != want[2] or tight.bypassed():
            fail(f"the 8K entry did not fit the 1,120 MiB budget ({tight.sizes()}); the scenario needs a larger budget")
        if tight.post(requests[0], "tenant-1") != want[0]:
            fail("an answer changed under the tight budget")
        if tight.bypassed():
            fail(f"a request whose stores reuse allocated slots was bypassed: projections {tight.sizes()}")
        if tight.post(requests[1], "tenant-1") != want[1] or tight.reused()[-1] == 0:
            fail("the replaced entry was not reused")
        print(f"server prefix cache slot reuse: projections {tight.sizes()}, no bypass")
    finally:
        tight.stop()
