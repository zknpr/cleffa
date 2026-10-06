"""The server's keep-warm passes remove the start delay of a request that follows an idle gap,
and change no answer.

Usage: test_keep_warm.py MODEL.gguf     (starts its own ./clef-server twice, one at a time)

After an idle gap, this machine delays the next pass between commit and GPU start
(~205 ms and up on the 27B, ~100 ms on clef-flash) while its execution interval stays the same.
Referencing the model buffers periodically removes the delay; the driver mechanism is unconfirmed.
The server logs both intervals with
CLEF_STAGE_TIME=1, so the test reads the delay directly as wait - exec instead of inferring it
from total latency, which also moves with GPU clocks and temperature.

  1. Control, --no-keep-warm: requests after a 5 s gap start at least CONTROL_MIN_MS late. This
     is what makes the test meaningful: if the platform stops showing the delay, the test says so
     (exit 2) instead of passing vacuously.
  2. Default server: the same requests start within WARM_MAX_MS.
  3. Both servers return the same answers: the keep-warm passes only read.
"""

from __future__ import annotations

import http.client
import json
import os
import re
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GAP_S, GAPS, CONTROL_MIN_MS, WARM_MAX_MS = 5.0, 3, 40.0, 20.0
STAGE = re.compile(r"clef: stage T=\d+ n=1: encode [\d.]+ ms, wait ([\d.]+) ms, exec ([\d.]+) ms")
BODY = json.dumps({
    "model": "clef", "state": "Checkout has been failing for every customer for the last hour.",
    "questions": {
        "urgent": {"type": "noul", "instructions": "Is this support request urgent?"},
        "team": {"type": "choice", "instructions": "Which team should handle this request?",
                 "criteria": {"billing": "Payments, invoices, and refunds",
                              "technical": "Outages, errors, and configuration", "sales": "Plans and upgrades"}},
    },
}).encode()


def arm(model: str, flags: list[str], tmp: Path, name: str) -> tuple[list[float], list, list[float]]:
    """Returns post-gap GPU start delays, all responses, and post-gap HTTP times in ms."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    log_path = tmp / (name + ".log")
    with log_path.open("w") as log:
        env = {k: v for k, v in os.environ.items() if not k.startswith("CLEF_")}
        server = subprocess.Popen([str(ROOT / "clef-server"), "-m", model, "--port", str(port), *flags],
                                  cwd=ROOT, stdout=log, stderr=log, env=dict(env, CLEF_STAGE_TIME="1"))
        conn = None
        try:
            deadline = time.monotonic() + 240
            while "on http://" not in log_path.read_text():
                if server.poll() is not None:
                    sys.exit(f"{name}: server exited during startup:\n{log_path.read_text()}")
                if time.monotonic() >= deadline:
                    sys.exit(f"{name}: server startup timed out")
                time.sleep(0.2)
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
            answers = []
            http_ms = []
            for i in range(3 + GAPS):
                if i >= 3:
                    time.sleep(GAP_S)
                start = time.monotonic()
                conn.request("POST", "/v1/systemone", BODY, {"Content-Type": "application/json"})
                resp = conn.getresponse()
                data = resp.read()
                http_ms.append((time.monotonic() - start) * 1e3)
                if resp.status != 200:
                    sys.exit(f"{name}: HTTP {resp.status}: {data[:200]!r}")
                json.loads(data)  # Also reject invalid JSON rather than comparing repeated error text.
                answers.append(data)
        finally:
            if conn:
                conn.close()
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
    text = log_path.read_text()
    if "keep-warm pass failed" in text:
        sys.exit(f"{name}: an idle GPU pass failed")
    if ("keep-warm off" in text) != ("--no-keep-warm" in flags):
        sys.exit(f"{name}: the server's keep-warm state does not match its flags")
    stages = [(float(w), float(x)) for w, x in STAGE.findall(text)]
    if len(stages) < 3 + GAPS:
        sys.exit(f"{name}: expected {3 + GAPS} stage lines, found {len(stages)}")
    return [w - x for w, x in stages[-GAPS:]], answers, http_ms[-GAPS:]


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    model = sys.argv[1]
    with tempfile.TemporaryDirectory(prefix="clef-keepwarm.") as d:
        cold, cold_answers, cold_http = arm(model, ["--no-keep-warm"], Path(d), "control")
        warm, warm_answers, warm_http = arm(model, [], Path(d), "default")
    print(f"control (--no-keep-warm): start delay after a {GAP_S:g} s gap: " + ", ".join(f"{v:.1f}" for v in cold) + " ms")
    print(f"default (keep-warm):      start delay after a {GAP_S:g} s gap: " + ", ".join(f"{v:.1f}" for v in warm) + " ms")
    print("control HTTP ms: " + ", ".join(f"{v:.1f}" for v in cold_http))
    print("default HTTP ms: " + ", ".join(f"{v:.1f}" for v in warm_http))
    if statistics.median(cold) < CONTROL_MIN_MS:
        print(f"control arm shows no idle start delay (< {CONTROL_MIN_MS:g} ms): keep-warm is not exercised here")
        sys.exit(2)
    if max(warm) > WARM_MAX_MS:
        sys.exit(f"FAIL: with keep-warm a request still started {max(warm):.1f} ms late (limit {WARM_MAX_MS:g} ms)")
    if any(a != cold_answers[0] for a in cold_answers + warm_answers):
        sys.exit("FAIL: answers differ between requests or between servers")
    print(f"ok: keep-warm removes the idle start delay ({statistics.median(cold):.0f} ms -> "
          f"{statistics.median(warm):.1f} ms) and every answer is identical")


if __name__ == "__main__":
    main()
