"""Measure full-input localhost HTTP latency on the Cloudflare blog's checkout example.

The longer inputs use synthetic filler, not the unavailable external chart corpus.
Starts one local server at a time; no request leaves this machine.

  .venv/bin/python -B bench/checkout_latency.py golden/checkout-latency.json
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import http.client
import json
import os
import platform
import random
import socket
import statistics
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = "https://blog.cloudflare.com/clef-decision-models/"


def fixtures() -> list[dict]:
    questions = {
        "urgent": {"type": "noul", "instructions": "Is this support request urgent?"},
        "team": {
            "type": "choice", "instructions": "Which team should handle this request?",
            "criteria": {"billing": "Payments, invoices, and refunds",
                         "technical": "Outages, errors, and configuration", "sales": "Plans and upgrades"},
        },
        "severity": {
            "type": "score", "instructions": "How severe is the customer impact?",
            "criteria": ["No impact", "Minor", "Major", "Critical"],
        },
    }
    outage = "Checkout has been failing for every customer for the last hour."
    filler = "Earlier routine service checks completed successfully. "
    rows = [{"id": "blog", "state": outage, "questions": questions}]
    for size, position in [(n, "end") for n in [128, 512, 2048, 8192, 16384, 32768]] + [
        (16384, "start"), (32768, "start")
    ]:
        pad = (filler * (size // len(filler) + 1))[:size - len(outage) - 1]
        state = pad + "\n" + outage if position == "end" else outage + "\n" + pad
        rows.append({"id": f"{size}-{position}", "state": state, "questions": questions})
    return rows


def request_body(row: dict, model: str) -> bytes:
    return json.dumps({**{k: v for k, v in row.items() if k != "id"}, "model": model}).encode()


def engine_env() -> dict[str, str]:
    """The server's environment: the shell's, without any CLEF_* diagnostic. CLEF_PROFILE
    serializes the GPU and CLEF_ATTN_TU=0 or CLEF_HEAD_BF16=1 select another execution mode;
    inherited silently, any of them would make the report describe something other than the
    default engine (review #102)."""
    return {k: v for k, v in os.environ.items() if not k.startswith("CLEF_")}


def gguf_identity(path: Path) -> dict:
    """The measured weights, by size and SHA-256 of the GGUF as it is on disk, hashed before
    the measurement: the model name alone would let a stale or regenerated file pass for the
    pinned model (review #107). tests/verify_gguf.py ties a hash to the pinned snapshot."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(16 << 20), b""):
            digest.update(chunk)
    return {"path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def measure(model: str, rows: list[dict], args: argparse.Namespace) -> list[dict]:
    gguf = ROOT / "gguf" / (model + ".gguf")
    encoded = subprocess.run(
        [str(ROOT / "clef-tool"), "encode-strict", str(gguf)], check=True, text=True,
        input="".join(json.dumps({**r, "model": model}) + "\n" for r in rows), capture_output=True,
    )
    counts = {r["id"]: len(json.loads(line)["input_ids"])
              for r, line in zip(rows, encoded.stdout.splitlines(), strict=True)}
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    log_path = args.out.with_name(args.out.stem + "-" + model + ".log")
    with log_path.open("w") as log:
        server = subprocess.Popen(
            [str(ROOT / "clef-server"), "-m", str(gguf), "--port", str(port)],
            cwd=ROOT, stdout=log, stderr=log, env=engine_env(),
        )
        conn = None
        try:
            deadline = time.monotonic() + 60
            while "on http://" not in log_path.read_text():
                if server.poll() is not None:
                    raise RuntimeError(f"server exited during startup; see {log_path}")
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"server startup timed out; see {log_path}")
                time.sleep(0.1)
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=120)

            def call(row: dict) -> dict:
                body = request_body(row, model)
                start = time.perf_counter()
                conn.request("POST", "/v1/systemone", body,
                             {"Content-Type": "application/json", "Connection": "keep-alive"})
                response = conn.getresponse()
                data = response.read()
                elapsed = (time.perf_counter() - start) * 1000
                if response.status != 200:
                    raise RuntimeError(f"{row['id']}: HTTP {response.status}: {data[:300]!r}")
                answer = json.loads(data)
                # Reject speed obtained by processing fewer tokens than the full request.
                if answer["usage"]["input_tokens"] != counts[row["id"]]:
                    raise RuntimeError(f"{row['id']}: input count differs from full strict encoding")
                seq[0] += 1
                return {"ms": elapsed, "response": answer, "seq": seq[0]}

            seq = [0]   # execution order of every call, warm-ups included, kept with each sample
            for _ in range(3):
                call(rows[0])
            calls = {r["id"]: [] for r in rows}
            for _ in range(args.passes):
                calls[rows[0]["id"]].append(call(rows[0]))
            for rep in range(args.long_passes):
                order = rows[1:].copy()
                random.Random(42 + rep).shuffle(order)
                for row in order:
                    result = call(row)
                    calls[row["id"]].append(result)
                    print(f"{model} {row['id']}: {result['ms']:.1f} ms, {counts[row['id']]} tokens", flush=True)
            result = []
            for row in rows:
                samples = calls[row["id"]]
                if any(s["response"] != samples[0]["response"] for s in samples):
                    raise RuntimeError(f"{row['id']}: identical requests produced different responses")
                result.append({"id": row["id"], "state_bytes": len(row["state"].encode()),
                               "request_sha256": hashlib.sha256(request_body(row, model)).hexdigest(),
                               "input_tokens": counts[row["id"]], "response": samples[0]["response"],
                               "median_ms": statistics.median(s["ms"] for s in samples),
                               "samples_ms": [s["ms"] for s in samples],
                               "sample_seq": [s["seq"] for s in samples]})
            return result
        finally:
            if conn:
                conn.close()
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


def positive(value: str) -> int:
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--models", nargs="+", choices=["clef-flash", "clef"], default=["clef-flash", "clef"])
    parser.add_argument("--passes", type=positive, default=12, help="measured calls of the exact blog example")
    parser.add_argument("--long-passes", type=positive, default=3, help="shuffled passes over padded fixtures")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = fixtures()
    report = {"source": SOURCE, "padding": "Synthetic filler; not external chart payloads",
              "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "platform": platform.platform(), "passes": args.passes, "long_passes": args.long_passes,
              "truncation": False, "requested_models": list(args.models), "complete": False, "models": {}, "gguf": {},
              "engine_overrides": "none: CLEF_* variables are removed from the server's environment",
              "engine_sha256": hashlib.sha256((ROOT / "clef-server").read_bytes()).hexdigest()}
    # Progress goes to a checkpoint beside the output; the output path receives the report only
    # once every requested model has finished, so an interrupted two-model run cannot be
    # mistaken for a complete single-model one. The report also names the models it was asked
    # for and carries a completion marker (review #92).
    partial = args.out.with_name(args.out.name + ".partial")
    for model in args.models:
        report["gguf"][model] = gguf_identity(ROOT / "gguf" / (model + ".gguf"))
        result = measure(model, rows, args)
        report["models"][model] = result
        partial.write_text(json.dumps(report, indent=2) + "\n")
        print(model, [(r["id"], round(r["median_ms"], 1)) for r in result], flush=True)
    report["complete"] = True
    partial.write_text(json.dumps(report, indent=2) + "\n")
    partial.replace(args.out)


if __name__ == "__main__":
    main()
