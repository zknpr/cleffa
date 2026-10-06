"""Prepare or collect the public checkout fixtures through Cloudflare Workers AI.

The default writes a plan without making network requests. Add --run to collect.
Credentials come only from CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN, or the
environment variable named by --token-env. Every completed call is saved immediately.
No retries, redirects, private input files, or automatic truncation are used.

  .venv/bin/python -B bench/cloudflare_checkout.py golden/cloudflare-checkout.jsonl
  .venv/bin/python -B bench/cloudflare_checkout.py golden/cloudflare-checkout.jsonl --run
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import http.client
import json
import os
import random
import re
import subprocess
import time
from pathlib import Path

from checkout_latency import ROOT, SOURCE, fixtures, positive, request_body


def plan(models: list[str], passes: int) -> dict:
    requests = []
    for model in models:
        rows = fixtures()
        payloads = [json.loads(request_body(r, model)) for r in rows]
        encoded = subprocess.run(
            [str(ROOT / "clef-tool"), "encode-strict", str(ROOT / "gguf" / f"{model}.gguf")],
            input="".join(json.dumps(p) + "\n" for p in payloads),
            text=True, capture_output=True, check=True,
        )
        for row, payload, line in zip(rows, payloads, encoded.stdout.splitlines(), strict=True):
            body = json.dumps(payload).encode()
            requests.append({"model": model, "id": row["id"], "request": payload,
                             "request_sha256": hashlib.sha256(body).hexdigest(),
                             "state_bytes": len(row["state"].encode()),
                             "full_input_tokens": len(json.loads(line)["input_ids"])})
    return {"type": "plan", "source": SOURCE,
            "padding": "Synthetic filler; not external chart payloads",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "passes": passes, "warmups_per_model": 1, "client_retries": 0,
            "planned_calls": len(models) + len(requests) * passes, "requests": requests}


def response_info(status: int, response: object, expected: dict) -> dict:
    if status != 200:
        raise ValueError(f"HTTP {status}")
    if not isinstance(response, dict):
        raise ValueError("response is not an object")
    if response.get("success") is False or response.get("errors"):
        raise ValueError("Cloudflare reported an unsuccessful request")
    answer = response.get("result", response)
    if not isinstance(answer, dict) or not isinstance(answer.get("answers"), dict):
        raise ValueError("response has no answers object")
    if set(answer["answers"]) != set(expected["request"]["questions"]):
        raise ValueError("returned question IDs differ from the request")
    usage = answer.get("usage")
    tokens = usage.get("input_tokens") if isinstance(usage, dict) else None
    if type(tokens) is not int or tokens < 0:
        tokens = None
    # A count mismatch does not identify its cause; exclude it from an equal-token comparison.
    return {"answer": answer, "reported_input_tokens": tokens,
            "full_input_reported": None if tokens is None else tokens == expected["full_input_tokens"]}


def collect(run_plan: dict, out, account: str, token: str) -> None:
    conn = http.client.HTTPSConnection("api.cloudflare.com", timeout=120)
    headers = {"Content-Type": "application/json", "Connection": "keep-alive",
               "Authorization": f"Bearer {token}"}
    try:
        models = list(dict.fromkeys(r["model"] for r in run_plan["requests"]))
        for model in models:
            rows = [r for r in run_plan["requests"] if r["model"] == model]
            schedule = [(-1, rows[0])]
            for rep in range(run_plan["passes"]):
                order = rows.copy()
                random.Random(42 + rep).shuffle(order)
                schedule.extend((rep, row) for row in order)
            path = f"/client/v4/accounts/{account}/ai/run/@cf/cloudflare/{model}"
            for rep, row in schedule:
                body = json.dumps(row["request"]).encode()
                if hashlib.sha256(body).hexdigest() != row["request_sha256"]:
                    raise ValueError("request changed after planning")
                record = {"type": "call", "model": model, "id": row["id"], "pass": rep,
                          "warmup": rep == -1, "request_sha256": row["request_sha256"],
                          "full_input_tokens": row["full_input_tokens"]}
                start = time.perf_counter()
                try:
                    conn.request("POST", path, body, headers)
                    result = conn.getresponse()
                    raw = result.read(1024 * 1024 + 1)
                    record.update(ms=(time.perf_counter() - start) * 1000, status=result.status,
                                  cf_ray=result.getheader("cf-ray"),
                                  server_timing=result.getheader("server-timing"))
                    if len(raw) > 1024 * 1024:
                        raise ValueError("response exceeds 1 MiB")
                    # Do not persist credentials, even if an error body echoes a request header.
                    clean = raw.decode("utf-8").replace(token, "[REDACTED]")
                    record["response"] = json.loads(clean)
                    record.update(response_info(result.status, record["response"], row))
                except Exception as exc:
                    record.setdefault("ms", (time.perf_counter() - start) * 1000)
                    record["error"] = str(exc).replace(token, "[REDACTED]")
                    out.write(json.dumps(record) + "\n")
                    out.flush()
                    raise RuntimeError(f"{model}/{row['id']}: collection failed; see output journal") from None
                out.write(json.dumps(record) + "\n")
                out.flush()
                print(f"{model} {row['id']} pass={rep}: {record['ms']:.1f} ms, "
                      f"reported/full tokens={record['reported_input_tokens']}/{row['full_input_tokens']}",
                      flush=True)
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--models", nargs="+", choices=["clef-flash", "clef"], default=["clef-flash", "clef"])
    parser.add_argument("--passes", type=positive, default=3)
    parser.add_argument("--run", action="store_true", help="send the planned public fixtures to Cloudflare")
    parser.add_argument("--token-env", default="CLOUDFLARE_API_TOKEN")
    args = parser.parse_args()
    account = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
    token = os.environ.get(args.token_env, "")
    if args.run:
        if not re.fullmatch(r"[a-fA-F0-9]{32}", account):
            parser.error("CLOUDFLARE_ACCOUNT_ID must be a 32-digit hexadecimal account ID")
        if not token or any(ord(c) <= 32 or ord(c) >= 127 for c in token):
            parser.error("token environment variable is missing or has invalid characters")
    run_plan = plan(list(dict.fromkeys(args.models)), args.passes)
    run_plan["mode"] = "collect" if args.run else "plan-only"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Refuse to overwrite evidence from an earlier paid run.
    with args.out.open("x") as out:
        out.write(json.dumps(run_plan) + "\n")
        out.flush()
        if args.run:
            collect(run_plan, out, account, token)
        else:
            print(f"Plan saved: {run_plan['planned_calls']} calls. No network requests sent.")


if __name__ == "__main__":
    main()
