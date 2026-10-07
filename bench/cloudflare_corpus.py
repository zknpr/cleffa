"""Capture a separate hosted response reference for the fixed public parity corpus.

Default: write a plan, with no network access. --run uses the named account from
the current Wrangler login and checks its daily free allowance before each model.
Never overwrites an existing journal. Does not replace the FP32 numerical oracle.

  .venv/bin/python -B bench/cloudflare_corpus.py golden/cloudflare-corpus.jsonl
  .venv/bin/python -B bench/cloudflare_corpus.py golden/cloudflare-corpus.jsonl --run --account ACCOUNT
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

from cloudflare_checkout import ROOT, collect

sys.path.insert(0, str(ROOT / "ref"))
from corpus import build

# Verified 2026-10-04. Reserve full submitted tokens even if the service truncates.
PRICING = "https://developers.cloudflare.com/workers-ai/platform/pricing/"
RATES = {"clef-flash": 8182, "clef": 21818}
FREE_NEURONS = 10000
HEADROOM = 2000  # Analytics are delayed and other clients may share the account.
RUN_LIMIT = 2500


def hosted_payload(request: dict, model: str) -> dict:
    payload = json.loads(json.dumps({k: v for k, v in request.items() if k != "id"}))
    payload["model"] = model
    # The hosted validator requires instructions; the pinned encoder defaults to
    # the question ID. Make that default explicit and prove token equality below.
    for qid, question in payload["questions"].items():
        question.setdefault("instructions", qid)
    return payload


def plan(passes: int) -> dict:
    rows = []
    for model in RATES:
        requests = build()
        originals = [{**{k: v for k, v in r.items() if k != "id"}, "model": model}
                     for r in requests]
        payloads = [hosted_payload(r, model) for r in requests]
        encoded = subprocess.run(
            [str(ROOT / "clef-tool"), "encode-strict", str(ROOT / "gguf" / f"{model}.gguf")],
            input="".join(json.dumps(p) + "\n" for p in originals + payloads),
            text=True, capture_output=True, check=True,
        )
        encodings = [json.loads(line) for line in encoded.stdout.splitlines()]
        if len(encodings) != 2 * len(requests) or encodings[:len(requests)] != encodings[len(requests):]:
            raise ValueError("Hosted payload adaptation changed the encoded corpus")
        for request, payload, enc in zip(requests, payloads, encodings[:len(requests)], strict=True):
            rows.append({"model": model, "id": request["id"], "request": payload,
                         "request_sha256": hashlib.sha256(json.dumps(payload).encode()).hexdigest(),
                         "full_input_tokens": len(enc["input_ids"]),
                         # The comparator requires this to equal the FP32 oracle encoding's:
                         # equal counts and spans do not prove equal tokens.
                         "input_ids_sha256": hashlib.sha256(json.dumps(enc["input_ids"]).encode()).hexdigest(),
                         "questions": enc["questions"]})
    estimates = {}
    for model, rate in RATES.items():
        tokens = [r["full_input_tokens"] for r in rows if r["model"] == model]
        estimates[model] = (tokens[0] + passes * sum(tokens)) * rate / 1e6
    return {"type": "plan", "source": "ref/corpus.py",
            "corpus_sha256": hashlib.sha256((ROOT / "ref/corpus.py").read_bytes()).hexdigest(),
            "adaptation": "Make missing instructions equal to question ID; exact token/span/option parity checked",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "passes": passes, "warmups_per_model": 1, "client_retries": 0,
            "planned_calls": len(RATES) + passes * len(rows), "requests": rows,
            "pricing_source": PRICING, "neurons_per_million_input_tokens": RATES,
            "full_token_neuron_estimate": estimates, "free_daily_neurons": FREE_NEURONS,
            "reserved_headroom_neurons": HEADROOM}


def credentials(account_name: str) -> tuple[str, str]:
    def wrangler(*args):
        result = subprocess.run(["wrangler", *args, "--json"], capture_output=True, text=True,
                                env={**os.environ, "WRANGLER_SEND_METRICS": "false"})
        if result.returncode:
            raise RuntimeError("Wrangler authentication failed; no inference sent")
        return json.loads(result.stdout)

    matches = [a for a in wrangler("whoami")["accounts"] if a["name"] == account_name]
    if len(matches) != 1 or not re.fullmatch(r"[a-fA-F0-9]{32}", matches[0]["id"]):
        raise ValueError("Account name must identify exactly one accessible account")
    token = wrangler("auth", "token")["token"]
    if not token or any(ord(c) <= 32 or ord(c) >= 127 for c in token):
        raise ValueError("Wrangler returned an invalid token")
    return matches[0]["id"], token


def daily_usage(account: str, token: str) -> dict:
    now = datetime.datetime.now(datetime.timezone.utc)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
    query = """query($account: string, $start: Time, $end: Time) {
      viewer { accounts(filter: {accountTag: $account}) {
        aiInferenceAdaptiveGroups(limit: 1,
          filter: {datetime_geq: $start, datetime_leq: $end}) {
          sum { totalNeurons totalInputTokens totalOutputTokens }
        }
      } }
    }"""
    conn = http.client.HTTPSConnection("api.cloudflare.com", timeout=30)
    try:
        conn.request("POST", "/client/v4/graphql", json.dumps({"query": query, "variables": {
            "account": account, "start": start, "end": now.isoformat()}}),
            {"Authorization": "Bearer " + token, "Content-Type": "application/json"})
        response = conn.getresponse()
        raw = response.read(1024 * 1024 + 1)
        if response.status != 200 or len(raw) > 1024 * 1024:
            raise RuntimeError("Cannot verify daily usage; stopping inference")
        data = json.loads(raw)
        if data.get("errors"):
            raise RuntimeError("Usage query failed; stopping inference")
        accounts = data["data"]["viewer"]["accounts"]
        if len(accounts) != 1:
            raise ValueError("Usage query did not resolve the selected account")
        groups = accounts[0]["aiInferenceAdaptiveGroups"]
        if len(groups) > 1:
            raise ValueError("Unexpected daily usage grouping")
        used = sum(g["sum"]["totalNeurons"] for g in groups)
        if not math.isfinite(used) or used < 0:
            raise ValueError("Invalid daily neuron usage")
        return {"type": "budget", "start": start, "timestamp": now.isoformat(),
                "used_neurons": used}
    finally:
        conn.close()


def check_budget(used: float, estimate: float) -> None:
    if not all(math.isfinite(x) and x >= 0 for x in (used, estimate)):
        raise ValueError("Invalid neuron budget")
    if estimate > RUN_LIMIT or used + estimate + HEADROOM > FREE_NEURONS:
        raise ValueError("Insufficient free allowance; no further inference will be sent")


def final_usage(out, account: str, token: str) -> None:
    """Record the post-run usage observation. The journal is already complete, so a failure
    here is recorded, with the token redacted, instead of failing a valid run."""
    try:
        record = daily_usage(account, token)
    except (RuntimeError, ValueError, OSError, KeyError, TypeError) as exc:
        record = {"type": "budget", "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  "error": str(exc).replace(token, "[REDACTED]")}
    out.write(json.dumps(record) + "\n")
    out.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--account", help="exact account name in the current Wrangler login")
    parser.add_argument("--passes", type=int, choices=[1, 2], default=2)
    args = parser.parse_args()
    if args.run and not args.account:
        parser.error("--run requires --account")
    run_plan = plan(args.passes)
    run_plan["mode"] = "collect" if args.run else "plan-only"
    estimate = sum(run_plan["full_token_neuron_estimate"].values())
    check_budget(0, estimate)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as out:
        out.write(json.dumps(run_plan) + "\n")
        out.flush()
        if not args.run:
            print(f"Plan only: {run_plan['planned_calls']} calls, at most {estimate:.2f} neurons by full input count.")
            return
        account, token = credentials(args.account)
        for model in RATES:
            # Keep the entire reservation after partial completion to cover analytics delay.
            usage = daily_usage(account, token)
            out.write(json.dumps({**usage, "before_model": model}) + "\n")
            out.flush()
            check_budget(usage["used_neurons"], estimate)
            print(f"Daily usage {usage['used_neurons']:.2f}; reserving {estimate:.2f} plus {HEADROOM} neurons.", flush=True)
            collect({**run_plan, "requests": [r for r in run_plan["requests"] if r["model"] == model]},
                    out, account, token)
        out.write(json.dumps({"type": "complete", "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()}) + "\n")
        out.flush()
        final_usage(out, account, token)


if __name__ == "__main__":
    main()
