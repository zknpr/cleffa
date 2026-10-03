"""Compare SystemOne endpoints (Jev's hosted API, clef-server) on the same requests.

  jev_compare.py collect URL MODEL OUT.jsonl REQUESTS.jsonl [--key-file F] [--passes 3]
      POST every request to URL (.../v1/systemone) with "model" set to MODEL, PASSES times
      back to back on one keep-alive connection, after one unrecorded warm-up request. Writes one
      line per call: pass, id, wall ms, HTTP status, response body. Also times a cheap
      authenticated GET on the same connection (GET /v1/models for an https URL, /health
      otherwise) as a network/front-end baseline.
      The API key comes from --key-file or TYPESAFE_API_KEY. It is sent only in the
      Authorization header and is never printed.
  jev_compare.py compare A.jsonl B.jsonl [C.jsonl ...]
      Per endpoint: median latency per request over the passes, and the median and p95 of those.
      Per pair: decision agreement per question (argmax of the probabilities), total variation
      distance between the probability vectors, and the questions they disagree on.

Wall time is measured at the client, so a remote endpoint's numbers include the network; the
baseline GET shows how much.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import statistics
import sys
import time
import urllib.parse
from pathlib import Path


def connect(url: urllib.parse.SplitResult) -> http.client.HTTPConnection:
    if url.scheme == "https":
        return http.client.HTTPSConnection(url.hostname or "", url.port or 443, timeout=300)
    return http.client.HTTPConnection(url.hostname or "", url.port or 80, timeout=300)


def call(conn: http.client.HTTPConnection, method: str, path: str, headers: dict, body: bytes | None):
    t0 = time.perf_counter()
    conn.request(method, path, body=body, headers=headers)
    r = conn.getresponse()
    data = r.read()
    ms = (time.perf_counter() - t0) * 1e3
    return ms, r.status, data


def collect(args: argparse.Namespace) -> None:
    url = urllib.parse.urlsplit(args.url)
    key = Path(args.key_file).read_text().strip() if args.key_file else os.environ.get("TYPESAFE_API_KEY", "")
    headers = {"Content-Type": "application/json", "Connection": "keep-alive"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    requests = [json.loads(l) for l in open(args.requests)]
    for r in requests:
        r["model"] = args.model
    conn = connect(url)
    base_path = "/v1/models" if url.scheme == "https" else "/health"
    base = [call(conn, "GET", base_path, headers, None) for _ in range(6)][1:]
    if any(status != 200 for _, status, _ in base):
        sys.exit(f"baseline GET {base_path}: HTTP {[s for _, s, _ in base]}")
    call(conn, "POST", url.path, headers, json.dumps({k: v for k, v in requests[0].items() if k != "id"}).encode())
    with open(args.out, "w") as out:
        out.write(json.dumps({"baseline_path": base_path, "baseline_ms": [ms for ms, _, _ in base]}) + "\n")
        for p in range(args.passes):
            for r in requests:
                body = json.dumps({k: v for k, v in r.items() if k != "id"}, ensure_ascii=False).encode()
                ms, status, data = call(conn, "POST", url.path, headers, body)
                try:
                    resp = json.loads(data)
                except ValueError:
                    resp = {"raw": data[:500].decode(errors="replace")}
                out.write(json.dumps({"pass": p, "id": r["id"], "ms": ms, "status": status, "response": resp},
                                     ensure_ascii=False) + "\n")
                print(f"pass {p} {r['id']}: HTTP {status} {ms:8.1f} ms", flush=True)
    conn.close()


def distribution(answer: dict) -> dict[str, float]:
    if isinstance(answer.get("probabilities"), dict):
        return {str(k): float(v) for k, v in answer["probabilities"].items()}
    if "noul" in answer:   # a noul answer may carry only the probability of true
        return {"true": float(answer["noul"]), "false": 1 - float(answer["noul"])}
    return {}


def load(path: str):
    lines = [json.loads(l) for l in open(path)]
    head, calls = lines[0], lines[1:]
    lat: dict[str, list[float]] = {}
    answers: dict[str, dict] = {}
    errors = []
    for c in calls:
        if c["status"] != 200 or "answers" not in c["response"]:
            errors.append((c["pass"], c["id"], c["status"], str(c["response"])[:160]))
            continue
        lat.setdefault(c["id"], []).append(c["ms"])
        answers.setdefault(c["id"], c["response"])   # decisions from the first successful pass
    model = next((c["response"].get("model") for c in calls if c["status"] == 200), "?")
    return head, lat, answers, errors, model


def compare(args: argparse.Namespace) -> None:
    runs = {Path(p).stem: load(p) for p in args.runs}
    # Summaries cover only the requests every endpoint answered, so a rejection cannot skew them.
    ids = set.intersection(*(set(r[1]) for r in runs.values()))
    print(f"{len(ids)} requests answered by every endpoint\n")
    for name, (head, lat, _, errors, model) in runs.items():
        med = {i: statistics.median(v) for i, v in lat.items() if i in ids}
        allm = sorted(med.values())
        p95 = allm[min(len(allm) - 1, int(round(0.95 * (len(allm) - 1))))] if allm else float("nan")
        print(f"== {name} (model {model}): {len(med)} requests answered, {len(errors)} errors")
        print(f"   baseline {head['baseline_path']}: median {statistics.median(head['baseline_ms']):.1f} ms")
        if allm:
            print(f"   latency over requests (median of passes): median {statistics.median(allm):.1f} ms, p95 {p95:.1f} ms")
        for e in errors[:5]:
            print(f"   error: pass {e[0]} {e[1]} HTTP {e[2]}: {e[3]}")
    names = list(runs)
    print("\nper-request median latency (ms):")
    print("  id     " + "".join(f"{n:>16s}" for n in names))
    for i in sorted(ids):
        print(f"  {i}   " + "".join(f"{statistics.median(runs[n][1][i]):16.1f}" for n in names))
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            A, B = runs[names[a]][2], runs[names[b]][2]
            n = agree = 0
            tvs, diff = [], []
            for i in sorted(set(A) & set(B)):
                for q, qa in A[i]["answers"].items():
                    qb = B[i]["answers"].get(q)
                    if qb is None:
                        continue
                    pa, pb = distribution(qa), distribution(qb)
                    if not pa or not pb:
                        continue
                    da, db = max(pa, key=lambda k: pa[k]), max(pb, key=lambda k: pb[k])
                    n += 1
                    agree += da == db
                    tvs.append(0.5 * sum(abs(pa.get(k, 0) - pb.get(k, 0)) for k in set(pa) | set(pb)))
                    if da != db:
                        diff.append(f"{i}/{q}: {names[a]} {da} ({pa[da]:.2f}) vs {names[b]} {db} ({pb[db]:.2f})")
            if n:
                print(f"\n{names[a]} vs {names[b]}: decisions agree on {agree}/{n} questions; "
                      f"probability TV distance median {statistics.median(tvs):.3f}, max {max(tvs):.3f}")
            for d in diff:
                print(f"  differs: {d}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("url")
    c.add_argument("model")
    c.add_argument("out")
    c.add_argument("requests")
    c.add_argument("--key-file")
    c.add_argument("--passes", type=int, default=3)
    m = sub.add_parser("compare")
    m.add_argument("runs", nargs="+")
    args = ap.parse_args()
    collect(args) if args.cmd == "collect" else compare(args)


if __name__ == "__main__":
    main()
