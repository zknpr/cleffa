"""Compare a complete hosted corpus journal with FP32 and optional local responses.

  .venv/bin/python -B bench/compare_cloudflare.py HOSTED.jsonl OUT.json [--local-dir DIR]

Local files are DIR/{clef,clef-flash}.jsonl, in corpus order. Agreement with the
hosted service is not labeled accuracy. A reported token count match does not
prove token identity; differing counts are summarized separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np
from safetensors.numpy import load_file

from cloudflare_corpus import ROOT, RATES, hosted_payload


def distribution(answer: dict, option_ids: list[str]) -> dict[str, float]:
    if answer.get("type") == "noul":
        p = answer["noul"]
        if type(p) not in (int, float):
            raise ValueError("Invalid noul probability")
        result = {"true": p, "false": 1 - p}
    else:
        result = answer["probabilities"]
    if set(result) != set(option_ids):
        raise ValueError("Response option IDs differ from the encoded request")
    if any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1
           for p in result.values()):
        raise ValueError("Invalid probability")
    if abs(sum(result.values()) - 1) > len(result) * 0.000051 + 1e-6:
        raise ValueError("Probabilities do not sum to one within response rounding")
    return {k: result[k] for k in option_ids}


def decision(answer: dict, probs: dict[str, float]) -> str:
    if answer.get("type") == "choice":
        choice = answer["choice"]
        if choice not in probs or probs[choice] < max(probs.values()) - 0.000101:
            raise ValueError("Choice is inconsistent with returned probabilities")
        return choice
    return max(probs, key=probs.get)


def load_hosted(path: Path) -> tuple[dict, dict]:
    with path.open() as source:
        records = [json.loads(line) for line in source]
    if not records or records[0].get("type") != "plan":
        raise ValueError("Missing reference plan")
    plan = records[0]
    if sum(r.get("type") == "complete" for r in records) != 1:
        raise ValueError("Incomplete hosted collection")
    rows = {(r["model"], r["id"]): r for r in plan["requests"]}
    if not rows or len(rows) != len(plan["requests"]) or plan["passes"] < 1:
        raise ValueError("Empty or duplicate reference requests")
    for row in rows.values():
        if hashlib.sha256(json.dumps(row["request"]).encode()).hexdigest() != row["request_sha256"]:
            raise ValueError("Planned request hash differs from its payload")
    calls = {}
    for record in records[1:]:
        if record.get("type") != "call":
            continue
        key = record["model"], record["id"]
        index = (*key, record["pass"])
        if key not in rows or index in calls:
            raise ValueError("Unexpected or duplicate hosted call")
        if record.get("error") or record.get("status") != 200:
            raise ValueError("Hosted call failed")
        row = rows[key]
        if record["request_sha256"] != row["request_sha256"]:
            raise ValueError("Call does not match planned request")
        if set(record["answer"]["answers"]) != set(row["request"]["questions"]):
            raise ValueError("Missing or unexpected hosted questions")
        if record["warmup"] != (record["pass"] == -1):
            raise ValueError("Invalid warmup marker")
        calls[index] = record
    expected = {(model, rid, rep) for model, rid in rows for rep in range(plan["passes"])}
    for model in dict.fromkeys(model for model, _ in rows):
        rid = next(rid for m, rid in rows if m == model)
        expected.add((model, rid, -1))
    if set(calls) != expected or len(calls) != plan["planned_calls"]:
        raise ValueError("Missing or unexpected calls/passes")
    return plan, calls


def metrics(rows: list[dict], left: str, right: str) -> dict:
    errors = [max(abs(r[left][k] - r[right][k]) for k in r[left]) for r in rows]
    return {"questions": len(rows), "decision_agreement": sum(
        r[left + "_decision"] == r[right + "_decision"] for r in rows),
        "mean_max_probability_delta": statistics.mean(errors) if errors else None,
        "max_probability_delta": max(errors) if errors else None,
        "disagreements": [r["id"] + "/" + r["question"] for r in rows
                          if r[left + "_decision"] != r[right + "_decision"]]}


def compare(path: Path, local_dir: Path | None) -> dict:
    plan, calls = load_hosted(path)
    summary = {"hosted_journal": str(path), "hosted_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "timestamp": plan["timestamp"], "passes": plan["passes"],
               "note": "Agreement and numerical distance, not labeled accuracy; probabilities rounded by API",
               "models": {}}
    for model in RATES:
        rows = [r for r in plan["requests"] if r["model"] == model]
        golden = ROOT / "golden" / (model + "-f32")
        refs = [json.loads(line) for line in (golden / "requests.jsonl").read_text().splitlines()]
        encoded = [json.loads(line) for line in (golden / "encoded.jsonl").read_text().splitlines()]
        if len(refs) != len(rows) or len(encoded) != len(rows):
            raise ValueError("Oracle corpus size differs")
        logits = load_file(str(golden / "logits.safetensors"))
        local = None
        if local_dir:
            local = [json.loads(line) for line in (local_dir / (model + ".jsonl")).read_text().splitlines()]
            if len(local) != len(rows):
                raise ValueError("Local response count differs")
        comparisons, coverage, unstable = [], [], []
        for i, (row, ref, enc) in enumerate(zip(rows, refs, encoded, strict=True)):
            if ref["id"] != row["id"] or hosted_payload(ref, model) != row["request"]:
                raise ValueError("Oracle and hosted requests differ")
            if enc["questions"] != row["questions"] or len(enc["input_ids"]) != row["full_input_tokens"]:
                raise ValueError("Oracle and hosted plan encodings differ")
            samples = [calls[model, row["id"], rep] for rep in range(plan["passes"])]
            counts = [s["reported_input_tokens"] for s in samples]
            equal_count = all(n == row["full_input_tokens"] for n in counts)
            coverage.append({"id": row["id"], "full_tokens": row["full_input_tokens"],
                             "hosted_tokens": counts, "counts_match": equal_count})
            if any(s["answer"]["answers"] != samples[0]["answer"]["answers"] for s in samples[1:]):
                unstable.append(row["id"])
            if local and (local[i]["model"] != model or
                          set(local[i]["answers"]) != set(row["request"]["questions"]) or
                          local[i]["usage"]["input_tokens"] != row["full_input_tokens"]):
                raise ValueError("Local response has incomplete questions or input")
            for question in row["questions"]:
                qid, options = question["id"], question["option_ids"]
                raw = logits[row["id"] + "/" + qid].astype(np.float64)
                if raw.shape != (len(options),) or not np.isfinite(raw).all():
                    raise ValueError("Invalid oracle logits")
                probs = np.exp(raw - raw.max())
                probs /= probs.sum()
                fp32 = dict(zip(options, map(float, probs), strict=True))
                answers = [s["answer"]["answers"][qid] for s in samples]
                hosted = [distribution(a, options) for a in answers]
                comparison = {"id": row["id"], "question": qid, "counts_match": equal_count,
                              "fp32": fp32, "fp32_decision": max(fp32, key=fp32.get),
                              "hosted": hosted[0], "hosted_decision": decision(answers[0], hosted[0]),
                              "hosted_decisions_all_passes": [decision(a, p) for a, p in zip(answers, hosted)],
                              "hosted_max_repeat_delta": max(abs(p[k] - hosted[0][k]) for p in hosted for k in p),
                              "hosted_confidence": answers[0].get("confidence")}
                if local:
                    answer = local[i]["answers"][qid]
                    comparison["local"] = distribution(answer, options)
                    comparison["local_decision"] = decision(answer, comparison["local"])
                    comparison["local_confidence"] = answer.get("confidence")
                if answers[0]["type"] == "score":
                    comparison["hosted_score"] = answers[0]["score"]
                    comparison["fp32_score"] = sum(int(k) * p for k, p in fp32.items())
                    if local:
                        comparison["local_score"] = local[i]["answers"][qid]["score"]
                comparisons.append(comparison)
        pairs = [("hosted", "fp32")]
        if local:
            pairs += [("local", "fp32"), ("local", "hosted")]
        summary["models"][model] = {
            "requests": len(rows), "questions": len(comparisons), "coverage": coverage,
            "unstable_responses": unstable,
            "comparisons": {a + "_vs_" + b: {
                "all": metrics(comparisons, a, b),
                "matching_reported_token_counts": metrics([r for r in comparisons if r["counts_match"]], a, b)
            } for a, b in pairs}, "questions_detail": comparisons,
            "fp32_logits_sha256": hashlib.sha256((golden / "logits.safetensors").read_bytes()).hexdigest(),
            "local_response_sha256": hashlib.sha256((local_dir / (model + ".jsonl")).read_bytes()).hexdigest()
            if local_dir else None}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hosted", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--local-dir", type=Path)
    args = parser.parse_args()
    summary = compare(args.hosted, args.local_dir)
    with args.out.open("x") as out:
        out.write(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    for model, result in summary["models"].items():
        print(model, json.dumps(result["comparisons"]))


if __name__ == "__main__":
    main()
