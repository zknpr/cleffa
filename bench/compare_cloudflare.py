"""Compare a complete hosted corpus journal with FP32 and optional local responses.

  .venv/bin/python -B bench/compare_cloudflare.py HOSTED.jsonl OUT.json [--local-dir DIR]

Local files are DIR/{clef,clef-flash}.jsonl, in corpus order. Agreement with the
hosted service is not labeled accuracy. A reported token count match does not
prove token identity; differing counts are summarized separately. Every answer
must carry the type the planned question declares; an answer of another type
whose probability keys happen to match is an error, never agreement. Each plan
row's input-ID hash must equal the FP32 oracle encoding's: equal counts and
spans do not prove equal tokens. A plan collected before the hash existed is
accepted only with --allow-unhashed-plan, and the summary then records that
token identity was not verified. Each local response row names its request
(`id`) and carries the planned request's hash (`request_sha256`); position in
the file is not identity. Rows captured before those fields existed are
accepted only with the same flag, recorded as unbound in the summary.
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

from cloudflare_checkout import response_info
from cloudflare_corpus import ROOT, RATES, hosted_payload


def distribution(answer: dict, option_ids: list[str], planned_type: str) -> dict[str, float]:
    if answer.get("type") != planned_type:
        raise ValueError("Answer type differs from the planned question")
    if planned_type == "noul":
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


def check_input_ids(row: dict, enc: dict, allow_unhashed: bool) -> bool:
    """True when the plan row's input-ID hash equals the oracle encoding's. A plan without the
    hash is accepted only with allow_unhashed, and the summary then says token identity was not
    verified; a hash that differs is an error, since the local responses and the FP32 logits
    would describe different token sequences."""
    digest = hashlib.sha256(json.dumps(enc["input_ids"]).encode()).hexdigest()
    if "input_ids_sha256" not in row:
        if not allow_unhashed:
            raise ValueError("Plan has no input-ID hash; regenerate it or pass --allow-unhashed-plan")
        return False
    if row["input_ids_sha256"] != digest:
        raise ValueError("Planned token IDs differ from the oracle encoding")
    return True


def check_local_binding(local_row: dict, row: dict, allow_unhashed: bool) -> bool:
    """True when the local response names the planned request and carries its hash. A row
    without the hash is accepted only with allow_unhashed, and the summary then says the local
    responses were not bound; a wrong id or hash is an error, since the local answer would be
    scored against another request's FP32 logits."""
    if "request_sha256" not in local_row:
        if not allow_unhashed:
            raise ValueError("Local response carries no request hash; capture it with id and request_sha256 or pass --allow-unhashed-plan")
        return False
    if local_row.get("id") != row["id"] or local_row["request_sha256"] != row["request_sha256"]:
        raise ValueError("Local response does not match its planned request")
    return True


def score_value(answer: dict, option_ids: list[str]) -> float:
    """A score answer's `score`: a finite number within the option range, or an error. The
    value is recorded as evidence, so an impossible one must not be published."""
    score = answer.get("score")
    if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= len(option_ids) - 1:
        raise ValueError("Invalid score")
    return score


def confidence_value(answer: dict) -> float | None:
    """A choice or score answer's `confidence`: required, a finite number in [0, 1]; a noul
    answer carries none. Hosted and local responses in the corpus journals both follow this.
    Not checked against the probabilities: the local engine reports the maximum probability,
    while the hosted service's confidence differs from it by up to 0.42 on the corpus, so it is
    recorded as evidence, not derived."""
    confidence = answer.get("confidence")
    if answer.get("type") == "noul":
        if confidence is not None:
            raise ValueError("Unexpected confidence on a noul answer")
        return None
    if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("Invalid confidence")
    return confidence


def decision(answer: dict, probs: dict[str, float]) -> str:
    if answer.get("type") == "choice":
        choice = answer["choice"]
        if choice not in probs or probs[choice] < max(probs.values()) - 0.000101:
            raise ValueError("Choice is inconsistent with returned probabilities")
        return choice
    return max(probs, key=probs.get)


def hosted_snapshot(path: Path) -> tuple[dict, dict, str]:
    """The plan, the calls and the SHA-256 of one read of the journal, so the digest the summary
    publishes is of the bytes that produced its calls and metrics, not of whatever the path
    holds by the time it is hashed."""
    data = path.read_bytes()
    plan, calls = load_hosted_bytes(data)
    return plan, calls, hashlib.sha256(data).hexdigest()


def load_hosted(path: Path) -> tuple[dict, dict]:
    return load_hosted_bytes(path.read_bytes())


def load_hosted_bytes(data: bytes) -> tuple[dict, dict]:
    records = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
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
        # The answer and token count a record carries are derived from its recorded body at
        # collection time; derive them again and require equality, so an edited or corrupted
        # duplicate cannot drive the comparison while the body says otherwise.
        derived = response_info(record["status"], record["response"], row)
        if any(record.get(k) != derived[k] for k in ("answer", "reported_input_tokens", "full_input_reported")):
            raise ValueError("Call fields differ from what the recorded response derives to")
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


def compare(path: Path, local_dir: Path | None, allow_unhashed: bool = False) -> dict:
    plan, calls, hosted_sha256 = hosted_snapshot(path)
    summary = {"hosted_journal": str(path), "hosted_sha256": hosted_sha256,
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
        ids_verified = local_bound = True
        for i, (row, ref, enc) in enumerate(zip(rows, refs, encoded, strict=True)):
            if ref["id"] != row["id"] or hosted_payload(ref, model) != row["request"]:
                raise ValueError("Oracle and hosted requests differ")
            if enc["questions"] != row["questions"] or len(enc["input_ids"]) != row["full_input_tokens"]:
                raise ValueError("Oracle and hosted plan encodings differ")
            ids_verified = check_input_ids(row, enc, allow_unhashed) and ids_verified
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
            if local:
                local_bound = check_local_binding(local[i], row, allow_unhashed) and local_bound
            for question in row["questions"]:
                qid, options = question["id"], question["option_ids"]
                planned_type = row["request"]["questions"][qid]["type"]
                raw = logits[row["id"] + "/" + qid].astype(np.float64)
                if raw.shape != (len(options),) or not np.isfinite(raw).all():
                    raise ValueError("Invalid oracle logits")
                probs = np.exp(raw - raw.max())
                probs /= probs.sum()
                fp32 = dict(zip(options, map(float, probs), strict=True))
                answers = [s["answer"]["answers"][qid] for s in samples]
                hosted = [distribution(a, options, planned_type) for a in answers]
                comparison = {"id": row["id"], "question": qid, "counts_match": equal_count,
                              "fp32": fp32, "fp32_decision": max(fp32, key=fp32.get),
                              "hosted": hosted[0], "hosted_decision": decision(answers[0], hosted[0]),
                              "hosted_decisions_all_passes": [decision(a, p) for a, p in zip(answers, hosted)],
                              "hosted_max_repeat_delta": max(abs(p[k] - hosted[0][k]) for p in hosted for k in p),
                              "hosted_confidence": confidence_value(answers[0])}
                for a in answers[1:]:
                    confidence_value(a)   # every pass, like the probabilities
                if local:
                    answer = local[i]["answers"][qid]
                    comparison["local"] = distribution(answer, options, planned_type)
                    comparison["local_decision"] = decision(answer, comparison["local"])
                    comparison["local_confidence"] = confidence_value(answer)
                if planned_type == "score":
                    for a in answers:
                        score_value(a, options)   # every pass, like the probabilities
                    comparison["hosted_score"] = answers[0]["score"]
                    comparison["fp32_score"] = sum(int(k) * p for k, p in fp32.items())
                    if local:
                        comparison["local_score"] = score_value(local[i]["answers"][qid], options)
                comparisons.append(comparison)
        pairs = [("hosted", "fp32")]
        if local:
            pairs += [("local", "fp32"), ("local", "hosted")]
        summary["models"][model] = {
            "requests": len(rows), "questions": len(comparisons), "coverage": coverage,
            "input_ids_verified": ids_verified,
            "local_bound": local_bound if local else None,
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
    parser.add_argument("--allow-unhashed-plan", action="store_true",
                        help="accept a plan without input-ID hashes and local rows without request hashes "
                             "(captured before they existed); both are recorded in the summary")
    args = parser.parse_args()
    summary = compare(args.hosted, args.local_dir, args.allow_unhashed_plan)
    with args.out.open("x") as out:
        out.write(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    for model, result in summary["models"].items():
        print(model, json.dumps(result["comparisons"]))


if __name__ == "__main__":
    main()
