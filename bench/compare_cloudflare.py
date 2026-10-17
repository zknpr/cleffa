"""Compare a complete hosted corpus journal with FP32 and optional local responses.

  .venv/bin/python -B bench/compare_cloudflare.py HOSTED.jsonl OUT.json [--local-dir DIR]

For the five-arm evaluation, add --evaluation-plan PLAN --precision-bands BANDS,
one --hosted-extra per additional model journal, and --native-capture / --stock-capture
for each capture or continuation. Each selected record must bind the final payload.

Local files are DIR/{clef,clef-flash}.jsonl, in corpus order. Agreement with the
hosted service is not labeled accuracy. A reported token count match does not
prove token identity; differing counts are summarized separately. Every answer
must carry the type the planned question declares; an answer of another type
whose probability keys happen to match is an error, never agreement. Each plan
row's input-ID hash must equal the FP32 oracle encoding's: equal counts and
spans do not prove equal tokens. A plan collected before the hash existed is
accepted only with --allow-unhashed-plan, and the summary then records that
token identity was not verified. The oracle's recorded dtype must be float32;
a directory that records none is accepted only with the same flag and reported
as unverified. Each local response row names its request
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
from safetensors.numpy import load as load_safetensors

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


def jsonl_rows(data: bytes) -> list[dict]:
    return [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]


def local_snapshot(path: Path) -> tuple[list[dict], str]:
    """The local responses and the SHA-256 of the one read they came from."""
    data = path.read_bytes()
    return jsonl_rows(data), hashlib.sha256(data).hexdigest()


def oracle_dtype(golden: Path, allow_unhashed: bool) -> str | None:
    """The oracle's recorded dtype, which must be float32: a `*-f32` directory is FP32 by name
    only. ref/oracle.py records it in latency.json; ref/oracle_f32_stream.py records nothing,
    and neither records the snapshot revision, so a missing record is accepted only with
    allow_unhashed and reported as unverified. Binding to the revision needs the oracles to
    write a manifest, which the existing golden directories do not have."""
    path = golden / "latency.json"
    if not path.exists():
        if not allow_unhashed:
            raise ValueError(f"{golden} records no oracle dtype; regenerate it or pass --allow-unhashed-plan")
        return None
    dtype = json.loads(path.read_bytes()).get("dtype")
    if dtype != "float32":
        raise ValueError(f"{golden} was generated with dtype {dtype!r}, not float32")
    return dtype


def oracle_snapshot(golden: Path) -> tuple[list[dict], list[dict], dict, str]:
    """The oracle requests, encodings and logits, the logits parsed from the same bytes that are
    hashed, so the published digest is of the tensors the metrics used."""
    refs = jsonl_rows((golden / "requests.jsonl").read_bytes())
    encoded = jsonl_rows((golden / "encoded.jsonl").read_bytes())
    data = (golden / "logits.safetensors").read_bytes()
    return refs, encoded, load_safetensors(data), hashlib.sha256(data).hexdigest()


def load_hosted_bytes(data: bytes, *, allow_validation_failures: bool = False) -> tuple[dict, dict]:
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
        row = rows[key]
        if record["request_sha256"] != row["request_sha256"]:
            raise ValueError("Call does not match planned request")
        if record["warmup"] != (record["pass"] == -1):
            raise ValueError("Invalid warmup marker")
        if record.get("error") or record.get("status") != 200:
            if (not allow_validation_failures or record.get("status") not in (400, 413, 422)
                    or not record.get("error") or "answer" in record):
                raise ValueError("Hosted call failed")
            calls[index] = record
            continue
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


def wilson_interval(successes: int, total: int) -> list[float] | None:
    if not 0 <= successes <= total:
        raise ValueError('invalid agreement count')
    if not total:
        return None
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z*z/total
    center = (p + z*z/(2*total)) / denominator
    radius = z*math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return [max(0., center-radius), min(1., center+radius)]


def metrics(rows: list[dict], left: str, right: str) -> dict:
    errors = [max(abs(r[left][k] - r[right][k]) for k in r[left]) for r in rows]
    agreement = sum(r[left + '_decision'] == r[right + '_decision'] for r in rows)
    return {"questions": len(rows), "decision_agreement": sum(
        r[left + "_decision"] == r[right + "_decision"] for r in rows),
        "decision_agreement_fraction": agreement / len(rows) if rows else None,
        "wilson_95": wilson_interval(agreement, len(rows)),
        "interval_unit": "question, one observation per hosted pass; nominal CI, questions may share a request",
        "mean_max_probability_delta": statistics.mean(errors) if errors else None,
        "p99_max_probability_delta": float(np.quantile(errors, .99, method='linear')) if errors else None,
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
        refs, encoded, logits, logits_sha256 = oracle_snapshot(golden)
        dtype = oracle_dtype(golden, allow_unhashed)
        if len(refs) != len(rows) or len(encoded) != len(rows):
            raise ValueError("Oracle corpus size differs")
        local = local_sha256 = None
        if local_dir:
            local, local_sha256 = local_snapshot(local_dir / (model + ".jsonl"))
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
            "input_ids_verified": ids_verified, "oracle_dtype_verified": dtype == "float32",
            "local_bound": local_bound if local else None,
            "unstable_responses": unstable,
            "comparisons": {a + "_vs_" + b: {
                "all": metrics(comparisons, a, b),
                "matching_reported_token_counts": metrics([r for r in comparisons if r["counts_match"]], a, b)
            } for a, b in pairs}, "questions_detail": comparisons,
            "fp32_logits_sha256": logits_sha256, "local_response_sha256": local_sha256}
    return summary


def capture_records(paths: list[Path], planned: dict, arm: str) -> tuple[dict, list[dict]]:
    """Select only exact-body captures. Failed stock records supersede stale successes.

    Continuations may refresh a transport representation. Older bodies remain in the
    provenance, but cannot silently contribute a response to the new request.
    """
    selected, sources, binaries = {}, [], None
    for path in paths:
        data = path.read_bytes()
        report = json.loads(data)
        if arm == "C":
            if report.get("strict") is not True or report.get("arm") != "strict native HTTP":
                raise ValueError("C requires strict HTTP captures")
            if binaries is not None and report["hashes"] != binaries:
                raise ValueError("Native captures use different binaries")
            binaries = report["hashes"]
        elif report.get("profile") != "stock model-card loader and Transformers defaults":
            raise ValueError("B requires the stock control")
        source = {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(),
                  "model": report["model"], "complete": report["complete"],
                  "phase": report.get("phase"), "error": report.get("error"),
                  "stale_bodies": [], "hashes": report.get("hashes")}
        for record in report["records"]:
            key = report["model"], record["id"]
            if key not in planned:
                raise ValueError("Unexpected capture request")
            if record["request_sha256"] != planned[key]["request_sha256"]:
                source["stale_bodies"].append(record["id"])
                continue
            previous = selected.get(key)
            if previous and any(previous.get(k) != record.get(k)
                                for k in ("response", "input_ids_sha256", "error")):
                raise ValueError("Conflicting captures for one request body")
            selected[key] = {**record, "source": str(path)}
        sources.append(source)
    return selected, sources


def load_control_audit(path: Path, bands_hash: str, stock_hashes: dict, models: set) -> dict:
    """A later correctness finding can revoke a control, never refit its frozen band."""
    data = path.read_bytes()
    audit = json.loads(data)
    if audit.get("complete") is not True or audit.get("bands_sha256") != bands_hash:
        raise ValueError("Control audit is incomplete or bound to different bands")
    if audit.get("stock_sources") != stock_hashes:
        raise ValueError("Control audit is bound to different stock sources")
    invalid = audit.get("invalid_models")
    if (not isinstance(invalid, dict) or not invalid or not invalid.keys() <= models or
            any(not isinstance(reason, str) or not reason.strip() for reason in invalid.values())):
        raise ValueError("Control audit must name invalid models and reasons")
    evidence = audit.get("evidence")
    if not isinstance(evidence, dict) or not evidence:
        raise ValueError("Control audit requires bound evidence")
    for source, expected in evidence.items():
        if hashlib.sha256(Path(source).read_bytes()).hexdigest() != expected:
            raise ValueError("Control audit evidence changed")
    return {**audit, "path": str(path), "sha256": hashlib.sha256(data).hexdigest()}


def classify_disagreement(row: dict, hosted: str, band: dict) -> dict:
    """Precision eligibility requires the frozen control and verified local input identity.

    A small margin is consistent with precision noise, not proof of its cause. In
    particular, missing stock evidence must never explain away a hosted disagreement.
    """
    included = {(r["id"], r["question"]) for r in band["included"]}
    threshold = band["threshold"]
    invalidated = band.get("invalidated_reason")
    eligible = (not invalidated and threshold is not None and "B" in row and row["B_F_input_ids_match"]
                and row["C_F_input_ids_match"] and row[hosted + "_counts_match"]
                and (row["id"], row["question"]) in included)
    if not row[hosted + "_counts_match"]:
        category = "input count differs; token identity unproven"
    elif invalidated:
        category = "unexplained; stock control failed correctness audit"
    elif eligible and row["F_top2_probability_margin"] <= threshold:
        category = "inside preregistered precision band"
    elif "B" not in row:
        category = "unexplained; stock control unavailable"
    elif not row["B_F_input_ids_match"] or not row["C_F_input_ids_match"]:
        category = "local input identity differs; investigate preprocessing"
    elif row["B_decision"] == row[hosted + "_decision"]:
        category = "B agrees with hosted; investigate native path"
    elif row["B_decision"] == row["C_decision"]:
        category = "B agrees with native; hosted and HF deployment differ"
    else:
        category = "all three decisions differ; unexplained"
    return {"id": row["id"], "question": row["question"], "modality": row["modality"],
            "hosted_pass": hosted, "category": category,
            "decisions": {a: row.get(a + "_decision") for a in ("C", hosted, "B", "F")},
            "max_probability_delta": max(abs(row["C"][k] - row[hosted][k]) for k in row["C"]),
            "fp32_margin": row["F_top2_probability_margin"], "precision_band": threshold,
            "precision_eligible": eligible, "hosted_input_identity": "unproven",
            "stock_control_invalidated": invalidated,
            "input_evidence": {k: row[k] for k in ("C_F_input_ids_match", "B_F_input_ids_match",
                                                   hosted + "_counts_match")}}


def compare_evaluation(plan_path: Path, bands_path: Path, hosted_paths: list[Path],
                       native_paths: list[Path], stock_paths: list[Path],
                       control_audit_path: Path | None = None) -> dict:
    """Five-arm evaluation; failures remain coverage failures, never inferred answers."""
    raw_plan, raw_bands = plan_path.read_bytes(), bands_path.read_bytes()
    plan, bands = json.loads(raw_plan), json.loads(raw_bands)
    if bands["plan_sha256"] != hashlib.sha256(raw_plan).hexdigest():
        raise ValueError("Precision bands are bound to a different plan")
    if bands["protocol_sha256"] != plan["protocol_sha256"] or plan["passes"] != 2:
        raise ValueError("Evaluation protocol differs")
    planned = {(r["model"], r["id"]): r for r in plan["requests"]}
    if not planned or len(planned) != len(plan["requests"]):
        raise ValueError("Empty or duplicate evaluation plan")
    for row in planned.values():
        if hashlib.sha256(json.dumps(row["request"]).encode()).hexdigest() != row["request_sha256"]:
            raise ValueError("Evaluation request hash differs from payload")
    captures, sources = {}, {}
    for arm, paths in (("C", native_paths), ("B", stock_paths)):
        captures[arm], sources[arm] = capture_records(paths, planned, arm)
    stock_hashes = {Path(s["path"]).name: s["sha256"] for s in sources["B"]}
    if stock_hashes != bands["stock_sources"]:
        raise ValueError("Stock captures differ from the frozen precision control")
    audit = (load_control_audit(control_audit_path, hashlib.sha256(raw_bands).hexdigest(),
                               stock_hashes, set(bands["models"])) if control_audit_path else None)
    if audit:
        for model, reason in audit["invalid_models"].items():
            bands["models"][model]["invalidated_reason"] = reason
    calls, hosted_sources = {}, []
    for path in hosted_paths:
        data = path.read_bytes()
        hosted_plan, subset = load_hosted_bytes(data, allow_validation_failures=True)
        if hosted_plan["passes"] != 2:
            raise ValueError("Hosted evidence requires two distinct passes")
        if any(planned.get((r["model"], r["id"])) != r for r in hosted_plan["requests"]):
            raise ValueError("Hosted plan differs from evaluation plan")
        if calls.keys() & subset.keys():
            raise ValueError("Duplicate hosted evidence")
        calls.update(subset)
        hosted_sources.append({"path": str(path), "sha256": hashlib.sha256(data).hexdigest()})
    expected = {(m, rid, rep) for m, rid in planned for rep in (0, 1)}
    if {k for k in calls if k[2] >= 0} != expected:
        raise ValueError("Missing hosted evaluation calls")
    result = {"plan_sha256": hashlib.sha256(raw_plan).hexdigest(),
              "bands_sha256": hashlib.sha256(raw_bands).hexdigest(), "bands_frozen_at": bands["frozen_at"],
              "control_audit": audit,
              "sources": {**sources, "H": hosted_sources, "F": {}}, "models": {},
              "note": "Agreement, not labeled accuracy. Historical FP32 is attribution only. "
                      "H1 and H2 are separate observations; matching counts do not prove hosted input identity."}
    cache = {}
    for model in dict.fromkeys(m for m, _ in planned):
        details, coverage, disagreements = [], [], []
        for (m, rid), row in planned.items():
            if m != model:
                continue
            golden = ROOT / row["golden"]
            if golden not in cache:
                refs, encs, logits, digest = oracle_snapshot(golden)
                cache[golden] = {e["id"]: e for e in encs}, logits
                result["sources"]["F"][str(golden)] = {"logits_sha256": digest,
                    "dtype": oracle_dtype(golden, allow_unhashed=True),
                    "encoded_sha256": hashlib.sha256((golden / "encoded.jsonl").read_bytes()).hexdigest(),
                    "historical": True}
            encs, logits = cache[golden]
            enc = encs[rid]
            input_sha = hashlib.sha256(json.dumps(enc["input_ids"]).encode()).hexdigest()
            if row["questions"] != enc["questions"]:
                raise ValueError("FP32 question encoding differs from plan")
            bound_fp32 = row.get("fp32_input_ids_sha256", row["input_ids_sha256"])
            if bound_fp32 != input_sha:
                raise ValueError("FP32 input IDs differ from registered plan")
            record = {a: captures[a].get((model, rid)) for a in ("C", "B")}
            record.update({f"H{rep+1}": calls[model, rid, rep] for rep in (0, 1)})
            answers, cov = {}, {"id": rid, "modality": row["modality"],
                               "request_sha256": row["request_sha256"], "full_tokens": row["full_input_tokens"]}
            for arm, captured in record.items():
                if not captured or captured.get("error"):
                    cov[arm] = {"available": False, "error": captured.get("error") if captured else "no capture",
                                "status": captured.get("status") if captured else None}
                    continue
                response = captured["answer"] if arm.startswith("H") else captured["response"]
                if (response.get("model") != model or
                        set(response["answers"]) != set(row["request"]["questions"])):
                    raise ValueError("Capture response model/questions differ")
                answers[arm] = response["answers"]
                tokens = response.get("usage", {}).get("input_tokens")
                cov[arm] = {"available": True, "input_tokens": tokens,
                            "counts_match": tokens == row["full_input_tokens"]}
                if arm in ("C", "B"):
                    if type(tokens) is not int or tokens != captured["tokens"]:
                        raise ValueError("Capture tokens differ from response")
                    if arm == "C" and captured["input_ids_sha256"] != row["input_ids_sha256"]:
                        raise ValueError("Native capture tokens differ from plan")
                    cov[arm]["F_input_ids_match"] = captured["input_ids_sha256"] == input_sha
            coverage.append(cov)
            for q in row["questions"]:
                qid, options = q["id"], q["option_ids"]
                raw = logits[rid + "/" + qid].astype(np.float64)
                if raw.shape != (len(options),) or not np.isfinite(raw).all():
                    raise ValueError("Invalid FP32 attribution logits")
                p = np.exp(raw - raw.max()); p /= p.sum()
                values = {"id": rid, "question": qid, "modality": row["modality"],
                          "F": dict(zip(options, map(float, p), strict=True)),
                          "F_decision": options[int(np.argmax(p))],
                          "F_top2_probability_margin": float(np.sort(p)[-1] - np.sort(p)[-2])}
                for arm in ("C", "B"):
                    values[arm + "_F_input_ids_match"] = cov[arm].get("F_input_ids_match", False)
                for arm in ("H1", "H2"):
                    values[arm + "_counts_match"] = cov[arm].get("counts_match", False)
                for arm, group in answers.items():
                    answer = group[qid]
                    values[arm] = distribution(answer, options, row["request"]["questions"][qid]["type"])
                    values[arm + "_decision"] = decision(answer, values[arm])
                    values[arm + "_confidence"] = confidence_value(answer)
                    if answer["type"] == "score":
                        values[arm + "_score"] = score_value(answer, options)
                for hosted in ("H1", "H2"):
                    if "C" in values and hosted in values and values["C_decision"] != values[hosted + "_decision"]:
                        disagreements.append(classify_disagreement(values, hosted, bands["models"][model]))
                details.append(values)
        # Audit the pre-registration against its pinned stock evidence. Do not let a
        # corrupted threshold or omitted outlier retroactively explain a disagreement.
        band = bands["models"][model]
        measured = {(r["id"], r["question"]): r for r in details
                    if "B" in r and r["B_F_input_ids_match"]}
        registered = {(r["id"], r["question"]): r for r in band["included"]}
        if len(registered) != len(band["included"]) or measured.keys() != registered.keys():
            raise ValueError("Precision band coverage differs from stock evidence")
        deltas = []
        for key, value in measured.items():
            delta = max(abs(value["B"][k] - value["F"][k]) for k in value["F"])
            if (abs(delta - registered[key]["max_abs_dp"]) > 1e-12 or
                    abs(value["F_top2_probability_margin"] -
                        registered[key]["fp32_top2_probability_margin"]) > 1e-12):
                raise ValueError("Precision band values differ from stock evidence")
            deltas.append(delta)
        threshold = float(np.quantile(deltas, .95, method="linear")) if deltas else None
        if (threshold is None) != (band["threshold"] is None) or (
                threshold is not None and abs(threshold - band["threshold"]) > 1e-12):
            raise ValueError("Precision threshold differs from preregistered p95 rule")
        pairs = [("C", "H1"), ("C", "H2"), ("H1", "H2"), ("B", "H1"), ("B", "H2"),
                 ("C", "B"), ("C", "F"), ("B", "F"), ("H1", "F"), ("H2", "F")]
        comparisons = {}
        for a, b in pairs:
            available = [r for r in details if a in r and b in r]
            # Preserve the primary identical-body comparison, alongside a stricter subset
            # based on the input evidence actually exposed by each implementation.
            equal = [r for r in available if all(r[arm + ("_counts_match" if arm.startswith("H")
                         else "_F_input_ids_match")] for arm in (a, b) if arm != "F")]
            comparisons[a + "_vs_" + b] = {"all_valid": metrics(available, a, b),
                "matching_input_evidence": metrics(equal, a, b),
                "by_modality": {mode: metrics([r for r in available if r["modality"] == mode], a, b)
                                for mode in dict.fromkeys(r["modality"] for r in details)}}
        result["models"][model] = {"requests": len(coverage), "questions": len(details),
            "coverage": coverage, "comparisons": comparisons, "disagreements": disagreements,
            "questions_detail": details, "precision_band": bands["models"][model]["threshold"],
            "stock_control_invalidated": bands["models"][model].get("invalidated_reason")}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hosted", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--local-dir", type=Path)
    parser.add_argument("--evaluation-plan", type=Path)
    parser.add_argument("--precision-bands", type=Path)
    parser.add_argument("--control-audit", type=Path,
                        help="source-bound correctness finding revoking a frozen stock control")
    parser.add_argument("--hosted-extra", type=Path, action="append", default=[])
    parser.add_argument("--native-capture", type=Path, action="append", default=[])
    parser.add_argument("--stock-capture", type=Path, action="append", default=[])
    parser.add_argument("--allow-unhashed-plan", action="store_true",
                        help="accept a plan without input-ID hashes and local rows without request hashes "
                             "(captured before they existed); both are recorded in the summary")
    args = parser.parse_args()
    if args.evaluation_plan:
        if not args.precision_bands or args.local_dir or args.allow_unhashed_plan:
            parser.error("evaluation requires --precision-bands and bound capture files")
        summary = compare_evaluation(args.evaluation_plan, args.precision_bands,
                                    [args.hosted, *args.hosted_extra], args.native_capture, args.stock_capture,
                                    args.control_audit)
    else:
        if args.control_audit:
            parser.error("--control-audit requires --evaluation-plan")
        summary = compare(args.hosted, args.local_dir, args.allow_unhashed_plan)
    with args.out.open("x") as out:
        out.write(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    for model, result in summary["models"].items():
        print(model, json.dumps(result["comparisons"]))


if __name__ == "__main__":
    main()
