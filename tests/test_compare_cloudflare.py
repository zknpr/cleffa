"""Incomplete or malformed hosted evidence must not improve apparent agreement."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from compare_cloudflare import (check_input_ids, check_local_binding, confidence_value, decision, distribution,
                                hosted_snapshot, load_hosted, local_snapshot, oracle_dtype, oracle_snapshot, score_value,
                                metrics, wilson_interval, capture_records, classify_disagreement,
                                compare_evaluation)


class ComparisonTests(unittest.TestCase):
    def test_invalidated_stock_control_cannot_explain_hosted_disagreement(self):
        row = dict(id="r", question="q", modality="text", F_top2_probability_margin=.01,
                   C={"a": .51, "b": .49}, H1={"a": .49, "b": .51},
                   B={"a": .48, "b": .52}, C_decision="a", H1_decision="b",
                   B_decision="b", F_decision="a", C_F_input_ids_match=True,
                   B_F_input_ids_match=True, H1_counts_match=True)
        band = dict(threshold=.1, included=[dict(id="r", question="q")],
                    invalidated_reason="causal-mask leakage reproduced")
        got = classify_disagreement(row, "H1", band)
        self.assertFalse(got["precision_eligible"])
        self.assertEqual(got["category"], "unexplained; stock control failed correctness audit")
        self.assertEqual(got["stock_control_invalidated"], band["invalidated_reason"])
        self.assertEqual(got["precision_band"], .1)  # Preserve the frozen historical value.

    def test_control_audit_requires_bound_evidence_and_only_revokes_controls(self):
        from compare_cloudflare import load_control_audit
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            evidence = root / "probe.json"
            evidence.write_text('{"causal_backend_valid": false}')
            audit = dict(complete=True, bands_sha256="bands", stock_sources={"stock.json": "sha"},
                         invalid_models={"clef-flash": "mask leakage"},
                         evidence={str(evidence): hashlib.sha256(evidence.read_bytes()).hexdigest()})
            path = root / "audit.json"
            path.write_text(json.dumps(audit))
            got = load_control_audit(path, "bands", audit["stock_sources"], {"clef-flash"})
            self.assertEqual(got["invalid_models"], audit["invalid_models"])
            with self.assertRaisesRegex(ValueError, "bands"):
                load_control_audit(path, "other", audit["stock_sources"], {"clef-flash"})
            with self.assertRaisesRegex(ValueError, "stock"):
                load_control_audit(path, "bands", {}, {"clef-flash"})
            with self.assertRaisesRegex(ValueError, "models"):
                load_control_audit(path, "bands", audit["stock_sources"], {"clef"})
            evidence.write_text("changed")
            with self.assertRaisesRegex(ValueError, "evidence"):
                load_control_audit(path, "bands", audit["stock_sources"], {"clef-flash"})

    def test_five_arm_evaluation_preserves_failures_and_audits_frozen_band(self):
        import numpy as np
        from safetensors.numpy import save
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            payload = {"questions": {"q": {"type": "noul"}}}
            body_hash = hashlib.sha256(json.dumps(payload).encode()).hexdigest()
            encoding = {"id": "r", "input_ids": [1, 2], "questions": [
                {"id": "q", "type": 0, "option_ids": ["true", "false"]}]}
            token_hash = hashlib.sha256(json.dumps(encoding["input_ids"]).encode()).hexdigest()
            row = dict(model="clef", id="r", modality="text", request=payload,
                       request_sha256=body_hash, input_ids_sha256=token_hash, full_input_tokens=2,
                       questions=encoding["questions"], golden=str(root))
            plan = dict(type="plan", requests=[row], passes=2, planned_calls=3, protocol_sha256="registered")
            def write(name, value):
                path = root / name; path.write_text(json.dumps(value)); return path
            plan_path = write("plan.json", plan)
            (root / "requests.jsonl").write_text(json.dumps({"id": "r", **payload}) + "\n")
            (root / "encoded.jsonl").write_text(json.dumps(encoding) + "\n")
            raw = np.array([2., 0.], dtype=np.float32)
            (root / "logits.safetensors").write_bytes(save({"r/q": raw}))
            write("latency.json", {"dtype": "float32"})
            def response(p):
                return {"model": "clef", "answers": {"q": {"type": "noul", "noul": p}},
                        "usage": {"input_tokens": 2}}
            common = dict(id="r", request_sha256=body_hash, input_ids_sha256=token_hash, tokens=2)
            c_path = write("native.json", {"model": "clef", "complete": True,
                "arm": "strict native HTTP", "strict": True, "hashes": {"clef-server": "binary"},
                "records": [{**common, "response": response(.9)}]})
            b_path = write("stock.json", {"model": "clef", "complete": True,
                "profile": "stock model-card loader and Transformers defaults",
                "records": [{**common, "response": response(.7)}]})
            prob = float(np.exp(2) / (np.exp(2) + 1))
            band = dict(threshold=prob-.7, included=[dict(id="r", question="q",
                        max_abs_dp=prob-.7, fp32_top2_probability_margin=2*prob-1)])
            bands = dict(plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
                         protocol_sha256="registered", frozen_at="before scoring",
                         stock_sources={"stock.json": hashlib.sha256(b_path.read_bytes()).hexdigest()},
                         models={"clef": band})
            bands_path = write("bands.json", bands)
            calls = [dict(type="call", model="clef", id="r", request_sha256=body_hash,
                          status=200, warmup=p == -1, **{"pass": p}, response=response(.1),
                          answer=response(.1), reported_input_tokens=2, full_input_reported=True)
                     for p in (-1, 0)]
            calls.append(dict(type="call", model="clef", id="r", request_sha256=body_hash,
                              status=422, warmup=False, **{"pass": 1}, error="HTTP 422",
                              response={"success": False}))
            hosted = root / "hosted.jsonl"
            def save_hosted():
                hosted.write_text("".join(json.dumps(r) + "\n" for r in [plan, *calls, {"type": "complete"}]))
            save_hosted()
            def compare():
                return compare_evaluation(plan_path, bands_path, [hosted], [c_path], [b_path])["models"]["clef"]
            got = compare()
            self.assertEqual(got["comparisons"]["C_vs_H1"]["all_valid"]["questions"], 1)
            self.assertEqual(got["comparisons"]["C_vs_H1"]["all_valid"]["decision_agreement"], 0)
            self.assertEqual(got["comparisons"]["C_vs_H2"]["all_valid"]["questions"], 0)
            self.assertEqual(got["comparisons"]["H1_vs_H2"]["all_valid"]["questions"], 0)
            self.assertEqual(got["coverage"][0]["H2"]["status"], 422)
            self.assertIn("hosted and HF deployment differ", got["disagreements"][0]["category"])
            evidence = write("probe.json", {"complete": True, "causal_backend_valid": False})
            audit = write("audit.json", dict(complete=True,
                bands_sha256=hashlib.sha256(bands_path.read_bytes()).hexdigest(),
                stock_sources=bands["stock_sources"], invalid_models={"clef": "mask leakage"},
                evidence={str(evidence): hashlib.sha256(evidence.read_bytes()).hexdigest()}))
            revised = compare_evaluation(plan_path, bands_path, [hosted], [c_path], [b_path], audit)
            self.assertEqual(revised["models"]["clef"]["comparisons"], got["comparisons"])
            self.assertEqual(revised["models"]["clef"]["stock_control_invalidated"], "mask leakage")
            self.assertEqual(revised["models"]["clef"]["disagreements"][0]["category"],
                             "unexplained; stock control failed correctness audit")
            self.assertEqual(revised["control_audit"]["sha256"], hashlib.sha256(audit.read_bytes()).hexdigest())
            band["threshold"] = 1.
            write("bands.json", bands)
            with self.assertRaisesRegex(ValueError, "threshold"):
                compare()
            band["threshold"] = prob-.7
            write("bands.json", bands)
            calls[0]["request_sha256"] = "changed"
            save_hosted()
            with self.assertRaisesRegex(ValueError, "planned request"):
                compare()

    def test_precision_classification_never_uses_missing_or_mismatched_controls(self):
        row = dict(id="r", question="q", modality="video", F_top2_probability_margin=.01,
                   C={"a": .51, "b": .49}, H1={"a": .49, "b": .51},
                   C_decision="a", H1_decision="b", F_decision="a",
                   C_F_input_ids_match=True, B_F_input_ids_match=False, H1_counts_match=True)
        band = dict(threshold=.1, included=[dict(id="r", question="q")])
        self.assertIn("stock control unavailable", classify_disagreement(row, "H1", band)["category"])
        row.update(B={"a": .48, "b": .52}, B_decision="b", B_F_input_ids_match=True)
        self.assertIn("inside preregistered", classify_disagreement(row, "H1", band)["category"])
        row["H1_counts_match"] = False
        self.assertIn("input count differs", classify_disagreement(row, "H1", band)["category"])
        self.assertFalse(classify_disagreement(row, "H1", band)["precision_eligible"])

    def test_capture_selection_cannot_reuse_stale_success_for_failed_final_body(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = {("clef", "r"): dict(request_sha256="final")}
            report = dict(model="clef", complete=True, profile="stock model-card loader and Transformers defaults")
            old = root / "old.json"; new = root / "new.json"
            old.write_text(json.dumps({**report, "records": [dict(id="r", request_sha256="old", response={})]}))
            new.write_text(json.dumps({**report, "records": [dict(id="r", request_sha256="final", error="decoder failed")]}))
            records, sources = capture_records([old, new], plan, "B")
            self.assertEqual(records["clef", "r"]["error"], "decoder failed")
            self.assertEqual(sources[0]["stale_bodies"], ["r"])

    def test_agreement_interval_and_probability_tail(self):
        low,high=wilson_interval(5,10)
        self.assertAlmostEqual(low,0.236593090512564,places=12)
        self.assertAlmostEqual(high,0.763406909487436,places=12)
        self.assertIsNone(wilson_interval(0,0))
        rows=[dict(id='r'+str(i),question='q',C={'a':.6,'b':.4},H={'a':p,'b':1-p},
                   C_decision='a',H_decision='a' if p>=.5 else 'b') for i,p in enumerate((.6,.8,.4))]
        got=metrics(rows,'C','H')
        self.assertEqual(got['questions'],3)
        self.assertEqual(got['decision_agreement'],2)
        self.assertAlmostEqual(got['mean_max_probability_delta'],.4/3)
        self.assertAlmostEqual(got['p99_max_probability_delta'],.2)
        self.assertEqual(got['disagreements'],['r2/q'])

    def test_probabilities_require_all_options_and_finite_mass(self):
        for probs in [{"a": 1}, {"a": True, "b": 0}, {"a": float("nan"), "b": 1},
                      {"a": 0.8, "b": 0.8}, {"a": 1.1, "b": -0.1}]:
            with self.subTest(probs=probs), self.assertRaises(ValueError):
                distribution({"type": "choice", "probabilities": probs}, ["a", "b"], "choice")
        self.assertEqual(distribution({"type": "noul", "noul": 0.6}, ["false", "true"], "noul"),
                         {"false": 0.4, "true": 0.6})

    def test_answer_type_must_match_the_planned_question(self):
        # A structurally valid answer of another type, whose probability keys happen to match
        # the options, must not count as agreement: the planned question's type is the oracle.
        probs = {"a": 0.6, "b": 0.4}
        for answer, planned in [({"type": "score", "probabilities": probs}, "choice"),
                                ({"type": "choice", "choice": "a", "probabilities": probs}, "score"),
                                ({"type": "noul", "noul": 0.6}, "choice"),
                                ({"probabilities": probs}, "choice")]:
            with self.subTest(answer=answer, planned=planned), self.assertRaises(ValueError):
                distribution(answer, list(probs), planned)
        self.assertEqual(distribution({"type": "score", "probabilities": probs}, ["a", "b"], "score"), probs)

    def test_score_must_be_a_finite_number_within_the_option_range(self):
        # A score answer's `score` is recorded as evidence; a string, a boolean, NaN or a value
        # outside the option range must be an error, not a published number.
        options = ["0", "1", "2"]
        for score in ["1", True, float("nan"), float("inf"), -0.5, 2.5, 999, None]:
            with self.subTest(score=score), self.assertRaises(ValueError):
                score_value({"type": "score", "score": score}, options)
        self.assertEqual(score_value({"type": "score", "score": 1.25}, options), 1.25)
        self.assertEqual(score_value({"type": "score", "score": 2}, options), 2)

    def test_confidence_must_be_a_finite_number_in_the_unit_interval(self):
        # Choice and score answers carry a confidence that is recorded as evidence; noul answers
        # carry none. It must be a finite number in [0, 1], and not a boolean.
        for conf in ["0.9", True, float("nan"), float("inf"), -0.1, 1.1, 999]:
            with self.subTest(conf=conf), self.assertRaises(ValueError):
                confidence_value({"type": "choice", "confidence": conf})
        with self.assertRaises(ValueError):
            confidence_value({"type": "score"})   # required for choice and score
        with self.assertRaises(ValueError):
            confidence_value({"type": "noul", "noul": 0.6, "confidence": 0.6})   # never for noul
        self.assertEqual(confidence_value({"type": "choice", "confidence": 0.9608}), 0.9608)
        self.assertEqual(confidence_value({"type": "score", "confidence": 1}), 1)
        self.assertIsNone(confidence_value({"type": "noul", "noul": 0.6}))

    def test_planned_token_ids_must_equal_the_oracle_encoding(self):
        # Equal counts and spans do not prove equal tokens; the plan carries a hash of its IDs.
        enc = {"input_ids": [1, 2, 3, 4], "questions": []}
        digest = hashlib.sha256(json.dumps(enc["input_ids"]).encode()).hexdigest()
        self.assertTrue(check_input_ids({"input_ids_sha256": digest}, enc, allow_unhashed=False))
        with self.assertRaises(ValueError):
            check_input_ids({"input_ids_sha256": digest}, {"input_ids": [1, 2, 3, 5]}, allow_unhashed=False)
        with self.assertRaises(ValueError):
            check_input_ids({}, enc, allow_unhashed=False)   # a plan from before the hash existed
        self.assertFalse(check_input_ids({}, enc, allow_unhashed=True))   # accepted, recorded as unverified

    def test_local_responses_bind_to_their_planned_request(self):
        # Position in the file is not identity: a local row names its request and carries the
        # planned request's hash, or it is accepted only as unbound.
        row = {"id": "r007", "request_sha256": "a" * 64}
        self.assertTrue(check_local_binding({"id": "r007", "request_sha256": "a" * 64}, row, allow_unhashed=False))
        for local in [{"id": "r008", "request_sha256": "a" * 64}, {"id": "r007", "request_sha256": "b" * 64},
                      {"request_sha256": "a" * 64}]:
            with self.subTest(local=local), self.assertRaises(ValueError):
                check_local_binding(local, row, allow_unhashed=False)
        with self.assertRaises(ValueError):
            check_local_binding({}, row, allow_unhashed=False)   # captured before the fields existed
        self.assertFalse(check_local_binding({}, row, allow_unhashed=True))

    def test_hosted_digest_is_of_the_bytes_parsed(self):
        # The journal is read once; the digest and the calls come from that one snapshot, so a
        # journal replaced between parsing and hashing cannot lend its digest to other calls.
        payload = {"questions": {"q": {"type": "noul"}}}
        digest = hashlib.sha256(json.dumps(payload).encode()).hexdigest()
        answer = {"answers": {"q": {"type": "noul", "noul": 0.9}}, "usage": {"input_tokens": 4}}
        plan = {"type": "plan", "passes": 1, "planned_calls": 2, "requests": [
            {"model": "clef", "id": "r000", "request": payload, "request_sha256": digest, "full_input_tokens": 4}]}
        calls = [{"type": "call", "model": "clef", "id": "r000", "pass": p, "status": 200, "warmup": p == -1,
                  "request_sha256": digest, "response": {"result": answer, "success": True, "errors": []},
                  "answer": answer, "reported_input_tokens": 4, "full_input_reported": True} for p in [-1, 0]]
        journal = "".join(json.dumps(r) + "\n" for r in [plan, *calls, {"type": "complete"}]).encode()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "hosted.jsonl"
            path.write_bytes(journal)
            with patch.object(Path, "read_bytes", side_effect=[journal, b"replaced"]):
                plan_out, calls_out, sha = hosted_snapshot(path)
            self.assertEqual(len(calls_out), 2)
            self.assertEqual(sha, hashlib.sha256(journal).hexdigest())

    def test_local_and_oracle_digests_are_of_the_bytes_parsed(self):
        # Each input is read once; its rows or tensors and its digest come from that snapshot.
        import numpy as np
        from safetensors.numpy import save
        rows = [{"model": "clef", "answers": {}, "usage": {"input_tokens": 4}}]
        local = ("".join(json.dumps(r) + "\n" for r in rows)).encode()
        tensors = save({"r000/q": np.zeros(2, dtype=np.float32)})
        with tempfile.TemporaryDirectory() as temp:
            golden = Path(temp)
            (golden / "requests.jsonl").write_text('{"id": "r000"}\n')
            (golden / "encoded.jsonl").write_text('{"id": "r000", "input_ids": [1]}\n')
            (golden / "logits.safetensors").write_bytes(tensors)
            (golden / "clef.jsonl").write_bytes(local)
            with patch.object(Path, "read_bytes", side_effect=[local, b"replaced"]):
                parsed, sha = local_snapshot(golden / "clef.jsonl")
            self.assertEqual(parsed, rows)
            self.assertEqual(sha, hashlib.sha256(local).hexdigest())
            refs, encoded, logits, sha = oracle_snapshot(golden)
            self.assertEqual(refs, [{"id": "r000"}])
            self.assertEqual(list(logits), ["r000/q"])
            self.assertEqual(sha, hashlib.sha256(tensors).hexdigest())

    def test_oracle_dtype_is_verified_where_recorded(self):
        # A golden directory is FP32 by name only; ref/oracle.py records its dtype in
        # latency.json, the streamed 27B oracle records nothing. The dtype is verified where it
        # exists, refused when it is not float32, and reported as unverified when absent only
        # with the allow flag.
        with tempfile.TemporaryDirectory() as temp:
            golden = Path(temp)
            (golden / "latency.json").write_text(json.dumps({"dtype": "float32", "device": "mps"}))
            self.assertEqual(oracle_dtype(golden, allow_unhashed=False), "float32")
            (golden / "latency.json").write_text(json.dumps({"dtype": "bfloat16"}))
            with self.assertRaises(ValueError):
                oracle_dtype(golden, allow_unhashed=False)
            (golden / "latency.json").unlink()
            with self.assertRaises(ValueError):
                oracle_dtype(golden, allow_unhashed=False)
            self.assertIsNone(oracle_dtype(golden, allow_unhashed=True))

    def test_choice_uses_explicit_winner_for_rounded_tie(self):
        self.assertEqual(decision({"type": "choice", "choice": "b"}, {"a": 0.5, "b": 0.5}), "b")
        with self.assertRaises(ValueError):
            decision({"type": "choice", "choice": "b"}, {"a": 0.8, "b": 0.2})

    def test_every_planned_pass_required_exactly_once(self):
        payload = {"questions": {"q": {"type": "noul"}}}
        digest = hashlib.sha256(json.dumps(payload).encode()).hexdigest()
        plan = {"type": "plan", "passes": 2, "planned_calls": 3, "requests": [
            {"model": "clef", "id": "r000", "request": payload, "request_sha256": digest, "full_input_tokens": 4}]}
        answer = {"answers": {"q": {"type": "noul", "noul": 0.9}}, "usage": {"input_tokens": 4}}
        calls = [{"type": "call", "model": "clef", "id": "r000", "pass": p,
                  "status": 200, "warmup": p == -1, "request_sha256": digest,
                  "response": {"result": answer, "success": True, "errors": []},
                  "answer": answer, "reported_input_tokens": 4, "full_input_reported": True}
                 for p in [-1, 0, 1]]
        complete = {"type": "complete"}
        good = [plan, *calls, complete]
        # The duplicated fields must be what response_info() derives from the recorded body: a
        # substituted answer or token count is rejected, not trusted.
        other = {"answers": {"q": {"type": "noul", "noul": 0.1}}, "usage": {"input_tokens": 4}}
        bad = [[plan, *calls], [plan, *calls[:-1], complete],
               [plan, *calls, calls[-1], complete],
               [plan, *calls[:-1], {**calls[-1], "error": "failure"}, complete],
               [plan, *calls[:-1], {**calls[-1], "request_sha256": "wrong"}, complete],
               [plan, *calls[:-1], {**calls[-1], "answer": other}, complete],
               [plan, *calls[:-1], {**calls[-1], "reported_input_tokens": 9}, complete],
               [plan, *calls[:-1], {**calls[-1], "full_input_reported": False}, complete]]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "reference.jsonl"
            for records in [good, *bad]:
                path.write_text("".join(json.dumps(r) + "\n" for r in records))
                if records is good:
                    self.assertEqual(len(load_hosted(path)[1]), 3)
                else:
                    with self.assertRaises(ValueError):
                        load_hosted(path)


if __name__ == "__main__":
    unittest.main()
