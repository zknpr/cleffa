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
                                hosted_snapshot, load_hosted, score_value)


class ComparisonTests(unittest.TestCase):
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
