"""Incomplete or malformed hosted evidence must not improve apparent agreement."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from compare_cloudflare import confidence_value, decision, distribution, load_hosted, score_value


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

    def test_choice_uses_explicit_winner_for_rounded_tie(self):
        self.assertEqual(decision({"type": "choice", "choice": "b"}, {"a": 0.5, "b": 0.5}), "b")
        with self.assertRaises(ValueError):
            decision({"type": "choice", "choice": "b"}, {"a": 0.8, "b": 0.2})

    def test_every_planned_pass_required_exactly_once(self):
        payload = {"questions": {"q": {"type": "noul"}}}
        digest = hashlib.sha256(json.dumps(payload).encode()).hexdigest()
        plan = {"type": "plan", "passes": 2, "planned_calls": 3, "requests": [
            {"model": "clef", "id": "r000", "request": payload, "request_sha256": digest}]}
        calls = [{"type": "call", "model": "clef", "id": "r000", "pass": p,
                  "status": 200, "warmup": p == -1, "request_sha256": digest,
                  "answer": {"answers": {"q": {"type": "noul", "noul": 0.9}}}}
                 for p in [-1, 0, 1]]
        complete = {"type": "complete"}
        good = [plan, *calls, complete]
        bad = [[plan, *calls], [plan, *calls[:-1], complete],
               [plan, *calls, calls[-1], complete],
               [plan, *calls[:-1], {**calls[-1], "error": "failure"}, complete],
               [plan, *calls[:-1], {**calls[-1], "request_sha256": "wrong"}, complete]]
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
