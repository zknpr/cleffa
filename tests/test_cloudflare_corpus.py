"""The hosted reference must fail closed on unavailable or exhausted free usage."""

import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from cloudflare_corpus import check_budget, daily_usage, final_usage, hosted_payload


class BudgetTests(unittest.TestCase):
    def test_explicit_fallback_does_not_mutate_corpus(self):
        request = {"id": "r000", "model": "clef", "state": "public", "questions": {
            "fallback": {"type": "noul"}, "explicit": {"type": "noul", "instructions": "Keep this."}}}
        payload = hosted_payload(request, "clef-flash")
        self.assertNotIn("id", payload)
        self.assertEqual(payload["model"], "clef-flash")
        self.assertEqual(payload["questions"]["fallback"]["instructions"], "fallback")
        self.assertEqual(payload["questions"]["explicit"]["instructions"], "Keep this.")
        self.assertNotIn("instructions", request["questions"]["fallback"])

    def test_full_reservation_and_headroom(self):
        check_budget(6000, 2000)
        for used, estimate in [(6000.01, 2000), (0, 2501), (-1, 10),
                               (0, float("nan")), (float("inf"), 1)]:
            with self.subTest(used=used, estimate=estimate), self.assertRaises(ValueError):
                check_budget(used, estimate)

    def test_usage_response_validation(self):
        valid = {"data": {"viewer": {"accounts": [{"aiInferenceAdaptiveGroups": [
            {"sum": {"totalNeurons": 123.5}}]}]}}}
        bad = [{"errors": ["permission denied"]}, {"data": {"viewer": {"accounts": []}}},
               {"data": {"viewer": {"accounts": [{"aiInferenceAdaptiveGroups": [
                   {"sum": {"totalNeurons": -1}}]}]}}}]
        for data in [valid, *bad]:
            with patch("cloudflare_corpus.http.client.HTTPSConnection") as connection:
                response = connection.return_value.getresponse.return_value
                response.status = 200
                response.read.return_value = json.dumps(data).encode()
                if data is valid:
                    self.assertEqual(daily_usage("a" * 32, "secret")["used_neurons"], 123.5)
                else:
                    with self.assertRaises((RuntimeError, ValueError)):
                        daily_usage("a" * 32, "secret")
                connection.return_value.request.assert_called_once()
                connection.return_value.close.assert_called_once()

    def test_final_usage_is_best_effort(self):
        import io
        out = io.StringIO()
        with patch("cloudflare_corpus.daily_usage", return_value={"type": "budget", "used_neurons": 5}):
            final_usage(out, "a" * 32, "secret-token")
        self.assertEqual(json.loads(out.getvalue().splitlines()[-1])["used_neurons"], 5)
        for exc in (RuntimeError("Cannot verify daily usage; stopping inference"), OSError("reset"),
                    KeyError("data"), ValueError("bad"), TypeError("none"), RuntimeError("leak secret-token here")):
            out = io.StringIO()
            with patch("cloudflare_corpus.daily_usage", side_effect=exc):
                final_usage(out, "a" * 32, "secret-token")   # must not raise: the run is already complete
            record = json.loads(out.getvalue().splitlines()[-1])
            self.assertEqual(record["type"], "budget")
            self.assertIn("error", record)
            self.assertNotIn("secret-token", json.dumps(record))


if __name__ == "__main__":
    unittest.main()
