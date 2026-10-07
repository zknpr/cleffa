"""The checkout latency report must not look complete until every requested model finished."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
import checkout_latency


class ReportTests(unittest.TestCase):
    def run_main(self, out: Path, measure):
        with patch.object(checkout_latency, "measure", measure), \
                patch.object(sys, "argv", ["checkout_latency.py", str(out), "--models", "clef-flash", "clef"]):
            checkout_latency.main()

    def test_server_environment_carries_no_engine_overrides(self):
        # A CLEF_* diagnostic inherited from the shell (CLEF_PROFILE serializes the GPU,
        # CLEF_ATTN_TU=0 selects another mode) would skew every sample without a trace.
        with patch.dict(os.environ, {"CLEF_PROFILE": "1", "CLEF_ATTN_TU": "0", "PATH": os.environ.get("PATH", "")}):
            env = checkout_latency.engine_env()
        self.assertFalse([k for k in env if k.startswith("CLEF_")], env)
        self.assertIn("PATH", env)

    def test_report_is_published_only_after_every_model(self):
        def fail_on_second(model, rows, args):
            if model == "clef":
                raise RuntimeError("server did not start")
            return [{"id": "blog", "median_ms": 1.0}]
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "latency.json"
            with self.assertRaises(RuntimeError):
                self.run_main(out, fail_on_second)
            self.assertFalse(out.exists(), "a one-model report was published as if complete")
            partial = [p.name for p in Path(temp).iterdir()]
            self.assertTrue(any("partial" in n for n in partial), partial)   # the checkpoint survives for inspection

    def test_complete_report_records_the_requested_models(self):
        def ok(model, rows, args):
            return [{"id": "blog", "median_ms": 1.0}]
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "latency.json"
            self.run_main(out, ok)
            report = json.loads(out.read_text())
            self.assertEqual(report["requested_models"], ["clef-flash", "clef"])
            self.assertIs(report["complete"], True)
            self.assertEqual(set(report["models"]), {"clef-flash", "clef"})
            self.assertEqual([p.name for p in Path(temp).iterdir()], ["latency.json"])   # no checkpoint left


if __name__ == "__main__":
    unittest.main()
