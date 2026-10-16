"""Route capture integration, run with the real Omni text export."""
import json
import math
from pathlib import Path
import tempfile
import unittest

from test_omni_runtime import NativeHelper, REQUEST


class OmniRoutes(NativeHelper, unittest.TestCase):
    def test_capture_last_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"routes.jsonl"
            result = self.run_native([REQUEST, REQUEST], "--dump-routes", str(path),
                                     "--dump-last", "3", "--batch", "2")
            rows = [json.loads(s) for s in path.read_text().splitlines()]
            self.assertEqual(len(rows), 48*3)
            first = rows[0]["token"]
            self.assertGreater(first, 0)
            for i, row in enumerate(rows):
                self.assertEqual((row["layer"], row["token"]), (i//3, first+i%3))
                self.assertEqual(len(row["logits"]), 128)
                self.assertEqual(len(set(row["ids"])), 8)
                self.assertTrue(all(0 <= e < 128 for e in row["ids"]))
                self.assertTrue(all(math.isfinite(x) for x in row["logits"]+row["weights"]))
                self.assertAlmostEqual(sum(row["weights"]), 1, places=6)
                order = sorted(range(128), key=lambda e: (-row["logits"][e], e))
                self.assertEqual(row["ids"], order[:8])
                self.assertAlmostEqual(row["margin"], row["logits"][order[7]]-row["logits"][order[8]], places=5)
            self.assertEqual(result[0], result[1])

    def test_failed_pass_does_not_write_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"routes.jsonl"
            self.run_native([REQUEST], "--dump-routes", str(path), "--dump", str(Path(tmp)/"layers"),
                            "--dump-last", "3", env={"CLEF_DEBUG_NIL_CMDBUF": "3"}, returncode=1)
            self.assertFalse(path.exists())

    def test_write_failure_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            # A directory is unwritable as a route file, even with elevated filesystem access.
            self.run_native([REQUEST], '--dump-routes', tmp, '--dump-last', '1', returncode=1)


if __name__ == "__main__":
    unittest.main()
