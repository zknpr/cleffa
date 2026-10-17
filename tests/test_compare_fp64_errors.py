"""Non-inferiority comparison of two Omni builds against FP64, on generated fixtures."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from safetensors.numpy import save_file

import compare_fp64_errors as compare


class CompareFP64Errors(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="clef-noninf-")
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.golden = root / "golden"
        # Ten text questions; FP64 logits, and two FP32 references 1% apart in error.
        self.keys = [f"r{i:03d}/q" for i in range(10)]
        fp64 = {k: np.array([1.0, 0.0], dtype=np.float32) for k in self.keys}
        blocked = {k: np.array([1.0 + 0.010, 0.0], dtype=np.float32) for k in self.keys}
        sdpa = {k: np.array([1.0 + 0.0101, 0.0], dtype=np.float32) for k in self.keys}
        for name, logits in (("clef-omni-text-short-f64-v1", fp64), ("clef-omni-text-f32-blocked-d2", blocked),
                             ("clef-omni-text-f32-sdpa-control-d2", sdpa)):
            (self.golden / name).mkdir(parents=True)
            save_file(logits, str(self.golden / name / "logits.safetensors"))
        self.base, self.cand = root / "base", root / "cand"
        self.base.mkdir(); self.cand.mkdir()

    def write(self, directory, logit_errors, limits=0.05):
        rows = [{"id": k.split("/")[0], "question": "q", "logit_error": e, "prob_error": e / 20,
                 "limit_logit": limits, "limit_prob": limits / 20, "argmax_equal": True}
                for k, e in zip(self.keys, logit_errors)]
        (directory / "text-short-f64-v1.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    def run_compare(self, corpus=None):
        out = io.StringIO()
        # The fixture holds one FP64 reference; the real inventory spans eight.
        with patch.object(compare, "CORPUS", corpus or {"text-short-f64-v1": "text"}), \
                patch("sys.argv", ["compare", str(self.base), str(self.cand), "--golden-root", str(self.golden)]), \
                contextlib.redirect_stdout(out), self.assertRaises(SystemExit) as code:
            compare.main()
        return code.exception.code, out.getvalue()

    def test_equal_builds_are_non_inferior(self):
        self.write(self.base, [0.01] * 10)
        self.write(self.cand, [0.01] * 10)
        self.assertEqual(self.run_compare()[0], 0)

    def test_small_growth_within_five_percent_passes(self):
        self.write(self.base, [0.01] * 10)
        self.write(self.cand, [0.0104] * 10)
        self.assertEqual(self.run_compare()[0], 0)

    def test_growth_beyond_the_allowance_fails_although_every_bound_passes(self):
        # Every candidate question stays within its (generous) bound, but the mean grows by 30%.
        self.write(self.base, [0.01] * 10, limits=0.2)
        self.write(self.cand, [0.013] * 10, limits=0.2)
        code, out = self.run_compare()
        self.assertEqual(code, 1)
        self.assertIn("NOT shown non-inferior", out)

    def test_question_outside_its_bound_fails(self):
        self.write(self.base, [0.01] * 10)
        self.write(self.cand, [0.01] * 9 + [0.06])
        self.assertEqual(self.run_compare()[0], 1)

    def test_question_sets_must_match(self):
        self.write(self.base, [0.01] * 10)
        rows = (self.base / "text-short-f64-v1.jsonl").read_text().splitlines()[:9]
        (self.cand / "text-short-f64-v1.jsonl").write_text("\n".join(rows) + "\n")
        self.assertIn("question sets differ", str(self.run_compare()[0]))

    def test_reports_must_cover_every_referenced_question(self):
        # On a subset the FP32 spread, and so the allowance, depends on which questions were picked.
        for directory in (self.base, self.cand):
            self.write(directory, [0.01] * 10)
            rows = (directory / "text-short-f64-v1.jsonl").read_text().splitlines()[:9]
            (directory / "text-short-f64-v1.jsonl").write_text("\n".join(rows) + "\n")
        self.assertIn("reports cover 9 of the 10", str(self.run_compare()[0]))

    def test_missing_references_fail(self):
        self.write(self.base, [0.01] * 10)
        self.write(self.cand, [0.01] * 10)
        corpus = {"text-short-f64-v1": "text", "text-mid-f64-v1": "text"}
        self.assertIn("missing", str(self.run_compare(corpus)[0]))
        sdpa = self.golden / "clef-omni-text-f32-sdpa-control-d2" / "logits.safetensors"
        save_file({k: np.array([1.0101, 0.0], dtype=np.float32) for k in self.keys[:9]}, str(sdpa))
        self.assertIn("FP32 references cover", str(self.run_compare()[0]))

    def test_bound_comes_from_the_reference_not_the_report(self):
        # Reports recorded with relaxed limits must not move a question inside its bound.
        self.write(self.base, [0.01] * 9 + [0.06], limits=5)
        self.write(self.cand, [0.01] * 9 + [0.06], limits=5)
        code, out = self.run_compare()
        self.assertEqual(code, 1)
        self.assertIn("1 questions outside the bound", out)


if __name__ == "__main__":
    unittest.main()
