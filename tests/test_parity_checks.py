"""Exercise parity acceptance criteria without running inference or modifying golden data."""

import contextlib
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import test_parity as parity


class ParityChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="clef-parity-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.golden = self.root / "golden"
        (self.golden / "layers").mkdir(parents=True)
        (self.golden / "layers/r0.safetensors").touch()
        (self.golden / "requests.jsonl").write_text('{"id":"r0"}\n')
        self.encoded = {"id": "r0", "input_ids": [1, 2], "questions": [
            {"id": "q", "option_ids": ["a", "b"]}]}
        (self.golden / "encoded.jsonl").write_text(json.dumps(self.encoded) + '\n')
        self.reference = np.array([0.1, 0.0], dtype=np.float32)
        self.engine = self.reference.copy()
        self.qid = "q"
        self.layers = {"embed": np.ones((2, 2)), "final_norm": np.ones((2, 2))}
        self.dump = np.ones((2, 2, 2), dtype=np.float32)

    def check(self, *flags):
        def run(args, **kwargs):
            if "encode" in args:
                text = json.dumps(self.encoded, ensure_ascii=False) + '\n'
            elif "--dump" in args:
                self.dump.tofile(args[args.index("--dump") + 1])
                text = ''
            else:
                text = json.dumps({self.qid: self.engine.tolist()}, ensure_ascii=False) + '\n'
            return subprocess.CompletedProcess(args, 0, text, '')

        def load(path):
            return {f"r0/{self.qid}": self.reference} if Path(path).name == "logits.safetensors" else self.layers

        output = io.StringIO()
        with patch.object(parity, 'ROOT', self.root), patch.object(parity.subprocess, 'run', run), \
                patch.object(parity, 'load_file', load), patch('sys.argv',
                ['test_parity.py', 'unused.gguf', str(self.golden), *flags]), contextlib.redirect_stdout(output):
            with self.assertRaises(SystemExit) as exit_result:
                parity.main()
        return exit_result.exception.code, output.getvalue()

    def test_exact_reference_passes_including_layers(self):
        self.assertEqual(self.check('--dump')[0], 0)

    def test_unicode_separators_are_not_jsonl_boundaries(self):
        for separator in ('\u0085', '\u2028', '\u2029'):
            with self.subTest(separator=ascii(separator)):
                req = {"id": "r0", "state": f"before{separator}after"}
                (self.golden / 'requests.jsonl').write_text(json.dumps(req, ensure_ascii=False) + '\n')
                self.assertEqual(self.check('--dump')[0], 0)

    def test_unicode_separators_in_question_ids(self):
        self.qid = 'q\u0085\u2028\u2029'
        self.encoded['questions'][0]['id'] = self.qid
        (self.golden / 'encoded.jsonl').write_text(json.dumps(self.encoded, ensure_ascii=False) + '\n')
        self.assertEqual(self.check()[0], 0)

    def test_same_argmax_wrong_confidence_fails(self):
        self.engine *= 10
        self.assertNotEqual(self.check()[0], 0)

    def test_probability_threshold_is_independent(self):
        self.engine *= 10
        self.assertNotEqual(self.check('--max-logit-error', '100')[0], 0)

    def test_common_logit_shift_fails_even_with_same_probabilities(self):
        self.engine += 1
        self.assertNotEqual(self.check()[0], 0)

    def test_argmax_still_required(self):
        self.engine = self.engine[::-1]
        self.assertNotEqual(self.check('--max-logit-error', '1', '--max-prob-error', '1')[0], 0)

    def test_nonfinite_engine_and_reference_fail(self):
        for value in (np.nan, np.inf, -np.inf):
            for target in (self.engine, self.reference):
                with self.subTest(value=value, target=target):
                    old = target[0]
                    target[0] = value
                    self.assertNotEqual(self.check()[0], 0)
                    target[0] = old

    def test_layer_drift_and_nonfinite_values_fail(self):
        for value in (1.2, np.nan, np.inf):
            with self.subTest(value=value):
                self.dump[1] = value
                self.assertNotEqual(self.check('--dump')[0], 0)

    def test_missing_requested_layers_fail(self):
        (self.golden / "layers/r0.safetensors").unlink()
        self.assertNotEqual(self.check('--dump')[0], 0)

    def test_empty_corpus_fails(self):
        (self.golden / "requests.jsonl").write_text('')
        (self.golden / "encoded.jsonl").write_text('')
        self.assertNotEqual(self.check()[0], 0)

    def test_invalid_tolerance_fails(self):
        for value in ('nan', 'inf', '-1'):
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertNotEqual(self.check('--max-prob-error', value)[0], 0)


if __name__ == "__main__":
    unittest.main()
