"""Exercise parity acceptance criteria without running inference or modifying golden data."""

import contextlib
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import test_parity as parity

sys.path.append(str(Path(__file__).resolve().parents[1] / "ref"))
import omni_fp64_policy  # noqa: E402
import oracle_fp64_bounded  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        self.bound_dir = self.root / "bound"
        self.bound_dir.mkdir()
        (self.bound_dir / "encoded.jsonl").write_text(json.dumps(self.encoded) + '\n')
        (self.bound_dir / "requests.jsonl").write_text('{"id":"r0"}\n')
        for directory in (self.golden, self.bound_dir):
            (directory / "logits.safetensors").touch()  # load_file is patched; the bytes are hashed
        self.bound = self.reference.copy()
        # Under --bound-by, GOLDEN must be the complete independent FP64 reference and the bound
        # the independent blocked-attention FP32 reference of the same corpus.
        self.fp64_manifest: dict[str, Any] = dict(complete=True, revision=omni_fp64_policy.REVISION, precision="float64-streamed",
                                  independent_from_native=True, fp64_policy=omni_fp64_policy.POLICY,
                                  corpus="text", records=[dict(id="r0", tokens=2, dump_start=0)])
        self.fp32_manifest: dict[str, Any] = dict(complete=True, revision=omni_fp64_policy.REVISION, precision="float32-streamed",
                                  independent_from_native=True, corpus="text",
                                  attention_reference=dict(value_reduction="blocked", first_layer=0))

    def seal(self):
        """Write both manifests over the current files; a test's own hash entries take precedence."""
        names = ["requests.jsonl", "encoded.jsonl", "logits.safetensors"]
        names += [f"layers/{r['id']}.safetensors" for r in self.fp64_manifest["records"]]
        hashes = {name: sha256(self.golden / name) for name in names}
        (self.golden / "manifest.json").write_text(json.dumps({"artifacts_sha256": hashes, **self.fp64_manifest}))
        (self.bound_dir / "manifest.json").write_text(json.dumps(
            {"corpus_sha256": sha256(self.bound_dir / "requests.jsonl"), **self.fp32_manifest}))

    def check(self, *flags):
        if '--bound-by' in flags:
            self.seal()
        def run(args, **kwargs):
            if "encode" in args:
                text = json.dumps(self.encoded, ensure_ascii=False) + '\n'
            elif "--dump" in args:
                rows = int(args[args.index("--dump-last") + 1]) if "--dump-last" in args else self.dump.shape[1]
                self.dump[:,-rows:].tofile(args[args.index("--dump") + 1])
                text = ''
            else:
                text = json.dumps({self.qid: self.engine.tolist()}, ensure_ascii=False) + '\n'
            return subprocess.CompletedProcess(args, 0, text, '')

        def load(path):
            if Path(path).name != "logits.safetensors":
                return self.layers
            return {f"r0/{self.qid}": self.bound if Path(path).parent == self.bound_dir else self.reference}

        output = io.StringIO()
        with patch.object(parity, 'ROOT', self.root), patch.object(parity.subprocess, 'run', run), \
                patch.object(parity, 'load_file', load), patch('sys.argv',
                ['test_parity.py', 'unused.gguf', str(self.golden), *flags]), contextlib.redirect_stdout(output):
            with self.assertRaises(SystemExit) as exit_result:
                parity.main()
        return exit_result.exception.code, output.getvalue()

    def test_exact_reference_passes_including_layers(self):
        self.assertEqual(self.check('--dump')[0], 0)

    def test_explicit_layer_suffix_uses_matching_native_rows(self):
        (self.golden/'manifest.json').write_text(json.dumps(dict(records=[dict(id='r0',tokens=2,dump_start=1)])))
        self.layers={key:value[1:].copy() for key,value in self.layers.items()}
        self.dump[:,0]=17  # The discarded prefix must not affect suffix comparison.
        self.assertEqual(self.check('--dump')[0],0)
        self.dump[1,1]=1.2
        self.assertNotEqual(self.check('--dump')[0],0)

    def test_layer_suffix_requires_valid_manifest_geometry(self):
        self.layers={key:value[1:].copy() for key,value in self.layers.items()}
        self.assertNotEqual(self.check('--dump')[0],0)
        for record in [dict(id='r0',tokens=3,dump_start=1),dict(id='r0',tokens=2,dump_start=-1),
                       dict(id='r0',tokens=2,dump_start=2),dict(id='r0',tokens=2,dump_start=True),
                       dict(id='other',tokens=2,dump_start=1)]:
            with self.subTest(record=record):
                (self.golden/'manifest.json').write_text(json.dumps(dict(records=[record])))
                self.assertNotEqual(self.check('--dump')[0],0)
        (self.golden/'manifest.json').write_text(json.dumps(dict(records=[dict(id='r0',tokens=2,dump_start=1)])))
        self.layers['final_norm']=np.ones((2,2))
        self.assertNotEqual(self.check('--dump')[0],0)

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

    def test_bound_allows_the_fp32_references_own_error(self):
        # FP64 reference, native and FP32 bound: native is outside the fixed logit limit but
        # closer to FP64 than the FP32 reference is on this question.
        self.engine = self.reference + np.float32(0.08)
        self.bound = self.reference + np.float32(0.09)
        self.assertNotEqual(self.check()[0], 0)
        self.assertEqual(self.check('--bound-by', str(self.bound_dir))[0], 0)

    def test_bound_rejects_error_beyond_the_fp32_reference(self):
        self.engine = self.reference + np.float32(0.1)
        self.bound = self.reference + np.float32(0.09)
        code, output = self.check('--bound-by', str(self.bound_dir))
        self.assertNotEqual(code, 0)
        self.assertIn("FAIL |d logit|", output)

    def test_bound_keeps_fixed_limits_when_reference_is_closer(self):
        self.engine = self.reference + np.float32(0.04)
        self.bound = self.reference.copy()
        self.assertEqual(self.check('--bound-by', str(self.bound_dir))[0], 0)
        self.engine = self.reference + np.float32(0.06)
        self.assertNotEqual(self.check('--bound-by', str(self.bound_dir))[0], 0)

    def test_bound_requires_identical_encoding(self):
        (self.bound_dir / "encoded.jsonl").write_text(json.dumps({**self.encoded, "input_ids": [1, 3]}) + '\n')
        code, output = self.check('--bound-by', str(self.bound_dir))
        self.assertNotEqual(code, 0)
        self.assertIn("encoded differently", output)

    def test_bound_argmax_may_follow_either_reference(self):
        # FP64 prefers option 0 by a hair, the FP32 reference option 1; native follows FP32.
        self.reference = np.array([0.001, 0.0], dtype=np.float32)
        self.bound = np.array([0.0, 0.001], dtype=np.float32)
        self.engine = self.bound.copy()
        self.assertNotEqual(self.check()[0], 0)
        self.assertEqual(self.check('--bound-by', str(self.bound_dir))[0], 0)

    def test_bound_makes_layer_drift_diagnostic_but_not_nonfinite(self):
        self.dump[1] *= 1.1
        self.assertNotEqual(self.check('--dump')[0], 0)
        code, output = self.check('--dump', '--bound-by', str(self.bound_dir))
        self.assertEqual(code, 0, output)
        self.assertIn("diagnostic", output)
        self.dump[1, 0, 0] = np.nan
        self.assertNotEqual(self.check('--dump', '--bound-by', str(self.bound_dir))[0], 0)

    def test_bound_accepts_both_fp64_capture_policies(self):
        for policy in (omni_fp64_policy.POLICY, oracle_fp64_bounded.POLICY):
            with self.subTest(policy=policy["name"]):
                self.fp64_manifest["fp64_policy"] = policy
                self.assertEqual(self.check('--bound-by', str(self.bound_dir))[0], 0)

    def test_bound_requires_the_complete_independent_fp64_reference(self):
        # Matching token ids alone accepted an FP32 or incomplete directory as GOLDEN.
        valid = dict(self.fp64_manifest)
        for change in (dict(precision="float32-streamed"), dict(complete=False), dict(arbitration={}),
                       dict(independent_from_native=False), dict(fp64_policy={}), dict(revision="main")):
            with self.subTest(change=change):
                self.fp64_manifest = {**valid, **change}
                code, output = self.check('--bound-by', str(self.bound_dir))
                self.assertNotEqual(code, 0)
                self.assertIn("not the complete independent FP64 reference", output)

    def test_bound_requires_the_fp64_artifacts_and_record_inventory(self):
        self.fp64_manifest["artifacts_sha256"] = {name: "0" * 64 for name in (
            "requests.jsonl", "encoded.jsonl", "logits.safetensors", "layers/r0.safetensors")}
        code, output = self.check('--bound-by', str(self.bound_dir))
        self.assertNotEqual(code, 0)
        self.assertIn("artifact hash mismatch", output)
        # A record dropped from requests.jsonl and encoded.jsonl must not shrink the gate silently.
        del self.fp64_manifest["artifacts_sha256"]
        (self.golden / "layers/r1.safetensors").touch()
        self.fp64_manifest["records"] = [dict(id="r0", tokens=2, dump_start=0), dict(id="r1", tokens=2, dump_start=0)]
        code, output = self.check('--bound-by', str(self.bound_dir))
        self.assertNotEqual(code, 0)
        self.assertIn("record inventory", output)

    def test_bound_must_be_the_independent_blocked_fp32_reference(self):
        # Arbitrated, SDPA-control, partial and BF16 references, or another corpus, are not the bound.
        valid = dict(self.fp32_manifest)
        for change in (dict(arbitration={}), dict(attention_reference=dict(value_reduction="sdpa", first_layer=0)),
                       dict(attention_reference=dict(value_reduction="blocked", first_layer=12)),
                       dict(precision="bfloat16"), dict(independent_from_native=False), dict(complete=False),
                       dict(corpus="omni-vision")):
            with self.subTest(change=change):
                self.fp32_manifest = {**valid, **change}
                code, output = self.check('--bound-by', str(self.bound_dir))
                self.assertNotEqual(code, 0)
                self.assertIn("not the independent blocked-attention FP32 reference", output)

    def test_bound_must_share_the_request_bytes(self):
        # Identical token ids do not tie media bytes: a bound built from other pixels passed.
        (self.bound_dir / "requests.jsonl").write_text('{"id":"r0","images":["other"]}\n')
        code, output = self.check('--bound-by', str(self.bound_dir))
        self.assertNotEqual(code, 0)
        self.assertIn("requests differ", output)
        (self.bound_dir / "requests.jsonl").write_text('{"id":"r0"}\n')
        self.fp32_manifest["corpus_sha256"] = "0" * 64
        code, output = self.check('--bound-by', str(self.bound_dir))
        self.assertNotEqual(code, 0)
        self.assertIn("corpus hash", output)

    def test_bound_refuses_relaxed_fixed_limits(self):
        for flags in (('--max-logit-error', '5'), ('--max-prob-error', '1')):
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()):
                self.assertNotEqual(self.check('--bound-by', str(self.bound_dir), *flags)[0], 0)
        self.assertEqual(self.check('--bound-by', str(self.bound_dir), '--max-logit-error', '0.01')[0], 0)

    def test_invalid_tolerance_fails(self):
        for value in ('nan', 'inf', '-1'):
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertNotEqual(self.check('--max-prob-error', value)[0], 0)


if __name__ == "__main__":
    unittest.main()
