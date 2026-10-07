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
        # measure_bound() hashes and guards the 18 GB files; the report tests stand in for it.
        def bound(model, rows, args, *engine):
            return {"path": f"gguf/{model}.gguf", "bytes": 1, "sha256": "0" * 64}, measure(model, rows, args)
        with patch.object(checkout_latency, "measure_bound", bound), \
                patch.object(sys, "argv", ["checkout_latency.py", str(out), "--models", "clef-flash", "clef"]):
            checkout_latency.main()

    def test_measurement_fails_if_the_gguf_changes_meanwhile(self):
        # The published hash must describe the bytes the server ran: a GGUF replaced after the
        # hash and before or during the measurement fails the run.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "gguf").mkdir()
            gguf = root / "gguf" / "clef-flash.gguf"
            gguf.write_bytes(b"GGUF" + bytes(range(256)))
            server = root / "clef-server"
            server.write_bytes(b"\xcf\xfa\xed\xfe engine")
            rows, args = [], None
            with patch.object(checkout_latency, "ROOT", root), \
                    patch.object(checkout_latency, "measure", lambda m, r, a: [{"id": "blog", "median_ms": 1.0}]):
                engine = checkout_latency.file_state(server)
                identity, result = checkout_latency.measure_bound("clef-flash", rows, args, engine)
                self.assertEqual(identity["bytes"], gguf.stat().st_size)

                def regenerate(model, r, a):
                    gguf.write_bytes(b"GGUF" + bytes(range(255, -1, -1)))   # same size, other bytes
                    return [{"id": "blog", "median_ms": 1.0}]
                with patch.object(checkout_latency, "measure", regenerate), \
                        self.assertRaisesRegex(RuntimeError, "changed"):
                    checkout_latency.measure_bound("clef-flash", rows, args, engine)

                # The server binary too: rebuilt between the Flash and 27B measurements, it
                # would run bytes the recorded engine hash does not describe.
                gguf.write_bytes(b"GGUF" + bytes(range(256)))
                engine = checkout_latency.file_state(server)

                def rebuild(model, r, a):
                    server.write_bytes(b"\xcf\xfa\xed\xfe other!")
                    return [{"id": "blog", "median_ms": 1.0}]
                with patch.object(checkout_latency, "measure", rebuild), \
                        self.assertRaisesRegex(RuntimeError, "clef-server.*changed"):
                    checkout_latency.measure_bound("clef-flash", rows, args, engine)

    def test_report_binds_to_the_measured_gguf(self):
        # The model name alone does not say which weights ran; the report carries each GGUF's
        # size and SHA-256, taken before its measurement.
        import hashlib
        with tempfile.TemporaryDirectory() as temp:
            gguf = Path(temp) / "x.gguf"
            gguf.write_bytes(b"GGUF" + bytes(range(256)) * 10)
            identity = checkout_latency.gguf_identity(gguf)
            self.assertEqual(identity["sha256"], hashlib.sha256(gguf.read_bytes()).hexdigest())
            self.assertEqual(identity["bytes"], gguf.stat().st_size)
            out = Path(temp) / "latency.json"
            self.run_main(out, lambda model, rows, args: [{"id": "blog", "median_ms": 1.0}])
            report = json.loads(out.read_text())
            self.assertEqual(set(report["gguf"]), {"clef-flash", "clef"})
            self.assertEqual(report["gguf"]["clef"]["sha256"], "0" * 64)

    def test_server_environment_carries_no_engine_overrides(self):
        # A CLEF_* diagnostic inherited from the shell (CLEF_PROFILE serializes the GPU,
        # CLEF_ATTN_TU=0 selects another mode) would skew every sample without a trace.
        with patch.dict(os.environ, {"CLEF_PROFILE": "1", "CLEF_ATTN_TU": "0", "PATH": os.environ.get("PATH", "")}):
            env = checkout_latency.engine_env()
        self.assertFalse([k for k in env if k.startswith("CLEF_")], env)
        self.assertIn("PATH", env)

    def test_engine_hash_describes_one_snapshot(self):
        # The binary's identity is taken before the hash and must hold afterwards; a rebuild
        # between the two would record a hash of bytes the measurements never ran.
        with tempfile.TemporaryDirectory() as temp:
            server = Path(temp) / "clef-server"
            server.write_bytes(b"\xcf\xfa\xed\xfe engine")
            digest, state = checkout_latency.engine_identity(server)
            self.assertEqual(state, checkout_latency.file_state(server))
            states = [checkout_latency.file_state(server), (0, 0, 0, 0, 0)]   # changed between the two looks
            with patch.object(checkout_latency, "file_state", side_effect=states), \
                    self.assertRaisesRegex(RuntimeError, "changed"):
                checkout_latency.engine_identity(server)

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
