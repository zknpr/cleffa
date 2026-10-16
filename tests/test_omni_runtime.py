"""Real-model native Omni smoke, packing, poison and late-failure regressions.

Run after producing gguf/clef-omni.gguf; kept out of model-free make test.
"""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
REQUEST = dict(model="clef-omni", state="The checkout is broken.",
               questions={"outage": dict(type="noul", instructions="Is a service down?")})


class NativeHelper:
    def run_native(self, requests, *args, env=None, returncode=0):
        p = subprocess.run([ROOT/"clef", "-m", ROOT/"gguf/clef-omni.gguf", "--logits", *args],
                           input="".join(json.dumps(r)+"\n" for r in requests),
                           capture_output=True, text=True, timeout=300,
                           env={**{k: v for k, v in os.environ.items() if not k.startswith("CLEF_")}, **(env or {})})
        self.assertEqual(p.returncode, returncode, p.stderr)
        result = [json.loads(line) for line in p.stdout.splitlines()]
        self.assertEqual(len(result), len(requests))
        return result


class OmniRuntime(NativeHelper, unittest.TestCase):
    def test_simd_width_guard(self):
        for name in ("omni_attn_prep", "omni_attention", "omni_attention_comp", "omni_route", "omni_router"):
            p = subprocess.run([ROOT/"clef", "-m", ROOT/"gguf/clef-omni.gguf"],
                               input="", capture_output=True, text=True,
                               env={**os.environ, "CLEF_DEBUG_SIMD_WIDTH_FOR": name})
            self.assertEqual(p.returncode, 1)
            self.assertIn(f"{name} requires a 32-lane SIMDgroup", p.stderr)

    def test_native_request(self):
        result = self.run_native([REQUEST])[0]
        self.assertIn("outage", result)
        self.assertEqual(len(result["outage"]), 2)

    def test_packing_and_poison(self):
        other = {**REQUEST, "state": "All services are healthy. "*31}
        separate = self.run_native([REQUEST, other])
        packed = self.run_native([REQUEST, other], "--batch", "2")
        poisoned = self.run_native([REQUEST, other], "--batch", "2", env={"CLEF_DEBUG_POISON": "1"})
        self.assertTrue(all("outage" in r for r in separate))
        self.assertEqual(separate, packed)
        self.assertEqual(separate, poisoned)

    def test_prefix_requests_fall_back(self):
        want = self.run_native([REQUEST, REQUEST])
        self.assertEqual(want, self.run_native([REQUEST, REQUEST], "--prefix-cache"))
        self.assertEqual(want, self.run_native([REQUEST, REQUEST], "--template-cache"))

    def test_late_command_failure_never_returns_logits(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Dumping splits at embedding/layer boundaries; creation 3 fails
            # after layer 0 has already run successfully on the device.
            results = self.run_native([REQUEST], "--dump", str(Path(tmp)/"layers.bin"), "--dump-last", "3",
                                      env={"CLEF_DEBUG_NIL_CMDBUF": "3"}, returncode=1)
            self.assertIn("command buffer", results[0].get("error", ""))
            self.assertNotIn("outage", results[0])


if __name__ == "__main__":
    unittest.main()
