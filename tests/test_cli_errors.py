"""CLI write failures and UTF-8 error responses. Usage: test_cli_errors.py MODEL.gguf

The HTTP arm starts its own localhost server and stops it before exiting.
"""

import http.client
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL = sys.argv.pop(1)
REQUEST = {"model": "clef", "state": "test", "questions": {"q": {"type": "noul"}}}


class CliErrors(unittest.TestCase):
    def test_template_flags_are_rejected_before_model_open(self):
        for flags in (["--prefix-cache", "--template-cache"],
                      ["--template-cache", "--prefix-cache"],
                      ["--template-cache", "--batch", "2"],
                      ["--template-cache", "--dump", "/unused"]):
            with self.subTest(flags=flags):
                p = subprocess.run([ROOT / "clef", "-m", "/nonexistent", *flags],
                                   text=True, capture_output=True, timeout=10)
                self.assertEqual(p.returncode, 2, p.stderr)
                self.assertIn("--template-cache", p.stderr)
                self.assertNotIn("cannot open", p.stderr)

    def test_truncation_requires_explicit_opt_in(self):
        request = {**REQUEST, "state": "alpha " * 20000}
        # A failed first command buffer detects whether encoding reached inference,
        # without spending GPU time on an input the default must reject intact.
        env = {k: v for k, v in os.environ.items() if not k.startswith("CLEF_")}
        env["CLEF_DEBUG_NIL_CMDBUF"] = "1"
        for flags in ([], ["--no-truncate"], ["--truncate", "--no-truncate"]):
            with self.subTest(flags=flags):
                p = subprocess.run([ROOT / "clef", "-m", MODEL, "--time", *flags],
                                   input=json.dumps(request) + "\n", text=True,
                                   capture_output=True, timeout=60, env=env)
                self.assertEqual(p.returncode, 0, p.stderr)
                self.assertIn("the reference would silently drop the rest", json.loads(p.stdout)["error"])
                self.assertNotIn("batch of", p.stderr)
        for flags in (["--truncate"], ["--no-truncate", "--truncate"]):
            with self.subTest(flags=flags):
                p = subprocess.run([ROOT / "clef", "-m", MODEL, "--time", *flags],
                                   input=json.dumps(request) + "\n", text=True,
                                   capture_output=True, timeout=60, env=env)
                self.assertNotEqual(p.returncode, 0)
                self.assertIn("command buffer", json.loads(p.stdout)["error"])
                self.assertIn("(16384 tokens)", p.stderr)

    def test_unwritable_stdout_is_an_error(self):
        # Small responses fail on fflush; the long model name also exercises a write
        # during printf. Both successful responses and validation errors must propagate it.
        cases = [(REQUEST, []), (REQUEST, ["--logits"]), ({}, []),
                 ({**REQUEST, "model": "m" * 16384}, [])]
        for request, flags in cases:
            with self.subTest(flags=flags, model_length=len(request.get("model", ""))):
                with open('/dev/null', 'rb') as unwritable:
                    p = subprocess.run([ROOT / "clef", "-m", MODEL, *flags],
                                       input=json.dumps(request).encode() + b"\n", stdout=unwritable,
                                       stderr=subprocess.PIPE, timeout=60)
                self.assertNotEqual(p.returncode, 0, "lost output was reported as success")
                self.assertIn(b"stdout", p.stderr)

    def test_unicode_errors_are_json_over_cli_and_http(self):
        requests = [{**REQUEST, "questions": {prefix + ch * 100: {"type": "unknown"}}}
                    for ch in ("é", "界", "🚀") for prefix in ("", "x", "xx")]
        p = subprocess.run([ROOT / "clef", "-m", MODEL],
                           input="".join(json.dumps(r) + "\n" for r in requests).encode(),
                           capture_output=True, timeout=60, check=True)
        lines = p.stdout.splitlines()
        self.assertEqual(len(lines), len(requests))
        for line in lines:
            with self.subTest(transport="CLI", response=line):
                self.assertIn("error", json.loads(line.decode("utf-8")))

        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        with tempfile.TemporaryFile() as log:
            server = subprocess.Popen([ROOT / "clef-server", "-m", MODEL, "--port", str(port),
                                       "--no-warmup"], stdout=subprocess.DEVNULL, stderr=log)
            try:
                deadline = time.monotonic() + 60
                while True:
                    self.assertIsNone(server.poll(), "server exited before accepting requests")
                    try:
                        with socket.create_connection(("127.0.0.1", port), timeout=1):
                            break
                    except OSError:
                        if time.monotonic() >= deadline:
                            self.fail("server did not start")
                        time.sleep(0.1)
                for request in requests:
                    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
                    try:
                        c.request("POST", "/v1/systemone", json.dumps(request).encode())
                        r = c.getresponse()
                        self.assertEqual(r.status, 400)
                        with self.subTest(transport="HTTP", question=next(iter(request["questions"]))):
                            self.assertIn("error", json.loads(r.read().decode("utf-8")))
                    finally:
                        c.close()
            finally:
                server.terminate()
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()


if __name__ == "__main__":
    unittest.main()
