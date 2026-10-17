"""Exercise setup's verify-before-promotion boundary without downloading weights."""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class DownloadModels(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        shutil.copy(ROOT/"download_models.sh", self.path)
        (self.path/".venv/bin").mkdir(parents=True)
        (self.path/"model-omni").mkdir()
        (self.path/"gguf").mkdir()
        (self.path/"gguf/clef-omni.gguf").write_text("previous verified model")
        py = self.path/".venv/bin/python"
        py.write_text('''#!/bin/sh
set -eu
[ "$1" != "-B" ] || shift
echo "$*" >> calls
case "$1" in
  tools/verify_snapshot.py) [ "${FAIL_AT:-}" != snapshot ] ;;
  tools/convert.py) printf 'new model' > "$3" ;;
  tests/verify_gguf.py) [ "${FAIL_AT:-}" != gguf ] ;;
  *) exit 90 ;;
esac
''')
        py.chmod(0o755)

    def run_setup(self, fail=""):
        return subprocess.run(["sh", "download_models.sh", "clef-omni", "--skip-download"],
                              cwd=self.path, env={**os.environ, "FAIL_AT": fail}, capture_output=True, text=True)

    def test_success_promotes_after_both_verifications(self):
        p = self.run_setup()
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual((self.path/"gguf/clef-omni.gguf").read_text(), "new model")
        calls = (self.path/"calls").read_text().splitlines()
        self.assertEqual([s.split()[0] for s in calls],
                         ["tools/verify_snapshot.py", "tools/convert.py", "tests/verify_gguf.py"])
        self.assertIn("0db1cd2607d76a7bdb2a382f659e7b313079f84b", calls[0])
        self.assertFalse((self.path/"gguf/clef-omni.gguf.tmp").exists())

    def test_failure_preserves_previous_model(self):
        for stage in ("snapshot", "gguf"):
            with self.subTest(stage=stage):
                p = self.run_setup(stage)
                self.assertNotEqual(p.returncode, 0)
                self.assertEqual((self.path/"gguf/clef-omni.gguf").read_text(), "previous verified model")
                self.assertTrue((self.path/"calls").exists(), p.stderr)
                self.assertIn("tools/verify_snapshot.py", (self.path/"calls").read_text())


if __name__ == "__main__":
    unittest.main()
