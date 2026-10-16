"""Independent verification must reject malformed Omni exports, not trust their table."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import gguf
import numpy as np

from omni_fixture import snapshot

ROOT = Path(__file__).resolve().parents[1]


class VerifyOmni(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.path = Path(cls.tmp.name)
        snapshot(cls.path)
        cls.good = cls.path/"good.gguf"
        p = subprocess.run([sys.executable, "-B", ROOT/"tools/convert.py", cls.path, cls.good], capture_output=True)
        if p.returncode:
            raise AssertionError(p.stderr.decode())

    def verify(self, path, error=None):
        p = subprocess.run([sys.executable, "-B", ROOT/"tests/verify_gguf.py", self.path, path],
                           capture_output=True, text=True)
        if error is None:
            self.assertEqual(p.returncode, 0, p.stdout+p.stderr)
        else:
            self.assertNotEqual(p.returncode, 0)
            self.assertIn(error.lower(), (p.stdout+p.stderr).lower())

    def rewrite(self, *, omit=None, change=None, meta=None, extra=False):
        r = gguf.GGUFReader(self.good)
        path = self.path/"bad.gguf"
        w = gguf.GGUFWriter(path, "clef")
        for key, field in r.fields.items():
            if key.startswith("GGUF.") or key == "general.architecture":
                continue
            value = field.contents() if meta is None or meta[0] != key else meta[1]
            w.add_key_value(key, value, field.types[0], field.types[-1] if len(field.types)>1 else None)
        for t in r.tensors:
            if t.name == omit:
                continue
            data = np.array(t.data, copy=True)
            data = data.reshape(-1).view(np.uint16 if t.tensor_type == gguf.GGMLQuantizationType.BF16 else np.float32)
            data = data.reshape(tuple(reversed(t.shape)))
            if change and change[0] == t.name:
                data = change[1](data)
            w.add_tensor(t.name, data, raw_dtype=t.tensor_type)
        if extra:
            w.add_tensor("unexpected.weight", np.ones(4, dtype=np.float32))
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()
        return path

    def test_valid_full(self):
        self.verify(self.good)

    def test_valid_partial(self):
        for mode in ("text", "vision"):
            path = self.path/(mode+".gguf")
            p = subprocess.run([sys.executable, "-B", ROOT/"tools/convert.py", self.path, path,
                                "--modalities", mode], capture_output=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.verify(path)

    def test_missing_and_extra(self):
        for n in ("blk.0.ffn_down_exps.weight", "v.deepstack.1.mm.0.weight", "a.conv_out.weight"):
            self.verify(self.rewrite(omit=n), "missing")
        self.verify(self.rewrite(extra=True), "unexpected")

    def test_duplicate(self):
        data = self.good.read_bytes().replace(b"blk.0.attn_k_norm.weight", b"blk.0.attn_q_norm.weight")
        path = self.path/"duplicate.gguf"
        path.write_bytes(data)
        self.verify(path, "duplicate")

    def test_wrong_expert_extent_and_rank(self):
        for shape in ((3, 4, 16), (3, 64)):
            self.verify(self.rewrite(change=("blk.0.ffn_gate_up_exps.weight", lambda x: x.reshape(shape))), "shape")

    def test_payload_and_folded_norm(self):
        self.verify(self.rewrite(change=("blk.0.ffn_down_exps.weight", lambda x: x ^ 1)), "payload")
        self.verify(self.rewrite(change=("output_norm.weight", lambda x: x+1)), "payload")

    def test_invalid_metadata(self):
        for key, value in (("clef.format_version", 2), ("clef.capabilities", 5),
                           ("clef.context_default", 65537), ("clef.moe.expert_used_count", 3),
                           ("clef.audio.config", "{}"), ("clef.moe.normalize_topk", False),
                           ('clef.vision.video_processor','{}'),('clef.vision.video.max_pixels',1)):
            with self.subTest(key=key):
                self.verify(self.rewrite(meta=(key, value)), key)


if __name__ == "__main__":
    unittest.main()
