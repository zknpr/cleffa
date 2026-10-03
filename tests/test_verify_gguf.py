"""The model verifier must reject missing tensors and equal-size wrong shapes.

Tiny safetensors/GGUF fixtures exercise both backbone layer kinds without model downloads.
Run with .venv/bin/python tests/test_verify_gguf.py.
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import gguf
import numpy as np
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parent.parent
P = "model.language_model."


class VerifyGguf(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="clef-verify-")
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.hf = Path(cls.tmp.name)
        source, cls.tensors = {}, {}

        def weight(name, shape):
            t = torch.arange(int(np.prod(shape)), dtype=torch.float32).reshape(shape).to(torch.bfloat16)
            source[name] = t
            return t

        cls.tensors["token_embd.weight"] = weight(P + "embed_tokens.weight", (2, 3))
        cls.tensors["output.weight"] = weight("lm_head.weight", (2, 3))
        cls.tensors["output_norm.weight"] = 1 + weight(P + "norm.weight", (3,)).float()
        for i, kind in enumerate(("full_attention", "linear_attention")):
            p, b = f"{P}layers.{i}.", f"blk.{i}."
            for src, dst in (("input_layernorm", "attn_norm"), ("post_attention_layernorm", "ffn_norm")):
                cls.tensors[b + dst + ".weight"] = 1 + weight(p + src + ".weight", (3,)).float()
            cls.tensors[b + "ffn_gate_up.weight"] = torch.cat([
                weight(p + "mlp.gate_proj.weight", (2, 3)), weight(p + "mlp.up_proj.weight", (2, 3))])
            cls.tensors[b + "ffn_down.weight"] = weight(p + "mlp.down_proj.weight", (3, 2))
            if kind == "full_attention":
                cls.tensors[b + "attn_qkv.weight"] = torch.cat([
                    weight(p + f"self_attn.{proj}_proj.weight", (1, 3)) for proj in ("q", "k", "v")])
                cls.tensors[b + "attn_output.weight"] = weight(p + "self_attn.o_proj.weight", (3, 1))
                for proj in ("q", "k"):
                    cls.tensors[b + f"attn_{proj}_norm.weight"] = 1 + weight(p + f"self_attn.{proj}_norm.weight", (1,)).float()
            else:
                p += "linear_attn."
                cls.tensors[b + "ssm_in.weight"] = torch.cat([
                    weight(p + "in_proj_qkv.weight", (3, 3)),
                    *[weight(p + f"in_proj_{proj}.weight", (1, 3)) for proj in ("z", "b", "a")]])
                cls.tensors[b + "ssm_out.weight"] = weight(p + "out_proj.weight", (3, 1))
                cls.tensors[b + "ssm_conv1d.weight"] = weight(p + "conv1d.weight", (3, 1, 2)).float().reshape(3, 2)
                cls.tensors[b + "ssm_dt.bias"] = weight(p + "dt_bias", (1,)).float()
                cls.tensors[b + "ssm_a"] = -weight(p + "A_log", (1,)).float().exp()
                cls.tensors[b + "ssm_norm.weight"] = weight(p + "norm.weight", (1,)).float()
        head = {"projection.weight": torch.ones((2, 3), dtype=torch.bfloat16),
                "scale": torch.tensor(1.0, dtype=torch.bfloat16)}
        cls.tensors.update({"head." + k: v.reshape(v.shape or (1,)) for k, v in head.items()})
        save_file(source, cls.hf / "weights.safetensors")
        save_file(head, cls.hf / "joint_head.safetensors")
        (cls.hf / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {k: "weights.safetensors" for k in source}}))
        (cls.hf / "config.json").write_text(json.dumps({"text_config": {
            "num_hidden_layers": 2, "layer_types": ["full_attention", "linear_attention"],
            "linear_conv_kernel_dim": 2}}))

    def verify(self, tensors):
        path = self.hf / "test.gguf"
        writer = gguf.GGUFWriter(path, "clef")
        for name, tensor in tensors.items():
            if tensor.dtype == torch.bfloat16:
                raw = tensor.contiguous().view(torch.int16).numpy().view(np.uint16)
                writer.add_tensor(name, raw, raw_dtype=gguf.GGMLQuantizationType.BF16)
            else:
                writer.add_tensor(name, tensor.numpy())
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        return subprocess.run([sys.executable, ROOT / "tests/verify_gguf.py", self.hf, path],
                              capture_output=True, text=True, timeout=30)

    def test_complete_export_passes(self):
        p = self.verify(self.tensors)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)

    def test_empty_inventory_fails(self):
        p = self.verify({})
        self.assertNotEqual(p.returncode, 0, p.stdout)
        self.assertIn("missing", (p.stdout + p.stderr).lower())

    def test_missing_tensors_fail(self):
        for name in ("output.weight", "blk.0.attn_norm.weight", "blk.0.attn_qkv.weight",
                     "blk.1.ssm_conv1d.weight", "head.projection.weight"):
            with self.subTest(tensor=name):
                p = self.verify({k: v for k, v in self.tensors.items() if k != name})
                self.assertNotEqual(p.returncode, 0, p.stdout)
                self.assertIn(name, p.stdout + p.stderr)

    def test_equal_size_wrong_shapes_fail(self):
        for name in ("token_embd.weight", "head.projection.weight"):
            with self.subTest(tensor=name):
                p = self.verify({**self.tensors, name: self.tensors[name].reshape(3, 2)})
                self.assertNotEqual(p.returncode, 0, p.stdout)
                self.assertIn(name, p.stdout + p.stderr)

    def test_unknown_tensor_fails_with_diagnostic(self):
        p = self.verify({**self.tensors, "unexpected.weight": torch.zeros(1)})
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("unexpected.weight", p.stdout + p.stderr)
        self.assertNotIn("Traceback", p.stderr)

    def test_changed_value_fails(self):
        p = self.verify({**self.tensors, "output.weight": self.tensors["output.weight"] + 1})
        self.assertNotEqual(p.returncode, 0)


if __name__ == "__main__":
    unittest.main()
