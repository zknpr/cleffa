"""Omni exports preserve source precision and explicit modality inventories."""

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

from omni_fixture import snapshot

ROOT = Path(__file__).resolve().parents[1]


class ConvertOmni(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.config, self.weights = snapshot(self.path)

    def convert(self, *args, success=True):
        p = subprocess.run([sys.executable, "-B", str(ROOT/"tools/convert.py"),
                            str(self.path), str(self.path/"out.gguf"), *args],
                           capture_output=True, text=True)
        if success:
            self.assertEqual(p.returncode, 0, p.stderr)
            return gguf.GGUFReader(self.path/"out.gguf")
        self.assertNotEqual(p.returncode, 0)
        return p.stderr

    def test_all_payloads_and_plain_norms(self):
        r = self.convert()
        tensors = {t.name: t for t in r.tensors}
        norm = tensors["output_norm.weight"]
        np.testing.assert_array_equal(norm.data, self.weights["thinker.model.norm.weight"].float())
        t = tensors["blk.0.ffn_gate_up_exps.weight"]
        self.assertEqual(tuple(t.shape), (8, 8, 3))
        data = t.data.reshape(-1).view(np.uint16).reshape(3, 8, 8)
        for e in range(3):
            p = f"thinker.model.layers.0.mlp.experts.{e}."
            want = torch.cat([self.weights[p+x+".weight"] for x in ("gate_proj", "up_proj")])
            np.testing.assert_array_equal(data[e], want.view(torch.int16).numpy().view(np.uint16))
        for k in range(3):
            np.testing.assert_array_equal(tensors[f"v.deepstack.{k}.post_ln.weight"].data,
                self.weights[f"thinker.visual.merger_list.{k}.ln_q.weight"].float())
        self.assertEqual(tensors["a.conv2d1.weight"].tensor_type, gguf.GGMLQuantizationType.BF16)
        self.assertEqual(tensors["a.layers.0.self_attn_layer_norm.weight"].tensor_type, gguf.GGMLQuantizationType.F32)
        self.assertFalse(any(n.startswith(("talker.", "code2wav.")) for n in tensors))
        for n, value in (("clef.backbone", "qwen3_omni_moe"), ("clef.format_version", 1),
                         ("clef.capabilities", 7), ("clef.context_default", 64000),
                         ("clef.context_length", 65536), ("clef.moe.normalize_topk", True)):
            self.assertEqual(r.fields[n].contents(), value)
        self.assertEqual(r.fields["clef.vision.deepstack_layers"].types[-1], gguf.GGUFValueType.INT32)

    def test_partial_exports(self):
        for mode, caps in (("text", 1), ("vision", 3)):
            with self.subTest(mode=mode):
                r = self.convert("--modalities", mode)
                self.assertEqual(r.fields["clef.capabilities"].contents(), caps)
                self.assertFalse(any(t.name.startswith("a.") for t in r.tensors))
                self.assertEqual(any(t.name.startswith("v.") for t in r.tensors), mode == "vision")

    def test_vocab_only(self):
        (self.path/"model.safetensors.index.json").unlink()
        r = self.convert("--vocab-only")
        self.assertEqual(len(r.tensors), 0)
        self.assertIn("tokenizer.ggml.tokens", r.fields)

    def test_missing_expert(self):
        idx = json.loads((self.path/"model.safetensors.index.json").read_text())
        del idx["weight_map"]["thinker.model.layers.0.mlp.experts.1.up_proj.weight"]
        (self.path/"model.safetensors.index.json").write_text(json.dumps(idx))
        self.assertIn("experts.1.up_proj.weight", self.convert(success=False))

    def test_wrong_matrix_dtype(self):
        self.weights["thinker.model.layers.0.mlp.gate.weight"] = self.weights["thinker.model.layers.0.mlp.gate.weight"].float()
        save_file(self.weights, self.path/"weights.safetensors")
        self.assertIn("BF16", self.convert(success=False))

    def test_unknown_thinker_tensor(self):
        idx = json.loads((self.path/"model.safetensors.index.json").read_text())
        idx["weight_map"]["thinker.unsupported.weight"] = "weights.safetensors"
        (self.path/"model.safetensors.index.json").write_text(json.dumps(idx))
        self.assertIn("unrecognized", self.convert(success=False))

    def test_incompatible_config(self):
        self.config["thinker_config"]["text_config"]["norm_topk_prob"] = False
        (self.path/"config.json").write_text(json.dumps(self.config))
        self.assertIn("norm_topk_prob", self.convert(success=False))

    def test_video_processor_is_checked_and_recorded(self):
        p=self.path/'processor_config.json'
        proc=json.loads(p.read_text())
        r=self.convert()
        self.assertEqual(json.loads(r.fields['clef.vision.video_processor'].contents()),proc['video_processor'])
        for key,value in (('do_sample_frames',True),('resample',2),('image_mean',[0.,0.,0.]),
                           ('size',dict(shortest_edge=1,longest_edge=2))):
            bad=json.loads(json.dumps(proc));bad['video_processor'][key]=value
            p.write_text(json.dumps(bad))
            self.assertIn('video processor',self.convert(success=False))


if __name__ == "__main__":
    unittest.main()
