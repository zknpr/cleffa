"""Native family/configuration validation against tiny header-only GGUF fixtures."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import gguf

from omni_fixture import snapshot

ROOT = Path(__file__).resolve().parents[1]


class OmniConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.path = Path(cls.tmp.name)
        config, _ = snapshot(cls.path)
        config["thinker_config"]["text_config"].update(hidden_size=2048, num_hidden_layers=48,
            num_attention_heads=32, num_key_value_heads=4, moe_intermediate_size=768, intermediate_size=768,
            num_experts=128, num_experts_per_tok=8, vocab_size=152064)
        config['thinker_config']['vision_config'].update(out_hidden_size=2048,depth=27,hidden_size=1152,
            intermediate_size=4304,num_heads=16,patch_size=16,image_size=768,deepstack_visual_indexes=[8,16,24])
        proc=json.loads((cls.path/'processor_config.json').read_text())
        for key in ('image_processor','video_processor'): proc[key]['patch_size']=16
        (cls.path/'processor_config.json').write_text(json.dumps(proc))
        config['thinker_config']['audio_config'].update(output_dim=2048,d_model=1280,encoder_layers=32,
            encoder_attention_heads=20,encoder_ffn_dim=5120,downsample_hidden_size=480,max_source_positions=1500)
        proc['feature_extractor'].update(chunk_length=30,n_samples=480000,nb_max_frames=3000,
                                        padding_side='right',return_attention_mask=True,padding_value=0.)
        (cls.path/'processor_config.json').write_text(json.dumps(proc))
        (cls.path/"config.json").write_text(json.dumps(config))
        (cls.path/"joint_head_config.json").write_text(json.dumps(dict(hidden_size=2048, width=1024,
            routing_layers=2, layers=4, heads=16, feedforward=4096)))
        p = subprocess.run([sys.executable, "-B", ROOT/"tools/convert.py", cls.path, cls.path/"vocab.gguf",
                            "--modalities", "text", "--vocab-only"], capture_output=True)
        if p.returncode:
            raise AssertionError(p.stderr.decode())
        cls.fields = gguf.GGUFReader(cls.path/"vocab.gguf").fields

    def run_header(self, expected="ok", override=None, drop=()):
        w = gguf.GGUFWriter(self.path/"test.gguf", "clef")
        for key, f in self.fields.items():
            if not key.startswith("clef.") or key in drop:
                continue
            val = override.get(key, f.contents()) if override else f.contents()
            w.add_key_value(key, val, f.types[0], f.types[-1] if len(f.types)>1 else None)
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.close()
        p = subprocess.run([ROOT/"tests/test-omni-config", self.path/"test.gguf", expected], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)

    def test_valid(self):
        self.run_header()

    def test_family_metadata(self):
        for key, val in (("clef.backbone", "unknown"), ("clef.format_version", 2), ("clef.capabilities", 5),
                          ("clef.context_default", 65537), ("clef.context_length", 0)):
            with self.subTest(key=key):
                self.run_header("model:", {key: val})
        for key in ("clef.backbone", "clef.format_version", "clef.capabilities", "clef.context_default", "clef.context_length"):
            self.run_header("model:", drop=(key,))

    def test_invalid_dimensions(self):
        for key in ("clef.embedding_length", "clef.moe.expert_count", "clef.moe.expert_used_count", "clef.attention.head_count"):
            for value in (0, 1 << 24, (1 << 32)-1):
                self.run_header("model:", {key: value})
        for key, value in (("clef.vision.image_token_id", 152064), ("clef.audio.audio_token_id", (1 << 32)-1),
                           ("clef.moe.normalize_topk", False), ("clef.attention.value_length", 256),
                           ("clef.rope.mrope_section", [20, 24, 20]),
                           ("clef.attention.layer_norm_rms_epsilon", float("nan"))):
            self.run_header("model:", {key: value})

    def test_no_implicit_legacy(self):
        self.run_header("model:", drop=("clef.backbone", "clef.format_version", "clef.capabilities",
                                       "clef.context_default", "clef.context_length"))

    def test_full_media_config(self):
        self.run_header(override={"clef.capabilities": 7})

    def test_audio_metadata(self):
        self.run_header('audio:ok')
        for key in ('clef.audio.config','clef.audio.processor'):
            original=json.loads(self.fields[key].contents())
            for field in ('d_model','n_window','encoder_attention_heads') if key.endswith('config') else ('feature_size','n_fft','hop_length'):
                for value in (0,1<<31,True,3.5):
                    changed={**original,field:value}
                    self.run_header('audio:unsupported', {key:json.dumps(changed)})
            self.run_header('audio:unsupported',{key:'{}'})
            self.run_header('audio:missing',drop=(key,))

    def test_vision_geometry_and_processor(self):
        self.run_header(override={'clef.capabilities':3})
        for key,value in (('clef.vision.deepstack_layers',[8,16,25]),('clef.vision.embedding_length',2048),
                          ('clef.vision.block_count',24),('clef.vision.position_embeddings',4096),
                          ('clef.vision.video.min_pixels',1),('clef.vision.video.max_pixels',1),
                          ('clef.vision.video_processor','{}')):
            self.run_header('model:',{'clef.capabilities':3,key:value})
        for key in ('clef.vision.block_count','clef.vision.deepstack_layers','clef.vision.video_processor'):
            self.run_header('model:',{'clef.capabilities':3},drop=(key,))

    def test_rank_checks(self):
        p = subprocess.run([ROOT/"tests/test-omni-config", "rank"], capture_output=True)
        self.assertEqual(p.returncode, 0, p.stderr)

    def test_audio_never_silently_ignored(self):
        p = subprocess.run([ROOT/"tests/test-omni-config", "audio"], capture_output=True)
        self.assertEqual(p.returncode, 0, p.stderr)


if __name__ == "__main__":
    subprocess.run(["make", "tests/test-omni-config"], cwd=ROOT, check=True)
    unittest.main()
