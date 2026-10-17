"""Independent FP64 arithmetic must preserve precision and bounded attention."""
from contextlib import nullcontext
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ref'))
import omni_fp64 as fp64


class Arithmetic(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(811)
        torch.set_num_threads(2)

    def test_online_attention_matches_dense_grouped_causal_and_noncausal(self):
        q = torch.randn(6, 19, 8, dtype=torch.float64)
        k = torch.randn(2, 19, 8, dtype=torch.float64)
        v = torch.randn_like(k)
        for causal in (False, True):
            scores = q @ k.repeat_interleave(3, 0).transpose(-1, -2) / 8**.5
            if causal:
                scores.masked_fill_(torch.ones(19, 19, dtype=torch.bool).triu(1), -torch.inf)
            expected = scores.softmax(-1) @ v.repeat_interleave(3, 0)
            for qb, kb in ((1, 1), (4, 7), (32, 32)):
                got = fp64.attention(q, k, v, causal=causal, query_block=qb, key_block=kb)
                torch.testing.assert_close(got, expected, rtol=2e-14, atol=2e-14)

    def test_attention_bounds_score_allocation_and_handles_large_scores(self):
        q = torch.randn(4, 53, 8, dtype=torch.float64) * 100
        k = torch.randn(2, 53, 8, dtype=torch.float64) * 100
        v = torch.randn_like(k)
        seen = []
        got = fp64.attention(q, k, v, query_block=7, key_block=11, observe_scores=seen.append)
        self.assertTrue(torch.isfinite(got).all())
        self.assertTrue(seen)
        self.assertLessEqual(max(shape[-2] for shape in seen), 7)
        self.assertLessEqual(max(shape[-1] for shape in seen), 11)
        with self.assertRaises(ValueError): fp64.attention(q.float(), k, v)
        with self.assertRaises(ValueError): fp64.attention(q, k, v, key_block=0)

    def test_interleaved_rotary_uses_all_three_axes_in_double(self):
        pos = torch.tensor([[0, 1, 80001], [3, 4, 90003], [6, 7, 100007]])
        with fp64.DoubleOnly():
            co, si = fp64.rotary(pos, 128, 1e6, (24, 20, 20))
        for column in range(64):
            axis = column % 3 if column < 60 else 0
            angle = pos[axis].double() / (1e6 ** (column / 64))
            torch.testing.assert_close(co[:, column], angle.cos(), rtol=2e-11, atol=2e-11)
            torch.testing.assert_close(si[:, column], angle.sin(), rtol=2e-11, atol=2e-11)
        torch.testing.assert_close(co[:, :64], co[:, 64:], rtol=0, atol=0)
        self.assertEqual(co.dtype, torch.float64)

    def test_router_keeps_sub_fp32_differences_without_external_choices(self):
        x = torch.eye(3, dtype=torch.float64)
        w = torch.tensor([[1, 0, 2], [1 + 1e-9, 0, -2], [0, 1, 1]], dtype=torch.float64)
        logits, weights, ids = fp64.route(x, w, 2)
        self.assertEqual(ids[0].tolist(), [1, 0])
        torch.testing.assert_close(logits, x @ w.T, rtol=0, atol=0)
        expected = logits.softmax(-1).gather(1, ids)
        expected /= expected.sum(-1, keepdim=True)
        torch.testing.assert_close(weights, expected, rtol=0, atol=0)

    def test_precision_audit_rejects_internal_downcast_even_if_output_is_double(self):
        with fp64.DoubleOnly():
            self.assertEqual((torch.ones(3, dtype=torch.float64) * 2).dtype, torch.float64)
        with self.assertRaisesRegex(ValueError, 'non-FP64'):
            with fp64.DoubleOnly():
                torch.ones(3, dtype=torch.float64).float().double()

    def test_bilinear_vision_positions_are_fp64_and_preserve_merge_order(self):
        from transformers.vision_utils import get_vision_position_ids
        grid = torch.tensor([[2, 4, 6], [1, 2, 2]])
        positions = get_vision_position_ids(grid, 2)
        table = torch.arange(16, dtype=torch.float64).reshape(16, 1)
        got = fp64.vision_positions(grid, table, 2)
        expected = []
        start = 0
        for t, h, w in grid.tolist():
            for y, x in positions[start:start+t*h*w].tolist():
                expected.append([3 * y/(h-1)*4 + 3*x/(w-1)])
            start += t*h*w
        torch.testing.assert_close(got, torch.tensor(expected, dtype=torch.float64), rtol=1e-15, atol=1e-15)


class Towers(unittest.TestCase):
    def setUp(self):
        from transformers import Qwen3OmniMoeThinkerConfig
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeThinkerForConditionalGeneration
        torch.manual_seed(193)
        cfg = Qwen3OmniMoeThinkerConfig(
            text_config=dict(vocab_size=32, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                             num_key_value_heads=1, head_dim=8, num_experts=4, num_experts_per_tok=2,
                             moe_intermediate_size=12, rope_scaling=dict(rope_type='default', mrope_section=[2,1,1])),
            vision_config=dict(depth=3, hidden_size=32, num_heads=4, intermediate_size=48,
                               patch_size=2, temporal_patch_size=2, spatial_merge_size=2, out_hidden_size=16,
                               num_position_embeddings=16, deepstack_visual_indexes=[0,1,2]),
            audio_config=dict(num_mel_bins=8, encoder_layers=2, encoder_attention_heads=2, encoder_ffn_dim=24,
                              d_model=16, output_dim=16, downsample_hidden_size=4, n_window=50, n_window_infer=800))
        cfg._attn_implementation = 'sdpa'
        cfg.text_config._experts_implementation = 'eager'
        self.model = Qwen3OmniMoeThinkerForConditionalGeneration(cfg).eval().double()
        self.pager = type('Resident', (), {'materialize': lambda _, *args, **kwargs: nullcontext()})()

    def test_decoder_attention_matches_dense_sdpa_with_qk_norm(self):
        layer = self.model.model.layers[0]
        hidden = torch.randn(17, 16, dtype=torch.float64)
        positions = torch.arange(17).expand(3, -1)
        pe = fp64.rotary(positions, 8, 1e6, (2, 1, 1))
        a = layer.self_attn
        with torch.inference_mode():
            x = fp64.rms(hidden, layer.input_layernorm)
            q = fp64.rms(a.q_proj(x).reshape(17, 2, 8), a.q_norm).transpose(0, 1)
            k = fp64.rms(a.k_proj(x).reshape(17, 1, 8), a.k_norm).transpose(0, 1)
            v = a.v_proj(x).reshape(17, 1, 8).transpose(0, 1)
            q, k = fp64.rotate(q, pe), fp64.rotate(k, pe)
            dense = torch.nn.functional.scaled_dot_product_attention(q, k.expand(2, -1, -1), v.expand(2, -1, -1), is_causal=True)
            expected = hidden+a.o_proj(dense.transpose(0, 1).reshape(17, 16))
            actual = fp64.text_attention(layer, hidden, pe)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_vision_matches_independent_dense_attention_and_emits_double(self):
        from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as hf
        tower = self.model.visual
        grid = torch.tensor([[2, 4, 6], [1, 2, 2]])
        pixels = torch.randn(52, 24, dtype=torch.float64)
        def rotate(q, k, cos, sin):
            def one(x):
                return x*cos[:, None] + torch.cat((-x[..., 4:], x[..., :4]), -1)*sin[:, None]
            return one(q), one(k)
        # Only replace the pinned routine's explicit float32 rotary cast and
        # frequency buffer; its dense SDPA remains a separate implementation.
        tower.rotary_pos_emb.inv_freq = 10000. ** (-torch.arange(0, 4, 2, dtype=torch.float64)/4)
        with torch.inference_mode(), patch.object(hf, 'apply_rotary_pos_emb_vision', rotate):
            # Supply independent dense positional interpolation in double.
            from transformers.vision_utils import get_vision_bilinear_indices_and_weights
            old = torch.get_default_dtype()
            torch.set_default_dtype(torch.float64)
            try:
                idx, coeff = get_vision_bilinear_indices_and_weights(grid, 4, 2)
            finally:
                torch.set_default_dtype(old)
            expected = tower(pixels, grid, bilinear_indices=idx, bilinear_weights=coeff)
        with torch.inference_mode():
            got, extras = fp64.vision(tower, self.pager, pixels, grid)
        torch.testing.assert_close(got, expected.pooler_output, rtol=1e-11, atol=1e-12)
        for actual, wanted in zip(extras, expected.deepstack_features):
            torch.testing.assert_close(actual, wanted, rtol=1e-11, atol=1e-12)
            self.assertEqual(actual.dtype, torch.float64)
        self.assertEqual(len(extras), 3)

    def test_audio_matches_dense_tower_for_multiple_clips_and_partial_chunks(self):
        tower = self.model.audio_tower
        lengths = torch.tensor([1, 99, 101, 900])
        features = torch.randn(4, 8, 900, dtype=torch.float64)
        table = fp64.sinusoids(tower.max_source_positions, tower.config.d_model)
        tower.positional_embedding.positional_embedding = table
        packed = torch.cat([f[:, :n] for f, n in zip(features, lengths)], -1)
        with torch.inference_mode():
            expected = tower(packed, lengths).last_hidden_state
            got, dumps = fp64.audio(tower, self.pager, features, lengths)
        torch.testing.assert_close(got, expected, rtol=1e-10, atol=1e-12)
        for i, n in enumerate(lengths.tolist()):
            self.assertEqual(len(dumps[f'audio.{i}.output']), 13*(n//100)+(n%100+7)//8)
        for name, value in dumps.items():
            self.assertEqual(value.dtype, torch.float64, name)


class Evidence(unittest.TestCase):
    def provenance(self, policy):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        for name in ('joint_schema_model.py','config.json','joint_head_config.json'):
            (root/name).write_text('unit fixture only')
        return policy.runtime_provenance(root,'text')

    def test_policy_rejects_arbitration_partial_precision_and_source_changes(self):
        import omni_fp64_policy as policy
        good = dict(complete=True, revision=policy.REVISION, precision='float64-streamed',
                    independent_from_native=True, fp64_policy=copy.deepcopy(policy.POLICY),
                    implementation_sha256=policy.implementation_sources(),corpus='text',**self.provenance(policy))
        policy.validate_policy(good)
        changes = [('complete', False), ('precision', 'float32-streamed'), ('arbitration', {}),
                   ('independent_from_native', False), ('implementation_sha256', {})]
        for key, value in changes:
            with self.subTest(key=key), self.assertRaises(ValueError):
                policy.validate_policy({**good, key: value})
        for key in ('source_sha256','torch','transformers','snapshot_path'):
            changed=copy.deepcopy(good);changed.pop(key)
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'provenance'):
                policy.validate_policy(changed)
        model_source=Path(good['snapshot_path'])/'config.json'
        model_source.write_text('changed source')
        with self.assertRaisesRegex(ValueError,'provenance'):policy.validate_policy(good)
        changed = copy.deepcopy(good)
        changed['fp64_policy']['inference'] = 'fp64-decoder-fp32-media'
        with self.assertRaises(ValueError): policy.validate_policy(changed)

    def test_missing_runtime_provenance_cannot_qualify(self):
        import omni_fp64_policy as policy
        manifest = dict(complete=True, revision=policy.REVISION, precision='float64-streamed',
                        independent_from_native=True, fp64_policy=policy.POLICY,
                        implementation_sha256=policy.implementation_sources())
        with self.assertRaisesRegex(ValueError, 'provenance'):
            policy.validate_policy(manifest)

    def test_capture_streams_stages_and_refuses_nonfinite_or_wrong_precision(self):
        from oracle_fp64 import Capture
        from safetensors.torch import load_file
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sink = Capture(root, 'r000', 6, 2)
            value = torch.arange(18, dtype=torch.float64).reshape(6, 3)
            sink.put('embed', value)
            self.assertTrue((root/'parts/r000/embed.safetensors').is_file())
            with self.assertRaises(ValueError): sink.put('embed', value)
            with self.assertRaises(ValueError): sink.put('layer.00', value.float())
            with self.assertRaises(ValueError): sink.put('layer.00', value*torch.inf)
            sink.finish(root/'record.safetensors')
            torch.testing.assert_close(load_file(root/'record.safetensors')['embed'], value[-2:], rtol=0, atol=0)

    def test_layer_comparison_requires_explicit_fp64_lane(self):
        import omni_fp64_policy as v1
        import oracle_fp64_bounded as v2
        for policy in (v1, v2):
            with self.subTest(policy=policy.POLICY['name']):
                self.check_layer_comparison(policy)

    def check_layer_comparison(self, policy):
        from compare_omni_layers import compare
        from safetensors.numpy import save_file
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); golden = root/'golden'; capture = root/'native'
            (golden/'layers').mkdir(parents=True); capture.mkdir()
            records = [dict(id='r000', tokens=3, dump_start=0, dumped=True)]
            meta = dict(complete=True, revision=policy.REVISION, precision='float64-streamed',
                        independent_from_native=True, fp64_policy=policy.POLICY,
                        implementation_sha256=policy.implementation_sources(), records=records, corpus_sha256='same',
                        corpus='text',**self.provenance(policy))
            (golden/'manifest.json').write_text(json.dumps(meta))
            (capture/'manifest.json').write_text(json.dumps(dict(complete=True, records=records, requests_sha256='same')))
            arrays = np.ones((3, 3, 2), dtype='f4')
            arrays.tofile(capture/'r000.bin')
            save_file({key: arrays[i].astype('f8') for i, key in enumerate(('embed', 'layer.00', 'final_norm'))}, golden/'layers/r000.safetensors')
            (golden/'requests.jsonl').write_text('{"id":"r000"}\n')
            (golden/'encoded.jsonl').write_text('{"id":"r000","input_ids":[1,2,3]}\n')
            save_file({'r000/q':np.ones(2,dtype='f8')},golden/'logits.safetensors')
            def bind():
                meta['artifacts_sha256']={name:policy.digest(golden/name) for name in
                    ('requests.jsonl','encoded.jsonl','logits.safetensors','layers/r000.safetensors')}
                (golden/'manifest.json').write_text(json.dumps(meta))
            bind()
            with self.assertRaises(ValueError): compare(golden, capture, layers=1)
            result = compare(golden, capture, layers=1, reference='fp64')
            self.assertTrue(result['complete'])
            arrays[2] *= 1.02
            arrays.tofile(capture/'r000.bin')
            self.assertFalse(compare(golden, capture, layers=1, reference='fp64')['complete'])
            # Finite FP64 values can overflow a squared norm; NaN must never
            # evade the unchanged numerical gate through a false comparison.
            save_file({key: np.full((3,2),1e308,dtype='f8') for key in ('embed','layer.00','final_norm')}, golden/'layers/r000.safetensors')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                compare(golden,capture,layers=1,reference='fp64')
            bind()
            with self.assertRaisesRegex(ValueError,'nonfinite.*metric'):
                compare(golden,capture,layers=1,reference='fp64')


if __name__ == '__main__':
    unittest.main()
