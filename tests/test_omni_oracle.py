"""The streamed FP32 adapter must equal a full FP32 forward of BF16 weights."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from transformers import Qwen3OmniMoeTextConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeThinkerTextModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"ref"))
import omni
from omni_weights import WeightPager


class OmniOracle(unittest.TestCase):
    def test_explicit_attention_profiles_match_full_forward_when_streamed(self):
        import safe_attn
        for mode in ('blocked', 'sdpa'):
            with self.subTest(mode=mode), torch.inference_mode():
                model = copy.deepcopy(self.model)
                safe_attn.install(model, omni=True, value_reduction=mode)
                full = copy.deepcopy(model).float()
                expected = [full(**batch, use_cache=False).last_hidden_state for batch in self.inputs]
                actual, _ = omni.stream_text(model, self.inputs)
                for a, b in zip(expected, actual):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)

    def setUp(self):
        torch.manual_seed(71)
        cfg = Qwen3OmniMoeTextConfig(vocab_size=64, hidden_size=32, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=1, head_dim=128, num_experts=4,
            num_experts_per_tok=2, moe_intermediate_size=16, intermediate_size=16,
            rope_scaling=dict(rope_type="default", mrope_interleaved=True, mrope_section=[24, 20, 20]))
        cfg._attn_implementation = "sdpa"
        cfg._experts_implementation = "eager"
        self.model = Qwen3OmniMoeThinkerTextModel(cfg).eval().to(torch.bfloat16)
        self.inputs = [dict(input_ids=torch.randint(0, 64, (1, n)), attention_mask=torch.ones(1, n, dtype=torch.long)) for n in (7, 13)]

    def test_full_and_streamed_exact(self):
        full = copy.deepcopy(self.model).float()
        head = torch.nn.Linear(32, 5).to(torch.bfloat16).float()
        with torch.inference_mode():
            want, dumps = [], []
            for batch in self.inputs:
                with omni.capture_text(full) as dump:
                    want.append(full(**batch, use_cache=False).last_hidden_state)
                dumps.append(dump)
            got, captured = omni.stream_text(self.model, self.inputs)
            for a, b, da, db in zip(want, got, dumps, captured):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
                torch.testing.assert_close(head(a), head(b), rtol=0, atol=0)
                self.assertEqual(da.keys(), db.keys())
                self.assertIn("routing.00.ids", da)
                self.assertIn("routing.01.margin", da)
                for k in da:
                    torch.testing.assert_close(da[k], db[k], rtol=0, atol=0, msg=k)
        self.assertTrue(all(p.dtype == torch.bfloat16 for p in self.model.parameters()))

    def test_disk_paged_execution_matches_resident_weights_exactly(self):
        expected,captured=omni.stream_text(self.model,self.inputs)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            state={'thinker.model.'+k:v.clone() for k,v in self.model.state_dict().items()}
            save_file(state,path/'part.safetensors')
            (path/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{k:'part.safetensors' for k in state}}))
            for name,p in list(self.model.named_parameters()):
                WeightPager.assign(self.model,name,torch.empty_like(p,device='meta'),parameter=True)
            pager=WeightPager(path,device='cpu')
            got,dumps=omni.stream_text(self.model,self.inputs,weight_pager=pager)
            for a,b,da,db in zip(expected,got,captured,dumps):
                torch.testing.assert_close(a,b,rtol=0,atol=0)
                for name in da:torch.testing.assert_close(da[name],db[name],rtol=0,atol=0)
            self.assertEqual(pager.active_bytes,0)
            self.assertTrue(all(p.is_meta for p in self.model.parameters()))

    def test_lossless_decoder_stages_match_unsplit(self):
        expected, dumps = omni.stream_text(self.model, self.inputs)
        state = {}
        omni.stream_text(self.model, self.inputs, stage_state=state, layer_stop=1)
        self.assertEqual(state['next_layer'], 1)
        self.assertNotIn('final_norm', state['dumps'][0])
        # Exercise the actual weights-only checkpoint boundary, including rotary
        # inputs and suffix captures, with a separately constructed next worker.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'state.pt'
            torch.save(state, path)
            restored = torch.load(path, weights_only=True)
        got, captured = omni.stream_text(copy.deepcopy(self.model), self.inputs, stage_state=restored)
        for a, b, da, db in zip(expected, got, dumps, captured):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertEqual(da.keys(), db.keys())
            for key in da:
                torch.testing.assert_close(da[key], db[key], rtol=0, atol=0, msg=key)
        self.assertEqual(restored['next_layer'], 2)
        with self.assertRaisesRegex(ValueError, 'stage'):
            omni.stream_text(self.model, self.inputs, stage_state=restored, layer_stop=1)

    def test_hooks_and_dtype_restored_after_failure(self):
        with self.assertRaisesRegex(RuntimeError, "injected"):
            with patch.object(self.model.layers[1], "forward", side_effect=RuntimeError("injected")):
                omni.stream_text(self.model, self.inputs)
        for m in self.model.modules():
            self.assertEqual(len(m._forward_hooks), 0)
            self.assertEqual(len(m._forward_pre_hooks), 0)
        self.assertTrue(all(p.dtype == torch.bfloat16 for p in self.model.parameters()))

    def test_dump_tail_does_not_change_execution(self):
        with torch.inference_mode():
            a, _ = omni.stream_text(self.model, self.inputs)
            b, captured = omni.stream_text(self.model, self.inputs, dump_last=3)
        for x, y, dump in zip(a, b, captured):
            torch.testing.assert_close(x, y, atol=0, rtol=0)
            self.assertTrue(all(t.shape[0] == 3 for t in dump.values()))

    def test_checked_routes_preserve_matching_execution_and_cleanup(self):
        expected, dumps = omni.stream_text(self.model, self.inputs)
        routes = [torch.stack([d[f'routing.{i:02d}.ids'] for i in range(2)]) for d in dumps]
        events = []
        got, _ = omni.stream_text(self.model, self.inputs, native_routes=routes,
                                   record_ids=['a','b'], arbitration_events=events)
        self.assertEqual(events, [])
        for a,b in zip(expected,got): torch.testing.assert_close(a,b,atol=0,rtol=0)
        with patch('omni_arbitration.arbitrate', side_effect=ValueError('FP64 rejects')):
            with self.assertRaisesRegex(ValueError, 'FP64 rejects'):
                omni.stream_text(self.model, self.inputs, native_routes=routes,
                                  record_ids=['a','b'], arbitration_events=[])
        for module in self.model.modules(): self.assertFalse(module._forward_hooks)
        self.assertTrue(all(p.dtype == torch.bfloat16 for p in self.model.parameters()))

    def test_native_activation_callback_uses_matching_layer_input(self):
        import omni_arbitration
        expected,dumps=omni.stream_text(self.model,self.inputs)
        routes=[torch.stack([d[f'routing.{i:02d}.ids'] for i in range(2)]) for d in dumps]
        activations=[torch.stack([d['embed'],d['layer.00'],d['layer.01']]).numpy() for d in dumps]
        seen=[]
        original=omni_arbitration.arbitrate
        def check(output,native,calculate,events,**kwargs):
            rows=torch.tensor([0,len(output[0])-1])
            high,metrics=kwargs['calculate_native'](rows)
            torch.testing.assert_close(high,calculate(rows),atol=0,rtol=0)
            self.assertEqual(metrics['incoming_relative_l2'],0)
            seen.append((kwargs['record'],kwargs['layer']))
            return original(output,native,calculate,events,**kwargs)
        with patch('omni_arbitration.arbitrate',side_effect=check):
            got,_=omni.stream_text(self.model,self.inputs,native_routes=routes,native_activations=activations,
                                    record_ids=['a','b'],arbitration_events=[])
        self.assertEqual(seen,[('a',0),('b',0),('a',1),('b',1)])
        for a,b in zip(expected,got): torch.testing.assert_close(a,b,atol=0,rtol=0)
        activations[0][0].fill(float('nan'))
        with patch('omni_arbitration.arbitrate',side_effect=check):
            with self.assertRaisesRegex(ValueError,'incoming activation'):
                omni.stream_text(self.model,self.inputs,native_routes=routes,native_activations=activations,
                                 record_ids=['a','b'],arbitration_events=[])
        for module in self.model.modules(): self.assertFalse(module._forward_hooks)
        self.assertTrue(all(p.dtype==torch.bfloat16 for p in self.model.parameters()))

    def test_failed_generation_does_not_publish(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)/"golden"
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                with omni.golden_directory(dest) as out:
                    (out/"encoded.jsonl").write_text("partial")
                    raise RuntimeError("incomplete")
            self.assertFalse(dest.exists())
            with omni.golden_directory(dest) as out:
                (out/"encoded.jsonl").write_text("complete")
            self.assertEqual((dest/"encoded.jsonl").read_text(), "complete")
            with self.assertRaises(FileExistsError):
                with omni.golden_directory(dest):
                    pass

    def test_rounding_callback_receives_native_input_and_rotary_positions(self):
        import omni_arbitration
        expected,dumps=omni.stream_text(self.model,self.inputs)
        routes=[torch.stack([d[f'routing.{i:02d}.ids'] for i in range(2)]) for d in dumps]
        activations=[torch.stack([d['embed'],d['layer.00'],d['layer.01']]).numpy() for d in dumps]
        seen=[]
        def audit(record,number,layer,incoming,pe,rows,positions):
            index=['a','b'].index(record)
            self.assertIs(layer,self.model.layers[number])
            self.assertTrue((incoming==activations[index][number]).all())
            torch.testing.assert_close(positions.cpu(),torch.arange(len(incoming)).view(1,1,-1).expand(3,1,-1))
            seen.append((record,number))
            return omni_arbitration.fp64_router(layer,torch.from_numpy(incoming)[None],pe,rows),[{} for _ in rows]
        original=omni_arbitration.arbitrate
        def check(output,native,calculate,events,**kwargs):
            rows=torch.tensor([0,len(output[0])-1])
            high,_=kwargs['calculate_rounding'](rows)
            torch.testing.assert_close(high,calculate(rows),atol=0,rtol=0)
            return original(output,native,calculate,events,**kwargs)
        with patch('omni_arbitration.arbitrate',side_effect=check):
            got,_=omni.stream_text(self.model,self.inputs,native_routes=routes,native_activations=activations,
                                  record_ids=['a','b'],arbitration_events=[],rounding_auditor=audit)
        self.assertEqual(seen,[('a',0),('b',0),('a',1),('b',1)])
        for a,b in zip(expected,got): torch.testing.assert_close(a,b,atol=0,rtol=0)


if __name__ == "__main__":
    unittest.main()
