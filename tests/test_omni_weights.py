"""Disk paging preserves BF16-source arithmetic and releases weights on failures."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from safetensors.torch import save_file

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ref'))
from omni_weights import WeightPager


class Paging(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)
        self.weights={'layer.weight':torch.arange(12,dtype=torch.float32).reshape(3,4)/17,
                      'layer.bias':torch.tensor([.2,.3,.4])}
        save_file(self.weights,self.path/'part.safetensors')
        (self.path/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{k:'part.safetensors' for k in self.weights}}))
        self.model=torch.nn.Linear(4,3,device='meta')
        self.model.register_buffer('position',torch.arange(3,dtype=torch.float32),persistent=False)
        self.pager=WeightPager(self.path,device='cpu',max_weight_bytes=60)

    def test_exact_bf16_source_then_fp32_and_release(self):
        with self.pager.materialize(self.model,'layer'):
            x=torch.arange(8,dtype=torch.float32).reshape(2,4)
            want=torch.nn.functional.linear(x,self.weights['layer.weight'].bfloat16().float(),self.weights['layer.bias'].bfloat16().float())
            torch.testing.assert_close(self.model(x),want,rtol=0,atol=0)
            self.assertEqual(self.pager.active_bytes,60)
        self.assertTrue(all(p.is_meta for p in self.model.parameters()))
        self.assertEqual(self.pager.active_bytes,0)
        self.assertEqual(self.pager.peak_weight_bytes,60)
        torch.testing.assert_close(self.model.position,torch.arange(3,dtype=torch.float32),rtol=0,atol=0)

    def test_budget_rejects_before_materialization(self):
        self.pager.max_weight_bytes=59
        with self.assertRaises(MemoryError):
            with self.pager.materialize(self.model,'layer'):self.fail('budget bypassed')
        self.assertTrue(all(p.is_meta for p in self.model.parameters()))

    def test_checkpoint_experts_stack_numerically_and_gate_precedes_up(self):
        weights={f'layer.mlp.experts.{e}.{kind}_proj.weight':
                 torch.full((3,4) if kind!='down' else (4,3),e*10+k+.13)
                 for e in range(12) for k,kind in enumerate(('gate','up','down'))}
        save_file(weights,self.path/'part.safetensors')
        (self.path/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{k:'part.safetensors' for k in weights}}))
        model=torch.nn.Module();model.mlp=torch.nn.Module();model.mlp.experts=torch.nn.Module()
        model.mlp.experts.gate_up_proj=torch.nn.Parameter(torch.empty(12,6,4,device='meta'))
        model.mlp.experts.down_proj=torch.nn.Parameter(torch.empty(12,4,3,device='meta'))
        pager=WeightPager(self.path,device='cpu',max_weight_bytes=2000)
        with pager.materialize(model,'layer'):
            for e in range(12):
                expected=torch.cat([weights[f'layer.mlp.experts.{e}.{k}_proj.weight'] for k in ('gate','up')]).bfloat16().float()
                torch.testing.assert_close(model.mlp.experts.gate_up_proj[e],expected,rtol=0,atol=0)
                torch.testing.assert_close(model.mlp.experts.down_proj[e],weights[f'layer.mlp.experts.{e}.down_proj.weight'].bfloat16().float(),rtol=0,atol=0)
        self.assertTrue(all(p.is_meta for p in model.parameters()))

    def test_shape_mismatch_and_failure_release(self):
        self.weights['layer.bias']=torch.ones(4)
        save_file(self.weights,self.path/'part.safetensors')
        with self.assertRaisesRegex(ValueError,'mismatched'):
            with self.pager.materialize(self.model,'layer'):pass
        self.weights['layer.bias']=torch.ones(3);save_file(self.weights,self.path/'part.safetensors')
        with self.assertRaisesRegex(RuntimeError,'injected'):
            with self.pager.materialize(self.model,'layer'):raise RuntimeError('injected')
        self.assertTrue(all(p.is_meta for p in self.model.parameters()))
        self.assertEqual(self.pager.active_bytes,0)


if __name__=='__main__':unittest.main()
