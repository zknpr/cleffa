"""FP64 routing arbitration must be independent and refuse clear disagreements."""
import copy
from pathlib import Path
import sys
import types
import unittest
import json
import tempfile

import torch
from transformers import Qwen3OmniMoeTextConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeThinkerTextDecoderLayer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'ref'))
import omni_arbitration as arb
from omni_rounding import audit_stages, native_positions, write_audit_report
MPS = '--mps' in sys.argv
if MPS: sys.argv.remove('--mps')


class Arbitration(unittest.TestCase):
    def test_long_audit_row_sets_have_bounded_distinct_report_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = torch.arange(10000, 11000)
            second = first + 1
            paths = []
            for rows in (first, second):
                report = {'rows': rows.tolist(), 'stages': {'router': {'max_absolute': 0.0}}}
                path = write_audit_report(tmp, rows, report)
                self.assertLessEqual(len(path.name.encode()), 255)
                self.assertEqual(json.loads(path.read_text()), report)
                paths.append(path)
            self.assertNotEqual(*paths)
            self.assertEqual(json.loads(paths[0].read_text())['rows'], first.tolist())

    def test_stage_probe_positions_are_exact_values_not_integer_dtype(self):
        import numpy as np
        positions=torch.arange(9).view(1,1,-1).expand(3,1,-1)
        expected=positions[:,0].T.numpy().astype('<i4')
        for dtype in (torch.int64,torch.float32,torch.float64):
            got=native_positions(positions.to(dtype),9)
            self.assertEqual(got.dtype,np.dtype('<i4'))
            self.assertEqual(got.tobytes(),expected.tobytes())
        for bad in (positions.float()+.5,positions.float()*float('nan'),positions-1,positions+65536,positions[:2]):
            with self.assertRaises(ValueError):native_positions(bad,9)

    def setUp(self):
        torch.manual_seed(83)
        cfg = Qwen3OmniMoeTextConfig(hidden_size=32, num_attention_heads=2, num_key_value_heads=1,
                                    head_dim=16, num_experts=4, num_experts_per_tok=2, moe_intermediate_size=8)
        cfg._attn_implementation = 'sdpa'
        cfg._experts_implementation = 'eager'
        self.layer = Qwen3OmniMoeThinkerTextDecoderLayer(cfg, 0).eval().bfloat16().float()
        # Direct construction does not run model post_init; give the router meaningful weights.
        with torch.no_grad(): self.layer.mlp.gate.weight.copy_(torch.randn(4,32).bfloat16().float())
        self.h = torch.randn(1,9,32)
        angle = torch.randn(1,9,8)
        self.pe = tuple(torch.cat((x,x),-1) for x in (angle.cos(), angle.sin()))

    def test_fp64_against_independent_hf_attention(self):
        layer = copy.deepcopy(self.layer).double()
        def norm(module, x):
            return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+module.variance_epsilon)*module.weight
        for module in layer.modules():
            if hasattr(module, 'variance_epsilon'):
                module.forward = types.MethodType(norm, module)
        # Capture immediately before experts. HF SDPA executes a different implementation
        # from the explicit CPU matrix/softmax arbiter.
        class Captured(Exception): pass
        logits = []
        def gate(module, args):
            logits.append(torch.nn.functional.linear(args[0], module.weight))
            raise Captured
        handle = layer.mlp.gate.register_forward_pre_hook(gate)
        with torch.no_grad(), self.assertRaises(Captured):
            layer(self.h.double(), position_embeddings=tuple(p.double() for p in self.pe), attention_mask=None)
        handle.remove()
        rows = torch.tensor([0,4,8])
        got = arb.fp64_router(self.layer, self.h, self.pe, rows)
        torch.testing.assert_close(got, logits[0][rows], atol=1e-12, rtol=1e-12)

    @unittest.skipUnless(MPS, 'explicit --mps runs the GPU transfer regression')
    def test_mps_transfer_keeps_fp64_on_cpu(self):
        rows = torch.tensor([0,4,8])
        want = arb.fp64_router(self.layer, self.h, self.pe, rows)
        got = arb.fp64_router(self.layer.to('mps'), self.h.to('mps'), tuple(x.to('mps') for x in self.pe), rows)
        torch.testing.assert_close(got,want,atol=0,rtol=0)
        output = (torch.tensor([[2.,1.,1.-1e-6,0.]],device='mps'),
                  torch.tensor([[.7,.3]],device='mps'), torch.tensor([[0,1]],device='mps'))
        high = torch.tensor([[2.,1.-1e-6,1.,0.]],dtype=torch.float64)
        checked = arb.arbitrate(output,torch.tensor([[0,2]]),lambda rows:high,[],layer=0,record='mps')
        self.assertEqual(checked[2].cpu().tolist(),[[0,2]])

    def test_accept_only_fp64_confirmed_near_boundary(self):
        native = torch.tensor([[0,2]])
        logits = torch.tensor([[2.,1.,1.-1e-6,0.]])
        ids = torch.tensor([[0,1]])
        output = (logits, torch.tensor([[.7,.3]]), ids)
        high = torch.tensor([[2.,1.-1e-6,1.,0.]],dtype=torch.float64)
        events = []
        got = arb.arbitrate(output, native, lambda rows: high, events, layer=7, record='probe')
        self.assertEqual(got[2].tolist(), [[0,2]])
        self.assertEqual(len(events), 1)
        self.assertGreater(events[0]['fp64_margin'], 0)
        with self.assertRaisesRegex(ValueError, 'FP64 rejects'):
            arb.arbitrate(output, native, lambda rows: logits.double(), [], layer=7, record='probe')
        far = (torch.tensor([[2.,1.,0.,-1.]]), output[1], ids)
        with self.assertRaisesRegex(ValueError, 'not a near tie'):
            arb.arbitrate(far, native, lambda rows: high, [], layer=7, record='probe')

    def test_no_disagreement_does_not_invoke_fp64(self):
        output = (torch.randn(2,4), torch.ones(2,2)/2, torch.tensor([[0,1],[2,3]]))
        def unwanted(rows): raise AssertionError('unnecessary arbitration')
        self.assertIs(arb.arbitrate(output, output[2], unwanted, [], layer=0, record='probe'), output)

    def test_input_sensitive_tie_requires_checked_native_fp64(self):
        output=(torch.tensor([[2.,1.,1.-1e-6,0.]]),torch.tensor([[.7,.3]]),torch.tensor([[0,1]]))
        high=output[0].double()
        other=torch.tensor([[2.,1.-1e-7,1.,0.]],dtype=torch.float64)
        events=[]
        got=arb.arbitrate(output,torch.tensor([[0,2]]),lambda rows:high,events,layer=4,record='sensitive',
                          calculate_native=lambda rows:(other,{'incoming_relative_l2':1e-7}))
        self.assertEqual(got[2].tolist(),[[0,2]])
        self.assertEqual(events[0]['method'],'input-sensitive')
        self.assertEqual(events[0]['reference_fp64_ids'],[0,1])
        # Only the choice is arbitrated. Reference logits and mixture weights
        # retain the reference input, never native activations or native logits.
        torch.testing.assert_close(got[0],output[0],atol=0,rtol=0)
        torch.testing.assert_close(got[1],high[:,[0,2]].softmax(-1).float(),atol=0,rtol=0)
        with self.assertRaisesRegex(ValueError,'FP64 rejects'):
            arb.arbitrate(output,torch.tensor([[0,2]]),lambda rows:high,[],layer=4,record='bad',
                          calculate_native=lambda rows:(high,{}))

    def test_native_input_checks_and_independent_calculation(self):
        rows=torch.tensor([4,8])
        incoming=self.h[0].numpy().copy()
        incoming[0,0]+=1e-6
        got,metrics=arb.fp64_native_router(self.layer,self.h,self.pe,rows,incoming)
        want=arb.fp64_router(self.layer,torch.from_numpy(incoming)[None],self.pe,rows)
        torch.testing.assert_close(got,want,atol=0,rtol=0)
        self.assertLess(metrics['incoming_max_row_relative_l2'],.01)
        for bad in (incoming[:-1],incoming*2,incoming*float('nan')):
            with self.assertRaises(ValueError): arb.fp64_native_router(self.layer,self.h,self.pe,rows,bad)

    def test_input_shift_bound_is_independent_and_keeps_limits(self):
        reference=torch.tensor([2.,1.,.9999,0.],dtype=torch.float64)
        native=torch.tensor([2.,.99989,.99991,0.],dtype=torch.float64)
        ids=torch.tensor([0,2]);metrics=dict(incoming_relative_l2=1e-5,incoming_max_row_relative_l2=3e-4,incoming_tolerance=.01)
        report=arb.input_shift_eligibility(reference.float(),reference,native,ids,metrics)
        self.assertGreater(max(p['gap'] for p in report['disputed_pairs']),report['rounding_window'])
        for pair in report['disputed_pairs']:self.assertLessEqual(pair['gap'],pair['bound'])
        with self.assertRaisesRegex(ValueError,'native-input FP64 rejects'):
            arb.input_shift_eligibility(reference.float(),reference,reference,ids,metrics)
        with self.assertRaisesRegex(ValueError,'incoming state'):
            arb.input_shift_eligibility(reference.float(),reference,native,ids,{**metrics,'incoming_max_row_relative_l2':.01001})
        with self.assertRaisesRegex(ValueError,'rounding window'):
            arb.input_shift_eligibility(reference.float()+.001,reference,native,ids,metrics)
        with self.assertRaisesRegex(ValueError,'geometry or values'):
            arb.input_shift_eligibility(reference.float(),reference,native*float('nan'),ids,metrics)
        # The proposed helper must not silently enable a new acceptance rule.
        output=(reference.float()[None],torch.ones(1,2)/2,torch.tensor([[0,1]]))
        with self.assertRaisesRegex(ValueError,'not a near tie'):
            arb.arbitrate(output,ids[None],lambda rows:reference[None],[],layer=0,record='unapproved',
                          calculate_native=lambda rows:(native[None],metrics))
        events=[]
        got=arb.arbitrate(output,ids[None],lambda rows:reference[None],events,layer=0,record='approved',
                          calculate_native=lambda rows:(native[None],metrics),allow_input_shift=True)
        self.assertEqual(got[2].tolist(),[[0,2]])
        self.assertEqual(events[0]['eligibility'],'input-shift')
        self.assertEqual(events[0]['input_shift_audit'],report)
        torch.testing.assert_close(got[0],output[0],atol=0,rtol=0)
        torch.testing.assert_close(got[1],reference[ids].softmax(-1).float()[None],atol=0,rtol=0)
        with self.assertRaisesRegex(ValueError,'native-input FP64 rejects'):
            arb.arbitrate(output,ids[None],lambda rows:reference[None],[],layer=0,record='rejected',
                          calculate_native=lambda rows:(reference[None],metrics),allow_input_shift=True,
                          calculate_rounding=lambda rows:self.fail('input-shift cannot authorize local rounding'))

    def test_input_shift_audits_actual_membership_crossings(self):
        # Two experts enter above both departing experts. Comparing either
        # departing expert with the old kth expert invents an ordering change.
        reference=torch.tensor([2.,1.0001,1.,.9999,.9998],dtype=torch.float64)
        native=torch.tensor([2.,1.0001,1.,1.0002,1.00015],dtype=torch.float64)
        metrics=dict(incoming_relative_l2=1e-5,incoming_max_row_relative_l2=3e-4,incoming_tolerance=.01)
        report=arb.input_shift_eligibility(reference.float(),reference,native,torch.tensor([0,3,4]),metrics)
        self.assertEqual({(p['lost'],p['gained']) for p in report['disputed_pairs']},
                         {(1,3),(1,4),(2,3),(2,4)})
        for pair in report['disputed_pairs']:
            self.assertGreaterEqual(pair['native_margin'],0)
            self.assertLessEqual(pair['gap'],pair['bound'])
            self.assertAlmostEqual(pair['bound'],abs(pair['lost_shift'])+abs(pair['gained_shift'])+report['rounding_window'])

    def test_widened_input_shift_cannot_combine_with_local_rounding(self):
        reference=torch.tensor([[2.,1.,.9999,0.]],dtype=torch.float64)
        native=torch.tensor([[2.,1.,1.-2e-8,0.]],dtype=torch.float64)
        output=(reference.float(),torch.ones(1,2)/2,torch.tensor([[0,1]]))
        metrics=dict(incoming_relative_l2=1e-5,incoming_max_row_relative_l2=3e-4,incoming_tolerance=.01)
        with self.assertRaisesRegex(ValueError,'probe layer 47 token 0: native-input FP64 rejects') as caught:
            arb.arbitrate(output,torch.tensor([[0,2]]),lambda rows:reference,[],layer=47,record='probe',
                          calculate_native=lambda rows:(native,metrics),allow_input_shift=True,
                          calculate_rounding=lambda rows:self.fail('cannot combine widened eligibility and rounding'))
        evidence=json.loads(str(caught.exception).split('input_shift_evidence=',1)[1])
        self.assertEqual(evidence['reference64'],reference[0].tolist())
        self.assertEqual(evidence['native64'],native[0].tolist())
        self.assertEqual(evidence['native_ids'],[0,2])

    def test_layer27_retained_boundary_input_shift(self):
        # The captured 64k failure replaces 49 with 73 while old boundary 39
        # remains selected. The actual crossing is 49/73, never 39/73.
        data=json.loads((Path(__file__).parent/'fixtures/omni_input_shift.json').read_text())
        reference32,reference,native=(torch.tensor(data[key],dtype=torch.float64)
                                      for key in ('reference32','reference64','native64'))
        ids=torch.tensor(data['native_ids'])
        report=arb.input_shift_eligibility(reference32,reference,native,ids,data['incoming'])
        self.assertEqual(len(report['disputed_pairs']),1)
        pair=report['disputed_pairs'][0]
        self.assertEqual((pair['lost'],pair['gained']),(49,73))
        self.assertAlmostEqual(pair['gap'],.001481838501107724)
        self.assertAlmostEqual(pair['bound'],.0016633247600569855)
        self.assertGreater(pair['native_margin'],0)
        output=(reference32.float()[None],torch.ones(1,8)/8,reference32.topk(8).indices[None])
        events=[]
        got=arb.arbitrate(output,ids[None],lambda rows:reference[None],events,layer=27,record='ol000',
                          calculate_native=lambda rows:(native[None],data['incoming']),allow_input_shift=True)
        self.assertEqual(set(got[2][0].tolist()),set(ids.tolist()))
        torch.testing.assert_close(got[0],reference.float()[None],atol=0,rtol=0)
        torch.testing.assert_close(got[1],reference[got[2]].softmax(-1).float(),atol=0,rtol=0)
        self.assertEqual(events[0]['input_shift_audit'],report)

    def test_local_rounding_requires_one_ulp_and_audited_native_choice(self):
        output=(torch.tensor([[2.,1.,1.-1e-6,0.]]),torch.tensor([[.7,.3]]),torch.tensor([[0,1]]))
        high=output[0].double()
        local=torch.tensor([[2.,1.,1.-2e-8,0.]],dtype=torch.float64)
        audited=torch.tensor([[2.,1.-2e-8,1.,0.]],dtype=torch.float64)
        def run(value=local, audit=lambda rows:(audited,[{'stages':{'router':{'max_absolute':1e-14}}}])):
            events=[]
            result=arb.arbitrate(output,torch.tensor([[0,2]]),lambda rows:high,events,
                layer=22,record='rounding',calculate_native=lambda rows:(value,{'incoming_relative_l2':1e-7}),
                calculate_rounding=audit)
            return result,events
        got,events=run()
        self.assertEqual(got[2].tolist(),[[0,2]])
        self.assertEqual(events[0]['method'],'local-rounding')
        self.assertEqual(events[0]['native_input_fp64_ids'],[0,1])
        self.assertLess(events[0]['rounding_gap'],events[0]['rounding_ulp'])
        self.assertIn('router',events[0]['rounding_audit']['stages'])
        torch.testing.assert_close(got[0],output[0],atol=0,rtol=0)
        torch.testing.assert_close(got[1],high[:,[0,2]].softmax(-1).float(),atol=0,rtol=0)
        with self.assertRaisesRegex(ValueError,'one FP32 ULP'):
            run(high,lambda rows:self.fail('ineligible gap must not run an audit'))
        with self.assertRaisesRegex(ValueError,'FP64 rejects'):
            run(audit=lambda rows:(local,[{}]))
        with self.assertRaisesRegex(ValueError,'stage corruption'):
            run(audit=lambda rows:(_ for _ in ()).throw(ValueError('stage corruption')))
        with self.assertRaisesRegex(ValueError,'FP64 rejects'):
            run(audit=None)

    def test_independent_stage_audit_detects_corruption_at_every_stage(self):
        a=self.layer.self_attn
        def norm(x,module):
            return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+module.variance_epsilon)*module.weight
        def linear(x,module): return x@module.weight.T
        def rope(x):
            return x*self.pe[0]+torch.cat((-x[...,8:],x[...,:8]),-1)*self.pe[1]
        with torch.no_grad():
            h=self.h[0]
            x=norm(h,self.layer.input_layernorm)
            q0,k0,v0=(linear(x,m) for m in (a.q_proj,a.k_proj,a.v_proj))
            q=rope(norm(q0.reshape(9,2,16).transpose(0,1),a.q_norm))
            k=rope(norm(k0.reshape(9,1,16).transpose(0,1),a.k_norm))
            v=v0.reshape(9,1,16).transpose(0,1)
            attn=torch.nn.functional.scaled_dot_product_attention(q,k.repeat(2,1,1),v.repeat(2,1,1),is_causal=True).transpose(0,1).reshape(9,32)
            residual=h+linear(attn,a.o_proj)
            ffn=norm(residual,self.layer.post_attention_layernorm)
            stages=dict(input_norm=x,qkv=torch.cat((q0,k0,v0),-1),q=q,k=k,v=v,attn=attn,residual=residual,norm=ffn)
            rows=torch.tensor([0,4,8])
            logits=ffn[rows].double()@self.layer.mlp.gate.weight.double().T
            got,metrics=audit_stages(self.layer,h,self.pe,rows,stages,logits)
            torch.testing.assert_close(got,logits,atol=0,rtol=0)
            self.assertEqual(len(metrics),11)
            # NumPy treats a length-one Torch tensor as a scalar index. Capture
            # files are memmaps; the single disputed token must retain its axis.
            array_stages={key:value.numpy() for key,value in stages.items()}
            single,_=audit_stages(self.layer,h.numpy(),self.pe,rows[1:2],array_stages,logits[1:2])
            torch.testing.assert_close(single,logits[1:2],atol=0,rtol=0)
            for name in stages:
                corrupted={key:value.clone() for key,value in stages.items()}
                corrupted[name][(0,)*corrupted[name].ndim]+=.1
                with self.subTest(stage=name),self.assertRaises(ValueError):
                    audit_stages(self.layer,h,self.pe,rows,corrupted,logits)
            with self.assertRaisesRegex(ValueError,'router'):
                audit_stages(self.layer,h,self.pe,rows,stages,logits+2e-9)
            with self.assertRaisesRegex(ValueError,'incomplete'):
                audit_stages(self.layer,h,self.pe,rows,{k:v for k,v in stages.items() if k!='norm'},logits)

    def test_activation_capture_geometry(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'r.bin'
            want=np.arange(4*9*32,dtype='<f4').reshape(4,9,32)
            path.write_bytes(want.tobytes())
            got=arb.load_activations(path,2,9,32)
            self.assertEqual(got.shape,want.shape)
            for i in range(len(got)):
                stage=got[i]
                np.testing.assert_array_equal(stage,want[i])
                self.assertEqual(stage.nbytes,9*32*4)
                self.assertEqual(stage.offset,i*9*32*4)
            with self.assertRaises(IndexError):got[4]
            del got
            path.write_bytes(want.tobytes()[:-1])
            with self.assertRaisesRegex(ValueError,'incorrect size'): arb.load_activations(path,2,9,32)

    def test_native_trace_must_be_complete_and_well_formed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'routes.jsonl'
            rows = [dict(layer=l,token=t,ids=[0,2]) for l in range(2) for t in range(3)]
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            self.assertEqual(tuple(arb.load_routes(path,2,3,4,2).shape), (2,3,2))
            for bad in (rows[:-1], rows+[rows[0]], list(reversed(rows)),
                        [dict(layer=0,token=0,ids=[0,0])]+rows[1:],
                        [dict(layer=0,token=0,ids=[0,4])]+rows[1:]):
                path.write_text(''.join(json.dumps(r)+'\n' for r in bad))
                with self.assertRaises(ValueError): arb.load_routes(path,2,3,4,2)


if __name__ == '__main__': unittest.main()
