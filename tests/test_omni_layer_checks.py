"""Missing/corrupt later records must not disappear from all-layer verification."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from safetensors.numpy import save_file

from compare_omni_layers import compare
from compare_omni_routes import compare as compare_routes, capture_rows


class LayerChecks(unittest.TestCase):
    def test_routing_difference_is_reported_but_missing_evidence_fails(self):
        ref = {'routing.00.ids':np.arange(8).reshape(1,8),
               'routing.00.logits':np.zeros((1,128)),
               'routing.00.weights':np.full((1,8),.125), 'routing.00.margin':np.array([0.])}
        row=dict(layer=0,token=2,ids=[1,0,2,3,4,5,6,7],logits=[0.]*128,weights=[.125]*8,margin=0.)
        got=compare_routes(iter([row]),ref,2)
        self.assertTrue(got['diagnostic_only']);self.assertEqual(got['mismatch_fraction'],1.)
        self.assertEqual(got['selected_set_mismatches'],0)
        row['ids'][-1]=8
        self.assertEqual(compare_routes([row],ref,2)['selected_set_mismatches'],1)
        with self.assertRaisesRegex(ValueError,'incomplete'):compare_routes([],ref,2)
        with self.assertRaisesRegex(ValueError,'extra'):compare_routes([row,row],ref,2)
        with self.assertRaisesRegex(ValueError,'weights'):
            compare_routes([{**row,'weights':[float('nan')]*8}],ref,2)
        ref['routing.00.margin'][0]=float('nan')
        with self.assertRaisesRegex(ValueError,'reference routing'):compare_routes([row],ref,2)
        ref['routing.00.margin'][0]=0.
        path=self.paired/'route.jsonl'
        path.write_text(''.join(json.dumps({**row,'token':i})+'\n' for i in range(3)))
        self.assertEqual([r['token'] for r in capture_rows(path,3,2,1)],[2])
        with self.assertRaisesRegex(ValueError,'incomplete'):list(capture_rows(path,3,2,2))

    def test_noise_control_measures_spread_without_qualifying_native(self):
        import hashlib
        import shutil
        from compare_omni_layers import compare_references
        request='{"id":"a"}\n{"id":"b"}\n'
        (self.golden/'requests.jsonl').write_text(request)
        (self.golden/'encoded.jsonl').write_text('{"input_ids":[1,2,3]}\n')
        manifest={**self.manifest,'revision':'same-pinned-revision','corpus_sha256':hashlib.sha256(request.encode()).hexdigest(),
                  'independent_from_native':True,'attention_reference':dict(value_reduction='blocked',first_layer=0)}
        (self.golden/'manifest.json').write_text(json.dumps(manifest))
        save_file({'a/q':np.array([1.,2.],dtype='f4')},self.golden/'logits.safetensors')
        control=self.golden.parent/'control';shutil.copytree(self.golden,control)
        other={**manifest,'attention_reference':dict(value_reduction='sdpa',first_layer=0)}
        (control/'manifest.json').write_text(json.dumps(other))
        save_file({'a/q':np.array([1.,2.2],dtype='f4')},control/'logits.safetensors')
        report=compare_references(self.golden,control,layers=2)
        self.assertTrue(report['complete']);self.assertTrue(report['native_limits_unchanged'])
        self.assertGreater(report['questions'][0]['max_logit_delta'],.19)
        self.assertEqual(report['records'][0]['layers'][0]['relative_l2'],0.)
        other['arbitration']={'accepted':0};(control/'manifest.json').write_text(json.dumps(other))
        with self.assertRaisesRegex(ValueError,'independent'):compare_references(self.golden,control,layers=2)

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.golden=Path(self.temp.name)/'golden';self.paired=Path(self.temp.name)/'paired'
        (self.golden/'layers').mkdir(parents=True);self.paired.mkdir()
        records=[dict(id='a',tokens=3,dump_start=1,dumped=True),dict(id='b',tokens=3,dump_start=0,dumped=True)]
        self.manifest=dict(complete=True,precision='float32-streamed',corpus_sha256='test',records=records)
        (self.golden/'manifest.json').write_text(json.dumps(self.manifest))
        (self.paired/'manifest.json').write_text(json.dumps(dict(complete=True,requests_sha256='test',records=records)))
        self.values=np.arange(24,dtype='<f4').reshape(4,3,2)+1
        for r in records:
            self.values.tofile(self.paired/(r['id']+'.bin'))
            save_file({key:self.values[i,r['dump_start']:].copy() for i,key in enumerate(['embed','layer.00','layer.01','final_norm'])},
                      self.golden/'layers'/(r['id']+'.safetensors'))

    def test_complete_suffixes_pass_and_later_failure_fails(self):
        report=compare(self.golden,self.paired,layers=2)
        self.assertTrue(report['complete']);self.assertEqual(len(report['records']),2)
        bad=self.values.copy();bad[2,1,0]*=2;bad.tofile(self.paired/'b.bin')
        report=compare(self.golden,self.paired,layers=2)
        self.assertFalse(report['complete']);self.assertEqual(report['failures'],1)

    def test_missing_record_or_layer_is_not_a_pass(self):
        (self.paired/'b.bin').unlink()
        with self.assertRaises(FileNotFoundError):compare(self.golden,self.paired,layers=2)
        self.values.tofile(self.paired/'b.bin')
        save_file({'embed':self.values[0]},self.golden/'layers/b.safetensors')
        with self.assertRaisesRegex(ValueError,'inventory'):compare(self.golden,self.paired,layers=2)

    def test_nonfinite_wrong_shape_and_provenance_fail(self):
        bad=self.values.copy();bad[3,2,1]=np.nan;bad.tofile(self.paired/'b.bin')
        with self.assertRaisesRegex(ValueError,'nonfinite'):compare(self.golden,self.paired,layers=2)
        (self.paired/'b.bin').write_bytes(self.values.tobytes()[:-4])
        with self.assertRaisesRegex(ValueError,'size'):compare(self.golden,self.paired,layers=2)
        self.values.tofile(self.paired/'b.bin')
        manifest=copy.deepcopy(self.manifest);manifest['arbitration']=dict(native_activations_sha256={'b':'wrong'})
        (self.golden/'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError,'provenance'):compare(self.golden,self.paired,layers=2)


if __name__=='__main__':unittest.main()
