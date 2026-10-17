"""Qualification must fail closed on missing, stale or corrupt evidence."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from safetensors.numpy import save_file
from test_omni_qualification import REVISION,run_gates,validate_golden
import test_omni_qualification as qualification


class Qualification(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name);(self.path/'layers').mkdir()
        self.requests=[dict(id='probe',audio=[[0.]*201])]
        raw=json.dumps(self.requests[0])+'\n';(self.path/'requests.jsonl').write_text(raw)
        (self.path/'encoded.jsonl').write_text(json.dumps(dict(id='probe',input_ids=[1,2,3]))+'\n')
        self.tensors={k:np.zeros((3,2),dtype='f4') for k in ['embed','final_norm']+[f'layer.{i:02d}' for i in range(48)]}
        audio=[f'audio.0.layer.{i:02d}' for i in range(32)]+['audio.0.features','audio.0.conv','audio.0.output',
             'audio.position_table','audio.conv2d1','audio.conv2d2','audio.conv2d3']
        self.tensors.update({k:np.zeros((1,2),dtype='f4') for k in audio})
        self.manifest=dict(complete=True,revision=REVISION,precision='float32-streamed',corpus='audio',
                           independent_from_native=True,
                           attention_reference=dict(value_reduction='blocked',first_layer=0),
                           corpus_sha256=hashlib.sha256(raw.encode()).hexdigest(),
                           records=[dict(id='probe',tokens=3,dump_start=0,dumped=True)],
                           media_shapes={'probe':{k:list(v.shape) for k,v in self.tensors.items() if k.startswith('audio.')}})
        save_file({'probe/q':np.array([.1,.2],dtype='f4')},self.path/'logits.safetensors')
        self.save()

    def save(self):
        (self.path/'manifest.json').write_text(json.dumps(self.manifest))
        save_file(self.tensors,self.path/'layers/probe.safetensors')

    def validate(self):return validate_golden(self.path,self.requests,'audio')

    def test_complete_fixture(self):self.assertEqual(self.validate()['records'],1)

    def test_fp64_is_separate_complete_hashed_and_checks_actual_tensor_precision(self):
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ref'))
        import omni_fp64_policy as policy
        self.check_fp64(policy)

    def test_bounded_fp64_keeps_the_same_gates_and_checks_actual_tensor_precision(self):
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ref'))
        import oracle_fp64_bounded as policy
        self.check_fp64(policy)

    def check_fp64(self, policy):
        from omni_fp64_policy import digest
        model=self.path/'model';model.mkdir()
        for name in ('joint_schema_model.py','config.json','joint_head_config.json'):
            (model/name).write_text('unit fixture only')
        self.manifest.update(precision='float64-streamed',fp64_policy=policy.POLICY,
                             implementation_sha256=policy.implementation_sources(),**policy.runtime_provenance(model,'audio'))
        self.tensors={k:v.astype('f8') if k!='audio.0.features' else v for k,v in self.tensors.items()}
        for i in range(48):
            self.tensors.update({f'routing.{i:02d}.ids':np.tile(np.arange(8),(3,1)),
                                 f'routing.{i:02d}.logits':np.zeros((3,128),dtype='f8'),
                                 f'routing.{i:02d}.weights':np.full((3,8),.125,dtype='f8'),
                                 f'routing.{i:02d}.margin':np.zeros(3,dtype='f8')})
        enc=dict(id='probe',input_ids=[1,2,3],questions=[dict(id='q',option_ids=['a','b'])])
        (self.path/'encoded.jsonl').write_text(json.dumps(enc)+'\n')
        save_file({'probe/q':np.array([.1,.2],dtype='f8')},self.path/'logits.safetensors')
        def save_hashed():
            self.save()
            self.manifest['artifacts_sha256']={name:digest(self.path/name) for name in
                ('requests.jsonl','encoded.jsonl','logits.safetensors','layers/probe.safetensors')}
            (self.path/'manifest.json').write_text(json.dumps(self.manifest))
        save_hashed()
        with self.assertRaises(ValueError):self.validate()
        self.assertEqual(validate_golden(self.path,self.requests,'audio',reference='fp64')['records'],1)
        for key in ('layer.47','audio.0.output','routing.47.weights'):
            original=self.tensors[key]
            self.tensors[key]=original.astype('f4');save_hashed()
            with self.subTest(key=key), self.assertRaisesRegex(ValueError,'dtype'):
                validate_golden(self.path,self.requests,'audio',reference='fp64')
            self.tensors[key]=original
        save_hashed()
        save_file({'probe/q':np.array([.1,.2],dtype='f8'),'extra/q':np.ones(2,dtype='f8')},self.path/'logits.safetensors')
        self.manifest['artifacts_sha256']['logits.safetensors']=digest(self.path/'logits.safetensors')
        (self.path/'manifest.json').write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError,'question inventory'):
            validate_golden(self.path,self.requests,'audio',reference='fp64')

    def test_arbitrated_or_mixed_attention_reference_cannot_qualify(self):
        original = dict(self.manifest)
        for changes in ({'arbitration': {'accepted': 0}}, {'independent_from_native': False},
                        {'attention_reference': {'value_reduction': 'sdpa', 'first_layer': 0}},
                        {'attention_reference': {'value_reduction': 'blocked', 'first_layer': 40}},
                        {'attention_reference': None}):
            with self.subTest(changes=changes):
                self.manifest = {**original, **changes}; self.save()
                with self.assertRaisesRegex(ValueError, 'independent blocked'):
                    self.validate()

    def test_capture_requires_current_model_and_executable(self):
        capture=self.path/'capture';capture.mkdir()
        path=capture/'manifest.json'
        manifest=dict(complete=True,model_sha256='weights',binary_sha256='current')
        path.write_text(json.dumps(manifest))
        qualification.validate_capture(capture,'weights','current')
        for key,value in [('complete',False),('model_sha256','old'),('binary_sha256','old')]:
            with self.subTest(key=key):
                changed={**manifest,key:value};path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):qualification.validate_capture(capture,'weights','current')
        del manifest['binary_sha256'];path.write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):qualification.validate_capture(capture,'weights','current')

    def test_build_all_precedes_executable_identification(self):
        executables=('clef','clef-server','clef-tool','clef-media')
        for name in executables:(self.path/name).write_text('old')
        calls=[]
        def runner(command,**kwargs):
            calls.append(command)
            for name in executables:(self.path/name).write_text('new')
            return subprocess.CompletedProcess(command,0,'rebuilt','')
        with patch.object(qualification,'ROOT',self.path):
            report,hashes=qualification.build_binaries(self.path/'build',runner=runner)
            self.assertTrue(report['complete']);self.assertEqual(calls,[['make','-j1','all','media']])
            self.assertEqual(set(hashes),set(executables))
            self.assertEqual(set(hashes.values()),{hashlib.sha256(b'new').hexdigest()})
            with self.assertRaisesRegex(ValueError,'build failed'):
                qualification.build_binaries(self.path/'failed',runner=lambda cmd,**kw:subprocess.CompletedProcess(cmd,1,'','failed'))

    def test_missing_stale_and_incomplete_inventory(self):
        for key,value in [('complete',False),('revision','stale'),('records',[]),('corpus_sha256','wrong')]:
            with self.subTest(key=key):
                original=self.manifest[key];self.manifest[key]=value;self.save()
                with self.assertRaises(ValueError):self.validate()
                self.manifest[key]=original
        self.save();(self.path/'encoded.jsonl').unlink()
        with self.assertRaises(OSError):self.validate()

    def test_missing_feature_layer_and_nonfinite_logits(self):
        for key in ('audio.0.layer.31','audio.0.features','layer.47'):
            with self.subTest(key=key):
                original=self.tensors.pop(key);self.save()
                with self.assertRaises(ValueError):self.validate()
                self.tensors[key]=original
        self.save();save_file({'probe/q':np.array([np.nan],dtype='f4')},self.path/'logits.safetensors')
        with self.assertRaises(ValueError):self.validate()

    def test_changed_corpus_and_record(self):
        with self.assertRaises(ValueError):validate_golden(self.path,[dict(id='other')],'audio')
        (self.path/'requests.jsonl').write_text('{}\n')
        with self.assertRaises(ValueError):self.validate()

    def gate_commands(self,reference):
        model=self.path/'model.gguf';model.write_bytes(b'gguf')
        inventory={name:dict(golden=name+'-golden',fp32=name+'-fp32',control=name+'-control',
                             captures=str(self.path/'captures')) for name in qualification.CORPORA}
        (self.path/'inventory.json').write_text(json.dumps(inventory))
        seen={}
        def run(gates,folder):seen.update(gates);return dict(complete=False,gates=[])
        class Corpus:build=staticmethod(lambda:[{}])
        argv=['test_omni_qualification.py','--model',str(model),'--golden-root',str(self.path),
              '--inventory',str(self.path/'inventory.json'),'--output',str(self.path/reference/'report.json'),
              '--reference',reference]
        hashes=dict(clef='0'*64)
        with patch.object(qualification,'build_binaries',return_value=(dict(gates=[]),hashes)),\
             patch.object(qualification,'binary_hashes',return_value=hashes),\
             patch.object(qualification,'validate_golden',return_value={}),\
             patch.object(qualification,'validate_capture'),\
             patch.object(qualification,'run_gates',side_effect=run),\
             patch.object(qualification.importlib,'import_module',return_value=Corpus),\
             patch.object(sys,'argv',argv),self.assertRaises(SystemExit):
            qualification.main()
        return seen

    def test_fp64_parity_gates_are_bounded_by_the_fp32_reference(self):
        gates=self.gate_commands('fp64')
        for name in qualification.CORPORA:
            command=gates[name.replace('_','-')+'-parity']
            if name=='long':self.assertNotIn('--bound-by',command)
            else:self.assertEqual(command[-2:],['--bound-by',str(self.path.resolve()/(name+'-fp32'))])
        self.assertFalse(any('--bound-by' in command for command in self.gate_commands('fp32').values()))

    def test_failed_and_missing_gates_never_complete(self):
        calls=[]
        def runner(command,**kwargs):
            calls.append(command);return subprocess.CompletedProcess(command,1 if command==['bad'] else 0,'result','')
        report=run_gates([('audio',['ok']),('av',['bad'])],self.path,runner=runner)
        self.assertFalse(report['complete']);self.assertEqual(len(calls),2)
        with self.assertRaises(ValueError):run_gates([],self.path,runner=runner)
        report=run_gates([('missing',['none'])],self.path,runner=lambda *a,**k:(_ for _ in ()).throw(FileNotFoundError('missing')))
        self.assertFalse(report['complete']);self.assertIn('missing',report['gates'][0]['error'])

    def test_interrupted_first_gate_cannot_leave_a_stale_success(self):
        path=self.path/'gates.json';path.write_text('{"complete":true}')
        with self.assertRaises(KeyboardInterrupt):
            run_gates([('interrupted',['run'])],self.path,
                      runner=lambda *a,**k:(_ for _ in ()).throw(KeyboardInterrupt()))
        self.assertFalse(json.loads(path.read_text())['complete'])


if __name__=='__main__':unittest.main()
