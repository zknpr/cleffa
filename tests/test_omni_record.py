"""Native Omni visual request encoding against the pinned encoder and processor."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from transformers import AutoProcessor, Qwen3OmniMoeThinkerConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeThinkerForConditionalGeneration

ROOT=Path(__file__).resolve().parents[1]
MODEL=Path(sys.argv.pop(1)) if len(sys.argv)>1 else ROOT/'gguf/clef-omni.gguf'
SNAPSHOT=Path(sys.argv.pop(1)) if len(sys.argv)>1 else ROOT/'model-omni'
sys.path.insert(0,str(ROOT/'ref'))
sys.path.insert(0,str(ROOT/'tools'))
import corpus_omni_vision,corpus_omni_video,golden_io
from omni import REVISION
from omni_media import media_layout,reference_request
from verify_snapshot import load_manifest,verify_snapshot


class OmniRecord(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        bad=verify_snapshot(SNAPSHOT,load_manifest(REVISION))
        if bad: raise ValueError('; '.join(bad))
        sys.path.insert(0,str(SNAPSHOT.resolve()))
        from joint_schema_model import encode_record,load_image
        cls.requests=corpus_omni_vision.build()+corpus_omni_video.build()
        image=copy.deepcopy(cls.requests[0]);image['id']='image-size-override'
        image['media_kwargs']={'size':{'shortest_edge':4096,'longest_edge':16384}}
        mixed=copy.deepcopy(cls.requests[8]);mixed['id']='mixed-size-override'
        mixed['media_kwargs']={'images_kwargs':{'size':{'shortest_edge':4096,'longest_edge':16384}},
                               'videos_kwargs':{'fps':2,'size':{'shortest_edge':4096,'longest_edge':16384}}}
        cls.requests += [image,mixed]
        processor=AutoProcessor.from_pretrained(SNAPSHOT)
        cfg=json.loads((SNAPSHOT/'config.json').read_text())['thinker_config']
        with torch.device('meta'):
            thinker=Qwen3OmniMoeThinkerForConditionalGeneration(Qwen3OmniMoeThinkerConfig(**cfg))
        cls.expected=[];cls.patches=[]
        for request in cls.requests:
            encoded=encode_record(processor.tokenizer,reference_request(request,load_image),processor=processor)
            layout=media_layout(encoded,thinker) if encoded.media else None
            cls.expected.append(json.loads(golden_io.encoded_line(request,encoded,media_layout=layout)))
            tensors=[encoded.media[k] for k in ('pixel_values','pixel_values_videos') if encoded.media and k in encoded.media]
            cls.patches.append(b''.join(t.float().contiguous().numpy().tobytes() for t in tensors))

    def native(self,request,*flags,mode='encode'):
        p=subprocess.run([str(ROOT/'clef-tool'),mode,str(MODEL),*map(str,flags)],input=json.dumps(request)+'\n',
                         text=True,capture_output=True)
        self.assertEqual(p.returncode,0,p.stderr)
        if p.stdout.startswith('ERR '): return {'error':p.stdout[4:].strip()}
        return json.loads(p.stdout)

    def test_tokens_grids_positions_and_patch_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'patches.bin'
            for req,want,patches in zip(self.requests,self.expected,self.patches):
                with self.subTest(record=req['id']):
                    got=self.native(req,path,mode='encode-patches')
                    self.assertNotIn('error',got)
                    for key in ('input_ids','questions','images','position_ids'):
                        self.assertEqual(got.get(key),want.get(key),key)
                    self.assertEqual(path.read_bytes(),patches,'processor patch bytes')

    def test_exact_context_and_explicit_bad_video_options(self):
        req=self.requests[0]; length=len(self.expected[0]['input_ids'])
        exact=self.native(req,'--max-length',length,mode='encode-notrunc')
        self.assertEqual(exact['input_ids'],self.expected[0]['input_ids'])
        self.assertIn('error',self.native(req,'--max-length',length-1,mode='encode-notrunc'))
        for change in ({'videos_kwargs':{'fps':3}}, {'videos_kwargs':{'do_sample_frames':True}},
                       {'videos_kwargs':{'size':{'shortest_edge':4096,'longest_edge':16384}}},
                       {'videos_kwargs':{'size':{'shortest_edge':1,'longest_edge':2147483648}}}):
            bad=copy.deepcopy(self.requests[5]);bad['media_kwargs']=change
            self.assertIn('error',self.native(bad))
        bad=copy.deepcopy(self.requests[5]);bad['videos'][0]['fps']=6
        self.assertIn('error',self.native(bad))
        for kw in ({'images_kwargs':{}},{'images_kwargs':{'size':{'shortest_edge':0,'longest_edge':4}}},
                   {'images_kwargs':{'do_resize':False}}, {'size':{'shortest_edge':4,'longest_edge':4}},
                   {'min_pixels':4,'max_pixels':4}):
            bad=copy.deepcopy(self.requests[8]);bad['media_kwargs']=kw
            self.assertIn('error',self.native(bad))


if __name__=='__main__': unittest.main()
