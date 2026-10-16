"""Omni HTTP media admission, strict parsing, cache fallback and failure recovery.

Usage: test_omni_server_media.py MODEL.gguf VISION_GOLDEN_DIR
"""
import base64
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor

from PIL import Image,ImageOps
import test_omni_server as common

ROOT=common.ROOT
MODEL=Path(sys.argv.pop(1)) if len(sys.argv)>1 else ROOT/'gguf/clef-omni-vision.gguf'
GOLDEN=Path(sys.argv.pop(1)) if len(sys.argv)>1 else ROOT/'golden/clef-omni-vision-f32'
sys.path.insert(0,str(ROOT/'ref'))
import corpus_omni_video


class OmniMediaServer(common.OmniServer):
    model=MODEL

    @classmethod
    def setUpClass(cls):
        cls.images=[json.loads(s) for s in (GOLDEN/'requests.jsonl').read_text().split('\n') if s]
        cls.videos=corpus_omni_video.build()

    def native(self,requests,*flags):
        p=subprocess.run([ROOT/'clef','-m',self.model,'--strict','--no-truncate',*flags],
                         input=''.join(json.dumps(r)+'\n' for r in requests),text=True,capture_output=True,
                         check=True,timeout=300,env=common.BASE_ENV)
        output=[json.loads(s) for s in p.stdout.split('\n') if s]
        self.assertEqual(len(output),len(requests));self.assertTrue(all('error' not in r for r in output))
        return output

    def test_media_transitions_and_concurrent_admission(self):
        changed=copy.deepcopy(self.images[0])
        pixels=Image.open(io.BytesIO(base64.b64decode(changed['images'][0].split(',',1)[1]))).convert('RGB')
        out=io.BytesIO();ImageOps.invert(pixels).save(out,'PNG')
        changed['images'][0]='data:image/png;base64,'+base64.b64encode(out.getvalue()).decode()
        requests=self.images+self.videos+[changed,self.images[0],self.images[-1]]
        raw=self.native(requests,'--logits')
        self.assertEqual(self.native(requests,'--logits','--prefix-cache'),raw)
        self.assertEqual(self.native(requests,'--logits','--template-cache'),raw)
        expected=self.native(requests)
        with self.server('--max-videos','2','--max-image-requests','1','--prefix-cache-mb','64',
                         env={'CLEF_DEBUG_POISON':'1'}) as (port,log):
            for headers in ({},{'X-Clef-Prefix-Cache':'media-a'},{'X-Clef-Prefix-Cache':'media-b'}):
                with ThreadPoolExecutor(max_workers=6) as pool:
                    actual=list(pool.map(lambda r:self.post(port,r,headers),requests))
                self.assertEqual(actual,[(200,r) for r in expected])

    def test_bad_media_releases_admission_and_strict_tokens_are_text(self):
        image,video=self.images[0],self.videos[0]
        good=self.native([image,video])
        bad=[dict(image,images=image['images']*5),dict(video,videos=video['videos']*3),
             dict(image,media_kwargs={'size':{'shortest_edge':67108864,'longest_edge':67108864}}),
             dict(video,media_kwargs={'videos_kwargs':{'fps':2,'size':{'shortest_edge':4194304,'longest_edge':4194304}}}),
             dict(video,videos=[dict(frames=video['videos'][0]['frames']*769,fps=2)]),
             dict(image,images=['data:image/png;base64,AAAA']),dict(image,audio=[[0.,1.]]),
             dict(video,videos=[dict(frames=video['videos'][0]['frames'],fps=6)])]
        literal=dict(image,state='<|video_pad|><|image_pad|><|vision_start|>')
        literal_expected=self.native([literal])[0]
        with self.server('--max-image-requests','1') as (port,log):
            for request in bad:
                status,body=self.post(port,request)
                self.assertEqual(status,400,body);self.assertIn('error',body)
                for req,want in zip([image,video],good):self.assertEqual(self.post(port,req),(200,want))
            self.assertEqual(self.post(port,literal),(200,literal_expected))

    def test_gpu_failure_between_media_and_text(self):
        requests=[self.images[0],self.videos[1],self.images[-1],self.videos[3],self.images[0]]
        expected=self.native(requests)
        for keyed in (False,True):
            with self.subTest(keyed=keyed),self.server('--max-image-requests','1','--prefix-cache-mb','64',
                             env={'CLEF_DEBUG_NIL_CMDBUF':'2','CLEF_DEBUG_POISON':'1'}) as (port,log):
                for i,(request,want) in enumerate(zip(requests,expected)):
                    status,body=self.post(port,request,{'X-Clef-Prefix-Cache':'same-key'} if keyed else {})
                    if i==1 and not keyed:
                        self.assertEqual(status,500);self.assertIn('command buffer',json.dumps(body))
                    else:self.assertEqual((status,body),(200,want))
                if keyed:
                    # The existing cache contract retries a failed pass uncached.
                    # Assert the failure actually occurred before accepting recovery.
                    log.seek(0);trace=log.read()
                    self.assertIn('command buffer',trace);self.assertIn('serving uncached',trace)


if __name__=='__main__':unittest.main()
