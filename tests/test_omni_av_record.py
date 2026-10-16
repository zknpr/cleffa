"""Soundtrack interleaving and rotary coordinates against the pinned processor."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
MODEL=Path(sys.argv.pop(1)) if len(sys.argv)>1 else ROOT/'gguf/clef-omni-full.gguf'
GOLDEN=Path(sys.argv.pop(1)) if len(sys.argv)>1 else ROOT/'golden/clef-omni-av-f32'


class AVRecord(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not json.loads((GOLDEN/'manifest.json').read_text())['complete']:raise ValueError('incomplete reference')
        cls.requests=list(map(json.loads,(GOLDEN/'requests.jsonl').read_text().splitlines()))
        cls.encoded=list(map(json.loads,(GOLDEN/'encoded.jsonl').read_text().splitlines()))

    def native(self,request,*flags):
        result=subprocess.run([ROOT/'clef-tool','encode-notrunc',MODEL,*map(str,flags)],
                              input=json.dumps(request)+'\n',capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        return {'error':result.stdout} if result.stdout.startswith('ERR ') else json.loads(result.stdout)

    def test_exact_interleaving_and_context(self):
        # Shorter/longer audio, equal timestamps, multiple tracks, standalone
        # audio and a still image all use independent reference token layouts.
        for request,want in zip(self.requests,self.encoded):
            with self.subTest(record=request['id']):
                got=self.native(request)
                self.assertNotIn('error',got)
                for key in ('input_ids','questions','images','position_ids','audio'):
                    self.assertEqual(got.get(key,[]),want.get(key,[]),key)
                self.assertNotIn('error',self.native(request,'--max-length',len(want['input_ids'])))
                self.assertIn('error',self.native(request,'--max-length',len(want['input_ids'])-1))

    def test_association_limits_and_processor_options(self):
        original=self.requests[0]
        bad=copy.deepcopy(original);bad['videos'].append(copy.deepcopy(bad['videos'][0]));del bad['videos'][1]['audio']
        self.assertIn('error',self.native(bad))
        bad=copy.deepcopy(original);bad['audio']=[original['videos'][0]['audio']]*4
        self.assertIn('error',self.native(bad))
        bad=copy.deepcopy(original);bad['videos'][0]['audio']=[]
        self.assertIn('error',self.native(bad))
        for options in ({'fps':2},{'fps':2,'use_audio_in_video':False},{'use_audio_in_video':True}):
            bad=copy.deepcopy(original);bad['media_kwargs']={'videos_kwargs':options}
            self.assertIn('error',self.native(bad))
        good=copy.deepcopy(original)
        good['media_kwargs']={'videos_kwargs':dict(fps=2,do_sample_frames=False,use_audio_in_video=True)}
        self.assertEqual(self.native(good),self.native(original))


if __name__=='__main__':unittest.main()
