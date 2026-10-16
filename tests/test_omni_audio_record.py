"""Audio host placement must exactly match independently generated pinned records."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
MODEL=Path(sys.argv.pop(1)) if len(sys.argv)>1 else ROOT/'gguf/clef-omni.gguf'
GOLDEN=Path(sys.argv.pop(1)) if len(sys.argv)>1 else ROOT/'golden/clef-omni-audio-f32'


class AudioRecord(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not json.loads((GOLDEN/'manifest.json').read_text())['complete']:raise ValueError('incomplete reference')
        cls.requests=list(map(json.loads,(GOLDEN/'requests.jsonl').read_text().splitlines()))
        cls.encoded=list(map(json.loads,(GOLDEN/'encoded.jsonl').read_text().splitlines()))

    def native(self,request,*flags,mode='encode'):
        result=subprocess.run([ROOT/'clef-tool',mode,MODEL,*map(str,flags)],input=json.dumps(request)+'\n',capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        return {'error':result.stdout} if result.stdout.startswith('ERR ') else json.loads(result.stdout)

    def test_exact_layout(self):
        for request,want in zip(self.requests,self.encoded):
            with self.subTest(record=request['id']):
                got=self.native(request)
                self.assertNotIn('error',got)
                for key in ('input_ids','questions','images','position_ids','audio'):
                    self.assertEqual(got.get(key,[]),want.get(key,[]),key)

    def test_limits_and_strict_placeholders(self):
        request=self.requests[0];length=len(self.encoded[0]['input_ids'])
        self.assertNotIn('error',self.native(request,'--max-length',length,mode='encode-notrunc'))
        self.assertIn('error',self.native(request,'--max-length',length-1,mode='encode-notrunc'))
        for audio in ([[]],[[True]],[[1e100]],['https://example.com/a.wav'],[request['audio'][0]]*5):
            bad=copy.deepcopy(request);bad['audio']=audio
            self.assertIn('error',self.native(bad))
        bad=copy.deepcopy(request);bad['state']='<|audio_pad|>'
        self.assertIn('error',self.native(bad))
        self.assertNotIn('error',self.native(bad,mode='encode-strict'))


if __name__=='__main__':unittest.main()
