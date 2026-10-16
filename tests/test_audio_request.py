"""Local audio conversion preserves pinned decoding and enforces resource limits."""
import base64
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import av
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from audio_request import convert
from verify_snapshot import load_manifest,verify_snapshot


class AudioRequest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model=ROOT/'model-omni'
        errors=verify_snapshot(model,load_manifest('0db1cd2607d76a7bdb2a382f659e7b313079f84b'))
        if errors:raise ValueError('; '.join(errors))
        sys.path.insert(0,str(model))
        from joint_schema_model import load_audio
        cls.load_audio=staticmethod(load_audio)

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.folder=Path(self.temp.name)
        self.request=dict(model='clef-omni',state='Listen.',questions={'sound':{'type':'boolean'}})

    def clip(self,suffix,codec):
        path=self.folder/('clip $(touch unexpected);.'+suffix)
        subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','sine=frequency=773:sample_rate=44100:duration=0.25',
                        '-ac','2','-c:a',codec,'-y',str(path)],check=True,capture_output=True)
        return path

    def test_samples_exact_for_compressed_stereo_and_wav(self):
        for suffix,codec in [('wav','pcm_s16le'),('wav','pcm_f32le'),('flac','flac'),('mp3','libmp3lame'),
                             ('ogg','libopus'),('m4a','aac')]:
            with self.subTest(codec=codec):
                path=self.clip(suffix,codec);want=self.load_audio(str(path))
                result=json.loads(convert(path,self.request))
                with av.open(io.BytesIO(base64.b64decode(result['audio'][0].split(',',1)[1]))) as container:
                    got=np.concatenate([f.to_ndarray().reshape(-1) for f in container.decode(audio=0)])
                self.assertEqual(got.dtype,np.float32)
                self.assertEqual(got.tobytes(),want.tobytes())
                self.assertNotIn('audio',self.request)

    def test_reject_limits_nonlocal_input_and_playlists(self):
        path=self.clip('wav','pcm_s16le')
        for kwargs in ({'max_samples':1},{'max_samples':0},{'max_body':1},{'timeout':.00001}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):convert(path,self.request,**kwargs)
        with self.assertRaises(ValueError):convert(path,dict(self.request,audio=[[0]]))
        fifo=self.folder/'fifo';os.mkfifo(fifo)
        with self.assertRaises(ValueError):convert(fifo,self.request)
        for payload in (b'#EXTM3U\nhttp://127.0.0.1/private\n',b'RIFF\x00\x00\x00\x00WAVEgarbage'):
            bad=self.folder/'bad';bad.write_bytes(payload)
            with self.assertRaises(ValueError):convert(bad,self.request)
        with self.assertRaises(OSError):convert('https://example.com/a.wav',self.request)


if __name__=='__main__':unittest.main()
