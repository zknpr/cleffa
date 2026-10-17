"""Bounded Omni container sampling/scaling against the pinned PyAV loader."""
import base64
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from video_request import convert
from verify_snapshot import load_manifest,verify_snapshot


class OmniVideoRequest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=ROOT/'model-omni'
        errors=verify_snapshot(path,load_manifest('0db1cd2607d76a7bdb2a382f659e7b313079f84b'))
        if errors: raise ValueError('; '.join(errors))
        sys.path.insert(0,str(path))
        from joint_schema_model import load_video
        cls.load_video=staticmethod(load_video)

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/'clip $(touch unexpected);.mp4'
        self.request={'model':'clef-omni','state':'Describe movement.','questions':{'moving':{'type':'boolean'}}}

    def clip(self,width=96,height=64,count=9,vfr=False,audio=False,codec='libx264',audio_offset=0,audio_codec='aac'):
        y,x=np.mgrid[:height,:width]
        frames=np.stack([np.stack([(x*7+i*17)%256,(y*5+i*19)%256,((x+y)*3+i*23)%256],-1).astype('uint8') for i in range(count)])
        command=['ffmpeg','-v','error','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{width}x{height}',
                 '-framerate','6','-i','pipe:0']
        if audio: command+=['-itsoffset',str(audio_offset),'-f','lavfi','-i','sine=frequency=440:sample_rate=16000:duration=1.5','-shortest','-c:a',audio_codec]
        if vfr: command+=['-vf',r'setpts=if(lt(N\,3)\,N\,N+2)/(6*TB)','-fps_mode','vfr']
        pixel={'mjpeg':'yuvj420p','prores_ks':'yuv422p10le'}.get(codec,'yuv420p')
        if codec=='libx265': command+=['-x265-params','pools=none:frame-threads=1:log-level=error']
        command+=['-c:v',codec,'-pix_fmt',pixel,'-threads','1','-f','mov','-y',str(self.path)]
        subprocess.run(command,input=frames.tobytes(),check=True,capture_output=True)

    def test_frames_match_pinned_loader(self):
        for width,height,count,vfr in ((96,64,1,False),(96,64,9,False),(800,640,9,False),
                                       (64,96,12,False),(96,64,9,True)):
            with self.subTest(shape=(width,height),count=count,vfr=vfr):
                self.clip(width,height,count,vfr)
                want,track=self.load_video(str(self.path));self.assertIsNone(track)
                result=json.loads(convert(self.path,self.request,model_family='qwen3_omni_moe'))
                video=result['videos'][0]
                self.assertEqual(set(video),{'frames','fps'});self.assertEqual(video['fps'],2)
                got=np.stack([np.asarray(Image.open(io.BytesIO(base64.b64decode(f.split(',',1)[1])))) for f in video['frames']])
                np.testing.assert_array_equal(got,want)
                self.assertEqual(result['media_kwargs']['videos_kwargs']['fps'],2)
                self.assertNotIn('media_kwargs',self.request)

    def test_other_supported_codecs_preserve_scaled_pixels(self):
        for codec in ('libx265','prores_ks','mjpeg'):
            with self.subTest(codec=codec):
                self.clip(800,640,3,codec=codec)
                want,_=self.load_video(str(self.path))
                result=json.loads(convert(self.path,self.request,model_family='qwen3_omni_moe'))
                got=np.stack([np.asarray(Image.open(io.BytesIO(base64.b64decode(f.split(',',1)[1]))))
                              for f in result['videos'][0]['frames']])
                np.testing.assert_array_equal(got,want)

    def test_soundtrack_is_lossless_and_limits_fail_explicitly(self):
        import av
        for offset in (0,.375):
            self.clip(audio=True,audio_offset=offset)
            want_frames,want=self.load_video(str(self.path))
            result=json.loads(convert(self.path,self.request,model_family='qwen3_omni_moe'))
            with av.open(io.BytesIO(base64.b64decode(result['videos'][0]['audio'].split(',',1)[1]))) as wav:
                got=np.concatenate([f.to_ndarray().reshape(-1) for f in wav.decode(audio=0)])
            np.testing.assert_array_equal(got,want)
            got_frames=np.stack([np.asarray(Image.open(io.BytesIO(base64.b64decode(f.split(',',1)[1]))))
                                 for f in result['videos'][0]['frames']])
            np.testing.assert_array_equal(got_frames,want_frames)
            self.assertTrue(result['media_kwargs']['videos_kwargs']['use_audio_in_video'])
        with self.assertRaisesRegex(ValueError,'sample limit|byte limit'):
            convert(self.path,self.request,model_family='qwen3_omni_moe',max_audio_samples=100)
        self.clip()
        for kwargs in ({'fps':3},{'num_frames':2},{'max_frames':3},{'max_body':1},{'timeout':.000001}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                convert(self.path,self.request,model_family='qwen3_omni_moe',**kwargs)


if __name__=='__main__':unittest.main()
