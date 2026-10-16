"""Native audio frontend: exact layout and approved bounded floating arithmetic."""
import json
from pathlib import Path
import subprocess
import tempfile
import time
import unittest

import numpy as np
import torch
from transformers import WhisperFeatureExtractor

ROOT=Path(__file__).resolve().parents[1]
SPEC=json.loads((ROOT/'tests/fixtures/audio_frontend.json').read_text())


class Frontend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.extractor=WhisperFeatureExtractor(**json.loads((ROOT/'model-omni/processor_config.json').read_text())['feature_extractor'])
        cls.window=torch.hann_window(400)
        cls.filters=cls.extractor.mel_filters.astype('f4')
        cls.worst_absolute=cls.worst_relative=0.

    @classmethod
    def tearDownClass(cls):
        print(f'frontend worst feature max_abs={cls.worst_absolute:.9g}, relative_L2={cls.worst_relative:.9g}',flush=True)

    def close_stage(self, actual, expected, name):
        self.assertEqual(actual.shape,expected.shape,name)
        self.assertTrue(np.isfinite(actual).all() and np.isfinite(expected).all(),name)
        delta=actual.astype('complex128')-expected.astype('complex128')
        scaled=float(abs(delta).max()/max(1,float(abs(expected).max())))
        relative=float(np.linalg.norm(delta)/max(1e-30,np.linalg.norm(expected)))
        self.assertLessEqual(scaled,1e-6,f'{name} scaled max {scaled}')
        self.assertLessEqual(relative,1e-6,f'{name} relative L2 {relative}')

    def check(self,clips):
        try:
            reference=self.extractor(clips,sampling_rate=16000,padding=True,truncation=False,return_attention_mask=True,return_tensors='np')
        except (ValueError,RuntimeError,IndexError): reference=None
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            paths=[]
            for i,clip in enumerate(clips):
                path=folder/f'input-{i}.f32';clip.astype('<f4').tofile(path);paths.append(path)
            p=subprocess.run([ROOT/'tests/audio-features',folder,*paths],capture_output=True,text=True,timeout=120)
            if reference is None:
                self.assertNotEqual(p.returncode,0);self.assertTrue(p.stderr.strip());return
            self.assertEqual(p.returncode,0,p.stderr)
            meta=json.loads(p.stdout)
            self.assertEqual(len(meta),len(clips))
            maximum=max(map(len,clips));frames=maximum//160
            def read(i,name,shape):return np.fromfile(folder/f'{i}-{name}.bin',dtype='<f4').reshape(shape)
            self.assertEqual(read(0,'window',(400,)).tobytes(),self.window.numpy().tobytes())
            self.assertEqual(read(0,'filters',(201,128)).tobytes(),self.filters.tobytes())
            for i,clip in enumerate(clips):
                valid=int(reference['attention_mask'][i].sum())
                self.assertEqual(meta[i],dict(frames=valid,padded_frames=frames,tokens=13*(valid//100)+(valid%100+7)//8))
                native_mask=np.arange(frames)<meta[i]['frames']
                np.testing.assert_array_equal(native_mask,reference['attention_mask'][i].astype(bool))
                padded=np.pad(clip,(0,maximum-len(clip)))
                centered=np.pad(padded,(200,200),mode='reflect')
                blocks=np.lib.stride_tricks.sliding_window_view(centered,400)[::160][:frames]*self.window.numpy()
                # NumPy uses a distinct double FFT implementation. Verify the
                # complete spectrum before examining power/mel/log stages.
                independent=np.fft.rfft(blocks.astype('f8'),axis=-1).T
                re,im=read(i,'real',(201,frames)),read(i,'imag',(201,frames))
                self.close_stage(re.astype('complex128')+1j*im,independent,'FFT')
                power=read(i,'power',(201,frames))
                self.close_stage(power,re.astype('f8')**2+im.astype('f8')**2,'power')
                mel=read(i,'mel',(128,frames))
                self.close_stage(mel,self.filters.astype('f8').T@power.astype('f8'),'mel')
                features=read(i,'features',(128,frames))
                log=np.log10(np.maximum(mel.astype('f8'),1e-10))
                local=(np.maximum(log,log.max()-8)+4)/4
                self.close_stage(features,local,'log and dynamic range')
                want=reference['input_features'][i].astype('f8');delta=features-want
                absolute=float(abs(delta).max());relative=float(np.linalg.norm(delta)/max(1e-30,np.linalg.norm(want)))
                Frontend.worst_absolute=max(Frontend.worst_absolute,absolute)
                Frontend.worst_relative=max(Frontend.worst_relative,relative)
                self.assertLessEqual(absolute,5e-5)
                self.assertLessEqual(relative,5e-6)

    def test_sample_and_chunk_boundaries(self):
        rng=np.random.default_rng(SPEC['seed'])
        for count in SPEC['sample_lengths']+[160*n for n in SPEC['frame_lengths']]:
            with self.subTest(samples=count): self.check([rng.uniform(-.3,.3,count).astype('f4')])

    def test_signals_and_unequal_clips(self):
        t=np.arange(16000,dtype='f8')/16000
        signals=dict(silence=np.zeros(16000),impulse=np.eye(1,16000,317).ravel(),sine=np.sin(2*np.pi*440*t),
                     quiet=1e-6*np.sin(2*np.pi*173*t),loud=100*np.sin(2*np.pi*991*t))
        for name in SPEC['signals']:
            with self.subTest(signal=name): self.check([signals[name].astype('f4')])
        rng=np.random.default_rng(SPEC['seed'])
        self.check([rng.uniform(-.2,.2,n).astype('f4') for n in SPEC['unequal_lengths']])

    def test_longer_than_thirty_seconds_is_not_truncated(self):
        rng=np.random.default_rng(SPEC['seed'])
        start=time.monotonic()
        self.check([rng.uniform(-.3,.3,SPEC['long_samples']).astype('f4')])
        print(f'31-second frontend and independent checks: {time.monotonic()-start:.3f}s',flush=True)


if __name__=='__main__': unittest.main()
