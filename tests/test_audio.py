"""Native WAV/sample decoding must preserve independent PCM/float32 bytes."""
import base64
import io
import json
from pathlib import Path
import struct
import subprocess
import unittest
import wave

import numpy as np

ROOT=Path(__file__).resolve().parents[1]


class Audio(unittest.TestCase):
    def tool(self, value):
        p=subprocess.run([str(ROOT/'clef-tool'),'audio'],input=json.dumps(value)+'\n',capture_output=True,text=True,check=True)
        if p.stdout.startswith('ERR '): return p.stdout.strip()
        return bytes.fromhex(json.loads(p.stdout)['hex'])

    def test_pcm16_matches_independent_wave_reader(self):
        samples=np.concatenate((np.array([-32768,-1,0,1,32767],dtype='<i2'),
                                np.random.default_rng(73).integers(-32768,32768,16000,dtype=np.int16)))
        output=io.BytesIO()
        with wave.open(output,'wb') as wav:
            wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(16000);wav.writeframes(samples.tobytes())
        with wave.open(io.BytesIO(output.getvalue()),'rb') as wav:
            want=np.frombuffer(wav.readframes(wav.getnframes()),dtype='<i2').astype('<f4')/32768
        got=self.tool(dict(wav=base64.b64encode(output.getvalue()).decode()))
        self.assertEqual(got,want.tobytes())

    def test_float32_payload_and_numeric_arrays(self):
        bits=np.array([0,0x80000000,1,0x007fffff,0x00800000,0x3f800000,0xc0000000,0x7f7fffff],dtype='<u4')
        fmt=struct.pack('<HHIIHH',3,1,16000,64000,4,32)
        body=b'WAVEfmt '+struct.pack('<I',len(fmt))+fmt+b'data'+struct.pack('<I',bits.nbytes)+bits.tobytes()
        wav=b'RIFF'+struct.pack('<I',len(body))+body
        self.assertEqual(self.tool(dict(wav=base64.b64encode(wav).decode())),bits.tobytes())
        samples=[-1,0,1,0.125,-1e-40,1e30]
        self.assertEqual(self.tool(dict(samples=samples)),np.asarray(samples,dtype='<f4').tobytes())

    def test_errors_and_recovery(self):
        for value in ({'samples':[]},{'samples':[True]},{'samples':[1e100]}, {'wav':'AAAA'}, {}):
            self.assertTrue(self.tool(value).startswith('ERR '))
        self.assertEqual(self.tool(dict(samples=[.5])),struct.pack('<f',.5))


if __name__=='__main__': unittest.main()
