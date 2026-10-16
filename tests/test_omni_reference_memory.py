"""Reading all reference stages must release each payload mapping between tensors."""
import json
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file


def probe(legacy=False):
    if not legacy:
        from compare_omni_layers import load_reference_tensor
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'record.safetensors';header={};size=16 << 20
        for i in range(16):header[f'layer.{i:02d}']=dict(dtype='F64',shape=[size//8],data_offsets=[i*size,(i+1)*size])
        raw=json.dumps(header).encode();raw+=b' '*(-len(raw)%8)
        with path.open('wb') as output:
            output.write(len(raw).to_bytes(8,'little'));output.write(raw)
            block=bytes(1 << 20)
            for _ in range(256):output.write(block)
        scale=1 if sys.platform=='darwin' else 1024
        before=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*scale
        # Keep the metadata handle open, matching both canonical consumers.
        with safe_open(path,framework='np') as metadata:
            for key in metadata.keys():
                value=metadata.get_tensor(key) if legacy else load_reference_tensor(path,key)
                assert value.shape==(size//8,) and value.dtype==np.float64 and not value.any()
        after=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*scale
        result=dict(legacy=legacy,payload_bytes=256 << 20,extra_peak_bytes=max(0,after-before),limit_bytes=64 << 20)
        print(json.dumps(result),flush=True)
        if result['extra_peak_bytes']>result['limit_bytes']:raise AssertionError('reader retained more than 64 MiB for a 256 MiB record')


class ReferenceMemory(unittest.TestCase):
    def test_exact_dtype_shape_and_value_bytes(self):
        from compare_omni_layers import load_reference_tensor
        values={'double':np.array([-0.,1e-300,1e300],dtype='f8'),
                'float':np.array([[-0.,1.]],dtype='f4'),
                'integer':np.array([0,127],dtype='i8'),
                'scalar':np.array(3.,dtype='f8')}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'values.safetensors';save_file(values,path)
            for key,expected in values.items():
                actual=load_reference_tensor(path,key)
                self.assertEqual((actual.dtype,actual.shape,actual.tobytes()),(expected.dtype,expected.shape,expected.tobytes()))

    def test_bounded_peak_while_validating_every_tensor(self):
        result=subprocess.run([sys.executable,'-B',__file__,'--probe'],text=True,capture_output=True,timeout=60)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        report=json.loads(result.stdout)
        self.assertLessEqual(report['extra_peak_bytes'],report['limit_bytes'])
        print('Reference read memory:',json.dumps(report),flush=True)


if __name__=='__main__':
    if sys.argv[1:]==['--probe']:probe()
    elif sys.argv[1:]==['--legacy-probe']:probe(True)
    else:unittest.main()
