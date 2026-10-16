"""Reference consolidation must preserve tensor bytes without retaining a record in RAM."""
import copy
import json
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'ref'))
import oracle_fp64 as legacy
import omni_fp64_policy as v1


def memory_probe(use_legacy=False):
    if use_legacy:
        capture = legacy.Capture
    else:
        from oracle_fp64_bounded import Capture as capture
    # Valid 256 MiB payload, created incrementally so setup cannot mask the peak.
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        sink = capture(root, 'probe', 1, 0)
        chunk = bytes(1 << 20)
        for index in range(16):
            name = f'layer.{index:02d}'
            header = json.dumps({name:dict(dtype='F64', shape=[(16 << 20)//8], data_offsets=[0, 16 << 20])}).encode()
            header += b' '*(-len(header) % 8)
            with (sink.path/(name+'.safetensors')).open('wb') as out:
                out.write(len(header).to_bytes(8, 'little')); out.write(header)
                for _ in range(16): out.write(chunk)
            sink.names.append(name)
        scale = 1 if sys.platform == 'darwin' else 1024
        before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*scale
        sink.finish(root/'record.safetensors')
        after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*scale
        with safe_open(root/'record.safetensors', framework='np') as tensors:
            assert len(tensors.keys()) == 16
            for name in tensors.keys():
                value = tensors.get_slice(name)
                assert value.get_shape() == [(16 << 20)//8]
                end = (16 << 20)//8
                assert not value[:2].any() and not value[end-2:end].any()
        result = dict(payload_bytes=256 << 20, before_peak_bytes=before, after_peak_bytes=after,
                      extra_peak_bytes=max(0, after-before), limit_bytes=64 << 20, legacy=use_legacy)
        print(json.dumps(result), flush=True)
        if result['extra_peak_bytes'] > result['limit_bytes']:
            raise AssertionError('consolidation retained more than 64 MiB for a 256 MiB record')


class Capture(unittest.TestCase):
    def setUp(self):
        import oracle_fp64_bounded as bounded
        self.bounded = bounded
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def sink(self):
        return self.bounded.Capture(self.root, 'probe', 6, 2)

    def test_matches_original_dtypes_shapes_suffixes_and_signed_zero_bytes(self):
        sink = self.sink()
        other_root = self.root/'old'; other_root.mkdir()
        other = legacy.Capture(other_root, 'probe', 6, 2)
        values = {'embed':torch.arange(18, dtype=torch.float64).reshape(6, 3),
                  'routing.00.ids':torch.arange(48, dtype=torch.int64).reshape(6, 8),
                  'audio.0.features':torch.tensor([[-0., 1., 0.]], dtype=torch.float32),
                  'vision.0':torch.tensor([[-0., 1e-300, 1e300]], dtype=torch.float64)}
        for name, value in values.items():
            for capture in (sink, other):capture.put(name, value, media=name.startswith(('audio.', 'vision.')))
        sink.finish(self.root/'new.safetensors'); other.finish(self.root/'old.safetensors')
        with safe_open(self.root/'new.safetensors', framework='np') as new, safe_open(self.root/'old.safetensors', framework='np') as old:
            self.assertEqual(set(new.keys()), set(old.keys()))
            for key in old.keys():
                a, b = new.get_tensor(key), old.get_tensor(key)
                self.assertEqual((a.dtype, a.shape, a.tobytes()), (b.dtype, b.shape, b.tobytes()))
        self.assertFalse(sink.path.exists())
        self.assertEqual(sink.media_shapes, other.media_shapes)

    def test_existing_destination_and_parts_survive(self):
        sink = self.sink(); sink.put('embed', torch.ones((6, 2), dtype=torch.float64))
        target = self.root/'existing.safetensors'; target.write_bytes(b'keep existing evidence')
        before = (sink.path/'embed.safetensors').read_bytes()
        with self.assertRaises(FileExistsError):sink.finish(target)
        self.assertEqual(target.read_bytes(), b'keep existing evidence')
        self.assertEqual((sink.path/'embed.safetensors').read_bytes(), before)
        self.assertEqual(set(self.root.iterdir()), {self.root/'parts', target})

    def test_write_failure_never_publishes_a_partial_file(self):
        sink = self.sink(); sink.put('embed', torch.ones((6, 2), dtype=torch.float64))
        before = (sink.path/'embed.safetensors').read_bytes()
        with patch.object(self.bounded.os, 'fsync', side_effect=OSError('injected write failure')):
            with self.assertRaisesRegex(OSError, 'injected'):sink.finish(self.root/'record.safetensors')
        self.assertFalse((self.root/'record.safetensors').exists())
        self.assertEqual((sink.path/'embed.safetensors').read_bytes(), before)
        self.assertEqual(set(self.root.iterdir()), {self.root/'parts'})

    def test_invalid_parts_are_preserved_without_output(self):
        for mode in ('truncated', 'trailing', 'huge-header', 'duplicate-key', 'wrong-name', 'unknown-file'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); sink = self.bounded.Capture(root, 'probe', 2, 0)
                sink.put('embed', torch.ones((2, 2), dtype=torch.float64))
                part = sink.path/'embed.safetensors'; raw = part.read_bytes()
                if mode == 'truncated':part.write_bytes(raw[:-1])
                elif mode == 'trailing':part.write_bytes(raw+b'extra')
                elif mode == 'huge-header':part.write_bytes((1 << 40).to_bytes(8, 'little'))
                elif mode == 'duplicate-key':
                    header = b'{"embed":{},"embed":{}}'
                    part.write_bytes(len(header).to_bytes(8, 'little')+header)
                elif mode == 'wrong-name':sink.names = ['other']
                else:(sink.path/'keep.txt').write_text('unrelated')
                before = {q.name:q.read_bytes() for q in sink.path.iterdir()}
                with self.assertRaises((ValueError, OSError)):sink.finish(root/'record.safetensors')
                self.assertEqual({q.name:q.read_bytes() for q in sink.path.iterdir()}, before)
                self.assertEqual(set(root.iterdir()), {root/'parts'})

    def test_nonfinite_and_wrong_precision_still_rejected_before_serialization(self):
        sink = self.sink()
        for value in (torch.ones((6, 2), dtype=torch.float32), torch.full((6, 2), float('nan'), dtype=torch.float64)):
            with self.assertRaises(ValueError):sink.put('embed', value)
        self.assertFalse(list(sink.path.iterdir()))

    def test_consolidation_peak_is_bounded(self):
        result = subprocess.run([sys.executable, '-B', __file__, '--memory-probe'], text=True, capture_output=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        report = json.loads(result.stdout)
        self.assertLessEqual(report['extra_peak_bytes'], report['limit_bytes'])
        print('Capture memory:', json.dumps(report), flush=True)


class Policy(unittest.TestCase):
    def setUp(self):
        import oracle_fp64_bounded as bounded
        self.bounded = bounded
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ('joint_schema_model.py', 'config.json', 'joint_head_config.json'):
            (self.root/name).write_text('unit fixture only')

    def manifest(self, policy):
        return dict(complete=True, revision=v1.REVISION, precision='float64-streamed', independent_from_native=True,
                    fp64_policy=copy.deepcopy(policy.POLICY), implementation_sha256=policy.implementation_sources(),
                    corpus='text', **policy.runtime_provenance(self.root, 'text'))

    def test_both_versions_validate_without_changing_v1(self):
        old = self.manifest(v1); new = self.manifest(self.bounded)
        v1.validate_policy(old)
        self.bounded.validate_policy(old); self.bounded.validate_policy(new)
        with self.assertRaises(ValueError):v1.validate_policy(new)
        new['implementation_sha256'] = old['implementation_sha256']
        with self.assertRaises(ValueError):self.bounded.validate_policy(new)

    def test_bounded_policy_still_requires_independence_precision_sources_and_runtime(self):
        good = self.manifest(self.bounded)
        changes = [('complete', False), ('independent_from_native', False), ('arbitration', {}),
                   ('precision', 'float32-streamed'), ('implementation_sha256', {}), ('safetensors', 'stale')]
        for key, value in changes:
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.bounded.validate_policy({**good, key:value})
        for key, value in [('name', 'unknown'), ('capture', 'unbounded'), ('max_probability_error', 1.)]:
            changed = copy.deepcopy(good); changed['fp64_policy'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):self.bounded.validate_policy(changed)
        changed = copy.deepcopy(good); changed.pop('source_sha256')
        with self.assertRaises(ValueError):self.bounded.validate_policy(changed)

    def test_cli_adapter_restores_executor_bindings_after_failure(self):
        keys = ('Capture', 'POLICY', 'implementation_sources', 'runtime_provenance')
        before = {key:getattr(legacy, key) for key in keys}
        def fail():
            self.assertIs(legacy.Capture, self.bounded.Capture)
            self.assertEqual(legacy.POLICY, self.bounded.POLICY)
            self.assertIn('ref/oracle_fp64_bounded.py', legacy.implementation_sources())
            raise RuntimeError('injected oracle failure')
        with patch.object(legacy, 'main', side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, 'injected'):self.bounded.main()
        self.assertEqual(before, {key:getattr(legacy, key) for key in keys})


if __name__ == '__main__':
    if sys.argv[1:] == ['--memory-probe']:memory_probe()
    elif sys.argv[1:] == ['--legacy-memory-probe']:memory_probe(True)
    else:unittest.main()
