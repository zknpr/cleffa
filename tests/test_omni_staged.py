"""Checkpoint corruption and changed snapshot inputs must stop reference reuse."""
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ref'))
import omni_staged
from omni_staged import check_snapshot, digest, inventory, load_state, save_state


class StagedOracle(unittest.TestCase):
    def test_failed_worker_retains_last_verified_checkpoint(self):
        import json
        import subprocess
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / 'ref').mkdir()
            model = root / 'model'; model.mkdir()
            (model / 'config.json').write_text(json.dumps({'thinker_config': {'text_config': {'num_hidden_layers': 2}}}))
            args = SimpleNamespace(model_dir=model, stage_layers=1, name='first')
            def child(command, **kwargs):
                job = json.loads(Path(command[-2]).read_text())
                if job['stage']['first'] == 0:
                    Path(job['stage']['output']).write_bytes(b'complete first stage')
                else:
                    raise subprocess.CalledProcessError(1, command)
            with patch.object(omni_staged, '__file__', str(root / 'ref/omni_staged.py')), \
                 patch('verify_snapshot.load_manifest', return_value={}), \
                 patch('verify_snapshot.verify_snapshot', return_value=[]), \
                 patch.object(omni_staged.subprocess, 'run', side_effect=child):
                with self.assertRaises(subprocess.CalledProcessError):
                    omni_staged.run(args)
            checkpoint = list((root / 'golden').glob('omni-stages-*/state-01.pt'))
            self.assertEqual(len(checkpoint), 1)
            self.assertEqual(checkpoint[0].read_bytes(), b'complete first stage')
            self.assertFalse((root / 'golden' / 'first').exists())

    def test_first_run_creates_golden_root(self):
        import json
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'ref').mkdir()
            model = root / 'model'
            model.mkdir()
            (model / 'config.json').write_text(json.dumps({'thinker_config': {'text_config': {'num_hidden_layers': 1}}}))
            args = SimpleNamespace(model_dir=model, stage_layers=1, name='first')
            def child(command, **kwargs):
                job = Path(command[-2])
                self.assertTrue(job.is_file())
                self.assertTrue((root / 'golden').is_dir())
            with patch.object(omni_staged, '__file__', str(root / 'ref/omni_staged.py')), \
                 patch('verify_snapshot.load_manifest', return_value={}), \
                 patch('verify_snapshot.verify_snapshot', return_value=[]), \
                 patch.object(omni_staged.subprocess, 'run', side_effect=child):
                omni_staged.run(args)

    def test_changed_or_added_snapshot_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / 'weight'
            path.write_bytes(b'one')
            expected = inventory(root)
            check_snapshot(root, expected)
            path.write_bytes(b'two')
            with self.assertRaisesRegex(ValueError, 'snapshot changed'):
                check_snapshot(root, expected)
            expected = inventory(root)
            (root / 'cached.pyc').write_bytes(b'new')
            with self.assertRaisesRegex(ValueError, 'snapshot changed'):
                check_snapshot(root, expected)

    def test_checkpoint_hash_stage_and_exact_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'state.pt'
            state = dict(next_layer=4, hidden=[torch.randn(1, 11, 8)],
                         extras=[(None, torch.randn(1, 11, 4))], dumps=[{'embed': torch.randn(11, 8)}])
            save_state({'output': path, 'policy_sha256': 'blocked-from-zero'}, state)
            stage = dict(input=path, input_sha256=digest(path), first=4, policy_sha256='blocked-from-zero')
            got = load_state(stage, 'cpu')
            self.assertTrue(torch.equal(got['hidden'][0], state['hidden'][0]))
            self.assertTrue(torch.equal(got['dumps'][0]['embed'], state['dumps'][0]['embed']))
            with self.assertRaisesRegex(ValueError, 'reference policy'):
                load_state({**stage, 'policy_sha256': 'different-attention'}, 'cpu')
            with self.assertRaisesRegex(ValueError, 'another stage'):
                load_state({**stage, 'first': 3}, 'cpu')
            with path.open('ab') as output:
                output.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                load_state(stage, 'cpu')
            with self.assertRaisesRegex(ValueError, 'replace'):
                save_state({'output': path}, state)


if __name__ == '__main__':
    unittest.main()
