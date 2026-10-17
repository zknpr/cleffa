"""Hosted collection may mark completion only after all calls succeed."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import copy
import hashlib
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench'))
import cloudflare_omni


class OmniHosted(unittest.TestCase):
    def test_comparison_rejects_other_weights_or_reference_revision(self):
        import compare_omni_cloudflare as comparison
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            golden = root / 'golden' / 'fixture'; golden.mkdir(parents=True)
            captures = root / 'captures'; captures.mkdir()
            (root / 'clef').write_bytes(b'current executable')
            model = root / 'model.gguf'; model.write_bytes(b'intended weights')
            manifest = dict(complete=True, precision='float32-streamed', revision=comparison.REVISION)
            paired = dict(complete=True, binary_sha256=hashlib.sha256((root / 'clef').read_bytes()).hexdigest(),
                          model_sha256=hashlib.sha256(model.read_bytes()).hexdigest())
            plan = {'requests': [dict(model='clef-omni', id='probe', modality='text')]}
            inventory = {'text': dict(golden='fixture', captures=str(captures))}
            for target, key, value, message in (
                ('paired', 'model_sha256', 'other weights', 'different model weights'),
                ('reference', 'revision', 'other snapshot', 'reference revision'),
                ('paired', 'model_sha256', None, 'different model weights'),
            ):
                with self.subTest(target=target, value=value):
                    ref = dict(manifest); native = dict(paired)
                    (native if target == 'paired' else ref)[key] = value
                    (golden / 'manifest.json').write_text(json.dumps(ref))
                    (captures / 'manifest.json').write_text(json.dumps(native))
                    with patch.object(comparison, 'ROOT', root), \
                         patch.object(comparison, 'hosted_snapshot', return_value=(plan, {}, 'journal')):
                        with self.assertRaisesRegex(ValueError, message):
                            comparison.compare(root / 'journal', inventory, model)

    def test_video_transport_rejects_changed_samples_even_with_matching_tokens(self):
        import torch
        from cloudflare_omni_video import tensor_hashes
        from compare_omni_cloudflare import verify_video_transport
        from omni import REVISION
        source = {'state': 'public', 'questions': {}, 'videos': []}
        before = SimpleNamespace(input_ids=(1, 2), questions=(), media={'pixels': torch.arange(6).reshape(2, 3)})
        row = dict(request={'encoded': 'video'}, transport=dict(revision=REVISION,
                   native_request_sha256=hashlib.sha256(json.dumps(source).encode()).hexdigest(),
                   media_tensors=tensor_hashes(before.media)))
        expected = {'input_ids': [1, 2]}
        for change in ('none', 'pixels', 'tokens', 'proof'):
            after = copy.deepcopy(before); candidate = copy.deepcopy(row)
            if change == 'pixels': after.media['pixels'][0, 0] += 1
            if change == 'tokens': after.input_ids = (1, 3)
            if change == 'proof': candidate['transport']['media_tensors']['pixels']['sha256'] = 'wrong'
            with patch('omni_media.encode_reference', side_effect=[before, after]):
                if change == 'none':
                    verify_video_transport(candidate, source, expected, (None, None))
                else:
                    with self.assertRaises(ValueError):
                        verify_video_transport(candidate, source, expected, (None, None))

    def test_success_marker_and_failed_collection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); inventory = root / 'inventory.json'
            inventory.write_text('{}')
            for fail in (False, True):
                output = root / ('failed.jsonl' if fail else 'complete.jsonl')
                argv = ['collector', '--inventory', str(inventory), '--output', str(output), '--run', '--account', 'test']
                def collect(plan, out, account, token):
                    out.write(json.dumps({'type': 'call'}) + '\n')
                    if fail:
                        raise RuntimeError('injected transport failure')
                with patch.object(sys, 'argv', argv), \
                     patch.object(cloudflare_omni, 'plan', return_value={'type': 'plan', 'full_token_neuron_estimate': 1}), \
                     patch.object(cloudflare_omni, 'credentials', return_value=('account', 'not-a-real-token')), \
                     patch.object(cloudflare_omni, 'daily_usage', return_value={'type': 'budget', 'used_neurons': 0}), \
                     patch.object(cloudflare_omni, 'collect', side_effect=collect), \
                     patch.object(cloudflare_omni, 'final_usage'):
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, 'transport'):
                            cloudflare_omni.main()
                    else:
                        cloudflare_omni.main()
                markers = [r for r in cloudflare_omni.rows(output) if r['type'] == 'complete']
                self.assertEqual(len(markers), 0 if fail else 1)


if __name__ == '__main__':
    unittest.main()
