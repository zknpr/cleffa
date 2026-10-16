"""Encoded HTTP/CLI video bodies must match the already-qualified frame path."""
import base64
import copy
import importlib.util
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
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TOOL = Path(os.environ.get('CLEF_TOOL', ROOT / 'clef-tool'))
MEDIA = TOOL.parent / 'clef-media'
sys.path.insert(0, str(ROOT / 'tools'))
from video_request import convert
import test_omni_video_request as fixture


class EncodedVideo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('encoded_video_oracle', ROOT / 'model-omni/joint_schema_model.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        cls.load_video = staticmethod(module.load_video)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'clip.mov'

    clip = fixture.OmniVideoRequest.clip

    def native(self, request, family):
        weights = 'clef-omni' if family == 'qwen3_omni_moe' else 'clef-flash'
        patches = Path(self.tmp.name) / 'patches'
        result = subprocess.run([TOOL, 'encode-patches',
                                 ROOT / 'gguf' / (weights + '.gguf'), patches],
                                input=json.dumps(request) + '\n', text=True,
                                capture_output=True, timeout=45, check=True)
        self.assertFalse(result.stdout.startswith('ERR'), result.stdout)
        return json.loads(result.stdout), patches.read_bytes()

    def test_container_matches_frames_for_both_models(self):
        for family, audio, vfr in [('qwen3_5', False, False),
                                    ('qwen3_omni_moe', False, True),
                                    ('qwen3_omni_moe', True, False)]:
            with self.subTest(family=family, audio=audio, vfr=vfr):
                self.clip(audio=audio, vfr=vfr)
                request = {'model': 'clef', 'state': 'Describe movement.',
                           'questions': {'moving': {'type': 'noul'}},
                           'media_kwargs': {'videos_kwargs': {'size': {
                               'shortest_edge': 4096, 'longest_edge': 16384}}}}
                frames = json.loads(convert(self.path, request, model_family=family))
                expected = self.native(frames, family)
                encoded = copy.deepcopy(request)
                if family == 'qwen3_omni_moe':
                    encoded['media_kwargs'] = copy.deepcopy(frames['media_kwargs'])
                payload = base64.b64encode(self.path.read_bytes()).decode()
                for wire in ({'content_type': 'video/quicktime', 'base64': payload},
                             'data:video/quicktime;base64,' + payload):
                    encoded['videos'] = [wire]
                    self.assertEqual(self.native(encoded, family), expected)

    def test_worker_pixels_and_soundtrack_match_pinned_decoder(self):
        for codec, width, height, count, offset in [('libx264', 800, 640, 9, 0),
                                                   ('libx264', 96, 64, 9, .375),
                                                   ('libx265', 96, 64, 3, 0),
                                                   ('prores_ks', 800, 640, 3, 0),
                                                   ('mjpeg', 96, 64, 1, 0)]:
            with self.subTest(codec=codec, count=count, offset=offset):
                self.clip(width, height, count, audio=True, codec=codec, audio_offset=offset)
                frames, samples = self.load_video(str(self.path))
                run = subprocess.run([MEDIA, 'omni', '2', '0', '0', '32',
                                      '16777216', '4800000'], input=self.path.read_bytes(),
                                     capture_output=True, timeout=35, check=True)
                decoded = json.loads(run.stdout)
                actual = np.stack([np.asarray(Image.open(io.BytesIO(base64.b64decode(s.split(',')[1]))))
                                   for s in decoded['frames']])
                np.testing.assert_array_equal(actual, frames)
                with av.open(io.BytesIO(base64.b64decode(decoded['audio'].split(',')[1]))) as wav:
                    actual = np.concatenate([frame.to_ndarray().reshape(-1) for frame in wav.decode(audio=0)])
                np.testing.assert_array_equal(actual, samples)

    def test_rejects_bad_envelopes_and_decoder_limits(self):
        self.clip()
        payload = base64.b64encode(self.path.read_bytes()).decode()
        request = {'model': 'clef', 'state': 'test', 'questions': {'q': {'type': 'noul'}}}
        bad = ['https://example.com/clip.mp4', 'file:///etc/passwd',
               {'content_type': 'image/png', 'base64': payload},
               {'content_type': 'video/mp4', 'base64': payload, 'path': '/etc/passwd'},
               {'content_type': 'video/mp4', 'base64': 'AAAA'},
               'data:video/mp4;base64,' + base64.b64encode(b'\0\0\0\x08ftypbroken').decode()]
        for value in bad:
            result = subprocess.run([TOOL, 'encode', ROOT / 'gguf/clef-flash.gguf'],
                                    input=json.dumps({**request, 'videos': [value]}) + '\n',
                                    capture_output=True, text=True, timeout=35, check=True)
            self.assertTrue(result.stdout.startswith('ERR'), result.stdout)
        for frames, pixels, samples in [('2', '16777216', '4800000'),
                                        ('32', '32', '4800000'), ('32', '16777216', '100')]:
            self.clip(audio=True)
            result = subprocess.run([MEDIA, 'omni', '2', '0', '0', frames, pixels, samples],
                                    input=self.path.read_bytes(), capture_output=True, timeout=35)
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue(result.stderr)
            if pixels == '32':
                self.assertIn(b'invalid or excessive frame count, duration or dimensions', result.stderr)


if __name__ == '__main__':
    unittest.main()
