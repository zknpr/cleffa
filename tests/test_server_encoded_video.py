"""Real strict HTTP encoded-video bodies, recovery and concurrent codec workers."""
from concurrent.futures import ThreadPoolExecutor
import base64
import json
from pathlib import Path
import sys
import tempfile
import unittest

import test_omni_server as common
import test_encoded_video as fixture

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from video_request import convert


class EncodedVideoServer(unittest.TestCase):
    server = common.OmniServer.server
    post = common.OmniServer.post
    clip = fixture.EncodedVideo.clip

    def test_identical_body_decoding_and_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.path = Path(tmp) / 'clip.mov'
            for model, family, audio in [('clef-flash', 'qwen3_5', False),
                                         ('clef-omni-full', 'qwen3_omni_moe', True)]:
                with self.subTest(model=model):
                    self.model = ROOT / 'gguf' / (model + '.gguf')
                    self.clip(audio=audio)
                    template = dict(model='clef', state='Does anything move?',
                                    questions={'moving': {'type': 'noul'}})
                    frames = json.loads(convert(self.path, template, model_family=family))
                    encoded = {**template, 'videos': [{'content_type': 'video/quicktime',
                               'base64': base64.b64encode(self.path.read_bytes()).decode()}]}
                    # A soft cap above the processor's hard cap must still accept
                    # bounded clips, exactly as the original frame-array path did.
                    with self.server('--max-video-frames', '1000') as (port, log):
                        status, expected = self.post(port, frames)
                        self.assertEqual(status, 200, expected)
                        with ThreadPoolExecutor(max_workers=2) as pool:
                            actual = list(pool.map(lambda _: self.post(port, encoded), range(2)))
                        self.assertEqual(actual, [(200, expected)] * 2)
                        bad = {**template, 'videos': ['data:video/mp4;base64,AAAA']}
                        status, body = self.post(port, bad)
                        self.assertEqual(status, 400, body)
                        self.assertEqual(self.post(port, encoded), (200, expected))


if __name__ == '__main__':
    unittest.main()
