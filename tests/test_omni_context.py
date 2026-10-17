"""Host context boundaries. These checks do not claim long-context numerical parity."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path(sys.argv.pop(1)) if len(sys.argv) > 1 and not sys.argv[1].startswith('-') else ROOT/'gguf/clef-omni.gguf'


def encode(req, maximum=None):
    cmd = [str(ROOT/'clef-tool'), 'encode-notrunc', str(MODEL)]
    if maximum:
        cmd += ['--max-length', str(maximum)]
    p = subprocess.run(cmd, input=json.dumps(req)+'\n', capture_output=True, text=True, check=True)
    return p.stdout.strip()


def request(repetitions):
    return dict(model='clef-omni', state=' alpha'*repetitions+' FINAL: billing outage.',
                questions={'outage': dict(type='noul', instructions='Is a service down?')})


def exact_request(length):
    baseline = len(json.loads(encode(request(1), 65536))['input_ids'])
    req = request(length-baseline+1)
    encoded = json.loads(encode(req, 65536))
    if len(encoded['input_ids']) != length:
        raise AssertionError('fixture did not reach its exact intended token count')
    return req, encoded


class OmniContext(unittest.TestCase):
    def test_exact_boundaries_and_explicit_override(self):
        for length in (16383, 16384, 16385, 63999, 64000, 64001, 65536):
            with self.subTest(length=length):
                req, full = exact_request(length)
                default = encode(req)
                if length <= 64000:
                    self.assertEqual(json.loads(default), full)
                else:
                    self.assertIn('reference would silently drop the rest', default)
                self.assertEqual(json.loads(encode(req, 65536)), full)

    def test_over_limit_is_rejected(self):
        p = subprocess.run([ROOT/'clef-tool', 'encode-notrunc', MODEL, '--max-length', '65537'],
                           input='', text=True, capture_output=True)
        self.assertEqual(p.returncode, 2)
        self.assertIn('invalid --max-length', p.stderr)

    def test_fixed_schema_cannot_be_truncated(self):
        req = request(1)
        req['questions']['outage']['instructions'] = ' instruction'*70000
        self.assertIn('ERR ', encode(req))

    def test_patch_encoder_accepts_context_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'patches.bin'
            p = subprocess.run([ROOT/'clef-tool', 'encode-patches', MODEL, path, '--max-length', '65536'],
                               input=json.dumps(request(1))+'\n', text=True, capture_output=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertEqual(path.read_bytes(), b'')


if __name__ == '__main__':
    unittest.main()
