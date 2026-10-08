"""JPEG scan-work, RGB and non-interleaved DC regressions from review #3.

Run with .venv/bin/python -B tests/test_jpeg_regressions.py. Fixtures are checked in;
--generate-fixtures PATH regenerates the cjpeg cases, requiring libjpeg-turbo's cjpeg.
--benchmark reports the original 4096x4096 empty-refinement attack through clef-tool.
CLEF_TOOL can select an ASan/UBSan build of the same host entry point.
"""
import base64
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'tests/fixtures/jpeg_scans.json'


def segment(marker, payload):
    return bytes([255, marker]) + (len(payload) + 2).to_bytes(2, 'big') + payload


def progressive(scans, width=64, height=64, entropy=True):
    """One grayscale component, quantizer 1, one-bit zero DC/EOB codes."""
    data = b'\xff\xd8' + segment(0xdb, b'\0' + bytes([1]) * 64)
    data += segment(0xc2, b'\x08' + height.to_bytes(2, 'big') + width.to_bytes(2, 'big') + b'\x01\x01\x11\0')
    for table in [0, 16]:
        data += segment(0xc4, bytes([table, 1]) + bytes(15) + b'\0')
    blocks = ((width + 7) // 8) * ((height + 7) // 8)
    for index, (ss, se, ah, al) in enumerate(scans):
        data += segment(0xda, bytes([1, 1, 0, ss, se, ah * 16 + al]))
        if entropy is True or isinstance(entropy, set) and index in entropy:
            data += bytes((blocks + 7) // 8)
    return data + b'\xff\xd9'


def spectral_work_bomb(width=4096, height=4096):
    # Valid progression headers still permit 882 AC traversals, even if no AC
    # entropy follows. Progression validation alone cannot bound decode CPU tightly.
    scans = [(0, 0, 0, 0)]
    for k in range(1, 64):
        scans += [(k, k, 0, 13)] + [(k, k, ah, ah - 1) for ah in range(13, 0, -1)]
    return progressive(scans, width, height, entropy={0})


def decode(data):
    with tempfile.TemporaryDirectory(prefix='jpeg-regression-') as td:
        prefix = Path(td) / 'out'
        request = {'image': base64.b64encode(data).decode(), 'out': str(prefix),
                   'min_pixels': 1024, 'max_pixels': 1024}
        start = time.monotonic()
        p = subprocess.run([os.environ.get('CLEF_TOOL', str(ROOT / 'clef-tool')), 'image'],
                           input=json.dumps(request) + '\n', text=True, capture_output=True,
                           check=True, timeout=20)
        elapsed = time.monotonic() - start
        if p.stdout.startswith('ERR'):
            return None, elapsed
        meta = json.loads(p.stdout)
        width, height = meta['decoded']
        return np.fromfile(str(prefix) + '.rgb', np.uint8).reshape(height, width, 3), elapsed


def without_app(data, marker):
    """Remove one APP marker before SOS, leaving entropy bytes unchanged."""
    out, at = data[:2], 2
    while at < len(data):
        kind = data[at + 1]
        if kind == 0xda:
            return out + data[at:]
        length = int.from_bytes(data[at + 2:at + 4], 'big') + 2
        if kind != marker:
            out += data[at:at + length]
        at += length
    raise AssertionError('missing SOS')


def headers(data):
    at = 2
    while at + 4 <= len(data):
        kind = data[at + 1]
        end = at + 2 + int.from_bytes(data[at + 2:at + 4], 'big')
        yield kind, at, end
        at = end
        if kind == 0xda:
            while at + 1 < len(data) and not (data[at] == 255 and data[at + 1] not in (0, *range(0xd0, 0xd8))):
                at += 1


def numeric_rgb_ids(data):
    result = bytearray(data)
    for kind, at, _ in headers(data):
        if kind in (0xc0, 0xc2):
            positions = [at + 10 + 3*i for i in range(data[at + 9])]
        elif kind == 0xda:
            positions = [at + 5 + 2*i for i in range(data[at + 4])]
        else:
            continue
        for pos in positions:
            result[pos] = {ord('R'): 1, ord('G'): 2, ord('B'): 3}[data[pos]]
    return bytes(result)


def generate_fixtures(path):
    cases = []
    with tempfile.TemporaryDirectory(prefix='jpeg-fixtures-') as td:
        script = Path(td) / 'scans.txt'
        # Separate initial and refinement DC scans, with full AC in between.
        script.write_text(''.join(f'{c}: 0 0 0 1;\n' for c in range(3)) +
                          ''.join(f'{c}: 1 63 0 0;\n' for c in range(3)) +
                          ''.join(f'{c}: 0 0 1 0;\n' for c in range(3)))
        for width, height in [(64, 48), (57, 41)]:
            y, x = np.indices((height, width))
            rgb = np.stack([(x * 4 + y) % 256, (y * 5 + x) % 256, (x * 3 + y * 2) % 256], -1).astype(np.uint8)
            ppm = f'P6\n{width} {height}\n255\n'.encode() + rgb.tobytes()
            for sampling in ['1x1', '2x1', '2x2']:
                for restart in ['0', '2B']:
                    p = subprocess.run(['cjpeg', '-quality', '90', '-sample', sampling + ',1x1,1x1',
                                        '-scans', str(script), '-restart', restart], input=ppm,
                                       capture_output=True, check=True)
                    cases.append({'name': f'{width}x{height}-{sampling}-restart{restart}',
                                  'base64': base64.b64encode(p.stdout).decode()})
    Path(path).write_text(json.dumps(cases, indent=2) + '\n')


class JPEGRegressions(unittest.TestCase):
    def parity(self, data):
        expected = np.asarray(Image.open(io.BytesIO(data)).convert('RGB'))
        actual, _ = decode(data)
        self.assertIsNotNone(actual, 'valid JPEG rejected')
        self.assertEqual(actual.shape, expected.shape)
        wrong = np.count_nonzero(actual != expected)
        self.assertEqual(wrong, 0, f'{wrong} channels differ; max error {np.abs(actual.astype(int) - expected).max()}')

    def test_invalid_progression(self):
        dc = (0, 0, 0, 1)
        bad = {
            'empty-refinement-bomb': ([(0, 0, 1, 0)] * 1000, False),
            'nonempty-refinement-before-initial': ([(0, 0, 1, 0)], True),
            'repeated-first-dc': ([dc, dc], True),
            'repeated-dc-refinement': ([dc, (0, 0, 1, 0), (0, 0, 1, 0)], True),
            'skipped-bitplane': ([(0, 0, 0, 3), (0, 0, 3, 1)], True),
            'wrong-prior-bitplane': ([(0, 0, 0, 3), (0, 0, 2, 1)], True),
            'unchanged-bitplane': ([dc, (0, 0, 1, 1)], True),
            'ac-before-dc': ([(1, 63, 0, 0)], True),
            'ac-refine-before-initial': ([dc, (1, 63, 1, 0)], True),
            'repeated-ac-first': ([dc, (1, 5, 0, 1), (1, 5, 0, 1)], True),
            'overlapping-ac-bands': ([dc, (1, 5, 0, 1), (4, 9, 0, 1)], True),
            'repeated-ac-refinement': ([dc, (1, 5, 0, 1), (1, 5, 1, 0), (1, 5, 1, 0)], True),
        }
        for name, (scans, entropy) in bad.items():
            with self.subTest(name=name):
                self.assertIsNone(decode(progressive(scans, entropy=entropy))[0])

    def test_valid_split_progression(self):
        scans = [(0, 0, 0, 13)] + [(0, 0, ah, ah - 1) for ah in range(13, 0, -1)]
        # Every coefficient can have its own initial/refinement scan. A scan-count shortcut
        # must not reject this legal 203-scan progression or confuse neighboring bands.
        scans += [(k, k, 0, 2) for k in range(1, 64)]
        scans += [(k, k, ah, ah - 1) for ah in [2, 1] for k in range(1, 64)]
        self.parity(progressive(scans))

    def test_scan_work_limit(self):
        self.assertIsNone(decode(spectral_work_bomb())[0])
        # With the host's 64 Mpx source cap, the work budget is 16 M block visits:
        # exactly 64 grayscale 4096x4096 scans. Even fully encoded, legal progressions
        # must fail once they exceed it; small high-scan-count files remain supported.
        # Complete the low-frequency AC coefficients so libjpeg needs no block smoothing;
        # otherwise the upstream incomplete-image refusal would mask the work boundary.
        scans = [(0, 0, 0, 0)] + [(k, k, 0, 0 if k < 10 else 1) for k in range(1, 64)]
        pixels, _ = decode(progressive(scans, 4096, 4096))
        self.assertIsNotNone(pixels)
        self.assertTrue(np.all(pixels == 128))
        self.assertIsNone(decode(progressive(scans + [(63, 63, 1, 0)], 4096, 4096))[0])

    def test_rgb(self):
        y, x = np.indices((48, 64))
        gradient = np.stack([(4*x) % 256, (5*y) % 256, (x+y) % 256], -1).astype(np.uint8)
        for progressive_mode in [False, True]:
            for pixels in [np.full((48, 64, 3), [255, 0, 0], np.uint8), gradient]:
                image = Image.fromarray(pixels)
                b = io.BytesIO(); image.save(b, 'JPEG', keep_rgb=True, progressive=progressive_mode, quality=90)
                data = b.getvalue()
                # Adobe transform 0 and ASCII RGB IDs are independent ways to signal RGB.
                for variant in [data, without_app(data, 0xee)]:
                    with self.subTest(progressive=progressive_mode, adobe=variant == data, solid=np.all(pixels == pixels[0, 0])):
                        self.parity(variant)

    def test_single_component_dc(self):
        for fixture in json.loads(FIXTURES.read_text()):
            with self.subTest(name=fixture['name']):
                self.parity(base64.b64decode(fixture['base64']))

    def test_color_marker_precedence(self):
        adobe = lambda transform: segment(0xee, b'Adobe\0d\0\0\0\0' + bytes([transform]))
        jfif = segment(0xe0, b'JFIF\0\x01\x01\0\0\x01\0\x01\0\0')
        for mode in [False, True]:
            buf = io.BytesIO()
            Image.new('RGB', (64, 48), (213, 57, 109)).save(buf, 'JPEG', keep_rgb=True, progressive=mode)
            rgb = buf.getvalue()
            numeric = numeric_rgb_ids(rgb)
            no_adobe = without_app(numeric, 0xee)
            sos = next(start for marker, start, _ in headers(no_adobe) if marker == 0xda)
            variants = {
                'adobe-numeric-ids': numeric,
                'adobe-after-sof': no_adobe[:sos] + adobe(0) + no_adobe[sos:],
                'adobe-ycbcr-over-rgb-ids': rgb[:2] + adobe(1) + without_app(rgb, 0xee)[2:],
                'jfif-over-adobe-rgb': rgb[:2] + jfif + rgb[2:],
            }
            for name, data in variants.items():
                with self.subTest(progressive=mode, name=name):
                    self.parity(data)
            # Match vision's libjpeg-compatible marker handling: unknown Adobe transforms
            # select YCbCr, while short Adobe payloads do not declare a colour space.
            for marker in [adobe(2), segment(0xee, b'Adobe\0d')]:
                with self.subTest(progressive=mode, marker=marker):
                    self.parity(no_adobe[:2] + marker + no_adobe[2:])
            short_jfif = no_adobe[:2] + segment(0xe0, b'JFIF\0') + no_adobe[2:]
            self.assertIsNone(decode(short_jfif)[0])


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--generate-fixtures':
        generate_fixtures(sys.argv[2])
    elif sys.argv[1:] == ['--benchmark']:
        for scans in [1, 1000]:
            data = progressive([(0, 0, 1, 0)] * scans, 4096, 4096, False)
            rgb, elapsed = decode(data)
            print(f'{scans} scans, {len(data)} bytes: {"REFUSED" if rgb is None else "ACCEPTED"} in {elapsed:.6f} s')
        data = spectral_work_bomb()
        rgb, elapsed = decode(data)
        print(f'883 split-band scans, {len(data)} bytes: {"REFUSED" if rgb is None else "ACCEPTED"} in {elapsed:.6f} s')
    else:
        unittest.main()
