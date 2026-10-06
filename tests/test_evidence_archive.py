"""The evidence archive must drop binaries, traces, tensors and private files, rewrite local
paths, and fail closed when a forbidden pattern survives."""
import json
import os
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import evidence_archive as ea  # noqa: E402

HOME = str(Path.home())
REPO = str(Path(__file__).resolve().parents[1])


def make_tree(root: Path):
    exp = root / 'gemm-probe-20261004'
    (exp / 'engine').mkdir(parents=True)
    (exp / 'result.json').write_text(json.dumps({'cli': f'{REPO}/clef', 'log': f'{HOME}/x.log', 'ms': 1.5}))
    (exp / 'run.py').write_text('print("hi")\n')
    (exp / 'engine' / 'clef').write_bytes(b'\xcf\xfa\xed\xfe' + b'\0' * 64)
    (exp / 'engine' / 'clef.o').write_bytes(b'\0' * 8)
    (exp / 'private-replay.json').write_text('{"n": 50}')
    (exp / 'replay.py').write_text('PATH = "~/dev/squid-bot/.personal/pop_v22.jsonl"\n')
    (exp / 'usage.json').write_text('{"account_id": "0123456789abcdef0123456789abcdef"}')
    (exp / 'journal.jsonl').write_text('{"account": "Zknpr", "calls": 3}\n')
    (exp / 'calls.jsonl').write_text('{"url": "https://api.cloudflare.com/client/v4/accounts/'
                                      '0123456789abcdef0123456789abcdef/ai/run/@cf/x", "ms": 2}\n')
    (exp / 'capture.trace').mkdir()
    (exp / 'capture.trace' / 'data.bin').write_bytes(b'\0' * 8)
    (exp / 'gpu-values.xml').write_text('<x/>')
    (exp / 'latin.log').write_bytes(b'path /Volumes/scratch/x \xff\xfe not utf-8\n')
    (exp / 'small.safetensors').write_bytes(b'\0' * 100)
    (exp / 'big.safetensors').write_bytes(b'\0' * (ea.SMALL_TENSOR + 1))
    art = root / 'article-batch-20261005'
    art.mkdir()
    (art / 'result.json').write_text('{"articles_per_s": 0.7}')
    ds4 = root / 'ds4-qwen-perf-20261004'
    (ds4 / 'source' / '.git').mkdir(parents=True)
    (ds4 / 'source' / 'ds4.c').write_text('int main(){}')
    (ds4 / 'groups-summary.json').write_text('{"groups": 4}')
    oracle = root / 'clef-flash-f32'
    (oracle / 'layers').mkdir(parents=True)
    (oracle / 'layers' / '0.safetensors').write_bytes(b'\0' * 10)
    (oracle / 'logits.safetensors').write_bytes(b'\0' * 10)
    (oracle / 'requests.jsonl').write_text('{"id":"r000"}\n')
    (oracle / 'latency.json').write_text('{}')
    (root / 'engine_logits.jsonl').write_text('{}\n')
    (root / 'engine_dump.bin').write_bytes(b'\0' * 4)
    (root / '.gpu.lock').write_text('')


class EvidenceArchive(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / 'golden'
        self.root.mkdir()
        make_tree(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_selection_and_rewrite(self):
        out = Path(self.tmp.name) / 'ev.tar.gz'
        manifest = ea.build(self.root, out, 'ev')
        names = {e['path'] for e in manifest['files']}
        self.assertEqual(names, {
            'gemm-probe-20261004/result.json', 'gemm-probe-20261004/run.py',
            'gemm-probe-20261004/journal.jsonl', 'gemm-probe-20261004/small.safetensors',
            'ds4-qwen-perf-20261004/groups-summary.json',
            'clef-flash-f32/logits.safetensors', 'clef-flash-f32/requests.jsonl', 'clef-flash-f32/latency.json',
            'engine_logits.jsonl',
        })
        with tarfile.open(out) as tar:
            members = {m.name for m in tar.getmembers()}
            self.assertIn('ev/README.md', members)
            self.assertIn('ev/manifest.json', members)
            f = tar.extractfile('ev/gemm-probe-20261004/result.json'); assert f
            result = json.loads(f.read())
            f = tar.extractfile('ev/gemm-probe-20261004/journal.jsonl'); assert f
            journal = f.read().decode()
        self.assertEqual(result['cli'], '<repo>/clef')
        self.assertEqual(result['log'], '<home>/x.log')
        self.assertIn('<cf-account>', journal)
        by_path = {e['path']: e for e in manifest['files']}
        self.assertTrue(by_path['gemm-probe-20261004/result.json']['rewritten'])
        self.assertFalse(by_path['gemm-probe-20261004/run.py']['rewritten'])
        self.assertNotEqual(by_path['gemm-probe-20261004/result.json']['sha256'],
                            by_path['gemm-probe-20261004/result.json']['source_sha256'])
        ex = manifest['excluded_counts']
        self.assertNotIn('gemm-probe-20261004/calls.jsonl', names)  # account ID inside a recorded URL
        self.assertNotIn('gemm-probe-20261004/latin.log', names)  # undecodable text is never archived raw
        self.assertIn('undecodable text', manifest['excluded_counts'])
        for reason in ('binary', 'private-named file', 'forbidden content', 'private workload directory',
                       'upstream clone', 'trace or dSYM bundle', 'tensor size', 'oracle directory'):
            self.assertIn(reason, ex, reason)

    def test_refuses_to_overwrite(self):
        out = Path(self.tmp.name) / 'ev.tar.gz'
        out.write_bytes(b'')
        with self.assertRaises(SystemExit):
            ea.build(self.root, out, 'ev')

    def test_final_scan_fails_closed(self):
        # A token-like line is excluded by the pre-filter; if the pre-filter is bypassed, the
        # final scan over archived bytes must stop the build instead of publishing it.
        exp = self.root / 'gemm-probe-20261004'
        (exp / 'token.log').write_text('Authorization: Bearer abcdefghijklmnopqrstuvwxyz0123\n')
        out = Path(self.tmp.name) / 'ev2.tar.gz'
        manifest = ea.build(self.root, out, 'ev')
        self.assertNotIn('gemm-probe-20261004/token.log', {e['path'] for e in manifest['files']})
        self.assertIn('forbidden content', manifest['excluded_counts'])
        orig = ea.prepare
        ea.prepare = lambda p: (p.read_bytes(), False, 'included') if p.name == 'token.log' else orig(p)
        try:
            out2 = Path(self.tmp.name) / 'ev3.tar.gz'
            with self.assertRaises(SystemExit):
                ea.build(self.root, out2, 'ev')
            self.assertFalse(out2.exists())
        finally:
            ea.prepare = orig

    def test_deterministic(self):
        a = Path(self.tmp.name) / 'a.tar.gz'
        b = Path(self.tmp.name) / 'b.tar.gz'
        ea.build(self.root, a, 'ev')
        ea.build(self.root, b, 'ev')
        with tarfile.open(a) as ta, tarfile.open(b) as tb:
            ma = [(m.name, m.size, m.mtime) for m in ta.getmembers() if m.name != 'ev/manifest.json']
            mb = [(m.name, m.size, m.mtime) for m in tb.getmembers() if m.name != 'ev/manifest.json']
        self.assertEqual(ma, mb)
        self.assertEqual(os.path.getsize(a), os.path.getsize(b))


if __name__ == '__main__':
    unittest.main()
