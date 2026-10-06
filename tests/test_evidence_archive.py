"""The evidence archive must drop binaries, traces, tensors and private files, rewrite local
paths, and fail closed when a forbidden pattern survives."""
import datetime
import hashlib
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
FIXED_MTIME = 1_759_700_000  # every fixture file gets this mtime; the manifest must derive from it
REPO = str(Path(__file__).resolve().parents[1])


def safetensors(header: dict) -> bytes:
    """Minimal safetensors container: 8-byte little-endian header length, JSON header, data."""
    body = json.dumps(header).encode()
    return len(body).to_bytes(8, 'little') + body + b'\0' * 16


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
    (exp / 'small.safetensors').write_bytes(safetensors({'__metadata__': {'format': 'pt'}, 'logits': {'dtype': 'F32', 'shape': [4], 'data_offsets': [0, 16]}}))
    (exp / 'leaky.safetensors').write_bytes(safetensors({'__metadata__': {'source': f'{HOME}/run/x.pt'}, 'logits': {'dtype': 'F32', 'shape': [4], 'data_offsets': [0, 16]}}))
    (exp / 'broken.safetensors').write_bytes(b'\0' * 100)   # not a safetensors header
    (exp / 'misshaped.safetensors').write_bytes(safetensors({'l': {'dtype': 'F32', 'shape': [1000], 'data_offsets': [0, 16]}}))   # 1000 floats in 16 bytes
    (exp / 'badshape.safetensors').write_bytes(safetensors({'l': {'dtype': 'F32', 'shape': [-4], 'data_offsets': [0, 16]}}))
    (exp / 'escaped.safetensors').write_bytes(len(b'{"__metadata__": {"source": "\\u002froot\\u002fsecret.txt"}, "l": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]}}').to_bytes(8, 'little')
                                             + b'{"__metadata__": {"source": "\\u002froot\\u002fsecret.txt"}, "l": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]}}' + b'\0' * 16)
    (exp / 'escaped.json').write_text('{"cli": "\\u002fUsers\\u002fsomeone\\u002fclef"}\n')   # escaped home path in JSON text
    (exp / 'escaped.jsonl').write_text('{"ok": 1}\n{"log": "\\u002froot\\u002fx.log"}\n')   # escaped path on one JSONL line
    named = exp / 'Bearer abcdefghijklmnopqrstuvwxyz0123'
    named.mkdir()
    (named / 'result.json').write_text('{"ok": 1}')   # forbidden string in a path component
    (exp / 'lower.log').write_text('authorization: bearer abcdefghijklmnopqrstuvwxyz0123\n')
    (exp / 'spaced.log').write_text('Authorization: Bearer   abcdefghijklmnopqrstuvwxyz0123\n')   # several spaces
    (exp / 'tabbed.log').write_text('Authorization: Bearer\tabcdefghijklmnopqrstuvwxyz0123\n')
    (exp / 'env.log').write_text('cloudflare_api_token=abcdefghijklmnopqrstuvwxyz0123\n')
    nested = exp / 'article-customer'
    nested.mkdir()
    (nested / 'requests.jsonl').write_text('{"state": "customer text"}\n')   # nested private workload
    (exp / '.env').write_text('OPENAI_API_KEY=sk-proj-abcdefghijklmnopqrstuvwxyz\n')   # credential dotfile
    (exp / 'notes').write_text('API_KEY=abcdefghijklmnopqrstuvwxyz0123\n')   # extensionless, not allowlisted
    (exp / 'dump.log').write_text('export MY_SECRET=abcdefghijklmnopqrstuvwxyz0123\n')   # key assignment in a log
    (exp / 'Makefile').write_text('all:\n\ttrue\n')
    (exp / 'creds.json').write_text('{"OPENAI_API_KEY": "abcdefghijklmnopqrst", "n": 1}\n')   # credential as a JSON field
    (exp / 'nested.json').write_text('{"env": {"password": "abcdefghijklmnopqrst"}}\n')
    (exp / 'counts.json').write_text('{"input_tokens": 4510, "token_env": "CLOUDFLARE_API_TOKEN"}\n')   # benign neighbours
    (exp / 'LICENSE').write_text('MIT\n')
    secret = b'Bearer abcdefghijklmnopqrstuvwxyz0123'
    (exp / 'late.log').write_bytes(b'ok line\n' * 700 + secret.decode().encode('utf-16-le') + b'\n')   # NULs past the 4 KiB probe
    hdr16 = json.dumps({'l': {'dtype': 'F16', 'shape': [len(secret)], 'data_offsets': [0, 2 * len(secret)]}}).encode()
    (exp / 'utf16.safetensors').write_bytes(len(hdr16).to_bytes(8, 'little') + hdr16 + secret.decode().encode('utf-16-le'))
    hdr8 = json.dumps({'blob': {'dtype': 'U8', 'shape': [len(secret)], 'data_offsets': [0, len(secret)]}}).encode()
    (exp / 'upper.SAFETENSORS').write_bytes(len(hdr8).to_bytes(8, 'little') + hdr8 + secret)   # suffix case variant
    hdr = json.dumps({'blob': {'dtype': 'U8', 'shape': [len(secret)], 'data_offsets': [0, len(secret)]}}).encode()
    (exp / 'payload.safetensors').write_bytes(len(hdr).to_bytes(8, 'little') + hdr + secret)   # text as tensor bytes
    (exp / 'usage-20261004.json').write_text('{"used_neurons": 1}')
    (exp / 'usage-model-27b.json').write_text('{"used_neurons": 1}')
    acme = root / 'customer-acme'
    acme.mkdir()
    (acme / 'requests.jsonl').write_text('{"state": "confidential"}\n')   # undated, oracle-shaped, not an oracle
    for name, home in (('root.safetensors', '/root/run/x.pt'), ('varroot.safetensors', '/var/root/x.pt'),
                       ('thishome.safetensors', f'{HOME}/x.pt')):
        (exp / name).write_bytes(safetensors({'__metadata__': {'source': home}, 'l': {'dtype': 'F32', 'shape': [4], 'data_offsets': [0, 16]}}))
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
    (oracle / 'logits.safetensors').write_bytes(safetensors({'logits': {'dtype': 'F32', 'shape': [4], 'data_offsets': [0, 16]}}))
    (oracle / 'requests.jsonl').write_text('{"id":"r000"}\n')
    (oracle / 'latency.json').write_text('{}')
    (root / 'engine_logits.jsonl').write_text('{}\n')
    (root / 'engine_logits-private.jsonl').write_text('{"state": "customer text"}\n')  # top-level, private-named
    (root / 'engine_dump.bin').write_bytes(b'\0' * 4)
    (root / '.gpu.lock').write_text('')
    priv = exp / 'private-customer'
    priv.mkdir()
    (priv / 'requests.jsonl').write_text('{"state": "customer text"}\n')   # private ancestor directory
    undated = root / 'customer-private'
    undated.mkdir()
    (undated / 'requests.jsonl').write_text('{"state": "customer text"}\n')  # undated, oracle-shaped
    for p in root.rglob('*'):
        os.utime(p, (FIXED_MTIME, FIXED_MTIME))


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
            'gemm-probe-20261004/Makefile', 'gemm-probe-20261004/LICENSE',
            'gemm-probe-20261004/counts.json',
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
        self.assertNotIn('gemm-probe-20261004/private-customer/requests.jsonl', names)  # private ancestor
        self.assertNotIn('customer-private/requests.jsonl', names)  # private undated directory
        self.assertNotIn('engine_logits-private.jsonl', names)  # private top-level file, ahead of the allowlist
        self.assertNotIn('gemm-probe-20261004/leaky.safetensors', names)  # forbidden string in the safetensors header
        self.assertNotIn('gemm-probe-20261004/broken.safetensors', names)  # unparseable safetensors header
        self.assertNotIn('gemm-probe-20261004/misshaped.safetensors', names)  # shape x dtype size != span
        self.assertNotIn('gemm-probe-20261004/badshape.safetensors', names)  # negative dimension
        self.assertNotIn('gemm-probe-20261004/payload.safetensors', names)  # a U8 tensor carrying text
        self.assertNotIn('gemm-probe-20261004/utf16.safetensors', names)  # text as UTF-16 inside an F32 payload
        self.assertNotIn('gemm-probe-20261004/late.log', names)  # UTF-16 text after a clean 4 KiB prefix
        self.assertNotIn('gemm-probe-20261004/.env', names)  # dotfiles never
        self.assertNotIn('gemm-probe-20261004/notes', names)  # extensionless only when allowlisted
        self.assertNotIn('gemm-probe-20261004/dump.log', names)  # key=value credential shape
        self.assertIn('gemm-probe-20261004/Makefile', names)
        self.assertNotIn('gemm-probe-20261004/creds.json', names)  # quoted credential key
        self.assertNotIn('gemm-probe-20261004/nested.json', names)
        self.assertIn('gemm-probe-20261004/counts.json', names)
        self.assertIn('gemm-probe-20261004/LICENSE', names)
        self.assertNotIn('gemm-probe-20261004/upper.SAFETENSORS', names)  # suffix case must not skip the checks
        self.assertFalse([n for n in names if 'Bearer' in n], 'forbidden string in a path component')
        self.assertNotIn('gemm-probe-20261004/lower.log', names)  # lowercase bearer
        self.assertNotIn('gemm-probe-20261004/spaced.log', names)  # whitespace run after the scheme
        self.assertNotIn('gemm-probe-20261004/tabbed.log', names)
        self.assertNotIn('gemm-probe-20261004/env.log', names)  # lowercase token variable
        self.assertNotIn('gemm-probe-20261004/article-customer/requests.jsonl', names)  # nested article dir
        self.assertNotIn('gemm-probe-20261004/usage-20261004.json', names)  # dated usage dump
        self.assertNotIn('gemm-probe-20261004/usage-model-27b.json', names)
        self.assertNotIn('customer-acme/requests.jsonl', names)  # only recognized oracle directories
        for name in ('escaped.safetensors', 'escaped.json', 'escaped.jsonl'):
            self.assertNotIn(f'gemm-probe-20261004/{name}', names)  # JSON-escaped paths decode to forbidden strings
        for name in ('root.safetensors', 'varroot.safetensors', 'thishome.safetensors'):
            self.assertNotIn(f'gemm-probe-20261004/{name}', names)  # home paths outside /Users and /home
        self.assertIn('undecodable text', manifest['excluded_counts'])
        for reason in ('binary', 'private-named path', 'forbidden content', 'private workload directory',
                       'upstream clone', 'trace or dSYM bundle', 'tensor size', 'oracle directory'):
            self.assertIn(reason, ex, reason)

    def test_refuses_to_overwrite(self):
        out = Path(self.tmp.name) / 'ev.tar.gz'
        out.write_bytes(b'')
        with self.assertRaises(SystemExit):
            ea.build(self.root, out, 'ev')

    def test_creation_is_exclusive(self):
        # A file that appears between the existence check and the open must not be truncated:
        # the check is simulated as having passed, and the open itself must refuse.
        from unittest.mock import patch
        out = Path(self.tmp.name) / 'race.tar.gz'
        out.write_bytes(b'another build')
        with patch.object(Path, 'exists', return_value=False), self.assertRaises(SystemExit):
            ea.build(self.root, out, 'ev')
        self.assertEqual(out.read_bytes(), b'another build')

    def test_manifest_describes_the_bytes_archived(self):
        # If a source changes after it was read, the manifest must still describe the bytes that
        # were archived, not the current file.
        target = self.root / 'gemm-probe-20261004' / 'run.py'
        original = target.read_bytes()
        real = ea.prepare

        def prepare_then_mutate(path, *args, **kwargs):
            result = real(path, *args, **kwargs)
            if path == target:
                target.write_bytes(original + b'# changed after the read\n')
            return result

        ea.prepare = prepare_then_mutate
        try:
            manifest = ea.build(self.root, Path(self.tmp.name) / 'prov.tar.gz', 'ev')
        finally:
            ea.prepare = real
        entry = next(e for e in manifest['files'] if e['path'] == 'gemm-probe-20261004/run.py')
        self.assertEqual(entry['source_sha256'], hashlib.sha256(original).hexdigest())
        self.assertEqual(entry['source_bytes'], len(original))

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
        ea.prepare = lambda p, raw=None: (p.read_bytes(), False, 'included') if p.name == 'token.log' else orig(p, raw)
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
        ma = ea.build(self.root, a, 'ev')
        ea.build(self.root, b, 'ev')
        # Byte-identical archives for the same tree: the manifest timestamp must come from the
        # tree, not the wall clock.
        self.assertEqual(hashlib.sha256(a.read_bytes()).hexdigest(), hashlib.sha256(b.read_bytes()).hexdigest())
        self.assertEqual(ma['built'], datetime.datetime.fromtimestamp(FIXED_MTIME, datetime.timezone.utc).isoformat())

    def test_allowlisted_files_obey_size_and_text_checks(self):
        (self.root / 'engine_logits_clef.jsonl').write_text('{}\n' * 2048)   # top-level, allowlisted, oversized
        (self.root / 'clef-flash-f32' / 'encoded.jsonl').write_bytes(b'\0' * 64)   # oracle-shaped name, binary content
        big = ea.MAX_TEXT
        ea.MAX_TEXT = 1024
        try:
            manifest = ea.build(self.root, Path(self.tmp.name) / 'size.tar.gz', 'ev')
        finally:
            ea.MAX_TEXT = big
        names = {e['path'] for e in manifest['files']}
        self.assertNotIn('engine_logits_clef.jsonl', names)
        self.assertNotIn('clef-flash-f32/encoded.jsonl', names)
        self.assertIn('engine_logits.jsonl', names)
        self.assertIn('oversized text', manifest['excluded_counts'])

    def test_default_label_from_tree(self):
        a = Path(self.tmp.name) / 'd1.tar.gz'
        b = Path(self.tmp.name) / 'd2.tar.gz'
        ma = ea.build(self.root, a, None)
        ea.build(self.root, b, None)
        day = datetime.datetime.fromtimestamp(FIXED_MTIME, datetime.timezone.utc).date().isoformat()
        self.assertEqual(ma['label'], f'cleffa-evidence-{day}')
        self.assertEqual(hashlib.sha256(a.read_bytes()).hexdigest(), hashlib.sha256(b.read_bytes()).hexdigest())
        with tarfile.open(a) as tar:
            self.assertTrue(all(m.name.startswith(f'cleffa-evidence-{day}/') for m in tar.getmembers()))

    def test_rejects_unsafe_label(self):
        # The label becomes every tar member's leading path component.
        for label in ('../outside', 'x/y', '.hidden', '', 'a b'):
            with self.subTest(label=label), self.assertRaises(SystemExit):
                ea.build(self.root, Path(self.tmp.name) / f'l{abs(hash(label))}.tar.gz', label)
        self.assertFalse(list(Path(self.tmp.name).glob('l*.tar.gz')))


if __name__ == '__main__':
    unittest.main()
