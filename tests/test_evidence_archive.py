"""The evidence archive must drop binaries, traces, tensors and private files, rewrite local
paths, and fail closed when a forbidden pattern survives."""
import datetime
import hashlib
import json
import os
import shutil
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
    (exp / 'badmeta.safetensors').write_bytes(safetensors({'__metadata__': 7, 'l': {'dtype': 'F32', 'shape': [4], 'data_offsets': [0, 16]}}))
    (exp / 'badmeta2.safetensors').write_bytes(safetensors({'__metadata__': {'format': 1}, 'l': {'dtype': 'F32', 'shape': [4], 'data_offsets': [0, 16]}}))
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
    doc = b'{"auth": "Bearer abcdefgh\\u002dijklmnopqrstuvwxyz0123"}'
    doc += b' ' * (-len(doc) % 4)   # trailing spaces keep it valid JSON and a whole number of floats
    hdrj = json.dumps({'l': {'dtype': 'F32', 'shape': [len(doc) // 4], 'data_offsets': [0, len(doc)]}}).encode()
    (exp / 'jsonpayload.safetensors').write_bytes(len(hdrj).to_bytes(8, 'little') + hdrj + doc)
    lead = b'\n' + json.dumps({'l': {'dtype': 'F32', 'shape': [4], 'data_offsets': [0, 16]}}).encode()
    (exp / 'leadingws.safetensors').write_bytes(len(lead).to_bytes(8, 'little') + lead + b'\0' * 16)   # header not starting with {
    (exp / 'macho.log').write_bytes(b'\xcf\xfa\xed\xfe' + b'1' * 12)   # Mach-O magic, no NUL, text name
    (exp / 'escaped.log').write_text('Authorization: Bearer abcdefgh\\u002dijklmnopqrstuvwxyz0123\n')   # escape in plain text
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
        self.assertNotIn('gemm-probe-20261004/badmeta.safetensors', names)  # __metadata__ not a map
        self.assertNotIn('gemm-probe-20261004/badmeta2.safetensors', names)  # __metadata__ values not strings
        self.assertNotIn('gemm-probe-20261004/payload.safetensors', names)  # a U8 tensor carrying text
        self.assertNotIn('gemm-probe-20261004/utf16.safetensors', names)  # text as UTF-16 inside an F32 payload
        self.assertNotIn('gemm-probe-20261004/jsonpayload.safetensors', names)  # JSON-escaped token in a payload
        self.assertNotIn('gemm-probe-20261004/escaped.log', names)  # \\u002d escape in plain text
        self.assertNotIn('gemm-probe-20261004/macho.log', names)  # Mach-O magic under a text name
        self.assertNotIn('gemm-probe-20261004/leadingws.safetensors', names)  # header must start with {
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
        for reason in ('binary', 'private-named path', 'forbidden content', 'tensor size',
                       # subtrees excluded by their path are pruned whole, so they count as directories
                       'private workload directory (directories)', 'upstream clone (directories)',
                       'trace or dSYM bundle (directories)', 'oracle directory (directories)',
                       'unrecognized directory (directories)', 'private-named path (directories)'):
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

    def test_source_changing_during_the_read_aborts(self):
        # The archive and its manifest must describe one consistent snapshot: a file whose
        # metadata differs between the stat before and the stat after the read stops the build.
        import os
        from unittest.mock import patch
        target = self.root / 'gemm-probe-20261004' / 'result.json'
        real_fstat = os.fstat
        calls = {'n': 0}

        def fstat_mutating_between(fd):
            st = real_fstat(fd)
            if st.st_ino == target.stat().st_ino:
                calls['n'] += 1
                if calls['n'] == 1:   # after the first stat, before the second: the file changes
                    target.write_bytes(target.read_bytes() + b'\n')
                    os.utime(target, (FIXED_MTIME + 5, FIXED_MTIME + 5))
            return st

        with patch.object(ea.os, 'fstat', fstat_mutating_between), self.assertRaises(SystemExit):
            ea.build(self.root, Path(self.tmp.name) / 'changing.tar.gz', 'ev')

    def test_source_changing_after_its_read_aborts(self):
        # A source that changes after it was read used to be archived as read, with the
        # manifest describing those bytes. Now the build refuses: the archived bytes would come
        # from one revision of the tree and files read later from another.
        target = self.root / 'gemm-probe-20261004' / 'run.py'
        original = target.read_bytes()
        real = ea.prepare

        def prepare_then_mutate(path, *args, **kwargs):
            result = real(path, *args, **kwargs)
            if path == target:
                target.write_bytes(original + b'# changed after the read\n')
            return result

        ea.prepare = prepare_then_mutate
        out = Path(self.tmp.name) / 'prov.tar.gz'
        try:
            with self.assertRaisesRegex(SystemExit, 'changed after it was examined'):
                ea.build(self.root, out, 'ev')
        finally:
            ea.prepare = real
        self.assertFalse(out.exists())

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

    def test_rejects_forbidden_label(self):
        # The label lands in every member name and in the generated README and manifest.
        for label in ('Zknpr', 'squid-run', 'pop_v22', 'logs.personal'):
            with self.subTest(label=label), self.assertRaises(SystemExit):
                ea.build(self.root, Path(self.tmp.name) / f'f{abs(hash(label))}.tar.gz', label)

    def test_size_limits_apply_to_the_bytes_read(self):
        # A file that passes the stat-time size check but is larger by the time it is read must
        # still be excluded, and never read whole.
        from unittest.mock import patch
        big = ea.MAX_TEXT
        ea.MAX_TEXT = 1024
        real = ea.classify
        try:
            def permissive(golden, path):
                ok, reason = real(golden, path)
                return (True, 'included') if path.name == 'grown.log' else (ok, reason)
            (self.root / 'gemm-probe-20261004' / 'grown.log').write_text('x' * 4096 + '\n')
            with patch.object(ea, 'classify', permissive):
                manifest = ea.build(self.root, Path(self.tmp.name) / 'grown.tar.gz', 'ev')
        finally:
            ea.MAX_TEXT = big
        self.assertNotIn('gemm-probe-20261004/grown.log', {e['path'] for e in manifest['files']})
        self.assertIn('oversized text', manifest['excluded_counts'])

    def test_symlink_swapped_in_after_classification_is_refused(self):
        # classify() rejects a symlink by name, but the name can be replaced by a link between
        # that check and the open. The read must not follow it: the target here is a raw,
        # unlabelled token that no FORBIDDEN pattern would catch.
        from unittest.mock import patch
        secret = Path(self.tmp.name) / 'jev.api'
        secret.write_text('k9f3q8z1x7v2b6n4m0c5l8p3w1e9r7t2\n')
        target = self.root / 'gemm-probe-20261004' / 'swap.log'
        target.write_text('benign\n')
        real = ea.classify

        def swap_after(golden, path):
            verdict = real(golden, path)
            if path == target:
                path.unlink()
                path.symlink_to(secret)
            return verdict
        with patch.object(ea, 'classify', swap_after), self.assertRaises(SystemExit):
            ea.build(self.root, Path(self.tmp.name) / 'swap.tar.gz', 'ev')
        root_fd = ea.open_tree(self.root)
        try:
            with self.assertRaises(SystemExit):
                ea.read_snapshot(root_fd, Path('gemm-probe-20261004/swap.log'), ea.MAX_TEXT)   # refused on its own
        finally:
            os.close(root_fd)

    def test_unreadable_directory_fails_the_build(self):
        # os.walk() skips a directory it cannot list unless told otherwise, and an archive that
        # is complete by appearance only is worse than none. A directory excluded by its path
        # is pruned before it is listed, so it may be unreadable.
        if os.geteuid() == 0:
            self.skipTest('root can list every directory')
        locked = self.root / 'gemm-probe-20261004' / 'locked'
        locked.mkdir()
        (locked / 'result.json').write_text('{}')
        pruned = self.root / 'gemm-probe-20261004' / 'private-locked'
        pruned.mkdir()
        (pruned / 'x.json').write_text('{}')
        try:
            pruned.chmod(0)
            manifest = ea.build(self.root, Path(self.tmp.name) / 'pruned.tar.gz', 'ev')
            self.assertNotIn('gemm-probe-20261004/private-locked/x.json', {e['path'] for e in manifest['files']})
            self.assertIn('gemm-probe-20261004/locked/result.json', {e['path'] for e in manifest['files']})
            locked.chmod(0)
            with self.assertRaisesRegex(SystemExit, 'locked'):
                ea.build(self.root, Path(self.tmp.name) / 'locked.tar.gz', 'ev')
        finally:
            locked.chmod(0o700)
            pruned.chmod(0o700)

    def test_aggregate_size_is_bounded(self):
        # Every included payload is held until the tar is written, and the per-file limit alone
        # does not bound that; the build fails explicitly instead of exhausting memory.
        from unittest.mock import patch
        with patch.object(ea, 'MAX_TOTAL', 64), self.assertRaisesRegex(SystemExit, 'MAX_TOTAL'):
            ea.build(self.root, Path(self.tmp.name) / 'total.tar.gz', 'ev')

    def test_classification_reads_no_content(self):
        # A size check at stat time cannot bound a read made afterwards, so classify() decides by
        # name and metadata alone; the bounded snapshot is the only read of a file's content.
        from unittest.mock import patch

        def refuse(*args, **kwargs):
            raise AssertionError('file content read during classification')
        for name in ('result.json', 'run.py', 'small.safetensors'):
            with patch.object(ea.Path, 'read_bytes', refuse), patch('builtins.open', refuse):
                self.assertTrue(ea.classify(self.root, self.root / 'gemm-probe-20261004' / name)[0], name)

    def test_credential_values_with_punctuation_are_caught(self):
        # The value class once stopped at punctuation, so a password with symbols or a base64
        # bearer token never reached sixteen matching characters.
        exp = self.root / 'gemm-probe-20261004'
        cases = {'punct.log': 'PASSWORD=Abc!Def@Ghi#Jkl$Mno%\n',
                 'bearer64.log': 'Authorization: Bearer abcd.efgh/ijkl+mnop=qrst\n',
                 'punct.json': '{"api_key": "Abc!Def@Ghi#Jkl$Mno%"}',
                 'spaced.json': '{"password": "correct horse battery staple!"}'}
        for name, text in cases.items():
            (exp / name).write_text(text)
        manifest = ea.build(self.root, Path(self.tmp.name) / 'cred.tar.gz', 'ev')
        names = {e['path'] for e in manifest['files']}
        for name in cases:
            self.assertNotIn(f'gemm-probe-20261004/{name}', names, name)

    def test_hard_linked_files_are_excluded(self):
        # A hard link is a regular file whose descriptor and name agree, so the symlink checks
        # pass. Linking needs no read access to the target, so the build would publish what
        # the linker could not read.
        from unittest.mock import patch
        secret = Path(self.tmp.name) / 'jev.api'
        secret.write_text('k9f3q8z1x7v2b6n4m0c5l8p3w1e9r7t2\n')
        exp = self.root / 'gemm-probe-20261004'
        os.link(secret, exp / 'linked.log')
        manifest = ea.build(self.root, Path(self.tmp.name) / 'linked.tar.gz', 'ev')
        self.assertNotIn('gemm-probe-20261004/linked.log', {e['path'] for e in manifest['files']})
        self.assertIn('hard link', manifest['excluded_counts'])
        target = exp / 'late-link.log'   # linked after classification: the descriptor's link count catches it
        target.write_text('benign\n')
        real = ea.classify

        def link_after(golden, path):
            verdict = real(golden, path)
            if path == target:
                path.unlink()
                os.link(secret, path)
            return verdict
        with patch.object(ea, 'classify', link_after), self.assertRaises(SystemExit):
            ea.build(self.root, Path(self.tmp.name) / 'late-link.tar.gz', 'ev')

    def test_fifo_does_not_block_the_build(self):
        # A FIFO under a text name has zero size and is not a link; a blocking open would wait
        # for a writer forever.
        import signal
        fifo = self.root / 'gemm-probe-20261004' / 'pipe.log'
        os.mkfifo(fifo)

        def expired(signum, frame):
            raise TimeoutError('the build blocked on the FIFO')
        previous = signal.signal(signal.SIGALRM, expired)
        signal.alarm(10)
        try:
            manifest = ea.build(self.root, Path(self.tmp.name) / 'fifo.tar.gz', 'ev')
            root_fd = ea.open_tree(self.root)
            try:
                with self.assertRaises(SystemExit):
                    ea.read_snapshot(root_fd, Path('gemm-probe-20261004/pipe.log'), ea.MAX_TEXT)   # refused without blocking
            finally:
                os.close(root_fd)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)
        self.assertNotIn('gemm-probe-20261004/pipe.log', {e['path'] for e in manifest['files']})
        self.assertIn('not a regular file', manifest['excluded_counts'])

    def test_ancestor_swapped_for_a_symlink_is_refused(self):
        # O_NOFOLLOW protects the final component only. A directory the walk already passed can
        # be replaced by a link to an external tree holding the same file name; opening every
        # component relative to its parent without following links refuses that.
        from unittest.mock import patch
        ext = Path(self.tmp.name) / 'ext'
        ext.mkdir()
        (ext / 'zz-swap.log').write_text('k9f3q8z1x7v2b6n4m0c5l8p3w1e9r7t2\n')   # unlabelled token
        probe = self.root / 'swap-probe-20261004'
        probe.mkdir()
        target = probe / 'zz-swap.log'
        target.write_text('benign\n')
        real = ea.classify

        def swap_ancestor(golden, path):
            verdict = real(golden, path)
            if path == target:
                shutil.rmtree(probe)
                probe.symlink_to(ext, target_is_directory=True)
            return verdict
        out = Path(self.tmp.name) / 'ancestor.tar.gz'
        with patch.object(ea, 'classify', swap_ancestor), self.assertRaises(SystemExit):
            ea.build(self.root, out, 'ev')
        self.assertFalse(out.exists())

    def test_quoted_credentials_with_spaces_are_caught(self):
        # A quoted value with spaces has no sixteen-character run, and plain key=value text
        # never reaches the JSON walk that normalizes whitespace.
        exp = self.root / 'gemm-probe-20261004'
        cases = {'quoted-space.log': 'PASSWORD="correct horse battery staple"\n',
                 'quoted-space2.log': "api_key: 'correct horse battery staple'\n"}
        for name, text in cases.items():
            (exp / name).write_text(text)
        manifest = ea.build(self.root, Path(self.tmp.name) / 'quoted.tar.gz', 'ev')
        names = {e['path'] for e in manifest['files']}
        for name in cases:
            self.assertNotIn(f'gemm-probe-20261004/{name}', names, name)

    def test_failed_write_leaves_no_archive(self):
        # A write that fails part-way must not leave a file at the output path: the next run
        # would refuse to overwrite it, and automation could take it for a finished archive.
        import errno
        from unittest.mock import patch
        out = Path(self.tmp.name) / 'enospc.tar.gz'

        def full(*args, **kwargs):
            raise OSError(errno.ENOSPC, 'No space left on device')
        with patch.object(ea.tarfile.TarFile, 'addfile', full), self.assertRaises(OSError):
            ea.build(self.root, out, 'ev')
        self.assertFalse(out.exists())
        self.assertEqual([p.name for p in out.parent.iterdir() if p.name.startswith('enospc')], [])

    def test_entries_changing_during_collection_abort(self):
        # A file created after its directory was enumerated is read by nobody, so the manifest
        # would describe neither the tree at the start nor at the end.
        from unittest.mock import patch
        exp = self.root / 'gemm-probe-20261004'
        real = ea.classify

        def add_late(golden, path):
            verdict = real(golden, path)
            if path == exp / 'result.json':
                (exp / 'late.json').write_text('{"ms": 2}')
            return verdict
        with patch.object(ea, 'classify', add_late), self.assertRaisesRegex(SystemExit, 'changed during collection'):
            ea.build(self.root, Path(self.tmp.name) / 'late.tar.gz', 'ev')

    def test_basic_auth_credentials_are_caught(self):
        # `Basic <base64>` carries user:password; the colon in the decoded value is what makes it
        # a credential, so the word "basic" before a word that happens to be valid base64 stays.
        exp = self.root / 'gemm-probe-20261004'
        (exp / 'basic.log').write_text('Authorization: Basic dXNlcjpwYXNzd29yZA==\n')
        (exp / 'basic-unpadded.log').write_text('Authorization: Basic dXNlcjpwYXNzd29yZA\n')   # padding stripped
        (exp / 'basic.json').write_text('{"headers": {"authorization": "basic dXNlcjpwYXNzd29yZA=="}}')
        (exp / 'prose.md').write_text('Basic test of the basic setup: a Basic auth note, basic abcd done.\n')
        manifest = ea.build(self.root, Path(self.tmp.name) / 'basic.tar.gz', 'ev')
        names = {e['path'] for e in manifest['files']}
        self.assertNotIn('gemm-probe-20261004/basic.log', names)
        self.assertNotIn('gemm-probe-20261004/basic-unpadded.log', names)
        self.assertNotIn('gemm-probe-20261004/basic.json', names)
        self.assertIn('gemm-probe-20261004/prose.md', names)

    def test_file_overwritten_after_its_read_aborts(self):
        # An in-place overwrite of an already-collected file leaves the directory's entries and
        # inode unchanged; the archived bytes would then come from one revision and files read
        # later from another.
        from unittest.mock import patch
        exp = self.root / 'gemm-probe-20261004'
        real = ea.classify

        def overwrite_earlier(golden, path):
            verdict = real(golden, path)
            if path == exp / 'run.py':   # sorted after result.json, which has been read by now
                (exp / 'result.json').write_text('{"ms": 9.5}')
            return verdict
        with patch.object(ea, 'classify', overwrite_earlier), self.assertRaisesRegex(SystemExit, 'changed after it was examined'):
            ea.build(self.root, Path(self.tmp.name) / 'overwrite.tar.gz', 'ev')

    def test_member_names_are_scanned_like_content(self):
        # A file name becomes a tar member name; the Basic check lives beside FORBIDDEN in
        # forbidden_in(), so a name is scanned with the same function as content.
        exp = self.root / 'gemm-probe-20261004'
        (exp / 'Authorization Basic dXNlcjpwYXNzd29yZA==.txt').write_text('benign\n')
        manifest = ea.build(self.root, Path(self.tmp.name) / 'name.tar.gz', 'ev')
        self.assertFalse(any('Basic' in e['path'] for e in manifest['files']))
        self.assertIn('forbidden path', manifest['excluded_counts'])

    def test_excluded_file_changing_after_classification_aborts(self):
        # A file excluded on mutable metadata (its size) and replaced below the limit afterwards
        # leaves the directory's entries unchanged; every enumerated file, included or not, must
        # still be what it was when it was examined.
        from unittest.mock import patch
        exp = self.root / 'gemm-probe-20261004'
        (exp / 'big.log').write_text('x' * 4096 + '\n')
        real = ea.classify

        def truncate_earlier(golden, path):
            verdict = real(golden, path)
            if path == exp / 'run.py':   # sorted after big.log, which was excluded as oversized by now
                (exp / 'big.log').write_text('small now\n')
            return verdict
        with patch.object(ea, 'MAX_TEXT', 1024), patch.object(ea, 'classify', truncate_earlier), \
                self.assertRaisesRegex(SystemExit, 'changed after it was examined'):
            ea.build(self.root, Path(self.tmp.name) / 'truncate.tar.gz', 'ev')

    def test_credential_key_spellings_are_caught(self):
        # CREDENTIAL_KEY recognizes api-key and apiKey, but the rejoined pair and plain text
        # were matched by an assignment pattern that knew API_KEY only.
        exp = self.root / 'gemm-probe-20261004'
        cases = {'hyphen.json': '{"api-key": "abcdefghijklmnop"}',
                 'camel.json': '{"apiKey": "abcdefghijklmnop"}',
                 'hyphen.log': 'api-key: abcdefghijklmnop\n',
                 'camel.log': 'apiKey=abcdefghijklmnop\n'}
        for name, text in cases.items():
            (exp / name).write_text(text)
        manifest = ea.build(self.root, Path(self.tmp.name) / 'keys.tar.gz', 'ev')
        names = {e['path'] for e in manifest['files']}
        for name in cases:
            self.assertNotIn(f'gemm-probe-20261004/{name}', names, name)

    def test_pem_private_keys_are_caught(self):
        exp = self.root / 'gemm-probe-20261004'
        cases = {'ssh.log': '-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXktdjEAAAAA\n-----END OPENSSH PRIVATE KEY-----\n',
                 'pkcs8.txt': '-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASC\n-----END PRIVATE KEY-----\n',
                 'rsa.md': 'dumped: -----BEGIN RSA PRIVATE KEY-----\n',
                 'pgp.txt': '-----BEGIN PGP PRIVATE KEY BLOCK-----\n'}
        for name, text in cases.items():
            (exp / name).write_text(text)
        (exp / 'cert.txt').write_text('-----BEGIN CERTIFICATE-----\nMIIB\n-----END CERTIFICATE-----\n')   # public: kept
        manifest = ea.build(self.root, Path(self.tmp.name) / 'pem.tar.gz', 'ev')
        names = {e['path'] for e in manifest['files']}
        for name in cases:
            self.assertNotIn(f'gemm-probe-20261004/{name}', names, name)
        self.assertIn('gemm-probe-20261004/cert.txt', names)

    def test_child_directory_swapped_before_descent_aborts(self):
        # os.walk() skips a child that became a symlink between the parent's enumeration and
        # the descent, silently and without onerror; the parent's entry names are unchanged.
        # Every retained child directory must be visited as the directory it was enumerated as.
        from unittest.mock import patch
        probe = self.root / 'gemm-probe-20261004'
        moved = Path(self.tmp.name) / 'moved-probe'
        real = ea.classify

        def swap_child(golden, path):
            verdict = real(golden, path)
            if path == self.root / 'engine_logits.jsonl':   # a root file: classified before the descent
                shutil.move(probe, moved)
                probe.symlink_to(moved, target_is_directory=True)
            return verdict
        out = Path(self.tmp.name) / 'child.tar.gz'
        with patch.object(ea, 'classify', swap_child), self.assertRaisesRegex(SystemExit, 'changed during collection'):
            ea.build(self.root, out, 'ev')
        self.assertFalse(out.exists())

    def test_compound_credential_names_are_caught(self):
        # The sensitive word may sit inside a compound name (AWS_SECRET_ACCESS_KEY,
        # secretAccessKey); a word that merely begins with it (tokenizer) is not a credential.
        exp = self.root / 'gemm-probe-20261004'
        secret = 'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY'
        caught = {'aws.log': f'AWS_SECRET_ACCESS_KEY={secret}\n',
                  'aws.json': f'{{"aws_secret_access_key": "{secret}"}}',
                  'camel.log': f'secretAccessKey={secret}\n',
                  'camel.json': f'{{"awsSecretAccessKey": "{secret}"}}',
                  'hash.log': 'password_hash: $2b$12$abcdefghijklmnopqrstuv\n'}
        kept = {'tok.log': 'tokenizer=<repo>/model-flash-9b-snapshot\nTOKENIZER_PATH=/opt/models/clef-flash-9b\n',
                'tok.json': '{"tokenizer_class": "Qwen2TokenizerFast", "max_tokens": 4096, "pad_token": "<|endoftext|>"}'}
        for name, text in {**caught, **kept}.items():
            (exp / name).write_text(text)
        manifest = ea.build(self.root, Path(self.tmp.name) / 'compound.tar.gz', 'ev')
        names = {e['path'] for e in manifest['files']}
        for name in caught:
            self.assertNotIn(f'gemm-probe-20261004/{name}', names, name)
        for name in kept:
            self.assertIn(f'gemm-probe-20261004/{name}', names, name)

    def test_key_material_assignments_are_caught(self):
        # Key material without PEM delimiters is named for what it is; public keys are not secrets.
        exp = self.root / 'gemm-probe-20261004'
        material = 'VGhpcyBpcyBhIHByaXZhdGUga2V5IG1hdGVyaWFsCg=='
        caught = {'pk.log': f'PRIVATE_KEY={material}\n',
                  'pk.json': f'{{"private_key": "{material}"}}',
                  'pk-camel.json': f'{{"privateKey": "{material}"}}',
                  'sign.log': f'SIGNING_KEY: {material}\n',
                  'enc.json': f'{{"encryption_key": "{material}"}}',
                  'aws-id.log': 'AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE\n'}
        kept = {'pub.json': '{"public_key": "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIEXAMPLEEXAMPLEEXAMPLE"}',
                'keys.log': 'key_count=16\nprimary_key_column=request_identifier_value\n'}
        for name, text in {**caught, **kept}.items():
            (exp / name).write_text(text)
        manifest = ea.build(self.root, Path(self.tmp.name) / 'material.tar.gz', 'ev')
        names = {e['path'] for e in manifest['files']}
        for name in caught:
            self.assertNotIn(f'gemm-probe-20261004/{name}', names, name)
        for name in kept:
            self.assertIn(f'gemm-probe-20261004/{name}', names, name)

    def test_rejects_unsafe_label(self):
        # The label becomes every tar member's leading path component.
        for label in ('../outside', 'x/y', '.hidden', '', 'a b'):
            with self.subTest(label=label), self.assertRaises(SystemExit):
                ea.build(self.root, Path(self.tmp.name) / f'l{abs(hash(label))}.tar.gz', label)
        self.assertFalse(list(Path(self.tmp.name).glob('l*.tar.gz')))


if __name__ == '__main__':
    unittest.main()
