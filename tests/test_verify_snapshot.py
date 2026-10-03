"""Offline snapshot verification against inert, independently specified fixtures."""

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from verify_snapshot import load_manifest, verify_snapshot


class VerifySnapshot(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="clef-snapshot-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        content = {"config.json": b'{}\n', "joint_schema_model.py": b'# inert fixture\n',
                   "weights.safetensors": b'fixture bytes'}
        self.manifest = {"files": {}}
        for name, data in content.items():
            (self.root / name).write_bytes(data)
            digest = (hashlib.sha256(data).hexdigest() if name.endswith("safetensors") else
                      hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest())
            self.manifest["files"][name] = {"size": len(data), "hash": digest}

    def verify(self):
        return verify_snapshot(self.root, self.manifest)

    def test_complete_inventory_without_download_metadata(self):
        self.assertEqual(self.verify(), [])

    def test_missing_file_is_rejected(self):
        (self.root / "joint_schema_model.py").unlink()
        self.assertIn("joint_schema_model.py: missing", self.verify())

    def test_download_metadata_cannot_replace_trusted_hash(self):
        path = self.root / "joint_schema_model.py"
        path.write_bytes(b'# other fixture\n')  # same size; the digest must decide
        meta = self.root / ".cache/huggingface/download/joint_schema_model.py.metadata"
        meta.parent.mkdir(parents=True)
        data = path.read_bytes()
        meta.write_text('a' * 40 + '\n' + hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest())
        self.assertIn("joint_schema_model.py: hash mismatch", self.verify())

    def test_size_mismatch_is_rejected(self):
        (self.root / "weights.safetensors").write_bytes(b'')
        self.assertIn("weights.safetensors: size mismatch", self.verify())

    def test_extra_loader_inputs_are_rejected(self):
        for name in ("model.safetensors", "special_tokens_map.json", "helper.py",
                     "__pycache__/joint_schema_model.cpython-312.pyc"):
            with self.subTest(name=name):
                path = self.root / name
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(b'inert')
                self.assertTrue(any(name + ": unexpected file" in error for error in self.verify()))
                path.unlink()

    def test_symlinks_are_rejected(self):
        path = self.root / "weights.safetensors"
        path.unlink()
        path.symlink_to(self.root / "config.json")
        self.assertTrue(any("symbolic link" in error for error in self.verify()))

    def test_download_bookkeeping_is_ignored(self):
        cache = self.root / ".cache/huggingface/download"
        cache.mkdir(parents=True)
        (cache / "config.json.metadata").write_text('untrusted bookkeeping')
        self.assertEqual(self.verify(), [])

    def test_only_pinned_revisions_have_manifests(self):
        for revision, count in (("17f0b0ad64efb65d273590632833508766b2aae6", 17),
                                ("2f3de3dd85f379784083b0814d997ab627200f0c", 25)):
            manifest = load_manifest(revision)
            self.assertEqual(len(manifest["files"]), count)
            self.assertIn("joint_schema_model.py", manifest["files"])
        for revision in ('main', '0' * 40, '../unknown'):
            with self.assertRaises(ValueError):
                load_manifest(revision)


if __name__ == "__main__":
    unittest.main()
