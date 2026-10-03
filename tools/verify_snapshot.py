"""Check a downloaded Hugging Face snapshot against the revision it is pinned to.

`hf download --local-dir DIR` records, per file, the commit it came from and the hash Hugging Face
published for it (DIR/.cache/huggingface/download/<path>.metadata: commit, etag, timestamp). LFS
files carry their SHA-256; small files carry their git blob SHA-1. This checks that every file came
from the pinned commit and still hashes to what was published. It matters beyond corrupt
downloads: the reference oracles import `joint_schema_model.py` from the snapshot, so it is code
that runs.

Usage: verify_snapshot.py DIR REVISION
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path


def file_hash(path: Path, etag: str) -> str:
    if len(etag) == 64:                                   # LFS: SHA-256 of the content
        h = hashlib.sha256()
    elif len(etag) == 40:                                 # git blob: SHA-1 of "blob <size>\0" + content
        h = hashlib.sha1()
        h.update(b"blob %d\0" % path.stat().st_size)
    else:
        raise ValueError(f"unrecognised etag {etag!r}")
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    root, rev = Path(sys.argv[1]), sys.argv[2]
    meta_root = root / ".cache" / "huggingface" / "download"
    metas = sorted(meta_root.rglob("*.metadata"))
    if not metas:
        sys.exit(f"{root}: no download metadata (not fetched with `hf download --local-dir`?)")
    bad = []
    for meta in metas:
        rel = meta.relative_to(meta_root).with_suffix("")   # strip ".metadata"
        lines = meta.read_text().splitlines()
        if len(lines) < 2:
            bad.append(f"{rel}: malformed metadata"); continue
        commit, etag = lines[0].strip(), lines[1].strip()
        path = root / rel
        if commit != rev:
            bad.append(f"{rel}: from commit {commit}, expected {rev}")
        elif not path.is_file():
            bad.append(f"{rel}: missing")
        elif file_hash(path, etag) != etag:
            bad.append(f"{rel}: hash mismatch")
    for b in bad:
        print(f"  {b}", file=sys.stderr)
    if bad:
        sys.exit(f"{root}: {len(bad)} of {len(metas)} files failed verification against {rev}")
    print(f"{root}: {len(metas)} files verified against {rev}")


if __name__ == "__main__":
    main()
