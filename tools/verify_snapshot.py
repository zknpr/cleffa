"""Check a downloaded Hugging Face snapshot against the revision it is pinned to.

The checked-in snapshots/<revision>.json manifests were fetched from Hugging Face's
HTTPS API at the exact pinned revisions. They supply the complete inventory, sizes,
LFS SHA-256 hashes and git blob SHA-1 hashes. Local download metadata is not trusted.
Unexpected files, including Python bytecode caches, are refused because loaders may
prefer them to the verified source or sharded weights. Verification is offline.

Usage: verify_snapshot.py DIR REVISION
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import sys
from pathlib import Path

MANIFESTS = Path(__file__).resolve().parent / "snapshots"


def load_manifest(revision: str) -> dict:
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("expected a full pinned revision")
    path = MANIFESTS / f"{revision}.json"
    if not path.is_file():
        raise ValueError(f"unsupported revision {revision}: no trusted manifest")
    manifest = json.loads(path.read_text())
    if manifest["revision"] != revision or not manifest["files"]:
        raise ValueError(f"invalid trusted manifest {path}")
    return manifest


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


def verify_snapshot(root: Path, manifest: dict) -> list[str]:
    if not root.is_dir():
        raise ValueError(f"{root}: snapshot directory missing")
    expected = manifest["files"]
    found = set()
    bad = []
    def walk_error(error: OSError) -> None:
        raise error

    for directory, dirs, files in os.walk(root, followlinks=False, onerror=walk_error):
        parent = Path(directory)
        for name in dirs[:]:
            path = parent / name
            rel = path.relative_to(root).as_posix()
            if rel == ".cache/huggingface":
                dirs.remove(name)  # download bookkeeping is not a model input
            elif path.is_symlink():
                bad.append(f"{rel}: symbolic link is not allowed")
                dirs.remove(name)
        for name in files:
            path = parent / name
            rel = path.relative_to(root).as_posix()
            found.add(rel)
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode):
                bad.append(f"{rel}: expected a regular file (no symbolic links)")
            elif rel not in expected:
                advice = "; remove stale bytecode and run reference scripts with python -B" if path.suffix == ".pyc" else ""
                bad.append(f"{rel}: unexpected file{advice}")
            elif info.st_size != expected[rel]["size"]:
                bad.append(f"{rel}: size mismatch")
    bad.extend(f"{rel}: missing" for rel in sorted(expected.keys() - found))
    if bad:
        return bad  # reject incomplete inventories before hashing multi-GB weights
    for rel, entry in expected.items():
        if file_hash(root / rel, entry["hash"]) != entry["hash"]:
            bad.append(f"{rel}: hash mismatch")
    return bad


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    root, rev = Path(sys.argv[1]), sys.argv[2]
    try:
        manifest = load_manifest(rev)
        bad = verify_snapshot(root, manifest)
    except (OSError, ValueError) as error:
        sys.exit(f"{root}: {error}")
    for b in bad:
        print(f"  {b}", file=sys.stderr)
    if bad:
        sys.exit(f"{root}: verification failed against {rev}")
    print(f"{root}: {len(manifest['files'])} files verified against {manifest['repo']}@{rev}")


if __name__ == "__main__":
    main()
