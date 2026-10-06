#!/usr/bin/env python3
"""Pack the evidence behind docs/ into a curated, publishable archive.

  .venv/bin/python -B tools/evidence_archive.py OUT.tar.gz [--golden golden] [--list]

`golden/` is git-ignored because it holds oracle tensors, Instruments traces, copied binaries
and an upstream clone (tens of GB). The reports in docs/ cite its `<experiment>-<date>/`
directories by name. This tool copies the parts of those directories that make the reports
checkable, under rules that are deliberately conservative:

Included
  - result, manifest, sample and log files (.json, .jsonl, .log, .txt, .md, .csv, .patch, ...)
  - experiment sources (.py, .m, .metal, .c, .h, .sh, Makefile)
  - .safetensors files up to SMALL_TENSOR bytes (saved logits; never layer dumps) whose JSON
    header passes the FORBIDDEN scan and whose data section holds only floating-point tensors
    that tile it exactly and do not read as forbidden text in any byte-, NUL-padded or UTF-16
    view; the suffix test is case-insensitive everywhere
  - for the oracle directories ref/oracle.py writes (clef, clef-flash and their documented
    variants, no date suffix): requests.jsonl, encoded.jsonl,
    logits.safetensors and latency.json only, so tests/test_parity.py can run without --dump
Excluded
  - Mach-O binaries, objects, dSYMs, .inc, Instruments .trace bundles and their exported
    counter tables (.xml, .npz), .bin/.npy tensors, archives, PDFs, .git clones, virtualenvs
  - the ds4 upstream `source/` tree, every `article-*` directory at any depth (private
    workload), any path that itself matches a FORBIDDEN pattern, and the
    text extracts of Apple's Metal Shading Language specification
  - every other undated directory, Cloudflare subscription and usage dumps (`subscriptions.json`,
    `usage-*.json`) and agents'
    `checkpoint*.json` working-state files
  - dotfiles and extensionless files other than Makefile and LICENSE, key=value assignments
    that look like credentials, hard-linked files and anything that is not a regular file
  - any file with "private" in any component of its path, and any text file that still matches a
    FORBIDDEN pattern after rewriting (private-workload paths, account identifiers,
    including a Cloudflare account ID inside a recorded `accounts/<id>/` API URL)
Rewritten (text files only, recorded per file in the manifest)
  - the local checkout path and home directory become <repo> and <home>
  - the Cloudflare account name becomes <cf-account>

The archive is deterministic for a given tree: sorted entries, source mtimes, a manifest
timestamp and default label taken from the newest source file, no owner names, gzip header
without a timestamp, and a label restricted to one safe path component. It refuses to overwrite an existing output. A final
scan over every archived text file fails the build if a FORBIDDEN pattern survives, so the
rewrite rules are checked rather than trusted. JSON text (documents, JSONL lines and
safetensors headers) is scanned both as written and as decoded strings, since an escaped
form such as \u002f survives a textual match and json.loads reassembles it. Text files may not contain a NUL byte
anywhere, which is what UTF-16 text would need to hide in them. Files are opened relative to
the evidence root one component at a time without following symlinks, and must still be the
singly linked regular file the path named once read; a directory
the walk cannot list fails the build unless its path excludes it, in which case it is pruned
unread; and the included payloads, held in memory until the archive is written, are bounded
by MAX_TOTAL in total.
"""
from __future__ import annotations

import argparse
import datetime
import gzip
import hashlib
import io
import json
import os
import re
import stat
import sys
import tarfile
from pathlib import Path

SMALL_TENSOR = 2 * 1024 * 1024
MAX_TEXT = 64 * 1024 * 1024
# Included payloads stay in memory until the archive is written; the per-file limits do not
# bound that sum, so the build fails explicitly past this total instead of exhausting memory
# (review #57). The 2026-10-06 tree is 63 MB uncompressed.
MAX_TOTAL = 1024 * 1024 * 1024

TEXT_EXT = {'.json', '.jsonl', '.log', '.txt', '.md', '.csv', '.patch', '.diff', '.yaml', '.yml',
            '.toml', '.py', '.m', '.metal', '.c', '.h', '.sh', '.mk', '.cfg'}
TEXT_NAMES = {'Makefile', 'LICENSE'}   # the only extensionless files admitted
ORACLE_FILES = {'requests.jsonl', 'encoded.jsonl', 'logits.safetensors', 'latency.json'}
EXPERIMENT = re.compile(r'^[a-z0-9-]+-20\d{6}$')
ORACLE_DIR = re.compile(r'^clef(-flash)?(-f32s?)?(-r02[01](r021)?)?(-unsafe(-attn)?|-safe)?$')
MACHO = {b'\xcf\xfa\xed\xfe', b'\xce\xfa\xed\xfe', b'\xca\xfe\xba\xbe', b'\xfe\xed\xfa\xcf', b'\xfe\xed\xfa\xce'}

EXCLUDE_DIR_PARTS = {'.git', 'mlx-env', '.venv', '__pycache__', 'node_modules'}
# Text extracts of Apple's Metal Shading Language specification kept beside some experiments.
THIRD_PARTY_DOC = re.compile(r'^(msl|metal-spec|Metal-Shading-Language-Specification)\.(txt|pdf)$')
# Cloudflare account subscription and usage dumps: budget bookkeeping, not evidence.
ACCOUNT_BOOKKEEPING = re.compile(r'^(subscriptions|usage([-_.].*)?)\.json$')
# Agents' own working-state files (goal, sessions, next action), not experiment evidence.
AGENT_STATE = re.compile(r'^checkpoint[-.a-zA-Z0-9]*\.json$')
EXCLUDE_EXT = {'.o', '.a', '.dylib', '.inc', '.bin', '.npy', '.npz', '.pt', '.xml', '.pdf',
               '.gz', '.zip', '.tar', '.zst', '.xz', '.bz2', '.7z', '.dmg', '.pkg'}

# Rewrites run in order; the checkout path must precede the home directory.
REWRITES = [
    (re.compile(re.escape(str(Path(__file__).resolve().parent.parent))), '<repo>'),
    (re.compile(re.escape(str(Path.home()))), '<home>'),
    (re.compile(r'Zknpr'), '<cf-account>'),
]
# Anything matching after rewriting excludes the file, and a match in the final scan fails
# the build. Case-insensitive: bearer schemes and variable names vary in case. Keep these broad: a false exclusion costs one evidence file, a miss publishes it.
# A credential value is any run of sixteen or more characters that are not whitespace or a quote:
# passwords carry punctuation and base64 tokens carry + / = (review #59).
FORBIDDEN = re.compile(r'/Users/[A-Za-z]|/home/[a-z]|/root/|/var/root/|' +
                       re.escape(str(Path.home())) + '|' + re.escape(str(Path(__file__).resolve().parent.parent)) + '|'
                       r'squid|\.personal|pop_v22|account_id["\']?\s*[=:]\s*["\']?[0-9a-f]{32}|'
                       r'Bearer\s+[^\s"\']{16,}|CLOUDFLARE_API_TOKEN=\S|Zknpr|session_id|'
                       r'accounts/[0-9a-f]{32}|CLOUDFLARE_ACCOUNT_ID["\']?\s*[=:]\s*["\']?[0-9a-f]{32}|'
                       r'\b[A-Z0-9_]*(API_KEY|SECRET|TOKEN|PASSWORD)["\']?\s*[=:]\s*["\']?[^\s"\']{16,}', re.IGNORECASE)
# A credential stored as a JSON field: a key named like one, with a string value long enough to be one.
CREDENTIAL_KEY = re.compile(r'(api[_-]?key|secret|token|password)$', re.IGNORECASE)


def is_text(path: Path) -> bool:
    """Text by name: a listed extension, or one of the two extensionless names (review #42).
    Content is judged by prepare() on the bounded snapshot, never here: a read made after the
    stat-time size check would be unbounded (review #58)."""
    return path.suffix.lower() in TEXT_EXT or path.name in TEXT_NAMES


def path_reason(golden: Path, path: Path, directory: bool) -> str | None:
    """The exclusion reason the relative path alone decides, for a file or for a directory. The
    walk prunes a directory with a reason before listing it, so an excluded subtree is never
    read and need not be readable (review #56); every file below it would get the same reason."""
    rel = path.relative_to(golden)
    parts = rel.parts
    top = parts[0]
    ancestors = parts if directory else parts[:-1]   # a directory is itself a bundle or workload
    if path.is_symlink():
        return 'symlink'
    # Before any allowlist: "private" anywhere in the relative path excludes the file, and so
    # does a forbidden pattern in the path itself, which becomes a tar member name.
    if any('private' in part.lower() for part in parts):
        return 'private-named path'
    if any(part.startswith('.') for part in parts):
        return 'dotfile'   # .env, .gitignore, editor state: never evidence
    if FORBIDDEN.search(rel.as_posix()):
        return 'forbidden path'
    if any(p in EXCLUDE_DIR_PARTS for p in parts):
        return 'clone or environment'
    if any(p.endswith('.trace') or p.endswith('.dSYM') for p in ancestors):
        return 'trace or dSYM bundle'
    if any(part.startswith('article-') for part in ancestors):
        return 'private workload directory'
    if top.startswith('ds4-') and len(ancestors) >= 2 and parts[1] == 'source':
        return 'upstream clone'
    if ancestors and not EXPERIMENT.match(top):
        # Only the oracle directories ref/oracle.py writes, and of those the small files, never
        # layers/ or dumps (size-checked in classify). Any other undated directory is not evidence.
        if not ORACLE_DIR.match(top):
            return 'unrecognized directory'
        if directory:
            if len(parts) >= 2:   # layers/ and any other subdirectory; the oracle dir itself is walked
                return 'oracle directory'
        elif not (len(parts) == 2 and path.name in ORACLE_FILES):
            return 'oracle directory'
    return None


def classify(golden: Path, path: Path) -> tuple[bool, str]:
    """(include?, reason). Reasons are stable strings used for the --list summary."""
    parts = path.relative_to(golden).parts
    reason = path_reason(golden, path, directory=False)
    if reason is not None:
        return False, reason
    if len(parts) == 1 and not (path.suffix == '.jsonl' and path.name.startswith('engine_logits')):
        # Top-level files: only the engine outputs the parity tests write (size-checked below).
        return False, 'top-level file'
    if path.name.startswith('cleffa-evidence-'):
        return False, 'archive output'
    if THIRD_PARTY_DOC.match(path.name):
        return False, 'third-party document'
    if ACCOUNT_BOOKKEEPING.match(path.name):
        return False, 'account bookkeeping'
    if AGENT_STATE.match(path.name):
        return False, 'agent checkpoint'
    st = path.lstat()
    if not stat.S_ISREG(st.st_mode):
        return False, 'not a regular file'   # a FIFO would block the open; a device is not evidence (review #61)
    if st.st_nlink != 1:
        # A hard link is a regular file whose descriptor and name agree; it can be made without
        # read access to its target, so publishing it would publish what the linker could not
        # read (review #60).
        return False, 'hard link'
    ext = path.suffix.lower()
    if ext in EXCLUDE_EXT:
        return False, f'excluded extension {ext}'
    if ext == '.safetensors':
        return st.st_size <= SMALL_TENSOR, 'tensor size'
    if not is_text(path):
        return False, 'binary'
    if st.st_size > MAX_TEXT:
        return False, 'oversized text'
    return True, 'included'


def json_strings(text: str) -> str | None:
    """Every string (keys and values) of a JSON document or JSONL file, joined, or None when
    the text is not JSON. A scan over the serialized text misses escaped forms such as
    \\u002f, which json.loads reassembles, so forbidden patterns are checked on the decoded
    strings as well."""
    found: list[str] = []

    def walk(v):
        if isinstance(v, str):
            found.append(v)
        elif isinstance(v, dict):
            for k, x in v.items():
                if isinstance(k, str) and isinstance(x, str) and CREDENTIAL_KEY.search(k) and len(x) >= 16:
                    # Rejoined so the credential pattern sees key and value together; whitespace
                    # inside the value becomes '_' so a passphrase counts as one value. This is
                    # a scanning view, never archived.
                    found.append(f'{k}={re.sub(r"\s", "_", x)}')
                walk(k)
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)

    try:
        walk(json.loads(text))
    except ValueError:
        lines = [l for l in text.splitlines() if l.strip()]
        if not lines:
            return None
        try:
            for line in lines:
                walk(json.loads(line))
        except ValueError:
            return None
    return '\n'.join(found)


ESCAPE = re.compile(r'\\(?:u([0-9a-fA-F]{4})|x([0-9a-fA-F]{2}))')


def unescape(text: str) -> str:
    """The text with \\uXXXX and \\xXX escapes replaced by the characters they denote. A token
    written as `abcdefgh\\u002dijkl` in a log line, or in a JSON document that is not parsed as a
    whole, would otherwise never line up with the FORBIDDEN pattern (review #53)."""
    if '\\' not in text:
        return text
    return ESCAPE.sub(lambda m: chr(int(m.group(1) or m.group(2), 16)), text)


def forbidden_in(text: str) -> bool:
    """FORBIDDEN over the text, over its decoded strings when it is JSON, and over both with
    character escapes decoded."""
    views = [text, unescape(text)]
    decoded = json_strings(text)
    if decoded is not None:
        views += [decoded, unescape(decoded)]
    return any(FORBIDDEN.search(v) for v in views)


FLOAT_DTYPES = {'F64': 8, 'F32': 4, 'F16': 2, 'BF16': 2}   # element sizes in bytes


def bytes_read_as_text(payload: bytes) -> bool:
    """Whether the bytes contain a forbidden pattern under any encoding text could hide in:
    one byte per character, UTF-16 in either order, or ASCII padded with NULs (what UTF-16 or
    UTF-32 ASCII looks like once the NULs are dropped)."""
    views = [payload.decode('latin-1'), payload.replace(b'\0', b'').decode('latin-1'),
             payload.decode('utf-16-le', 'ignore'), payload.decode('utf-16-be', 'ignore')]
    return any(forbidden_in(v) for v in views)


def safetensors_payload_ok(raw: bytes, header: str) -> bool:
    """The data section must hold only floating-point tensors whose offsets tile it exactly and
    whose shape times element size equals their span (so a loader can read them), and must not
    read as forbidden text: a U8 tensor can carry arbitrary bytes."""
    n = int.from_bytes(raw[:8], 'little')
    payload = raw[8 + n:]
    spans = []
    for name, spec in json.loads(header).items():
        if name == '__metadata__':
            if not (isinstance(spec, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in spec.items())):
                return False   # the format allows only a string-to-string map here
            continue
        if not isinstance(spec, dict) or spec.get('dtype') not in FLOAT_DTYPES:
            return False
        off, shape = spec.get('data_offsets'), spec.get('shape')
        if not (isinstance(off, list) and len(off) == 2 and all(type(x) is int for x in off) and 0 <= off[0] <= off[1] <= len(payload)):
            return False
        if not (isinstance(shape, list) and all(type(d) is int and d >= 0 for d in shape)):
            return False
        count = 1
        for d in shape:
            count *= d   # Python integers do not overflow
        if count * FLOAT_DTYPES[spec['dtype']] != off[1] - off[0]:
            return False
        spans.append((off[0], off[1]))
    spans.sort()
    end = 0
    for a, b in spans:
        if a != end:
            return False
        end = b
    return end == len(payload) and not bytes_read_as_text(payload)


def safetensors_header(raw: bytes) -> str | None:
    """The JSON header of a safetensors file as text, or None if the container is malformed.
    The header is the only place such a file can carry strings (tensor names, dtypes,
    `__metadata__`), so it is what the forbidden-pattern scan reads."""
    if len(raw) < 8:
        return None
    n = int.from_bytes(raw[:8], 'little')
    if n == 0 or 8 + n > len(raw):
        return None
    try:
        text = raw[8:8 + n].decode('utf-8')
        if not isinstance(json.loads(text), dict):
            return None
    except (UnicodeDecodeError, ValueError):
        return None
    return text


def prepare(path: Path, raw: bytes) -> tuple[bytes | None, bool, str]:
    """Return (archived bytes, rewritten?, reason). None means exclude. `raw` is the bounded
    snapshot read_snapshot() returned: that one read serves the content checks, the archive
    and the manifest, and nothing here opens the path again."""
    if path.suffix.lower() == '.safetensors':
        header = safetensors_header(raw)
        if header is None:
            return None, False, 'malformed safetensors'
        if forbidden_in(header):
            return None, False, 'forbidden content'
        if not safetensors_payload_ok(raw, header):
            return None, False, 'safetensors payload'
        return raw, False, 'included'
    if raw[:4] in MACHO or b'\0' in raw:
        return None, False, 'binary'   # Mach-O, NUL-padded (UTF-16) or binary data under a text name (review #40)
    try:
        text = raw.decode('utf-8')
    except UnicodeDecodeError:
        # Evidence files are UTF-8. Anything else cannot be rewritten or scanned reliably, so it
        # is excluded rather than archived raw (review #11).
        return None, False, 'undecodable text'
    new = text
    for pat, repl in REWRITES:
        new = pat.sub(repl, new)
    if forbidden_in(new):
        return None, False, 'forbidden content'
    return new.encode('utf-8'), new != text, 'included'


def open_tree(golden: Path) -> int:
    """A descriptor for the evidence root. Every snapshot is opened relative to it, one
    component at a time, so no directory below the root may be a symlink at read time: a
    directory the walk already passed can be swapped for a link to an external tree holding
    the same file name, which O_NOFOLLOW on the final component alone would traverse
    (review #62). The root itself may be a link; the operator chose it."""
    if os.open not in os.supports_dir_fd or os.stat not in os.supports_dir_fd:   # lstat is stat without following
        raise SystemExit('this platform cannot open paths relative to a directory descriptor')
    return os.open(golden, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)


def read_snapshot(root_fd: int, rel: Path, limit: int) -> tuple[bytes | None, os.stat_result]:
    """The bytes and metadata of `rel` below the root descriptor, read once; a file whose size, inode or mtime
    differ between the stat before and the stat after the read is being changed, and the build
    stops rather than describe a snapshot that never existed (review #50). A file longer than
    `limit` by the time it is read returns None: the stat-time size check does not bind the
    bytes actually read (review #54). The open does not follow a symlink, the descriptor must
    be a regular file, and the path must still name that inode after the read: classify()
    rejected links by name, and the name can be replaced by one in between (review #55). The
    open is non-blocking so a FIFO cannot stall it before the type check (review #61); reads of
    a regular file ignore that flag. A link count above one is refused here as in classify()
    (review #60). Each directory component is opened with O_DIRECTORY | O_NOFOLLOW relative
    to its parent, and the final lstat is relative to that parent too (review #62)."""
    dirs: list[int] = []
    try:
        parent = root_fd
        try:
            for part in rel.parts[:-1]:
                parent = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=parent)
                dirs.append(parent)
            fd = os.open(rel.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=parent)
        except OSError as e:   # ELOOP for a link, ENOTDIR for a swapped directory, a vanished file
            raise SystemExit(f'{rel}: cannot open: {e.strerror}')
        with os.fdopen(fd, 'rb') as fh:
            before = os.fstat(fh.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise SystemExit(f'{rel} is not a regular file')
            if before.st_nlink != 1:
                raise SystemExit(f'{rel} has {before.st_nlink} links')
            raw = fh.read(limit + 1)   # never more than the limit allows into memory
            after = os.fstat(fh.fileno())
        named = os.stat(rel.name, dir_fd=parent, follow_symlinks=False)
    finally:
        for d in dirs:
            os.close(d)
    if (named.st_dev, named.st_ino) != (before.st_dev, before.st_ino):
        raise SystemExit(f'{rel} was replaced while it was being archived')
    if len(raw) > limit:
        return None, before
    same = (before.st_size, before.st_ino, before.st_dev, before.st_mtime_ns) == (after.st_size, after.st_ino, after.st_dev, after.st_mtime_ns)
    if not same or len(raw) != before.st_size:
        raise SystemExit(f'{rel} changed while it was being archived')
    return raw, before


def walk_error(error: OSError):
    # os.walk() would otherwise skip the directory and the archive would be complete by
    # appearance only (review #56).
    raise SystemExit(f'{error.filename}: cannot scan: {error.strerror}')


def collect(golden: Path):
    root_fd = open_tree(golden)
    try:
        return collect_below(golden, root_fd)
    finally:
        os.close(root_fd)


def collect_below(golden: Path, root_fd: int):
    included, excluded = [], {}
    total = 0
    for dirpath, dirnames, filenames in os.walk(golden, onerror=walk_error):
        dirnames.sort()
        for name in list(dirnames):
            reason = path_reason(golden, Path(dirpath) / name, directory=True)
            if reason is not None:
                dirnames.remove(name)
                excluded[f'{reason} (directories)'] = excluded.get(f'{reason} (directories)', 0) + 1
        for name in sorted(filenames):
            path = Path(dirpath) / name
            ok, reason = classify(golden, path)
            if not ok:
                excluded[reason] = excluded.get(reason, 0) + 1
                continue
            tensor = path.suffix.lower() == '.safetensors'
            raw, st = read_snapshot(root_fd, path.relative_to(golden), SMALL_TENSOR if tensor else MAX_TEXT)
            if raw is None:
                reason = 'tensor size' if tensor else 'oversized text'
                excluded[reason] = excluded.get(reason, 0) + 1
                continue
            data, rewritten, reason = prepare(path, raw)
            if data is None:
                excluded[reason] = excluded.get(reason, 0) + 1
                continue
            # Provenance comes from this one read and stat, not from a later look at a file that
            # may have changed meanwhile (review #48).
            source = {'source_bytes': len(raw), 'source_sha256': hashlib.sha256(raw).hexdigest(),
                      'mtime': datetime.datetime.fromtimestamp(st.st_mtime, datetime.timezone.utc).isoformat(),
                      'mtime_s': st.st_mtime}
            total += len(data)
            if total > MAX_TOTAL:
                raise SystemExit(f'included evidence exceeds MAX_TOTAL ({MAX_TOTAL} bytes) at {path}')
            included.append((path.relative_to(golden).as_posix(), path, data, rewritten, source))
    return included, excluded


LABEL = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$')


def build(golden: Path, out: Path, label: str | None) -> dict:
    if label is not None and (not LABEL.match(label) or '..' in label or forbidden_in(label)):
        # The label is every member's leading path component; keep it a single safe name so
        # an extractor that honours '..' or '/' cannot be steered outside its directory.
        raise SystemExit(f'label {label!r} must be a single path component [A-Za-z0-9._-]')
    if out.exists():
        raise SystemExit(f'{out} exists; evidence archives are never overwritten')
    included, excluded = collect(golden)
    entries = []
    for rel, path, data, rewritten, source in included:
        entries.append({
            'path': rel, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
            'source_bytes': source['source_bytes'], 'source_sha256': source['source_sha256'],
            'rewritten': rewritten, 'mtime': source['mtime'],
        })
    if label is None:
        # Derived from the tree, like the manifest timestamp, so the default stays reproducible.
        newest = max((e['mtime'] for e in entries), default='1970-01-01T00:00:00+00:00')
        label = f'cleffa-evidence-{newest[:10]}'
    manifest = {
        'label': label,
        # Derived from the tree, not the wall clock, so rebuilding the same tree gives the same bytes.
        'built': max((e['mtime'] for e in entries), default='1970-01-01T00:00:00+00:00'),
        'rules': (__doc__ or '').split('\n', 2)[2].strip(),
        'rewrites': ['<repo>', '<home>', '<cf-account>'],
        'files': entries,
        'excluded_counts': dict(sorted(excluded.items())),
    }
    readme = README.format(label=label, n=len(entries), rewritten=sum(e['rewritten'] for e in entries),
                           excluded=json.dumps(manifest['excluded_counts'], indent=2))
    # Final control: the archived bytes, the member names and the generated members must not
    # contain any forbidden pattern.
    for name, text in (('README.md', readme), ('manifest.json', json.dumps(manifest))):
        if forbidden_in(text):
            raise SystemExit(f'forbidden pattern in the generated {name}')
    for rel, path, data, _, _ in included:
        if FORBIDDEN.search(rel):
            raise SystemExit(f'forbidden pattern in the path {rel}')
        tensor = path.suffix.lower() == '.safetensors'
        if not tensor and b'\0' in data:
            raise SystemExit(f'NUL byte survived in {rel}')
        text = safetensors_header(data) if tensor else data.decode('utf-8', 'replace')
        if text is None or forbidden_in(text) or (tensor and not safetensors_payload_ok(data, text)):
            raise SystemExit(f'forbidden pattern survived rewriting in {rel}')
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        fh = open(out, 'xb')   # exclusive: the check above is not atomic with the create (review #47)
    except FileExistsError:
        raise SystemExit(f'{out} exists; evidence archives are never overwritten') from None
    with fh, gzip.GzipFile(filename='', mode='wb', fileobj=fh, mtime=0) as gz, \
            tarfile.open(fileobj=gz, mode='w', format=tarfile.PAX_FORMAT) as tar:
        def add(name: str, data: bytes, mtime: float):
            info = tarfile.TarInfo(f'{label}/{name}')
            info.size = len(data)
            info.mtime = int(mtime)
            info.mode = 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ''
            tar.addfile(info, io.BytesIO(data))
        add('README.md', readme.encode(), 0)
        add('manifest.json', json.dumps(manifest, indent=1).encode(), 0)
        for rel, path, data, _, source in included:
            add(rel, data, source['mtime_s'])
    return manifest


README = '''# {label}

Curated evidence for the reports in `docs/` of the cleffa repository: {n} files copied from
the local `golden/` tree by `tools/evidence_archive.py`, with their SHA-256 before and after
copying in `manifest.json`. {rewritten} text files were rewritten to replace the local
checkout path, home directory and Cloudflare account name with `<repo>`, `<home>` and
`<cf-account>`; nothing else was edited. Each report names the directories it relies on.

Not included, by rule: engine binaries and objects, Instruments traces and their counter
exports, model tensors and layer dumps, the MLX environment, the ds4 upstream clone, the
ContractNLI dataset archive (its commit hash is in `contractnli-*/source.json`; the dataset is
CC BY 4.0, Koreeda and Manning, Findings of EMNLP 2021), Cloudflare account and usage dumps,
and every file from the private article-classification workload.

Excluded counts by reason (files, or whole directories pruned before being listed):

```
{excluded}
```

The oracle directories (`clef-flash`, `clef-flash-f32`, `clef`, `clef-f32`, ...) contain only
`requests.jsonl`, `encoded.jsonl`, `logits.safetensors` and `latency.json`, which is enough
for `tests/test_parity.py` without `--dump`; regenerate `layers/` with `ref/oracle.py` when a
per-layer comparison is needed.
'''


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or '').split('\n')[0])
    ap.add_argument('out', type=Path, help='output .tar.gz (must not exist)')
    ap.add_argument('--golden', type=Path, default=Path(__file__).resolve().parent.parent / 'golden')
    ap.add_argument('--label', default=None,
                    help='top-level directory name inside the archive; default cleffa-evidence-<date of the newest included file>')
    ap.add_argument('--list', action='store_true', help='print the selection and exit without writing')
    args = ap.parse_args(argv)
    golden = args.golden.resolve()
    if not golden.is_dir():
        raise SystemExit(f'{golden} is not a directory')
    if args.list:
        included, excluded = collect(golden)
        total = sum(len(d) for _, _, d, _, _ in included)
        rewritten = sum(1 for _, _, _, r, _ in included if r)
        for rel, _, data, r, _ in included:
            print(f'{len(data):10d}  {"R" if r else " "}  {rel}')
        print(f'\n{len(included)} files, {total/1e6:.1f} MB, {rewritten} rewritten', file=sys.stderr)
        for reason, n in sorted(excluded.items()):
            print(f'  excluded {n:6d}  {reason}', file=sys.stderr)
        return 0
    manifest = build(golden, args.out, args.label)
    total = sum(e['bytes'] for e in manifest['files'])
    print(f'{args.out}: {len(manifest["files"])} files, {total/1e6:.1f} MB uncompressed, '
          f'{args.out.stat().st_size/1e6:.1f} MB compressed, sha256 '
          f'{hashlib.sha256(args.out.read_bytes()).hexdigest()}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
