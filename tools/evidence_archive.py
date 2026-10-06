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
    that look like credentials,
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
anywhere, which is what UTF-16 text would need to hide in them.
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
import sys
import tarfile
from pathlib import Path

SMALL_TENSOR = 2 * 1024 * 1024
MAX_TEXT = 64 * 1024 * 1024

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
FORBIDDEN = re.compile(r'/Users/[A-Za-z]|/home/[a-z]|/root/|/var/root/|' +
                       re.escape(str(Path.home())) + '|' + re.escape(str(Path(__file__).resolve().parent.parent)) + '|'
                       r'squid|\.personal|pop_v22|account_id["\']?\s*[=:]\s*["\']?[0-9a-f]{32}|'
                       r'Bearer [A-Za-z0-9_\-]{16,}|CLOUDFLARE_API_TOKEN=\S|Zknpr|session_id|'
                       r'accounts/[0-9a-f]{32}|CLOUDFLARE_ACCOUNT_ID["\']?\s*[=:]\s*["\']?[0-9a-f]{32}|'
                       r'\b[A-Z0-9_]*(API_KEY|SECRET|TOKEN|PASSWORD)\s*[=:]\s*["\']?[A-Za-z0-9_\-]{16,}', re.IGNORECASE)


def is_text(path: Path) -> bool:
    """Text by extension or, without one, by content; a NUL byte anywhere disqualifies either,
    so binary or NUL-padded (UTF-16) data under a text name is excluded rather than archived.
    Files are bounded by MAX_TEXT, so reading them whole here is affordable."""
    if not (path.suffix.lower() in TEXT_EXT or path.name in TEXT_NAMES):
        return False   # an extensionless file is admitted by name only (review #42)
    if path.stat().st_size > MAX_TEXT:
        return True   # classify() excludes it as oversized without reading it
    raw = path.read_bytes()
    return raw[:4] not in MACHO and b'\0' not in raw


def classify(golden: Path, path: Path) -> tuple[bool, str]:
    """(include?, reason). Reasons are stable strings used for the --list summary."""
    rel = path.relative_to(golden)
    parts = rel.parts
    top = parts[0]
    if path.is_symlink():
        return False, 'symlink'
    # Before any allowlist: "private" anywhere in the relative path excludes the file, and so
    # does a forbidden pattern in the path itself, which becomes a tar member name.
    if any('private' in part.lower() for part in parts):
        return False, 'private-named path'
    if any(part.startswith('.') for part in parts):
        return False, 'dotfile'   # .env, .gitignore, editor state: never evidence
    if FORBIDDEN.search(rel.as_posix()):
        return False, 'forbidden path'
    if any(p in EXCLUDE_DIR_PARTS for p in parts):
        return False, 'clone or environment'
    if any(p.endswith('.trace') or p.endswith('.dSYM') for p in parts[:-1]):
        return False, 'trace or dSYM bundle'
    if len(parts) == 1 and not (path.suffix == '.jsonl' and path.name.startswith('engine_logits')):
        # Top-level files: only the engine outputs the parity tests write (size-checked below).
        return False, 'top-level file'
    if any(part.startswith('article-') for part in parts[:-1]):
        return False, 'private workload directory'
    if top.startswith('ds4-') and len(parts) > 2 and parts[1] == 'source':
        return False, 'upstream clone'
    if path.name.startswith('cleffa-evidence-'):
        return False, 'archive output'
    if THIRD_PARTY_DOC.match(path.name):
        return False, 'third-party document'
    if ACCOUNT_BOOKKEEPING.match(path.name):
        return False, 'account bookkeeping'
    if AGENT_STATE.match(path.name):
        return False, 'agent checkpoint'
    if len(parts) > 1 and not EXPERIMENT.match(top):
        # Only the oracle directories ref/oracle.py writes, and of those the small files, never
        # layers/ or dumps (size-checked below). Any other undated directory is not evidence.
        if not ORACLE_DIR.match(top):
            return False, 'unrecognized directory'
        if not (len(parts) == 2 and path.name in ORACLE_FILES):
            return False, 'oracle directory'
    ext = path.suffix.lower()
    if ext in EXCLUDE_EXT:
        return False, f'excluded extension {ext}'
    if ext == '.safetensors':
        return path.stat().st_size <= SMALL_TENSOR, 'tensor size'
    if not is_text(path):
        return False, 'binary'
    if path.stat().st_size > MAX_TEXT:
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


def forbidden_in(text: str) -> bool:
    """FORBIDDEN over the text and, when it is JSON, over its decoded strings."""
    if FORBIDDEN.search(text):
        return True
    decoded = json_strings(text)
    return decoded is not None and FORBIDDEN.search(decoded) is not None


FLOAT_DTYPES = {'F64', 'F32', 'F16', 'BF16'}


def bytes_read_as_text(payload: bytes) -> bool:
    """Whether the bytes contain a forbidden pattern under any encoding text could hide in:
    one byte per character, UTF-16 in either order, or ASCII padded with NULs (what UTF-16 or
    UTF-32 ASCII looks like once the NULs are dropped)."""
    views = [payload.decode('latin-1'), payload.replace(b'\0', b'').decode('latin-1'),
             payload.decode('utf-16-le', 'ignore'), payload.decode('utf-16-be', 'ignore')]
    return any(FORBIDDEN.search(v) for v in views)


def safetensors_payload_ok(raw: bytes, header: str) -> bool:
    """The data section must hold only floating-point tensors whose offsets tile it exactly,
    and must not read as forbidden text: a U8 tensor can carry arbitrary bytes."""
    n = int.from_bytes(raw[:8], 'little')
    payload = raw[8 + n:]
    spans = []
    for name, spec in json.loads(header).items():
        if name == '__metadata__':
            continue
        if not isinstance(spec, dict) or spec.get('dtype') not in FLOAT_DTYPES:
            return False
        off = spec.get('data_offsets')
        if not (isinstance(off, list) and len(off) == 2 and all(isinstance(x, int) for x in off) and 0 <= off[0] <= off[1] <= len(payload)):
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


def prepare(path: Path) -> tuple[bytes | None, bool, str]:
    """Return (archived bytes, rewritten?, reason). None means exclude."""
    raw = path.read_bytes()
    if path.suffix.lower() == '.safetensors':
        header = safetensors_header(raw)
        if header is None:
            return None, False, 'malformed safetensors'
        if forbidden_in(header):
            return None, False, 'forbidden content'
        if not safetensors_payload_ok(raw, header):
            return None, False, 'safetensors payload'
        return raw, False, 'included'
    if b'\0' in raw:
        return None, False, 'binary'   # NUL-padded (UTF-16) or binary data under a text name (review #40)
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


def collect(golden: Path):
    included, excluded = [], {}
    for dirpath, dirnames, filenames in os.walk(golden):
        dirnames.sort()
        for name in sorted(filenames):
            path = Path(dirpath) / name
            ok, reason = classify(golden, path)
            if not ok:
                excluded[reason] = excluded.get(reason, 0) + 1
                continue
            data, rewritten, reason = prepare(path)
            if data is None:
                excluded[reason] = excluded.get(reason, 0) + 1
                continue
            included.append((path.relative_to(golden).as_posix(), path, data, rewritten))
    return included, excluded


LABEL = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$')


def build(golden: Path, out: Path, label: str | None) -> dict:
    if label is not None and (not LABEL.match(label) or '..' in label):
        # The label is every member's leading path component; keep it a single safe name so
        # an extractor that honours '..' or '/' cannot be steered outside its directory.
        raise SystemExit(f'label {label!r} must be a single path component [A-Za-z0-9._-]')
    if out.exists():
        raise SystemExit(f'{out} exists; evidence archives are never overwritten')
    included, excluded = collect(golden)
    entries = []
    for rel, path, data, rewritten in included:
        st = path.stat()
        entries.append({
            'path': rel, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
            'source_bytes': st.st_size, 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'rewritten': rewritten, 'mtime': datetime.datetime.fromtimestamp(st.st_mtime, datetime.timezone.utc).isoformat(),
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
    # Final control: the archived bytes must not contain any forbidden pattern.
    for rel, path, data, _ in included:
        if FORBIDDEN.search(rel):
            raise SystemExit(f'forbidden pattern in the path {rel}')
        tensor = path.suffix.lower() == '.safetensors'
        if not tensor and b'\0' in data:
            raise SystemExit(f'NUL byte survived in {rel}')
        text = safetensors_header(data) if tensor else data.decode('utf-8', 'replace')
        if text is None or forbidden_in(text) or (tensor and not safetensors_payload_ok(data, text)):
            raise SystemExit(f'forbidden pattern survived rewriting in {rel}')
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'wb') as fh, gzip.GzipFile(filename='', mode='wb', fileobj=fh, mtime=0) as gz, \
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
        for rel, path, data, _ in included:
            add(rel, data, path.stat().st_mtime)
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

Excluded file counts by reason:

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
        total = sum(len(d) for _, _, d, _ in included)
        rewritten = sum(1 for *_, r in included if r)
        for rel, _, data, r in included:
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
