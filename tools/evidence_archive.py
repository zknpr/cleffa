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
  - documentation pages saved from Apple's developer site (recognized by their terms link)
  - the engine source inside copied engine trees (a directory holding clef.c or
    clef_engine.h: its tests/, bench/, tools/, metal/, ref/ and docs/ directories and its root
    clef* sources, Makefile and documents), since the repository holds the source; result
    files and an experiment's own probe programs stored beside it stay
  - the ds4 upstream `source/` tree, every `article-*` directory at any depth (private
    workload), any path that itself matches a FORBIDDEN pattern, and the
    text extracts of Apple's Metal Shading Language specification
  - every other undated directory, Cloudflare subscription and usage dumps (`subscriptions.json`,
    `usage-*.json`) and agents'
    `checkpoint*.json` working-state files
  - dotfiles and extensionless files other than Makefile and LICENSE, key=value assignments
    that look like credentials, Bearer and Basic authorization values, PEM private keys,
    hard-linked files and anything that is not a regular file
  - any file with "private" in any component of its path, and any text file that still matches a
    FORBIDDEN pattern after rewriting (private-workload paths, account identifiers,
    including a Cloudflare account ID inside a recorded `accounts/<id>/` API URL)
Rewritten (text files only, recorded per file in the manifest)
  - the local checkout path and home directory become <repo> and <home>; an agent's scratchpad
    path under /private/tmp, which carries the checkout path with dashes, becomes <scratch>
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
import base64
import binascii
import datetime
import functools
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
# Experiments copy the engine tree (engine/, reviewed/, builds/baseline/, or the experiment root
# itself). A directory holding either marker is such a copy; inside it the engine's own
# directories and root files are source the repository already holds, while result files
# stored beside them (qualification logs, build manifests) are evidence and stay.
ENGINE_MARKERS = ('clef.c', 'clef_engine.h')
ENGINE_DIRS = {'tests', 'bench', 'tools', 'metal', 'ref', 'docs', 'golden', 'gguf', 'model', 'model-flash'}
ENGINE_ROOT_FILES = {'Makefile', 'LICENSE', 'THIRD_PARTY_NOTICES.md', 'README.md', 'CLAUDE.md', 'AGENTS.md',
                     'pyrightconfig.json', 'requirements.txt', 'smoke_test.py', 'download_models.sh', 'release.sh'}
ENGINE_ROOT_EXT = {'.d', '.o'}
# The engine's own root sources are all named clef*; an experiment's probe program beside them
# (scorer.c, timed_head.c) is that experiment's evidence and stays.
ENGINE_ROOT_SOURCE = re.compile(r'^clef[A-Za-z0-9_]*\.(c|h|m|inc)$')
# Text extracts of Apple's Metal Shading Language specification kept beside some experiments.
THIRD_PARTY_DOC = re.compile(r'^(msl|metal-spec|Metal-Shading-Language-Specification)\.(txt|pdf)$', re.IGNORECASE)
# Cloudflare account subscription and usage dumps: budget bookkeeping, not evidence.
ACCOUNT_BOOKKEEPING = re.compile(r'^(subscriptions|usage([-_.].*)?)\.json$', re.IGNORECASE)
# Agents' own working-state files (goal, sessions, next action), not experiment evidence.
AGENT_STATE = re.compile(r'^checkpoint[-.a-zA-Z0-9]*\.json$', re.IGNORECASE)   # name rules fold case (review #103)
EXCLUDE_EXT = {'.o', '.a', '.dylib', '.inc', '.bin', '.npy', '.npz', '.pt', '.xml', '.pdf',
               '.gz', '.zip', '.tar', '.zst', '.xz', '.bz2', '.7z', '.dmg', '.pkg'}

def path_pattern(path: str) -> str:
    """`path` where it is a path: not inside a URL authority (the `//root` of
    `mysql://root:pw@db` is not a home directory) and not followed by a name
    character (`/rooted/` is another directory). Shared by the rewrite and the forbidden
    pattern, so a home of /root neither rewrites a connection string's userinfo nor excludes
    a file for `/rooted/path` after the rewrite left it alone (reviews #94, #100). What makes
    `//root` an authority is the `:/` before it; a path after `file://` or with a doubled
    leading slash (`//workspace/cleffa/x`) is still a path (reviews #120, #128)."""
    return r'(?<!:/)' + re.escape(path) + r'(?![A-Za-z0-9_.-])'


def path_rewrite(path: str, placeholder: str) -> tuple[re.Pattern, str]:
    return re.compile(path_pattern(path)), placeholder


# Claude Code keeps a per-session scratchpad at /private/tmp/claude-<uid>/<checkout with the
# slashes turned into dashes>/<session uuid>/scratchpad; experiment scripts embed it. The path
# rewrites do not see the mangled checkout, so it is rewritten whole, and any mangled home that
# survives, or another session's scratchpad, is forbidden (hand skim of the 2026-10-06 tree).
MANGLED_REPO = '-' + str(Path(__file__).resolve().parent.parent).strip('/').replace('/', '-')
MANGLED_HOME = '-' + str(Path.home()).strip('/').replace('/', '-')
# Rewrites run in order; the scratchpad precedes the checkout path, which precedes the home.
REWRITES = [
    (re.compile(r'/private/tmp/claude-\d+/' + re.escape(MANGLED_REPO) + r'/[0-9a-f-]{36}/scratchpad'), '<scratch>'),
    path_rewrite(str(Path(__file__).resolve().parent.parent), '<repo>'),
    path_rewrite(str(Path.home()), '<home>'),
    (re.compile(r'Zknpr'), '<cf-account>'),
]
# Anything matching after rewriting excludes the file, and a match in the final scan fails
# the build. Case-insensitive: bearer schemes and variable names vary in case. Keep these broad: a false exclusion costs one evidence file, a miss publishes it.
# A credential value is a quoted string of sixteen or more characters, spaces and line breaks
# included up to a bound that keeps an unterminated quote from spanning a file, or an unquoted
# run of sixteen or more characters that are not whitespace or a quote: passwords carry
# punctuation and spaces, base64 tokens carry + / = (reviews #59, #63, #86). A URL whose userinfo
# has a password (scheme://user:password@host, any key name) is a credential too (review #85);
# a user alone or a port is not. The key accepts the
# spellings CREDENTIAL_KEY does (api-key, apiKey, x-api-key) so the rejoined JSON pair and plain
# text match alike (review #75). PEM private-key delimiters of every kind are forbidden outright
# (review #76); certificates and public keys are not secrets. The sensitive word may be followed
# by further segments of a compound credential name, separated by _ or - or by a camel-case
# capital (AWS_SECRET_ACCESS_KEY, secretAccessKey, password_hash), drawn from the words such
# names are built of; "tokenizer" is not a token, and a result field such as
# token_neuron_estimate or token_negative_control is a measurement, not a credential
# (review #79); base, salt, seed and material cover SECRET_KEY_BASE and its kin (review #91).
# The capital test is case-sensitive inside an otherwise case-insensitive pattern.
KEY_WORDS = 'access|key|id|secret|token|value|hash|str|string|pass|pwd|auth|private|signing|base|salt|seed|material'
# The words that name a credential: API keys, key material named for its use (private, signing,
# encryption, access), secrets, tokens and passwords in their abbreviations too (PASS, PWD,
# PASSWD, PASSPHRASE; reviews #80, #88). Public keys are not secrets.
# Longer password spellings precede PASS so the whole word is consumed before the suffix rule.
# The abbreviations PASS and PWD need a preceding name segment (DB_PASS, dbPass, ssh-pwd): on
# their own they are a loop variable (`pass=0;pass<2;pass++`), a Python statement or a `PASS:`
# log marker, and the unrestricted form excluded ten source files of the 2026-10-06 tree.
SENSITIVE = (r'API[_-]?KEY|PRIVATE[_-]?KEY|SIGNING[_-]?KEY|ENCRYPTION[_-]?KEY|ACCESS[_-]?KEY|SECRET|TOKEN|'
             r'PASSPHRASE|PASSWORD|PASSWD|(?:[A-Z0-9]+[_-]|(?-i:[a-z0-9]+))(?:PASS|PWD)')
KEY_SUFFIX = (r'(?:[_-](?:' + KEY_WORDS + r')|(?-i:(?:' +
              '|'.join(w.capitalize() for w in KEY_WORDS.split('|')) + r')))*')
# A strongly named credential (password, passphrase, secret, API or key material, not the
# generic token or the pass/pwd abbreviations) is a credential whatever its value's length:
# `PASSWORD=hunter2` is a password (review #95). The value may start with any character
# (`!hunter2`, `/abc123`, `$upersecret`; reviews #105, #111) except an escape or an opening
# structure: a backslash (`secret:\` before an escaped line break in three ContractNLI texts),
# a brace (a schema object), a bracket, a parenthesis or an angle bracket (placeholders), and
# runs to a quote, comma, semicolon or bracket, however short (`PASSWORD=123`; review #114);
# on the 2026-10-06 tree that form excludes nothing. Only a placeholder is not a value: an
# empty one,
# null/none/true/false, <redacted>, (none), ${VAR}, {{var}}, {var} or a run of asterisks, and
# the value is on the key's line: prose
# such as "kept secret:" followed by a new sentence is not an assignment. A YAML block scalar
# after a sensitive key (`password: |-` with the value on the next lines) is rejected on the
# indicator alone, since the value cannot be matched inline (review #96).
# A bare "token" stays generic (a loop variable, a tokenizer field); a compound token name
# (API_TOKEN, AUTH_TOKEN, accessToken, refresh_token) names a credential at any length (review #122).
STRONG = (r'API[_-]?KEY|PRIVATE[_-]?KEY|SIGNING[_-]?KEY|ENCRYPTION[_-]?KEY|ACCESS[_-]?KEY|SECRET|PASSPHRASE|PASSWORD|PASSWD|'
          r'(?:API|AUTH|ACCESS|BEARER|REFRESH|SESSION|CLIENT|SERVICE|ADMIN|USER|OAUTH)[_-]?TOKEN')
# A TOML or Python triple-quoted value after a sensitive key (review #101).
# The opening delimiter alone: a bounded value let a longer key through, an unbounded one would
# run to the end of a file with an unterminated string, and a sensitive key followed by a
# triple-quoted value is a credential whatever follows (reviews #119, #125).
TRIPLE = r'"""|\'\'\''
# A shell variable reference ($SECRET, uppercase by convention; case-sensitive inside the
# otherwise case-insensitive pattern) is a placeholder; $upersecret is a password.
PLACEHOLDER = (r'(?:null|none|nil|true|false|\*+|<[^>\s]*>|\$\{[^}]*\}|\{\{[^}]*\}\}|\{[^}\s]*\}|\([^)\s]*\)|'
               r'(?-i:\$[A-Z_][A-Z0-9_]*))(?![A-Za-z0-9_])')
# Two groups: paths and names the rewrites replace (scanned on the rewritten text, since a
# surviving path is a leak) and credentials (scanned on the original text as well, since a
# rewrite could alter the bytes around a secret before the pattern sees them; review #94).
PATH_PATTERNS = (r'/Users/[A-Za-z]|/home/[a-z]|/root/|/var/root/|/private/tmp/claude-\d+/|' +
                 r'(?<![A-Za-z0-9_])' + re.escape(MANGLED_HOME) + r'(?![A-Za-z0-9_])|' +
                 path_pattern(str(Path.home())) + '|' + path_pattern(str(Path(__file__).resolve().parent.parent)) + '|'
                 r'squid|\.personal|pop_v22|Zknpr|session_id|accounts/[0-9a-f]{32}|'
                 # a 32-hex value under any account label: account, account_id, accountId,
                 # accountTag, CLOUDFLARE_ACCOUNT_ID (review #99)
                 r'account(?:[_-]?(?:id|tag))?["\']?\s*[=:]\s*["\']?[0-9a-f]{32}\b')
CREDENTIAL_PATTERNS = (r'Bearer\s+["\']?[^\s"\']{16,}|CLOUDFLARE_API_TOKEN=\S|'
                       # a bare `Bearer <value>` of any length when the value ends the line or a quoted
                       # string: `Bearer hunter2`; prose ("a Bearer token in the header") runs on, and
                       # the words that follow "Bearer" in prose are not values (review #121)
                       r'Bearer[ \t]+["\']?(?!' + PLACEHOLDER + r')(?!(?:token|tokens|auth|authentication|scheme|header|value)\b)'
                       r'[^\s"\']+[ \t]*(?:\r?\n|$|["\'])|'
                       r'-----BEGIN [A-Z ]*PRIVATE KEY|'
                       r'Authorization\s*[=:]\s*(?:[A-Z][A-Z0-9-]*\s+)?(?:[^\s"\']{16,}|[^\n]*?["\'=][^\n]{8,})|'
                       # the key is explicitly Authorization: any value that is not a placeholder, of any
                       # length (`x`; review #116) and whatever its first character (`/abc1234`; review #108),
                       # when it ends the line or the quoted string; prose after the colon runs on,
                       # a `{template}` is a placeholder, and a scheme word standing alone
                       # (`"Bearer " + token` in code) is no value (review #98)
                       # quoted: the value closes with the quote that opened it; unquoted: it ends the
                       # line. An f-string prefix before a quote is then never a one-letter value.
                       r'Authorization["\']?[ \t]*[=:][ \t]*(?P<aq>["\'])(?:[A-Z][A-Z0-9-]*[ \t]+)?(?!' + PLACEHOLDER + r')'
                       r'(?!(?:Bearer|Basic|Token|ApiKey|Digest|Negotiate|NTLM|OAuth|HOBA)[ \t]*(?P=aq))'
                       r'[^\s"\']+[ \t]*(?P=aq)|'
                       # a quoted token after a scheme word (`Token 'abc'`, `ApiKey "x"`), including inside a
                       # JSON string where the inner quotes are escaped
                       r'Authorization["\']?[ \t]*[=:][ \t]*["\']?[A-Z][A-Z0-9-]*[ \t]+\\?(?P<tq>["\'])(?!' + PLACEHOLDER + r')[^\s"\'\\]+\\?(?P=tq)|'
                       # ... or a comment or annotation delimiter (`# staging`, `// prod`, `; note`)
                       r'Authorization[ \t]*[=:][ \t]*(?:[A-Z][A-Z0-9-]*[ \t]+)?(?!' + PLACEHOLDER + r')'
                       r'(?!(?:Bearer|Basic|Token|ApiKey|Digest|Negotiate|NTLM|OAuth|HOBA)[ \t]*(?:\r?\n|$|#|//|;))'
                       r'[^\s"\'#;]+[ \t]*(?:\r?\n|$|#|//|;)|'
                       r'Authorization["\']?\s*[=:]\s*(?P<dq>["\'])(?:[A-Z][A-Z0-9-]*\s+)?(?:[^\s"\'\\]{16,}|(?:(?!(?P=dq))[^\\\n])*?(?:=|\\["\'])[^\n]{8,})|'
                       r'\bgh[pousr]_[A-Z0-9]{20,}|\bgithub_pat_[A-Z0-9_]{20,}|'
                       r'\b[A-Z][A-Z0-9+.-]*://[^\s/:@"\']*:[^\s/@"\']+@|'
                       r'\b[A-Z0-9_-]*(' + SENSITIVE + r')' + KEY_SUFFIX +
                       r'["\']?\s*[=:]\s*(?:' + TRIPLE + r'|"[^"]{16,256}"|\'[^\']{16,256}\'|["\']?[^\s"\']{16,})|'
                       r'\b[A-Z0-9_-]*(?:' + STRONG + r')' + KEY_SUFFIX +
                       r'["\']?[ \t]*[=:][ \t]*["\']?(?!' + PLACEHOLDER + r')[^\s"\'\\{\[(<][^\s"\',;}\]{]*|'
                       # a command-line option named for a credential, with its value after whitespace, of
                       # any length: `--api-key abcdefghijklmnop`, `--password x` (reviews #113, #117)
                       r'(?<![A-Z0-9-])--?[A-Z0-9-]*(?:' + SENSITIVE + r')' + KEY_SUFFIX +
                       r'[ \t]+["\']?(?!' + PLACEHOLDER + r')[^\s"\'\\{\[(<][^\s"\',;}\]{]*|'
                       # curl's user:password argument, whose option is not named for a credential
                       r'(?<![A-Z0-9-])(?:--user[ \t=]+|-u[ \t=]*)["\']?[^\s"\':]+:[^\s"\']+|'
                       r'\b[A-Z0-9_-]*(?:' + SENSITIVE + r')' + KEY_SUFFIX + r'["\']?\s*:\s*[|>][-+0-9]*[ \t]*\n')
FORBIDDEN = re.compile(PATH_PATTERNS + '|' + CREDENTIAL_PATTERNS, re.IGNORECASE)
CREDENTIALS = re.compile(CREDENTIAL_PATTERNS, re.IGNORECASE)
# An Authorization header carries a credential whatever its scheme (Bearer, Basic, token, ApiKey,
# Digest or none). Unquoted (first form): the value has a run of sixteen or more characters
# after an optional scheme word, or carries a quote or an '=' followed by eight or more
# characters, which is a quoted token or Digest parameters. Quoted, as a JSON field or a YAML
# value (second form): the same, with inner quotes escaped and the match confined to the
# quoted string, so a template such as `f"Bearer {api_token}"` followed by other fields on the
# line stays. GitHub tokens carry a recognizable prefix and are forbidden on their own
# (reviews #81, #82).
# `Basic <base64>` authorization: the value decodes to user:password. Only a decoded colon makes
# it a credential; "basic test" is the word before a word that happens to be valid base64 (review #66).
BASIC_AUTH = re.compile(r'\bBasic\s+["\']?([A-Za-z0-9+/]{4,}={0,2})', re.IGNORECASE)   # the value may be quoted
# A credential stored as a JSON field: a key named like one, with a string value long enough to be one.
CREDENTIAL_KEY = re.compile('(' + SENSITIVE + ')' + KEY_SUFFIX + '$', re.IGNORECASE)


def is_text(path: Path) -> bool:
    """Text by name: a listed extension, or one of the two extensionless names (review #42).
    Content is judged by prepare() on the bounded snapshot, never here: a read made after the
    stat-time size check would be unbounded (review #58)."""
    return path.suffix.lower() in TEXT_EXT or path.name in TEXT_NAMES


@functools.lru_cache(maxsize=None)
def engine_copy(directory: str) -> bool:
    return any(os.path.lexists(os.path.join(directory, marker)) for marker in ENGINE_MARKERS)


def copied_source(golden: Path, parts: tuple[str, ...], directory: bool) -> bool:
    """True for a path that is engine source inside a copied engine tree: an engine directory
    (tests/, bench/, ...) directly under the copy's root, or a root file of the engine by name
    or extension. Everything else in the copy is a result and stays."""
    for depth in range(len(parts) - 1, -1, -1):   # the copy's root is the nearest marker directory
        root = golden.joinpath(*parts[:depth])
        if engine_copy(str(root)):
            inside = parts[depth:]
            if len(inside) == 1:
                name = inside[0]
                return (directory and name in ENGINE_DIRS) or (not directory and (
                    name in ENGINE_ROOT_FILES or ENGINE_ROOT_SOURCE.match(name) is not None
                    or Path(name).suffix.lower() in ENGINE_ROOT_EXT))
            return inside[0] in ENGINE_DIRS
    return False


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
    if any('\\' in part or part in ('.', '..') or any(ord(c) < 32 for c in part) for part in parts):
        # The name becomes a tar member name. A backslash is legal on macOS but a path
        # separator to a Windows extractor, which would read an embedded `..` as traversal.
        return 'unsafe name'
    # Before any allowlist: "private" anywhere in the relative path excludes the file, and so
    # does a forbidden pattern in the path itself, which becomes a tar member name.
    if any('private' in part.lower() for part in parts):
        return 'private-named path'
    if any(part.startswith('.') for part in parts):
        return 'dotfile'   # .env, .gitignore, editor state: never evidence
    if forbidden_in(rel.as_posix()):   # the same scan as content: FORBIDDEN and Basic (review #73)
        return 'forbidden path'
    if any(p.lower() in EXCLUDE_DIR_PARTS for p in parts):   # directory names fold case (review #106)
        return 'clone or environment'
    if any(p.lower().endswith(('.trace', '.dsym')) for p in ancestors):
        return 'trace or dSYM bundle'
    if any(part.lower().startswith('article-') for part in ancestors):   # any capitalization (review #90)
        return 'private workload directory'
    if top.lower().startswith('ds4-') and len(ancestors) >= 2 and parts[1].lower() == 'source':
        return 'upstream clone'
    if copied_source(golden, parts, directory):
        return 'copied engine source'
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
    if path.name.lower().startswith('cleffa-evidence-'):
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
                if isinstance(k, str) and isinstance(x, str) and CREDENTIAL_KEY.search(k) and x:
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


ESCAPE = re.compile(r'\\(?:u([0-9a-fA-F]{4})|x([0-9a-fA-F]{2})|([/"\\bfnrt]))')
SIMPLE_ESCAPES = {'/': '/', '"': '"', '\\': '\\', 'b': '\b', 'f': '\f', 'n': '\n', 'r': '\r', 't': '\t'}


def unescape(text: str) -> str:
    """The text with \\uXXXX, \\xXX and the simple JSON escapes (\\/ \\" \\\\ \\n ...) replaced by the
    characters they denote. A token written as `abcdefgh\\u002dijkl` in a log line, or an
    `accounts\\/<id>` URL inside a JSON fragment that is not parsed as a whole, would otherwise
    never line up with the FORBIDDEN pattern (reviews #53, #83)."""
    if '\\' not in text:
        return text
    return ESCAPE.sub(lambda m: SIMPLE_ESCAPES[m.group(3)] if m.group(3) else chr(int(m.group(1) or m.group(2), 16)), text)


def basic_credential(text: str) -> bool:
    """True when a `Basic <base64>` value decodes to something with a colon: user:password.
    Padding is restored before the strict decode: clients and logs drop it (review #68)."""
    for m in BASIC_AUTH.finditer(text):
        value = m.group(1).rstrip('=')
        try:
            decoded = base64.b64decode(value + '=' * (-len(value) % 4), validate=True)
        except (binascii.Error, ValueError):
            continue
        if b':' in decoded:
            return True
    return False


STRONG_KEY = re.compile(r'^[A-Z0-9_-]*(?:' + STRONG + r')' + KEY_SUFFIX + '$', re.IGNORECASE)
PLACEHOLDER_VALUE = re.compile(r'^(?:' + PLACEHOLDER + r')$', re.IGNORECASE)


def json_credential(text: str) -> bool:
    """A strongly named field of a JSON document with a non-empty string value that is not a
    placeholder. In decoded JSON the key and the value are unambiguous, so the first-character
    exclusions that keep plain-text escapes and structures out do not apply (review #127)."""
    def walk(v) -> bool:
        if isinstance(v, dict):
            for k, x in v.items():
                if isinstance(k, str) and isinstance(x, str) and x and STRONG_KEY.match(k) and not PLACEHOLDER_VALUE.match(x):
                    return True
                if walk(x):
                    return True
        elif isinstance(v, list):
            return any(walk(x) for x in v)
        return False
    try:
        return walk(json.loads(text))
    except ValueError:
        pass
    # A JSONL file is not one document; each record is judged like one.
    records = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except ValueError:
            return False
    return any(walk(r) for r in records)


def forbidden_in(text: str, pattern: re.Pattern = None) -> bool:
    """FORBIDDEN (or the given pattern) and the Basic-authorization check over the text, over
    its decoded strings when it is JSON, and over both with character escapes decoded."""
    pattern = FORBIDDEN if pattern is None else pattern
    views = [text, unescape(text)]
    decoded = json_strings(text)
    if decoded is not None:
        views += [decoded, unescape(decoded)]
    return any(pattern.search(v) or basic_credential(v) or json_credential(v) for v in views)


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
    if raw[8:9] != b'{':
        # The format requires the header to begin with '{' (trailing space padding only), and
        # the reference loader refuses anything else; json.loads alone would accept leading
        # whitespace and archive a tensor its consumers cannot load (review #69).
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
    # Credentials are scanned on the original text: a rewrite could alter the bytes around a
    # secret before the pattern sees them (review #94). Paths are not, since replacing them is
    # what the rewrites are for; the rewritten text is scanned for everything.
    if forbidden_in(text, CREDENTIALS):
        return None, False, 'forbidden content'
    if 'apple.com/legal/internet-services/terms/site.html' in text:   # whatever the saved page's suffix
        # A documentation page saved from Apple's developer site beside an experiment, like the
        # Metal specification extracts: third-party text, not evidence (hand skim).
        return None, False, 'third-party document'
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
    same = ((before.st_size, before.st_ino, before.st_dev, before.st_mtime_ns, before.st_ctime_ns) ==
            (after.st_size, after.st_ino, after.st_dev, after.st_mtime_ns, after.st_ctime_ns))
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
    seen: list[tuple[str, frozenset[str], tuple[int, int]]] = []
    children: list[tuple[str, tuple[int, int]]] = []   # retained child directories, as enumerated
    pruned: list[tuple[str, tuple[int, int, int]]] = []   # pruned children: device, inode, file type
    snapshots: list[tuple[Path, tuple[int, int, int, int, int]]] = []   # what each enumerated file was when examined

    def note(path: Path, st: os.stat_result):
        # ctime included: a writer can restore size and mtime after an in-place rewrite, but
        # not ctime (review #84).
        snapshots.append((path, (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)))
    for dirpath, dirnames, filenames in os.walk(golden, onerror=walk_error):
        st = os.stat(dirpath, follow_symlinks=False)
        seen.append((dirpath, frozenset(dirnames) | frozenset(filenames), (st.st_dev, st.st_ino)))
        dirnames.sort()
        for name in list(dirnames):
            reason = path_reason(golden, Path(dirpath) / name, directory=True)
            if reason is not None:
                dirnames.remove(name)
                excluded[f'{reason} (directories)'] = excluded.get(f'{reason} (directories)', 0) + 1
                # A pruned child (a link, say) replaced by a real directory afterwards keeps
                # the parent's entry names; its identity is checked at the end (review #124).
                pst = os.stat(os.path.join(dirpath, name), follow_symlinks=False)
                pruned.append((os.path.join(dirpath, name), (pst.st_dev, pst.st_ino, stat.S_IFMT(pst.st_mode))))
                continue
            # os.walk() skips a child that becomes a link before the descent, silently and with
            # no onerror call; the child must later turn up as a visited directory with this
            # identity (review #77). Keyed the way os.walk() names it.
            cst = os.stat(os.path.join(dirpath, name), follow_symlinks=False)
            if not stat.S_ISDIR(cst.st_mode):
                raise SystemExit(f'{os.path.join(dirpath, name)} changed during collection')
            children.append((os.path.join(dirpath, name), (cst.st_dev, cst.st_ino)))
        for name in sorted(filenames):
            path = Path(dirpath) / name
            try:
                st = os.stat(path, follow_symlinks=False)
                ok, reason = classify(golden, path)
            except FileNotFoundError:
                raise SystemExit(f'{path} vanished during collection') from None
            if not ok:
                # Excluded on its name or its metadata: recorded too, so a file that becomes
                # eligible after this look (replaced below a size limit, say) aborts the build
                # instead of being omitted from a tree it is part of by completion (review #74).
                note(path, st)
                excluded[reason] = excluded.get(reason, 0) + 1
                continue
            tensor = path.suffix.lower() == '.safetensors'
            raw, st = read_snapshot(root_fd, path.relative_to(golden), SMALL_TENSOR if tensor else MAX_TEXT)
            note(path, st)
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
            # No digest or size of the original bytes: the values the rewrites replace are
            # low-entropy (a username, a checkout path, an account name), and a digest of the
            # original would let a reader hash candidates against the archived text and confirm
            # them offline (review #109). The archived bytes are what the manifest describes.
            source = {'mtime': datetime.datetime.fromtimestamp(st.st_mtime, datetime.timezone.utc).isoformat(),
                      'mtime_s': st.st_mtime}
            total += len(data)
            if total > MAX_TOTAL:
                raise SystemExit(f'included evidence exceeds MAX_TOTAL ({MAX_TOTAL} bytes) at {path}')
            included.append((path.relative_to(golden).as_posix(), path, data, rewritten, source))
    # Quiescence: a file created or removed after its directory was enumerated is seen by no
    # per-file check, and the manifest would describe neither the tree at the start nor at the
    # end. Every enumerated directory must still be the same directory with the same entries
    # (review #65).
    for dirpath, listed, ident in seen:
        try:
            st = os.stat(dirpath, follow_symlinks=False)
            now = frozenset(os.listdir(dirpath))
        except OSError as e:
            raise SystemExit(f'{dirpath}: cannot re-list: {e.strerror}')
        if (st.st_dev, st.st_ino) != ident or now != listed:
            raise SystemExit(f'{dirpath} changed during collection')
    visited = {dirpath: ident for dirpath, _, ident in seen}
    for child, ident in children:
        if visited.get(child) != ident:
            raise SystemExit(f'{child} changed during collection')
    for child, ident in pruned:
        try:
            pst = os.stat(child, follow_symlinks=False)
        except OSError as e:
            raise SystemExit(f'{child}: cannot re-stat: {e.strerror}')
        if (pst.st_dev, pst.st_ino, stat.S_IFMT(pst.st_mode)) != ident:
            raise SystemExit(f'{child} changed during collection')
    # An in-place overwrite of a file already examined leaves its directory's entries unchanged,
    # so every enumerated file must still have the identity, size and mtime it was read or
    # classified with once the whole tree has been collected (reviews #72, #74). A rewrite that
    # restores all three is beyond this check; the archive describes files, not a filesystem
    # snapshot.
    for path, ident in snapshots:
        try:
            st = os.stat(path, follow_symlinks=False)
        except OSError as e:
            raise SystemExit(f'{path}: cannot re-stat: {e.strerror}')
        if (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns) != ident:
            raise SystemExit(f'{path} changed after it was examined')
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
        'rewrites': ['<scratch>', '<repo>', '<home>', '<cf-account>'],
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
        if forbidden_in(rel):
            raise SystemExit(f'forbidden pattern in the path {rel}')
        if rel.startswith('/') or any('\\' in part or part in ('.', '..') or any(ord(c) < 32 for c in part) for part in rel.split('/')):
            raise SystemExit(f'unsafe member name {rel!r}')
        tensor = path.suffix.lower() == '.safetensors'
        if not tensor and b'\0' in data:
            raise SystemExit(f'NUL byte survived in {rel}')
        text = safetensors_header(data) if tensor else data.decode('utf-8', 'replace')
        if text is None or forbidden_in(text) or (tensor and not safetensors_payload_ok(data, text)):
            raise SystemExit(f'forbidden pattern survived rewriting in {rel}')
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        # The archive is written to a sibling and linked into place only once it is complete
        # and synced, so the output path never holds a partial file that the next run would
        # refuse to overwrite or that automation could take for a finished asset (review #64).
        # link() rather than rename(): it fails if the output appeared meanwhile, so the
        # existence check above stays atomic with the create (review #47).
        partial = out.with_name(out.name + '.partial')
        fh = open(partial, 'xb')
    except FileExistsError:
        raise SystemExit(f'{partial} exists: a build was interrupted; remove it to continue') from None
    try:
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
            tar.close()
            gz.close()
            fh.flush()
            os.fsync(fh.fileno())
        os.link(partial, out)
    except FileExistsError:
        partial.unlink()
        raise SystemExit(f'{out} exists; evidence archives are never overwritten') from None
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    partial.unlink()
    return manifest


README = '''# {label}

Curated evidence for the reports in `docs/` of the cleffa repository: {n} files copied from
the local `golden/` tree by `tools/evidence_archive.py`, with the SHA-256 of the archived
bytes in `manifest.json`. {rewritten} text files were rewritten to replace the local
checkout path, home directory and Cloudflare account name with `<repo>`, `<home>` and
`<cf-account>`, and an agent scratchpad path with `<scratch>`; nothing else was edited, and
they are flagged `rewritten`. No digest of a
file's original bytes is published: the replaced values are low-entropy, and such a digest
would let a reader confirm guesses offline. Each report names the directories it relies on.

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
