"""Prefix cache checkpoints: a request that shares only part of an entry's tokens resumes from
the last checkpoint inside them, and its logits are still exactly the uncached ones.

  .venv/bin/python -B tests/test_prefix_checkpoints.py MODEL.gguf REQUESTS.jsonl --attention tu
  .venv/bin/python -B tests/test_prefix_checkpoints.py MODEL.gguf REQUESTS.jsonl --attention fp32

Run under the shared GPU lock. One CLI process serves each sequence through one entry
(--prefix-cache), with poisoned buffers, and is compared with the plain path request by request.

Two kinds of expectation. The scenarios state what the feature is for: a fixed preamble with
varying tails pays for its tail alone from the third request on, an edit costs fewer than 2,048
shared tokens, a record of the other DeltaNet class reuses nothing, a failure leaves nothing
usable. And every reused count must equal `Entry`, the policy of clef.c written out again here:
resume from the last checkpoint inside the shared tokens; store the snapshot, every 2,048th row
and the block where the request left the entry's tokens; when the slots run out, the least
recently used checkpoint goes.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess

from test_prefix_cache import exact, payload, rows

ROOT = Path(__file__).resolve().parents[1]
SLOTS, PERIOD, MIN, MARGIN = 12, 2048, 128, 8


class Entry:
    def __init__(self):
        self.ids, self.cls = [], None
        self.row, self.use, self.clock = [0] * SLOTS, [0] * SLOTS, 0

    def request(self, ids, snapshot, cls):
        """Returns the row the request resumes from and the tokens it shares with the entry."""
        same = 0
        if self.ids and self.cls == cls:
            n = min(len(self.ids), snapshot)
            while same < n and self.ids[same] == ids[same]:
                same += 1
        left = bool(self.ids) and self.cls == cls and same < len(self.ids) and same < snapshot
        self.row = [r if r <= same else 0 for r in self.row]
        resume = max(self.row)
        load = self.row.index(resume) if resume else -1
        period = PERIOD
        while (snapshot - 1) // period > SLOTS - 3:
            period *= 2
        store = set(range((resume // period + 1) * period, snapshot, period))
        at = same // 32 * 32
        if left and at > resume and at >= MIN:
            store.add(at)
        if snapshot > resume:
            store.add(snapshot)
        self.clock += 1
        taken = []
        for r in sorted(store):
            free = [j for j in range(SLOTS) if j != load and j not in taken]
            empty = [j for j in free if self.row[j] == 0]
            slot = empty[0] if empty else min(free, key=lambda j: (self.use[j], j))
            taken.append(slot)
            self.row[slot] = 0
        for slot, r in zip(taken, sorted(store)):
            self.row[slot], self.use[slot] = r, self.clock
        if load >= 0:
            self.use[load] = self.clock
        self.ids, self.cls = ids[:snapshot], cls
        return resume, same


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('corpus', type=Path)
    parser.add_argument('--binary', type=Path, default=ROOT / 'clef')
    parser.add_argument('--attention', choices=('fp32', 'tu'), default='tu')
    args = parser.parse_args()
    env = {k: v for k, v in os.environ.items() if not k.startswith('CLEF_')}
    # Select after sanitizing the environment; an inherited selector is otherwise discarded.
    env['CLEF_ATTN_TU'] = '0' if args.attention == 'fp32' else '1'
    print(f'Attention mode: {args.attention}; CLEF_ATTN_TU={env["CLEF_ATTN_TU"]}', flush=True)

    def run(requests, cache=False, settings=None, rc=0):
        command = [str(args.binary.resolve()), '-m', str(args.model.resolve()), '--logits',
                   '--strict', '--no-truncate', '--time', '--batch', '1']
        if cache:
            command.append('--prefix-cache')
        result = subprocess.run(command, input=payload(requests), text=True, capture_output=True,
                                env={**env, **(settings or {})})
        if result.returncode != rc:
            raise ValueError(f'CLI returned {result.returncode}, expected {rc}: {result.stderr[-2000:]}')
        if len(rows(result.stdout)) != len(requests):
            raise ValueError('incomplete CLI responses')
        stats = [(int(a), int(b)) for a, b in re.findall(r'prefix cache reused (\d+) of (\d+)', result.stderr)]
        if cache and len(stats) != len(requests):
            raise ValueError(f'{len(stats)} cache statistics for {len(requests)} requests: {result.stderr[-1500:]}')
        if rc == 0 and any('error' in r for r in rows(result.stdout)):
            raise ValueError('a request was rejected: ' + result.stdout[:300])
        return result.stdout, stats

    def encode(requests):
        p = subprocess.run([str(ROOT / 'clef-tool'), 'encode-notrunc', str(args.model)], input=payload(requests),
                           text=True, capture_output=True, check=True, env=env)
        out = [json.loads(s) for s in p.stdout.splitlines()]
        if len(out) != len(requests) or any('input_ids' not in o for o in out):
            raise ValueError('fixture rejected by the encoder')
        return [o['input_ids'] for o in out]

    after = {}

    def snapshot_row(request, tokens):
        """The entry's snapshot for this request: 8 tokens before the schema, on a 32-token boundary.
        The tokens after the state are what two requests with one-word states have in common at the end."""
        key = json.dumps(request['questions'])
        if key not in after:
            x, y = encode([dict(request, state='alpha'), dict(request, state='omega')])
            n = 0
            while x[-1 - n] == y[-1 - n]:
                n += 1
            after[key] = n
        start = len(tokens) - after[key]
        return (start - MARGIN) // 32 * 32 if start > MARGIN else 0

    corpus = rows(args.corpus.read_text())
    logs = sorted((r for r in corpus if isinstance(r['state'], str) and len(r['state']) > 2000), key=lambda r: len(r['state']))
    b, c = logs[1], logs[-1]
    other_q = [r['questions'] for r in corpus if r['questions'] != b['questions']][0]
    chunk_from = None if 'flash' in args.model.name else 4096   # the 27B's DeltaNet class boundary

    def check(label, requests):
        """Exact logits against the plain path, and the reused counts the policy gives."""
        expected, _ = run(requests)
        actual, stats = run(requests, cache=True, settings={'CLEF_DEBUG_POISON': '1'})
        count = exact(actual, expected)
        entry, reused, shared, total = Entry(), [], [], []
        for request, tokens, (got, n) in zip(requests, encode(requests), stats, strict=True):
            if n != len(tokens):
                raise ValueError(f'{label}: the CLI processed {n} tokens, the encoder gives {len(tokens)}')
            snapshot = snapshot_row(request, tokens)
            want, same = (0, 0) if snapshot < MIN else entry.request(tokens, snapshot, bool(chunk_from) and len(tokens) >= chunk_from)
            if got != want:
                raise ValueError(f'{label}: request {len(reused)} reused {got} tokens, the policy gives {want}')
            reused.append(got)
            shared.append(same)
            total.append(len(tokens))
        print(f'PASS: {label}: {count} exact logits, reused {reused} of {total}', flush=True)
        return reused, shared, total

    def tail(r, text, questions=None):
        return dict(r, state=r['state'] + '\n' + text, questions=questions or r['questions'])

    def cut(r, share):
        lines = r['state'].split('\n')
        return dict(r, state='\n'.join(lines[:max(1, int(len(lines) * share))]))

    # A fixed preamble with varying tails. The second request resumes from a periodic checkpoint,
    # every later one from the block where the tails begin, and a repeated state from its snapshot.
    # A tail has to be longer than the snapshot's margin and one 32-token block, or it lies past
    # every snapshot and the requests share all the entry holds. These are a few hundred tokens.
    tails = ['\n'.join(f'{tag} {i}: {text}' for i in range(24)) for tag, text in (
        ('ALERT', 'payment gateway timeouts rising in eu-west.'), ('NOTE', 'cache node 7 drained for maintenance.'),
        ('WARN', 'checkout latency back to normal after rollback.'))]
    # the longest state leaves no room for a tail inside the context, so nine tenths of it
    for name, base in (('8k', b), ('15k', cut(c, .9))):
        seq = [tail(base, tails[0]), tail(base, tails[1]), tail(base, tails[2]), tail(base, tails[0]),
               tail(base, tails[0], other_q)]
        reused, shared, total = check(f'{name} preamble with varying tails', seq)
        if reused[0] or not shared[1] - PERIOD < reused[1] <= shared[1]:
            raise ValueError(f'{name}: the second tail did not resume from a periodic checkpoint')
        # within two 32-token blocks of the first differing token, whichever pair of tails it is
        if not all(shared[i] - 64 < reused[i] <= shared[i] for i in (2, 3)):
            raise ValueError(f'{name}: a later tail did not resume where the tails begin')
        if reused[4] <= reused[3]:
            raise ValueError(f'{name}: a repeated state did not resume from its snapshot')

    # An edit inside a state, another edit at the same place, the original again, the state cut
    # short, and a state of the other DeltaNet class that shares its first tokens.
    middle = len(b['state']) // 2
    edits = [dict(b, state=b['state'][:middle] + text + b['state'][middle:]) for text in (' CHANGED EVENT ', ' ANOTHER EVENT ')]
    reused, shared, total = check('edits, shrinking and a class change', [b, edits[0], edits[1], b, cut(b, .6), b, cut(b, .3), b])
    if not shared[1] - PERIOD < reused[1] <= shared[1] or not shared[2] - 64 < reused[2] <= shared[2]:
        raise ValueError('an edit did not resume from the checkpoints before it')
    if not reused[3] or not reused[4] or not reused[5]:
        raise ValueError('the original, a shortened or a regrown state reused nothing')
    if chunk_from and total[6] < chunk_from <= total[5] and (reused[6] or reused[7]):
        raise ValueError('a record of the other DeltaNet class reused the entry')
    if not chunk_from and not (reused[6] and reused[7]):
        raise ValueError('a shortened state and its regrowth reused nothing')

    # More places to leave the entry's tokens than it has slots. Each variant has one extra line,
    # later in the state than the one before, so each shares more with its predecessor than that one
    # did with its own: the blocks where they part all stay valid, fourteen of them beside three
    # periodic rows and the snapshot. Twelve slots hold them, so the least recently used go. The
    # last two requests return to earlier variants.
    lines = b['state'].split('\n')
    step = max(1, len(lines) // 16)
    variants = [dict(b, state='\n'.join(lines[:i * step] + ['VARIANT %d' % i] + lines[i * step:])) for i in range(1, 16)]
    reused, shared, _ = check('fifteen divergence points', variants + [variants[13], variants[7]])
    if any(r > s for r, s in zip(reused, shared)) or not all(reused[15:]):
        raise ValueError('reuse beyond the shared tokens, or none after the slots were recycled')

    # A discarded pass and a failed allocation leave nothing usable behind.
    seq = [tail(b, tails[0]), tail(b, tails[1]), tail(b, tails[2])]
    expected, _ = run(seq, settings={'CLEF_ACT_F16': '0'})
    actual, stats = run(seq, cache=True, settings={'CLEF_DEBUG_F16_LIMIT': '1e-6', 'CLEF_DEBUG_POISON': '1'})
    exact(actual, expected)
    if any(n for n, _ in stats):
        raise ValueError('reused an overflowing discarded pass')
    print('PASS: overflow invalidates every checkpoint and returns the exact BF16 fallback', flush=True)

    # The fill allocates its periodic checkpoints and its snapshot. The second request stores two
    # (the block where the tails begin, and its snapshot) and has one freed slot: one allocation.
    seq = [tail(b, tails[0]), tail(b, tails[1]), tail(b, tails[2]), tail(b, tails[0])]
    expected, _ = run(seq)
    fills = (snapshot_row(seq[0], encode(seq[:1])[0]) - 1) // PERIOD + 1
    actual, stats = run(seq, cache=True, settings={'CLEF_DEBUG_PREFIX_CKPT_FAIL': str(fills + 1)}, rc=1)
    failed = [i for i, r in enumerate(rows(actual)) if 'error' in r]
    if failed != [1]:
        raise ValueError(f'the allocation failure hit requests {failed}, expected the second')
    exact(actual, expected, errors=True)
    if stats[0][0] or stats[2][0] or not stats[3][0]:
        raise ValueError('a failed checkpoint allocation did not invalidate and recover the entry')
    print('PASS: a failed checkpoint allocation is explicit; the next fill and its successor recover exactly', flush=True)


if __name__ == '__main__':
    main()
