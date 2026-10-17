"""Bound MPS graph-cache lifetime with lossless decoder subprocess stages.

The parent verifies the pinned snapshot once. Every worker checks its complete
file identity inventory before imports and after execution. Checkpoints are
private, hash-bound, weights-only tensor files; an incomplete run publishes no
golden. This is a numerical verifier, not a PyTorch latency baseline.
"""
import hashlib
import json
import os
from pathlib import Path
import stat
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace


def digest(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def policy_digest(arguments, sources):
    # Bind the arithmetic and all verifier sources across checkpoint boundaries.
    # A later stage cannot relabel states produced by a different attention path.
    return hashlib.sha256(json.dumps(dict(arguments=arguments, sources=sources),
                                     sort_keys=True).encode()).hexdigest()


def inventory(path):
    result = {}
    for directory, dirs, files in os.walk(path, followlinks=False):
        parent = Path(directory)
        for name in dirs[:]:
            child = parent / name
            if child.relative_to(path).as_posix() == '.cache/huggingface':
                dirs.remove(name)
            elif child.is_symlink():
                raise ValueError('snapshot directory became a symbolic link')
        for name in files:
            child = parent / name
            info = child.lstat()
            if not stat.S_ISREG(info.st_mode):
                raise ValueError('snapshot input is not a regular file')
            result[child.relative_to(path).as_posix()] = [info.st_dev, info.st_ino, info.st_mode,
                                                       info.st_size, info.st_mtime_ns, info.st_ctime_ns]
    return result


def check_snapshot(path, expected):
    if inventory(path) != expected:
        raise ValueError('verified snapshot changed during staged execution')


def move(value, device):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().to(device).contiguous()
    if isinstance(value, dict):
        return {k: move(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [move(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(move(v, device) for v in value)
    return value


def load_state(stage, device):
    import torch
    if not stage['input']:
        return {}
    if digest(stage['input']) != stage['input_sha256']:
        raise ValueError('decoder checkpoint hash mismatch')
    saved = torch.load(stage['input'], map_location='cpu', weights_only=True)
    if not stage.get('policy_sha256') or saved.get('policy_sha256') != stage['policy_sha256']:
        raise ValueError('checkpoint reference policy differs from this run')
    state = saved['state']
    if state['next_layer'] != stage['first']:
        raise ValueError('decoder checkpoint belongs to another stage')
    for key in ('hidden', 'extras'):
        state[key] = move(state[key], device)
    return state


def save_state(stage, state):
    import torch
    output = Path(stage['output'])
    if output.exists():
        raise ValueError('refusing to replace a decoder checkpoint')
    with output.open('xb') as target:
        torch.save(dict(state=move(state, 'cpu'), policy_sha256=stage['policy_sha256']), target)


def run(args):
    root = Path(__file__).resolve().parents[1]
    if Path(args.name).name != args.name or args.name in ('.', '..'):
        raise ValueError('--name must be a single golden directory name')
    if (root / 'golden' / args.name).exists():
        raise FileExistsError('refusing to replace existing golden directory')
    sys.path.insert(0, str(root / 'tools'))
    from verify_snapshot import load_manifest, verify_snapshot
    from omni import REVISION
    before = inventory(args.model_dir)
    bad = verify_snapshot(args.model_dir, load_manifest(REVISION))
    if bad:
        raise ValueError('snapshot verification failed: ' + '; '.join(bad))
    check_snapshot(args.model_dir, before)
    config = json.loads((args.model_dir / 'config.json').read_text())
    layers = config['thinker_config']['text_config']['num_hidden_layers']
    if not 1 <= args.stage_layers <= layers:
        raise ValueError('invalid number of layers per subprocess')
    # Bind all local verifier code as well; a mid-run edit invalidates evidence.
    sources = {str(path): digest(path) for path in (root / 'ref').glob('*.py')}
    arguments = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    policy = policy_digest(arguments, sources)
    (root / 'golden').mkdir(parents=True, exist_ok=True)
    # Retain failed stages for diagnosis/recovery instead of discarding hours of
    # verified decoder work. Workers still verify job/state hashes on every use.
    with tempfile.TemporaryDirectory(prefix='omni-stages-', dir=root / 'golden', delete=False) as tmp:
        folder = Path(tmp)
        print(f'Omni checkpoint directory: {folder}', flush=True)
        previous = None
        for first in range(0, layers, args.stage_layers):
            stop = min(layers, first + args.stage_layers)
            output = folder / f'state-{stop:02d}.pt'
            stage = dict(first=first, stop=stop, input=str(previous) if previous else None,
                         input_sha256=digest(previous) if previous else None, output=str(output),
                         snapshot_inventory=before, sources=sources, policy_sha256=policy)
            job = folder / f'job-{stop:02d}.json'
            job.write_text(json.dumps(dict(arguments=arguments, stage=stage)))
            print(f'Omni subprocess layers {first}..{stop-1}', flush=True)
            subprocess.run([sys.executable, '-B', __file__, str(job), digest(job)], check=True)
            if stop < layers and not output.is_file():
                raise ValueError('worker did not save its decoder state')
            if previous:
                previous.unlink()
            previous = output
    shutil.rmtree(folder)


def worker(job_path, expected_hash):
    if digest(job_path) != expected_hash:
        raise ValueError('staged worker configuration changed')
    job = json.loads(Path(job_path).read_text())
    if job['stage'].get('policy_sha256') != policy_digest(job['arguments'], job['stage']['sources']):
        raise ValueError('staged reference policy changed')
    for path, expected in job['stage']['sources'].items():
        if digest(path) != expected:
            raise ValueError('reference source changed during staged execution')
    values = job['arguments']
    values['model_dir'] = Path(values['model_dir'])
    args = SimpleNamespace(**values, _stage=job['stage'])
    import omni
    omni.run(args, streamed=True)


if __name__ == '__main__':
    worker(*sys.argv[1:])
