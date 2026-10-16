"""Run the unchanged FP64 oracle with bounded, versioned capture consolidation.

Use the same CLI arguments as oracle_fp64.py. Existing v1 references retain
their exact source contract; v2 adds this adapter and the safetensors version.
"""
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile

import safetensors
from safetensors import safe_open

import omni_fp64_policy as v1
import oracle_fp64 as executor

REVISION = v1.REVISION
digest = v1.digest
POLICY = {**v1.POLICY, 'name':'independent-cpu-fp64-v2',
          'capture':'bounded-safetensors-payload-copy-v1'}
_CHUNK = 1 << 20
_PART_HEADER_LIMIT = 64 << 10
_FINAL_HEADER_LIMIT = 8 << 20
_WIDTH = {'F64':8, 'I64':8, 'F32':4}


def implementation_sources():
    sources = v1.implementation_sources()
    path = Path(__file__).resolve()
    sources[str(path.relative_to(path.parents[1]))] = v1.digest(path)
    return sources


def runtime_provenance(model_dir, corpus):
    return {**v1.runtime_provenance(model_dir, corpus), 'safetensors':safetensors.__version__}


def validate_policy(manifest):
    if manifest.get('fp64_policy') == v1.POLICY:
        return v1.validate_policy(manifest)
    if (manifest.get('fp64_policy') != POLICY or
            manifest.get('implementation_sha256') != implementation_sources() or
            manifest.get('safetensors') != safetensors.__version__):
        raise ValueError('FP64 bounded capture requires its exact policy, sources and runtime provenance')
    # The v2-only contract is checked above. Reuse every original arithmetic,
    # independence and runtime check without weakening v1 or rewriting evidence.
    return v1.validate_policy({**manifest, 'fp64_policy':v1.POLICY,
                               'implementation_sha256':v1.implementation_sources()})


verify_artifacts = v1.verify_artifacts


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate safetensors header key')
        result[key] = value
    return result


def _identity(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def _part(path, name):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError('capture part must be a regular file')
    with path.open('rb') as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError('truncated capture header length')
        size = int.from_bytes(prefix, 'little')
        if not 0 < size <= _PART_HEADER_LIMIT:
            raise ValueError('capture header exceeds its bound')
        raw = stream.read(size)
        if len(raw) != size:
            raise ValueError('truncated capture header')
        header = json.loads(raw, object_pairs_hook=_unique_object)
        if not isinstance(header, dict) or set(header) != {name}:
            raise ValueError('capture part tensor inventory differs')
        meta = header[name]
        if not isinstance(meta, dict) or set(meta) != {'dtype', 'shape', 'data_offsets'}:
            raise ValueError('invalid capture tensor metadata')
        shape = meta['shape']; dtype = meta['dtype']
        if (not isinstance(dtype, str) or dtype not in _WIDTH or not isinstance(shape, list) or
                any(type(n) is not int or n < 0 for n in shape)):
            raise ValueError('invalid capture dtype or shape')
        count = math.prod(shape)*_WIDTH[dtype]
        if (not count or meta['data_offsets'] != [0, count] or info.st_size != 8+size+count or
                _identity(os.fstat(stream.fileno())) != _identity(info)):
            raise ValueError('capture payload size or identity differs')
    # Validate the bounded header with the canonical reader without loading data.
    with safe_open(path, framework='np') as tensors:
        value = tensors.get_slice(name)
        if tensors.keys() != [name] or value.get_shape() != shape or value.get_dtype() != dtype:
            raise ValueError('capture reader metadata differs')
    return dict(path=path, name=name, dtype=dtype, shape=shape, offset=8+size,
                count=count, identity=_identity(info))


class Capture(executor.Capture):
    def finish(self, destination):
        destination = Path(destination)
        if destination.exists():
            raise FileExistsError(destination)
        if (not self.names or len(set(self.names)) != len(self.names) or
                any(not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_.]+', name) or
                    name == '__metadata__' for name in self.names)):
            raise ValueError('invalid capture tensor inventory')
        expected = {name+'.safetensors' for name in self.names}
        if {path.name for path in self.path.iterdir()} != expected:
            raise ValueError('capture part file inventory differs')
        parts = [_part(self.path/(name+'.safetensors'), name) for name in self.names]
        # Keep wider values aligned, then use a deterministic tensor order.
        parts.sort(key=lambda part:(-_WIDTH[part['dtype']], part['name']))
        header = {}; end = 0
        for part in parts:
            header[part['name']] = dict(dtype=part['dtype'], shape=part['shape'],
                                       data_offsets=[end, end+part['count']])
            end += part['count']
        # Format: https://github.com/safetensors/safetensors#format
        raw = json.dumps(header, separators=(',', ':')).encode('utf-8')
        raw += b' '*(-len(raw) % 8)
        if len(raw) > _FINAL_HEADER_LIMIT:
            raise ValueError('consolidated capture header exceeds its bound')
        fd, temporary_name = tempfile.mkstemp(prefix='.'+destination.name+'.', suffix='.tmp', dir=destination.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, 'wb') as output:
                output.write(len(raw).to_bytes(8, 'little')); output.write(raw)
                for part in parts:
                    with part['path'].open('rb') as source:
                        if _identity(os.fstat(source.fileno())) != part['identity']:
                            raise ValueError('capture part changed before consolidation')
                        source.seek(part['offset']); left = part['count']
                        while left:
                            block = source.read(min(_CHUNK, left))
                            if not block:
                                raise ValueError('truncated capture payload')
                            output.write(block); left -= len(block)
                        if source.read(1) or _identity(os.fstat(source.fileno())) != part['identity']:
                            raise ValueError('capture part changed during consolidation')
                output.flush(); os.fsync(output.fileno())
            with safe_open(temporary, framework='np') as tensors:
                if set(tensors.keys()) != set(self.names):
                    raise ValueError('consolidated capture inventory differs')
            # Publish only the complete file. A racing destination cannot be
            # overwritten; before publication, every source part is retained.
            os.link(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        # Cleanup failures propagate with the complete destination still present.
        for part in parts:
            part['path'].unlink()
        self.path.rmdir()


@contextmanager
def _configured_executor():
    bindings = dict(Capture=Capture, POLICY=POLICY, implementation_sources=implementation_sources,
                    runtime_provenance=runtime_provenance)
    previous = {name:getattr(executor, name) for name in bindings}
    try:
        for name, value in bindings.items():
            setattr(executor, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(executor, name, value)


def main():
    # This CLI has one executor. Restore its globals for import/test callers,
    # including failures; all inference and snapshot verification stay in v1.
    with _configured_executor():
        executor.main()


if __name__ == '__main__':
    main()
