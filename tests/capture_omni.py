"""Capture native routes/residuals for numerical checks and routing diagnostics.

Usage: capture_omni.py MODEL.gguf REQUESTS.jsonl OUTPUT_DIR
The output directory is published atomically only after every request succeeds.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('model',type=Path)
    ap.add_argument('requests',type=Path)
    ap.add_argument('output',type=Path)
    args=ap.parse_args()
    if args.output.exists(): raise ValueError('refusing to replace an existing capture directory')
    lines=args.requests.read_text().split('\n')
    requests=[(line,json.loads(line)) for line in lines if line.strip()]
    ids=[r['id'] for _,r in requests]
    if not ids or len(set(ids))!=len(ids) or any(not isinstance(i,str) or not re.fullmatch(r'[A-Za-z0-9_-]+',i) for i in ids):
        raise ValueError('capture requires unique nonempty alphanumeric record IDs')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=args.output.name+'.incomplete-',dir=args.output.parent) as tmp:
        folder=Path(tmp)
        records=[]
        for line,req in requests:
            name=req['id'];start=time.monotonic()
            command=[str(ROOT/'clef'),'-m',str(args.model),'--dump',str(folder/(name+'.bin')),
                     '--dump-routes',str(folder/(name+'.jsonl'))]
            result=subprocess.run(command,input=line+'\n',capture_output=True,text=True,timeout=3600)
            if result.returncode:
                raise RuntimeError(f'{name}: native capture failed: {result.stderr}')
            response=json.loads(result.stdout)
            if 'error' in response: raise RuntimeError(f'{name}: {response["error"]}')
            if not (folder/(name+'.bin')).stat().st_size or not (folder/(name+'.jsonl')).stat().st_size:
                raise ValueError(f'{name}: empty capture')
            (folder/(name+'.output.json')).write_text(result.stdout)
            records.append(dict(id=name,seconds=time.monotonic()-start))
            print(f'{name}: captured paired routes/residuals in {records[-1]["seconds"]:.2f}s',flush=True)
        with (ROOT/'clef').open('rb') as binary:
            binary_hash=hashlib.file_digest(binary,'sha256').hexdigest()
        with args.model.open('rb') as weights:
            model_hash=hashlib.file_digest(weights,'sha256').hexdigest()
        manifest=dict(complete=True,model=str(args.model.resolve()),model_sha256=model_hash,
                      binary_sha256=binary_hash,requests_sha256=hashlib.sha256(args.requests.read_bytes()).hexdigest(),
                      records=records)
        (folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        folder.rename(args.output)
    print(f'published {args.output}',flush=True)


if __name__=='__main__': main()
