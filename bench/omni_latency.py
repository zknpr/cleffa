"""Measure native full-request warm/sparse latency with one resident Omni engine."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import selectors
import signal
import statistics
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',type=Path,default=ROOT/'gguf/clef-omni.gguf')
    ap.add_argument('--golden-root',type=Path,default=ROOT/'golden')
    ap.add_argument('--inventory',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--samples',type=int,default=5)
    ap.add_argument('--sparse-gap',type=float,default=2)
    args=ap.parse_args()
    if args.samples<3 or not 0<args.sparse_gap<=60:ap.error('require >=3 samples and a sparse gap in (0,60]')
    if args.output.exists():ap.error('refusing to replace benchmark evidence')
    inventory=json.loads(args.inventory.read_text());cases=[]
    for name,index in [('text',0),('images',0),('video',2),('audio',5),('audio_video',0)]:
        path=args.golden_root/inventory[name]['golden']
        requests=[json.loads(s) for s in io.StringIO((path/'requests.jsonl').read_text())]
        encoded=[json.loads(s) for s in io.StringIO((path/'encoded.jsonl').read_text())]
        request,layout=requests[index],encoded[index]
        cases.append(dict(modality=name,id=request['id'],tokens=len(layout['input_ids']),
                          state_characters=len(json.dumps(request.get('state'),ensure_ascii=False)),
                          images=len(request.get('images') or []),video_frames=[len(v['frames']) for v in request.get('videos') or []],
                          audio_frames=[a['frames'] for a in layout.get('audio',[])],
                          request_sha256=hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest(),request=request))
    report=dict(complete=False,metric='JSON write through complete native response, milliseconds; model load excluded',
                platform=platform.platform(),macos=platform.mac_ver()[0],samples=args.samples,
                sparse_gap_seconds=args.sparse_gap,mode='default FP32 Omni; batch 1; CLEF_* environment cleared',records=[])
    report['hardware']=subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip()
    report['physical_memory_bytes']=int(subprocess.check_output(['sysctl','-n','hw.memsize'],text=True))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with (ROOT/'clef').open('rb') as source:report['binary_sha256']=hashlib.file_digest(source,'sha256').hexdigest()
    with args.model.open('rb') as source:report['model_sha256']=hashlib.file_digest(source,'sha256').hexdigest()
    report['model']=str(args.model.resolve())
    clean={k:v for k,v in os.environ.items() if not k.startswith('CLEF_')}
    logpath=args.output.with_suffix('.engine.log')
    try:
        with logpath.open('w') as log:
            process=subprocess.Popen(['/usr/bin/time','-l',ROOT/'clef','-m',args.model,'--no-truncate','--batch','1'],
                                     stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,env=clean,start_new_session=True)
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout,selectors.EVENT_READ)
                try:
                    def send(payload):
                        start=time.perf_counter()
                        process.stdin.write(payload);process.stdin.flush()
                        if not selector.select(600):raise RuntimeError('native response timeout')
                        line=process.stdout.readline();elapsed=(time.perf_counter()-start)*1000
                        if not line or 'error' in json.loads(line):raise RuntimeError('missing or failed native response: '+line)
                        return elapsed,line
                    for case in cases:
                        payload=json.dumps(case.pop('request'))+'\n'
                        _,expected=send(payload);send(payload)
                        case.update(request_bytes=len(payload.encode()),response_sha256=hashlib.sha256(expected.encode()).hexdigest())
                        for mode in ('warm','sparse'):
                            values=[]
                            for _ in range(args.samples):
                                if mode=='sparse':time.sleep(args.sparse_gap)
                                elapsed,response=send(payload)
                                if response!=expected:raise ValueError('repeated request changed its response')
                                values.append(elapsed)
                            case[mode]=dict(samples_ms=values,median_ms=statistics.median(values),min_ms=min(values),max_ms=max(values))
                        report['records'].append(case)
                        args.output.write_text(json.dumps(report,indent=2)+'\n')
                        print(f'{case["modality"]}: {case["tokens"]} tokens, warm {case["warm"]["median_ms"]:.2f} ms, sparse {case["sparse"]["median_ms"]:.2f} ms',flush=True)
                finally:
                    try:process.stdin.close()
                    except BrokenPipeError:pass  # The response/exit check reports the engine failure.
                    try:process.wait(timeout=20)
                    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                    process.stdout.close()
            if process.returncode:raise RuntimeError(f'engine exited {process.returncode}; see {logpath}')
        timing=logpath.read_text()
        for key,label in [('peak_rss_bytes','maximum resident set size'),('peak_footprint_bytes','peak memory footprint')]:
            match=re.search(r'^\s*(\d+)\s+'+label+r'\s*$',timing,re.M)
            if not match:raise ValueError('missing native peak-memory measurement: '+label)
            report[key]=int(match[1])
        report['complete']=True
    except (OSError,ValueError,RuntimeError) as exc:
        report['error']=str(exc)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    raise SystemExit(0 if report['complete'] else 1)


if __name__=='__main__':main()
