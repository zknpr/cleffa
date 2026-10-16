"""Omni media grouping, record packing and poisoned scratch must preserve logits."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ref'))
import corpus_omni_vision,corpus_omni_video,corpus_audio,corpus_omni_av
from corpus_video import frames,image


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model',type=Path,nargs='?',default=ROOT/'gguf/clef-omni-full.gguf')
    parser.add_argument('--bf16',action='store_true',help='check invariance within the legacy 16-bit vision diagnostic mode')
    args=parser.parse_args();model=args.model
    cases=corpus_omni_vision.build()+corpus_omni_video.build()+corpus_audio.build()+corpus_omni_av.build()
    # Unequal grids and enough frame patches to cross the grouped GEMM cap.
    for count,h,w in ((9,64,96),(33,128,128),(5,192,288)):
        request=dict(cases[5],id=f'groups-{count}',videos=[dict(frames=[image(f) for f in frames(count,h,w)],fps=2)])
        request['media_kwargs']={'videos_kwargs':dict(fps=2,do_sample_frames=False)}
        cases.append(request)
    payload=''.join(json.dumps(r)+'\n' for r in cases)
    clean={k:v for k,v in os.environ.items() if not k.startswith('CLEF_')}
    expected=None
    modes=[{'CLEF_VIS_F32':'0'}] if args.bf16 else [{},{'CLEF_VIS_COMP':'0'},{'CLEF_ACT_F16':'0'}]
    for mode in modes:
        for grouped,batch in ((False,1),(True,1),(True,8)):
            env={**clean,**mode,'CLEF_VIS_GROUP':str(int(grouped)),'CLEF_DEBUG_POISON':'1'}
            result=subprocess.run([ROOT/'clef','-m',model,'--logits','--no-truncate','--max-video-frames','64',
                                   '--batch',str(batch)],input=payload,text=True,capture_output=True,env=env,timeout=900)
            if result.returncode:raise RuntimeError(result.stderr+'\n'+result.stdout)
            rows=result.stdout.splitlines()
            if len(rows)!=len(cases) or any('error' in json.loads(row) for row in rows):raise ValueError('missing/failed media response')
            if expected is None:expected=rows
            if rows!=expected:raise ValueError(f'logits depend on grouping/packing: {mode}, grouped={grouped}, batch={batch}')
        print(f'{mode or "default"}: {len(cases)} media records, grouped/separate, batch1/8, poison: exact logits',flush=True)


if __name__=='__main__':main()
