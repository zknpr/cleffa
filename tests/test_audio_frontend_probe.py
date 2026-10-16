"""Measure the approved byte-exact frontend gate, retaining all first differences.

This standalone diagnostic intentionally exits nonzero for any byte mismatch.
It is not included in make test until the native frontend contract is resolved.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from transformers import WhisperFeatureExtractor

ROOT=Path(__file__).resolve().parents[1]


def compare(actual,expected):
    actual,expected=np.asarray(actual,dtype='<f4'),np.asarray(expected,dtype='<f4')
    if actual.shape != expected.shape: raise ValueError('frontend stage shape differs')
    delta=actual.astype('f8')-expected.astype('f8')
    differs=np.flatnonzero(actual.view('u4').ravel()!=expected.view('u4').ravel())
    result=dict(shape=list(expected.shape),different=len(differs),max_absolute=float(abs(delta).max()),
                relative_l2=float(np.linalg.norm(delta)/max(1e-30,np.linalg.norm(expected.astype('f8')))))
    if len(differs):
        first=int(differs[0])
        result['first']=dict(index=first,native=float(actual.ravel()[first]),reference=float(expected.ravel()[first]))
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',type=Path,default=ROOT/'model-omni')
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    config_path=args.model/'processor_config.json'
    config=json.loads(config_path.read_text())['feature_extractor']
    extractor=WhisperFeatureExtractor(**config)
    window=torch.hann_window(400)
    filters=torch.from_numpy(extractor.mel_filters).float()
    window.numpy().tofile(args.output/'window.f32')
    filters.numpy().tofile(args.output/'filters.f32')
    rng=np.random.default_rng(718)
    cases=[]
    for n in (0,1,199,200,201,399,400,401,15999,16000,16001):
        cases.append((f'random-{n}',[rng.uniform(-.3,.3,n).astype('f4')]))
    t=np.arange(16000,dtype='f8')/16000
    cases += [('silence',[np.zeros(16000,dtype='f4')]),
              ('impulse',[np.eye(1,16000,317,dtype='f4').ravel()]),
              ('sine',[np.sin(2*np.pi*440*t).astype('f4')]),
              ('quiet',[(1e-6*np.sin(2*np.pi*173*t)).astype('f4')]),
              ('loud',[(100*np.sin(2*np.pi*991*t)).astype('f4')]),
              ('unequal',[rng.uniform(-.2,.2,n).astype('f4') for n in (1,199,201,401,16001)])]
    report=dict(torch=torch.__version__,processor_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                method='independent CPU FP64 direct DFT, FP32 storage, Accelerate mel GEMM, libm log10',
                stft_window='reference bytes supplied to isolate FFT differences; native Hann checked separately',cases=[])
    mismatch=False
    for name,clips in cases:
        start=time.monotonic()
        row=dict(name=name,samples=[len(c) for c in clips])
        try:
            reference=extractor(clips,sampling_rate=16000,padding=True,truncation=False,return_attention_mask=True,return_tensors='np')
        except (ValueError,RuntimeError,IndexError) as error:
            row.update(reference_rejected=True,error=str(error))
            report['cases'].append(row)
            print(f'{name}: reference rejected',flush=True)
            continue
        maximum=max(map(len,clips))
        row['clips']=[]
        for i,clip in enumerate(clips):
            folder=args.output/f'{name}-{i}'
            folder.mkdir()
            padded=np.pad(clip,(0,maximum-len(clip)))
            padded.tofile(folder/'samples.f32')
            subprocess.run([str(ROOT/'tests/audio-frontend-probe'),str(folder/'samples.f32'),str(args.output/'window.f32'),
                            str(args.output/'filters.f32'),str(folder),str(maximum)],check=True)
            stft=torch.stft(torch.from_numpy(padded),400,160,window=window,return_complex=True)[...,:-1]
            power=stft.abs()**2
            mel=filters.T@power
            expected=dict(window=window.numpy(),real=stft.real.numpy(),imag=stft.imag.numpy(),power=power.numpy(),
                          mel=mel.numpy(),features=reference['input_features'][i])
            stages={key:compare(np.fromfile(folder/(key+'.bin'),dtype='<f4').reshape(value.shape),value)
                    for key,value in expected.items()}
            for key,value in expected.items(): np.ascontiguousarray(value).tofile(folder/(key+'.reference.bin'))
            valid=int(reference['attention_mask'][i].sum())
            expected_valid=min((len(clip)+159)//160,maximum//160)
            if valid != expected_valid: raise ValueError(f'{name}: invalid assumed frame mask')
            tokens=13*(valid//100)+(valid%100+7)//8
            row['clips'].append(dict(valid_frames=valid,padded_frames=maximum//160,tokens=tokens,stages=stages))
            mismatch |= any(stage['different'] for stage in stages.values())
        row['seconds']=time.monotonic()-start
        report['cases'].append(row)
        print(f'{name}: max feature error {max(c["stages"]["features"]["max_absolute"] for c in row["clips"]):.9g}',flush=True)
    report['byte_exact']=not mismatch
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'byte_exact={not mismatch}; {args.output}/report.json',flush=True)
    raise SystemExit(1 if mismatch else 0)


if __name__=='__main__': main()
