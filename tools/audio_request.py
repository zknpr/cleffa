#!/usr/bin/env python3
"""Convert a bounded local audio file into an Omni request with lossless float32 WAV."""
from __future__ import annotations

import argparse
import base64
import copy
import io
import json
import math
import os
from pathlib import Path
import stat
import struct
import sys
import tempfile
import time

from video_request import MAX_INPUT, run_bounded


def read_local(path):
    fd=os.open(path,os.O_RDONLY|os.O_NONBLOCK)
    with os.fdopen(fd,'rb') as source:
        info=os.fstat(source.fileno())
        if not stat.S_ISREG(info.st_mode) or not 8<=info.st_size<=MAX_INPUT:
            raise ValueError('audio must be a regular local file of 8 bytes to 64 MiB')
        data=source.read(MAX_INPUT+1)
    if not 8<=len(data)<=MAX_INPUT:raise ValueError('audio exceeds input byte limit or is truncated')
    return data


def container_format(data):
    # Force a single-file demuxer. In particular, never autodetect playlists
    # capable of fetching network URLs or opening arbitrary local resources.
    if data[:4]==b'RIFF' and data[8:12]==b'WAVE':return 'wav'
    if data[:4]==b'fLaC':return 'flac'
    if data[:4]==b'OggS':return 'ogg'
    if data[:3]==b'ID3' or (data[0]==255 and data[1]&0xe0==0xe0):return 'mp3'
    if data[4:8] in (b'ftyp',b'moov',b'mdat',b'wide',b'free'):return 'mov'
    raise ValueError('supported audio containers: WAV, FLAC, Ogg, MP3 and MP4/MOV')


def decode_worker(path,max_samples):
    import av
    import numpy as np
    data=read_local(path)
    options={'protocol_whitelist':'file','enable_drefs':'0','use_absolute_path':'0',
             'max_streams':'16','probesize':'1048576','analyzeduration':'1000000'}
    with av.open(io.BytesIO(data),format=container_format(data),options=options) as container:
        streams=[s for s in container.streams if s.type=='audio']
        if len(streams)!=1:raise ValueError('exactly one audio track is required')
        stream=streams[0]
        stream.codec_context.thread_count=1
        resampler=av.AudioResampler(format='flt',layout='mono',rate=16000)
        count=0
        def emit(frame):
            nonlocal count
            for output in resampler.resample(frame):
                samples=output.to_ndarray().reshape(-1)
                count+=len(samples)
                if count>max_samples:raise ValueError('decoded audio exceeds sample limit')
                if not np.isfinite(samples).all():raise ValueError('decoded audio contains nonfinite samples')
                sys.stdout.buffer.write(samples.astype('<f4',copy=False).tobytes())
        for frame in container.decode(stream):emit(frame)
        emit(None)
        if not count:raise ValueError('audio has no decoded samples')


def decode_audio(path,*,max_samples,deadline):
    if type(max_samples)!=int or not 1<=max_samples<=9600000:raise ValueError('audio sample limit must be in [1,9600000]')
    raw=run_bounded([sys.executable,'-B',str(Path(__file__).resolve()),'--decode-worker',str(path),str(max_samples)],
                    max_samples*4,deadline)
    if not raw or len(raw)%4:raise ValueError('invalid float32 audio from decoder')
    return raw


def wav_data_url(raw):
    # IEEE float WAV preserves resampler output exactly; PCM16 would quantize
    # the source and change frontend features before native inference begins.
    header=struct.pack('<4sI4s4sIHHIIHH4sI',b'RIFF',36+len(raw),b'WAVE',b'fmt ',16,3,1,
                       16000,64000,4,32,b'data',len(raw))
    return 'data:audio/wav;base64,'+base64.b64encode(header+raw).decode('ascii')


def convert(path,request,*,max_samples=4800000,max_body=8<<20,timeout=30):
    if not isinstance(request,dict) or not {'model','state','questions'}<=request.keys():
        raise ValueError('request must contain model, state and questions')
    if request.get('audio'):raise ValueError('request already contains audio; convert one clip per request')
    if not 1<=max_body<=1<<30 or not math.isfinite(timeout) or not 0<timeout<=600:
        raise ValueError('invalid output byte limit or timeout')
    data=read_local(path);container_format(data)
    deadline=time.monotonic()+timeout
    with tempfile.TemporaryDirectory(prefix='clef-audio-') as folder:
        snapshot=Path(folder)/'input';snapshot.write_bytes(data)
        raw=decode_audio(snapshot,max_samples=max_samples,deadline=deadline)
    result=copy.deepcopy(request);result['audio']=[wav_data_url(raw)]
    serialized=json.dumps(result,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n'
    if len(serialized.encode())>max_body:raise ValueError('request exceeds --max-body')
    return serialized


def main():
    if len(sys.argv)==4 and sys.argv[1]=='--decode-worker':
        try:decode_worker(Path(sys.argv[2]),int(sys.argv[3]))
        except Exception as exc:sys.exit(f'audio decode: {exc}')
        return
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('clip',type=Path)
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--max-samples',type=int,default=4800000)
    parser.add_argument('--max-body',type=int,default=8<<20)
    parser.add_argument('--timeout',type=float,default=30)
    args=parser.parse_args()
    try:
        with args.request.open('rb') as source:raw=source.read((8<<20)+1)
        if len(raw)>8<<20:raise ValueError('request template exceeds 8 MiB')
        result=convert(args.clip,json.loads(raw),max_samples=args.max_samples,max_body=args.max_body,timeout=args.timeout)
    except (OSError,ValueError,KeyError) as exc:parser.exit(1,f'audio_request: {exc}\n')
    sys.stdout.write(result)


if __name__=='__main__':main()
