"""Fixed synthetic Omni audio; no external files, speech recordings or network."""
import base64
import json
import struct

import numpy as np

QUESTIONS={
    'sound':dict(type='choice',instructions='Which best describes the attached audio?',
                 criteria={'silence':'No audible signal','tone':'Steady musical tone','noise':'Broadband noise'}),
    'audible':dict(type='noul',instructions='Does the audio contain an audible signal?'),
}


def samples(count,kind='tone',seed=718):
    if kind=='silence': return np.zeros(count,dtype='<f4')
    if kind=='noise': return np.random.default_rng(seed).uniform(-.2,.2,count).astype('<f4')
    t=np.arange(count,dtype='f8')/16000
    return (.3*np.sin(2*np.pi*440*t)+.08*np.sin(2*np.pi*991*t)).astype('<f4')


def wav(values,pcm16=False):
    if pcm16:
        payload=np.rint(np.clip(values,-1,32767/32768)*32768).astype('<i2').tobytes();code,width=1,2
    else: payload=np.asarray(values,dtype='<f4').tobytes();code,width=3,4
    fmt=struct.pack('<HHIIHH',code,1,16000,16000*width,width,8*width)
    body=b'WAVEfmt '+struct.pack('<I',16)+fmt+b'data'+struct.pack('<I',len(payload))+payload
    return 'data:audio/wav;base64,'+base64.b64encode(b'RIFF'+struct.pack('<I',len(body))+body).decode()


def build():
    from corpus_omni_vision import build as images
    clips=[
        [wav(samples(201,'silence'),True)],
        [wav(samples(16000),True)],
        [wav(samples(16001,'noise'))],
        [wav(samples(399)),wav(samples(16160,'noise'))],
        [wav(samples(16000))],
        [wav(samples(496000,'noise'))],
    ]
    result=[dict(id=f'oa{i:03d}',model='clef-omni',state='Review the attached media.',audio=audio,questions=QUESTIONS)
            for i,audio in enumerate(clips)]
    result[4]['images']=images()[0]['images']
    return result


if __name__=='__main__':
    for item in build():print(json.dumps(item))
