"""Fixed native frame/soundtrack associations with unequal durations and ties."""
import json

from corpus_audio import samples,wav,QUESTIONS
from corpus_video import frames,image


def video(n,count,seed=718):
    return dict(frames=[image(f) for f in frames(n,96,128)],fps=2,audio=wav(samples(count,'noise',seed)))


def build():
    from corpus_omni_vision import build as images
    result=[dict(videos=[video(6,16000)]),dict(videos=[video(2,80000)]),
            dict(videos=[video(4,32000)]),
            dict(audio=[wav(samples(399))],videos=[video(2,16001),video(3,32001,719)]),
            dict(images=images()[0]['images'],videos=[video(3,16001)])]
    for i,item in enumerate(result):
        item.update(id=f'oav{i:03d}',model='clef-omni',state='Review the video and its soundtrack.',questions=QUESTIONS)
    return result


if __name__=='__main__':
    for item in build():print(json.dumps(item))
