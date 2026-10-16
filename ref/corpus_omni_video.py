"""Fixed Omni video inputs already sampled at the pinned encoder's 2 FPS."""
import json

from corpus_video import frames, image, request


def build():
    result = []
    for n,h,w in ((1,96,128),(3,96,128),(6,128,96)):
        result.append(request(dict(frames=[image(f) for f in frames(n,h,w)],fps=2)))
    result.append(request(dict(frames=[image(f) for f in frames(4,96,128)],fps=2),
                          images=[image(frames(1,160,224)[0])]))
    result.append(request(dict(frames=[image(f) for f in frames(2,96,128)],fps=2),
                          videos=[dict(frames=[image(f) for f in frames(2,96,128)],fps=2),
                                  dict(frames=[image(f) for f in frames(3,128,96)],fps=2)]))
    for i,item in enumerate(result):
        item.update(id=f'ot{i:03d}',model='clef-omni')
    return result


if __name__ == '__main__':
    for item in build(): print(json.dumps(item))
