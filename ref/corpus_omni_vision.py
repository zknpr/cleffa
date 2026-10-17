"""Fixed Omni still-image corpus; independent of the legacy golden corpus."""
import json

from corpus_vision import _gradient, _jpeg, _png, _scene, SCENE_QUESTIONS


def build():
    media = [
        [_png(_gradient(16, 16, 301),data_url=True)],
        [_png(_gradient(257, 385, 302),data_url=True)],
        ['data:image/jpeg;base64,'+_jpeg(_scene(192, 320, 303))],
        [_png(_gradient(288, 192, 304),data_url=True), _png(_scene(192, 352, 305),data_url=True)],
        [],
    ]
    return [dict(id=f'ov{i:03d}', model='clef-omni', state='Review the attached images.',
                 images=images, questions=SCENE_QUESTIONS) for i,images in enumerate(media)]


if __name__ == '__main__':
    for request in build(): print(json.dumps(request))
