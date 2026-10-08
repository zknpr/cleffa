"""Deterministic temporal inputs, separate from the existing image/text golden corpora."""
from __future__ import annotations

import base64
import io
import json

import numpy as np
from PIL import Image


def frames(n=9, h=96, w=128):
    result = []
    for i in range(n):
        y, x = np.indices((h, w))
        rgb = np.stack([(x * 2 + i * 17) % 256, (y * 3 + i * 7) % 256, (x + y + i * 13) % 256], -1).astype(np.uint8)
        rgb[h // 4:3 * h // 4, i * (w - 24) // max(n - 1, 1):i * (w - 24) // max(n - 1, 1) + 24] = (240, 20, 30)
        result.append(rgb)
    return np.stack(result)


def image(rgb):
    f = io.BytesIO()
    Image.fromarray(rgb).save(f, 'PNG')
    return 'data:image/png;base64,' + base64.b64encode(f.getvalue()).decode()


def request(video, **kwargs):
    return {'model': 'clef', 'state': 'Review the video in time order.', 'videos': [video],
            'questions': {'moving': {'type': 'noul', 'instructions': 'Does the red object move?'},
                          'direction': {'type': 'choice', 'instructions': 'Which direction does the red object move?',
                                        'criteria': {'left': 'Left', 'right': 'Right', 'still': 'It stays still'}}}, **kwargs}


def build():
    rgb = frames()
    base = {'frames': [image(f) for f in rgb], 'fps': 6}
    result = [request(base),
              request(base, media_kwargs={'videos_kwargs': {'do_sample_frames': False}}),
              request({'frames': [image(f) for f in rgb[::-1]], 'fps': 6}),
              request({'frames': [image(rgb[0])], 'fps': 2}),
              request(base, images=[image(rgb[-1])], media_kwargs={'min_pixels': 4096, 'max_pixels': 16384}),
              request({'frames': [image(rgb[i]) for i in [0, 3, 5, 8]], 'fps': 6,
                       'frame_indices': [0, 3, 5, 8], 'total_num_frames': 9},
                      media_kwargs={'videos_kwargs': {'do_sample_frames': False}}),
              request({'frames': [image(rgb[i]) for i in [0, 4, 8]], 'fps': 29.97,
                       'frame_indices': [7, 109, 301], 'total_num_frames': 450},
                      media_kwargs={'videos_kwargs': {'do_sample_frames': False}}),
              request(base, videos=[base, {'frames': [image(f) for f in frames(5, 64, 96)], 'fps': 3}])]
    for i, req in enumerate(result):
        req['id'] = f'vd{i:03d}'
    return result


if __name__ == '__main__':
    for r in build():
        print(json.dumps(r))
