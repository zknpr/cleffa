"""Video host byte parity against the pinned processor, including original timestamps after adapter sampling.
Usage: .venv/bin/python -B tests/test_video.py [MODEL.gguf MODEL_DIR]
"""
import copy
import json
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

ROOT = Path(__file__).resolve().parents[1]
model = sys.argv[1] if len(sys.argv) > 1 else 'gguf/clef-flash.gguf'
model_dir = sys.argv[2] if len(sys.argv) > 2 else 'model-flash'
sys.path.insert(0, str(ROOT / model_dir))
sys.path.insert(0, str(ROOT / 'ref'))
from joint_schema_model import encode_record
from oracle import with_pil_images
import corpus_video
import golden_io


def native(req, mode='encode', path=None):
    cmd = [os.environ.get('CLEF_TOOL', str(ROOT / 'clef-tool')), mode, model]
    if path is not None:
        cmd.append(str(path))
    serialized = req if isinstance(req, str) else json.dumps(req)
    return subprocess.run(cmd, input=serialized + '\n', text=True, capture_output=True, check=True).stdout.strip()


def main():
    proc = AutoProcessor.from_pretrained(model_dir)
    stub = types.SimpleNamespace(config=types.SimpleNamespace(vision_config=types.SimpleNamespace(spatial_merge_size=2)))
    stub.get_vision_position_ids = lambda *a, **k: Qwen3_5Model.get_vision_position_ids(stub, *a, **k)
    cases = corpus_video.build()
    rgb = corpus_video.frames(11, 67, 99)
    for count in [1, 2, 3, 5, 8, 11]:
        cases.append(corpus_video.request({'frames': [corpus_video.image(f) for f in rgb], 'fps': 7.5},
                      media_kwargs={'videos_kwargs': {'num_frames': count, 'size': {'shortest_edge': 4096, 'longest_edge': 12000}}}))
    with tempfile.TemporaryDirectory(prefix='clef-video-test-') as td:
        for i, req in enumerate(cases):
            req['id'] = f't{i}'
            enc = encode_record(proc.tokenizer, with_pil_images(req), processor=proc)
            ids = torch.tensor([enc.input_ids])
            kinds = torch.zeros_like(ids)
            offset = enc.media['token_offset']
            kinds[0, offset:offset + len(enc.media['mm_token_type_ids'])] = torch.tensor(enc.media['mm_token_type_ids'])
            pos, _ = Qwen3_5Model.get_rope_index(stub, ids, kinds, image_grid_thw=enc.media.get('image_grid_thw'),
                                               video_grid_thw=enc.media.get('video_grid_thw'))
            want = json.loads(golden_io.encoded_line(req, enc, pos[:, 0].tolist()))
            want.pop('id')
            path = Path(td) / 'patches'
            raw = native(req, 'encode-patches', path)
            assert not raw.startswith('ERR'), (i, raw)
            got = json.loads(raw)
            assert got == want, (i, [k for k in want if got.get(k) != want[k]])
            expected = torch.cat([enc.media[k] for k in ('pixel_values', 'pixel_values_videos') if k in enc.media]).numpy()
            actual = np.fromfile(path, np.float32).reshape(expected.shape)
            assert np.array_equal(actual, expected), (i, 'patches', np.max(np.abs(actual - expected)))
            assert json.loads(native(req, 'encode-strict')) == got
    bad = []
    base = cases[0]
    for value in [0, -1, float('nan'), float('inf'), 0.00001, '24', True]:
        req = copy.deepcopy(base); req['videos'][0]['fps'] = value; bad.append(req)
    for video in [[], 'https://example.com/video.mp4', 'file:///etc/passwd', {'frames': [], 'fps': 24},
                  {'content_type': 'video/mp4', 'base64': 'AAAA'}, {'content_type': 'image/png', 'base64': 'AAAA'}, 'data:video/mp4;base64,AAAA']:
        bad.append(corpus_video.request(video))
    req = copy.deepcopy(base); req['videos'][0]['frames'][-1] = corpus_video.image(corpus_video.frames(1, 64, 64)[0]); bad.append(req)
    sampled = cases[5]
    for indices in [[0, 3], [0, 3, 3, 8], [-1, 3, 5, 8], [0, 3, 5, 9], [0, 3, 5, 1.5], [False, 3, 5, 8], [0, 3, 5, 2**64]]:
        req = copy.deepcopy(sampled); req['videos'][0]['frame_indices'] = indices; bad.append(req)
    for total in [0, 8, 18001, 2**64, True]:
        req = copy.deepcopy(sampled); req['videos'][0]['total_num_frames'] = total; bad.append(req)
    req = copy.deepcopy(sampled); req.pop('media_kwargs'); bad.append(req)
    req = copy.deepcopy(sampled); req['videos'][0].pop('total_num_frames'); bad.append(req)
    req = copy.deepcopy(base); req['videos'][0]['total_num_frames'] = 9; bad.append(req)
    for kw in [{'num_frames': 1000}, {'num_frames': 0}, {'fps': 0}, {'fps': 2, 'num_frames': 4},
               {'do_sample_frames': False, 'fps': 2}, {'size': {'shortest_edge': 2**31, 'longest_edge': 2**31}},
               {'size': {'shortest_edge': 2**30, 'longest_edge': 2**30}}, {'patch_size': 8}]:
        req = copy.deepcopy(base); req['media_kwargs'] = {'videos_kwargs': kw}; bad.append(req)
    req = copy.deepcopy(base); req['state'] = '<|video_pad|>'; bad.append(req)
    for i, req in enumerate(bad):
        assert native(req).startswith('ERR'), ('accepted invalid input', i)
    # A size mismatch in either frame of a pair must name the dimensions, not fall through to
    # the pair preprocessor's generic geometry error. base samples frames [0, 3, 5, 8]:
    # frame 5 opens the second pair and frame 8 closes it.
    odd = corpus_video.image(corpus_video.frames(1, 64, 64)[0])
    for position in [5, 8]:
        req = copy.deepcopy(base); req['videos'][0]['frames'][position] = odd
        assert 'video frame dimensions differ' in native(req), ('pair frame size message', position)
    assert not native(bad[-1], 'encode-strict').startswith('ERR')
    # Uniform adapter sampling must preserve every token and temporal patch of core sampling.
    assert native(cases[0]) == native(cases[5])
    assert native(json.dumps(cases[5]).replace('"frame_indices": [0', '"frame_indices": [-0')) == native(cases[5])
    # The schema and any untruncated state must count before video patches are allocated.
    # This pair would occupy 14,400 tokens, leaving too little room for either 2,500-token
    # field. A deliberately invalid second frame proves the budget check happens first.
    tiny = corpus_video.image(corpus_video.frames(1, 32, 32)[0])
    budget = corpus_video.request({'frames': [tiny, 'AAAA'], 'fps': 2},
        media_kwargs={'videos_kwargs': {'do_sample_frames': False,
                      'size': {'shortest_edge': 2 * 14400 * 1024, 'longest_edge': 2 * 14400 * 1024}}})
    for mode in ['encode', 'encode-notrunc']:
        req = copy.deepcopy(budget)
        if mode == 'encode':
            next(iter(req['questions'].values()))['instructions'] = 'alpha ' * 2500
        else:
            req['state'] = 'alpha ' * 2500
        assert 'video tokens and prompt cannot fit the context' in native(req, mode), mode
    # With truncation allowed, the state yields to the video. The encoder must reach the
    # second frame rather than incorrectly charging the full state against its budget.
    assert 'image: not a PNG or JPEG' in native(req, 'encode')
    print(f'video: {len(cases)} exact token/span/position/patch cases; {len(bad)} rejections; '
          'schema/state budgets and strict placeholders PASS')


if __name__ == '__main__':
    main()
