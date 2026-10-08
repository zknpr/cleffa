"""Video HTTP limits, strict tokens, mixed-media batching and exact keyed-cache transitions.
Usage: .venv/bin/python -B tests/test_server_video.py MODEL.gguf [VIDEO_REQUESTS.jsonl]
"""
import copy
import base64
import http.client
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ref'))
import corpus_video
from test_jpeg_regressions import progressive, spectral_work_bomb


def main():
    model = sys.argv[1]
    corpus = ([json.loads(x) for x in Path(sys.argv[2]).read_text().splitlines()]
              if len(sys.argv) > 2 else corpus_video.build())
    env = dict(os.environ, CLEF_DEBUG_POISON='1')
    def cli(requests, *flags):
        p = subprocess.run([str(ROOT / 'clef'), '-m', model, '--strict', *flags],
                           input=''.join(json.dumps(r) + '\n' for r in requests),
                           text=True, capture_output=True, check=True, env=env)
        result = [json.loads(s) for s in p.stdout.splitlines()]
        assert len(result) == len(requests) and all('error' not in s for s in result), result
        return result

    # The timestamp changes while patch bytes stay identical; the other edits change one tap,
    # frame order, geometry, media kind or the question after the cached visual prefix.
    base = corpus[1]
    sequence = [base, base]
    changed = copy.deepcopy(base); changed['videos'][0]['fps'] = 3; sequence += [changed, changed]
    changed = copy.deepcopy(base); changed['videos'][0]['frames'][1] = changed['videos'][0]['frames'][-1]; sequence += [changed, changed]
    changed = copy.deepcopy(base); changed['videos'][0]['frames'].reverse(); sequence += [changed, changed]
    changed = copy.deepcopy(base); changed['state'] = 'Watch the red object carefully.'; sequence += [changed, changed]
    sequence += [corpus[4], corpus[4], corpus[5], corpus[5], corpus[7], corpus[7]]
    expected = cli(sequence, '--logits')
    assert cli(sequence, '--logits', '--prefix-cache') == expected
    assert cli(sequence, '--logits', '--template-cache') == expected
    print(f'video caches: {len(sequence)} poisoned transitions, exact raw logits PASS', flush=True)
    expected = cli(corpus)
    with tempfile.TemporaryDirectory(prefix='clef-http-video-') as td:
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
        with (Path(td) / 'server.log').open('w') as log:
            server = subprocess.Popen([str(ROOT / 'clef-server'), '-m', model, '--port', str(port),
                                       '--no-warmup', '--no-keep-warm', '--max-videos', '2', '--max-image-requests', '1',
                                       '--prefix-cache-mb', '1024'], stdout=log, stderr=log, env=env)
            def post(req, key=None):
                c = http.client.HTTPConnection('127.0.0.1', port, timeout=120)
                try:
                    headers = {'Content-Type': 'application/json'}
                    if key: headers['X-Clef-Prefix-Cache'] = key
                    c.request('POST', '/v1/systemone', json.dumps(req), headers)
                    r = c.getresponse()
                    return r.status, json.loads(r.read())
                finally:
                    c.close()
            try:
                deadline = time.monotonic() + 120
                while True:
                    try:
                        c = http.client.HTTPConnection('127.0.0.1', port, timeout=1)
                        c.request('GET', '/health'); ready = c.getresponse().status == 200; c.close()
                        if ready: break
                    except OSError:
                        pass
                    assert server.poll() is None and time.monotonic() < deadline, 'server startup failed'
                    time.sleep(0.5)
                for key in [None, 'video-a', 'video-a', 'video-b']:
                    with ThreadPoolExecutor(max_workers=8) as pool:
                        actual = list(pool.map(lambda r: post(r, key), corpus))
                    assert actual == [(200, r) for r in expected], actual
                bad = []
                bad.append((dict(corpus[0], videos=corpus[0]['videos'] * 3), 'too many videos'))
                bad.append((corpus_video.request({'frames': corpus[0]['videos'][0]['frames'] * 4, 'fps': 2},
                             media_kwargs={'videos_kwargs': {'do_sample_frames': False}}), 'sampled frames'))
                bad.append((dict(corpus[0], media_kwargs={'videos_kwargs': {'size': {'shortest_edge': 4194304, 'longest_edge': 4194304}}}), 'tokens'))
                bad.append((corpus_video.request({'content_type': 'video/mp4', 'base64': 'AAAA'}), 'video'))
                # The source fits the default pixel/body limits. Still images and video frames
                # must reach the shared scan guard and release the same media admission slot.
                for data in [progressive([(0, 0, 1, 0)] * 1000, 4096, 4096, False), spectral_work_bomb()]:
                    bomb = base64.b64encode(data).decode()
                    for image in [bomb, 'data:image/jpeg;base64,' + bomb, {'content_type': 'image/jpeg', 'base64': bomb}]:
                        bad.append((dict(corpus[0], videos=[], images=[image]), 'invalid or unsupported JPEG'))
                        bad.append((corpus_video.request({'frames': [image], 'fps': 2}), 'invalid or unsupported JPEG'))
                for req, message in bad:
                    status, body = post(req)
                    assert status == 400 and message in body.get('error', ''), (status, body)
                # Admission must be released on rejection, including a failure during decode.
                assert post(corpus[5]) == (200, expected[5])
                injected = dict(corpus[0], state='<|video_pad|><|vision_start|>')
                assert post(injected) == (200, cli([injected])[0])
            finally:
                server.terminate()
                try: server.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    server.kill(); server.wait()
    print('video HTTP: concurrent CLI parity, key isolation, count/frame/token limits, strict tokens and failure recovery PASS')


if __name__ == '__main__':
    main()
