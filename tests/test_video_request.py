"""Optional MP4/MOV adapter tests. Requires ffmpeg/ffprobe in PATH; excluded from make test."""
import json
import base64
import io
import socket
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT / 'ref'))
from video_request import convert, run_bounded, png_frames
import corpus_video


def external_reference(data, url):
    boxes, at = [], 0
    while at < len(data):
        size, kind = struct.unpack_from('>I4s', data, at)
        assert size >= 8 and at + size <= len(data)
        body = data[at + 8:at + size]
        if kind in (b'moov', b'trak', b'mdia', b'minf', b'dinf'):
            body = external_reference(body, url)
        elif kind == b'dref':
            ref = b'\0\0\0\0' + url.encode() + b'\0'
            body = body[:8] + struct.pack('>I4s', len(ref) + 8, b'url ') + ref
        boxes.append(struct.pack('>I4s', len(body) + 8, kind) + body)
        at += size
    return b''.join(boxes)


def refuse(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except (ValueError, OSError):
        return
    raise AssertionError('accepted invalid input')


def main():
    template = corpus_video.request(None)
    template.pop('videos')
    rgb = corpus_video.frames()
    cases = [('mp4', 'libx264'), ('mov', 'libx264'), ('mov', 'libx265'), ('mov', 'prores_ks'), ('mov', 'mjpeg')]
    with tempfile.TemporaryDirectory(prefix='clef-adapter-test-') as td:
        td = Path(td)
        req_path = td / 'request.json'
        req_path.write_text(json.dumps(template))
        for container, codec in cases:
            clip = td / ('clip.' + container)
            pixel_format = {'prores_ks': 'yuv422p10le', 'mjpeg': 'yuvj420p'}.get(codec, 'yuv420p')
            extra = ['-x265-params', 'pools=none:frame-threads=1:log-level=error'] if codec == 'libx265' else []
            subprocess.run(['ffmpeg', '-v', 'error', '-f', 'rawvideo', '-pixel_format', 'rgb24',
                            '-video_size', '128x96', '-framerate', '6', '-i', 'pipe:0', '-an', '-c:v', codec,
                            '-pix_fmt', pixel_format, '-threads', '1', *extra, '-y', str(clip)],
                           input=rgb.tobytes(), check=True, capture_output=True)
            converted = json.loads(convert(clip, template))
            assert converted['videos'][0]['frame_indices'] == [0, 3, 5, 8]
            assert converted['videos'][0]['fps'] == 6
            assert converted['videos'][0]['total_num_frames'] == 9
            assert template.get('media_kwargs') is None, 'converter mutated caller'
            # Decode all frames independently, then let C choose samples. Compare full host
            # encoding and patches to the adapter's preselected frames with original indices.
            raw = subprocess.run(['ffmpeg', '-v', 'error', '-noautorotate', '-i', str(clip), '-map', '0:v:0',
                                  '-fps_mode', 'passthrough', '-sws_flags', 'bilinear', '-pix_fmt', 'rgb24',
                                  '-f', 'rawvideo', 'pipe:1'], check=True, capture_output=True).stdout
            full = dict(template, videos=[{'frames': [corpus_video.image(f) for f in np.frombuffer(raw, np.uint8).reshape(-1, 96, 128, 3)], 'fps': 6}])
            compact = json.loads(json.dumps(converted))
            compact['videos'][0]['frames'] = [full['videos'][0]['frames'][i] for i in [0, 3, 5, 8]]
            compact_size = len((json.dumps(compact, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n').encode())
            # The fast, unfiltered PNGs can exceed a tight body limit. The fallback must
            # retain every frame and produce the old compact representation exactly.
            assert json.loads(convert(clip, template, max_body=compact_size)) == compact
            results = []
            for i, req in enumerate([converted, full]):
                patches = td / f'patches-{i}'
                p = subprocess.run([str(ROOT / 'clef-tool'), 'encode-patches', 'gguf/clef-flash.gguf', str(patches)],
                                   input=json.dumps(req), text=True, capture_output=True, check=True)
                assert not p.stdout.startswith('ERR'), p.stdout
                results.append((json.loads(p.stdout), patches.read_bytes()))
            assert results[0] == results[1], (container, codec)
            process = subprocess.run([sys.executable, '-B', str(ROOT / 'tools/video_request.py'), str(clip),
                                      '--request', str(req_path)], capture_output=True, text=True, check=True)
            assert json.loads(process.stdout) == converted
        for kwargs in [{'num_frames': 10}, {'fps': 0}, {'fps': float('nan')}, {'max_frames': 3},
                       {'num_frames': 0}, {'max_body': 1}, {'timeout': 0.000001}]:
            refuse(convert, clip, template, **kwargs)
        refuse(convert, clip, dict(template, videos=[{}]))
        refuse(convert, clip, dict(template, media_kwargs={'videos_kwargs': {'fps': 2}}))
        bad = td / 'bad.mov'
        for data in [b'', b'invalid video data', clip.read_bytes()[:len(clip.read_bytes()) // 2]]:
            bad.write_bytes(data)
            refuse(convert, bad, template)
        # Disabled MOV data references must never fetch from the network or a local file.
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0)); listener.listen(); listener.settimeout(0.1)
            for url in [f'http://127.0.0.1:{listener.getsockname()[1]}/external.mov', req_path.as_uri()]:
                bad.write_bytes(external_reference(clip.read_bytes(), url))
                try:
                    got = json.loads(convert(bad, template))
                except ValueError:
                    pass
                else:
                    assert got == converted, 'external reference changed decoded content'
            try:
                conn, _ = listener.accept(); conn.close()
            except TimeoutError:
                pass
            else:
                raise AssertionError('external reference opened a network connection')
        p = subprocess.run([sys.executable, '-B', str(ROOT / 'tools/video_request.py'), str(td / 'missing'),
                            '--request', str(req_path)], capture_output=True, text=True)
        assert p.returncode and not p.stdout and 'video_request:' in p.stderr
    refuse(run_bounded, [sys.executable, '-c', 'import time; time.sleep(30)'], 10, time.monotonic() + 0.1)
    refuse(run_bounded, [sys.executable, '-c', 'import os; os.write(1,b"x"*100000)'], 10, time.monotonic() + 5)
    refuse(run_bounded, [sys.executable, '-c', 'import os; os.write(2,b"x"*100000)'], 10, time.monotonic() + 5)
    b = io.BytesIO(); Image.new('RGB', (32, 32), (23, 87, 195)).save(b, 'PNG')
    png = b.getvalue()
    assert base64.b64decode(png_frames(png + png, 2, 32, 32)[0].split(',')[1]) == png
    for damaged in [png[:-1], b'x' + png[1:], png[:40] + bytes([png[40] ^ 1]) + png[41:],
                    png + png, png[:8] + b'\xff\xff\xff\xff' + png[12:],
                    png[:33] + png[8:33] + png[33:], png[:33] + png[-12:]]:
        refuse(png_frames, damaged, 1, 32, 32)
    refuse(png_frames, png, 1, 64, 32)
    refuse(png_frames, png, 2, 32, 32)
    print('video adapter: 5 codec/container combinations match all-frame encoding and patches; compact fallback, PNG pipe, CLI, bounds, deadlines and external references PASS')


if __name__ == '__main__':
    main()
