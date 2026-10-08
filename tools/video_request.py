#!/usr/bin/env python3
"""Convert one local MP4/MOV into a Cleffa JSON request. FFmpeg stays outside the engine."""
from __future__ import annotations

import argparse
import base64
import copy
import io
import json
import math
import os
import selectors
import stat
import subprocess
import sys
import tempfile
import time
import zlib
from fractions import Fraction
from pathlib import Path

from PIL import Image

MAX_INPUT = 64 << 20
MAX_PIXELS = 16777216
MAX_RAW = 256 << 20
INPUT_OPTIONS = ['-protocol_whitelist', 'file', '-f', 'mov', '-enable_drefs', '0',
                 '-use_absolute_path', '0', '-max_streams', '16', '-probesize', '1048576',
                 '-analyzeduration', '1000000', '-max_alloc', '67108864']


class OutputLimitError(ValueError):
    """A lossless encoding can be retried more compactly within the same deadline."""


def png_frames(data, count, width, height):
    """Validate FFmpeg's bounded image pipe before turning it into frame data URLs."""
    view = memoryview(data)
    frames, at = [], 0
    while at < len(view):
        start = at
        if len(frames) >= count or view[at:at + 8] != b'\x89PNG\r\n\x1a\n':
            raise ValueError('invalid PNG frame count or signature from FFmpeg')
        at += 8
        header, pixels = False, False
        while True:
            if at + 12 > len(view):
                raise ValueError('truncated PNG frame from FFmpeg')
            size = int.from_bytes(view[at:at + 4], 'big')
            end = at + size + 12
            if end > len(view):
                raise ValueError('truncated PNG chunk from FFmpeg')
            kind = view[at + 4:at + 8]
            if zlib.crc32(view[at + 4:end - 4]) != int.from_bytes(view[end - 4:end], 'big'):
                raise ValueError('invalid PNG checksum from FFmpeg')
            if not header:
                expected = width.to_bytes(4, 'big') + height.to_bytes(4, 'big') + b'\x08\x02\0\0\0'
                if kind != b'IHDR' or view[at + 8:end - 4] != expected:
                    raise ValueError('PNG dimensions or format disagree with the container')
                header = True
            elif kind == b'IHDR':
                raise ValueError('duplicate PNG header from FFmpeg')
            if kind == b'IDAT':
                pixels = True
            at = end
            if kind == b'IEND':
                if size or not pixels:
                    raise ValueError('invalid PNG end from FFmpeg')
                break
        frames.append('data:image/png;base64,' + base64.b64encode(view[start:at]).decode('ascii'))
    if len(frames) != count:
        raise ValueError('decoded frame count disagrees with the container')
    return frames


def run_bounded(command, limit, deadline):
    """Drain both pipes with byte limits and kill/reap even a codec stuck inside a call."""
    output, errors = bytearray(), bytearray()
    with subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE) as process:
        try:
            with selectors.DefaultSelector() as selector:
                for stream, data, cap in [(process.stdout, output, limit), (process.stderr, errors, 65536)]:
                    os.set_blocking(stream.fileno(), False)
                    selector.register(stream, selectors.EVENT_READ, (data, cap))
                while selector.get_map():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise ValueError('FFmpeg processing deadline exceeded')
                    for key, _ in selector.select(remaining):
                        chunk = os.read(key.fd, 65536)
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        data, cap = key.data
                        if len(data) + len(chunk) > cap:
                            error = OutputLimitError if data is output else ValueError
                            raise error('FFmpeg output exceeds byte limit')
                        data.extend(chunk)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ValueError('FFmpeg processing deadline exceeded')
            try:
                code = process.wait(timeout=remaining)
            except subprocess.TimeoutExpired as exc:
                raise ValueError('FFmpeg processing deadline exceeded') from exc
            if code:
                raise ValueError(f'{Path(command[0]).name} failed: {errors.decode(errors="replace").strip()}')
            return output
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()


def convert(path, request, *, fps=2.0, num_frames=None, max_frames=32, max_body=8 << 20, timeout=30):
    if not math.isfinite(fps) or not 0 < fps <= 1000:
        raise ValueError('sampling fps must be in (0,1000]')
    if not 1 <= max_frames <= 768 or num_frames is not None and not 1 <= num_frames <= 768:
        raise ValueError('frame limits must be in [1,768]')
    if not 1 <= max_body <= 1 << 30 or not math.isfinite(timeout) or not 0 < timeout <= 600:
        raise ValueError('invalid output byte limit or timeout')
    if not isinstance(request, dict) or not {'model', 'state', 'questions'} <= request.keys():
        raise ValueError('request must contain model, state and questions')
    if request.get('videos'):
        raise ValueError('request already contains videos; convert one clip per request')
    result = copy.deepcopy(request)
    mk = result.setdefault('media_kwargs', {})
    if not isinstance(mk, dict):
        raise ValueError('media_kwargs must be an object')
    vk = mk.setdefault('videos_kwargs', {})
    if not isinstance(vk, dict) or set(vk) - {'size', 'do_sample_frames'}:
        raise ValueError('set sampling with --fps or --num-frames; request videos_kwargs may contain size and do_sample_frames=false')
    if 'do_sample_frames' in vk and vk['do_sample_frames'] is not False:
        raise ValueError('adapter output requires do_sample_frames=false')

    # Own a bounded snapshot so probing and decoding cannot read different versions of the input.
    # Only regular local files are accepted; nonblocking open also avoids hanging on a FIFO.
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as source:
        info = os.fstat(source.fileno())
        if not stat.S_ISREG(info.st_mode) or not 8 <= info.st_size <= MAX_INPUT:
            raise ValueError('clip must be a regular local file of 8 bytes to 64 MiB')
        data = source.read(MAX_INPUT + 1)
    if not 8 <= len(data) <= MAX_INPUT:
        raise ValueError('clip exceeds input byte limit or is truncated')

    deadline = time.monotonic() + timeout
    with tempfile.TemporaryDirectory(prefix='clef-video-') as td:
        clip = Path(td) / 'input.mov'
        clip.write_bytes(data)
        del data
        probe = run_bounded(['ffprobe', '-v', 'error', *INPUT_OPTIONS, '-select_streams', 'v',
                             '-show_entries', 'stream=codec_name,width,height,nb_frames,avg_frame_rate',
                             '-of', 'json', str(clip)], 65536, deadline)
        streams = json.loads(probe)['streams']
        if len(streams) != 1:
            raise ValueError('exactly one video track is required')
        stream = streams[0]
        if stream.get('codec_name') not in {'h264', 'hevc', 'prores', 'mjpeg'}:
            raise ValueError('supported codecs: H.264, HEVC, ProRes and MJPEG')
        try:
            total = int(stream['nb_frames'])
            source_fps = float(Fraction(stream['avg_frame_rate']))
            width, height = int(stream['width']), int(stream['height'])
        except (KeyError, ValueError, ZeroDivisionError, OverflowError) as exc:
            raise ValueError('clip needs known frame count, frame rate and dimensions') from exc
        if not 1 <= total <= 18000 or not 0 < source_fps <= 1000 or total / source_fps > 600:
            raise ValueError('clip exceeds 18000 frames / 600 seconds or has invalid FPS')
        if not 32 <= width <= 16384 or not 32 <= height <= 16384 or width * height > MAX_PIXELS:
            raise ValueError('source dimensions exceed pixel limit or are below 32 pixels')
        count = num_frames if num_frames is not None else min(max(int(total / source_fps * fps), 4), 768, total)
        if count > total or count > max_frames:
            raise ValueError(f'{count} sampled frames exceeds source count or --max-frames={max_frames}; lower --fps/--num-frames')
        indices = [round(i * ((total - 1) / (count - 1))) for i in range(count)] if count > 1 else [0]
        if count > 1:
            indices[-1] = total - 1
        size = width * height * 3
        if size * count > MAX_RAW:
            raise ValueError('sampled RGB exceeds 256 MiB; lower sampling count or source dimensions')
        select = "select='" + '+'.join(f'eq(n,{i})' for i in indices) + "'"
        # Software frame threads preserve the exact RGB conversion. Cap each stage at four
        # workers and reduce parallelism with source size, down to one for large sources.
        threads = str(max(1, min(4, MAX_PIXELS // (width * height))))
        command = ['ffmpeg', '-v', 'error', '-nostdin', '-xerror', *INPUT_OPTIONS,
                           '-threads', threads, '-max_pixels', str(MAX_PIXELS), '-err_detect', 'explode',
                           '-noautorotate', '-i', str(clip), '-map', '0:v:0', '-an', '-sn', '-dn',
                           '-vf', select, '-fps_mode', 'passthrough', '-sws_flags', 'bilinear',
                           '-pix_fmt', 'rgb24']

        def serialize(images):
            vk['do_sample_frames'] = False
            result['videos'] = [{'frames': images, 'fps': source_fps, 'frame_indices': indices, 'total_num_frames': total}]
            serialized = json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n'
            if len(serialized.encode('utf-8')) > max_body:
                raise OutputLimitError('request exceeds --max-body')
            return serialized

        try:
            # Avoid the raw-RGB pipe and Python's serial PNG encoding. Filter 0 also avoids
            # Paeth reconstruction in the engine. Pixels, sampling and timestamps are unchanged.
            encoded = run_bounded(command + ['-c:v', 'png', '-threads', threads, '-pred', 'none',
                                  '-compression_level', '1', '-f', 'image2pipe', 'pipe:1'],
                                  max_body * 3 // 4, deadline)
            return serialize(png_frames(encoded, count, width, height))
        except OutputLimitError:
            # Fast PNGs can be larger. Retry the original compact representation without
            # resetting the deadline or relaxing any input limit.
            raw = run_bounded(command + ['-threads', '1', '-f', 'rawvideo', 'pipe:1'], size * count, deadline)
            if len(raw) != size * count:
                raise ValueError('decoded frame count/dimensions disagree with the container')
            images, encoded_size = [], 0
            for i in range(count):
                image = Image.frombytes('RGB', (width, height), bytes(raw[i * size:(i + 1) * size]))
                output = io.BytesIO()
                image.save(output, 'PNG')
                encoded = 'data:image/png;base64,' + base64.b64encode(output.getvalue()).decode('ascii')
                encoded_size += len(encoded)
                if encoded_size > max_body:
                    raise ValueError('encoded frames exceed --max-body; lower sampling count or source dimensions')
                images.append(encoded)
            return serialize(images)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('clip', type=Path)
    parser.add_argument('--request', type=Path, required=True, help='JSON object with model, state and questions')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--fps', type=float, default=2.0, help='target sampling rate, default 2')
    group.add_argument('--num-frames', type=int, help='explicit sampled frame count')
    parser.add_argument('--max-frames', type=int, default=32)
    parser.add_argument('--max-body', type=int, default=8 << 20, help='maximum output bytes, default 8 MiB')
    parser.add_argument('--timeout', type=float, default=30, help='probe plus decode deadline in seconds')
    args = parser.parse_args()
    try:
        with args.request.open('rb') as source:
            raw = source.read((8 << 20) + 1)
        if len(raw) > 8 << 20:
            raise ValueError('request template exceeds 8 MiB')
        result = convert(args.clip, json.loads(raw), fps=args.fps, num_frames=args.num_frames,
                         max_frames=args.max_frames, max_body=args.max_body, timeout=args.timeout)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f'video_request: {exc}\n')
    sys.stdout.write(result)


if __name__ == '__main__':
    main()
