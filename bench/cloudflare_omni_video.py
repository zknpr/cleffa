"""Lossless video transport for the hosted Omni comparison.

Encode local frame/soundtrack fixtures as MOV, then require every pinned processor
tensor and token to equal the original fixture before sending anything remotely.
"""
import argparse
import base64
import datetime
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch

from cloudflare_checkout import ROOT, collect
from cloudflare_corpus import check_budget, credentials, daily_usage, final_usage, hosted_payload
from cloudflare_omni import RATE, rows

sys.path[:0] = [str(ROOT / 'ref'), str(ROOT / 'tools')]
from omni import REVISION
from omni_media import encode_reference
from verify_snapshot import load_manifest, verify_snapshot


def tensor_hashes(media):
    result = {}
    for key, value in media.items():
        if isinstance(value, torch.Tensor):
            value = value.cpu().contiguous()
            result[key] = dict(shape=list(value.shape), dtype=str(value.dtype),
                               sha256=hashlib.sha256(value.numpy().tobytes()).hexdigest())
        else:
            result[key] = value
    return result


def pack_video(video, folder, load_image):
    frames = np.stack([np.asarray(load_image(frame)) for frame in video['frames']])
    fps = video['fps']
    if (frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3 or
            type(fps) not in (int, float) or not math.isfinite(fps) or not 0 < fps <= 60):
        raise ValueError('invalid RGB frame fixture')
    output = folder / 'clip.mov'
    command = ['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-y',
               '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', f'{frames.shape[2]}x{frames.shape[1]}',
               '-framerate', str(fps), '-i', 'pipe:0']
    if video.get('audio'):
        audio = folder / 'track.wav'
        audio.write_bytes(base64.b64decode(video['audio'].split(',', 1)[1], validate=True))
        command += ['-i', str(audio)]
    command += ['-c:v', 'libx264rgb', '-preset', 'ultrafast', '-crf', '0']
    if video.get('audio'):
        command += ['-c:a', 'pcm_f32le']
    command += ['-f', 'mov', str(output)]
    subprocess.run(command, input=frames.tobytes(), capture_output=True, check=True, timeout=30)
    data = output.read_bytes()
    if len(data) > 16 << 20:
        raise ValueError('encoded fixture exceeds the hosted video limit')
    return 'data:video/quicktime;base64,' + base64.b64encode(data).decode()


def plan(inventory, passes):
    bad = verify_snapshot(ROOT / 'model-omni', load_manifest(REVISION))
    if bad:
        raise ValueError('snapshot verification failed: ' + '; '.join(bad))
    sys.path.insert(0, str(ROOT / 'model-omni'))
    from joint_schema_model import encode_record, load_image
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(ROOT / 'model-omni')
    planned = []
    with tempfile.TemporaryDirectory(prefix='omni-hosted-video-') as tmp:
        folder = Path(tmp)
        for modality in ('video', 'audio_video'):
            golden = ROOT / 'golden' / inventory[modality]['golden']
            requests, encoded = rows(golden / 'requests.jsonl'), rows(golden / 'encoded.jsonl')
            for original, expected in zip(requests, encoded, strict=True):
                source = hosted_payload(original, 'clef-omni')
                payload = {**source, 'videos': [pack_video(v, folder, load_image) for v in source['videos']]}
                before = encode_reference(encode_record, source, processor)
                after = encode_reference(encode_record, payload, processor)
                source_media, packed_media = tensor_hashes(before.media), tensor_hashes(after.media)
                if list(before.input_ids) != expected['input_ids'] or before.input_ids != after.input_ids or before.questions != after.questions:
                    raise ValueError('video transport changed tokens or question spans')
                if source_media != packed_media:
                    raise ValueError('lossless video transport changed processor tensors')
                native_request_sha256 = hashlib.sha256(json.dumps(source).encode()).hexdigest()
                planned.append(dict(model='clef-omni', modality=modality, id=original['id'], request=payload,
                    golden=str(golden.relative_to(ROOT)), questions=expected['questions'], full_input_tokens=len(after.input_ids),
                    request_sha256=hashlib.sha256(json.dumps(payload).encode()).hexdigest(),
                    input_ids_sha256=hashlib.sha256(json.dumps(list(after.input_ids)).encode()).hexdigest(),
                    transport=dict(method='lossless MOV; exact pinned processor tensors', revision=REVISION,
                                   native_request_sha256=native_request_sha256, media_tensors=source_media)))
    estimate = (planned[0]['full_input_tokens'] + passes * sum(r['full_input_tokens'] for r in planned)) * RATE / 1e6
    check_budget(0, estimate)
    return dict(type='plan', timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(), requests=planned,
                passes=passes, planned_calls=1 + passes * len(planned), warmups_per_model=1, client_retries=0,
                full_token_neuron_estimate=estimate, neurons_per_million_input_tokens=RATE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--passes', type=int, choices=(1, 2, 3, 5), default=2)
    parser.add_argument('--account')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.output.exists() or args.run and not args.account:
        parser.error('require a new output and an account for --run')
    run_plan = plan(json.loads(args.inventory.read_text()), args.passes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as out:
        out.write(json.dumps(run_plan) + '\n'); out.flush()
        if args.run:
            account, token = credentials(args.account)
            usage = daily_usage(account, token)
            check_budget(usage['used_neurons'], run_plan['full_token_neuron_estimate'])
            out.write(json.dumps(usage) + '\n'); out.flush()
            collect(run_plan, out, account, token)
            out.write(json.dumps(dict(type='complete', timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat())) + '\n')
            out.flush()
            final_usage(out, account, token)
        else:
            print(f'{run_plan["planned_calls"]} calls planned; processor tensors exactly match; no inference sent')


if __name__ == '__main__':
    main()
