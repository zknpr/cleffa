"""Generate independent CPU FP64 Omni evidence with bounded resident weights.

Usage: .venv/bin/python -B ref/oracle_fp64.py model-omni --corpus text --output DIR
Run under tools/memory_guard.py and normally the shared GPU lock: CPU and Metal
share memory. The small reproduction may use the documented 4 GiB/48 GiB CPU-only
lane. Existing output directories are refused.
"""
from contextlib import ExitStack
import argparse
import importlib
import json
from pathlib import Path
import re
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

import omni_fp64 as fp64
from omni_fp64_policy import POLICY, REVISION, CORPORA, digest, implementation_sources, runtime_provenance

ROOT = Path(__file__).resolve().parents[1]


class Capture:
    """Write each stage immediately; consolidation holds only one record in RAM."""
    def __init__(self, output, record, tokens, last):
        self.path = output/'parts'/record
        self.path.mkdir(parents=True)
        self.start = max(0, tokens-last) if last else 0
        self.names = []
        self.media_shapes = {}

    def put(self, name, value, *, media=False):
        if name in self.names or not re.fullmatch(r'[A-Za-z0-9_.]+', name):
            raise ValueError('duplicate or invalid tensor name')
        value = (value if media else value[self.start:]).detach().contiguous()
        if not value.numel() or not torch.isfinite(value).all():
            raise ValueError('nonfinite or empty FP64 capture: '+name)
        if value.is_floating_point() and value.dtype != torch.float64:
            if not (media and re.fullmatch(r'audio\.\d+\.features', name) and value.dtype == torch.float32):
                raise ValueError('unexpected capture precision: '+name)
        save_file({name: value}, self.path/(name+'.safetensors'))
        self.names.append(name)
        if media:
            self.media_shapes[name] = list(value.shape)

    def finish(self, destination):
        tensors = {}
        for name in self.names:
            tensors.update(load_file(self.path/(name+'.safetensors')))
        save_file(tensors, destination)
        # Parts are retained on failures. A completed immutable tensor file is
        # sufficient after consolidation; removing these avoids duplicate storage.
        for path in self.path.iterdir():
            path.unlink()
        self.path.rmdir()


class Weights:
    def __init__(self, pager):
        self.pager = pager

    def read(self, key, ids=None):
        shard = self.pager.index.get(key)
        if not shard or Path(shard).name != shard:
            raise ValueError('invalid weight source: '+key)
        with safe_open(self.pager.path/shard, framework='pt', device='cpu') as source:
            if ids is None:
                return source.get_tensor(key).to(torch.bfloat16).double()
            rows = source.get_slice(key)
            # Only requested lexical/embedding rows are loaded, never V*H doubles.
            unique, inverse = torch.unique(torch.as_tensor(ids, dtype=torch.long), sorted=True, return_inverse=True)
            values = torch.stack([rows[index].to(torch.bfloat16).double() for index in unique.tolist()])
            return values[inverse]


def prepare(model, pager, weights, encoded, sink):
    from omni_media import media_layout
    hidden = weights.read('thinker.model.embed_tokens.weight', encoded.input_ids)
    media = encoded.media or {}
    thinker = model.thinker
    layout = media_layout(encoded, thinker) if media else None
    positions = torch.tensor(layout['position_ids']) if layout else torch.arange(len(hidden)).expand(3, -1)
    ids = torch.tensor(encoded.input_ids)
    visual_masks, extras = [], []
    if media.get('input_features') is not None:
        output, dumps = fp64.audio(thinker.audio_tower, pager, media['input_features'], media['feature_attention_mask'].sum(-1))
        mask = ids == thinker.config.audio_token_id
        if int(mask.sum()) != len(output):
            raise ValueError('audio placeholder count differs')
        hidden[mask] = output
        for name, value in dumps.items():
            sink.put(name, value, media=True)
        for index, value in enumerate(media['input_features']):
            sink.put(f'audio.{index}.features', value, media=True)
    image_index = 0
    for pixels, grid, token in (('pixel_values', 'image_grid_thw', thinker.config.image_token_id),
                                ('pixel_values_videos', 'video_grid_thw', thinker.config.video_token_id)):
        if pixels not in media:
            continue
        output, deepstack = fp64.vision(thinker.visual, pager, media[pixels], media[grid])
        mask = ids == token
        if int(mask.sum()) != len(output):
            raise ValueError('vision placeholder count differs')
        hidden[mask] = output
        visual_masks.append(mask)
        extras.append(deepstack)
        merge = thinker.config.vision_config.spatial_merge_size
        offset = 0
        for t, h, w in media[grid].tolist():
            for _ in range(t):
                count = h*w//(merge*merge)
                sink.put(f'vision.{image_index}', output[offset:offset+count], media=True)
                for level, value in enumerate(deepstack):
                    sink.put(f'vision.{image_index}.deepstack.{level}', value[offset:offset+count], media=True)
                offset += count
                image_index += 1
        if offset != len(output):
            raise ValueError('visual grid and output counts differ')
    mask, deepstack = None, []
    if visual_masks:
        mask = torch.stack(visual_masks).any(0)
        if any(len(values) != len(extras[0]) for values in extras):
            raise ValueError('image/video DeepStack depths differ')
        for level in range(len(extras[0])):
            joint = hidden.new_zeros(int(mask.sum()), hidden.shape[-1])
            for part_mask, values in zip(visual_masks, extras):
                joint[part_mask[mask]] = values[level]
            deepstack.append(joint)
    config = thinker.config.text_config
    with fp64.DoubleOnly():
        pe = fp64.rotary(positions, config.head_dim, config.rope_parameters['rope_theta'],
                        config.rope_parameters['mrope_section'])
    sink.put('embed', hidden)
    return dict(hidden=hidden, pe=pe, visual_mask=mask, deepstack=deepstack, layout=layout)


@torch.inference_mode()
def run(args):
    if args.output.exists():
        raise ValueError('refusing to replace existing reference evidence')
    if not 0 < args.max_weight_gib <= 16 or not 1 <= args.threads <= 16 or args.dump_last < 0:
        raise ValueError('invalid memory, thread or capture bound')
    sys.path.insert(0, str(ROOT/'tools'))
    from verify_snapshot import load_manifest, verify_snapshot
    problems = verify_snapshot(args.model_dir, load_manifest(REVISION))
    if problems:
        raise ValueError('snapshot verification failed: '+'; '.join(problems))
    sys.path.insert(0, str(args.model_dir.resolve()))
    from joint_schema_model import encode_record
    from omni_media import encode_reference
    from omni_weights import load_paged
    from golden_io import encoded_line
    torch.set_num_threads(args.threads)
    model, processor, pager = load_paged(args.model_dir, 'cpu', max_weight_bytes=int(args.max_weight_gib*(1<<30)))
    weights = Weights(pager)
    module = importlib.import_module(CORPORA[args.corpus])
    requests = module.build()
    if args.only:
        if len(set(args.only)) != len(args.only):
            raise ValueError('duplicate requested records')
        requests = [r for r in requests if r['id'] in args.only]
        if {r['id'] for r in requests} != set(args.only):
            raise ValueError('unknown requested record')
    ids = [r['id'] for r in requests]
    if not ids or len(set(ids)) != len(ids) or any(not re.fullmatch(r'[A-Za-z0-9_-]+', name) for name in ids):
        raise ValueError('invalid corpus record inventory')
    encoded = [encode_reference(encode_record, request, processor) for request in requests]
    if args.corpus == 'omni-long' and any(len(e.input_ids) != 64000 for e in encoded):
        raise ValueError('long corpus must encode to 64000 tokens')
    args.output.mkdir(parents=True)
    (args.output/'layers').mkdir()
    report = dict(complete=False, revision=REVISION, precision='float64-streamed', device='cpu',
                  independent_from_native=True, fp64_policy=POLICY, corpus=args.corpus,
                  implementation_sha256=implementation_sources(), records=[], next_layer=0,
                  memory=dict(max_weight_gib=args.max_weight_gib, threads=args.threads),
                  **runtime_provenance(args.model_dir,args.corpus))
    def status():
        (args.output/'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    status()
    started = time.monotonic()
    try:
        request_text = ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in requests)
        (args.output/'requests.jsonl').write_text(request_text)
        report['corpus_sha256'] = digest(args.output/'requests.jsonl')
        sinks = [Capture(args.output, r['id'], len(e.input_ids), args.dump_last) for r, e in zip(requests, encoded)]
        states = []
        for request, enc, sink in zip(requests, encoded, sinks):
            states.append(prepare(model, pager, weights, enc, sink))
            report['records'].append(dict(id=request['id'], tokens=len(enc.input_ids), dump_start=sink.start, dumped=True))
            print(f'{request["id"]}: prepared {len(enc.input_ids)} tokens', flush=True)
        (args.output/'encoded.jsonl').write_text(''.join(encoded_line(r, e, media_layout=s['layout'])
                                                for r, e, s in zip(requests, encoded, states)))
        tm = model.thinker.model
        for li, layer in enumerate(tm.layers):
            prefix = f'thinker.model.layers.{li}'
            with ExitStack() as stack:
                for part in ('self_attn', 'input_layernorm', 'post_attention_layernorm', 'mlp.gate'):
                    stack.enter_context(pager.materialize(layer.get_submodule(part), prefix+'.'+part, dtype=torch.float64))
                prepared = []
                for state, sink in zip(states, sinks):
                    residual = fp64.text_attention(layer, state['hidden'], state['pe'])
                    with fp64.DoubleOnly():
                        x = fp64.rms(residual, layer.post_attention_layernorm)
                        logits, routing, choices = fp64.route(x, layer.mlp.gate.weight, layer.mlp.gate.top_k)
                        top = logits.topk(layer.mlp.gate.top_k+1, dim=-1).values
                        for key, value in (('ids', choices), ('logits', logits), ('weights', routing), ('margin', top[:, -2]-top[:, -1])):
                            sink.put(f'routing.{li:02d}.{key}', value)
                    prepared.append((residual, x, choices, routing, torch.zeros_like(residual)))
                for expert in range(layer.mlp.gate.num_experts):
                    selections = [torch.where(record[2] == expert) for record in prepared]
                    if not any(len(rows) for rows, _ in selections):
                        continue
                    gate, up, down = [weights.read(f'{prefix}.mlp.experts.{expert}.{part}_proj.weight') for part in ('gate', 'up', 'down')]
                    with fp64.DoubleOnly():
                        for record, (rows, slots) in zip(prepared, selections):
                            if not len(rows):
                                continue
                            _, x, _, routing, out = record
                            # Bound expert temporaries independently of context length.
                            for first in range(0, len(rows), 2048):
                                selected, selected_slots = rows[first:first+2048], slots[first:first+2048]
                                chosen = x[selected]
                                act = torch.nn.functional.silu(chosen @ gate.T)*(chosen @ up.T)
                                out[selected] += (act @ down.T)*routing[selected, selected_slots, None]
                    del gate, up, down
                with fp64.DoubleOnly():
                    for state, sink, (residual, _, _, _, out) in zip(states, sinks, prepared):
                        hidden = residual+out
                        if li < len(state['deepstack']):
                            hidden[state['visual_mask']] += state['deepstack'][li]
                        state['hidden'] = hidden
                        sink.put(f'layer.{li:02d}', hidden)
            del prepared
            report['next_layer'] = li+1
            report['seconds'] = time.monotonic()-started
            status()
            print(f'CPU FP64 layer {li}: {report["seconds"]:.1f}s', flush=True)
        with pager.materialize(tm.norm, 'thinker.model.norm', dtype=torch.float64), fp64.DoubleOnly():
            for state, sink in zip(states, sinks):
                state['hidden'] = fp64.rms(state['hidden'], tm.norm)
                sink.put('final_norm', state['hidden'])
        lexical_ids = sorted({e.input_ids[token] for e in encoded for q in e.questions
                              for a, b in q.option_spans for token in range(a, b)})
        lexical_values = weights.read('thinker.lm_head.weight', lexical_ids)
        lookup = {token: i for i, token in enumerate(lexical_ids)}
        class LexicalRows:
            def __getitem__(self, tokens):
                return lexical_values[[lookup[token] for token in tokens.tolist()]]
        outputs = {}
        with pager.materialize(model.head, '', dtype=torch.float64, head=True), fp64.DoubleOnly():
            for request, enc, state in zip(requests, encoded, states):
                tokens = torch.tensor(enc.input_ids)[None]
                values = model.head(state['hidden'][None], tokens, torch.ones_like(tokens), [enc], LexicalRows())[0]
                if len(values) != len(enc.questions):
                    raise ValueError('head output inventory differs')
                for question, value in zip(enc.questions, values):
                    if not torch.isfinite(value).all():
                        raise ValueError('nonfinite FP64 logits')
                    outputs[request['id']+'/'+question.question_id] = value.contiguous()
        save_file(outputs, args.output/'logits.safetensors')
        for request, sink in zip(requests, sinks):
            sink.finish(args.output/'layers'/(request['id']+'.safetensors'))
        (args.output/'parts').rmdir()
        report['media_shapes'] = {r['id']: sink.media_shapes for r, sink in zip(requests, sinks)}
        report['artifacts_sha256'] = {str(p.relative_to(args.output)): digest(p) for p in
            [args.output/name for name in ('requests.jsonl', 'encoded.jsonl', 'logits.safetensors')] +
            [args.output/'layers'/(name+'.safetensors') for name in ids]}
        report['complete'] = True
    except Exception as error:
        report['error'] = str(error)
        raise
    finally:
        report['seconds'] = time.monotonic()-started
        report['memory']['peak_paged_weight_bytes'] = pager.peak_weight_bytes
        status()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model_dir', type=Path)
    parser.add_argument('--corpus', choices=CORPORA, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--only', nargs='+')
    parser.add_argument('--dump-last', type=int, default=512)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--max-weight-gib', type=float, default=2.)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
