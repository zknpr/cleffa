"""Omni media reference preparation; no native inference dependency."""

import torch
from dataclasses import dataclass
import sys


@dataclass(frozen=True)
class DecodedVideo:
    frames: object
    audio: object


def reference_request(request, load_image, load_audio=None):
    """Translate Cleffa's frame-array wire format to pinned load_video's ndarray input."""
    import numpy as np
    converted = dict(request)
    videos = []
    for video in request.get('videos') or []:
        if not isinstance(video, dict):
            videos.append(video)
            continue
        if not {'frames','fps'} <= set(video) or set(video)-{'frames','fps','audio'} or video['fps'] != 2 or not video['frames']:
            raise ValueError('Omni reference frame arrays require nonempty frames sampled at 2 FPS')
        frames = np.stack([np.asarray(load_image(frame)) for frame in video['frames']])
        if 'audio' in video:
            if load_audio is None: raise ValueError('soundtrack reference requires a verified audio loader')
            samples = load_audio(video['audio'])
            if samples.ndim != 1 or not len(samples) or not np.isfinite(samples).all():
                raise ValueError('invalid decoded soundtrack')
            videos.append(DecodedVideo(frames,samples))
        else: videos.append(frames)
    if videos:
        tracks = [isinstance(v,DecodedVideo) for v in videos]
        if any(tracks) and not all(tracks): raise ValueError('mixed soundtrack presence is unsupported')
        converted['videos'] = videos
    return converted


def encode_reference(encode_record, request, processor):
    """Scoped adapter for already-decoded frame/audio pairs, without snapshot edits.

    The pinned loader returns these same two arrays for a compressed container.
    Keep token assembly, processor calls and interleaving in the pinned encoder.
    Reference generation is sequential; never leave its temporary loader installed.
    """
    module = sys.modules[encode_record.__module__]
    converted = reference_request(request,module.load_image,module.load_audio)
    original = module.load_video
    def load_video(value):
        return (value.frames,value.audio) if isinstance(value,DecodedVideo) else original(value)
    try:
        module.load_video = load_video
        return encode_record(processor.tokenizer,converted,processor=processor)
    finally:
        module.load_video = original


def _positions(thinker, ids, mask, media):
    lengths = media['feature_attention_mask'].sum(-1) if media.get('input_features') is not None else None
    return thinker.get_rope_index(ids, media.get('image_grid_thw'), media.get('video_grid_thw'),
                                  mask, media.get('use_audio_in_video',False), lengths, media.get('video_second_per_grid'))[0]


@torch.inference_mode()
def _audio_features(thinker,media):
    """Capture pinned FP32 audio execution, including valid rows after CNN packing."""
    tower = thinker.audio_tower
    dtype = next(tower.parameters()).dtype
    # Nonpersistent sinusoid buffers can remain FP32 in a BF16 checkpoint load.
    # Restoring the weight dtype must not quantize those buffers between records.
    buffers = {name:value.detach().clone() for name,value in tower.named_buffers()}
    lengths = media['feature_attention_mask'].sum(-1).tolist()
    if not lengths or any(n <= 0 for n in lengths):
        raise ValueError('audio requires nonempty valid features')
    counts = [13*(n//100)+(n%100+7)//8 for n in lengths]
    dumps,handles,convolutions = {},[],{}
    def save_split(name,value):
        if len(value) != sum(counts):
            raise ValueError('audio feature and valid-token counts differ')
        for i,part in enumerate(value.split(counts,dim=0)):
            dumps[f'audio.{i}.{name}'] = part.detach().cpu().clone().contiguous()
    try:
        tower.float()
        dumps['audio.position_table'] = tower.positional_embedding.positional_embedding[:13].detach().cpu().clone().contiguous()
        for name in ('conv2d1','conv2d2','conv2d3'):
            convolutions[name] = []
            handles.append(getattr(tower,name).register_forward_hook(
                lambda module,args,output,name=name: convolutions[name].append(output.detach().cpu().clone().contiguous())))
        handles.append(tower.layers[0].register_forward_pre_hook(lambda module,args: save_split('conv',args[0])))
        for li,layer in enumerate(tower.layers):
            handles.append(layer.register_forward_hook(lambda module,args,output,li=li: save_split(f'layer.{li:02d}',output[0])))
        features = thinker.get_audio_features(media['input_features'].float(),media['feature_attention_mask'],return_dict=True).last_hidden_state
        save_split('output',features)
        for name,parts in convolutions.items(): dumps['audio.'+name] = torch.cat(parts,dim=0)
        for i,values in enumerate(media['input_features']): dumps[f'audio.{i}.features'] = values.detach().cpu().clone().contiguous()
        return features,dumps
    finally:
        for handle in handles: handle.remove()
        tower.to(dtype)
        for name,value in buffers.items():
            parent,_,key = name.rpartition('.')
            setattr(tower.get_submodule(parent) if parent else tower,key,value)


@torch.inference_mode()
def prepare_vision(thinker, inputs):
    """Run media towers in FP32, retaining audio stages and three DeepStack outputs."""
    prepared = []
    dtype = next(thinker.visual.parameters()).dtype
    try:
        thinker.visual.float()
        for batch in inputs:
            ids, media = batch['input_ids'], batch.get('media') or {}
            hidden = thinker.get_input_embeddings()(ids).float()
            positions = _positions(thinker, ids, batch['attention_mask'], media)
            features, masks, dumps, image_index = [], [], {}, 0
            if media.get('input_features') is not None:
                audio,audio_dumps = _audio_features(thinker,media)
                audio_mask = ids == thinker.config.audio_token_id
                if int(audio_mask.sum()) != audio.shape[0]:
                    raise ValueError('audio features and placeholders differ')
                hidden = hidden.masked_scatter(audio_mask.unsqueeze(-1),audio.to(hidden.dtype).flatten())
                dumps.update(audio_dumps)
            for pixels, grid, token, function in (
                ('pixel_values', 'image_grid_thw', thinker.config.image_token_id, thinker.get_image_features),
                ('pixel_values_videos', 'video_grid_thw', thinker.config.video_token_id, thinker.get_video_features),
            ):
                if pixels not in media:
                    continue
                result = function(media[pixels].float(), media[grid], return_dict=True)
                main, extra = result.pooler_output, result.deepstack_features
                mask = ids == token
                if int(mask.sum()) != main.shape[0]:
                    raise ValueError('visual feature and placeholder counts differ')
                # MPS masked_scatter flattens its source tensor's metadata in
                # place. Give it a separate view so per-image dump geometry
                # remains intact after the scatter.
                hidden = hidden.masked_scatter(mask.unsqueeze(-1), main.to(hidden.dtype).flatten())
                features.append(extra)
                masks.append(mask)
                merge = thinker.config.vision_config.spatial_merge_size
                sizes = [h*w//(merge*merge) for t,h,w in media[grid].tolist() for _ in range(t)]
                start = 0
                for size in sizes:
                    key = f'vision.{image_index}'
                    dumps[key] = main[start:start+size].detach().cpu().clone().contiguous()
                    for level, tensor in enumerate(extra):
                        dumps[f'{key}.deepstack.{level}'] = tensor[start:start+size].detach().cpu().clone().contiguous()
                    start += size
                    image_index += 1
                if start != main.shape[0]:
                    raise ValueError(f'visual grid and feature counts differ: grid={media[grid].tolist()}, merge={merge}, sizes={sizes}, features={tuple(main.shape)}')
            visual_mask, deepstack = None, None
            if masks:
                visual_mask = torch.stack(masks).any(0)
                if any(len(f) != len(features[0]) for f in features):
                    raise ValueError('image and video DeepStack depths differ')
                deepstack = []
                for level in range(len(features[0])):
                    joint = hidden.new_zeros(int(visual_mask.sum()), hidden.shape[-1])
                    for mask, extra in zip(masks, features):
                        joint[mask[visual_mask]] = extra[level]
                    deepstack.append(joint)
            prepared.append(dict(inputs_embeds=hidden, position_ids=positions, visual_pos_masks=visual_mask,
                                 deepstack_visual_embeds=deepstack, dumps=dumps))
    finally:
        thinker.visual.to(dtype)
    return prepared


def media_layout(encoded, thinker, processor=None):
    """Describe temporal grids inside Omni's single video placeholder run."""
    media = encoded.media or {}
    ids = torch.tensor([encoded.input_ids], dtype=torch.long)
    positions = _positions(thinker, ids, torch.ones_like(ids), media)
    starts = []
    merge = thinker.config.vision_config.spatial_merge_size
    for key, token in (('image_grid_thw', thinker.config.image_token_id),
                       ('video_grid_thw', thinker.config.video_token_id)):
        cursor = 0
        slots = (ids[0] == token).nonzero().flatten().tolist()
        for t,h,w in media.get(key, torch.empty(0,3,dtype=torch.long)).tolist():
            size = h*w//(merge*merge)
            for _ in range(t):
                part = slots[cursor:cursor+size]
                if len(part) != size or not part or part != list(range(part[0],part[0]+size)):
                    raise ValueError('invalid visual placeholder run')
                starts.append([part[0],h,w])
                cursor += size
        if cursor != len(slots):
            raise ValueError('visual placeholders lack a matching grid')
    result = dict(images=sorted(starts), position_ids=positions[:,0].cpu().tolist())
    if media.get('input_features') is not None:
        lengths = media['feature_attention_mask'].sum(-1).tolist()
        slots = (ids[0] == thinker.config.audio_token_id).nonzero().flatten().tolist()
        videos = len(media.get('video_grid_thw',[])) if media.get('use_audio_in_video',False) else 0
        if videos > len(lengths): raise ValueError('video soundtracks exceed audio feature count')
        clips,cursor = [],0
        for i,frames in enumerate(lengths):
            count = 13*(frames//100)+(frames%100+7)//8
            rows = slots[cursor:cursor+count]
            if count <= 0 or len(rows) != count: raise ValueError('audio placeholders lack matching features')
            clips.append(dict(frames=frames,padded_frames=media['input_features'].shape[-1],token_rows=rows,
                              video_index=i-(len(lengths)-videos) if i>=len(lengths)-videos else -1))
            cursor += count
        if cursor != len(slots): raise ValueError('audio feature counts differ from placeholders')
        result['audio'] = clips
    return result
