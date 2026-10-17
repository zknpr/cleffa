"""Independent CPU FP64 inference arithmetic for the pinned Omni architecture.

Processor outputs and BF16 checkpoint values are inputs to this implementation.
It never consumes native residuals or expert selections. Explicit forwards avoid
the pinned model's internal FP32 RMSNorm, rotary and router casts.
"""
from contextlib import ExitStack
import math

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves


class DoubleOnly(TorchDispatchMode):
    """Fail at the operation that downcasts, rather than trusting output dtype."""
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        for value in tree_leaves(result):
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                if value.dtype != torch.float64 or value.device.type != 'cpu':
                    raise ValueError(f'non-FP64 CPU inference operation: {func}: {value.dtype}/{value.device}')
        return result


def require_double(*values):
    if any(v.dtype != torch.float64 or v.device.type != 'cpu' for v in values):
        raise ValueError('inference requires CPU float64 tensors')


def rms(x, module):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + module.variance_epsilon) * module.weight


def attention(q, k, v, *, causal=True, query_block=128, key_block=2048, observe_scores=None):
    """Online softmax with bounded Q/K tiles; grouped heads share K/V storage.

Each row's running maximum rescales both its denominator and value numerator.
Future-only tiles are skipped, so a causal row always has a finite first maximum.
The optional shape observer is for allocation-bound tests, not alternate math.
"""
    require_double(q, k, v)
    if (q.ndim != 3 or k.ndim != 3 or k.shape != v.shape or
            q.shape[1:] != k.shape[1:] or not q.numel() or not k.numel() or
            q.shape[0] % k.shape[0] or query_block <= 0 or key_block <= 0):
        raise ValueError('invalid attention geometry or tile bound')
    heads, length, width = q.shape
    group = heads // k.shape[0]
    result = torch.empty_like(q)
    scale = width ** -.5
    for head in range(k.shape[0]):
        lo, hi = head*group, (head+1)*group
        for start in range(0, length, query_block):
            end = min(length, start+query_block)
            queries = q[lo:hi, start:end]
            maximum = q.new_full((group, end-start, 1), -torch.inf)
            denominator = q.new_zeros(group, end-start, 1)
            numerator = torch.zeros_like(queries)
            for first in range(0, end if causal else length, key_block):
                last = min(length, first+key_block, end if causal else length)
                scores = (queries @ k[head, first:last].T) * scale
                if observe_scores is not None:
                    observe_scores(tuple(scores.shape))
                if causal and last > start:
                    scores.masked_fill_(torch.arange(first, last)[None] > torch.arange(start, end)[:, None], -torch.inf)
                next_maximum = torch.maximum(maximum, scores.amax(-1, keepdim=True))
                correction = (maximum-next_maximum).exp()
                probabilities = (scores-next_maximum).exp()
                numerator = numerator*correction + probabilities @ v[head, first:last]
                denominator = denominator*correction + probabilities.sum(-1, keepdim=True)
                maximum = next_maximum
            result[lo:hi, start:end] = numerator / denominator
    return result


def rotary(positions, head_dim, theta, sections):
    if (positions.ndim != 2 or positions.shape[0] != 3 or head_dim % 2 or
            len(sections) != 3 or sum(sections) != head_dim//2 or theta <= 0):
        raise ValueError('invalid interleaved rotary geometry')
    frequency = theta ** (-torch.arange(0, head_dim, 2, dtype=torch.float64)/head_dim)
    angles = positions.double()[..., None] * frequency
    selected = angles[0].clone()
    for axis in (1, 2):
        selected[:, axis:sections[axis]*3:3] = angles[axis, :, axis:sections[axis]*3:3]
    doubled = torch.cat((selected, selected), -1)
    return doubled.cos(), doubled.sin()


def rotate(x, position_embeddings):
    cosine, sine = position_embeddings
    half = x.shape[-1]//2
    return x*cosine + torch.cat((-x[..., half:], x[..., :half]), -1)*sine


def text_attention(layer, hidden, position_embeddings):
    a = layer.self_attn
    length, width = len(hidden), a.head_dim
    with DoubleOnly():
        x = rms(hidden, layer.input_layernorm)
        q = rms(a.q_proj(x).reshape(length, -1, width), a.q_norm).transpose(0, 1)
        k = rms(a.k_proj(x).reshape(length, -1, width), a.k_norm).transpose(0, 1)
        v = a.v_proj(x).reshape(length, -1, width).transpose(0, 1)
        q, k = rotate(q, position_embeddings), rotate(k, position_embeddings)
        values = attention(q, k, v).transpose(0, 1).reshape(length, -1)
        return hidden + a.o_proj(values)


def route(x, weight, top_k):
    require_double(x, weight)
    logits = x @ weight.T
    weights, ids = logits.softmax(-1).topk(top_k, dim=-1)
    return logits, weights/weights.sum(-1, keepdim=True), ids


def sinusoids(length, channels):
    if channels < 4 or channels % 2:
        raise ValueError('invalid audio sinusoid width')
    scale = math.log(10000)/(channels//2-1)
    frequencies = (-scale*torch.arange(channels//2, dtype=torch.float64)).exp()
    angles = torch.arange(length, dtype=torch.float64)[:, None]*frequencies
    return torch.cat((angles.sin(), angles.cos()), -1)


def vision_positions(grid, table, merge):
    """Bilinear learned positions in processor patch order, computed in FP64."""
    from transformers.vision_utils import get_vision_position_ids
    positions = get_vision_position_ids(grid, merge)
    side = math.isqrt(len(table))
    if side*side != len(table):
        raise ValueError('vision position table is not square')
    values, offset = [], 0
    for t, h, w in grid.tolist():
        count = t*h*w
        y, x = positions[offset:offset+count].T.double()
        y *= (side-1)/max(1, h-1)
        x *= (side-1)/max(1, w-1)
        yl, xl = y.long(), x.long()
        yh, xh = (yl+1).clamp(max=side-1), (xl+1).clamp(max=side-1)
        fy, fx = (y-yl)[:, None], (x-xl)[:, None]
        values.append(table[yl*side+xl]*(1-fy)*(1-fx) + table[yl*side+xh]*(1-fy)*fx +
                      table[yh*side+xl]*fy*(1-fx) + table[yh*side+xh]*fy*fx)
        offset += count
    return torch.cat(values)


def segmented_attention(q, k, v, boundaries):
    output = torch.empty_like(q)
    if boundaries[0] != 0 or boundaries[-1] != q.shape[1]:
        raise ValueError('invalid media attention boundaries')
    for first, last in zip(boundaries, boundaries[1:]):
        if first >= last:
            raise ValueError('empty media attention segment')
        output[:, first:last] = attention(q[:, first:last], k[:, first:last], v[:, first:last], causal=False)
    return output


@torch.inference_mode()
def vision(tower, pager, pixels, grid):
    from transformers.vision_utils import get_vision_position_ids, get_vision_cu_seqlens
    prefix = 'thinker.visual'
    pixels = pixels.double()
    with ExitStack() as stack:
        stack.enter_context(pager.materialize(tower.patch_embed, prefix+'.patch_embed', dtype=torch.float64))
        stack.enter_context(pager.materialize(tower.pos_embed, prefix+'.pos_embed', dtype=torch.float64))
        with DoubleOnly():
            hidden = tower.patch_embed(pixels) + vision_positions(grid, tower.pos_embed.weight, tower.spatial_merge_size)
            positions = get_vision_position_ids(grid, tower.spatial_merge_size)
            dim, theta = tower.rotary_pos_emb.dim, tower.rotary_pos_emb.theta
            inv = theta ** (-torch.arange(0, dim, 2, dtype=torch.float64)/dim)
            angle = (positions.double()[..., None]*inv).flatten(1)
            angle = torch.cat((angle, angle), -1)
            pe = (angle.cos().unsqueeze(0), angle.sin().unsqueeze(0))
            boundaries = get_vision_cu_seqlens(grid).tolist()
    extras = []
    for index, block in enumerate(tower.blocks):
        with pager.materialize(block, f'{prefix}.blocks.{index}', dtype=torch.float64), DoubleOnly():
            a = block.attn
            q, k, v = a.qkv(block.norm1(hidden)).reshape(len(hidden), 3, a.num_heads, a.head_dim).permute(1, 2, 0, 3)
            q, k = rotate(q, pe), rotate(k, pe)
            out = segmented_attention(q, k, v, boundaries).transpose(0, 1).reshape(len(hidden), -1)
            hidden = hidden + a.proj(out)
            hidden = hidden + block.mlp(block.norm2(hidden))
        if index in tower.deepstack_visual_indexes:
            level = tower.deepstack_visual_indexes.index(index)
            with pager.materialize(tower.merger_list[level], f'{prefix}.merger_list.{level}', dtype=torch.float64), DoubleOnly():
                extras.append(tower.merger_list[level](hidden))
    with pager.materialize(tower.merger, prefix+'.merger', dtype=torch.float64), DoubleOnly():
        return tower.merger(hidden), extras


@torch.inference_mode()
def audio(tower, pager, features, lengths):
    """Preserve clip/chunk boundaries while running every learned stage in FP64."""
    prefix = 'thinker.audio_tower'
    features = features.double()
    lengths = lengths.tolist()
    window = tower.n_window*2
    if not lengths or any(n <= 0 or n > features.shape[-1] for n in lengths):
        raise ValueError('invalid audio feature lengths')
    chunks = [clip[:, first:min(n, first+window)].T for clip, n in zip(features, lengths)
              for first in range(0, n, window)]
    chunk_lengths = [len(chunk) for chunk in chunks]
    counts = [13*(n//100)+(n%100+7)//8 for n in lengths]
    if window != 100:
        raise ValueError('only the pinned 100-frame audio CNN layout is supported')
    chunk_counts = [(n+7)//8 for n in chunk_lengths]
    dumps = {}
    def save(name, value):
        if len(value) != sum(counts):
            raise ValueError('audio capture token count differs')
        for index, part in enumerate(value.split(counts)):
            dumps[f'audio.{index}.{name}'] = part.clone().contiguous()
    with DoubleOnly():
        padded = torch.nn.utils.rnn.pad_sequence(chunks, batch_first=True).transpose(1, 2).unsqueeze(1)
    for key in ('conv2d1', 'conv2d2', 'conv2d3'):
        with pager.materialize(getattr(tower, key), prefix+'.'+key, dtype=torch.float64), DoubleOnly():
            value = torch.cat([getattr(tower, key)(part) for part in padded.split(tower.conv_chunksize)])
            dumps['audio.'+key] = value.clone().contiguous()
            padded = torch.nn.functional.gelu(value)
    with pager.materialize(tower.conv_out, prefix+'.conv_out', dtype=torch.float64), DoubleOnly():
        b, c, f, t = padded.shape
        hidden = tower.conv_out(padded.permute(0, 3, 1, 2).reshape(b, t, c*f))
        table = sinusoids(t, tower.config.d_model)
        dumps['audio.position_table'] = sinusoids(13, tower.config.d_model)
        hidden = hidden + table[None]
        hidden = torch.cat([row[:n] for row, n in zip(hidden, chunk_counts)])
        save('conv', hidden)
    infer_window = max(chunk_counts)*(tower.n_window_infer//window)
    boundaries = [0]
    for count in counts:
        for first in range(0, count, infer_window):
            boundaries.append(boundaries[-1]+min(infer_window, count-first))
    for index, layer in enumerate(tower.layers):
        with pager.materialize(layer, f'{prefix}.layers.{index}', dtype=torch.float64), DoubleOnly():
            a = layer.self_attn
            x = layer.self_attn_layer_norm(hidden)
            q, k, v = [proj(x).reshape(len(x), a.num_heads, a.head_dim).transpose(0, 1)
                       for proj in (a.q_proj, a.k_proj, a.v_proj)]
            out = segmented_attention(q, k, v, boundaries).transpose(0, 1).reshape(len(x), -1)
            hidden = hidden + a.out_proj(out)
            hidden = hidden + layer.fc2(layer.activation_fn(layer.fc1(layer.final_layer_norm(hidden))))
            save(f'layer.{index:02d}', hidden)
    with ExitStack() as stack:
        for name in ('ln_post', 'proj1', 'proj2'):
            stack.enter_context(pager.materialize(getattr(tower, name), prefix+'.'+name, dtype=torch.float64))
        with DoubleOnly():
            hidden = tower.proj2(tower.act(tower.proj1(tower.ln_post(hidden))))
            save('output', hidden)
    return hidden, dumps
