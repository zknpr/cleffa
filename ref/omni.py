"""Omni reference adapter; the native engine never imports or executes this module."""

from contextlib import contextmanager, ExitStack, nullcontext
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import torch
import transformers
from safetensors.torch import save_file
from transformers.masking_utils import create_causal_mask

REVISION = "0db1cd2607d76a7bdb2a382f659e7b313079f84b"


def _save(tensor, last):
    return (tensor[-last:] if last > 0 else tensor).detach().cpu().clone().contiguous()


def _routing(output, captured, layer, last):
    logits, weights, ids = output
    p = f"routing.{layer:02d}."
    captured[p+"logits"] = _save(logits.float(), last)
    captured[p+"weights"] = _save(weights.float(), last)
    captured[p+"ids"] = _save(ids, last)
    scores = logits.float().topk(ids.shape[-1]+1, dim=-1).values
    captured[p+"margin"] = _save(scores[:, -2]-scores[:, -1], last)


@contextmanager
def capture_text(text_model, *, dump_last=0):
    captured, hooks = {}, []
    try:
        for li, layer in enumerate(text_model.layers):
            # A decoder pre-hook sees the preceding DeepStack addition too. Layer
            # output hooks alone would capture the wrong boundary for layers 0..2.
            def before(module, args, kwargs, li=li):
                h = args[0] if args else kwargs["hidden_states"]
                captured["embed" if li == 0 else f"layer.{li-1:02d}"] = _save(h[0].float(), dump_last)
            hooks.append(layer.register_forward_pre_hook(before, with_kwargs=True))
            hooks.append(layer.mlp.gate.register_forward_hook(
                lambda m, a, o, li=li: _routing(o, captured, li, dump_last)))
        hooks.append(text_model.norm.register_forward_pre_hook(
            lambda m, a: captured.__setitem__(f"layer.{len(text_model.layers)-1:02d}", _save(a[0][0].float(), dump_last))))
        hooks.append(text_model.norm.register_forward_hook(
            lambda m, a, o: captured.__setitem__("final_norm", _save(o[0].float(), dump_last))))
        yield captured
    finally:
        for hook in hooks:
            hook.remove()


@contextmanager
def float_module(module,pager,prefix):
    if pager is not None:
        with pager.materialize(module,prefix):yield module
    else:
        dtype=next(module.parameters()).dtype
        try:
            module.float()
            yield module
        finally:module.to(dtype)


@torch.inference_mode()
def stream_text(text_model, inputs, *, dump_last=0, dump_count=None, prepared=None,
                weight_pager=None, stage_state=None, layer_stop=None):
    """Layer-major FP32 prefill; only one decoder layer is upcast at a time.

The source model retains BF16 weights. Its embedding lookup is byte-exact before
upcasting; all arithmetic, routing and residual updates after that use FP32.
"""
    tm = text_model
    count = len(inputs) if dump_count is None else dump_count
    first = stage_state.get('next_layer', 0) if stage_state is not None else 0
    stop = len(tm.layers) if layer_stop is None else layer_stop
    if not 0 <= first < stop <= len(tm.layers):
        raise ValueError('invalid decoder stage range')
    if prepared is not None and len(prepared) != len(inputs):
        raise ValueError('prepared media and input counts differ')
    hidden, extras, dumps = [], [], [{} for _ in inputs]
    with ExitStack() as startup:
        if first:
            hidden, extras, dumps = (stage_state[k] for k in ('hidden', 'extras', 'dumps'))
            if not len(hidden) == len(extras) == len(dumps) == len(inputs):
                raise ValueError('decoder stage record count differs')
        if weight_pager is not None:
            if prepared is None and not first:
                startup.enter_context(weight_pager.materialize(tm.embed_tokens,'thinker.model.embed_tokens',dtype=torch.bfloat16))
            if not first:
                startup.enter_context(weight_pager.materialize(tm.rotary_emb,'thinker.model.rotary_emb'))
        for i, batch in enumerate(inputs if not first else []):
            if prepared is None:
                if batch.get("media"):
                    raise ValueError("Omni streamed media requires prepared visual features")
                h = tm.embed_tokens(batch["input_ids"]).float()
                pos = torch.arange(h.shape[1], device=h.device).view(1, 1, -1).expand(4, h.shape[0], -1)
                visual_mask = visual_extra = None
            else:
                ready = prepared[i]
                h, pos = ready['inputs_embeds'], ready['position_ids']
                visual_mask, visual_extra = ready['visual_pos_masks'], ready['deepstack_visual_embeds']
                if i < count:
                    dumps[i].update(ready['dumps'])
            text_pos, rotary_pos = (pos[0],pos[1:]) if pos.shape[0] == 4 else (None,pos)
            mask = create_causal_mask(config=tm.config, inputs_embeds=h, attention_mask=batch["attention_mask"],
                                      past_key_values=None, position_ids=text_pos)
            pe = tm.rotary_emb(h, rotary_pos)
            hidden.append(h)
            extras.append((mask, pe, text_pos, visual_mask, visual_extra, rotary_pos))
            if i < count:
                dumps[i]["embed"] = _save(h[0], dump_last)
    for li, layer in enumerate(tm.layers):
        if not first <= li < stop:
            continue
        with float_module(layer,weight_pager,f'thinker.model.layers.{li}'):
            for i, (mask, pe, pos, visual_mask, visual_extra, rotary_pos) in enumerate(extras):
                hook = None
                try:
                    if i < count:
                        hook = layer.mlp.gate.register_forward_hook(
                            lambda m, a, o, i=i, li=li: _routing(o, dumps[i], li, dump_last))
                    hidden[i] = layer(hidden[i], attention_mask=mask, position_embeddings=pe,
                                       position_ids=pos, past_key_values=None, use_cache=False)
                    if visual_extra is not None and li < len(visual_extra):
                        hidden[i] = tm._deepstack_process(hidden[i], visual_mask, visual_extra[li])
                    if i < count:
                        dumps[i][f"layer.{li:02d}"] = _save(hidden[i][0], dump_last)
                finally:
                    if hook:
                        hook.remove()
        if hidden and hidden[0].device.type == "mps":
            torch.mps.synchronize()
            torch.mps.empty_cache()
        print(f"Omni layer {li} done", flush=True)
    if stop == len(tm.layers):
        with float_module(tm.norm,weight_pager,'thinker.model.norm'):
            for i in range(len(hidden)):
                hidden[i] = tm.norm(hidden[i])
                if i < count:
                    dumps[i]["final_norm"] = _save(hidden[i][0], dump_last)
    if stage_state is not None:
        # CPU transfers in the subprocess controller preserve every FP32 bit.
        # In particular, retain rotary inputs instead of recalculating them.
        stage_state.update(next_layer=stop, hidden=hidden, extras=extras, dumps=dumps)
    return hidden, dumps


@contextmanager
def golden_directory(destination):
    """Incomplete generations never occupy the advertised golden path."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"refusing to replace existing golden directory {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=destination.name+".incomplete-", dir=destination.parent) as tmp:
        path = Path(tmp)
        (path/"layers").mkdir()
        yield path
        os.rename(path, destination)


def run(args, *, streamed):
    import corpus
    import golden_io
    import safe_attn
    root = Path(__file__).resolve().parents[1]
    if Path(args.name).name != args.name or args.name in (".", ".."):
        raise ValueError("--name must be a single golden directory name")
    sys.path.insert(0, str(root/"tools"))
    from verify_snapshot import load_manifest, verify_snapshot
    stage = getattr(args, '_stage', None)
    if stage:
        import omni_staged
        omni_staged.check_snapshot(args.model_dir, stage['snapshot_inventory'])
    bad = [] if stage else verify_snapshot(args.model_dir, load_manifest(REVISION))
    if bad:
        raise ValueError("snapshot verification failed: " + "; ".join(bad))
    sys.path.insert(0, str(args.model_dir.resolve()))
    from joint_schema_model import collate_records, encode_record, load_release_model, load_image
    from omni_media import media_layout, prepare_vision, encode_reference
    import corpus_omni_vision
    import corpus_omni_video
    import corpus_omni_long
    import corpus_audio
    import corpus_omni_av
    corpora = {'text':corpus, 'omni-vision':corpus_omni_vision, 'omni-video':corpus_omni_video,
               'omni-long':corpus_omni_long,'audio':corpus_audio,'omni-av':corpus_omni_av}
    if args.corpus not in corpora:
        raise ValueError('unsupported Omni corpus')
    requests = corpora[args.corpus].build()
    if args.only:
        requested = set(args.only)
        requests = [r for r in requests if r["id"] in requested]
        if {r["id"] for r in requests} != requested:
            raise ValueError("--only contains unknown corpus record IDs")
        args.dump_layers = len(requests)
    start = time.perf_counter()
    dtype = torch.bfloat16 if streamed else getattr(torch, args.dtype)
    budget_gib=getattr(args,'max_memory_gib',24.)
    if not 0<budget_gib<=96:raise ValueError('reference memory budget must be in (0,96] GiB')
    gpu_budget=int(budget_gib*(1<<30)*2/3)
    if torch.device(args.device).type=='mps':
        torch.mps.set_per_process_memory_fraction(gpu_budget/torch.mps.recommended_max_memory())
    weight_pager=None
    if streamed:
        from omni_weights import load_paged
        model,processor,weight_pager=load_paged(args.model_dir,args.device,max_weight_bytes=gpu_budget//2)
    else:
        weight_bytes=sum(p.stat().st_size for p in args.model_dir.glob('*.safetensors'))
        if weight_bytes*(2 if dtype==torch.float32 else 1)>gpu_budget:
            raise MemoryError('resident Omni weights exceed reference budget; use oracle_f32_stream.py')
        model, processor = load_release_model(args.model_dir, device=args.device, dtype=dtype,
                                               experts_implementation="eager")
    attention_reference = getattr(args, 'omni_attention', None)
    if attention_reference or args.safe_attn:
        safe_attn.install(model, omni=True, value_reduction=attention_reference or 'auto')
    print(f"loaded Omni in {time.perf_counter()-start:.1f}s", flush=True)
    tok, tm = processor.tokenizer, model.thinker.model
    encoded = [encode_reference(encode_record,r,processor) for r in requests]
    if args.corpus=='omni-long' and any(len(e.input_ids)!=64000 for e in encoded):
        raise ValueError('Omni long corpus no longer encodes to exactly 64000 tokens')
    batches = [collate_records([enc], tok.pad_token_id, torch.device(args.device)) for enc in encoded]
    logits_out, layer_dumps = {}, []
    stage_state = omni_staged.load_state(stage, args.device) if stage else None
    intermediate = stage and stage['stop'] < len(tm.layers)
    destination = nullcontext(None) if intermediate else golden_directory(root/"golden"/args.name)
    with destination as out, torch.inference_mode():
        if streamed:
            with ExitStack() as media_weights:
                if any(e.media for e in encoded) and not (stage and stage['first']):
                    for module,prefix in [(model.thinker.model.embed_tokens,'thinker.model.embed_tokens'),
                                          (model.thinker.visual,'thinker.visual'),(model.thinker.audio_tower,'thinker.audio_tower')]:
                        media_weights.enter_context(weight_pager.materialize(module,prefix,dtype=torch.bfloat16))
                    prepared=prepare_vision(model.thinker,batches)
                else:prepared=None
            hidden, layer_dumps = stream_text(tm, batches, dump_last=args.dump_last, dump_count=args.dump_layers,
                                             prepared=prepared, weight_pager=weight_pager,
                                             stage_state=stage_state,layer_stop=stage['stop'] if stage else None)
            if stage:
                omni_staged.check_snapshot(args.model_dir, stage['snapshot_inventory'])
            if intermediate:
                omni_staged.save_state(stage, stage_state)
                return
            with weight_pager.materialize(model.head,'',head=True), weight_pager.materialize(model.output_embeddings,'thinker.lm_head'):
                output_weight = model.output_embeddings.weight
                for req, enc, batch, last in zip(requests, encoded, batches, hidden):
                    logits = model.head(last, batch["input_ids"], batch["attention_mask"], batch["records"], output_weight)[0]
                    for q, values in zip(enc.questions, logits):
                        logits_out[f"{req['id']}/{q.question_id}"] = values.float().cpu().contiguous()
        else:
            for i, (req, enc, batch) in enumerate(zip(requests, encoded, batches)):
                with capture_text(tm, dump_last=args.dump_last) as captured:
                    logits = model(batch)[0]
                layer_dumps.append(captured if i < args.dump_layers else {})
                for q, values in zip(enc.questions, logits):
                    logits_out[f"{req['id']}/{q.question_id}"] = values.float().cpu().contiguous()
                print(f"{req['id']}: {len(enc.input_ids)} tokens", flush=True)
        request_text = "".join(json.dumps(r, ensure_ascii=False)+"\n" for r in requests)
        (out/"requests.jsonl").write_text(request_text)
        (out/"encoded.jsonl").write_text("".join(golden_io.encoded_line(r, enc,
            media_layout=media_layout(enc,model.thinker,processor) if enc.media else None)
            for r, enc in zip(requests, encoded)))
        if not all(torch.isfinite(t).all() for t in logits_out.values()):
            raise ValueError("nonfinite oracle logits")
        save_file(logits_out, out/"logits.safetensors")
        for req, captured in zip(requests, layer_dumps):
            if captured:
                if not all(torch.isfinite(t).all() for t in captured.values()):
                    raise ValueError("nonfinite oracle layer/router capture")
                save_file(captured, out/"layers"/(req["id"]+".safetensors"))
        manifest = dict(complete=True, revision=REVISION, torch=torch.__version__, transformers=transformers.__version__,
                        experts_implementation="eager", precision="float32-streamed" if streamed else args.dtype,
                        safe_attention=bool(args.safe_attn or attention_reference), corpus=args.corpus,
                        corpus_sha256=hashlib.sha256(request_text.encode()).hexdigest(),
                        records=[dict(id=r["id"], tokens=len(e.input_ids),
                                      dump_start=max(0, len(e.input_ids)-args.dump_last) if args.dump_last else 0,
                                      dumped=bool(d)) for r, e, d in zip(requests, encoded, layer_dumps)])
        manifest['independent_from_native'] = True
        manifest['attention_reference'] = dict(value_reduction=attention_reference or
                ('legacy-auto' if args.safe_attn else 'stock'), first_layer=0,
                value_block=safe_attn.VALUE_BLOCK, score_elements=safe_attn.SCORE_ELEMENTS)
        manifest['media_shapes'] = {r['id']:{key:list(value.shape) for key,value in captured.items()
                                            if key.startswith(('vision.','audio.'))}
                                    for r,captured in zip(requests,layer_dumps)}
        if weight_pager is not None:
            manifest['memory']=dict(weight_paging=True,budget_gib=budget_gib,mps_allocator_limit_bytes=gpu_budget,
                                    peak_paged_weight_bytes=weight_pager.peak_weight_bytes)
            if stage:
                manifest['memory']['layers_per_subprocess'] = args.stage_layers
                manifest['reference_sources_sha256'] = stage['sources']
                manifest['reference_policy_sha256'] = stage['policy_sha256']
        if stage:
            omni_staged.check_snapshot(args.model_dir, stage['snapshot_inventory'])
            for path, expected in stage['sources'].items():
                if omni_staged.digest(path) != expected:
                    raise ValueError('reference source changed during staged execution')
        (out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(f"wrote golden/{args.name}", flush=True)
