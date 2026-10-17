"""Page verified snapshot weights by module; inactive parameters stay on meta."""
from contextlib import contextmanager, ExitStack
import json
from pathlib import Path

import torch
from safetensors import safe_open


class WeightPager:
    def __init__(self, model_path, *, device, max_weight_bytes=4 << 30):
        self.path=Path(model_path)
        self.index=json.loads((self.path/'model.safetensors.index.json').read_text())['weight_map']
        self.device=torch.device(device)
        self.max_weight_bytes=max_weight_bytes
        self.active_bytes=0
        self.peak_weight_bytes=0

    @staticmethod
    def assign(module,name,value,*,parameter):
        parent,_,key=name.rpartition('.')
        owner=module.get_submodule(parent) if parent else module
        setattr(owner,key,torch.nn.Parameter(value,requires_grad=False) if parameter else value)

    def source_parts(self,key,shape,head):
        # Transformers merges the checkpoint's per-expert matrices into stacked
        # parameters. Numeric expert order and gate-before-up match its converter.
        if not head and key not in self.index and key.endswith(('.mlp.experts.gate_up_proj','.mlp.experts.down_proj')):
            if len(shape)!=3:raise ValueError(f'invalid stacked expert shape: {key}')
            base,kind=key.rsplit('.',1)
            if kind=='gate_up_proj':
                if shape[1]%2:raise ValueError(f'invalid gate/up shape: {key}')
                half=shape[1]//2
                return [(f'{base}.{e}.{part}_proj.weight',(e,slice(k*half,(k+1)*half)),(half,shape[2]))
                        for e in range(shape[0]) for k,part in enumerate(('gate','up'))]
            return [(f'{base}.{e}.down_proj.weight',e,tuple(shape[1:])) for e in range(shape[0])]
        return [(key,None,tuple(shape))]

    @contextmanager
    def materialize(self,module,prefix,*,dtype=torch.float32,head=False):
        with ExitStack() as files:
            opened={}
            def source(shard):
                if shard not in opened:
                    opened[shard]=files.enter_context(safe_open(self.path/shard,framework='pt',device='cpu'))
                return opened[shard]
            with self._materialize(module,prefix,dtype,head,source,files.close):yield module

    @contextmanager
    def _materialize(self,module,prefix,dtype,head,source,close_sources):
        parameters=list(module.named_parameters())
        buffers=list(module.named_buffers())
        if any(p.device.type!='meta' for _,p in parameters):
            raise ValueError('weight pager requires inactive meta parameters')
        size=sum(p.numel()*torch.empty((),dtype=dtype).element_size() for _,p in parameters)
        if size+self.active_bytes>self.max_weight_bytes:
            raise MemoryError('resident reference weights exceed the configured budget')
        # Check the complete inventory and shapes before copying anything to MPS.
        sources=[]
        for name,p in parameters:
            key=name if head else prefix+'.'+name
            parts=[]
            for source_key,selection,shape in self.source_parts(key,p.shape,head):
                shard='joint_head.safetensors' if head else self.index.get(source_key)
                if not shard or Path(shard).name!=shard:
                    raise ValueError(f'missing or invalid weight source: {source_key}')
                handle=source(shard)
                if tuple(handle.get_slice(source_key).get_shape())!=shape:
                    raise ValueError(f'missing or mismatched reference weight: {source_key}')
                parts.append((source_key,shard,selection))
            sources.append((name,p.shape,parts))
        self.active_bytes+=size
        self.peak_weight_bytes=max(self.peak_weight_bytes,self.active_bytes)
        try:
            for name,shape,parts in sources:
                value=torch.empty(shape,device=self.device,dtype=dtype) if len(parts)>1 else None
                for key,shard,selection in parts:
                    # Match load_release_model(dtype=BF16) followed by .float(),
                    # including any source head tensors originally stored in FP32.
                    part=source(shard).get_tensor(key).to(torch.bfloat16).to(device=self.device,dtype=dtype)
                    if selection is None:value=part
                    else:value[selection].copy_(part)
                    del part
                self.assign(module,name,value,parameter=True)
                del value
            for name,value in buffers:
                if value.device.type=='meta':raise ValueError('reference buffer was not initialized')
                self.assign(module,name,value.to(self.device),parameter=False)
            # Reuse shard metadata while assembling experts, then drop mappings
            # before inference and FP64 audits need their temporary storage.
            close_sources()
            yield module
        finally:
            for name,p in parameters:
                self.assign(module,name,torch.empty_like(p,device='meta'),parameter=True)
            for name,value in buffers:self.assign(module,name,value,parameter=False)
            self.active_bytes-=size
            if self.device.type=='mps':
                torch.mps.synchronize()
                torch.mps.empty_cache()


def load_paged(model_path,device,*,max_weight_bytes=4 << 30):
    """Construct the pinned classes without allocating their parameter storage."""
    from accelerate import init_empty_weights
    from transformers import AutoConfig,AutoProcessor,Qwen3OmniMoeForConditionalGeneration
    from joint_schema_model import ClefModel,JointSchemaHead
    path=Path(model_path)
    config=AutoConfig.from_pretrained(path)
    config.enable_audio_output=False
    config._experts_implementation='eager'
    config.thinker_config.text_config._experts_implementation='eager'
    previous=torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        # Nonpersistent rotary/audio buffers must be initialized on CPU, not meta.
        with init_empty_weights(include_buffers=False):
            backbone=Qwen3OmniMoeForConditionalGeneration(config)
            head=JointSchemaHead(**json.loads((path/'joint_head_config.json').read_text()))
    finally:
        torch.set_default_dtype(previous)
    model=ClefModel(backbone.thinker,head).eval()
    if any(p.device.type!='meta' for p in model.parameters()):
        raise ValueError('paged model construction allocated resident parameters')
    return model,AutoProcessor.from_pretrained(path),WeightPager(path,device=device,max_weight_bytes=max_weight_bytes)
