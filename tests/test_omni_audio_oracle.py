"""Small pinned audio tower, media scatter and cleanup before real-model capture."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import types
import numpy as np

import torch
from transformers import Qwen3OmniMoeThinkerConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeThinkerForConditionalGeneration, SinusoidsPositionEmbedding

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ref'))
import omni
import omni_media
MPS='--mps' in sys.argv
if MPS:sys.argv.remove('--mps')


class AudioOracle(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(192)
        cfg=Qwen3OmniMoeThinkerConfig(
            text_config=dict(vocab_size=64,hidden_size=32,num_hidden_layers=3,num_attention_heads=2,
                             num_key_value_heads=1,head_dim=128,num_experts=4,num_experts_per_tok=2,
                             moe_intermediate_size=16,rope_scaling=dict(rope_type='default',mrope_section=[24,20,20])),
            vision_config=dict(depth=3,hidden_size=32,num_heads=4,intermediate_size=48,patch_size=2,
                               temporal_patch_size=2,spatial_merge_size=2,out_hidden_size=32,
                               num_position_embeddings=16,deepstack_visual_indexes=[0,1,2]),
            audio_config=dict(num_mel_bins=8,encoder_layers=2,encoder_attention_heads=2,encoder_ffn_dim=32,
                              d_model=16,output_dim=32,downsample_hidden_size=4,n_window=50,n_window_infer=800),
            image_token_id=10,video_token_id=11,vision_start_token_id=12,vision_end_token_id=13,
            audio_token_id=14,audio_start_token_id=15,audio_end_token_id=16,position_id_per_seconds=13)
        cfg._attn_implementation='sdpa';cfg.text_config._experts_implementation='eager'
        self.thinker=Qwen3OmniMoeThinkerForConditionalGeneration(cfg).eval().bfloat16()
        self.thinker.audio_tower.positional_embedding=SinusoidsPositionEmbedding(cfg.audio_config.max_source_positions,16)
        self.thinker.lm_head=torch.nn.Identity()

    def batch(self,lengths,video=False):
        counts=[13*(n//100)+(n%100+7)//8 for n in lengths]
        features=torch.randn(len(lengths),8,max(lengths))
        mask=torch.arange(max(lengths))[None]<torch.tensor(lengths)[:,None]
        ids=[1]
        media=dict(input_features=features,feature_attention_mask=mask.long(),use_audio_in_video=video)
        if not video:
            for count in counts: ids += [15]+[14]*count+[16,2]
        else:
            self.assertEqual(len(lengths),1)
            vt=[0]*4+[13]*4;at=list(range(counts[0]));vi=ai=0;ids += [12,15]
            while vi<len(vt) or ai<len(at):
                if vi<len(vt) and (ai==len(at) or vt[vi]<=at[ai]): ids.append(11);vi+=1
                else: ids.append(14);ai+=1
            ids += [16,13,2]
            media.update(pixel_values_videos=torch.randn(32,24),video_grid_thw=torch.tensor([[2,4,4]]),
                         video_second_per_grid=torch.tensor([1.]))
        ids += [3,4]
        return dict(input_ids=torch.tensor([ids]),attention_mask=torch.ones(1,len(ids),dtype=torch.long),media=media)

    def test_audio_and_av_full_vs_prepared_streamed(self):
        batches=[self.batch([1,99]),self.batch([101],True)]
        full=copy.deepcopy(self.thinker).float()
        original_buffers={k:v.clone() for k,v in self.thinker.audio_tower.named_buffers()}
        ready=omni_media.prepare_vision(self.thinker,batches)
        got,dumps=omni.stream_text(self.thinker.model,batches,prepared=ready)
        for batch,prepared,actual,captured in zip(batches,ready,got,dumps):
            with torch.inference_mode(),omni.capture_text(full.model) as expected:
                want=full(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],use_cache=False,**batch['media']).logits
            torch.testing.assert_close(prepared['inputs_embeds'][0],expected['embed'],rtol=0,atol=0)
            torch.testing.assert_close(actual,want,rtol=0,atol=0)
            for i,length in enumerate(batch['media']['feature_attention_mask'].sum(-1).tolist()):
                count=13*(length//100)+(length%100+7)//8
                for key in ('conv','layer.00','layer.01','output'):
                    self.assertEqual(prepared['dumps'][f'audio.{i}.{key}'].shape[0],count)
                torch.testing.assert_close(prepared['dumps'][f'audio.{i}.features'],batch['media']['input_features'][i],rtol=0,atol=0)
            self.assertIn('audio.conv2d1',captured)
            self.assertIn('audio.conv2d2',captured)
            self.assertIn('audio.conv2d3',captured)
        self.assertTrue(all(p.dtype==torch.bfloat16 for p in self.thinker.parameters()))
        for name,value in self.thinker.audio_tower.named_buffers():
            torch.testing.assert_close(value,original_buffers[name],rtol=0,atol=0)

    def test_audio_layout_token_rows_and_video_association(self):
        from types import SimpleNamespace
        for batch in (self.batch([1,101]),self.batch([101],True)):
            encoded=SimpleNamespace(input_ids=batch['input_ids'][0].tolist(),media=batch['media'])
            layout=omni_media.media_layout(encoded,self.thinker)
            rows=[i for i,token in enumerate(encoded.input_ids) if token==14]
            self.assertEqual([r for clip in layout['audio'] for r in clip['token_rows']],rows)
            self.assertEqual([c['video_index'] for c in layout['audio']], [0] if batch['media']['use_audio_in_video'] else [-1,-1])
            self.assertEqual([c['frames'] for c in layout['audio']],batch['media']['feature_attention_mask'].sum(-1).tolist())
            if batch['media']['use_audio_in_video']:
                self.assertEqual(layout['position_ids'][0][encoded.input_ids.index(11)],layout['position_ids'][0][encoded.input_ids.index(14)])

    def test_audio_hooks_and_dtype_restored_on_failure(self):
        batch=self.batch([99])
        with patch.object(self.thinker.audio_tower.layers[0],'forward',side_effect=RuntimeError('tower failure')):
            with self.assertRaisesRegex(RuntimeError,'tower failure'): omni_media.prepare_vision(self.thinker,[batch])
        self.assertTrue(all(p.dtype==torch.bfloat16 for p in self.thinker.parameters()))
        for module in self.thinker.modules():
            self.assertFalse(module._forward_hooks);self.assertFalse(module._forward_pre_hooks)

    def test_decoded_video_adapter_restores_loader_and_rejects_partial_tracks(self):
        module=types.ModuleType('omni_test_encoder')
        module.load_image=lambda value:np.zeros((4,4,3),dtype=np.uint8)
        module.load_audio=lambda value:np.asarray(value,dtype=np.float32)
        original=lambda value:('original',None)
        module.load_video=original
        def encode(tok,request,processor):
            self.assertIs(tok,processor.tokenizer)
            return module.load_video(request['videos'][0])
        encode.__module__=module.__name__
        processor=types.SimpleNamespace(tokenizer=object())
        req=dict(videos=[dict(frames=['frame','frame'],fps=2,audio=[0.,.25])])
        with patch.dict(sys.modules,{module.__name__:module}):
            frames,audio=omni_media.encode_reference(encode,req,processor)
            self.assertEqual(frames.shape,(2,4,4,3));np.testing.assert_array_equal(audio,[0.,.25])
            self.assertIs(module.load_video,original)
            def broken(*args,**kwargs):raise RuntimeError('encoder failure')
            broken.__module__=module.__name__
            with self.assertRaisesRegex(RuntimeError,'encoder failure'):omni_media.encode_reference(broken,req,processor)
            self.assertIs(module.load_video,original)
            req['videos'].append(dict(frames=['frame'],fps=2))
            with self.assertRaisesRegex(ValueError,'mixed soundtrack'):omni_media.encode_reference(encode,req,processor)

    @unittest.skipUnless(MPS,'explicit --mps runs the GPU media capture regression')
    def test_mps_audio_capture_keeps_output_geometry(self):
        batch=self.batch([99,101])
        batch={key:({k:v.to('mps') if isinstance(v,torch.Tensor) else v for k,v in value.items()}
                     if isinstance(value,dict) else value.to('mps')) for key,value in batch.items()}
        self.thinker.to('mps')
        ready=omni_media.prepare_vision(self.thinker,[batch])[0]
        self.assertEqual(ready['dumps']['audio.0.output'].shape,(13,32))
        self.assertEqual(ready['dumps']['audio.1.output'].shape,(14,32))


if __name__=='__main__': unittest.main()
