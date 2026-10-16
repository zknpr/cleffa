"""Small independent HF vision/DeepStack execution and media-layout contracts."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from transformers import Qwen3OmniMoeThinkerConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeThinkerForConditionalGeneration

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ref'))
import golden_io
import omni
import omni_media
MPS='--mps' in sys.argv
if MPS: sys.argv.remove('--mps')


class OmniVisionOracle(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(91)
        cfg=Qwen3OmniMoeThinkerConfig(
            text_config=dict(vocab_size=64,hidden_size=32,num_hidden_layers=3,num_attention_heads=2,
                             num_key_value_heads=1,head_dim=128,num_experts=4,num_experts_per_tok=2,
                             moe_intermediate_size=16,rope_scaling=dict(rope_type='default',mrope_section=[24,20,20])),
            vision_config=dict(depth=3,hidden_size=32,num_heads=4,intermediate_size=48,
                               patch_size=2,temporal_patch_size=2,spatial_merge_size=2,out_hidden_size=32,
                               num_position_embeddings=16,deepstack_visual_indexes=[0,1,2]),
            audio_config=dict(num_mel_bins=8,encoder_layers=1,encoder_attention_heads=2,encoder_ffn_dim=32,
                              d_model=16,output_dim=32,downsample_hidden_size=4),
            image_token_id=10,video_token_id=11,vision_start_token_id=12,vision_end_token_id=13,
            audio_token_id=14,audio_start_token_id=15,audio_end_token_id=16,position_id_per_seconds=13)
        cfg._attn_implementation='sdpa'
        cfg.text_config._experts_implementation='eager'
        self.thinker=Qwen3OmniMoeThinkerForConditionalGeneration(cfg).eval().bfloat16()
        self.thinker.lm_head=torch.nn.Identity()
        self.inputs=[]
        for image,video in ((True,False),(False,True),(True,True),(False,False)):
            ids=[1]; media={}
            if image:
                ids += [12]+[10]*6+[13,2]
                media.update(pixel_values=torch.randn(24,24),image_grid_thw=torch.tensor([[1,4,6]]))
            if video:
                ids += [12]+[11]*12+[13,3]
                media.update(pixel_values_videos=torch.randn(48,24),video_grid_thw=torch.tensor([[2,4,6]]),
                             video_second_per_grid=torch.tensor([1.]))
            ids += [4,5]
            self.inputs.append(dict(input_ids=torch.tensor([ids]),attention_mask=torch.ones(1,len(ids),dtype=torch.long),media=media))

    def test_preparation_matches_full_forward_and_restores_dtype(self):
        full=copy.deepcopy(self.thinker).float()
        prepared=omni_media.prepare_vision(self.thinker,self.inputs)
        for batch,ready in zip(self.inputs,prepared):
            with torch.inference_mode(), omni.capture_text(full.model) as capture:
                want=full(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],use_cache=False,
                          **batch['media']).logits
            torch.testing.assert_close(ready['inputs_embeds'][0].cpu(),capture['embed'],rtol=0,atol=0)
            with torch.inference_mode(), omni.capture_text(full.model) as supplied:
                got=full.model(inputs_embeds=ready['inputs_embeds'],attention_mask=batch['attention_mask'],
                               position_ids=ready['position_ids'],visual_pos_masks=ready['visual_pos_masks'],
                               deepstack_visual_embeds=ready['deepstack_visual_embeds'],use_cache=False).last_hidden_state
            torch.testing.assert_close(got,want,rtol=0,atol=0)
            for k in capture:
                torch.testing.assert_close(supplied[k],capture[k],rtol=0,atol=0)
            if batch['media']:
                self.assertEqual(len(ready['deepstack_visual_embeds']),3)
                self.assertTrue(all(torch.count_nonzero(v) for v in ready['deepstack_visual_embeds']))
                self.assertIn('vision.0.deepstack.2',ready['dumps'])
        self.assertTrue(all(p.dtype==torch.bfloat16 for p in self.thinker.parameters()))

    def test_staged_decoder_preserves_pending_deepstack_and_media_dumps(self):
        from omni_staged import move
        prepared = omni_media.prepare_vision(self.thinker, self.inputs)
        expected, dumps = omni.stream_text(self.thinker.model, self.inputs, prepared=prepared)
        state = {}
        omni.stream_text(self.thinker.model, self.inputs, prepared=prepared, stage_state=state, layer_stop=1)
        restored = copy.deepcopy(move(state, 'cpu'))
        got, captured = omni.stream_text(self.thinker.model, self.inputs, stage_state=restored)
        for a, b, da, db in zip(expected, got, dumps, captured):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
            self.assertEqual(da.keys(), db.keys())
            for key in da:
                torch.testing.assert_close(da[key], db[key], atol=0, rtol=0, msg=key)

    def test_layout_splits_temporal_grids_without_extra_wrappers(self):
        batch=self.inputs[2]
        encoded=SimpleNamespace(input_ids=batch['input_ids'][0].tolist(),media=batch['media'],questions=[])
        layout=omni_media.media_layout(encoded,self.thinker)
        self.assertEqual(layout['images'],[[2,4,6],[11,4,6],[17,4,6]])
        pos,_=self.thinker.get_rope_index(batch['input_ids'],batch['media']['image_grid_thw'],
                                         batch['media']['video_grid_thw'],batch['attention_mask'],False,None,
                                         batch['media']['video_second_per_grid'])
        self.assertEqual(layout['position_ids'],pos[:,0].tolist())
        self.assertEqual(layout['position_ids'][0][17]-layout['position_ids'][0][11],13)
        line=json.loads(golden_io.encoded_line({'id':'mixed'},encoded,media_layout=layout))
        self.assertEqual(line['images'],layout['images'])
        self.assertEqual(line['position_ids'],layout['position_ids'])
        legacy=json.loads(golden_io.encoded_line({'id':'mixed'},encoded))
        self.assertNotIn('images',legacy)
        self.assertNotIn('position_ids',legacy)

    def test_streamed_matches_post_injection_boundaries(self):
        full=copy.deepcopy(self.thinker).float()
        prepared=omni_media.prepare_vision(self.thinker,self.inputs)
        got,dumps=omni.stream_text(self.thinker.model,self.inputs,prepared=prepared)
        for batch,hidden,dump,ready in zip(self.inputs,got,dumps,prepared):
            with torch.inference_mode(),omni.capture_text(full.model) as capture:
                want=full(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],
                          use_cache=False,**batch['media']).logits
            torch.testing.assert_close(hidden,want,atol=0,rtol=0)
            for key in capture:
                torch.testing.assert_close(dump[key],capture[key],atol=0,rtol=0,msg=key)
            for key in ready['dumps']:
                torch.testing.assert_close(dump[key],ready['dumps'][key],atol=0,rtol=0)
        # Nonzero extras must affect each boundary; this catches captures taken
        # from the raw decoder output before its visual residual addition.
        selected=self.inputs[:1]
        synthetic=copy.deepcopy(prepared[:1])
        for i,extra in enumerate(synthetic[0]['deepstack_visual_embeds']): extra.fill_(i+1)
        after,observed=omni.stream_text(self.thinker.model,selected,prepared=synthetic)
        with torch.inference_mode(),omni.capture_text(full.model) as expected:
            want=full.model(inputs_embeds=synthetic[0]['inputs_embeds'],
                            attention_mask=selected[0]['attention_mask'],
                            position_ids=synthetic[0]['position_ids'],
                            visual_pos_masks=synthetic[0]['visual_pos_masks'],
                            deepstack_visual_embeds=synthetic[0]['deepstack_visual_embeds'],use_cache=False).last_hidden_state
        torch.testing.assert_close(after[0],want,atol=0,rtol=0)
        for i in range(3):
            key=f'layer.{i:02d}'
            torch.testing.assert_close(observed[0][key],expected[key],atol=0,rtol=0)
            self.assertFalse(torch.equal(observed[0][key],dumps[0][key]))

    def test_visual_dtype_restored_on_failure(self):
        with patch.object(self.thinker.visual,'forward',side_effect=ValueError('visual probe')):
            with self.assertRaisesRegex(ValueError,'visual probe'):
                omni_media.prepare_vision(self.thinker,self.inputs)
        self.assertTrue(all(p.dtype==torch.bfloat16 for p in self.thinker.visual.parameters()))

    @unittest.skipUnless(MPS,'explicit --mps runs the GPU media regression')
    def test_mps_visual_preparation(self):
        thinker=self.thinker.to('mps')
        batches=[dict(input_ids=b['input_ids'].to('mps'),attention_mask=b['attention_mask'].to('mps'),
                      media={k:v.to('mps') for k,v in b['media'].items()}) for b in self.inputs]
        prepared=omni_media.prepare_vision(thinker,batches)
        self.assertEqual(len(prepared),4)
        for b,ready in zip(batches,prepared):
            if b['media']:
                self.assertTrue(ready['dumps'])
                for tensor in ready['dumps'].values():
                    self.assertEqual(tuple(tensor.shape),(6,32))
                    self.assertTrue(torch.isfinite(tensor).all())

    def test_meta_positions_and_reference_frame_adapter(self):
        with torch.device('meta'):
            meta=Qwen3OmniMoeThinkerForConditionalGeneration(self.thinker.config)
        batch=self.inputs[2]
        encoded=SimpleNamespace(input_ids=batch['input_ids'][0].tolist(),media=batch['media'],questions=[])
        self.assertEqual(omni_media.media_layout(encoded,meta),omni_media.media_layout(encoded,self.thinker))
        import numpy as np
        frames={name:np.full((2,4,3),i,dtype=np.uint8) for i,name in enumerate(('one','two','three'))}
        request={'videos':[dict(frames=list(frames),fps=2)],'state':'unchanged'}
        converted=omni_media.reference_request(request,lambda name:frames[name])
        np.testing.assert_array_equal(converted['videos'][0],np.stack(list(frames.values())))
        self.assertEqual(request['videos'][0]['frames'],list(frames))
        self.assertEqual(converted['state'],'unchanged')
        for video in (dict(frames=[],fps=2),dict(frames=['one'],fps=6)):
            with self.assertRaisesRegex(ValueError,'2 FPS'):
                omni_media.reference_request({'videos':[video]},lambda name:frames[name])
        with self.assertRaisesRegex(ValueError,'verified audio loader'):
            omni_media.reference_request({'videos':[dict(frames=['one'],fps=2,audio=[])]},lambda name:frames[name])


if __name__ == '__main__': unittest.main()
