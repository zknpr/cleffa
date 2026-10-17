"""Independent pinned chunk, valid-row and attention-window geometry."""
import json
from pathlib import Path
import subprocess
import unittest

import torch
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import chunk_and_pad_features,get_valid_indices,get_audio_cu_seqlens

ROOT=Path(__file__).resolve().parents[1]


class Layout(unittest.TestCase):
    def test_chunk_boundaries_and_unequal_records(self):
        cases=[[n] for n in (1,7,8,9,99,100,101,799,800,801)]+[[1,7],[1,99],[99,101],[1,801,33,199]]
        for lengths in cases:
            with self.subTest(lengths=lengths):
                p=subprocess.run([ROOT/'tests/audio-layout',*map(str,lengths)],capture_output=True,text=True,check=True)
                native=json.loads(p.stdout)
                lens=torch.tensor(lengths)
                values=torch.arange(sum(lengths),dtype=torch.float32)[None].expand(128,-1)
                padded,chunks=chunk_and_pad_features(values,lens,50)
                indices=get_valid_indices(chunks)
                max_rows=(native['max_chunk']+7)//8
                self.assertEqual(native['positions'],(indices%max_rows).tolist())
                self.assertEqual(native['max_chunk'],padded.shape[-1])
                bounds=get_audio_cu_seqlens(chunks,lens,800,50).tolist()
                self.assertEqual(native['starts'],[a for a,b in zip(bounds,bounds[1:]) for _ in range(a,b)])
                self.assertEqual(native['ends'],[b for a,b in zip(bounds,bounds[1:]) for _ in range(a,b)])
                self.assertEqual([c[2] for c in native['chunks']],chunks.tolist())
                for ci,start,count,first in native['chunks']:
                    self.assertLess(ci,len(lengths));self.assertLessEqual(start+count,lengths[ci])
                    self.assertEqual(native['positions'][first:first+(count+7)//8],list(range((count+7)//8)))


if __name__=='__main__':unittest.main()
