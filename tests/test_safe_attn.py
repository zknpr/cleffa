"""The safe reference must bound every SDPA call, including one GQA group."""

import sys
import subprocess
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"ref"))
import safe_attn


class SafeAttention(unittest.TestCase):
    def test_explicit_blocked_reference_also_covers_short_inputs(self):
        torch.manual_seed(7)
        q = torch.randn(1, 4, 9, 8)
        k, v = (torch.randn(1, 2, 9, 8) for _ in range(2))
        module = SimpleNamespace(num_key_value_groups=2, is_causal=True)
        expected = torch.nn.functional.scaled_dot_product_attention(
            q.double(), k.double(), v.double(), is_causal=True, enable_gqa=True).float().transpose(1, 2)
        with patch.object(safe_attn, 'sdpa_attention_forward', side_effect=AssertionError('unblocked fallback')):
            actual, _ = safe_attn.sdpa_omni_blocked(module, q, k, v, None)
        torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)
        with patch.object(safe_attn, 'VALUE_BLOCK_THRESHOLD', 1), \
             patch.object(safe_attn, 'sdpa_blocked_values', side_effect=AssertionError('blocked control')):
            control, _ = safe_attn.sdpa_omni_unblocked(module, q, k, v, None)
        torch.testing.assert_close(control, expected, atol=2e-7, rtol=2e-6)

    def test_dense_attention_matches_committed_implementation(self):
        root = Path(__file__).resolve().parents[1]
        baseline = types.ModuleType('committed_safe_attention')
        exec(subprocess.check_output(['git', 'show', 'HEAD:ref/safe_attn.py'],
                                     cwd=root, text=True), baseline.__dict__)
        torch.manual_seed(83)
        q = torch.randn(2, 16, 19, 8)
        k, v = (torch.randn(2, 2, 19, 8) for _ in range(2))
        module = SimpleNamespace(num_key_value_groups=8, is_causal=True)
        expected, _ = baseline.sdpa_safe(module, q, k, v, None)
        # The Omni memory budget must have no effect on the dense entry point.
        with patch.object(safe_attn, 'SCORE_ELEMENTS', 1024), \
             patch.object(safe_attn, 'sdpa_attention_forward',
                          wraps=safe_attn.sdpa_attention_forward) as forward:
            actual, _ = safe_attn.sdpa_safe(module, q, k, v, None)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(forward.call_count, 1)

    def check_attention(self, nq, nk, *, mask_kind=None, causal=True, batch=1):
        torch.manual_seed(5)
        q = torch.randn(batch, 16, nq, 8, dtype=torch.float64)
        k, v = (torch.randn(batch, 2, nk, 8, dtype=torch.float64) for _ in range(2))
        allowed = torch.ones(batch, 1, nq, nk, dtype=torch.bool)
        if causal and nq > 1:
            allowed &= torch.arange(nk)[None, :] <= torch.arange(nq)[:, None]
        mask = None
        if mask_kind:
            allowed[..., -1] = False
            mask = allowed if mask_kind == "bool" else torch.zeros_like(allowed, dtype=q.dtype).masked_fill(~allowed, -torch.inf)
        scores = q @ k.repeat_interleave(8, dim=1).transpose(-1, -2) / 8**.5
        scores.masked_fill_(~allowed, -torch.inf)
        expected = (scores.softmax(-1) @ v.repeat_interleave(8, dim=1)).transpose(1, 2)
        calls = []
        original = safe_attn.sdpa_attention_forward

        def checked(module, query, key, value, attention_mask, **kwargs):
            count = query.shape[0]*query.shape[1]*query.shape[2]*key.shape[2]
            self.assertLess(count, 1024, "one GQA group still exceeds the offset budget")
            calls.append(query.shape)
            return original(module, query, key, value, attention_mask, **kwargs)

        with patch.object(safe_attn, "LIMIT", 1024), patch.object(safe_attn, "sdpa_attention_forward", checked):
            result, _ = safe_attn.sdpa_omni(SimpleNamespace(num_key_value_groups=8, is_causal=causal), q, k, v, mask)
        torch.testing.assert_close(result, expected, atol=1e-12, rtol=1e-12)
        self.assertGreater(len(calls), 1)

    def test_large_gqa_group_requires_query_splits(self):
        self.check_attention(19, 19)

    def test_unequal_lengths_global_causal_positions(self):
        self.check_attention(15, 23)

    def test_explicit_masks_and_batch(self):
        for kind in ("bool", "float"):
            self.check_attention(17, 17, mask_kind=kind, batch=2)

    def test_noncausal(self):
        self.check_attention(17, 19, causal=False)

    def test_long_fp32_value_reduction_preserves_constant_values(self):
        # A long serial P@V reduction loses accuracy even with uniform scores
        # and identical values. The reference should preserve this invariant.
        q=torch.zeros(1,8,1,16)
        k=torch.zeros(1,1,64000,16)
        v=torch.full_like(k,.1)
        got,_=safe_attn.sdpa_omni(SimpleNamespace(num_key_value_groups=8,is_causal=False),q,k,v,None)
        torch.testing.assert_close(got,torch.full_like(got,.1),atol=2e-7,rtol=0)

    def test_blocked_values_preserve_gqa_masks_and_partial_blocks(self):
        torch.manual_seed(41)
        q=torch.randn(2,4,7,8)
        k,v=(torch.randn(2,2,2051,8) for _ in range(2))
        allowed=torch.arange(2051)[None,:]<=torch.arange(7)[:,None]+1020
        allowed[-1]=False
        for mask in (allowed,torch.zeros(7,2051).masked_fill(~allowed,-torch.inf)):
            expected=torch.nn.functional.scaled_dot_product_attention(q.double(),k.double(),v.double(),
                        attn_mask=mask if mask.dtype==torch.bool else mask.double(),enable_gqa=True).float().transpose(1,2)
            with patch.object(safe_attn,'VALUE_BLOCK_THRESHOLD',1), \
                 patch.object(safe_attn,'SCORE_ELEMENTS',2*2*3*2051):
                got,_=safe_attn.sdpa_omni(SimpleNamespace(num_key_value_groups=2,is_causal=False),q,k,v,mask)
            torch.testing.assert_close(got,expected,atol=2e-7,rtol=2e-6)

    def test_blocked_values_keep_global_causal_rows(self):
        torch.manual_seed(42)
        q=torch.randn(1,16,19,8)
        k,v=(torch.randn(1,2,23,8) for _ in range(2))
        expected=torch.nn.functional.scaled_dot_product_attention(q.double(),k.double(),v.double(),
                    is_causal=True,enable_gqa=True).float().transpose(1,2)
        with patch.object(safe_attn,'VALUE_BLOCK_THRESHOLD',1),patch.object(safe_attn,'LIMIT',1024):
            got,_=safe_attn.sdpa_omni(SimpleNamespace(num_key_value_groups=8,is_causal=True),q,k,v,None)
        torch.testing.assert_close(got,expected,atol=5e-7,rtol=2e-6)


if __name__ == "__main__":
    unittest.main()
