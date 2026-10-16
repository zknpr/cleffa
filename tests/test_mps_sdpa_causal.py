"""CPU checks for the model-free MPS causal diagnostic, without GPU ownership."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ref"))
from mps_sdpa_causal import reference, relative, run_case


class CausalAuditTests(unittest.TestCase):
    def test_oracle_is_prefix_mean_for_zero_scores(self):
        q = k = torch.zeros(1, 1, 4, 2)
        v = torch.tensor([0., 2., 4., 6.]).reshape(1, 1, 4, 1)
        got = reference(q, k, v, True)
        torch.testing.assert_close(got.flatten(), torch.tensor([0., 1., 2., 3.]).double())
        torch.testing.assert_close(reference(q, k, v, False), torch.full_like(got, 3.))
        v[:, :, 2:] += 100
        torch.testing.assert_close(reference(q, k, v, True)[:, :, :2], got[:, :, :2], rtol=0, atol=0)

    def test_cpu_sdpa_passes_both_causal_checks(self):
        case = run_case("cpu", torch.float32, 8, tokens=16, heads=2)
        self.assertTrue(case["implicit_pass"])
        self.assertTrue(case["controls_pass"])

    def test_nonfinite_error_cannot_pass(self):
        for value in (float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "non-finite"):
                relative(torch.tensor([value]), torch.ones(1))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            relative(torch.zeros(1), torch.zeros(1))

    def test_near_future_leak_is_not_hidden_by_a_distant_prefix_cut(self):
        import torch.nn.functional as F
        sdpa = F.scaled_dot_product_attention
        def leaky(q, k, v, **kwargs):
            out = sdpa(q, k, v, **kwargs)
            if kwargs.get("is_causal"):
                out[:, :, 0] = (v[:, :, 0] + v[:, :, 1]) / 2
            return out
        with patch.object(F, "scaled_dot_product_attention", leaky):
            case = run_case("cpu", torch.float32, 8, tokens=32, heads=2)
        self.assertTrue(case["controls_pass"])
        self.assertGreater(case["implicit_prefix_max_change"], 1)


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main()
