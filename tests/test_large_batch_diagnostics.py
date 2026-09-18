"""Check exact logging percentiles above PyTorch's 2**24-element limit."""
import math
import unittest

import torch

from torchsx.nn.metrics import _diagnostic_quantile, scale_metrics


class LargeBatchDiagnosticTests(unittest.TestCase):
    def test_small_inputs_preserve_native_behavior(self):
        x = torch.tensor([4., 1., 1., 9., -3.])
        for q in [0., .01, .5, .99, 1.]:
            torch.testing.assert_close(_diagnostic_quantile(x, q), x.quantile(q), rtol=0, atol=0)

    def check_large_batch(self, device):
        n = 24_456_800
        x = torch.arange(n // 2, device=device, dtype=torch.float32).repeat_interleave(2).flip(0)
        with self.assertRaisesRegex(RuntimeError, 'too large'):
            x.quantile(.01)
        for q in [0., .01, .5, .99, 1.]:
            rank = q * (n - 1)
            lo, hi = math.floor(rank), math.ceil(rank)
            expected = x.new_tensor(lo // 2 + (hi // 2 - lo // 2) * (rank-lo))
            torch.testing.assert_close(_diagnostic_quantile(x, q), expected, rtol=0, atol=0)
        ids = torch.zeros(n, device=device, dtype=torch.long)
        result = scale_metrics(x, ids, 1)
        self.assertTrue(math.isfinite(result['p01_log_K']))
        self.assertTrue(math.isfinite(result['p99_log_K']))
        del ids
        x[0] = float('nan')
        self.assertTrue(torch.isnan(_diagnostic_quantile(x, .01)))

    def test_large_cpu(self):
        self.check_large_batch('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_large_cuda(self):
        self.check_large_batch('cuda')


if __name__ == '__main__':
    unittest.main(verbosity=2)
