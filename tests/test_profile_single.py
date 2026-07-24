from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from profile_single import _classify_cuda_op


class ProfileSingleTest(unittest.TestCase):
    def test_classifies_primary_cuda_kernel_families(self) -> None:
        cases = {
            "cublasGemvTensorStridedBatched": "gemv",
            "ampere_bf16_gemm": "gemm",
            "conv_depthwise2d_forward": "convolution",
            "Memcpy DtoD": "memory_copy",
            "reduce_kernel": "reduction",
            "vectorized_elementwise_kernel": "elementwise",
            "unknown_kernel": "other",
        }
        for kernel_name, expected in cases.items():
            with self.subTest(kernel_name=kernel_name):
                self.assertEqual(_classify_cuda_op(kernel_name), expected)


if __name__ == "__main__":
    unittest.main()
