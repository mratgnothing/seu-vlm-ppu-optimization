from __future__ import annotations

import sys
import unittest
from pathlib import Path

try:
    import torch
except ImportError:  # The local contract-only test environment need not ship torch.
    torch = None


CUSTOM_OPS = Path(__file__).resolve().parents[1] / "ppu" / "custom_ops"
if str(CUSTOM_OPS) not in sys.path:
    sys.path.insert(0, str(CUSTOM_OPS))

if torch is not None:
    from ppu_first_token_cache import ReservedDynamicLayer


@unittest.skipIf(torch is None, "torch is validated in the official PPU image")
class ReservedDynamicLayerTests(unittest.TestCase):
    def test_matches_dynamic_concatenation_and_resets_length(self) -> None:
        layer = ReservedDynamicLayer(capacity=8)
        first_k = torch.arange(12, dtype=torch.float32).reshape(1, 2, 3, 2)
        first_v = first_k + 100
        keys, values = layer.update(first_k, first_v)
        torch.testing.assert_close(keys, first_k)
        torch.testing.assert_close(values, first_v)
        self.assertEqual(layer.get_seq_length(), 3)
        self.assertEqual(layer.get_mask_sizes(1), (4, 0))

        next_k = torch.full((1, 2, 1, 2), 7.0)
        next_v = torch.full((1, 2, 1, 2), 9.0)
        keys, values = layer.update(next_k, next_v)
        torch.testing.assert_close(keys, torch.cat((first_k, next_k), dim=-2))
        torch.testing.assert_close(values, torch.cat((first_v, next_v), dim=-2))

        layer.reset()
        replacement = torch.full((1, 2, 2, 2), -3.0)
        keys, _ = layer.update(replacement, replacement)
        self.assertEqual(layer.get_seq_length(), 2)
        torch.testing.assert_close(keys, replacement)

    def test_rejects_capacity_overflow(self) -> None:
        layer = ReservedDynamicLayer(capacity=2)
        with self.assertRaisesRegex(RuntimeError, "capacity"):
            layer.update(
                torch.zeros((1, 1, 3, 1)),
                torch.zeros((1, 1, 3, 1)),
            )


if __name__ == "__main__":
    unittest.main()
