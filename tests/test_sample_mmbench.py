from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from sample_mmbench import allocate_proportional_counts, stratified_sample


class SampleMmbenchTest(unittest.TestCase):
    def test_allocates_exact_proportional_target(self) -> None:
        group_sizes = {
            ("large",): 70,
            ("medium",): 20,
            ("small",): 10,
        }
        allocation = allocate_proportional_counts(group_sizes, target=20)
        self.assertEqual(allocation, {
            ("large",): 14,
            ("medium",): 4,
            ("small",): 2,
        })
        self.assertEqual(sum(allocation.values()), 20)

    def test_sampling_is_deterministic_and_stratified(self) -> None:
        rows = [
            {"index": str(index), "category": "A" if index < 8 else "B"}
            for index in range(10)
        ]
        first = stratified_sample(
            rows,
            target=5,
            seed=42,
            strata=("category",),
        )
        second = stratified_sample(
            rows,
            target=5,
            seed=42,
            strata=("category",),
        )
        self.assertEqual(first, second)
        self.assertEqual(
            Counter(row["category"] for row in first),
            Counter({"A": 4, "B": 1}),
        )


if __name__ == "__main__":
    unittest.main()
