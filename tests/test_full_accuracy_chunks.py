from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.merge_benchmark_chunks import merge_results
from scripts.prepare_dataset_chunks import prepare_chunks


class FullAccuracyChunksTest(unittest.TestCase):
    def _write_dataset(self, root: Path) -> Path:
        path = root / "dataset.tsv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["index", "question", "answer"],
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            for index in range(5):
                writer.writerow(
                    {
                        "index": str(index),
                        "question": f"question {index}",
                        "answer": "A",
                    }
                )
        return path

    def _write_result(
        self,
        root: Path,
        chunk_path: Path,
        *,
        chunk_index: int,
        question_ids: list[str],
        correct: list[bool],
    ) -> Path:
        answers = [
            {
                "question_id": question_id,
                "parsed_answer": "A" if is_correct else "B",
                "correct": is_correct,
                "token_count": 1,
                "ttft_ms": 10.0 + index,
                "throughput_tokens_per_sec": 20.0 + index,
                "validation_errors": [],
                "meta": {
                    "optimization_profile": "o1_inference_mode",
                    "ttft_measurement": "first_generated_token_put",
                },
            }
            for index, (question_id, is_correct) in enumerate(
                zip(question_ids, correct, strict=True)
            )
        ]
        payload = {
            "benchmark_version": "test",
            "timestamp": "2026-07-24T00:00:00",
            "dataset_path": str(chunk_path),
            "sample_count": len(answers),
            "seed": 20260625,
            "backend": "transformers",
            "performance": {},
            "timing": {"benchmark_elapsed_seconds": float(len(answers))},
            "accuracy": {},
            "public_validation": {"passed": True, "failed_samples": 0},
            "answers": answers,
        }
        path = root / f"result-{chunk_index:04d}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_prepares_deterministic_complete_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = self._write_dataset(root)
            first = prepare_chunks(dataset, root / "chunks-a", chunk_size=2)
            second = prepare_chunks(dataset, root / "chunks-b", chunk_size=2)

            self.assertEqual(first["source_sample_count"], 5)
            self.assertEqual(first["chunk_count"], 3)
            self.assertEqual(
                [chunk["sample_count"] for chunk in first["chunks"]],
                [2, 2, 1],
            )
            self.assertEqual(
                [chunk["sha256"] for chunk in first["chunks"]],
                [chunk["sha256"] for chunk in second["chunks"]],
            )

    def test_merges_complete_chunks_with_exact_accuracy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = self._write_dataset(root)
            manifest = prepare_chunks(dataset, root / "chunks", chunk_size=2)
            chunks = [
                root / "chunks" / chunk["file"]
                for chunk in manifest["chunks"]
            ]
            paths = [
                self._write_result(
                    root,
                    chunks[0],
                    chunk_index=1,
                    question_ids=["0", "1"],
                    correct=[True, False],
                ),
                self._write_result(
                    root,
                    chunks[1],
                    chunk_index=2,
                    question_ids=["2", "3"],
                    correct=[True, True],
                ),
                self._write_result(
                    root,
                    chunks[2],
                    chunk_index=3,
                    question_ids=["4"],
                    correct=[False],
                ),
            ]

            merged = merge_results(manifest, paths)

            self.assertEqual(merged["sample_count"], 5)
            self.assertEqual(merged["accuracy"]["correct"], 3)
            self.assertEqual(merged["accuracy"]["score"], 0.6)
            self.assertTrue(merged["public_validation"]["passed"])
            self.assertTrue(merged["chunk_merge"]["complete"])
            self.assertEqual(merged["chunk_merge"]["missing_indices"], [])

    def test_rejects_partial_results_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = self._write_dataset(root)
            manifest = prepare_chunks(dataset, root / "chunks", chunk_size=2)
            first_chunk = root / "chunks" / manifest["chunks"][0]["file"]
            result = self._write_result(
                root,
                first_chunk,
                chunk_index=1,
                question_ids=["0", "1"],
                correct=[True, True],
            )

            with self.assertRaisesRegex(ValueError, "Missing chunk results"):
                merge_results(manifest, [result])


if __name__ == "__main__":
    unittest.main()
