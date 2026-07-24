from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark_public import (  # noqa: E402
    build_prompt,
    compute_throughput,
    decode_image,
    extract_answer,
    fixed_generation_config,
    load_mmbench_tsv,
)
from evaluation_wrapper import VLMModel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug one public benchmark sample")
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--position", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    samples = load_mmbench_tsv(args.dataset_path)
    if args.position < 0 or args.position >= len(samples):
        raise IndexError(f"position {args.position} is outside 0..{len(samples) - 1}")
    sample = samples[args.position]

    model = VLMModel(str(args.model_path), backend="transformers", device="auto")
    result = model.generate_with_metrics(
        image=decode_image(sample.image_b64),
        prompt=build_prompt(sample),
        choices=sample.choices,
        generation_config=fixed_generation_config(),
        sample_id=sample.sample_id,
    )
    parsed_answer = extract_answer(result.text)
    payload = {
        "sample_id": sample.sample_id,
        "question": sample.question,
        "choices": sample.choices,
        "reference_answer": sample.answer,
        "output_text": result.text,
        "output_repr": repr(result.text),
        "parsed_answer": parsed_answer,
        "correct": parsed_answer == sample.answer,
        "token_count": result.token_count,
        "ttft_ms": round(result.ttft_seconds * 1000.0, 3),
        "elapsed_seconds": round(result.elapsed_seconds, 3),
        "throughput_tokens_per_sec": round(
            compute_throughput(
                result.token_count,
                result.ttft_seconds,
                result.elapsed_seconds,
            ),
            3,
        ),
        "meta": result.meta,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

