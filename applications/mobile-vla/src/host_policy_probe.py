"""Run one screenshot-to-action policy step with the local BF16 Qwen3.5-2B model.

This probe never executes the proposed action. It emits the raw model text, a
parsed action, deterministic safety-gate decision, and timing evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from action_gate import decide, load_policy, validate_action


def extract_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("model output did not contain a JSON object")


def build_user_prompt(
    instruction: str,
    width: int,
    height: int,
    required_target_tokens: tuple[str, ...] = (),
) -> str:
    target_constraint = (
        "\n动作目标必须同时包含这些已检索并确认的语义："
        + "、".join(required_target_tokens)
        if required_target_tokens
        else ""
    )
    return f"""用户任务：{instruction}
当前截图分辨率：{width}x{height}
{target_constraint}

请只决定下一步动作。输出一个 JSON 对象，至少包含：
- action: tap/long_press/swipe/type/back/home/open_app/wait/clarify/handoff/terminate
- operation: 动作的语义类型，例如 send_message、payment、open_settings
- target: 目标控件文字或用途；无目标时为 null
- coordinates: tap 为 [x,y]，swipe 为 [x1,y1,x2,y2]，其他情况为 null
- text: 需要输入的文字；否则为 null
- risk: low/medium/high
- confidence: 0 到 1
- requires_confirmation: true/false
- reason: 简短依据
- expected_next_state: 执行后应观察到的状态；不执行时说明原因
"""


def average_hash(image, size: int = 16) -> str:
    grayscale = image.convert("L").resize((size, size))
    pixels = list(
        grayscale.get_flattened_data()
        if hasattr(grayscale, "get_flattened_data")
        else grayscale.getdata()
    )
    mean = sum(pixels) / len(pixels)
    bits = "".join("1" if value >= mean else "0" for value in pixels)
    return f"{int(bits, 2):0{size * size // 4}x}"


def run_probe(
    *,
    model_path: Path,
    image_path: Path,
    instruction: str,
    system_prompt_path: Path,
    policy_path: Path,
    max_new_tokens: int,
    required_target_tokens: tuple[str, ...] = (),
) -> dict[str, Any]:
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this BF16 host probe")
    image = Image.open(image_path).convert("RGB")
    system_prompt = system_prompt_path.read_text(encoding="utf-8")
    policy = load_policy(policy_path)

    load_start = time.perf_counter()
    processor = AutoProcessor.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="cuda:0",
    ).eval()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_start

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": build_user_prompt(
                        instruction,
                        image.width,
                        image.height,
                        required_target_tokens,
                    ),
                },
            ],
        },
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    input_tokens = int(inputs.input_ids.shape[1])

    torch.cuda.synchronize()
    generation_start = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    torch.cuda.synchronize()
    generation_seconds = time.perf_counter() - generation_start
    generated_ids = output_ids[0, input_tokens:]
    output_tokens = int(generated_ids.shape[0])
    raw_text = processor.tokenizer.decode(
        generated_ids, skip_special_tokens=True
    ).strip()

    result: dict[str, Any] = {
        "model_path": str(model_path),
        "image_path": str(image_path),
        "instruction": instruction,
        "required_target_tokens": list(required_target_tokens),
        "image_size": [image.width, image.height],
        "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
        "image_ahash": average_hash(image),
        "load_seconds": load_seconds,
        "generation_seconds": generation_seconds,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "throughput_tokens_per_second": (
            output_tokens / generation_seconds if generation_seconds else None
        ),
        "raw_text": raw_text,
        "parsed_action": None,
        "gate_decision": "PARSE_ERROR",
        "gate_reason": "model output could not be parsed",
    }
    try:
        action = extract_json_object(raw_text)
        validate_action(action)
        gate = decide(
            action,
            policy,
            (image.width, image.height),
            required_target_tokens,
        )
    except (ValueError, KeyError, TypeError) as exc:
        result["parse_error"] = str(exc)
    else:
        result["parsed_action"] = action
        result["gate_decision"] = gate.decision
        result["gate_reason"] = gate.reason
    return result


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument(
        "--system-prompt",
        type=Path,
        default=root / "prompts" / "mobile_policy_zh.txt",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=root / "configs" / "safety_policy.json",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--required-target-token", action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = run_probe(
        model_path=args.model_path,
        image_path=args.image,
        instruction=args.instruction,
        system_prompt_path=args.system_prompt,
        policy_path=args.policy,
        max_new_tokens=args.max_new_tokens,
        required_target_tokens=tuple(args.required_target_token),
    )
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
