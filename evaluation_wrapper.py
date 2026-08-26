"""Participant model wrapper for the DNDX benchmark."""

from __future__ import annotations

import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CHOICE_MARKUP_PATTERN = re.compile(
    r"(?<![A-Za-z])(?P<open>\*{1,2}|_{1,2}|`)(?P<choice>[ABCD])"
    r"(?P=open)(?![A-Za-z])",
    re.IGNORECASE,
)
CHOICE_LEADING_MARKUP_PATTERN = re.compile(
    r"(?<![A-Za-z])(?:\*{1,2}|_{1,2}|`)(?P<choice>[ABCD])"
    r"(?=[\s\.。,\，:：、\)\]）】])",
    re.IGNORECASE,
)
EXPLICIT_BOLDED_CONCLUSION_PATTERN = re.compile(
    r"^\s*[-*]\s*(?P<choice>[ABCD])\s*[:：][^\r\n]*"
    r"\*\*(?:matches exactly|is correct|is the correct choice|"
    r"完全符合|正确选项|符合题意)\*\*\s*[.!。]?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
CANONICAL_CHOICE_OUTPUT_PATTERN = re.compile(
    r"(?:final\s*)?(?:answer|choice|option|答案|选项|选择|正确答案|最终答案)"
    r"\s*(?:(?:is|为|是|[:：])\s*)*[\(\[（【]?\s*[ABCD]"
    r"\s*[\)\]）】]?",
    re.IGNORECASE,
)


def normalize_choice_markup(text: str) -> str:
    """Remove Markdown wrappers around a generated A/B/C/D choice only."""
    normalized = CHOICE_MARKUP_PATTERN.sub(
        lambda match: match.group("choice").upper(),
        text,
    )
    return CHOICE_LEADING_MARKUP_PATTERN.sub(
        lambda match: match.group("choice").upper(),
        normalized,
    )


def normalize_explicit_choice_conclusion(text: str) -> str:
    """Append a canonical answer for one unambiguous bolded option verdict.

    Some long multiple-choice explanations reach the token budget after marking
    one list item as an explicit positive conclusion but before emitting their
    final ``Answer: X`` line. This normalization is deliberately strict: it
    requires exactly one bolded positive verdict and never uses question IDs or
    reference answers.
    """
    if not text or CANONICAL_CHOICE_OUTPUT_PATTERN.search(text):
        return text
    choices = {
        match.group("choice").upper()
        for match in EXPLICIT_BOLDED_CONCLUSION_PATTERN.finditer(text)
    }
    if len(choices) != 1:
        return text
    return f"{text.rstrip()}\nAnswer: {next(iter(choices))}"


@dataclass
class GenerationConfig:
    max_new_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0


@dataclass
class GenerationResult:
    text: str
    token_count: int
    ttft_seconds: float
    elapsed_seconds: float
    meta: dict[str, Any]


@dataclass
class FirstTokenTiming:
    """Capture the first generated-token event, excluding the prompt put."""

    timestamp: float | None = None

    def observe_stream_put(
        self,
        *,
        skip_prompt: bool,
        next_tokens_are_prompt: bool,
        now: float,
    ) -> None:
        if skip_prompt and next_tokens_are_prompt:
            return
        if self.timestamp is None:
            self.timestamp = now


class VLMModel:
    """
    Default participant wrapper.

    `backend="dummy"` is for demo-only smoke tests.
    `backend="transformers"` uses a local Hugging Face model directory.
    Participants can replace the internals while preserving `generate_with_metrics`.
    """

    def __init__(
        self,
        model_path: str,
        *,
        backend: str = "auto",
        device: str = "auto",
        optimization_profile: str = "auto",
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.backend = backend
        self.optimization_profile = (
            os.getenv("VLM_OPT_PROFILE", "o1_inference_mode")
            if optimization_profile == "auto"
            else optimization_profile
        )
        if self.optimization_profile not in {
            "o0_no_grad",
            "o1_inference_mode",
        }:
            raise ValueError(
                f"Unsupported optimization profile: {self.optimization_profile}"
            )
        self._model = None
        self._processor = None
        self._tokenizer = None
        self._backend_name = "dummy"

        if backend in {"auto", "transformers"}:
            try:
                self._load_transformers_backend()
                self._backend_name = "transformers"
            except Exception as exc:
                if backend == "transformers":
                    raise
                self._load_dummy_backend(str(exc))
        else:
            self._load_dummy_backend("backend=dummy")

    @property
    def backend_name(self) -> str:
        return self._backend_name

    def generate_with_metrics(
        self,
        *,
        image,
        prompt: str,
        choices: dict[str, str],
        generation_config: GenerationConfig,
        sample_id: str,
    ) -> GenerationResult:
        if self._backend_name == "transformers":
            return self._generate_with_transformers(
                image=image,
                prompt=prompt,
                generation_config=generation_config,
            )
        return self._generate_with_dummy(
            prompt=prompt,
            choices=choices,
            generation_config=generation_config,
            sample_id=sample_id,
        )

    def _load_transformers_backend(self) -> None:
        default_ppu_sdk = Path("/usr/local/PPU_SDK")
        if default_ppu_sdk.is_dir():
            os.environ.setdefault("PPU_SDK", str(default_ppu_sdk))
            os.environ.setdefault("PPU_HOME", str(default_ppu_sdk))
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._torch = torch
        self._processor = AutoProcessor.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map=self.device,
        ).eval()
        self._tokenizer = getattr(self._processor, "tokenizer", None)
        self._ppu_gdn_patched_modules = 0
        gdn_library_path = os.getenv("SEU_PPU_GDN_LIBRARY")
        if gdn_library_path:
            custom_op_dir = Path(
                os.getenv(
                    "SEU_PPU_GDN_PYTHON_DIR",
                    str(Path(__file__).resolve().parent / "ppu" / "custom_ops"),
                )
            ).resolve()
            if str(custom_op_dir) not in sys.path:
                sys.path.insert(0, str(custom_op_dir))
            from ppu_gdn import PPUGDNLibrary

            tiles_per_head = int(os.getenv("SEU_PPU_GDN_TILES", "4"))
            self._ppu_gdn_library = PPUGDNLibrary(
                gdn_library_path,
                tiles_per_head=tiles_per_head,
                conv_threads=int(os.getenv("SEU_PPU_CONV_THREADS", "96")),
                rmsnorm_threads=int(os.getenv("SEU_PPU_RMSNORM_THREADS", "512")),
                gated_rmsnorm_threads=int(
                    os.getenv("SEU_PPU_GATED_RMSNORM_THREADS", "128")
                ),
            )
            fused_callable = self._ppu_gdn_library.transformers_callable()
            fuse_causal_conv = os.getenv("SEU_PPU_CONV_ENABLE", "0") == "1"
            self._ppu_conv_patched_modules = 0
            for module in self._model.modules():
                if type(module).__name__ == "Qwen3_5GatedDeltaNet":
                    module.recurrent_gated_delta_rule = fused_callable
                    self._ppu_gdn_patched_modules += 1
                    if fuse_causal_conv:
                        module.causal_conv1d_update = (
                            self._ppu_gdn_library.causal_conv1d_decode
                        )
                        self._ppu_conv_patched_modules += 1
            if self._ppu_gdn_patched_modules != 18:
                raise RuntimeError(
                    "SEU PPU GDN integration expected 18 Qwen3.5 modules, "
                    f"patched {self._ppu_gdn_patched_modules}"
                )
            self._ppu_qk_rope_patched_modules = 0
            if os.getenv("SEU_PPU_QK_ROPE_ENABLE", "0") == "1":
                from transformers.models.qwen3_5 import (
                    modeling_qwen3_5 as qwen35_modeling,
                )

                for module in self._model.modules():
                    if type(module).__name__ != "Qwen3_5Attention":
                        continue
                    module.forward = (
                        self._ppu_gdn_library.transformers_attention_callable(
                            module,
                            qwen35_modeling.ALL_ATTENTION_FUNCTIONS,
                            qwen35_modeling.eager_attention_forward,
                        )
                    )
                    self._ppu_qk_rope_patched_modules += 1
                if self._ppu_qk_rope_patched_modules != 6:
                    raise RuntimeError(
                        "SEU PPU q/k RMSNorm+RoPE expected 6 attention modules, "
                        f"patched {self._ppu_qk_rope_patched_modules}"
                    )
            self._ppu_rmsnorm_patched_modules = 0
            if os.getenv("SEU_PPU_RMSNORM_ENABLE", "0") == "1":
                for module_name, module in self._model.named_modules():
                    if (
                        type(module).__name__ != "Qwen3_5RMSNorm"
                        or not module_name.startswith("model.language_model")
                        or module.weight.numel() != 2048
                    ):
                        continue
                    eager_forward = module.forward

                    def decode_rmsnorm(x, *, _module=module, _eager=eager_forward):
                        if x.ndim >= 2 and x.shape[-2] == 1 and x.shape[-1] == 2048:
                            return self._ppu_gdn_library.rmsnorm_decode(
                                x, _module.weight, _module.eps
                            )
                        return _eager(x)

                    module.forward = decode_rmsnorm
                    self._ppu_rmsnorm_patched_modules += 1
                if self._ppu_rmsnorm_patched_modules != 49:
                    raise RuntimeError(
                        "SEU PPU RMSNorm integration expected 49 Qwen3.5 modules, "
                        f"patched {self._ppu_rmsnorm_patched_modules}"
                    )
            self._ppu_gated_rmsnorm_patched_modules = 0
            if os.getenv("SEU_PPU_GATED_RMSNORM_ENABLE", "0") == "1":
                for module in self._model.modules():
                    if type(module).__name__ != "Qwen3_5RMSNormGated":
                        continue
                    eager_forward = module.forward

                    def decode_gated_rmsnorm(
                        hidden_states,
                        gate=None,
                        *,
                        _module=module,
                        _eager=eager_forward,
                    ):
                        if (
                            gate is not None
                            and hidden_states.ndim == 2
                            and hidden_states.shape == (16, 128)
                            and gate.shape == hidden_states.shape
                        ):
                            return self._ppu_gdn_library.gated_rmsnorm_decode(
                                hidden_states,
                                gate,
                                _module.weight,
                                _module.variance_epsilon,
                            )
                        return _eager(hidden_states, gate)

                    module.forward = decode_gated_rmsnorm
                    self._ppu_gated_rmsnorm_patched_modules += 1
                if self._ppu_gated_rmsnorm_patched_modules != 18:
                    raise RuntimeError(
                        "SEU PPU gated RMSNorm expected 18 modules, "
                        f"patched {self._ppu_gated_rmsnorm_patched_modules}"
                    )

    def _load_dummy_backend(self, reason: str) -> None:
        self._dummy_reason = reason

    def _generate_with_transformers(
        self,
        *,
        image,
        prompt: str,
        generation_config: GenerationConfig,
    ) -> GenerationResult:
        import torch
        from transformers import TextIteratorStreamer

        timing = FirstTokenTiming()

        class TimedTextIteratorStreamer(TextIteratorStreamer):
            def put(self, value) -> None:
                timing.observe_stream_put(
                    skip_prompt=self.skip_prompt,
                    next_tokens_are_prompt=self.next_tokens_are_prompt,
                    now=time.perf_counter(),
                )
                super().put(value)

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }]
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)
        input_len = inputs.input_ids.shape[1]
        streamer = TimedTextIteratorStreamer(
            self._processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        generation_kwargs = {
            **inputs,
            "max_new_tokens": generation_config.max_new_tokens,
            "do_sample": generation_config.temperature > 0,
            "use_cache": True,
            "streamer": streamer,
        }
        if generation_config.temperature > 0:
            generation_kwargs.update(
                temperature=generation_config.temperature,
                top_p=generation_config.top_p,
            )

        output_holder: dict[str, Any] = {}

        def _run_generate() -> None:
            context = (
                torch.no_grad
                if self.optimization_profile == "o0_no_grad"
                else torch.inference_mode
            )
            with context():
                output_holder["output_ids"] = self._model.generate(**generation_kwargs)

        worker = threading.Thread(target=_run_generate, daemon=True)
        start = time.perf_counter()
        worker.start()

        chunks: list[str] = []
        for chunk in streamer:
            chunks.append(chunk)
        worker.join()
        end = time.perf_counter()

        output_ids = output_holder["output_ids"]
        generated_ids = output_ids[0][input_len:]
        text = "".join(chunks).strip()
        if not text:
            text = self._processor.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
        markup_normalized_text = normalize_choice_markup(text)
        normalized_text = normalize_explicit_choice_conclusion(
            markup_normalized_text
        )

        ttft = (
            timing.timestamp - start
            if timing.timestamp is not None
            else end - start
        )
        return GenerationResult(
            text=normalized_text,
            token_count=int(generated_ids.shape[0]),
            ttft_seconds=ttft,
            elapsed_seconds=end - start,
            meta={
                "backend": "transformers",
                "choice_markup_normalized": markup_normalized_text != text,
                "choice_conclusion_normalized": (
                    normalized_text != markup_normalized_text
                ),
                "optimization_profile": self.optimization_profile,
                "ttft_measurement": "first_generated_token_put",
                "ppu_gdn_patched_modules": self._ppu_gdn_patched_modules,
                "ppu_conv_patched_modules": getattr(
                    self, "_ppu_conv_patched_modules", 0
                ),
                "ppu_rmsnorm_patched_modules": getattr(
                    self, "_ppu_rmsnorm_patched_modules", 0
                ),
                "ppu_gated_rmsnorm_patched_modules": getattr(
                    self, "_ppu_gated_rmsnorm_patched_modules", 0
                ),
                "ppu_qk_rope_patched_modules": getattr(
                    self, "_ppu_qk_rope_patched_modules", 0
                ),
            },
        )

    def _generate_with_dummy(
        self,
        *,
        prompt: str,
        choices: dict[str, str],
        generation_config: GenerationConfig,
        sample_id: str,
    ) -> GenerationResult:
        start = time.perf_counter()
        usable_choices = [key for key, value in choices.items() if (value or "").strip()]
        picked = usable_choices[hash(sample_id) % len(usable_choices)] if usable_choices else "A"
        text = (
            f"Answer: {picked}\n"
            f"Explanation: dummy backend selected a deterministic option for smoke testing."
        )
        token_count = max(1, min(generation_config.max_new_tokens, len(text.split())))
        end = time.perf_counter()
        return GenerationResult(
            text=text,
            token_count=token_count,
            ttft_seconds=max(end - start, 1e-4),
            elapsed_seconds=max(end - start, 2e-4),
            meta={"backend": "dummy", "reason": getattr(self, "_dummy_reason", "n/a"), "prompt_chars": len(prompt)},
        )
