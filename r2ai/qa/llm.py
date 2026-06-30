"""LLM clients and answer-generation orchestration for grounded QA."""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from r2ai.qa.generation import (
    build_qa_messages,
    build_repair_messages,
    extract_article_numbers,
    fallback_answer,
    normalize_article_number,
    sanitize_generated_answer,
    validate_answer_articles,
)

GenerationMode = Literal["no-thinking", "thinking"]


class ChatLLM(Protocol):
    """Minimal interface used by the QA pipeline."""

    def generate(self, messages: list[dict[str, str]], *, max_new_tokens: int | None = None) -> str:
        """Generate a final answer from chat messages."""


@dataclass(frozen=True)
class LLMGenerationConfig:
    """Decoding options shared across model backends."""

    mode: GenerationMode = "no-thinking"
    max_new_tokens: int = 768
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None

    def sampling_kwargs(self) -> dict[str, Any]:
        temperature = self.temperature
        top_p = self.top_p
        top_k = self.top_k
        if self.mode == "thinking":
            temperature = 0.6 if temperature is None else temperature
            top_p = 0.95 if top_p is None else top_p
            top_k = 20 if top_k is None else top_k
        else:
            temperature = 0.7 if temperature is None else temperature
            top_p = 0.8 if top_p is None else top_p
            top_k = 20 if top_k is None else top_k

        kwargs: dict[str, Any] = {"do_sample": temperature > 0}
        if temperature > 0:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p
        if top_k is not None:
            kwargs["top_k"] = top_k
        return kwargs


@dataclass(frozen=True)
class TransformersLLMConfig:
    """Config for a local Hugging Face causal LM."""

    model_id: str = "Qwen/Qwen3-8B"
    generation: LLMGenerationConfig = field(default_factory=LLMGenerationConfig)
    torch_dtype: str | None = "auto"
    device_map: str | None = "auto"
    trust_remote_code: bool = False
    model_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class CallableChatLLM:
    """Small adapter for tests or external LLM providers."""

    generator: Callable[[list[dict[str, str]]], str]

    def generate(self, messages: list[dict[str, str]], *, max_new_tokens: int | None = None) -> str:
        del max_new_tokens
        return sanitize_generated_answer(str(self.generator(messages)))


class TransformersChatLLM:
    """Hugging Face chat model wrapper used by Kaggle generation."""

    def __init__(
        self,
        tokenizer: Any,
        model: Any,
        *,
        generation: LLMGenerationConfig | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.generation = generation or LLMGenerationConfig()

    @classmethod
    def load(cls, config: TransformersLLMConfig) -> "TransformersChatLLM":
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "QA generation requires torch and transformers. "
                "Install the search/LLM dependencies before running this script."
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(
            config.model_id,
            trust_remote_code=config.trust_remote_code,
        )
        model_kwargs = {
            "trust_remote_code": config.trust_remote_code,
            **config.model_kwargs,
        }
        torch_dtype = _resolve_torch_dtype(config.torch_dtype, torch)
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype
        if config.device_map:
            model_kwargs["device_map"] = config.device_map

        model = AutoModelForCausalLM.from_pretrained(config.model_id, **model_kwargs)
        if hasattr(model, "eval"):
            model.eval()
        if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return cls(tokenizer, model, generation=config.generation)

    def generate(self, messages: list[dict[str, str]], *, max_new_tokens: int | None = None) -> str:
        text = self._format_messages(messages)
        model_inputs = self.tokenizer([text], return_tensors="pt")
        device = getattr(self.model, "device", None)
        if device is not None and hasattr(model_inputs, "to"):
            model_inputs = model_inputs.to(device)

        generation_kwargs = {
            "max_new_tokens": max_new_tokens or self.generation.max_new_tokens,
            "pad_token_id": getattr(self.tokenizer, "pad_token_id", None)
            or getattr(self.tokenizer, "eos_token_id", None),
            **self.generation.sampling_kwargs(),
        }
        try:
            import torch
        except ImportError:
            torch = None
        no_grad = torch.no_grad() if torch is not None else contextlib.nullcontext()
        with no_grad:
            output_ids = self.model.generate(**model_inputs, **generation_kwargs)

        input_ids = model_inputs["input_ids"] if isinstance(model_inputs, dict) else model_inputs.input_ids
        generated_ids = output_ids[0][len(input_ids[0]) :].tolist()
        if self.generation.mode == "thinking":
            generated_ids = strip_qwen_thinking_ids(generated_ids)
        answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return sanitize_generated_answer(answer)

    def _format_messages(self, messages: list[dict[str, str]]) -> str:
        if not hasattr(self.tokenizer, "apply_chat_template"):
            return "\n".join(message["content"] for message in messages)

        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if self.generation.mode in {"thinking", "no-thinking"}:
            kwargs["enable_thinking"] = self.generation.mode == "thinking"
        try:
            return self.tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            return self.tokenizer.apply_chat_template(messages, **kwargs)


def generate_valid_answer(llm: ChatLLM, row: dict[str, Any]) -> str:
    """Generate, repair if needed, and fall back to retrieval-only citations."""
    allowed_articles = row.get("allowed_article_numbers") or []
    answer = llm.generate(build_qa_messages(row))
    if not answer_needs_citation_repair(answer, allowed_articles):
        return answer

    repaired = llm.generate(build_repair_messages(row, answer))
    if not answer_needs_citation_repair(repaired, allowed_articles):
        return repaired
    return fallback_answer(allowed_articles)


def answer_needs_citation_repair(answer: str, allowed_articles: list[str]) -> bool:
    """Require no disallowed articles and at least one allowed article when available."""
    if validate_answer_articles(answer, allowed_articles):
        return True
    allowed = {normalize_article_number(article) for article in allowed_articles}
    allowed.discard("")
    if not allowed:
        return False
    mentioned = set(extract_article_numbers(answer))
    return not bool(allowed & mentioned)


def strip_qwen_thinking_ids(output_ids: list[int]) -> list[int]:
    qwen_think_end_token = 151668
    try:
        index = len(output_ids) - output_ids[::-1].index(qwen_think_end_token)
    except ValueError:
        return output_ids
    return output_ids[index:]


def _resolve_torch_dtype(value: str | None, torch_module: Any) -> Any:
    if value in {None, ""}:
        return None
    if value == "auto":
        return value
    if not isinstance(value, str):
        return value
    if not hasattr(torch_module, value):
        raise ValueError(f"Unknown torch dtype: {value}")
    return getattr(torch_module, value)
