"""Generate grounded legal QA answers with Qwen3 on Kaggle GPU."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from r2ai.qa import (
    build_qa_messages,
    build_repair_messages,
    fallback_answer,
    sanitize_generated_answer,
    validate_answer_articles,
)
from r2ai.qa.generation import article_numbers_from_refs, load_json_rows, load_jsonl_rows
from r2ai.retrieval.qdrant_search import write_submission, write_submission_zip


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate R2AI QA answers with Qwen/Qwen3-8B")
    parser.add_argument("--input-contexts", type=Path, required=True, help="QA context JSONL from submit-qdrant")
    parser.add_argument("--base-results", type=Path, required=True, help="Retrieval-only results.json")
    parser.add_argument("--output", type=Path, default=Path("results.json"))
    parser.add_argument("--zip-output", type=Path, default=None)
    parser.add_argument("--model-id", default="Qwen/Qwen3-8B")
    parser.add_argument("--mode", choices=["no-thinking", "thinking"], default="no-thinking")
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test row limit")
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    args = parse_args()
    base_rows = load_json_rows(args.base_results)
    context_rows = load_jsonl_rows(args.input_contexts)
    if args.limit is not None:
        base_rows = base_rows[: args.limit]
        context_rows = context_rows[: args.limit]
    context_by_id = {str(row["id"]): row for row in context_rows}

    tokenizer, model = load_qwen(args.model_id)
    output_rows = []
    started_at = time.monotonic()
    for index, base_row in enumerate(base_rows, start=1):
        row = dict(base_row)
        context_row = context_by_id.get(str(row["id"]))
        if context_row is None:
            allowed_articles = article_numbers_from_refs(row.get("relevant_articles") or [])
            row["answer"] = fallback_answer(allowed_articles)
        else:
            row["answer"] = generate_valid_answer(tokenizer, model, context_row, args)
        output_rows.append(row)
        if args.progress_every > 0 and (index == 1 or index == len(base_rows) or index % args.progress_every == 0):
            elapsed = time.monotonic() - started_at
            print(f"Generated {index}/{len(base_rows)} answers in {elapsed:.1f}s", flush=True)

    write_submission(args.output, output_rows)
    if args.zip_output:
        write_submission_zip(args.zip_output, args.output)
    print(f"Wrote {len(output_rows)} QA predictions to {args.output}")
    if args.zip_output:
        print(f"Wrote flat submission zip to {args.zip_output}")


def load_qwen(model_id: str) -> tuple[Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Kaggle QA generation requires torch and transformers>=4.51.0. "
            "Install them in the notebook before running this script."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return tokenizer, model


def generate_valid_answer(tokenizer: Any, model: Any, row: dict[str, Any], args: argparse.Namespace) -> str:
    allowed_articles = row.get("allowed_article_numbers") or []
    answer = generate_answer(tokenizer, model, build_qa_messages(row), mode=args.mode, max_new_tokens=args.max_new_tokens)
    if not validate_answer_articles(answer, allowed_articles):
        return answer

    repaired = generate_answer(
        tokenizer,
        model,
        build_repair_messages(row, answer),
        mode=args.mode,
        max_new_tokens=args.max_new_tokens,
    )
    if not validate_answer_articles(repaired, allowed_articles):
        return repaired
    return fallback_answer(allowed_articles)


def generate_answer(
    tokenizer: Any,
    model: Any,
    messages: list[dict[str, str]],
    *,
    mode: str,
    max_new_tokens: int,
) -> str:
    enable_thinking = mode == "thinking"
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generation_kwargs = generation_config(mode)
    output_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        **generation_kwargs,
    )[0][len(model_inputs.input_ids[0]) :].tolist()
    if enable_thinking:
        output_ids = strip_qwen_thinking_ids(output_ids)
    answer = tokenizer.decode(output_ids, skip_special_tokens=True)
    return sanitize_generated_answer(answer)


def generation_config(mode: str) -> dict[str, Any]:
    if mode == "thinking":
        return {"do_sample": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20}
    return {"do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20}


def strip_qwen_thinking_ids(output_ids: list[int]) -> list[int]:
    qwen_think_end_token = 151668
    try:
        index = len(output_ids) - output_ids[::-1].index(qwen_think_end_token)
    except ValueError:
        return output_ids
    return output_ids[index:]


if __name__ == "__main__":
    main()
