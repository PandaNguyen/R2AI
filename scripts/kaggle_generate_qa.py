"""Generate grounded legal QA answers with Qwen3 on Kaggle GPU."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from r2ai.qa import fallback_answer
from r2ai.qa.generation import article_numbers_from_refs, load_json_rows, load_jsonl_rows
from r2ai.qa.llm import (
    LLMGenerationConfig,
    TransformersChatLLM,
    TransformersLLMConfig,
    generate_valid_answer,
)
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
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--torch-dtype", default="auto", help="auto, float16, bfloat16, float32, or empty for default")
    parser.add_argument("--device-map", default="auto", help="Transformers device_map; use empty string to disable")
    parser.add_argument("--trust-remote-code", action="store_true")
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

    generation_config = LLMGenerationConfig(
        mode=args.mode,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    llm = TransformersChatLLM.load(
        TransformersLLMConfig(
            model_id=args.model_id,
            generation=generation_config,
            torch_dtype=args.torch_dtype or None,
            device_map=args.device_map or None,
            trust_remote_code=args.trust_remote_code,
        )
    )
    output_rows = []
    started_at = time.monotonic()
    for index, base_row in enumerate(base_rows, start=1):
        row = dict(base_row)
        context_row = context_by_id.get(str(row["id"]))
        if context_row is None:
            allowed_articles = article_numbers_from_refs(row.get("relevant_articles") or [])
            row["answer"] = fallback_answer(allowed_articles)
        else:
            row["answer"] = generate_valid_answer(llm, context_row)
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


if __name__ == "__main__":
    main()
