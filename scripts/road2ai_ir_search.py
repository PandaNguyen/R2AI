#!/usr/bin/env python3
"""Create ROAD2AI-style IR candidate results before answer generation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from r2ai.indexing.config import DEFAULT_DENSE_VECTOR_NAME, DEFAULT_SPARSE_VECTOR_NAME  # noqa: E402
from r2ai.retrieval.qdrant_search import load_query_vectors, load_questions  # noqa: E402
from r2ai.search import SearchQuery, search_many_ir, write_ir_rows  # noqa: E402
from r2ai.search.road2ai_search import (  # noqa: E402
    DEFAULT_BM25_CACHE,
    DEFAULT_EMBED_MODEL,
    DEFAULT_RERANK_MODEL,
    Road2AISearchBackend,
    Road2AISearchConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True, help="JSON/JSONL/CSV questions file.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL IR candidate rows.")
    parser.add_argument("--query-embeddings", type=Path, default=None, help="Optional .npy dense query vectors.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--collection", default=None)
    parser.add_argument("--mode", choices=["dense", "hybrid"], default="hybrid")
    parser.add_argument("--top-k", type=int, default=4, help="Final IR hits after eligibility/dedup/rerank.")
    parser.add_argument("--retrieve-pool", type=int, default=15, help="Candidates per dense/BM25 branch.")
    parser.add_argument("--rrf-top-k", type=int, default=20, help="Candidates kept after RRF before rerank.")
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--dense-model", type=Path, default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--dense-vector-name", default=DEFAULT_DENSE_VECTOR_NAME)
    parser.add_argument("--sparse-vector-name", default=DEFAULT_SPARSE_VECTOR_NAME)
    parser.add_argument("--reranker-model", type=Path, default=DEFAULT_RERANK_MODEL)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--bm25-cache", type=Path, default=DEFAULT_BM25_CACHE)
    parser.add_argument("--device-embed", default="cuda")
    parser.add_argument("--device-rerank", default="cuda")
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    questions = load_questions(args.questions)
    query_vectors = load_query_vectors(args.query_embeddings) if args.query_embeddings else None
    if args.limit is not None:
        questions = questions[: args.limit]
        if query_vectors is not None:
            query_vectors = query_vectors[: args.limit]
    if query_vectors is not None and len(query_vectors) != len(questions):
        raise ValueError(f"questions ({len(questions)}) and query embeddings ({len(query_vectors)}) differ.")

    backend = Road2AISearchBackend(
        Road2AISearchConfig.from_env(
            embed_model_path=args.dense_model,
            collection=args.collection,
            top_k=args.top_k,
            device_embed=args.device_embed,
            vector_name=args.dense_vector_name,
            sparse_vector_name=args.sparse_vector_name,
            use_bm25=args.mode == "hybrid",
            bm25_cache=args.bm25_cache,
            retrieve_pool=args.retrieve_pool,
            rrf_top_k=args.rrf_top_k,
            rrf_k=args.rrf_k,
            use_rerank=not args.no_rerank,
            rerank_model_path=args.reranker_model,
            device_rerank=args.device_rerank,
            enable_subquery=False,
        )
    )
    queries = [
        SearchQuery(
            id=question["id"],
            text=question["question"],
            query_vector=query_vectors[index] if query_vectors is not None else None,
        )
        for index, question in enumerate(questions)
    ]
    rows = search_many_ir(backend, queries, progress_every=args.progress_every)
    write_ir_rows(args.output, rows)
    print(f"Wrote {len(rows)} ROAD2AI-style IR rows to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
