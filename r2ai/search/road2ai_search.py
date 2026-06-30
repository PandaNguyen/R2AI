#!/usr/bin/env python3
"""RAG pipeline: retrieve chunks from Qdrant and answer R2AIStage1 questions."""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import pickle
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover - exercised only in minimal environments
    torch = None


class _MissingDependency:
    def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("ROAD2AI search dependencies are not installed. Install the search extra first.")


try:
    from qdrant_client import QdrantClient
except ImportError:  # pragma: no cover
    QdrantClient = _MissingDependency

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover
    BM25Okapi = _MissingDependency

try:
    from sentence_transformers import CrossEncoder
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    CrossEncoder = _MissingDependency
    SentenceTransformer = _MissingDependency

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # pragma: no cover
    AutoModelForCausalLM = _MissingDependency
    AutoTokenizer = _MissingDependency

from r2ai.search.contracts import SearchHit, SearchQuery, SearchResponse

ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# ---- Standalone helpers copied from the former local modules. ----


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_COLLECTION = "legal_documents"
DEFAULT_EMBED_MODEL = ROOT / "models" / "Vietnamese_Embedding_v2"
VECTOR_SIZE = 1024
_ENV_LOADED = False


def resolve_env_path(env_file: str | Path | None = None) -> Path | None:
    if env_file is not None:
        path = Path(env_file)
        return path if path.is_file() else None
    override = os.environ.get("QDRANT_ENV_FILE")
    if override:
        path = Path(override)
        return path if path.is_file() else None
    default = ROOT / ".env"
    return default if default.is_file() else None


def load_env(env_file: str | Path | None = None, *, force: bool = False) -> None:
    """Load ROAD2AI/.env (or QDRANT_ENV_FILE) if python-dotenv is available."""
    global _ENV_LOADED
    if _ENV_LOADED and not force:
        return
    env_path = resolve_env_path(env_file)
    if not env_path:
        _ENV_LOADED = True
        return
    try:
        from dotenv import load_dotenv

        explicit = env_file is not None or os.environ.get("QDRANT_ENV_FILE")
        load_dotenv(env_path, override=bool(explicit))
    except ImportError:
        pass
    _ENV_LOADED = True


def get_qdrant_url(url: str | None = None) -> str:
    load_env()
    return url or os.environ.get("QDRANT_URL", DEFAULT_QDRANT_URL)


def get_qdrant_api_key(api_key: str | None = None) -> str | None:
    load_env()
    return api_key or os.environ.get("QDRANT_API_KEY") or None


def get_collection_name(collection: str | None = None) -> str:
    load_env()
    return collection or os.environ.get("QDRANT_COLLECTION", DEFAULT_COLLECTION)


def get_vector_name(vector_name: str | None = None) -> str | None:
    """Named vector for multi-vector collections (e.g. Qdrant Cloud 'dense')."""
    load_env()
    resolved = vector_name or os.environ.get("QDRANT_VECTOR_NAME")
    return resolved or None


def get_sparse_vector_name(sparse_vector_name: str | None = None) -> str | None:
    """Named sparse vector for Qdrant Cloud BM25 (e.g. 'bm25')."""
    load_env()
    resolved = sparse_vector_name or os.environ.get("QDRANT_SPARSE_VECTOR_NAME")
    return resolved or None


def get_embed_model_path(path: str | Path | None = None) -> str | Path:
    load_env()
    if isinstance(path, str):
        candidate = Path(path)
        if candidate.is_dir():
            return candidate
        return path
    if path is not None and path.is_dir():
        return path
    env_path = os.environ.get("EMBED_MODEL_PATH")
    if env_path and Path(env_path).is_dir():
        return Path(env_path)
    legacy = ROOT / "models" / "Vietnamese_Embedding_v2"
    if legacy.is_dir():
        return legacy
    return DEFAULT_EMBED_MODEL


def normalize_chunk_payload(payload: dict | None) -> dict:
    """Map Qdrant Cloud (vld_business_law) payloads to the local ingest schema."""
    p = dict(payload or {})
    if p.get("law_code") or p.get("law_title"):
        p.setdefault("text", str(p.get("text") or ""))
        return p

    doc_id = p.get("document_id", p.get("doc_id", ""))
    p.setdefault("doc_id", str(doc_id) if doc_id is not None else "")
    p.setdefault("law_code", str(p.get("document_number") or ""))
    p.setdefault("law_title", str(p.get("document_title") or ""))
    p.setdefault("law_type", str(p.get("legal_type") or ""))
    article = str(p.get("article_no") or p.get("node_label") or "").strip()
    if article.lower().startswith("điều "):
        article = article[5:].strip()
    p.setdefault("article_number", article)
    p.setdefault(
        "text",
        str(
            p.get("retrieval_text")
            or p.get("content_text")
            or p.get("text")
            or ""
        ),
    )
    p.setdefault("url", str(p.get("source_url") or ""))
    p.setdefault("file_name", str(p.get("chunk_id") or p.get("doc_id") or ""))
    return p


def query_dense(
    client: QdrantClient,
    collection: str,
    vector: list[float],
    *,
    limit: int,
    vector_name: str | None = None,
    **kwargs,
):
    using = get_vector_name(vector_name)
    if using:
        return client.query_points(
            collection_name=collection,
            query=vector,
            using=using,
            limit=limit,
            **kwargs,
        )
    return client.query_points(
        collection_name=collection,
        query=vector,
        limit=limit,
        **kwargs,
    )


def query_sparse_bm25(
    client: QdrantClient,
    collection: str,
    query_text: str,
    *,
    limit: int,
    sparse_vector_name: str | None = None,
    **kwargs,
):
    """BM25 sparse search via Qdrant Cloud (pre-indexed sparse vector)."""
    from qdrant_client import models

    using = get_sparse_vector_name(sparse_vector_name)
    if not using:
        raise ValueError("QDRANT_SPARSE_VECTOR_NAME chưa cấu hình (vd. bm25)")
    return client.query_points(
        collection_name=collection,
        query=models.Document(text=query_text, model="Qdrant/bm25"),
        using=using,
        limit=limit,
        **kwargs,
    )


def make_qdrant_client(
    url: str | None = None,
    api_key: str | None = None,
    *,
    timeout: float | None = None,
) -> QdrantClient:
    resolved_url = get_qdrant_url(url)
    resolved_key = get_qdrant_api_key(api_key)
    if timeout is None:
        timeout = 120.0 if resolved_key else 30.0
    if resolved_key:
        return QdrantClient(url=resolved_url, api_key=resolved_key, timeout=timeout)
    return QdrantClient(url=resolved_url, timeout=timeout)


def add_qdrant_args(parser) -> None:
    """Register common Qdrant CLI arguments on an argparse parser."""
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Load Qdrant settings from this .env file (default: ROAD2AI/.env or QDRANT_ENV_FILE)",
    )
    parser.add_argument(
        "--qdrant-url",
        default=None,
        help=f"Qdrant server URL (default: env QDRANT_URL or {DEFAULT_QDRANT_URL})",
    )
    parser.add_argument(
        "--qdrant-api-key",
        default=None,
        help="Qdrant API key (default: env QDRANT_API_KEY; not needed for local Qdrant)",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help=f"Qdrant collection name (default: env QDRANT_COLLECTION or {DEFAULT_COLLECTION})",
    )
    parser.add_argument(
        "--vector-name",
        default=None,
        help="Named dense vector (default: env QDRANT_VECTOR_NAME; omit for local legal_documents)",
    )
    parser.add_argument(
        "--sparse-vector-name",
        default=None,
        help="Named sparse BM25 vector on Qdrant Cloud (default: env QDRANT_SPARSE_VECTOR_NAME)",
    )


def init_qdrant_from_args(args) -> None:
    """Load env file from argparse namespace before resolving Qdrant settings."""
    env_file = getattr(args, "env_file", None)
    if env_file is not None:
        load_env(env_file, force=True)


TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text) if len(t) > 1]


def chunk_key(payload: dict[str, Any]) -> str:
    doc_id = str(payload.get("doc_id", ""))
    chunk_id = str(payload.get("chunk_id", ""))
    if doc_id or chunk_id:
        return f"{doc_id}:{chunk_id}"
    return str(payload.get("point_id", ""))


def payload_from_qdrant(point) -> dict[str, Any]:
    payload = normalize_chunk_payload(point.payload or {})
    payload["point_id"] = str(point.id)
    return payload


def scroll_all_chunks(client: QdrantClient, collection: str) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            payload = payload_from_qdrant(point)
            text = str(payload.get("text") or "").strip()
            if text:
                chunks.append(payload)
        if offset is None:
            break
    return chunks


def load_or_build_bm25_index(
    client: QdrantClient,
    collection: str,
    cache_path: Path | None,
) -> tuple[BM25Okapi, list[dict[str, Any]]]:
    if cache_path and cache_path.is_file():
        with cache_path.open("rb") as f:
            cached = pickle.load(f)
        if cached.get("collection") == collection:
            print(f"  BM25 cache: {cache_path} ({len(cached['corpus'])} chunks)")
            return cached["bm25"], cached["corpus"]

    print(f"  Building BM25 index from Qdrant collection '{collection}'...")
    corpus = scroll_all_chunks(client, collection)
    if not corpus:
        raise RuntimeError(f"Không có chunk trong collection '{collection}'")

    tokenized = [tokenize(c.get("text", "")) for c in corpus]
    bm25 = BM25Okapi(tokenized)
    print(f"  BM25 index: {len(corpus)} documents")

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as f:
            pickle.dump({"collection": collection, "corpus": corpus, "bm25": bm25}, f)
        print(f"  Saved BM25 cache: {cache_path}")

    return bm25, corpus


def bm25_top_k(
    bm25: BM25Okapi,
    corpus: list[dict[str, Any]],
    query: str,
    top_k: int,
) -> list[tuple[dict[str, Any], float]]:
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    return [(corpus[i], float(score)) for i, score in ranked if score > 0]


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float | None]]],
    *,
    rrf_k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse multiple ranked lists keyed by chunk_key. Each item: (key, optional raw score)."""
    fused: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, (key, _) in enumerate(ranked, start=1):
            fused[key] = fused.get(key, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


def chunk_record(
    payload: dict[str, Any],
    *,
    rank: int,
    score: float,
    dense_score: float | None = None,
    bm25_score: float | None = None,
    rerank_score: float | None = None,
) -> dict[str, Any]:
    return {
        "rank": rank,
        "score": score,
        "rrf_score": score,
        "dense_score": dense_score,
        "bm25_score": bm25_score,
        "rerank_score": rerank_score,
        "point_id": str(payload.get("point_id", "")),
        "doc_id": str(payload.get("doc_id", "")),
        "chunk_id": str(payload.get("chunk_id", "")),
        "law_type": payload.get("law_type", ""),
        "law_code": payload.get("law_code", ""),
        "law_title": payload.get("law_title", ""),
        "file_name": payload.get("file_name", ""),
        "article_number": payload.get("article_number", ""),
        "text": payload.get("text", ""),
    }


def chunk_key_from_record(chunk: dict[str, Any]) -> str:
    doc_id = str(chunk.get("doc_id", ""))
    chunk_id = str(chunk.get("chunk_id", ""))
    if doc_id or chunk_id:
        return f"{doc_id}:{chunk_id}"
    point_id = str(chunk.get("point_id", ""))
    if point_id:
        return point_id
    return f"{chunk.get('law_code', '')}:{chunk.get('file_name', '')}:{chunk.get('article_number', '')}"


def merge_hybrid_results(
    chunk_lists: list[list[dict[str, Any]]],
    *,
    rrf_k: int = 60,
    top_k: int,
) -> list[dict[str, Any]]:
    """RRF-merge multiple hybrid retrieval result lists into one ranked list."""
    if not chunk_lists:
        return []
    if len(chunk_lists) == 1:
        return chunk_lists[0][:top_k]

    ranked_lists: list[list[tuple[str, float | None]]] = []
    chunk_by_key: dict[str, dict[str, Any]] = {}

    for chunks in chunk_lists:
        ranked: list[tuple[str, float | None]] = []
        for chunk in chunks:
            key = chunk_key_from_record(chunk)
            chunk_by_key[key] = chunk
            ranked.append((key, chunk.get("score")))
        if ranked:
            ranked_lists.append(ranked)

    if not ranked_lists:
        return []

    fused = reciprocal_rank_fusion(ranked_lists, rrf_k=rrf_k)[:top_k]
    merged: list[dict[str, Any]] = []
    for rank, (key, fused_score) in enumerate(fused, start=1):
        source = chunk_by_key[key]
        merged.append(
            chunk_record(
                source,
                rank=rank,
                score=fused_score,
                dense_score=source.get("dense_score"),
                bm25_score=source.get("bm25_score"),
                rerank_score=source.get("rerank_score"),
            )
        )
    return merged


def hybrid_retrieve_one(
    query: str,
    *,
    dense_hits: list,
    top_k: int,
    pool_size: int,
    rrf_k: int,
    bm25: BM25Okapi | None = None,
    corpus: list[dict[str, Any]] | None = None,
    sparse_hits: list | None = None,
) -> list[dict[str, Any]]:
    dense_ranked: list[tuple[str, float | None]] = []
    payload_by_key: dict[str, dict[str, Any]] = {}
    dense_score_by_key: dict[str, float] = {}

    for hit in dense_hits[:pool_size]:
        payload = normalize_chunk_payload(hit.payload or {})
        payload["point_id"] = str(hit.id)
        key = chunk_key(payload)
        payload_by_key[key] = payload
        dense_score_by_key[key] = float(hit.score)
        dense_ranked.append((key, hit.score))

    bm25_ranked: list[tuple[str, float | None]] = []
    bm25_score_by_key: dict[str, float] = {}

    if sparse_hits is not None:
        for hit in sparse_hits[:pool_size]:
            payload = normalize_chunk_payload(hit.payload or {})
            payload["point_id"] = str(hit.id)
            key = chunk_key(payload)
            payload_by_key[key] = payload
            score = float(hit.score)
            bm25_score_by_key[key] = score
            bm25_ranked.append((key, score))
    elif bm25 is not None and corpus is not None:
        bm25_hits = bm25_top_k(bm25, corpus, query, pool_size)
        for payload, score in bm25_hits:
            key = chunk_key(payload)
            payload_by_key[key] = payload
            bm25_score_by_key[key] = score
            bm25_ranked.append((key, score))
    else:
        raise ValueError("Cần sparse_hits (Qdrant BM25) hoặc bm25+corpus (local index)")

    ranked_lists = [dense_ranked]
    if bm25_ranked:
        ranked_lists.append(bm25_ranked)
    fused = reciprocal_rank_fusion(ranked_lists, rrf_k=rrf_k)[:top_k]

    chunks: list[dict[str, Any]] = []
    for rank, (key, fused_score) in enumerate(fused, start=1):
        payload = payload_by_key[key]
        chunks.append(
            chunk_record(
                payload,
                rank=rank,
                score=fused_score,
                dense_score=dense_score_by_key.get(key),
                bm25_score=bm25_score_by_key.get(key),
            )
        )
    return chunks


MAX_RERANK_LENGTH = 2304


def load_reranker(model_path: str | Path, device: str = "cuda") -> CrossEncoder:
    try:
        return CrossEncoder(str(model_path), max_length=MAX_RERANK_LENGTH, device=device)
    except (ImportError, ValueError) as exc:
        message = str(exc)
        if "SentencePiece" in message or "processing class" in message:
            raise RuntimeError(
                "Could not load the ROAD2AI reranker tokenizer. Install the search extra with "
                "`uv sync --extra search` or install `sentencepiece`, then rerun. "
                "For a quick retrieval-only smoke test, set `road2ai.use_rerank: false` "
                "in configs/ir_main_flow.yaml."
            ) from exc
        raise


def dense_hits_to_chunks(hits) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for rank, hit in enumerate(hits, start=1):
        payload = normalize_chunk_payload(hit.payload or {})
        payload["point_id"] = str(hit.id)
        chunks.append(
            {
                "rank": rank,
                "score": float(hit.score),
                "rrf_score": None,
                "dense_score": float(hit.score),
                "bm25_score": None,
                "rerank_score": None,
                "point_id": str(hit.id),
                "doc_id": str(payload.get("doc_id", "")),
                "chunk_id": str(payload.get("chunk_id", "")),
                "law_type": payload.get("law_type", ""),
                "law_code": payload.get("law_code", ""),
                "law_title": payload.get("law_title", ""),
                "file_name": payload.get("file_name", ""),
                "article_number": payload.get("article_number", ""),
                "text": payload.get("text", ""),
            }
        )
    return chunks


def rerank_chunks(
    query: str,
    chunks: list[dict[str, Any]],
    reranker: CrossEncoder,
    *,
    top_k: int,
    batch_size: int = 32,
) -> list[dict[str, Any]]:
    if not chunks:
        return []

    pairs = [(query, str(c.get("text", ""))) for c in chunks]
    scores = reranker.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    ranked = sorted(zip(chunks, scores), key=lambda x: float(x[1]), reverse=True)[:top_k]

    result: list[dict[str, Any]] = []
    for rank, (chunk, score) in enumerate(ranked, start=1):
        out = dict(chunk)
        if out.get("rrf_score") is None and out.get("bm25_score") is not None:
            out["rrf_score"] = out.get("score")
        elif out.get("rrf_score") is None and out.get("dense_score") is not None and out.get("bm25_score") is None:
            out["rrf_score"] = None
        else:
            out["rrf_score"] = out.get("rrf_score", out.get("score"))
        out["rerank_score"] = float(score)
        out["score"] = float(score)
        out["rank"] = rank
        result.append(out)
    return result


DEFAULT_THRESHOLD_1 = 30
DEFAULT_THRESHOLD_2 = 58

_JSON_ARRAY_RE = re.compile(r"\[[\s\S]*\]")


def load_embed_tokenizer(embed_model_path: Path):
    return AutoTokenizer.from_pretrained(str(embed_model_path))


def count_tokens(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def num_subqueries(
    token_count: int,
    *,
    threshold_1: int = DEFAULT_THRESHOLD_1,
    threshold_2: int = DEFAULT_THRESHOLD_2,
) -> int:
    if token_count < threshold_1:
        return 1
    if token_count < threshold_2:
        return 2
    return 3


def plan_queries(
    question: str,
    tokenizer,
    *,
    threshold_1: int = DEFAULT_THRESHOLD_1,
    threshold_2: int = DEFAULT_THRESHOLD_2,
) -> tuple[int, list[str] | None]:
    """Return (n_subqueries, None) — decomposition filled in later by LLM."""
    n = num_subqueries(
        count_tokens(question, tokenizer),
        threshold_1=threshold_1,
        threshold_2=threshold_2,
    )
    if n == 1:
        return 1, [question]
    return n, None


def _build_decompose_prompt(tokenizer, question: str, n: int) -> str:
    system = (
        "Bạn là chuyên gia pháp luật Việt Nam. "
        f"Tách câu hỏi pháp luật thành đúng {n} sub-query độc lập, "
        "mỗi sub-query tập trung một khía cạnh (điều kiện, mức phạt, thủ tục, thời hạn, v.v.). "
        "Chỉ trả về JSON array các chuỗi, không giải thích."
    )
    user = (
        f"Câu hỏi: {question}\n\n"
        f'Trả về đúng {n} sub-query dạng JSON array, ví dụ: ["sub-query 1", "sub-query 2"]'
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _parse_subqueries(text: str, n: int, fallback: str) -> list[str]:
    text = text.strip()
    match = _JSON_ARRAY_RE.search(text)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                subs = [str(s).strip() for s in parsed if str(s).strip()]
                if subs:
                    return subs[:n] if len(subs) >= n else subs + [fallback] * (n - len(subs))
        except json.JSONDecodeError:
            pass
    lines = [line.strip(" \"'-•") for line in text.splitlines() if line.strip()]
    subs = [line for line in lines if len(line) > 10]
    if len(subs) >= n:
        return subs[:n]
    return [fallback]


def decompose_question(
    question: str,
    n: int,
    model,
    tokenizer,
    eos_ids: list[int],
    *,
    max_new_tokens: int = 256,
) -> list[str]:
    if n <= 1:
        return [question]

    prompt = _build_decompose_prompt(tokenizer, question, n)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=eos_ids,
            pad_token_id=tokenizer.pad_token_id,
        )

    raw = tokenizer.decode(generated[0][prompt_len:], skip_special_tokens=True)
    subs = _parse_subqueries(raw, n, question)
    return subs


def batch_decompose_questions(
    questions: list[dict],
    model,
    tokenizer,
    eos_ids: list[int],
    embed_tokenizer,
    *,
    threshold_1: int = DEFAULT_THRESHOLD_1,
    threshold_2: int = DEFAULT_THRESHOLD_2,
) -> dict[int, list[str]]:
    """Return question id -> list of query strings (including single-query cases)."""
    result: dict[int, list[str]] = {}
    pending: list[tuple[dict, int]] = []

    for q in questions:
        qid = q["id"]
        n, preset = plan_queries(
            q["question"],
            embed_tokenizer,
            threshold_1=threshold_1,
            threshold_2=threshold_2,
        )
        if preset is not None:
            result[qid] = preset
        else:
            pending.append((q, n))

    for q, n in pending:
        subs = decompose_question(
            q["question"],
            n,
            model,
            tokenizer,
            eos_ids,
        )
        result[q["id"]] = subs
        print(f"  Decompose id={q['id']} → {n} sub-query")

    return result


# ---- End standalone helpers. ----

DEFAULT_QUESTIONS = ROOT / "test" / "R2AIStage1DATA.json"
DEFAULT_EMBED_MODEL = get_embed_model_path()
DEFAULT_LLM_MODEL = ROOT / "models" / "Vi-Qwen2-1.5B-RAG"
DEFAULT_RERANK_MODEL = ROOT / "models" / "Vietnamese_Reranker"
DEFAULT_OUTPUT = ROOT / "test" / "R2AIStage1_answers.json"
DEFAULT_RETRIEVED = ROOT / "test" / "R2AIStage1_retrieved.json"
DEFAULT_BM25_CACHE = ROOT / "output" / "bm25_corpus.pkl"

VI_QWEN_SYSTEM = (
    "Bạn là một trợ lí Tiệng Việt nhiệt tình và trung thực. "
    "Hãy luôn trả lời một cách hữu ích nhất có thể."
)
VI_QWEN_RAG_USER = """Chú ý các yêu cầu sau:
- Câu trả lời phải chính xác và đầy đủ nếu ngữ cảnh có câu trả lời.
- Chỉ sử dụng các thông tin có trong ngữ cảnh được cung cấp.
- Viết một đoạn văn liền mạch, trả lời trực tiếp câu hỏi, nêu đủ điều kiện, mức hỗ trợ, thời hạn, trách nhiệm nếu có.
- Không thêm tiêu đề, không thêm mục Kết luận/Phân tích, không liệt kê bullet.
Hãy trả lời câu hỏi dựa trên ngữ cảnh:
### Ngữ cảnh :
{context}

### Câu hỏi :
{question}

### Trả lời :"""

_ARTICLE_RE = re.compile(r"Điều\s+(\d+[a-zA-Z]?)", re.IGNORECASE)
_CODE_RE = re.compile(
    r"\b(\d{1,3}/\d{4}/(?:QH\d+|NĐ-CP|TT-[A-Z]+|QĐ-[A-Z]+|NQ-HĐND))\b",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    p.add_argument(
        "--input-json",
        type=Path,
        default=None,
        help=(
            "Standalone submit interface: JSON object/list with qid and question. "
            "Use '-' to read JSON from stdin. Output remains submit format."
        ),
    )
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--retrieved-cache", type=Path, default=DEFAULT_RETRIEVED)
    p.add_argument("--embed-model", type=Path, default=DEFAULT_EMBED_MODEL)
    p.add_argument("--llm-model", type=Path, default=DEFAULT_LLM_MODEL)
    add_qdrant_args(p)
    p.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Final chunks after rerank",
    )
    p.add_argument(
        "--llm-top-k",
        type=int,
        default=4,
        help="Chunks for LLM context and relevant_articles submission",
    )
    p.add_argument("--retrieve-batch", type=int, default=4)
    p.add_argument("--gen-batch", type=int, default=2, help="LLM generation batch size")
    p.add_argument("--max-context-chars", type=int, default=3500)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--start-id", type=int, default=1)
    p.add_argument("--skip-retrieve", action="store_true")
    p.add_argument("--skip-answered", action="store_true", help="Skip question ids already in output file")
    p.add_argument("--retrieve-only", action="store_true")
    p.add_argument(
        "--citations-only",
        action="store_true",
        help="Chỉ retrieve và xuất id, relevant_docs, relevant_articles (không dùng LLM)",
    )
    p.add_argument("--device-embed", default="cuda")
    p.add_argument("--device-llm", default="cuda")
    p.add_argument(
        "--include-chunks",
        action="store_true",
        help="Include retrieved_chunks in output JSON (debug)",
    )
    p.add_argument(
        "--no-bm25",
        action="store_true",
        help="Disable hybrid retrieval (dense + BM25 + RRF); default is hybrid on",
    )
    p.add_argument(
        "--bm25-cache",
        type=Path,
        default=DEFAULT_BM25_CACHE,
        help="Cache file for BM25 corpus index",
    )
    p.add_argument(
        "--retrieve-pool",
        type=int,
        default=15,
        help="Candidate pool per retrieval method (ANN / BM25) before fusion",
    )
    p.add_argument(
        "--rrf-top-k",
        type=int,
        default=20,
        help="Chunks kept after RRF fusion, before rerank (hybrid only)",
    )
    p.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="RRF constant k for hybrid score fusion",
    )
    p.add_argument(
        "--rerank-model",
        type=Path,
        default=DEFAULT_RERANK_MODEL,
        help="Cross-encoder reranker model path",
    )
    p.add_argument("--device-rerank", default="cuda")
    p.add_argument("--rerank-batch", type=int, default=8)
    p.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable cross-encoder reranking",
    )
    p.add_argument(
        "--no-subquery",
        action="store_true",
        help="Disable sub-query decomposition for long questions",
    )
    p.add_argument(
        "--token-threshold-1",
        type=int,
        default=DEFAULT_THRESHOLD_1,
        help="Token count below this uses 1 query (default: 30)",
    )
    p.add_argument(
        "--token-threshold-2",
        type=int,
        default=DEFAULT_THRESHOLD_2,
        help="Token count below this uses 2 sub-queries; >= uses 3 (default: 58)",
    )
    p.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit quantization for LLM (uses fp16 on CUDA)",
    )
    return p.parse_args()


def _looks_bad_title(title: str) -> bool:
    if not title:
        return True
    stripped = title.strip()
    if not stripped or all(c in "-_ " for c in stripped):
        return True
    lowered = stripped.lower()
    return lowered.startswith("căn cứ") or lowered.startswith("theo ")


def _title_from_file_name(file_name: str, law_code: str) -> str:
    if not file_name:
        return ""
    stem = Path(file_name).stem
    code_token = law_code.replace("/", "_") if law_code else ""
    if code_token and code_token in stem:
        rest = stem.split(code_token, 1)[-1].lstrip("_")
        if rest:
            return rest.replace("_", " ").strip()
    parts = stem.split("_", 2)
    if len(parts) >= 3:
        return parts[2].replace("_", " ").strip()
    return stem.replace("_", " ").strip()


def _title_from_text(text: str, law_code: str) -> str:
    if not text:
        return ""
    head = text[:500]
    if law_code:
        pattern = re.compile(
            rf"{re.escape(law_code)}\s+(.+?)(?:\n|$)",
            re.IGNORECASE,
        )
        match = pattern.search(head)
        if match:
            return match.group(1).strip(" -")
    first_line = head.split("\n", 1)[0].strip()
    if law_code and law_code in first_line:
        return first_line.split(law_code, 1)[-1].strip(" -")
    return first_line


def resolve_law_code(chunk: dict) -> str:
    code = str(chunk.get("law_code") or "").strip()
    if code:
        return code
    text = chunk.get("text") or ""
    match = _CODE_RE.search(text[:400])
    return match.group(1) if match else ""


def resolve_law_title(chunk: dict) -> str:
    title = str(chunk.get("law_title") or "").strip()
    law_code = resolve_law_code(chunk)
    if not _looks_bad_title(title):
        return title
    title = _title_from_file_name(chunk.get("file_name") or "", law_code)
    if not _looks_bad_title(title):
        return title
    return _title_from_text(chunk.get("text") or "", law_code)


def is_valid_chunk(chunk: dict) -> bool:
    law_code = resolve_law_code(chunk)
    law_title = resolve_law_title(chunk)
    return bool(law_code) and bool(law_title) and not _looks_bad_title(law_title)


def filter_valid_chunks(chunks: list[dict], limit: int | None = None) -> list[dict]:
    valid = [c for c in chunks if is_valid_chunk(c)]
    if limit is not None:
        valid = valid[:limit]
    for rank, chunk in enumerate(valid, start=1):
        chunk["rank"] = rank
    return valid


def resolve_article_label(chunk: dict) -> str | None:
    article = str(chunk.get("article_number") or "").strip()
    if article:
        return f"Điều {article}" if not article.lower().startswith("điều") else article
    text = chunk.get("text") or ""
    match = _ARTICLE_RE.search(text[:800])
    if match:
        return f"Điều {match.group(1)}"
    return None


def chunk_to_doc_ref(chunk: dict) -> str | None:
    if not is_valid_chunk(chunk):
        return None
    law_code = resolve_law_code(chunk)
    law_title = resolve_law_title(chunk)
    return f"{law_code}|{law_title}"


def chunk_to_article_ref(chunk: dict) -> str | None:
    if not is_valid_chunk(chunk):
        return None
    law_code = resolve_law_code(chunk)
    law_title = resolve_law_title(chunk)
    article = resolve_article_label(chunk)
    if not article:
        return None
    return f"{law_code}|{law_title}|{article}"


def extract_citations(chunks: list[dict]) -> tuple[list[str], list[str]]:
    relevant_docs: list[str] = []
    relevant_articles: list[str] = []
    seen_docs: set[str] = set()
    seen_articles: set[str] = set()
    for chunk in filter_valid_chunks(chunks):
        doc_ref = chunk_to_doc_ref(chunk)
        if doc_ref and doc_ref not in seen_docs:
            seen_docs.add(doc_ref)
            relevant_docs.append(doc_ref)
        article_ref = chunk_to_article_ref(chunk)
        if article_ref and article_ref not in seen_articles:
            seen_articles.add(article_ref)
            relevant_articles.append(article_ref)
    return relevant_docs, relevant_articles


def build_citation_results(items: list[dict], llm_top_k: int) -> list[dict]:
    results: list[dict] = []
    for item in items:
        relevant_docs, relevant_articles = extract_citations(item["chunks"][:llm_top_k])
        results.append({
            "id": item["id"],
            "question": item.get("question", ""),
            "answer": "",
            "relevant_docs": relevant_docs,
            "relevant_articles": relevant_articles,
        })
    return results


def load_questions(path: Path, start_id: int, limit: int | None) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = [q for q in data if q.get("id", 0) >= start_id]
    if limit:
        items = items[:limit]
    return items



def _coerce_qid(raw_qid):
    if isinstance(raw_qid, str):
        stripped = raw_qid.strip()
        if stripped.isdigit():
            return int(stripped)
        return stripped
    return raw_qid


def normalize_question_input(item: dict) -> dict:
    """Accept external {qid, question} and convert to internal {id, question}."""
    if not isinstance(item, dict):
        raise ValueError("Mỗi input phải là object JSON có qid và question")
    raw_qid = item.get("qid", item.get("id"))
    question = str(item.get("question") or "").strip()
    if raw_qid is None:
        raise ValueError("Thiếu qid trong input JSON")
    if not question:
        raise ValueError(f"Thiếu question cho qid={raw_qid}")
    return {"id": _coerce_qid(raw_qid), "question": question}


def load_interface_questions(path: Path) -> list[dict]:
    """Load standalone JSON input: {qid, question}, a list, or {questions: [...]}"""
    raw = sys.stdin.read() if str(path) == "-" else path.read_text(encoding="utf-8")
    payload = json.loads(raw)
    if isinstance(payload, dict) and "questions" in payload:
        payload = payload["questions"]
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        raise ValueError("Input JSON phải là object {qid, question} hoặc list các object")
    return [normalize_question_input(item) for item in payload]


class RetrievalEngine:
    """Embedding + BM25 + rerank retrieval; supports GPU offload between batches."""

    def __init__(
        self,
        *,
        embed_model_path: Path,
        qdrant_url: str | None,
        collection: str | None,
        top_k: int,
        device: str,
        qdrant_api_key: str | None = None,
        vector_name: str | None = None,
        sparse_vector_name: str | None = None,
        use_bm25: bool = False,
        bm25_cache: Path | None = None,
        retrieve_pool: int = 50,
        rrf_top_k: int = 50,
        rrf_k: int = 60,
        use_rerank: bool = True,
        rerank_model_path: Path | None = None,
        device_rerank: str = "cuda",
        rerank_batch: int = 32,
        enable_subquery: bool = True,
        sub_query_map: dict[int, list[str]] | None = None,
        token_threshold_1: int = DEFAULT_THRESHOLD_1,
        token_threshold_2: int = DEFAULT_THRESHOLD_2,
        embed_tokenizer=None,
    ) -> None:
        self.embed_model_path = embed_model_path
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key
        self.vector_name = vector_name
        self.sparse_vector_name = get_sparse_vector_name(sparse_vector_name)
        self.use_qdrant_bm25 = bool(use_bm25 and self.sparse_vector_name)
        self.collection = get_collection_name(collection)
        self.top_k = top_k
        self.device = device
        self.use_bm25 = use_bm25
        self.bm25_cache = bm25_cache
        self.pool_size = retrieve_pool
        self.rrf_top_k = rrf_top_k
        self.rrf_k = rrf_k
        self.use_rerank = use_rerank
        self.rerank_model_path = rerank_model_path
        self.device_rerank = device_rerank
        self.rerank_batch = rerank_batch
        self.enable_subquery = enable_subquery
        self.sub_query_map = sub_query_map or {}
        self.token_threshold_1 = token_threshold_1
        self.token_threshold_2 = token_threshold_2
        self.embed_tokenizer = embed_tokenizer

        self.fusion_top_k = rrf_top_k if (use_bm25 and use_rerank) else top_k
        self.ann_limit = retrieve_pool if (use_bm25 or use_rerank) else top_k

        self.client: QdrantClient | None = None
        self.embed_model: SentenceTransformer | None = None
        self.reranker = None
        self.bm25 = None
        self.corpus = None
        self._on_gpu = False

    def load(self) -> None:
        if self.client is not None:
            return
        self.client = make_qdrant_client(self.qdrant_url, self.qdrant_api_key)
        if self.use_bm25:
            if self.use_qdrant_bm25:
                bm25_label = f"Qdrant BM25 ({self.sparse_vector_name})"
            else:
                self.bm25, self.corpus = load_or_build_bm25_index(
                    self.client, self.collection, self.bm25_cache
                )
                bm25_label = "local BM25"
            if self.use_rerank:
                print(
                    f"  Hybrid retrieve: dense + {bm25_label} + rerank "
                    f"(pool={self.pool_size}, rrf_top_k={self.rrf_top_k}, "
                    f"top_k={self.top_k}, rrf_k={self.rrf_k})"
                )
            else:
                print(
                    f"  Hybrid retrieve: dense + {bm25_label} "
                    f"(pool={self.pool_size}, top_k={self.top_k}, rrf_k={self.rrf_k})"
                )
        elif self.use_rerank:
            print(f"  Dense retrieve + rerank (pool={self.pool_size}, top_k={self.top_k})")

        self.ensure_on_gpu()

    def ensure_on_gpu(self) -> None:
        if self.embed_model is None:
            self.embed_model = SentenceTransformer(str(self.embed_model_path), device=self.device)
            self.embed_model.max_seq_length = 2048
        elif not self._on_gpu and self.device == "cuda":
            self.embed_model.to(self.device)

        if self.use_rerank and self.reranker is None:
            self.reranker = load_reranker(
                self.rerank_model_path or DEFAULT_RERANK_MODEL,
                self.device_rerank,
            )
        elif (
            self.use_rerank
            and self.reranker is not None
            and not self._on_gpu
            and self.device_rerank == "cuda"
        ):
            self.reranker.model.to(self.device_rerank)

        self._on_gpu = self.device == "cuda" or self.device_rerank == "cuda"

    def offload_gpu(self) -> None:
        if self.embed_model is not None and self.device == "cuda":
            self.embed_model.to("cpu")
        if self.reranker is not None and self.device_rerank == "cuda":
            self.reranker.model.to("cpu")
        self._on_gpu = False
        if self.device == "cuda" or self.device_rerank == "cuda":
            torch.cuda.empty_cache()

    def unload(self) -> None:
        self.offload_gpu()
        self.embed_model = None
        self.reranker = None
        self.client = None
        self.bm25 = None
        self.corpus = None

    def retrieve_one(
        self,
        q_item: dict,
        *,
        query_vectors: dict[str, list[float]] | None = None,
    ) -> dict:
        assert self.client is not None and self.embed_model is not None

        question = q_item["question"]
        qid = q_item["id"]

        if self.enable_subquery and qid in self.sub_query_map:
            queries = self.sub_query_map[qid]
        elif self.enable_subquery and self.embed_tokenizer is not None:
            _, queries = plan_queries(
                question,
                self.embed_tokenizer,
                threshold_1=self.token_threshold_1,
                threshold_2=self.token_threshold_2,
            )
            queries = queries or [question]
        else:
            queries = [question]

        chunk_lists: list[list[dict]] = []
        for query in queries:
            if query_vectors and query in query_vectors:
                vec_list = query_vectors[query]
            else:
                encoded = self.embed_model.encode(
                    [query], normalize_embeddings=True, show_progress_bar=False
                )[0]
                vec_list = encoded.tolist()

            hits = query_dense(
                self.client,
                self.collection,
                vec_list,
                limit=self.ann_limit,
                vector_name=self.vector_name,
            )
            if self.use_bm25:
                sparse_hits = None
                if self.use_qdrant_bm25:
                    sparse_res = query_sparse_bm25(
                        self.client,
                        self.collection,
                        query,
                        limit=self.pool_size,
                        sparse_vector_name=self.sparse_vector_name,
                    )
                    sparse_hits = sparse_res.points
                chunks = hybrid_retrieve_one(
                    query,
                    dense_hits=hits.points,
                    bm25=self.bm25,
                    corpus=self.corpus,
                    sparse_hits=sparse_hits,
                    top_k=self.fusion_top_k,
                    pool_size=self.pool_size,
                    rrf_k=self.rrf_k,
                )
            else:
                chunks = dense_hits_to_chunks(hits.points)[: self.fusion_top_k]
            chunk_lists.append(chunks)

        if len(chunk_lists) > 1:
            chunks = merge_hybrid_results(
                chunk_lists,
                rrf_k=self.rrf_k,
                top_k=self.fusion_top_k,
            )
        else:
            chunks = chunk_lists[0] if chunk_lists else []

        if self.use_rerank and self.reranker is not None:
            chunks = rerank_chunks(
                question,
                chunks,
                self.reranker,
                top_k=len(chunks),
                batch_size=self.rerank_batch,
            )
        chunks = filter_valid_chunks(chunks, limit=self.top_k)

        return {
            "id": qid,
            "question": question,
            "chunks": chunks,
            "sub_queries": queries if len(queries) > 1 else None,
        }

    def retrieve_batch(self, questions: list[dict]) -> list[dict]:
        if not questions:
            return []
        self.load()
        self.ensure_on_gpu()
        assert self.client is not None and self.embed_model is not None

        all_queries: list[str] = []
        query_to_encode: set[str] = set()
        for q_item in questions:
            qid = q_item["id"]
            if self.enable_subquery and qid in self.sub_query_map:
                queries = self.sub_query_map[qid]
            elif self.enable_subquery and self.embed_tokenizer is not None:
                _, queries = plan_queries(
                    q_item["question"],
                    self.embed_tokenizer,
                    threshold_1=self.token_threshold_1,
                    threshold_2=self.token_threshold_2,
                )
                queries = queries or [q_item["question"]]
            else:
                queries = [q_item["question"]]
            all_queries.extend(queries)
            query_to_encode.update(queries)

        vectors_map: dict[str, list[float]] = {}
        if query_to_encode:
            unique_queries = list(query_to_encode)
            vectors = self.embed_model.encode(
                unique_queries, normalize_embeddings=True, show_progress_bar=False
            )
            for query, vec in zip(unique_queries, vectors):
                vectors_map[query] = vec.tolist()

        results: list[dict] = []
        for q_item in questions:
            qid = q_item["id"]
            if self.enable_subquery and qid in self.sub_query_map:
                queries = self.sub_query_map[qid]
            elif self.enable_subquery and self.embed_tokenizer is not None:
                _, queries = plan_queries(
                    q_item["question"],
                    self.embed_tokenizer,
                    threshold_1=self.token_threshold_1,
                    threshold_2=self.token_threshold_2,
                )
                queries = queries or [q_item["question"]]
            else:
                queries = [q_item["question"]]

            per_query_vectors = {q: vectors_map[q] for q in queries if q in vectors_map}
            result = self.retrieve_one(q_item, query_vectors=per_query_vectors)
            results.append(result)
        return results


def build_context(chunks: list[dict], max_chars: int, max_chunks: int | None = None) -> str:
    chunks = filter_valid_chunks(chunks, limit=max_chunks)
    parts = []
    total = 0
    for c in chunks:
        law_code = resolve_law_code(c)
        law_title = resolve_law_title(c)
        article = resolve_article_label(c) or ""
        header = f"[{law_code}|{law_title}|{article}]".rstrip("|")
        block = f"{header}\n{c.get('text', '')}"
        if total + len(block) > max_chars:
            remain = max_chars - total
            if remain > 200:
                parts.append(block[:remain])
            break
        parts.append(block)
        total += len(block) + 4
    return "\n\n---\n\n".join(parts)


def apply_chat_template_safe(tokenizer, messages: list[dict], **kwargs) -> str:
    kwargs.setdefault("tokenize", False)
    kwargs.setdefault("add_generation_prompt", True)
    try:
        return tokenizer.apply_chat_template(messages, **kwargs, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def build_messages(question: str, context: str) -> list[dict[str, str]]:
    user = VI_QWEN_RAG_USER.format(context=context, question=question)
    return [{"role": "system", "content": VI_QWEN_SYSTEM}, {"role": "user", "content": user}]


def build_prompt(tokenizer, question: str, context: str) -> str:
    messages = build_messages(question, context)
    return apply_chat_template_safe(tokenizer, messages)


def clean_answer(text: str) -> str:
    text = text.strip()
    think_close = "</" + "redacted_thinking>"
    think_open = "<" + "redacted_thinking>"
    if think_close in text:
        text = text.split(think_close, 1)[-1].strip()
    text = re.sub(rf"{re.escape(think_open)}.*?{re.escape(think_close)}", "", text, flags=re.DOTALL).strip()
    for marker in ("Câu hỏi:", "### Câu hỏi", "assistant", "Ngữ cảnh pháp luật:", "### Ngữ cảnh"):
        if marker in text:
            text = text.split(marker, 1)[0].strip()
    text = re.sub(r"\s+", " ", text).strip()
    lines = [line.strip() for line in text.splitlines()]
    cleaned: list[str] = []
    for line in lines:
        if line in {"Có", "Không"} and cleaned and cleaned[-1] == line:
            continue
        cleaned.append(line)
    return " ".join(cleaned).strip()


def generate_batch_answers(
    model,
    tokenizer,
    batch_items: list[dict],
    *,
    max_context_chars: int,
    max_new_tokens: int,
    eos_ids: list[int],
    llm_top_k: int,
) -> list[dict]:
    prompts = [
        build_prompt(
            tokenizer,
            item["question"],
            build_context(item["chunks"], max_context_chars, max_chunks=llm_top_k),
        )
        for item in batch_items
    ]
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=6144,
    ).to(model.device)
    prompt_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.1,
            eos_token_id=eos_ids,
            pad_token_id=tokenizer.pad_token_id,
        )

    results: list[dict] = []
    for item, seq in zip(batch_items, generated):
        answer = clean_answer(
            tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
        )
        relevant_docs, relevant_articles = extract_citations(item["chunks"][:llm_top_k])
        results.append({
            "id": item["id"],
            "question": item["question"],
            "answer": answer,
            "relevant_docs": relevant_docs,
            "relevant_articles": relevant_articles,
        })
    return results


@dataclass
class AdaptiveGenBatch:
    """Gen batch size; tự giảm một nửa khi OOM (8→4→2→1)."""

    size: int

    def try_reduce(self) -> bool:
        if self.size <= 1:
            return False
        new_size = max(1, self.size // 2)
        print(f"  OOM gen_batch {self.size} → giảm xuống {new_size}")
        self.size = new_size
        return True


def generate_single_answer_safe(
    model,
    tokenizer,
    item: dict,
    *,
    max_context_chars: int,
    max_new_tokens: int,
    eos_ids: list[int],
    llm_top_k: int,
    device: str,
) -> dict:
    batch_items = [item]
    try:
        return generate_batch_answers(
            model,
            tokenizer,
            batch_items,
            max_context_chars=max_context_chars,
            max_new_tokens=max_new_tokens,
            eos_ids=eos_ids,
            llm_top_k=llm_top_k,
        )[0]
    except torch.cuda.OutOfMemoryError:
        if device == "cuda":
            torch.cuda.empty_cache()
        if llm_top_k > 5:
            reduced_k = max(5, llm_top_k // 2)
            reduced_chars = max(1500, max_context_chars // 2)
            print(
                f"  OOM câu id={item['id']} → "
                f"llm_top_k={reduced_k}, max_context_chars={reduced_chars}"
            )
            return generate_single_answer_safe(
                model,
                tokenizer,
                item,
                max_context_chars=reduced_chars,
                max_new_tokens=max_new_tokens,
                eos_ids=eos_ids,
                llm_top_k=reduced_k,
                device=device,
            )
        if max_new_tokens > 128:
            reduced_tokens = max(128, max_new_tokens // 2)
            print(f"  OOM câu id={item['id']} → max_new_tokens={reduced_tokens}")
            return generate_single_answer_safe(
                model,
                tokenizer,
                item,
                max_context_chars=max_context_chars,
                max_new_tokens=reduced_tokens,
                eos_ids=eos_ids,
                llm_top_k=llm_top_k,
                device=device,
            )
        raise


def generate_items_adaptive(
    model,
    tokenizer,
    items: list[dict],
    *,
    gen_batch: AdaptiveGenBatch,
    max_context_chars: int,
    max_new_tokens: int,
    eos_ids: list[int],
    llm_top_k: int,
    device: str,
) -> list[dict]:
    """Generate answers; giảm gen_batch khi OOM và giữ size mới cho các batch sau."""
    results: list[dict] = []
    idx = 0
    while idx < len(items):
        chunk = items[idx : idx + gen_batch.size]
        try:
            batch_results = generate_batch_answers(
                model,
                tokenizer,
                chunk,
                max_context_chars=max_context_chars,
                max_new_tokens=max_new_tokens,
                eos_ids=eos_ids,
                llm_top_k=llm_top_k,
            )
            results.extend(batch_results)
            idx += len(chunk)
        except torch.cuda.OutOfMemoryError:
            if device == "cuda":
                torch.cuda.empty_cache()
            if len(chunk) > 1 and gen_batch.try_reduce():
                continue
            for item in chunk:
                results.append(
                    generate_single_answer_safe(
                        model,
                        tokenizer,
                        item,
                        max_context_chars=max_context_chars,
                        max_new_tokens=max_new_tokens,
                        eos_ids=eos_ids,
                        llm_top_k=llm_top_k,
                        device=device,
                    )
                )
            idx += len(chunk)
    return results


def load_llm(llm_model_path: Path, device: str, *, load_in_4bit: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(str(llm_model_path))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs: dict = {}
    use_4bit = device == "cuda" and load_in_4bit
    if use_4bit:
        try:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            model_kwargs["device_map"] = "auto"
        except ImportError:
            use_4bit = False

    if device == "cuda" and not use_4bit:
        model_kwargs["torch_dtype"] = torch.bfloat16
        model_kwargs["device_map"] = "auto"
    elif device != "cuda":
        model_kwargs["torch_dtype"] = torch.float32

    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(llm_model_path),
            **model_kwargs,
        )
        if use_4bit:
            print("  LLM: 4-bit quantization enabled")
        elif device == "cuda":
            print("  LLM: bfloat16 on CUDA")
    except (RuntimeError, OSError) as exc:
        if not use_4bit:
            raise
        print(f"  LLM: 4-bit failed ({exc}), falling back to bfloat16")
        model_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
        model = AutoModelForCausalLM.from_pretrained(
            str(llm_model_path),
            **model_kwargs,
        )

    if device != "cuda" or "device_map" not in model_kwargs:
        model = model.to(device)

    eos_ids = [tokenizer.eos_token_id]
    for token in ("", "<|endoftext|>", "<|im_end|>"):
        tid = tokenizer.convert_tokens_to_ids(token)
        if tid is not None and tid != tokenizer.unk_token_id and tid not in eos_ids:
            eos_ids.append(tid)
    return model, tokenizer, eos_ids


def offload_llm(model, device: str) -> None:
    """Move LLM off GPU without reloading from disk."""
    if device == "cuda":
        model.to("cpu")
        torch.cuda.empty_cache()


def unload_llm(model, device: str) -> None:
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


def ensure_llm_on_device(model, device: str) -> None:
    if device == "cuda" and next(model.parameters()).device.type != "cuda":
        model.to(device)


def append_retrieved_cache(cache_path: Path, new_items: list[dict]) -> None:
    if not new_items:
        return
    save_json(cache_path, merge_existing(cache_path, new_items))


def _needs_decompose(questions: list[dict], embed_tokenizer, args) -> bool:
    if args.no_subquery:
        return False
    for q in questions:
        n, preset = plan_queries(
            q["question"],
            embed_tokenizer,
            threshold_1=args.token_threshold_1,
            threshold_2=args.token_threshold_2,
        )
        if preset is None:
            return True
    return False


def decompose_sub_queries(
    questions: list[dict],
    *,
    args: argparse.Namespace,
    embed_model_path: Path,
) -> dict[int, list[str]]:
    embed_tokenizer = load_embed_tokenizer(embed_model_path)
    if not _needs_decompose(questions, embed_tokenizer, args):
        return {}

    print("=== Decompose sub-queries (LLM) ===")
    load_in_4bit = not args.no_4bit
    model, tokenizer, eos_ids = load_llm(args.llm_model, args.device_llm, load_in_4bit=load_in_4bit)
    try:
        sub_map = batch_decompose_questions(
            questions,
            model,
            tokenizer,
            eos_ids,
            embed_tokenizer,
            threshold_1=args.token_threshold_1,
            threshold_2=args.token_threshold_2,
        )
    finally:
        unload_llm(model, args.device_llm)
    return sub_map


def make_retrieval_engine(
    args: argparse.Namespace,
    *,
    sub_query_map: dict[int, list[str]] | None = None,
) -> RetrievalEngine:
    embed_tokenizer = None
    if not args.no_subquery:
        embed_tokenizer = load_embed_tokenizer(args.embed_model)

    return RetrievalEngine(
        embed_model_path=args.embed_model,
        qdrant_url=args.qdrant_url,
        qdrant_api_key=args.qdrant_api_key,
        vector_name=args.vector_name,
        sparse_vector_name=getattr(args, "sparse_vector_name", None),
        collection=args.collection,
        top_k=args.top_k,
        device=args.device_embed,
        use_bm25=not args.no_bm25,
        bm25_cache=args.bm25_cache,
        retrieve_pool=args.retrieve_pool,
        rrf_top_k=args.rrf_top_k,
        rrf_k=args.rrf_k,
        use_rerank=not args.no_rerank,
        rerank_model_path=args.rerank_model,
        device_rerank=args.device_rerank,
        rerank_batch=args.rerank_batch,
        enable_subquery=not args.no_subquery,
        sub_query_map=sub_query_map,
        token_threshold_1=args.token_threshold_1,
        token_threshold_2=args.token_threshold_2,
        embed_tokenizer=embed_tokenizer,
    )


def run_citations_pipeline(
    questions: list[dict],
    *,
    args: argparse.Namespace,
) -> int:
    """Retrieve chunks and export citations only (no LLM)."""
    total = len(questions)
    pipeline_batch = args.retrieve_batch
    engine = make_retrieval_engine(args)

    citation_count = 0
    print(
        f"=== Citations only: retrieve + extract (batch={pipeline_batch}, "
        f"llm_top_k={args.llm_top_k}, hybrid={not args.no_bm25}) ==="
    )
    if (
        args.output is not None
        and not args.skip_answered
        and args.start_id <= 1
        and args.limit is None
    ):
        save_json(args.output, [])
    t0 = time.time()

    try:
        for start in range(0, total, pipeline_batch):
            batch_qs = questions[start : start + pipeline_batch]
            if not args.no_subquery:
                engine.sub_query_map = decompose_sub_queries(
                    batch_qs,
                    args=args,
                    embed_model_path=args.embed_model,
                )
            retrieved_batch = engine.retrieve_batch(batch_qs)
            done_retrieve = min(start + pipeline_batch, total)
            print(f"  Retrieved {done_retrieve}/{total}")

            append_retrieved_cache(args.retrieved_cache, retrieved_batch)

            batch_results = build_citation_results(retrieved_batch, args.llm_top_k)
            citation_count += len(batch_results)

            if args.output is not None and batch_results:
                save_json(args.output, merge_existing(args.output, batch_results))

            done = min(start + pipeline_batch, total)
            print(f"  Done {done}/{total} (retrieve + citations)")
            del retrieved_batch
    finally:
        engine.unload()

    print(f"Citations xong: {citation_count} câu ({time.time() - t0:.1f}s)")
    return citation_count


def run_interleaved_pipeline(
    questions: list[dict],
    *,
    args: argparse.Namespace,
) -> int:
    """Retrieve and generate per batch; only one pipeline batch of chunks in RAM."""
    total = len(questions)
    pipeline_batch = args.retrieve_batch
    engine = make_retrieval_engine(args)
    load_in_4bit = not args.no_4bit

    llm_model = None
    tokenizer = None
    eos_ids: list[int] = []
    answer_count = 0
    gen_batch = AdaptiveGenBatch(args.gen_batch)

    print(
        f"=== Pipeline: retrieve + generate (batch={pipeline_batch}, "
        f"gen_batch={gen_batch.size}, hybrid={not args.no_bm25}) ==="
    )
    if args.output is not None and not args.skip_answered:
        save_json(args.output, [])
    t0 = time.time()

    try:
        for start in range(0, total, pipeline_batch):
            batch_qs = questions[start : start + pipeline_batch]
            if not args.no_subquery:
                engine.sub_query_map = decompose_sub_queries(
                    batch_qs,
                    args=args,
                    embed_model_path=args.embed_model,
                )
            retrieved_batch = engine.retrieve_batch(batch_qs)
            done_retrieve = min(start + pipeline_batch, total)
            print(f"  Retrieved {done_retrieve}/{total}")

            append_retrieved_cache(args.retrieved_cache, retrieved_batch)

            engine.offload_gpu()
            if llm_model is None:
                llm_model, tokenizer, eos_ids = load_llm(
                    args.llm_model, args.device_llm, load_in_4bit=load_in_4bit
                )
            else:
                ensure_llm_on_device(llm_model, args.device_llm)

            gen_start = 0
            while gen_start < len(retrieved_batch):
                gen_items = retrieved_batch[gen_start : gen_start + gen_batch.size]
                batch_results = generate_items_adaptive(
                    llm_model,
                    tokenizer,
                    gen_items,
                    gen_batch=gen_batch,
                    max_context_chars=args.max_context_chars,
                    max_new_tokens=args.max_new_tokens,
                    eos_ids=eos_ids,
                    llm_top_k=args.llm_top_k,
                    device=args.device_llm,
                )
                if args.device_llm == "cuda":
                    torch.cuda.empty_cache()

                for result in batch_results:
                    if args.include_chunks:
                        chunks = next(
                            x["chunks"] for x in gen_items if x["id"] == result["id"]
                        )
                        result["retrieved_chunks"] = chunks
                    answer_count += 1

                if args.output is not None and batch_results:
                    save_json(args.output, merge_existing(args.output, batch_results))
                gen_start += len(gen_items)

            done = min(start + pipeline_batch, total)
            print(f"  Done {done}/{total} (retrieve + generate)")
            del retrieved_batch

            if llm_model is not None:
                offload_llm(llm_model, args.device_llm)
    finally:
        engine.unload()
        if llm_model is not None:
            unload_llm(llm_model, args.device_llm)

    print(f"Pipeline xong: {answer_count} câu ({time.time() - t0:.1f}s)")
    return answer_count


def generate_answers(
    retrieved: list[dict],
    *,
    llm_model_path: Path,
    max_context_chars: int,
    max_new_tokens: int,
    llm_top_k: int,
    device: str,
    gen_batch: int,
    output_path: Path | None = None,
    include_chunks: bool = False,
    fresh_output: bool = False,
    load_in_4bit: bool = True,
) -> list[dict]:
    model, tokenizer, eos_ids = load_llm(llm_model_path, device, load_in_4bit=load_in_4bit)

    outputs: list[dict] = []
    total = len(retrieved)
    adaptive_batch = AdaptiveGenBatch(gen_batch)
    if fresh_output and output_path is not None:
        save_json(output_path, [])

    start = 0
    while start < total:
        batch_items = retrieved[start : start + adaptive_batch.size]
        batch_results = generate_items_adaptive(
            model,
            tokenizer,
            batch_items,
            gen_batch=adaptive_batch,
            max_context_chars=max_context_chars,
            max_new_tokens=max_new_tokens,
            eos_ids=eos_ids,
            llm_top_k=llm_top_k,
            device=device,
        )
        if device == "cuda":
            torch.cuda.empty_cache()
        for result in batch_results:
            if include_chunks:
                chunks = next(x["chunks"] for x in batch_items if x["id"] == result["id"])
                result["retrieved_chunks"] = chunks
            outputs.append(result)

        done = min(start + len(batch_items), total)
        print(f"  Generated {done}/{total}")
        if output_path is not None:
            if fresh_output:
                existing = json.loads(output_path.read_text(encoding="utf-8")) if output_path.exists() else []
                existing.extend(batch_results)
                save_json(output_path, existing)
            else:
                save_json(output_path, merge_existing(output_path, outputs))
        start += len(batch_items)

    unload_llm(model, device)
    return outputs




def retrieve_submit_items(
    questions: list[dict],
    *,
    args: argparse.Namespace,
) -> list[dict]:
    engine = make_retrieval_engine(args)
    retrieved: list[dict] = []
    try:
        total = len(questions)
        for start in range(0, total, args.retrieve_batch):
            batch_qs = questions[start : start + args.retrieve_batch]
            if not args.no_subquery:
                engine.sub_query_map = decompose_sub_queries(
                    batch_qs,
                    args=args,
                    embed_model_path=args.embed_model,
                )
            retrieved_batch = engine.retrieve_batch(batch_qs)
            retrieved.extend(retrieved_batch)
            if args.retrieved_cache is not None and str(args.retrieved_cache) != "-":
                append_retrieved_cache(args.retrieved_cache, retrieved_batch)
            print(f"  Retrieved {min(start + args.retrieve_batch, total)}/{total}")
    finally:
        engine.unload()
    return retrieved


def run_submit_interface(questions: list[dict], *, args: argparse.Namespace) -> list[dict]:
    """Return submit-format rows for external {qid, question} JSON input."""
    if args.skip_retrieve and args.retrieved_cache.exists():
        cached = json.loads(args.retrieved_cache.read_text(encoding="utf-8"))
        ids = {q["id"] for q in questions}
        retrieved = [item for item in cached if item.get("id") in ids]
        by_id = {item["id"]: item for item in retrieved}
        retrieved = [by_id[q["id"]] for q in questions if q["id"] in by_id]
        print(f"Dùng cache retrieve: {len(retrieved)} câu")
    else:
        retrieved = retrieve_submit_items(questions, args=args)

    if args.citations_only or args.retrieve_only:
        outputs = build_citation_results(retrieved, args.llm_top_k)
    else:
        outputs = generate_answers(
            retrieved,
            llm_model_path=args.llm_model,
            max_context_chars=args.max_context_chars,
            max_new_tokens=args.max_new_tokens,
            llm_top_k=args.llm_top_k,
            device=args.device_llm,
            gen_batch=args.gen_batch,
            output_path=None,
            include_chunks=args.include_chunks,
            fresh_output=False,
            load_in_4bit=not args.no_4bit,
        )
    return outputs


def write_submit_json(path: Path, data: list[dict]) -> None:
    payload = json.dumps(data, ensure_ascii=False, indent=2)
    if str(path) == "-":
        print(payload)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")

def save_json(path: Path, data: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def merge_existing(output_path: Path, new_items: list[dict]) -> list[dict]:
    if not output_path.exists():
        return new_items
    existing = json.loads(output_path.read_text(encoding="utf-8"))
    by_id = {x["id"]: x for x in existing}
    for item in new_items:
        by_id[item["id"]] = item
    return [by_id[k] for k in sorted(by_id)]



@dataclass(frozen=True)
class Road2AISearchConfig:
    """Native config for exposing the copied ROAD2AI retrieval flow as SearchBackend."""

    embed_model_path: str | Path = DEFAULT_EMBED_MODEL
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    collection: str | None = None
    top_k: int = 4
    device_embed: str = "cuda"
    vector_name: str | None = None
    sparse_vector_name: str | None = None
    use_bm25: bool = True
    bm25_cache: Path | None = DEFAULT_BM25_CACHE
    retrieve_pool: int = 15
    rrf_top_k: int = 20
    rrf_k: int = 60
    use_rerank: bool = True
    rerank_model_path: str | Path | None = DEFAULT_RERANK_MODEL
    device_rerank: str = "cuda"
    rerank_batch: int = 8
    enable_subquery: bool = False
    sub_query_map: dict[Any, list[str]] | None = None
    token_threshold_1: int = DEFAULT_THRESHOLD_1
    token_threshold_2: int = DEFAULT_THRESHOLD_2
    embed_tokenizer: Any | None = None

    @classmethod
    def from_env(
        cls,
        *,
        embed_model_path: str | Path | None = None,
        qdrant_url: str | None = None,
        qdrant_api_key: str | None = None,
        collection: str | None = None,
        top_k: int = 4,
        device_embed: str = "cuda",
        vector_name: str | None = None,
        sparse_vector_name: str | None = None,
        use_bm25: bool = True,
        bm25_cache: Path | None = DEFAULT_BM25_CACHE,
        retrieve_pool: int = 15,
        rrf_top_k: int = 20,
        rrf_k: int = 60,
        use_rerank: bool = True,
        rerank_model_path: str | Path | None = DEFAULT_RERANK_MODEL,
        device_rerank: str = "cuda",
        rerank_batch: int = 8,
        enable_subquery: bool = False,
        sub_query_map: dict[Any, list[str]] | None = None,
        token_threshold_1: int = DEFAULT_THRESHOLD_1,
        token_threshold_2: int = DEFAULT_THRESHOLD_2,
        embed_tokenizer: Any | None = None,
    ) -> "Road2AISearchConfig":
        return cls(
            embed_model_path=get_embed_model_path(embed_model_path),
            qdrant_url=qdrant_url or get_qdrant_url(None),
            qdrant_api_key=qdrant_api_key or get_qdrant_api_key(None),
            collection=collection or get_collection_name(None),
            top_k=top_k,
            device_embed=device_embed,
            vector_name=get_vector_name(vector_name),
            sparse_vector_name=get_sparse_vector_name(sparse_vector_name),
            use_bm25=use_bm25,
            bm25_cache=bm25_cache,
            retrieve_pool=retrieve_pool,
            rrf_top_k=rrf_top_k,
            rrf_k=rrf_k,
            use_rerank=use_rerank,
            rerank_model_path=rerank_model_path,
            device_rerank=device_rerank,
            rerank_batch=rerank_batch,
            enable_subquery=enable_subquery,
            sub_query_map=sub_query_map,
            token_threshold_1=token_threshold_1,
            token_threshold_2=token_threshold_2,
            embed_tokenizer=embed_tokenizer,
        )

    def make_engine(self) -> RetrievalEngine:
        return RetrievalEngine(
            embed_model_path=self.embed_model_path,
            qdrant_url=self.qdrant_url,
            qdrant_api_key=self.qdrant_api_key,
            vector_name=self.vector_name,
            sparse_vector_name=self.sparse_vector_name,
            collection=self.collection,
            top_k=self.top_k,
            device=self.device_embed,
            use_bm25=self.use_bm25,
            bm25_cache=self.bm25_cache,
            retrieve_pool=self.retrieve_pool,
            rrf_top_k=self.rrf_top_k,
            rrf_k=self.rrf_k,
            use_rerank=self.use_rerank,
            rerank_model_path=self.rerank_model_path,
            device_rerank=self.device_rerank,
            rerank_batch=self.rerank_batch,
            enable_subquery=self.enable_subquery,
            sub_query_map=self.sub_query_map,
            token_threshold_1=self.token_threshold_1,
            token_threshold_2=self.token_threshold_2,
            embed_tokenizer=self.embed_tokenizer,
        )


class Road2AISearchBackend:
    """SearchBackend wrapper around the copied ROAD2AI RetrievalEngine."""

    def __init__(self, config: Road2AISearchConfig, *, engine: RetrievalEngine | None = None) -> None:
        self.config = config
        self.engine = engine

    def search(self, query: SearchQuery) -> SearchResponse:
        engine = self._engine()
        q_item = {"id": query.id, "question": query.text}
        query_vectors = {query.text: query.query_vector} if query.query_vector is not None else None
        retrieved = engine.retrieve_one(q_item, query_vectors=query_vectors)
        chunks = retrieved.get("chunks") or []
        return SearchResponse(
            query=query,
            hits=[chunk_to_search_hit(chunk) for chunk in chunks],
            backend="road2ai",
            metadata={
                "collection_name": getattr(engine, "collection", self.config.collection),
                "top_k": self.config.top_k,
                "retrieve_pool": self.config.retrieve_pool,
                "rrf_top_k": self.config.rrf_top_k,
                "rrf_k": self.config.rrf_k,
                "use_bm25": self.config.use_bm25,
                "use_rerank": self.config.use_rerank,
                "used_precomputed_dense_vector": query.query_vector is not None,
                "sub_queries": retrieved.get("sub_queries"),
            },
        )

    def search_many(self, queries: list[SearchQuery]) -> list[SearchResponse]:
        engine = self._engine()
        questions = [{"id": query.id, "question": query.text} for query in queries]
        if any(query.query_vector is not None for query in queries):
            return [self.search(query) for query in queries]
        retrieved_items = engine.retrieve_batch(questions)
        by_id = {item.get("id"): item for item in retrieved_items}
        responses = []
        for query in queries:
            retrieved = by_id.get(query.id, {"chunks": []})
            responses.append(
                SearchResponse(
                    query=query,
                    hits=[chunk_to_search_hit(chunk) for chunk in retrieved.get("chunks", [])],
                    backend="road2ai",
                    metadata={
                        "collection_name": getattr(engine, "collection", self.config.collection),
                        "top_k": self.config.top_k,
                        "retrieve_pool": self.config.retrieve_pool,
                        "rrf_top_k": self.config.rrf_top_k,
                        "rrf_k": self.config.rrf_k,
                        "use_bm25": self.config.use_bm25,
                        "use_rerank": self.config.use_rerank,
                        "used_precomputed_dense_vector": False,
                        "sub_queries": retrieved.get("sub_queries"),
                    },
                )
            )
        return responses

    def unload(self) -> None:
        if self.engine is not None:
            self.engine.unload()

    def _engine(self) -> RetrievalEngine:
        if self.engine is None:
            self.engine = self.config.make_engine()
            self.engine.load()
        return self.engine


def chunk_to_search_hit(chunk: dict[str, Any]) -> SearchHit:
    doc_ref = chunk_to_doc_ref(chunk)
    article_ref = chunk_to_article_ref(chunk)
    payload = chunk_to_search_payload(chunk, doc_ref=doc_ref, article_ref=article_ref)
    score = chunk.get("score")
    return SearchHit(
        id=chunk.get("point_id") or chunk.get("chunk_id"),
        rank=chunk.get("rank"),
        score=float(score) if isinstance(score, int | float) else None,
        text=str(chunk.get("text") or ""),
        payload=payload,
        doc_refs=[doc_ref] if doc_ref else [],
        article_refs=[article_ref] if article_ref else [],
        metadata={key: value for key, value in chunk.items() if key != "text"},
    )


def chunk_to_search_payload(
    chunk: dict[str, Any],
    *,
    doc_ref: str | None,
    article_ref: str | None,
) -> dict[str, Any]:
    law_code = resolve_law_code(chunk)
    law_title = resolve_law_title(chunk)
    article = resolve_article_label(chunk) or ""
    payload = {
        "point_id": chunk.get("point_id", ""),
        "doc_id": chunk.get("doc_id", ""),
        "chunk_id": chunk.get("chunk_id", ""),
        "law_type": chunk.get("law_type", ""),
        "law_code": law_code,
        "law_title": law_title,
        "file_name": chunk.get("file_name", ""),
        "article_number": chunk.get("article_number", ""),
        "text": chunk.get("text", ""),
        "retrieval_text": chunk.get("text", ""),
        "document_number": law_code,
        "document_title": law_title,
        "article_no": article,
        "competition_law_id": law_code,
        "competition_doc_title_type1": law_title,
        "competition_article_no": article,
        "source_law_id_candidates": [law_code] if law_code else [],
        "source_doc_title_candidates": [law_title] if law_title else [],
        "source_article_no_candidates": [article] if article else [],
    }
    if doc_ref:
        payload["doc_ref"] = doc_ref
    if article_ref:
        payload["article_ref"] = article_ref
    return payload

def main() -> int:
    args = parse_args()
    init_qdrant_from_args(args)
    args.embed_model = get_embed_model_path(args.embed_model)

    if args.input_json is not None:
        questions = load_interface_questions(args.input_json)
        if args.limit:
            questions = questions[: args.limit]
        if str(args.output) == "-":
            with contextlib.redirect_stdout(sys.stderr):
                print(f"Input JSON: {len(questions)} câu")
                outputs = run_submit_interface(questions, args=args)
                print(f"Đã xuất submit format: stdout ({len(outputs)} câu)")
            write_submit_json(args.output, outputs)
        else:
            print(f"Input JSON: {len(questions)} câu")
            outputs = run_submit_interface(questions, args=args)
            write_submit_json(args.output, outputs)
            print(f"Đã xuất submit format: {args.output} ({len(outputs)} câu)")
        return 0

    if not args.questions.exists():
        print(f"Không tìm thấy: {args.questions}", file=sys.stderr)
        return 1

    questions = load_questions(args.questions, args.start_id, args.limit)
    if args.skip_answered and args.output.exists():
        answered_ids = {
            item["id"] for item in json.loads(args.output.read_text(encoding="utf-8"))
        }
        before = len(questions)
        questions = [q for q in questions if q["id"] not in answered_ids]
        print(f"Bỏ qua {before - len(questions)} câu đã trả lời, còn {len(questions)}")
    print(f"Câu hỏi: {len(questions)} (start_id={args.start_id})")
    if not questions:
        print("Không còn câu hỏi cần xử lý.")
        return 0

    if args.citations_only and args.skip_retrieve and args.retrieved_cache.exists():
        retrieved = json.loads(args.retrieved_cache.read_text(encoding="utf-8"))
        ids = {q["id"] for q in questions}
        retrieved = [r for r in retrieved if r["id"] in ids]
        print(f"Dùng cache retrieve: {len(retrieved)} câu")
        print("=== Extract citations (from cache, no LLM) ===")
        t1 = time.time()
        citations = build_citation_results(retrieved, args.llm_top_k)
        del retrieved
        if args.skip_answered:
            merged = merge_existing(args.output, citations)
            save_json(args.output, merged)
            print(f"Đã lưu: {args.output} ({len(merged)} câu, {time.time()-t1:.1f}s)")
        else:
            save_json(args.output, citations)
            print(f"Đã lưu: {args.output} ({len(citations)} câu, {time.time()-t1:.1f}s)")
        return 0

    if args.citations_only:
        citation_count = run_citations_pipeline(questions, args=args)
        print(f"Đã lưu: {args.output} ({citation_count} câu)")
        return 0

    if args.retrieve_only:
        print("=== Retrieve only (incremental cache) ===")
        t0 = time.time()
        engine = make_retrieval_engine(args)
        try:
            total = len(questions)
            for start in range(0, total, args.retrieve_batch):
                batch_qs = questions[start : start + args.retrieve_batch]
                if not args.no_subquery:
                    engine.sub_query_map = decompose_sub_queries(
                        batch_qs,
                        args=args,
                        embed_model_path=args.embed_model,
                    )
                retrieved_batch = engine.retrieve_batch(batch_qs)
                append_retrieved_cache(args.retrieved_cache, retrieved_batch)
                done = min(start + args.retrieve_batch, total)
                print(f"  Retrieved {done}/{total}")
                del retrieved_batch
        finally:
            engine.unload()
        print(f"Đã lưu retrieve cache: {args.retrieved_cache} ({time.time()-t0:.1f}s)")
        return 0

    if args.skip_retrieve and args.retrieved_cache.exists():
        retrieved = json.loads(args.retrieved_cache.read_text(encoding="utf-8"))
        ids = {q["id"] for q in questions}
        retrieved = [r for r in retrieved if r["id"] in ids]
        print(f"Dùng cache retrieve: {len(retrieved)} câu")
        print("=== Generate answers (from cache) ===")
        t1 = time.time()
        answers = generate_answers(
            retrieved,
            llm_model_path=args.llm_model,
            max_context_chars=args.max_context_chars,
            max_new_tokens=args.max_new_tokens,
            llm_top_k=args.llm_top_k,
            device=args.device_llm,
            gen_batch=args.gen_batch,
            output_path=args.output,
            include_chunks=args.include_chunks,
            fresh_output=not args.skip_answered,
            load_in_4bit=not args.no_4bit,
        )
        del retrieved
        if args.skip_answered:
            merged = merge_existing(args.output, answers)
            save_json(args.output, merged)
            print(f"Đã lưu: {args.output} ({len(merged)} câu, {time.time()-t1:.1f}s)")
        else:
            print(f"Đã lưu: {args.output} ({len(answers)} câu, {time.time()-t1:.1f}s)")
        return 0

    answer_count = run_interleaved_pipeline(questions, args=args)
    print(f"Đã lưu: {args.output} ({answer_count} câu)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
