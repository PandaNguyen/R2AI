"""Config-driven CLI flow for retrieval-only IR candidate generation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from r2ai.indexing.config import DEFAULT_DENSE_VECTOR_NAME, DEFAULT_SPARSE_VECTOR_NAME
from r2ai.qa.generation import article_numbers_from_refs, fallback_answer
from r2ai.qa.llm import LLMGenerationConfig, TransformersChatLLM, TransformersLLMConfig, generate_valid_answer
from r2ai.retrieval.qdrant_search import load_query_vectors, load_questions
from r2ai.search.contracts import SearchBackend, SearchQuery, SearchResponse
from r2ai.search.ir_results import response_to_ir_row, write_ir_rows
from r2ai.search.pipeline import build_context_blocks, collect_references
from r2ai.search.road2ai_search import (
    DEFAULT_BM25_CACHE,
    DEFAULT_EMBED_MODEL,
    DEFAULT_RERANK_MODEL,
    Road2AISearchBackend,
    Road2AISearchConfig,
    load_embed_tokenizer,
)


DEFAULT_IR_MAIN_FLOW_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "ir_main_flow.yaml"

_FLOW_KEYS = {"backend", "query_embeddings", "limit", "progress_every", "ir_output"}
_QA_KEYS = {"answer_mode", "context_limit", "max_context_chars", "progress_every", "llm"}
_ROAD2AI_KEYS = {
    "embed_model_path",
    "qdrant_url",
    "qdrant_api_key",
    "collection",
    "mode",
    "top_k",
    "device_embed",
    "vector_name",
    "sparse_vector_name",
    "use_bm25",
    "bm25_cache",
    "retrieve_pool",
    "rrf_top_k",
    "rrf_k",
    "use_rerank",
    "rerank_model_path",
    "device_rerank",
    "rerank_batch",
    "enable_subquery",
    "sub_query_map",
    "token_threshold_1",
    "token_threshold_2",
}


@dataclass(frozen=True)
class IRMainFlowConfig:
    """Settings for reading questions, running search, and writing IR rows."""

    backend: str = "road2ai"
    query_embeddings: Path | None = None
    limit: int | None = None
    progress_every: int = 25
    ir_output: Path | None = None
    road2ai: dict[str, Any] = field(default_factory=dict)
    qa: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "IRMainFlowConfig":
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise TypeError("IR main-flow config must be a mapping.")

        flow = data.get("flow", {})
        if flow is None:
            flow = {}
        if not isinstance(flow, dict):
            raise TypeError("config.flow must be a mapping.")
        unknown_flow = set(flow) - _FLOW_KEYS
        if unknown_flow:
            raise ValueError(f"Unknown config.flow keys: {sorted(unknown_flow)}")

        road2ai = data.get("road2ai", {})
        if road2ai is None:
            road2ai = {}
        if not isinstance(road2ai, dict):
            raise TypeError("config.road2ai must be a mapping.")
        unknown_road2ai = set(road2ai) - _ROAD2AI_KEYS
        if unknown_road2ai:
            raise ValueError(f"Unknown config.road2ai keys: {sorted(unknown_road2ai)}")

        qa = data.get("qa", {})
        if qa is None:
            qa = {}
        if not isinstance(qa, dict):
            raise TypeError("config.qa must be a mapping.")
        unknown_qa = set(qa) - _QA_KEYS
        if unknown_qa:
            raise ValueError(f"Unknown config.qa keys: {sorted(unknown_qa)}")

        return cls(
            backend=str(flow.get("backend", "road2ai")),
            query_embeddings=_optional_path(flow.get("query_embeddings"), "flow.query_embeddings"),
            limit=_optional_int(flow.get("limit"), "flow.limit"),
            progress_every=int(flow.get("progress_every", 25)),
            ir_output=_optional_path(flow.get("ir_output"), "flow.ir_output"),
            road2ai=dict(road2ai),
            qa=dict(qa),
        )

    def build_backend(self) -> SearchBackend:
        if self.backend != "road2ai":
            raise ValueError(f"Unsupported IR backend: {self.backend!r}. Supported backends: road2ai.")
        return Road2AISearchBackend(build_road2ai_search_config(self.road2ai))


def load_ir_main_flow_config(path: Path | None = None) -> IRMainFlowConfig:
    """Load YAML or JSON config for the IR main flow."""

    config_path = path or DEFAULT_IR_MAIN_FLOW_CONFIG_PATH
    if not config_path.exists():
        if path is None or config_path == DEFAULT_IR_MAIN_FLOW_CONFIG_PATH:
            return IRMainFlowConfig()
        raise FileNotFoundError(f"IR main-flow config not found: {config_path}")
    return IRMainFlowConfig.from_mapping(_load_config_mapping(config_path))


def build_road2ai_search_config(settings: dict[str, Any]) -> Road2AISearchConfig:
    mode = str(settings.get("mode", "hybrid")).strip().lower()
    if mode not in {"dense", "hybrid"}:
        raise ValueError("road2ai.mode must be either 'dense' or 'hybrid'.")
    if "mode" in settings and "use_bm25" in settings:
        raise ValueError("Use either road2ai.mode or road2ai.use_bm25, not both.")

    use_bm25 = bool(settings.get("use_bm25", mode == "hybrid"))
    embed_model_path = _model_setting(settings, "embed_model_path", DEFAULT_EMBED_MODEL, "road2ai.embed_model_path")
    bm25_cache = _path_setting(settings, "bm25_cache", DEFAULT_BM25_CACHE, "road2ai.bm25_cache")
    rerank_model_path = _model_setting(settings, "rerank_model_path", DEFAULT_RERANK_MODEL, "road2ai.rerank_model_path")

    enable_subquery = bool(settings.get("enable_subquery", False))
    sub_query_map_path = _optional_path(settings.get("sub_query_map"), "road2ai.sub_query_map")
    sub_query_map = load_sub_query_map(sub_query_map_path) if sub_query_map_path else None
    embed_tokenizer = load_embed_tokenizer(embed_model_path) if enable_subquery else None

    return Road2AISearchConfig.from_env(
        embed_model_path=embed_model_path,
        qdrant_url=_optional_str(settings.get("qdrant_url")),
        qdrant_api_key=_optional_str(settings.get("qdrant_api_key")),
        collection=_optional_str(settings.get("collection")),
        top_k=int(settings.get("top_k", 4)),
        device_embed=str(settings.get("device_embed", "cuda")),
        vector_name=_optional_str(settings.get("vector_name", DEFAULT_DENSE_VECTOR_NAME)),
        sparse_vector_name=_optional_str(settings.get("sparse_vector_name", DEFAULT_SPARSE_VECTOR_NAME)),
        use_bm25=use_bm25,
        bm25_cache=bm25_cache,
        retrieve_pool=int(settings.get("retrieve_pool", 15)),
        rrf_top_k=int(settings.get("rrf_top_k", 20)),
        rrf_k=int(settings.get("rrf_k", 60)),
        use_rerank=bool(settings.get("use_rerank", True)),
        rerank_model_path=rerank_model_path,
        device_rerank=str(settings.get("device_rerank", "cuda")),
        rerank_batch=int(settings.get("rerank_batch", 8)),
        enable_subquery=enable_subquery,
        sub_query_map=sub_query_map,
        token_threshold_1=int(settings.get("token_threshold_1", 30)),
        token_threshold_2=int(settings.get("token_threshold_2", 58)),
        embed_tokenizer=embed_tokenizer,
    )


def run_ir_main_flow(
    questions_path: Path,
    output_path: Path,
    *,
    config: IRMainFlowConfig | None = None,
    backend: SearchBackend | None = None,
) -> int:
    """Read questions, run retrieval, write IR candidate rows, and return row count."""

    flow_config = config or load_ir_main_flow_config()
    questions = load_questions(questions_path)
    query_vectors = load_query_vectors(flow_config.query_embeddings) if flow_config.query_embeddings else None

    if flow_config.limit is not None:
        questions = questions[: flow_config.limit]
        if query_vectors is not None:
            query_vectors = query_vectors[: flow_config.limit]
    if query_vectors is not None and len(query_vectors) != len(questions):
        raise ValueError(f"questions ({len(questions)}) and query embeddings ({len(query_vectors)}) differ.")

    queries = [
        SearchQuery(
            id=question["id"],
            text=question["question"],
            query_vector=query_vectors[index] if query_vectors is not None else None,
        )
        for index, question in enumerate(questions)
    ]
    search_backend = backend or flow_config.build_backend()
    try:
        responses = search_many_responses(search_backend, queries, progress_every=flow_config.progress_every)
    finally:
        unload = getattr(search_backend, "unload", None)
        if callable(unload):
            unload()

    if flow_config.ir_output is not None:
        write_ir_rows(flow_config.ir_output, [response_to_ir_row(response) for response in responses])
    llm = load_qa_llm(flow_config.qa)
    rows = responses_to_submission_rows(responses, flow_config.qa, llm=llm)
    write_submission_rows(output_path, rows)
    return len(rows)


def search_many_responses(
    backend: SearchBackend,
    queries: list[SearchQuery],
    *,
    progress_every: int = 0,
) -> list[SearchResponse]:
    search_many = getattr(backend, "search_many", None)
    if callable(search_many):
        return list(search_many(queries))

    responses: list[SearchResponse] = []
    for index, query in enumerate(queries, start=1):
        responses.append(backend.search(query))
        if progress_every > 0 and index % progress_every == 0:
            print(f"Searched {index} queries", flush=True)
    return responses


def responses_to_submission_rows(
    responses: list[SearchResponse],
    qa_config: dict[str, Any] | None = None,
    *,
    llm: Any | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    progress_every = int((qa_config or {}).get("progress_every", 10))
    for index, response in enumerate(responses, start=1):
        rows.append(response_to_submission_row(response, qa_config, llm=llm))
        if llm is not None and progress_every > 0 and (index == 1 or index == len(responses) or index % progress_every == 0):
            print(f"Generated QA {index}/{len(responses)}", flush=True)
    return rows


def response_to_submission_row(
    response: SearchResponse,
    qa_config: dict[str, Any] | None = None,
    *,
    llm: Any | None = None,
) -> dict[str, Any]:
    qa_row = response_to_qa_context_row(response, qa_config)
    answer_mode = str((qa_config or {}).get("answer_mode", "empty"))
    if answer_mode == "empty":
        answer = ""
    elif answer_mode == "retrieval":
        answer = fallback_answer(qa_row["allowed_article_numbers"])
    elif answer_mode == "llm":
        if llm is None:
            raise RuntimeError("qa.answer_mode is 'llm' but no LLM is loaded.")
        answer = generate_valid_answer(llm, qa_row)
    else:
        raise ValueError("qa.answer_mode must be 'empty', 'retrieval', or 'llm'.")
    return {
        "answer": answer,
        "id": response.query.id,
        "question": response.query.text,
        "relevant_articles": qa_row["relevant_articles"],
        "relevant_docs": qa_row["relevant_docs"],
    }


def response_to_qa_context_row(response: SearchResponse, qa_config: dict[str, Any] | None = None) -> dict[str, Any]:
    relevant_docs, relevant_articles = collect_references(response.hits)
    context_limit = _optional_int((qa_config or {}).get("context_limit"), "qa.context_limit")
    max_context_chars = int((qa_config or {}).get("max_context_chars", 3500))
    contexts = build_context_blocks(response.hits, max_chars=max_context_chars, limit=context_limit)
    allowed_article_numbers = article_numbers_from_refs(relevant_articles)
    return {
        "id": response.query.id,
        "question": response.query.text,
        "relevant_docs": relevant_docs,
        "relevant_articles": relevant_articles,
        "allowed_article_numbers": allowed_article_numbers,
        "contexts": [context.as_qa_context() for context in contexts],
    }


def load_qa_llm(qa_config: dict[str, Any] | None) -> TransformersChatLLM | None:
    config = qa_config or {}
    if str(config.get("answer_mode", "empty")) != "llm":
        return None
    llm_config = config.get("llm") or {}
    if not isinstance(llm_config, dict):
        raise TypeError("qa.llm must be a mapping.")
    generation_config = llm_config.get("generation") or {}
    if not isinstance(generation_config, dict):
        raise TypeError("qa.llm.generation must be a mapping.")
    generation = LLMGenerationConfig(
        mode=str(generation_config.get("mode", "no-thinking")),
        max_new_tokens=int(generation_config.get("max_new_tokens", 768)),
        temperature=generation_config.get("temperature"),
        top_p=generation_config.get("top_p"),
        top_k=generation_config.get("top_k"),
    )
    model_kwargs = llm_config.get("model_kwargs") or {}
    if not isinstance(model_kwargs, dict):
        raise TypeError("qa.llm.model_kwargs must be a mapping.")
    return TransformersChatLLM.load(
        TransformersLLMConfig(
            model_id=str(llm_config.get("model_id", "Qwen/Qwen3-8B")),
            generation=generation,
            torch_dtype=llm_config.get("torch_dtype", "auto") or None,
            device_map=llm_config.get("device_map", "auto") or None,
            trust_remote_code=bool(llm_config.get("trust_remote_code", False)),
            model_kwargs=model_kwargs,
        )
    )


def write_submission_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_config_mapping(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - depends on installed extras
            raise RuntimeError("PyYAML is required to read YAML config files. Install with: pip install PyYAML") from exc
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    elif suffix == ".json":
        loaded = json.loads(path.read_text(encoding="utf-8"))
    else:
        raise ValueError(f"Unsupported config file type: {path.suffix}. Use .yaml, .yml, or .json.")

    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise TypeError(f"Config file must contain a mapping: {path}")
    return loaded


def load_sub_query_map(path: Path) -> dict[Any, list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: dict[Any, list[str]] = {}
    if isinstance(payload, dict):
        for qid, queries in payload.items():
            _add_sub_query_entry(result, qid, queries)
    elif isinstance(payload, list):
        for index, row in enumerate(payload):
            if not isinstance(row, dict):
                raise TypeError(f"sub_query_map row {index} must be an object.")
            qid = row.get("id", row.get("qid"))
            queries = row.get("sub_queries", row.get("queries"))
            _add_sub_query_entry(result, qid, queries)
    else:
        raise TypeError("sub_query_map must be a JSON object or a list of rows.")
    return result


def _add_sub_query_entry(result: dict[Any, list[str]], qid: Any, queries: Any) -> None:
    if qid is None:
        raise ValueError("sub_query_map entry is missing id/qid.")
    if not isinstance(queries, list):
        raise TypeError(f"sub_query_map[{qid!r}] must be a list of strings.")
    normalized = [str(query).strip() for query in queries if str(query).strip()]
    if not normalized:
        return
    result[qid] = normalized
    result[str(qid)] = normalized
    if isinstance(qid, str) and qid.isdigit():
        result[int(qid)] = normalized


def _optional_path(value: Any, field_name: str) -> Path | None:
    if value in (None, ""):
        return None
    if isinstance(value, Path):
        return value
    if isinstance(value, str):
        return Path(value)
    raise TypeError(f"{field_name} must be a string path or null.")


def _model_setting(settings: dict[str, Any], key: str, default: str | Path | None, field_name: str) -> str | Path | None:
    value = settings.get(key)
    if value in (None, ""):
        return default
    if isinstance(value, Path):
        return value
    if isinstance(value, str):
        candidate = Path(value)
        if candidate.exists() or value.startswith(".") or value.startswith("/"):
            return candidate
        return value
    raise TypeError(f"{field_name} must be a model id, string path, or null.")


def _path_setting(settings: dict[str, Any], key: str, default: Path | None, field_name: str) -> Path | None:
    value = settings.get(key)
    if value in (None, ""):
        return default
    return _optional_path(value, field_name)


def _optional_int(value: Any, field_name: str) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _optional_str(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)
