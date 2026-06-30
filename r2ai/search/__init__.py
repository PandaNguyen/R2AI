"""Stable search/answer interfaces built beside the existing retrieval stack."""

from r2ai.search.contracts import (
    AnswerResult,
    ContextBlock,
    SearchBackend,
    SearchHit,
    SearchQuery,
    SearchResponse,
)
from r2ai.search.pipeline import SearchAnswerPipeline, SearchPipelineConfig
from r2ai.search.qdrant_backend import QdrantSearchBackend
from r2ai.search.ir_results import response_to_ir_row, search_many_ir, write_ir_rows
from r2ai.search.ir_main_flow import (
    DEFAULT_IR_MAIN_FLOW_CONFIG_PATH,
    IRMainFlowConfig,
    load_ir_main_flow_config,
    run_ir_main_flow,
)
from r2ai.search.road2ai_search import Road2AISearchBackend, Road2AISearchConfig

__all__ = [
    "AnswerResult",
    "ContextBlock",
    "DEFAULT_IR_MAIN_FLOW_CONFIG_PATH",
    "IRMainFlowConfig",
    "QdrantSearchBackend",
    "Road2AISearchBackend",
    "Road2AISearchConfig",
    "SearchAnswerPipeline",
    "SearchBackend",
    "SearchHit",
    "SearchPipelineConfig",
    "SearchQuery",
    "SearchResponse",
    "load_ir_main_flow_config",
    "response_to_ir_row",
    "run_ir_main_flow",
    "search_many_ir",
    "write_ir_rows",
]
