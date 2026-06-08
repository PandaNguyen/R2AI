"""Experimental filtered retrieval submitter.

This script is intentionally separate from the main CLI. It explores a
multi-branch retrieval strategy for the SME legal QA setting:

1. Run the current global search.
2. Detect coarse legal domains from question keywords.
3. Run extra filtered searches by likely topic_title and law_id.
4. Merge and rerank the candidates before writing Challenge-format output.

Use this for experiments only; keep `main.py submit-qdrant` as the stable
baseline path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from r2ai.cli import load_env_file
from r2ai.indexing.config import (
    DEFAULT_COLLECTION,
    DEFAULT_DOC_TITLE_FORMAT,
    DEFAULT_PREFETCH_LIMIT,
    DEFAULT_SEARCH_QDRANT_TIMEOUT,
    QdrantSearchConfig,
)
from r2ai.indexing.qdrant_ingest import _load_sparse_model, _make_qdrant_client
from r2ai.indexing.retry import is_retryable_request_error
from r2ai.retrieval.qdrant_search import (
    append_submission_checkpoint,
    format_competition_row,
    load_submission_checkpoint,
    load_query_vectors,
    load_questions,
    payload_to_competition_articles,
    payload_to_competition_docs,
    search_qdrant,
    write_submission,
    write_submission_zip,
)


DOMAIN_RULES: dict[str, dict[str, Any]] = {
    "doanh_nghiep_dnnvv": {
        "keywords": ["doanh nghiệp", "dnnvv", "nhỏ và vừa", "sme", "công ty", "hộ kinh doanh", "hợp tác xã"],
        "topics": ["Doanh nghiệp, hợp tác xã"],
        "law_ids": ["59/2020/QH14", "01/2021/NĐ-CP", "04/2017/QH14", "80/2021/NĐ-CP", "07/VBHN-VPQH"],
    },
    "thue_phi": {
        "keywords": ["thuế", "phí", "lệ phí", "hóa đơn", "doanh thu", "thu nhập doanh nghiệp", "gtgt", "vat"],
        "topics": ["Thuế, phí, lệ phí, các khoản thu khác", "Kế toán, kiểm toán"],
        "law_ids": ["38/2019/QH14", "126/2020/NĐ-CP", "123/2020/NĐ-CP", "125/2020/NĐ-CP", "03/VBHN-VPQH"],
    },
    "lao_dong_bhxh": {
        "keywords": [
            "lao động",
            "người lao động",
            "nhân viên",
            "hợp đồng lao động",
            "bằng cấp",
            "chứng chỉ",
            "bản chính",
            "bảo hiểm xã hội",
            "bhxh",
            "tiền lương",
            "sa thải",
            "kỷ luật",
        ],
        "topics": ["Lao động", "Bảo hiểm"],
        "law_ids": ["45/2019/QH14", "145/2020/NĐ-CP", "12/2022/NĐ-CP", "58/2014/QH13", "14/VBHN-VPQH"],
    },
    "hop_dong_dan_su": {
        "keywords": ["hợp đồng", "giao dịch", "thỏa thuận", "bồi thường", "phạt vi phạm", "dân sự"],
        "topics": ["Dân sự", "Thương mại, đầu tư, chứng khoán"],
        "law_ids": ["91/2015/QH13", "36/2005/QH11", "17/VBHN-VPQH", "54/2010/QH12"],
    },
    "dau_thau": {
        "keywords": ["đấu thầu", "nhà thầu", "gói thầu", "dự thầu", "mở thầu"],
        "topics": ["Thương mại, đầu tư, chứng khoán"],
        "law_ids": ["22/2023/QH15", "24/2024/NĐ-CP", "95/2020/NĐ-CP"],
    },
    "ke_toan_tai_chinh": {
        "keywords": ["kế toán", "tài chính", "vốn", "khoản vay", "tín dụng", "quỹ"],
        "topics": ["Kế toán, kiểm toán", "Tài chính", "Ngân hàng, tiền tệ"],
        "law_ids": ["200/2014/TT-BTC", "133/2016/TT-BTC", "04/VBHN-BTC", "39/2019/NĐ-CP", "34/2018/NĐ-CP"],
    },
    "so_huu_tri_tue_cong_nghe": {
        "keywords": ["sở hữu trí tuệ", "công nghệ", "chuyển giao công nghệ", "đổi mới sáng tạo", "nhãn hiệu"],
        "topics": ["Khoa học, công nghệ"],
        "law_ids": ["50/2005/QH11", "11/VBHN-VPQH", "65/2023/NĐ-CP", "07/2022/QH15"],
    },
    "dat_dai_mat_bang": {
        "keywords": ["đất", "mặt bằng", "thuê đất", "sử dụng đất", "nhà xưởng"],
        "topics": ["Tài nguyên", "Đất đai"],
        "law_ids": ["31/2021/NĐ-CP", "10/2024/NĐ-CP", "35/2022/NĐ-CP"],
    },
}


BRANCH_WEIGHTS = {
    "global": 1.0,
    "law": 0.85,
    "topic": 0.65,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experimental filtered Qdrant submission builder")
    parser.add_argument("--questions", type=Path, default=Path("data/R2AIStage1DATA.json"))
    parser.add_argument("--query-embeddings", type=Path, default=Path("data/question_embeddings.npy"))
    parser.add_argument("--output", type=Path, default=Path("results_experiment_filters.json"))
    parser.add_argument("--zip-output", type=Path, default=None)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--doc-title-format", choices=["type1", "type2"], default=DEFAULT_DOC_TITLE_FORMAT)
    parser.add_argument("--top-k", type=int, default=5, help="Final number of submitted references per question")
    parser.add_argument("--global-top-k", type=int, default=8, help="Global branch retrieval depth")
    parser.add_argument("--branch-top-k", type=int, default=3, help="Filtered branch retrieval depth")
    parser.add_argument("--prefetch-limit", type=int, default=DEFAULT_PREFETCH_LIMIT)
    parser.add_argument("--qdrant-timeout", type=float, default=DEFAULT_SEARCH_QDRANT_TIMEOUT)
    parser.add_argument("--limit", type=int, default=20, help="Smoke-test question limit; pass 0 for all")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--max-domains", type=int, default=2)
    parser.add_argument("--max-law-branches", type=int, default=3)
    parser.add_argument("--max-topic-branches", type=int, default=1)
    parser.add_argument("--no-law-branches", action="store_true")
    parser.add_argument("--no-topic-branches", action="store_true")
    parser.add_argument("--answer-article-limit", type=int, default=None)
    parser.add_argument("--debug-output", type=Path, default=None)
    parser.add_argument(
        "--checkpoint-output",
        type=Path,
        default=None,
        help="JSONL checkpoint for completed question predictions; defaults to <output>.checkpoint.jsonl",
    )
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    load_env_file()
    args = parse_args()

    questions = load_questions(args.questions)
    query_vectors = load_query_vectors(args.query_embeddings)
    if len(questions) != len(query_vectors):
        raise ValueError(f"questions ({len(questions)}) and query embeddings ({len(query_vectors)}) differ.")
    if args.limit and args.limit > 0:
        questions = questions[: args.limit]
        query_vectors = query_vectors[: args.limit]
    checkpoint_output = None if args.no_resume else args.checkpoint_output or args.output.with_name(f"{args.output.name}.checkpoint.jsonl")
    completed_rows = load_submission_checkpoint(checkpoint_output)
    if completed_rows:
        print(f"Resuming experiment: {len(completed_rows)} question ids already checkpointed", flush=True)
    pending_questions = []
    pending_vectors = []
    for question, vector in zip(questions, query_vectors, strict=True):
        if str(question["id"]) in completed_rows:
            continue
        pending_questions.append(question)
        pending_vectors.append(vector)

    qdrant_url = os.getenv("QDRANT_URL", "").strip()
    qdrant_api_key = os.getenv("QDRANT_API_KEY", "").strip()
    if not qdrant_url or not qdrant_api_key:
        raise RuntimeError("Missing QDRANT_URL or QDRANT_API_KEY.")
    client, models = _make_qdrant_client(qdrant_url, qdrant_api_key, timeout=args.qdrant_timeout)
    sparse_model = _load_sparse_model("Qdrant/bm25", None)

    base_config = QdrantSearchConfig.from_env(
        query_text="placeholder",
        collection_name=args.collection,
        search_mode="hybrid",
        top_k=args.global_top_k,
        prefetch_limit=args.prefetch_limit,
        qdrant_timeout=args.qdrant_timeout,
        doc_title_format=args.doc_title_format,
        answer_article_limit=args.answer_article_limit,
    )

    rows_by_id: dict[str, dict[str, Any]] = dict(completed_rows)
    debug_rows: list[dict[str, Any]] = []
    started_at = time.monotonic()
    for index, (question, vector) in enumerate(zip(pending_questions, pending_vectors, strict=True), start=1):
        domains = detect_domains(question["question"], max_domains=args.max_domains)
        candidates: dict[str, dict[str, Any]] = {}

        global_result = run_branch(
            base_config,
            question,
            vector,
            client,
            models,
            sparse_model,
            branch_name="global",
            weight=BRANCH_WEIGHTS["global"],
        )
        merge_candidates(candidates, global_result["results"], "global", BRANCH_WEIGHTS["global"], args.doc_title_format)

        branch_descriptions = [{"kind": "global", "value": "", "count": len(global_result["results"])}]
        for branch in build_filtered_branches(args, domains):
            branch_config = replace(
                base_config,
                top_k=args.branch_top_k,
                source_law_id=branch["law_id"],
                topic_title=branch["topic_title"],
            )
            branch_result = run_branch(
                branch_config,
                question,
                vector,
                client,
                models,
                sparse_model,
                branch_name=branch["kind"],
                weight=branch["weight"],
            )
            merge_candidates(
                candidates,
                branch_result["results"],
                branch["kind"],
                branch["weight"],
                args.doc_title_format,
            )
            branch_descriptions.append(
                {
                    "kind": branch["kind"],
                    "value": branch["law_id"] or branch["topic_title"],
                    "count": len(branch_result["results"]),
                }
            )

        result = {
            "doc_title_format": args.doc_title_format,
            "answer_article_limit": args.answer_article_limit,
            "results": rerank_candidates(candidates, limit=args.top_k),
        }
        row = format_competition_row(question, result)
        rows_by_id[str(question["id"])] = row
        append_submission_checkpoint(checkpoint_output, row)
        debug_rows.append(
            {
                "id": question["id"],
                "domains": domains,
                "branches": branch_descriptions,
                "candidate_count": len(candidates),
                "submitted_articles": row["relevant_articles"],
            }
        )

        if args.progress_every > 0 and (
            index == 1 or index == len(pending_questions) or index % args.progress_every == 0
        ):
            elapsed = time.monotonic() - started_at
            print(f"Processed {index}/{len(pending_questions)} pending questions in {elapsed:.1f}s", flush=True)

    rows = [rows_by_id[str(question["id"])] for question in questions if str(question["id"]) in rows_by_id]
    write_submission(args.output, rows)
    if args.zip_output:
        write_submission_zip(args.zip_output, args.output)
    if args.debug_output:
        args.debug_output.write_text(json.dumps(debug_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} experimental predictions to {args.output}")


def detect_domains(question: str, max_domains: int) -> list[str]:
    text = question.lower()
    scored = []
    for domain, rule in DOMAIN_RULES.items():
        hits = sum(1 for keyword in rule["keywords"] if keyword in text)
        if hits:
            scored.append((hits, domain))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [domain for _, domain in scored[:max_domains]]


def build_filtered_branches(args: argparse.Namespace, domains: list[str]) -> list[dict[str, Any]]:
    branches = []
    seen_laws = set()
    seen_topics = set()
    if not args.no_law_branches:
        domain_laws = [DOMAIN_RULES[domain]["law_ids"] for domain in domains]
        max_laws_per_domain = max((len(laws) for laws in domain_laws), default=0)
        for law_index in range(max_laws_per_domain):
            for laws in domain_laws:
                if law_index >= len(laws):
                    continue
                law_id = laws[law_index]
                if law_id in seen_laws:
                    continue
                seen_laws.add(law_id)
                branches.append({"kind": "law", "law_id": law_id, "topic_title": None, "weight": BRANCH_WEIGHTS["law"]})
                if len(seen_laws) >= args.max_law_branches:
                    break
            if len(seen_laws) >= args.max_law_branches:
                break
    if not args.no_topic_branches:
        for domain in domains:
            for topic_title in DOMAIN_RULES[domain]["topics"]:
                if topic_title in seen_topics:
                    continue
                seen_topics.add(topic_title)
                branches.append(
                    {"kind": "topic", "law_id": None, "topic_title": topic_title, "weight": BRANCH_WEIGHTS["topic"]}
                )
                if len(seen_topics) >= args.max_topic_branches:
                    break
            if len(seen_topics) >= args.max_topic_branches:
                break
    return branches


def run_branch(
    config: QdrantSearchConfig,
    question: dict[str, Any],
    vector: list[float],
    client: Any,
    models: Any,
    sparse_model: Any,
    *,
    branch_name: str,
    weight: float,
) -> dict[str, Any]:
    try:
        result = search_qdrant(
            replace(config, query_text=question["question"], query_vector=vector),
            sparse_model=sparse_model,
            client=client,
            models=models,
        )
    except Exception as exc:
        if not is_retryable_request_error(exc):
            raise
        print(
            f"Skipping {branch_name} branch for question {question['id']} after exhausted request retries: {exc}",
            flush=True,
        )
        result = {
            "doc_title_format": config.doc_title_format,
            "answer_article_limit": config.answer_article_limit,
            "results": [],
        }
    result["branch_name"] = branch_name
    result["branch_weight"] = weight
    return result


def merge_candidates(
    candidates: dict[str, dict[str, Any]],
    results: list[dict[str, Any]],
    branch_name: str,
    weight: float,
    doc_title_format: str,
) -> None:
    for rank, item in enumerate(results, start=1):
        payload = item.get("payload") or {}
        articles = payload_to_competition_articles(payload, doc_title_format=doc_title_format)
        docs = payload_to_competition_docs(payload, doc_title_format=doc_title_format)
        key = articles[0] if articles else (docs[0] if docs else str(item.get("id")))
        candidate = candidates.setdefault(
            key,
            {
                "id": item.get("id"),
                "payload": payload,
                "score": 0.0,
                "branches": [],
                "best_rank": rank,
                "dense_score": item.get("dense_score"),
                "sparse_score": item.get("sparse_score"),
                "hybrid_score": item.get("hybrid_score") or item.get("score"),
            },
        )
        candidate["score"] += weight / (60 + rank)
        candidate["best_rank"] = min(candidate["best_rank"], rank)
        candidate["branches"].append({"name": branch_name, "rank": rank, "weight": weight})


def rerank_candidates(candidates: dict[str, dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    ranked = sorted(
        candidates.values(),
        key=lambda item: (
            -item["score"],
            item["best_rank"],
            str(item["id"]),
        ),
    )
    output = []
    for rank, item in enumerate(ranked[:limit], start=1):
        output.append(
            {
                "id": item["id"],
                "rank": rank,
                "score": item["score"],
                "payload": item["payload"],
                "branches": item["branches"],
            }
        )
    return output


if __name__ == "__main__":
    main()
