"""Orchestration for Phase 1 phapdien data build."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from r2ai.data_ingest.phapdien.io import load_csv_rows, load_parquet_rows, write_json, write_jsonl
from r2ai.data_ingest.phapdien.qdrant_payload import make_qdrant_preview
from r2ai.data_ingest.phapdien.quality import build_quality_report
from r2ai.data_ingest.phapdien.records import canonical_hash_basis, canonicalize_article, make_retrieval_units


@dataclass(frozen=True)
class BuildPaths:
    source_dir: Path
    output_dir: Path


def build_phapdien_data(paths: BuildPaths) -> dict[str, Any]:
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    topic_rows = load_csv_rows(paths.source_dir / "ontology_topics.csv")
    subject_rows = load_csv_rows(paths.source_dir / "ontology_subjects.csv")
    glossary_rows = load_csv_rows(paths.source_dir / "ontology_glossary.csv")

    raw_rows = load_parquet_rows(paths.source_dir)
    basis_counts: Counter[str] = Counter()
    articles = []
    for row in raw_rows:
        basis = canonical_hash_basis(row)
        disambiguator = basis_counts[basis]
        basis_counts[basis] += 1
        articles.append(canonicalize_article(row, disambiguator=disambiguator))
    units = [unit for article in articles for unit in make_retrieval_units(article)]
    qdrant_preview = [make_qdrant_preview(unit) for unit in units]

    write_jsonl(paths.output_dir / "articles.jsonl", articles)
    write_jsonl(paths.output_dir / "retrieval_units.jsonl", units)
    write_jsonl(paths.output_dir / "qdrant_payload_preview.jsonl", qdrant_preview)

    report = build_quality_report(
        articles,
        units,
        {
            "topics": len(topic_rows),
            "subjects": len(subject_rows),
            "glossary": len(glossary_rows),
        },
    )
    write_json(paths.output_dir / "data_quality_report.json", report)
    return report
