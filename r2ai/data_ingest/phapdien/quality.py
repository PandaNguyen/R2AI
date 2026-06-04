"""Data quality report for phapdien build outputs."""

from __future__ import annotations

from collections import Counter
from typing import Any

from r2ai.data_ingest.phapdien.constants import LONG_ARTICLE_CHARS


def build_quality_report(
    articles: list[dict[str, Any]],
    units: list[dict[str, Any]],
    ontology_counts: dict[str, int],
) -> dict[str, Any]:
    canonical_ids = [article["canonical_article_id"] for article in articles]
    anchors = [article["article_anchor"] for article in articles if article["article_anchor"]]
    confidence_counts = Counter(article["citation_confidence"] for article in articles)
    topic_counts = Counter(article["topic_title"] for article in articles)
    subject_counts = Counter(article["subject_title"] for article in articles)

    return {
        "counts": {
            "articles": len(articles),
            "retrieval_units": len(units),
            "ontology_topics": ontology_counts.get("topics", 0),
            "ontology_subjects": ontology_counts.get("subjects", 0),
            "ontology_glossary": ontology_counts.get("glossary", 0),
        },
        "content": {
            "empty_content_articles": sum(1 for article in articles if not article["content_text"]),
            "long_articles_over_3000_chars": sum(
                1 for article in articles if len(article["content_text"]) > LONG_ARTICLE_CHARS
            ),
            "max_chunk_count": max((unit["chunk_count"] for unit in units), default=0),
        },
        "chunking": {
            "method_distribution": dict(Counter(unit.get("chunk_method", "") for unit in units).most_common()),
            "with_tree_path": sum(1 for unit in units if unit.get("content_tree_path")),
        },
        "duplicates": {
            "canonical_article_ids": [item for item, count in Counter(canonical_ids).items() if count > 1],
            "article_anchors": [item for item, count in Counter(anchors).items() if count > 1],
        },
        "citation_confidence": dict(sorted(confidence_counts.items())),
        "topic_distribution": dict(topic_counts.most_common()),
        "subject_distribution_top_50": dict(subject_counts.most_common(50)),
    }
