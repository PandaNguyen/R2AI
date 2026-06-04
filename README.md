# R2AI Phapdien Baseline

Phase 1 builds canonical article/chunk artifacts from the local
`data/phapdien-moj-gov-vn` corpus. It does not call Qdrant Cloud and does not
create embeddings.

## Setup

```powershell
uv sync --extra data
```

## Build Phase 1 Data

```powershell
.\.venv\Scripts\python.exe main.py build-phapdien-data --output-dir build
```

Outputs:

- `build/articles.jsonl`: canonical article rows.
- `build/retrieval_units.jsonl`: article/chunk retrieval units.
- `build/qdrant_payload_preview.jsonl`: Qdrant point payload preview without vectors.
- `build/data_quality_report.json`: counts and validation diagnostics.

The default retrieval unit is one legal article. Articles longer than 3000
characters are split into smaller chunks, but every chunk keeps
`canonical_article_id` so later retrieval can group results back to article-level
citations.

## Package Layout

- `r2ai/data_ingest/phapdien/`: phapdien loaders, citation parsing, chunking,
  canonical records, Qdrant payload previews, and data-quality reports.
- `r2ai/indexing/`: reserved for Qdrant collection creation, embedding, sparse
  setup, and upsert code.
- `r2ai/retrieval/`: reserved for hybrid search, grouping chunks to articles,
  and reranking.
- `r2ai/qa/`: reserved for grounded answer generation.
- `r2ai/evaluation/`: reserved for schema checks, citation extraction, and
  submission diagnostics.
- `r2ai/pipelines/`: reserved for end-to-end commands that compose modules.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```
