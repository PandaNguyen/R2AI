# VLD Pipeline Refactor Notes

## Context

Phapdien data is now considered stale for the main R2AI retrieval flow. The active baseline should focus on the Vietnamese Legal Documents (VLD) collection, especially the business-law subset currently ingested into Qdrant.

The production command that must remain supported is:

```bash
R2AI_BATCH_SIZE=128 R2AI_UPSERT_BATCH_SIZE=32 R2AI_FORCE_REBUILD=1 R2AI_RECREATE_COLLECTION=1 bash scripts/ingest_vld_qdrant_cloud.sh
```

This command still drives the same high-level flow: build VLD Qdrant preview artifacts, validate the build report, then call `r2ai ingest-qdrant --skip-build` to embed and upsert those preview rows.

## Current VLD Stages

1. Data availability
   - Input dataset root: `R2AI_VLD_DIR`, default `data/vietnamese-legal-documents`.
   - Expected files: `metadata/data-00000-of-00001.parquet` and `content/*.parquet`.
   - Optional fetch path: set `R2AI_ENSURE_VLD_DATA=1` to call `scripts/ensure_vld_data.py`.

2. Metadata filtering
   - Reads VLD metadata parquet.
   - Enriches effect status from `R2AI_VLD_EFFECT_METADATA_PATH` when available.
   - Keeps business-relevant documents using `classify_metadata`.
   - Important knobs: `R2AI_VLD_MIN_YEAR`, `R2AI_VLD_IDS_FILE`, `R2AI_VLD_LIMIT`.

3. Legal tree parsing
   - Converts markdown-like legal document content into a structured legal tree.
   - Preserves hierarchy such as part, chapter, section, article, clause, point, appendix, and table blocks.

4. Chunking and Qdrant preview build
   - Builds chunks from legal tree nodes.
   - Current ingest path uses `--article-only`, so embedding chunks are limited to article text/title/split chunks with normalized article numbers and without tables or appendices.
   - Writes document-level checkpoint part files before merging.
   - Main output for ingest: `qdrant_payload_preview.jsonl`.
   - Build report: `build_report.json`.

5. Build validation
   - `scripts/ingest_vld_qdrant_cloud.sh` checks that the build report exists, preview file exists, chunk count is non-zero, and `error_count <= R2AI_ALLOW_VLD_BUILD_ERRORS`.

6. Qdrant ingest
   - Uses existing generic `r2ai ingest-qdrant --skip-build` path.
   - Important knobs: `QDRANT_COLLECTION`, `R2AI_DENSE_MODEL`, `R2AI_SPARSE_MODEL`, `R2AI_BATCH_SIZE`, `R2AI_UPSERT_BATCH_SIZE`, `R2AI_RECREATE_COLLECTION`, `R2AI_HNSW_M`, `R2AI_HNSW_EF`.
   - Default collection for this script is `vld_business_law_v2`.

## Refactor Completed In This Step

The VLD builder logic has been moved out of standalone scripts and into importable package modules:

```text
r2ai/data_ingest/vld/
  __init__.py
  tree.py                 # moved from scripts/build_vld_tree_preview.py
  chunking.py             # moved from scripts/build_vld_chunk_preview.py
  business_collection.py  # moved from scripts/build_vld_business_collection.py
  business_qdrant.py      # moved from scripts/build_vld_business_qdrant.py
```

Backward-compatible wrappers remain in `scripts/`:

```text
scripts/build_vld_tree_preview.py
scripts/build_vld_chunk_preview.py
scripts/build_vld_business_collection.py
scripts/build_vld_business_qdrant.py
```

The main ingest shell script now calls the package module directly:

```bash
uv run python -m r2ai.data_ingest.vld.business_qdrant ...
```

This means the canonical implementation is now under `r2ai/data_ingest/vld`, while old script entrypoints still work for ad-hoc usage.

## Current Module Boundaries

- `tree.py`: VLD markdown/content parser and tree builder.
- `chunking.py`: token counting, tree-to-chunk conversion, Qdrant payload preview formatting used by VLD chunks.
- `business_collection.py`: older business-focused collection builder and metadata classification helpers.
- `business_qdrant.py`: production-oriented VLD Qdrant artifact builder used by `ingest_vld_qdrant_cloud.sh`.

The boundaries are not final yet. `business_collection.py` still mixes classification, preview artifacts, and older builder behavior. `business_qdrant.py` still owns filesystem checkpointing and report writing. Those should be separated in later steps.

## Next Refactor Targets

1. Split `business_collection.py`
   - `metadata.py`: normalize metadata, effect status merge, business-domain classification.
   - `filters.py`: business scope predicates and keyword lists.
   - `payload.py`: Qdrant preview payload format.

2. Split `business_qdrant.py`
   - `builder.py`: orchestration for production artifact build.
   - `checkpoints.py`: document part paths, atomic JSONL writes, merge/summarize helpers.
   - `reports.py`: build report schema and validation helpers.

3. Add a first-class CLI command
   - Proposed: `r2ai build-vld-qdrant-data`.
   - Then `scripts/ingest_vld_qdrant_cloud.sh` can call `uv run r2ai build-vld-qdrant-data ...` instead of `python -m ...`.

4. Separate VLD ingest config from Phapdien naming
   - Current Qdrant ingest still uses generic `QdrantIngestConfig` fields named `source_dir`, `max_chunk_tokens`, and `chunk_overlap_tokens` inherited from Phapdien.
   - VLD should eventually have explicit config names like `vld_root`, `max_text_tokens`, and `overlap_tokens`.

5. Update tests around VLD build units
   - Smoke import tests for wrappers and package modules.
   - Unit tests for metadata classification and article-only chunk filtering.
   - Report validation test for zero chunks and build error threshold.

## Compatibility Notes

The original command remains the supported command during this refactor:

```bash
R2AI_BATCH_SIZE=128 R2AI_UPSERT_BATCH_SIZE=32 R2AI_FORCE_REBUILD=1 R2AI_RECREATE_COLLECTION=1 bash scripts/ingest_vld_qdrant_cloud.sh
```

Expected behavior should stay the same after this step. The key internal change is that the build stage now runs through `r2ai.data_ingest.vld.business_qdrant`.
