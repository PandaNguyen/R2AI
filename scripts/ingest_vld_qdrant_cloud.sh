#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

DATA_DIR="${R2AI_VLD_DIR:-data/vietnamese-legal-documents}"
MODEL_CACHE_DIR="${R2AI_MODEL_CACHE_DIR:-.cache/models}"
COLLECTION="${QDRANT_COLLECTION:-vld_business_law}"
VLD_REPO_ID="${VLD_REPO_ID:-vohuutridung/vietnamese-legal-documents}"
VLD_GIT_URL="${VLD_GIT_URL:-https://huggingface.co/datasets/vohuutridung/vietnamese-legal-documents}"

DENSE_MODEL="${R2AI_DENSE_MODEL:-jinaai/jina-embeddings-v5-text-small}"
SPARSE_MODEL="${R2AI_SPARSE_MODEL:-Qdrant/bm25}"
BATCH_SIZE="${R2AI_BATCH_SIZE:-2}"
MAX_TEXT_TOKENS="${R2AI_MAX_TEXT_TOKENS:-32768}"
OVERLAP_TOKENS="${R2AI_VLD_OVERLAP_TOKENS:-${R2AI_OVERLAP_TOKENS:-2048}}"
TABLE_ROWS_PER_CHUNK="${R2AI_TABLE_ROWS_PER_CHUNK:-16}"
MIN_YEAR="${R2AI_VLD_MIN_YEAR:-2000}"
PROGRESS_EVERY="${R2AI_PROGRESS_EVERY:-100}"
FORCE_REBUILD="${R2AI_FORCE_REBUILD:-0}"
RECREATE_COLLECTION="${R2AI_RECREATE_COLLECTION:-0}"
ALLOW_BUILD_ERRORS="${R2AI_ALLOW_VLD_BUILD_ERRORS:-0}"
HNSW_M="${R2AI_HNSW_M-48}"
HNSW_EF="${R2AI_HNSW_EF-256}"
BUILD_DIR="${R2AI_VLD_BUILD_DIR:-build/vld_business_min${MIN_YEAR}_tok${MAX_TEXT_TOKENS}_ov${OVERLAP_TOKENS}_tbl${TABLE_ROWS_PER_CHUNK}}"

mkdir -p "$BUILD_DIR" "$MODEL_CACHE_DIR"

run_with_retries() {
  local label="$1"
  shift
  local attempts="${R2AI_PIPELINE_RETRIES:-20}"
  local delay="${R2AI_PIPELINE_RETRY_DELAY:-30}"
  local attempt=1
  until "$@"; do
    if (( attempt >= attempts )); then
      echo "$label failed after $attempt/$attempts attempts." >&2
      return 1
    fi
    echo "$label failed on attempt $attempt/$attempts. Sleeping ${delay}s before continuing..." >&2
    sleep "$delay"
    attempt=$((attempt + 1))
  done
}

if [[ -z "${QDRANT_URL:-}" ]]; then
  echo "Missing QDRANT_URL. Set it in .env or export it before running." >&2
  exit 1
fi

if [[ -z "${QDRANT_API_KEY:-}" ]]; then
  echo "Missing QDRANT_API_KEY. Set it in .env or export it before running." >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  run_with_retries "Install uv" python -m pip install -q uv
fi

echo "VLD ingest config:"
echo "  DATA_DIR=$DATA_DIR"
echo "  BUILD_DIR=$BUILD_DIR"
echo "  COLLECTION=$COLLECTION"
echo "  DENSE_MODEL=$DENSE_MODEL"
echo "  SPARSE_MODEL=$SPARSE_MODEL"
echo "  MAX_TEXT_TOKENS=$MAX_TEXT_TOKENS"
echo "  OVERLAP_TOKENS=$OVERLAP_TOKENS"
echo "  TABLE_ROWS_PER_CHUNK=$TABLE_ROWS_PER_CHUNK"
echo "  BATCH_SIZE=$BATCH_SIZE"
echo "  HNSW_M=${HNSW_M:-<unchanged>}"
echo "  HNSW_EF=${HNSW_EF:-<unchanged>}"
echo "  FORCE_REBUILD=$FORCE_REBUILD"
echo "  RECREATE_COLLECTION=$RECREATE_COLLECTION"

run_with_retries "uv sync" uv sync --extra data --extra ingest

if [[ "${R2AI_ENSURE_VLD_DATA:-0}" == "1" ]]; then
  run_with_retries "Ensure VLD data" uv run python scripts/ensure_vld_data.py \
    --source-dir "$DATA_DIR" \
    --repo-id "$VLD_REPO_ID" \
    --git-url "$VLD_GIT_URL" \
    --download-method "${R2AI_VLD_DOWNLOAD_METHOD:-git-first}"
elif [[ ! -f "$DATA_DIR/metadata/data-00000-of-00001.parquet" || ! -d "$DATA_DIR/content" ]]; then
  echo "Missing local VLD dataset under $DATA_DIR." >&2
  echo "Clone it first or set R2AI_VLD_DIR=/path/to/vietnamese-legal-documents." >&2
  echo "Alternatively set R2AI_ENSURE_VLD_DATA=1 to let this script fetch it." >&2
  exit 1
fi

BUILD_ARGS=(
  scripts/build_vld_business_qdrant.py
  --vld-root "$DATA_DIR"
  --output-dir "$BUILD_DIR"
  --min-year "$MIN_YEAR"
  --max-text-tokens "$MAX_TEXT_TOKENS"
  --overlap-tokens "$OVERLAP_TOKENS"
  --table-rows-per-chunk "$TABLE_ROWS_PER_CHUNK"
  --progress-every "$PROGRESS_EVERY"
  --preview-only
)

if [[ -n "${R2AI_VLD_IDS_FILE:-}" ]]; then
  BUILD_ARGS+=(--ids-file "$R2AI_VLD_IDS_FILE")
fi

if [[ -n "${R2AI_VLD_LIMIT:-}" ]]; then
  BUILD_ARGS+=(--limit "$R2AI_VLD_LIMIT")
fi

if [[ "$FORCE_REBUILD" == "1" ]]; then
  BUILD_ARGS+=(--force-rebuild)
fi

run_with_retries "Build VLD Qdrant artifacts" uv run python "${BUILD_ARGS[@]}"

REPORT_PATH="$BUILD_DIR/build_report.json"
PREVIEW_PATH="$BUILD_DIR/qdrant_payload_preview.jsonl"
uv run python -c '
import json
import sys
from pathlib import Path

report_path = Path(sys.argv[1])
preview_path = Path(sys.argv[2])
allowed_errors = int(sys.argv[3])
if not report_path.exists():
    raise SystemExit(f"Missing build report: {report_path}")
if not preview_path.exists() or preview_path.stat().st_size == 0:
    raise SystemExit(f"Missing or empty Qdrant preview: {preview_path}")
report = json.loads(report_path.read_text(encoding="utf-8"))
chunk_count = int(report.get("chunk_count") or 0)
document_count = int(report.get("documents_built") or 0)
error_count = int(report.get("error_count") or 0)
print(f"VLD build summary: documents={document_count}, chunks={chunk_count}, errors={error_count}", flush=True)
if chunk_count <= 0:
    raise SystemExit("VLD build produced zero chunks; refusing to ingest.")
if error_count > allowed_errors:
    raise SystemExit(
        f"VLD build had {error_count} errors, above allowed {allowed_errors}. "
        "Set R2AI_ALLOW_VLD_BUILD_ERRORS to allow known bad documents."
    )
' "$REPORT_PATH" "$PREVIEW_PATH" "$ALLOW_BUILD_ERRORS"

if command -v uv >/dev/null 2>&1; then
  uv cache prune || true
fi

INGEST_ARGS=(
  ingest-qdrant
  --source-dir "$DATA_DIR"
  --build-dir "$BUILD_DIR"
  --model-cache-dir "$MODEL_CACHE_DIR"
  --dense-model "$DENSE_MODEL"
  --sparse-model "$SPARSE_MODEL"
  --batch-size "$BATCH_SIZE"
  --collection "$COLLECTION"
  --skip-build
)

if [[ -n "$HNSW_M" ]]; then
  INGEST_ARGS+=(--hnsw-m "$HNSW_M")
fi

if [[ -n "${R2AI_HNSW_EF_CONSTRUCT:-}" ]]; then
  INGEST_ARGS+=(--hnsw-ef-construct "$R2AI_HNSW_EF_CONSTRUCT")
elif [[ -n "$HNSW_EF" ]]; then
  INGEST_ARGS+=(--hnsw-ef "$HNSW_EF")
fi

if [[ "$RECREATE_COLLECTION" == "1" ]]; then
  INGEST_ARGS+=(--recreate-collection)
fi

if [[ -n "${R2AI_INGEST_LIMIT:-}" ]]; then
  INGEST_ARGS+=(--limit "$R2AI_INGEST_LIMIT")
fi

run_with_retries "Ingest VLD Qdrant artifacts" uv run r2ai "${INGEST_ARGS[@]}"
