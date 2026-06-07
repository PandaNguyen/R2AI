#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATA_DIR="${R2AI_VLD_DIR:-data/vietnamese-legal-documents}"
BUILD_DIR="${R2AI_VLD_BUILD_DIR:-build/vld_business_scope}"
MODEL_CACHE_DIR="${R2AI_MODEL_CACHE_DIR:-.cache/models}"
VLD_REPO_ID="${VLD_REPO_ID:-vohuutridung/vietnamese-legal-documents}"
VLD_GIT_URL="${VLD_GIT_URL:-https://huggingface.co/datasets/vohuutridung/vietnamese-legal-documents}"
COLLECTION="${QDRANT_COLLECTION:-vld_business_law}"

if [[ -z "${QDRANT_URL:-}" ]]; then
  echo "Missing QDRANT_URL. Set it as a Kaggle secret/env var before running." >&2
  exit 1
fi

if [[ -z "${QDRANT_API_KEY:-}" ]]; then
  echo "Missing QDRANT_API_KEY. Set it as a Kaggle secret/env var before running." >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  python -m pip install -q uv
fi

uv sync --extra data --extra ingest

uv run python scripts/ensure_vld_data.py \
  --source-dir "$DATA_DIR" \
  --repo-id "$VLD_REPO_ID" \
  --git-url "$VLD_GIT_URL"

BUILD_ARGS=(
  scripts/build_vld_business_qdrant.py
  --vld-root "$DATA_DIR"
  --output-dir "$BUILD_DIR"
  --min-year "${R2AI_VLD_MIN_YEAR:-2010}"
  --max-text-tokens "${R2AI_MAX_TEXT_TOKENS:-2048}"
  --table-rows-per-chunk "${R2AI_TABLE_ROWS_PER_CHUNK:-8}"
  --progress-every "${R2AI_PROGRESS_EVERY:-100}"
)

if [[ -n "${R2AI_VLD_IDS_FILE:-}" ]]; then
  BUILD_ARGS+=(--ids-file "$R2AI_VLD_IDS_FILE")
fi

if [[ -n "${R2AI_VLD_LIMIT:-}" ]]; then
  BUILD_ARGS+=(--limit "$R2AI_VLD_LIMIT")
fi

uv run python "${BUILD_ARGS[@]}"

INGEST_ARGS=(
  ingest-qdrant
  --source-dir "$DATA_DIR"
  --build-dir "$BUILD_DIR"
  --model-cache-dir "$MODEL_CACHE_DIR"
  --batch-size "${R2AI_BATCH_SIZE:-64}"
  --collection "$COLLECTION"
  --skip-build
)

if [[ -n "${R2AI_RECREATE_COLLECTION:-}" ]]; then
  INGEST_ARGS+=(--recreate-collection)
fi

if [[ -n "${R2AI_INGEST_LIMIT:-}" ]]; then
  INGEST_ARGS+=(--limit "$R2AI_INGEST_LIMIT")
fi

uv run r2ai "${INGEST_ARGS[@]}"
