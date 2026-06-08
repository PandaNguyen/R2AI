#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

KAGGLE_TEMP="${KAGGLE_TEMP:-/kaggle/temp}"
DATA_DIR="${R2AI_VLD_DIR:-$KAGGLE_TEMP/r2ai-data/vietnamese-legal-documents}"
BUILD_DIR="${R2AI_VLD_BUILD_DIR:-$KAGGLE_TEMP/r2ai-build/vld_business_scope}"
MODEL_CACHE_DIR="${R2AI_MODEL_CACHE_DIR:-$KAGGLE_TEMP/r2ai-cache/models}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$KAGGLE_TEMP/r2ai-cache/uv}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$KAGGLE_TEMP/r2ai-venv}"
export HF_HOME="${HF_HOME:-$KAGGLE_TEMP/r2ai-cache/huggingface}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$KAGGLE_TEMP/r2ai-cache/pip}"
export TMPDIR="${TMPDIR:-$KAGGLE_TEMP/r2ai-tmp}"
VLD_REPO_ID="${VLD_REPO_ID:-vohuutridung/vietnamese-legal-documents}"
VLD_GIT_URL="${VLD_GIT_URL:-https://huggingface.co/datasets/vohuutridung/vietnamese-legal-documents}"
COLLECTION="${QDRANT_COLLECTION:-vld_business_law}"

mkdir -p "$MODEL_CACHE_DIR" "$UV_CACHE_DIR" "$HF_HOME" "$PIP_CACHE_DIR" "$TMPDIR"

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
  echo "Missing QDRANT_URL. Set it as a Kaggle secret/env var before running." >&2
  exit 1
fi

if [[ -z "${QDRANT_API_KEY:-}" ]]; then
  echo "Missing QDRANT_API_KEY. Set it as a Kaggle secret/env var before running." >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  run_with_retries "Install uv" python -m pip install -q uv
fi

run_with_retries "uv sync" uv sync --extra data --extra ingest

run_with_retries "Ensure VLD data" uv run python scripts/ensure_vld_data.py \
  --source-dir "$DATA_DIR" \
  --repo-id "$VLD_REPO_ID" \
  --git-url "$VLD_GIT_URL" \
  --download-method snapshot

BUILD_ARGS=(
  scripts/build_vld_business_qdrant.py
  --vld-root "$DATA_DIR"
  --output-dir "$BUILD_DIR"
  --min-year "${R2AI_VLD_MIN_YEAR:-2010}"
  --max-text-tokens "${R2AI_MAX_TEXT_TOKENS:-2048}"
  --table-rows-per-chunk "${R2AI_TABLE_ROWS_PER_CHUNK:-8}"
  --progress-every "${R2AI_PROGRESS_EVERY:-100}"
  --preview-only
)

if [[ -n "${R2AI_VLD_IDS_FILE:-}" ]]; then
  BUILD_ARGS+=(--ids-file "$R2AI_VLD_IDS_FILE")
fi

if [[ -n "${R2AI_VLD_LIMIT:-}" ]]; then
  BUILD_ARGS+=(--limit "$R2AI_VLD_LIMIT")
fi

if [[ "${R2AI_FORCE_REBUILD:-0}" == "1" ]]; then
  BUILD_ARGS+=(--force-rebuild)
fi

run_with_retries "Build VLD Qdrant artifacts" uv run python "${BUILD_ARGS[@]}"

if [[ "${R2AI_DROP_DATA_AFTER_BUILD:-1}" == "1" && "$DATA_DIR" == "$KAGGLE_TEMP"* ]]; then
  echo "Dropping temporary VLD parquet data after build to save disk: $DATA_DIR"
  rm -rf "$DATA_DIR"
fi

if command -v uv >/dev/null 2>&1; then
  uv cache prune || true
fi

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

run_with_retries "Ingest VLD Qdrant artifacts" uv run r2ai "${INGEST_ARGS[@]}"

if [[ "${R2AI_CLEANUP_AFTER_INGEST:-0}" == "1" && "$BUILD_DIR" == "$KAGGLE_TEMP"* ]]; then
  echo "Dropping temporary VLD build artifacts after ingest: $BUILD_DIR"
  rm -rf "$BUILD_DIR"
fi
