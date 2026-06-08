#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATA_DIR="${R2AI_PHAPDIEN_DIR:-data/phapdien-moj-gov-vn}"
BUILD_DIR="${R2AI_BUILD_DIR:-build}"
MODEL_CACHE_DIR="${R2AI_MODEL_CACHE_DIR:-.cache/models}"
PHAPDIEN_GIT_URL="${PHAPDIEN_GIT_URL:-https://huggingface.co/datasets/tmquan/phapdien-moj-gov-vn}"

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

if [[ ! -d "$DATA_DIR" ]]; then
  mkdir -p "$(dirname "$DATA_DIR")"
  run_with_retries "Clone phapdien data" git clone "$PHAPDIEN_GIT_URL" "$DATA_DIR" || true
fi

run_with_retries "uv sync" uv sync --extra ingest

run_with_retries "Ensure phapdien data" uv run r2ai ensure-phapdien-data --source-dir "$DATA_DIR"

INGEST_ARGS=(
  ingest-qdrant
  --source-dir "$DATA_DIR"
  --build-dir "$BUILD_DIR"
  --model-cache-dir "$MODEL_CACHE_DIR"
  --batch-size "${R2AI_BATCH_SIZE:-64}"
  --max-chunk-tokens "${R2AI_MAX_CHUNK_TOKENS:-2048}"
  --chunk-overlap-tokens "${R2AI_CHUNK_OVERLAP_TOKENS:-256}"
)

if [[ -n "${QDRANT_COLLECTION:-}" ]]; then
  INGEST_ARGS+=(--collection "$QDRANT_COLLECTION")
fi

if [[ -n "${R2AI_RECREATE_COLLECTION:-}" ]]; then
  INGEST_ARGS+=(--recreate-collection)
fi

run_with_retries "Ingest Qdrant artifacts" uv run r2ai "${INGEST_ARGS[@]}"
