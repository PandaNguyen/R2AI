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
characters are split with a tree-aware chunker. The chunker preserves parent
legal structure such as `1 -> 1.1` or `3 -> 3.1 -> a`, then falls back to
sentence splitting with overlap only when a leaf is still too long. If a leaf is
still too large, it recursively splits at punctuation boundaries (`.`, `;`, `,`)
and then whitespace as the last resort, so chunks do not cut through words.
Every chunk keeps `canonical_article_id` so later retrieval can group results
back to article-level citations.

Useful local build knobs:

```powershell
.\.venv\Scripts\python.exe main.py build-phapdien-data --max-chunk-tokens 384 --chunk-overlap-tokens 48
```

## One-Shot Kaggle Qdrant Ingest

Set Kaggle secrets/environment variables first:

```bash
export QDRANT_URL="https://YOUR_CLUSTER.qdrant.io"
export QDRANT_API_KEY="..."
export QDRANT_COLLECTION="r2ai_phapdien_baseline_v1"
```

Then run one script from a fresh git clone:

```bash
bash scripts/ingest_qdrant_cloud.sh
```

The script installs/uses `uv`, clones the phapdien dataset if missing, falls
back to Hugging Face snapshot download when Git LFS is unavailable, builds Phase
1 JSONL artifacts, downloads `mainguyen9/vietlegal-harrier-0.6b` and
`Qdrant/bm25`, creates the Qdrant collection, and upserts points in batches.

The default dense model is `mainguyen9/vietlegal-harrier-0.6b`
(SentenceTransformer, 1024-dim cosine, 512-token max sequence length). Passages
are embedded as raw `retrieval_text`. Later query retrieval should prepend:

```text
Instruct: Given a Vietnamese legal question, retrieve relevant legal passages that answer the question
Query: <question>
```

Useful knobs:

- `R2AI_BATCH_SIZE=8` to reduce memory pressure.
- `R2AI_MAX_CHUNK_TOKENS=2048` to control tree-aware content chunk size.
- `R2AI_CHUNK_OVERLAP_TOKENS=256` to control sentence fallback overlap.
- `R2AI_RECREATE_COLLECTION=1` to delete and recreate the collection.
- `R2AI_MODEL_CACHE_DIR=/kaggle/working/models` to control model cache path.

## Qdrant Search With Reranking

Install search dependencies:

```powershell
uv sync --extra search
```

Then enable the Vietnamese cross-encoder reranker after Qdrant retrieval:

```powershell
.\.venv\Scripts\python.exe main.py search-qdrant "Trí tuệ nhân tạo là gì?" --mode hybrid --top-k 5 --prefetch-limit 20 --rerank
```

The default reranker is `AITeamVN/Vietnamese_Reranker` with max sequence length
`2304`. Use `--reranker-model` or `--reranker-max-length` to override it.

### Kaggle Rerank Run

On Kaggle, add these secrets first: `QDRANT_URL`, `QDRANT_API_KEY`, and
optionally `HF_TOKEN` for higher Hugging Face rate limits.

Clone and install with Kaggle's Python environment so it can reuse the existing
CUDA PyTorch installation:

```bash
git clone https://github.com/PandaNguyen/R2AI.git
cd R2AI
pip install -q -e ".[search]"
```

Create `.env` from Kaggle secrets:

```python
from kaggle_secrets import UserSecretsClient

secrets = UserSecretsClient()
env_rows = [
    ("QDRANT_URL", secrets.get_secret("QDRANT_URL")),
    ("QDRANT_API_KEY", secrets.get_secret("QDRANT_API_KEY")),
    ("QDRANT_COLLECTION", "r2ai_phapdien_baseline_v1"),
]
try:
    env_rows.append(("HF_TOKEN", secrets.get_secret("HF_TOKEN")))
except Exception:
    pass

with open(".env", "w", encoding="utf-8", newline="\n") as handle:
    for key, value in env_rows:
        handle.write(f"{key}={value}\n")
```

Smoke test one question:

```bash
python main.py search-qdrant "Doanh nghiệp nhỏ và vừa được hưởng ưu đãi gì khi tham gia đấu thầu?" \
  --mode hybrid \
  --top-k 5 \
  --prefetch-limit 20 \
  --doc-title-format type1 \
  --model-cache-dir /kaggle/working/models \
  --rerank \
  --reranker-max-length 2304
```

Run the full file:

```bash
python main.py submit-qdrant \
  --questions data/R2AIStage1DATA.json \
  --output /kaggle/working/results_rerank.json \
  --mode hybrid \
  --top-k 20 \
  --prefetch-limit 100 \
  --doc-title-format type1 \
  --model-cache-dir /kaggle/working/models \
  --rerank \
  --reranker-model AITeamVN/Vietnamese_Reranker \
  --reranker-max-length 2304 \
  --qdrant-timeout 120 \
  --progress-every 10
```

For faster reranking or lower GPU memory use, reduce `--prefetch-limit` to `50`
or `30`; rerank cost grows with the number of retrieved candidates per question.

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
