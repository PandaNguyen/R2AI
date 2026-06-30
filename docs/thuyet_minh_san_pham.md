# Thuyết minh sản phẩm


## 1. DATA

### 1.1 Nguồn dữ liệu sử dụng

| Nguồn | Vai trò | Vị tris trong repo | Link công khai/link chia ser trên HF |
|---|---|---|---|
| Vietnamese Legal Documents tu `vbpl.vn` | Dữ liệu phụ trợ bổ sung metadata | `data/vietnam-legal-documentv2/` | Public dataset: `th1nhng0/vietnamese-legal-documents` |
| Vietnamese Legal Documents tu `thuvienphapluat.vn` | Dữ liệu văn bản chính để build chunk và tree | `data/vietnamese-legal-documents/` | Public dataset: `vohuutridung/vietnamese-legal-documents` |



### 1.2 Huong Dan Truy Cap Hoac Su Dung Du Lieu


Build artifact truy hoi:

```bash
R2AI_BATCH_SIZE=128 R2AI_UPSERT_BATCH_SIZE=32 R2AI_FORCE_REBUILD=1 R2AI_RECREATE_COLLECTION=1 bash scripts/ingest_vld_qdrant_cloud.sh
```



## 3. MÔ hình sử dụng và thử nghiệm

### 3.1 Danh sách mô Hình


AITeamVN/Vietnamese_Embedding_v2

AITeamVN/Vietnamese_Reranker

Qdrant/bm25

Qwen/Qwen3-8B  (và các dòng khác như qwen2 qwen2.5)

jinaai/jina-embeddings-v5-text-small

AITeamVN/Vi-Qwen2-7B-RAG

AITeamVN/Vi-Qwen2-3B-RAG

AITeamVN/Vi-Qwen2-1.5B-RAG



## 4. Mã nguồn

### 4.1 Source Code

|Path | Vai trò |
|---|---|
| `main.py` | Entry point goi `r2ai.cli:main` |
| `r2ai/cli.py` | Định nghĩa CLI: build data, ingest Qdrant, search, submit, ir-main-flow |
| `r2ai/data_ingest/vld/` | Xử lý Vietnamese Legal Documents |
| `r2ai/indexing/` | Cấu hình va ingest Qdrant |
| `r2ai/retrieval/` | Qdrant search, batch search, dđịnh dạng kết quả submit |
| `r2ai/search/` | Search backend, main IR flow, pipeline context va IR result schema |
| `r2ai/qa/` | Tạo prompt |
| `scripts/` | Script ingest, experiment, phân tích và xử lý bổ trợ helper |
| `tests/` | Unit tests cho CLI, ingest, retrieval, search va QA |

### 4.2 Framework Va Dependencies

Dependencies chính trong `pyproject.toml`:

| Nhóm | Dependencies |
|---|---|
| Core | `numpy`, `PyYAML` |
| Data | `pyarrow`, `tiktoken` |
| Search/Ingest | `fastembed`, `huggingface-hub`, `peft`, `qdrant-client`, `sentence-transformers`, `sentencepiece`, `torch`, `transformers` |
| QA | `torch`, `transformers` |
| Dev/test | Python `unittest` trong repo |


### 4.3 Tệp cấu hình cần thiết

| Tep | Mo ta |
|---|---|
| `configs/ir_main_flow.yaml` | Cau hinh end-to-end IR + QA: backend, top-k, retrieve pool, rerank, model LLM |
| `.env` | Bien moi truong cuc bo, khong commit secret |
| `pyproject.toml` | Metadata package va dependency groups |
| `uv.lock` | Lockfile tai hien moi truong khi dung `uv` |

Biến môi trường:

```bash
export QDRANT_URL="https://YOUR_CLUSTER.qdrant.io"
export QDRANT_API_KEY="..."
export QDRANT_COLLECTION="vld_business"
```

## 5. tài liệu


### 5.1 cài đặt

Dung `uv`:

```bash
uv sync --extra data --extra search --extra qa
```

hoặc `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[data,search,qa]"
```

sau đó

```bash
R2AI_BATCH_SIZE=128 R2AI_UPSERT_BATCH_SIZE=32 R2AI_FORCE_REBUILD=1 R2AI_RECREATE_COLLECTION=1 bash scripts/ingest_vld_qdrant_cloud.sh
```

### 5.5 End-to-End IR + QA Main Flow

Lệnh chính:

```bash
r2ai ir-main-flow data/R2AIStage1DATA.json build/results.json
```

Hoặc qua `main.py`:

```bash
python main.py ir-main-flow data/R2AIStage1DATA.json build/results.json \
  --config configs/ir_main_flow.yaml
```

File `configs/ir_main_flow.yaml` đang cấu hình:

| Nhom | Gia tri chinh |
|---|---|
| `flow.backend` | `road2ai` |
| `flow.ir_output` | `build/ir_results.jsonl` |
| `qa.answer_mode` | `llm` |
| `qa.context_limit` | `4` |
| `qa.llm.model_id` | `Qwen/Qwen3-8B` |
| `road2ai.mode` | `hybrid` |
| `road2ai.top_k` | `4` |
| `road2ai.retrieve_pool` | `15` |
| `road2ai.rrf_top_k` | `20` |
| `road2ai.use_rerank` | `true` |


