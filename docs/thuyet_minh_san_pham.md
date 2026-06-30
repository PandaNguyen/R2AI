# Thuyet Minh San Pham R2AI

Tai lieu nay tong hop cac thong tin can nop de phuc vu danh gia va nghiem thu san pham R2AI. Cac muc co ky hieu `TODO` can duoc cap nhat truoc khi nop chinh thuc, dac biet la link chia se du lieu, checkpoint va commit/revision su dung de tai hien ket qua.

## 1. Thong Tin Chung

| Hang muc | Noi dung |
|---|---|
| Ten san pham | R2AI - he thong truy hoi va tra loi cau hoi phap luat tieng Viet |
| Muc tieu | Truy hoi dieu/khoan phap luat lien quan va sinh cau tra loi co can cu tu ngu canh trich xuat |
| Pipeline chinh | Retrieval hybrid tren Qdrant + BM25 sparse + dense embedding, rerank cross-encoder, sau do sinh cau tra loi bang LLM |
| Entry point | `main.py`, console script `r2ai` |
| Cau hinh chinh | `configs/ir_main_flow.yaml` |
| Ket qua dau ra | File JSON gom `id`, `question`, `answer`, `relevant_docs`, `relevant_articles` |
| Repository/source code | TODO: dien link repository hoac link Google Drive/OneDrive chua source code |
| Nguoi/doi phu trach | TODO |

## 2. Tai Lieu Mo Ta Du Lieu

### 2.1 Nguon Du Lieu Su Dung

| Nguon | Vai tro trong san pham | Vi tri trong repo | Link cong khai/link chia se |
|---|---|---|---|
| Bo Phap Dien Viet Nam tu `phapdien.moj.gov.vn` | Corpus chinh cap Dieu dung de tao retrieval units, payload Qdrant va trich dan | `data/phapdien-moj-gov-vn/` | Public dataset: `tmquan/phapdien-moj-gov-vn`;  |
| Vietnamese Legal Documents tu `vbpl.vn` | Du lieu van ban phap luat bo tro/thu nghiem mo rong | `data/vietnam-legal-documentv2/` | Public dataset: `th1nhng0/vietnamese-legal-documents`; TODO: bo sung link Drive/OneDrive neu su dung trong ban nop |
| Vietnamese Legal Documents tu `thuvienphapluat.vn` | Du lieu van ban phap luat bo tro/thu nghiem mo rong | `data/vietnamese-legal-documents/` | Public dataset: `vohuutridung/vietnamese-legal-documents`; TODO: bo sung link Drive/OneDrive neu su dung trong ban nop |
| Tap cau hoi danh gia | Dau vao cho pipeline truy hoi/QA | Vi du: `data/R2AIStage1DATA.json` neu duoc cung cap ngoai repo | TODO: dien link tap cau hoi neu duoc phep chia se |


### 2.2 Cau Truc Va Dinh Dang Du Lieu

#### Bo Phap Dien Viet Nam

Du lieu chinh nam trong `data/phapdien-moj-gov-vn/`, gom cac file Parquet va ontology:

| Nhom file | Dinh dang | Mo ta |
|---|---|---|
| `articles-*.parquet` | Parquet | Moi dong la mot Dieu trong Bo Phap Dien, kem noi dung da chuan hoa, ma trich dan, chu de, de muc va lien ket nguon |
| `subjects.parquet`, `tree_nodes.parquet` | Parquet | Cau truc cay chu de/de muc/dieu phuc vu chunking va dieu huong ngu canh |
| `ontology*.parquet`, `ontology*.csv`, `ontology.json` | Parquet/CSV/JSON | Tu dien va ontology chu de, de muc, glossary song ngu |
| `ontology_*.png`, `ontology_mindmap.mmd` | PNG/Mermaid | Hinh anh/so do minh hoa ontology |

Qua lenh build, he thong sinh cac artifact trung gian:

| File sinh ra | Mo ta |
|---|---|
| `build/articles.jsonl` | Ban ghi article chuan hoa |
| `build/retrieval_units.jsonl` | Don vi truy hoi cap article/chunk |
| `build/qdrant_payload_preview.jsonl` | Payload Qdrant chua vector, dung de kiem tra truoc khi ingest |
| `build/data_quality_report.json` | Bao cao so luong va chan doan chat luong du lieu |

Mac dinh, moi Dieu duoc xem la mot retrieval unit. Cac Dieu dai duoc cat bang tree-aware chunker, giu cau truc phap ly cha-con va chi cat theo cau/dau cau/tu khi can thiet.

#### Vietnamese Legal Documents

Hai bo du lieu phu tro co cau truc Parquet:

| Thu muc | Cac bang chinh | Mo ta |
|---|---|---|
| `data/vietnam-legal-documentv2/` | `data/metadata.parquet`, `data/content.parquet`, `data/relationships.parquet` | Metadata van ban, HTML noi dung, quan he phap ly giua van ban |
| `data/vietnamese-legal-documents/` | `metadata/data-*.parquet`, `content/data-*.parquet` | Metadata va noi dung Markdown cua van ban phap luat |

Khoa noi bang thuong dung la `id`. Voi bang quan he, `doc_id` la van ban nguon va `other_doc_id` la van ban dich.

### 2.3 Huong Dan Truy Cap Hoac Su Dung Du Lieu

Tai du lieu chinh neu chua co san:

```bash
uv sync --extra data
python main.py ensure-phapdien-data --source-dir data/phapdien-moj-gov-vn
```

Build artifact truy hoi:

```bash
python main.py build-phapdien-data --source-dir data/phapdien-moj-gov-vn --output-dir build
```

Khi nop nghiem thu, can chia se du lieu qua mot link truy cap on dinh:

| Hang muc can chia se | Link |
|---|---|
| Ban dong goi du lieu goc | TODO: Google Drive/OneDrive/link tuong duong |
| Artifact `build/` da tao | TODO: Google Drive/OneDrive/link tuong duong |
| File cau hoi dau vao | TODO |
| File ket qua dau ra | TODO |

## 3. Mo Hinh Su Dung

### 3.1 Danh Sach Mo Hinh

| Thanh phan | Model/checkpoint mac dinh | Vai tro | Cau hinh |
|---|---|---|---|
| Dense embedding | `AITeamVN/Vietnamese_Embedding_v2` | Ma hoa cau hoi/van ban thanh vector dense | `road2ai.embed_model_path` trong `configs/ir_main_flow.yaml` |
| Sparse retrieval | `Qdrant/bm25` hoac BM25 cache cuc bo | Truy hoi sparse/BM25 de ket hop hybrid | `road2ai.sparse_vector_name`, `road2ai.bm25_cache` |
| Cross-encoder reranker | `AITeamVN/Vietnamese_Reranker` | Sap xep lai candidate sau truy hoi hybrid | `road2ai.rerank_model_path` |
| QA LLM | `Qwen/Qwen3-8B` | Sinh cau tra loi tu ngu canh da truy hoi | `qa.llm.model_id` |
| Dense ingest model thay the trong README | `jinaai/jina-embeddings-v5-text-small` | Tuy chon khi ingest Qdrant theo script cu | `R2AI_DENSE_MODEL` hoac tham so CLI |

### 3.2 Phien Ban Checkpoint

Truoc khi nop, can ghi ro revision/commit hash cua tung checkpoint de tai hien dung ket qua.

| Model/checkpoint | Revision/commit hash | Link checkpoint |
|---|---|---|
| `AITeamVN/Vietnamese_Embedding_v2` | TODO | TODO |
| `AITeamVN/Vietnamese_Reranker` | TODO | TODO |
| `Qwen/Qwen3-8B` | TODO | TODO |
| Qdrant collection/vector index da ingest | TODO: ten collection, ngay tao, so point | TODO: link snapshot/export neu co |
| BM25 cache cuc bo | TODO: duong dan/file hash | TODO: link chia se neu co |

### 3.3 Huong Dan Tai Va Su Dung Checkpoint

Cai dat dependencies lien quan model:

```bash
uv sync --extra search --extra qa
```

Neu chay bang `pip`:

```bash
pip install -e ".[search,qa]"
```

Thiet lap cache model neu can:

```bash
export HF_HOME=/path/to/hf-cache
export HF_TOKEN=TODO_NEU_CAN
```

Model Hugging Face se duoc tai tu dong khi pipeline chay lan dau. Neu da tai san checkpoint noi bo, cap nhat `configs/ir_main_flow.yaml` nhu sau:

```yaml
qa:
  llm:
    model_id: /path/to/checkpoints/Qwen3-8B

road2ai:
  embed_model_path: /path/to/checkpoints/Vietnamese_Embedding_v2
  rerank_model_path: /path/to/checkpoints/Vietnamese_Reranker
```

Checkpoint can duoc chia se qua link truy cap:

| Hang muc | Link chia se |
|---|---|
| Thu muc checkpoint embedding | TODO |
| Thu muc checkpoint reranker | TODO |
| Thu muc checkpoint LLM | TODO |
| Snapshot/export Qdrant collection | TODO |
| BM25 cache/model cache neu can | TODO |

## 4. Ma Nguon

### 4.1 Cau Truc Source Code

| Duong dan | Vai tro |
|---|---|
| `main.py` | Entry point goi `r2ai.cli:main` |
| `r2ai/cli.py` | Dinh nghia CLI: build data, ingest Qdrant, search, submit, ir-main-flow |
| `r2ai/data_ingest/phapdien/` | Loader, parser citation, chunking, record builder va quality report cho Bo Phap Dien |
| `r2ai/data_ingest/vld/` | Xu ly cac bo Vietnamese Legal Documents phu tro |
| `r2ai/indexing/` | Cau hinh va ingest Qdrant |
| `r2ai/retrieval/` | Qdrant search, batch search, dinh dang ket qua submit |
| `r2ai/search/` | Search backend, main IR flow, pipeline context va IR result schema |
| `r2ai/qa/` | Tao prompt/ngu canh va sinh cau tra loi bang Transformers LLM |
| `scripts/` | Script ingest, experiment, phan tich va tien xu ly bo tro |
| `tests/` | Unit tests cho CLI, ingest, retrieval, search va QA |

### 4.2 Thu Vien, Framework Va Dependencies

Dependencies chinh trong `pyproject.toml`:

| Nhom | Dependencies |
|---|---|
| Core | `numpy`, `PyYAML` |
| Data | `pyarrow`, `tiktoken` |
| Search/Ingest | `fastembed`, `huggingface-hub`, `peft`, `qdrant-client`, `sentence-transformers`, `sentencepiece`, `torch`, `transformers` |
| QA | `torch`, `transformers` |
| Dev/test | Python `unittest` trong repo |

Yeu cau moi truong:

| Hang muc | Gia tri de xuat |
|---|---|
| Python | `>=3.11` |
| Package manager | `uv` hoac `pip` |
| GPU | Khuyen nghi CUDA GPU cho embedding/rerank/LLM; co the chinh `device_embed`, `device_rerank`, `device_map` trong config |
| Vector database | Qdrant Cloud hoac Qdrant self-hosted |

### 4.3 Tep Cau Hinh Can Thiet

| Tep | Mo ta |
|---|---|
| `configs/ir_main_flow.yaml` | Cau hinh end-to-end IR + QA: backend, top-k, retrieve pool, rerank, model LLM |
| `.env` | Bien moi truong cuc bo, khong commit secret |
| `pyproject.toml` | Metadata package va dependency groups |
| `uv.lock` | Lockfile tai hien moi truong khi dung `uv` |

Bien moi truong toi thieu khi dung Qdrant:

```bash
export QDRANT_URL="https://YOUR_CLUSTER.qdrant.io"
export QDRANT_API_KEY="TODO"
export QDRANT_COLLECTION="r2ai_phapdien_baseline_v1"
```

## 5. Tai Lieu Huong Dan Cai Dat Va Chay Lai

README chinh cua repo la `README.md`. Phan duoi day tom tat cac buoc de nguoi khac co the reproduce tu dau.

### 5.1 Cai Dat Moi Truong

Dung `uv`:

```bash
uv sync --extra data --extra search --extra qa
```

Hoac dung `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[data,search,qa]"
```

### 5.2 Chuan Bi Du Lieu

Tai/kiem tra du lieu Bo Phap Dien:

```bash
python main.py ensure-phapdien-data --source-dir data/phapdien-moj-gov-vn
```

Build artifact:

```bash
python main.py build-phapdien-data \
  --source-dir data/phapdien-moj-gov-vn \
  --output-dir build \
  --max-chunk-tokens 2048 \
  --chunk-overlap-tokens 256
```

### 5.3 Ingest Vao Qdrant

Thiet lap secret:

```bash
export QDRANT_URL="https://YOUR_CLUSTER.qdrant.io"
export QDRANT_API_KEY="TODO"
export QDRANT_COLLECTION="r2ai_phapdien_baseline_v1"
```

Chay ingest:

```bash
python main.py ingest-qdrant \
  --source-dir data/phapdien-moj-gov-vn \
  --build-dir build \
  --collection "$QDRANT_COLLECTION" \
  --dense-model AITeamVN/Vietnamese_Embedding_v2 \
  --sparse-model Qdrant/bm25 \
  --batch-size 8
```

Neu dung script co san:

```bash
bash scripts/ingest_qdrant_cloud.sh
```

### 5.4 Chay Smoke Test Truy Hoi

```bash
python main.py search-qdrant "Doanh nghiep nho va vua duoc huong uu dai gi khi tham gia dau thau?" \
  --mode hybrid \
  --top-k 5 \
  --prefetch-limit 20 \
  --rerank \
  --reranker-model AITeamVN/Vietnamese_Reranker
```

### 5.5 Chay End-to-End IR + QA Main Flow

Lenh chinh:

```bash
r2ai ir-main-flow data/R2AIStage1DATA.json build/results.json
```

Hoac qua `main.py`:

```bash
python main.py ir-main-flow data/R2AIStage1DATA.json build/results.json \
  --config configs/ir_main_flow.yaml
```

File `configs/ir_main_flow.yaml` dang cau hinh:

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

### 5.6 Chay Batch Submission Retrieval-Only/Rerank

Neu chi can sinh file ket qua truy hoi:

```bash
python main.py submit-qdrant \
  --questions data/R2AIStage1DATA.json \
  --output build/results_rerank.json \
  --mode hybrid \
  --top-k 4 \
  --prefetch-limit 30 \
  --rerank \
  --reranker-model AITeamVN/Vietnamese_Reranker \
  --progress-every 10
```

### 5.7 Kiem Thu

Chay unit tests:

```bash
python -m unittest discover -s tests
```

Kiem tra artifact sau khi chay:

| Artifact | Dieu kien |
|---|---|
| `build/results.json` | La JSON hop le, moi dong co `id`, `question`, `answer`, `relevant_docs`, `relevant_articles` |
| `build/ir_results.jsonl` | Co candidate IR cho tung cau hoi neu `flow.ir_output` duoc bat |
| Log console | Khong co loi Qdrant/model/tokenizer; co tien do search/generate |

## 6. Danh Muc Can Nop

| Hang muc bat buoc | Trang thai | Ghi chu |
|---|---|---|
| Tai lieu mo ta du lieu | Da co trong file nay | Can bo sung link chia se |
| Du lieu goc va artifact build | TODO | Chia se qua Google Drive/OneDrive/link tuong duong |
| Thong tin model/checkpoint | Da co khung thong tin | Can bo sung revision va link checkpoint |
| Checkpoint/model cache/Qdrant snapshot | TODO | Chia se qua link truy cap |
| Toan bo ma nguon | Da co trong repository | Can bo sung link repository/ban nen |
| Dependencies va config | Da mo ta | Kem `pyproject.toml`, `uv.lock`, `configs/ir_main_flow.yaml`, `.env.example` neu co |
| README huong dan cai dat/chay lai | Co `README.md` va huong dan tom tat trong file nay | Nen cap nhat README neu co thay doi moi |
| Ket qua dau ra mau | TODO | Vi du `results_llm_type1_top3_prefetch30_v5.json` hoac file submit chinh thuc |

## 7. Checklist Truoc Khi Nop

- [ ] Cap nhat link repository/source code.
- [ ] Cap nhat link Google Drive/OneDrive cho du lieu goc.
- [ ] Cap nhat link artifact `build/` va file ket qua.
- [ ] Ghi ro revision/commit hash cua tung checkpoint Hugging Face.
- [ ] Chia se snapshot Qdrant collection hoac huong dan recreate collection tu dau.
- [ ] Xoa secret that khoi `.env`; chi nop `.env.example` neu can.
- [ ] Chay lai `python -m unittest discover -s tests`.
- [ ] Chay smoke test `search-qdrant`.
- [ ] Chay lai `ir-main-flow` tu dau va doi chieu schema ket qua.
- [ ] Dam bao nguoi khac co quyen truy cap tat ca link chia se.
