# 🎬 AniRec — Production Conversational Anime Recommendation Engine

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Architecture: RAG](https://img.shields.io/badge/architecture-RAG-orange.svg)](MASTER_ROADMAP_AND_INDEX.md)

**AniRec** is a production-grade, conversational anime recommendation engine powered by:
- **Retrieval-Augmented Generation (RAG)**
- **AniList GraphQL API Ingestion** with polite rate limiting and atomic snapshotting
- **ChromaDB Vector Store** with domain-tuned embeddings
- **Cross-Encoder Reranking** (`ms-marco-MiniLM-L-6-v2`) and MMR diversity
- **Anti-Hallucination Guardrail Verification** and citation grounding
- **FastAPI Backend** with SSE streaming & **Streamlit Chat UI**
- **LLM-as-a-Judge Evaluation Harness** (Hit Rate@5, MRR, Faithfulness)

---

## 🗺️ Implementation Roadmap

The full engineering specification spans 60 granular steps. See [MASTER_ROADMAP_AND_INDEX.md](MASTER_ROADMAP_AND_INDEX.md) for full phase-by-phase details:

| Phase | Focus | Status |
|---|---|---|
| **Phase 1** | Project Foundations, Architecture & Environment (Steps 01–07) | ✅ Complete |
| **Phase 2** | AniList GraphQL Ingestion Pipeline (Steps 08–14) | ✅ Complete |
| **Phase 3** | Document Engineering & Chunking Strategy (Steps 15–19) | ✅ Complete |
| **Phase 4** | Embeddings & ChromaDB Vector Indexing (Steps 20–24) | ⏳ Planned |
| **Phase 5** | Hybrid Retrieval, Reranking & Diversity (Steps 25–31) | ⏳ Planned |
| **Phase 6** | Generation Layer, Grounding & Guardrails (Steps 32–37) | ⏳ Planned |
| **Phase 7** | FastAPI Backend API Engineering (Steps 38–43) | ⏳ Planned |
| **Phase 8** | Streamlit Frontend Chat UI (Steps 44–48) | ⏳ Planned |
| **Phase 9** | Golden-Set Evaluation & LLM-as-Judge (Steps 49–54) | ⏳ Planned |
| **Phase 10** | Production Containerization & Cloud Deployment (Steps 55–60) | ⏳ Planned |

---

## ✅ What's Built So Far

### Phase 1 — Foundations (Steps 01–07)
- `src/core/config.py` — Pydantic-settings config singleton with all path & model settings
- `src/core/logger.py` — Structured JSON logger with JSONL trace output
- `src/core/interfaces.py` — Abstract `BaseEmbedder` and `BaseChatModel` provider contracts
- `src/core/cost.py` — Token cost tracker
- `src/core/metrics_spec.py` — Evaluation metric definitions

### Phase 2 — AniList Ingestion Pipeline (Steps 08–14)
- `src/ingest/queries.py` — Full GraphQL query (tags, relations, studios, scores)
- `src/ingest/rate_limiter.py` — Async token-bucket rate limiter with `Retry-After` support
- `src/ingest/storage.py` — Atomic page snapshot cache (`data/raw/page_001.json` …)
- `src/ingest/anilist_client.py` — Fault-tolerant paginator (60 pages, exponential backoff)
- `src/ingest/transform.py` — `SanitizedAnime` extractor: HTML strip, spoiler filtering, tag sorting
- `src/ingest/franchise_graph.py` — BFS franchise graph → `franchise_root_id` per show
- `src/ingest/validate.py` — `DataQualityAuditor` with 6 checks + `run_full_pipeline()`

### Phase 3 — Document Engineering (Steps 15–19)
- `src/docs/schema.py` — `AnimeDocument` Pydantic type + `CHUNKING_STRATEGY` constants
- `src/docs/tags.py` — Weighted tag rank engine: filter (min_rank=40), sort, format `"survival (92)"`
- `src/docs/assembler.py` — `build_document_text()` + `build_anime_document()` + batch builder
- `src/docs/metadata.py` — ChromaDB metadata schema (20 fields), `build_where_clause()` composer
- `src/docs/snapshot.py` — Atomic `save_processed_corpus()` / `load_processed_corpus()` + manifest

---

## ⚡ Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/AniRec.git
cd AniRec
```

### 2. Set up virtual environment
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -e .[dev]
```

### 4. Configure environment
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 5. Run smoke tests
```bash
python tests/smoke_step15.py
python tests/smoke_step16.py
python tests/smoke_step17.py
python tests/smoke_step18.py
python tests/smoke_step19.py
```

---

## 📂 Repository Directory Layout

```
anirec/
├── MASTER_ROADMAP_AND_INDEX.md     # 60-step roadmap specification
├── README.md                       # This file
├── pyproject.toml                  # Packaging and tool configurations
├── requirements.txt                # Pinned dependency requirements
├── ruff.toml                       # Ruff linter and formatter config
├── .env.example                    # Environment variable template
│
├── data/                           # Data storage (gitignored)
│   ├── raw/                        # Atomic AniList API snapshots (page_001.json ...)
│   ├── processed/                  # anime_corpus.json + corpus_manifest.json
│   └── chroma/                     # ChromaDB persistent vector index
│
├── logs/                           # Structured JSONL traces (gitignored)
│
├── evals/                          # Evaluation harness & golden datasets
│
├── src/
│   ├── core/                       # Config, logger, cost tracker, base interfaces
│   ├── ingest/                     # GraphQL client, rate limiter, transform, franchise graph, validator
│   ├── docs/                       # Document schema, tag engine, assembler, metadata, snapshot
│   ├── retrieval/                  # (Phase 4+) Embedders, ChromaDB store, rerankers
│   ├── generation/                 # (Phase 6+) LLM clients, prompt templates, guardrails
│   ├── api/                        # (Phase 7+) FastAPI application and schemas
│   └── ui/                         # (Phase 8+) Streamlit chat interface
│
├── steps/                          # 60 implementation specification guides
└── tests/                          # Smoke tests for each completed step
```

---

## 🛡️ License & Acknowledgments
- Distributed under the **MIT License**. See [LICENSE](LICENSE) for more information.
- Anime metadata provided publicly by the [AniList GraphQL API](https://anilist.gitbook.io/anilist-apiv2-docs/).
