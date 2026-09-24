# 🎬 AniRec — Production Conversational Anime Recommendation Engine

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Tests Passing](https://img.shields.io/badge/tests-42%20passed-brightgreen.svg)](tests/)
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

## 🗺️ Master Implementation Roadmap

The full engineering specification spans 60 granular steps. See [MASTER_ROADMAP_AND_INDEX.md](MASTER_ROADMAP_AND_INDEX.md) for full phase-by-phase details:

| Phase | Focus | Status |
|---|---|---|
| **Phase 1** | Project Foundations, Architecture & Environment (Steps 01–07) | ✅ Completed |
| **Phase 2** | AniList GraphQL Ingestion Pipeline (Steps 08–14) | 🔄 In Progress (Steps 08–11 Done) |
| **Phase 3** | Document Engineering & Chunking Strategy (Steps 15–19) | ⏳ Planned |
| **Phase 4** | Embeddings & ChromaDB Vector Indexing (Steps 20–24) | ⏳ Planned |
| **Phase 5** | Hybrid Retrieval, Reranking & Diversity (Steps 25–31) | ⏳ Planned |
| **Phase 6** | Generation Layer, Grounding & Guardrails (Steps 32–37) | ⏳ Planned |
| **Phase 7** | FastAPI Backend API Engineering (Steps 38–43) | ⏳ Planned |
| **Phase 8** | Streamlit Frontend Chat UI (Steps 44–48) | ⏳ Planned |
| **Phase 9** | Golden-Set Evaluation & LLM-as-Judge (Steps 49–54) | ⏳ Planned |
| **Phase 10** | Production Containerization & Cloud Deployment (Steps 55–60) | ⏳ Planned |

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
pip install -r requirements.txt
# Or install in editable mode with development tools:
pip install -e .[dev]
```

### 4. Configure environment
```bash
cp .env.example .env
```

### 5. Run test suite
```bash
pytest
```
All unit tests should pass (currently **42/42 tests passing**).

---

## 📂 Repository Directory Layout

```
anirec/
├── MASTER_ROADMAP_AND_INDEX.md     # 60-step roadmap specification
├── README.md                       # Repository overview and setup guide
├── pyproject.toml                  # Packaging and tool configurations
├── requirements.txt                # Pinned dependency requirements
├── ruff.toml                       # Ruff linter and formatter config
├── .env.example                    # Environment variable template
│
├── data/                           # Data storage (gitignored)
│   ├── raw/                        # Atomic AniList API snapshots (page_001.json, etc.)
│   ├── processed/                  # Assembled documents & corpus
│   └── chroma/                     # ChromaDB vector index
│
├── logs/                           # Observability & traces (gitignored)
│   └── app.jsonl
│
├── evals/                          # Evaluation harness & benchmarks
│
├── src/                            # Application modules
│   ├── core/                       # Config, structured logger, cost tracking, base interfaces
│   ├── ingest/                     # GraphQL client, token-bucket limiter, disk cache
│   ├── retrieval/                  # ChromaDB vector store, embedders, rerankers
│   ├── generation/                 # LLM provider clients, prompt templates, guardrails
│   ├── api/                        # FastAPI application and schemas
│   └── ui/                         # Streamlit chat interface
│
├── steps/                          # 60 implementation specification guides
└── tests/                          # Automated unit and integration test suite
```

---

## 🛡️ License & Acknowledgments
- Data provided publicly by [AniList API](https://anilist.gitbook.io/anilist-apiv2-docs/).
