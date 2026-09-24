# Step 07: Unified Repository Directory Structure Blueprint

## 1. Process & Purpose
Design a clean, modular, and maintainable project directory structure that separates concerns between ingestion, document synthesis, retrieval, generation, API serving, UI presentation, and evaluation harnesses.

---

## 2. Directory Layout Blueprint

```
anirec/
├── MASTER_ROADMAP_AND_INDEX.md     # Master Index of all 60 steps & timeline
├── Overview.txt                    # Original design specification
├── README.md                       # High-level architecture, metrics table, demo
├── pyproject.toml                  # Packaging, metadata & dependencies
├── ruff.toml                       # Linting & formatting configuration
├── .env.example                    # Environment variable template
├── Dockerfile                      # Multi-stage production container build
├── docker-compose.yml              # Multi-container orchestration
│
├── data/                           # Data storage layer (gitignored)
│   ├── raw/                        # Paginated AniList JSON snapshots (page_01.json, etc.)
│   ├── processed/                  # Formatted documents & metadata (anime_corpus.json)
│   └── chroma/                     # Persistent ChromaDB vector index files
│
├── logs/                           # System traces & latency/cost logs
│   ├── app.jsonl                   # Request-level JSONL traces
│   └── violations.jsonl            # Hallucination/guardrail violation audit log
│
├── evals/                          # Evaluation harness & benchmarks
│   ├── golden_set.json             # 50 expert-curated Q&A evaluation pairs
│   ├── run_evals.py                # Automated evaluation benchmark runner
│   ├── judge.py                    # LLM-as-a-judge faithfulness evaluator
│   ├── metrics.py                  # Hit Rate@K, MRR & constraint accuracy metrics
│   └── results/                    # Timestamped eval run comparisons
│
├── src/                            # Source code modules
│   ├── core/                       # Shared models, settings, logging & interfaces
│   │   ├── __init__.py
│   │   ├── config.py               # pydantic-settings configuration
│   │   ├── logger.py               # Structured JSONL logger
│   │   ├── cost.py                 # Token cost estimator
│   │   ├── interfaces.py           # BaseEmbedder & BaseChatModel abstractions
│   │   └── models.py               # Core domain entities & Pydantic models
│   │
│   ├── ingest/                     # AniList GraphQL crawling & snapshotting
│   │   ├── __init__.py
│   │   ├── anilist_client.py       # GraphQL HTTP client with rate-limiter
│   │   ├── rate_limiter.py         # Polite token bucket / delay throttler
│   │   ├── transform.py            # Raw JSON to AnimeMetadata converter
│   │   └── build_index.py          # End-to-end index build runner script
│   │
│   ├── retrieval/                  # Multi-stage retrieval & ranking engine
│   │   ├── __init__.py
│   │   ├── vectorstore.py          # ChromaDB collection wrapper
│   │   ├── embedder.py             # SentenceTransformers & OpenAI embedders
│   │   ├── query_parser.py         # Rules-based & LLM structured query parser
│   │   ├── relaxation.py           # Progressive filter relaxation strategy
│   │   ├── reranker.py             # Cross-Encoder rescorer
│   │   └── diversity.py            # MMR diversity & franchise deduplicator
│   │
│   ├── generation/                 # LLM generation & guardrail layer
│   │   ├── __init__.py
│   │   ├── llm_client.py           # OpenAI & Ollama client adapters
│   │   ├── prompts.py              # Strict JSON & citation system prompts
│   │   ├── recommender.py          # Grounded recommendation generator
│   │   └── guardrails.py           # Anti-hallucination title post-checker
│   │
│   ├── api/                        # FastAPI HTTP gateway
│   │   ├── __init__.py
│   │   ├── main.py                 # FastAPI application instance & routing
│   │   ├── schemas.py              # HTTP request & response models
│   │   └── middleware.py           # Request ID injection & timing middleware
│   │
│   └── ui/                         # Streamlit frontend
│       ├── __init__.py
│       └── streamlit_app.py        # Interactive chat interface & explainability panels
│
├── steps/                          # Granular 60-step implementation guides
│   ├── STEP_01_PROJECT_PHILOSOPHY_AND_HIRING_IMPACT.md
│   └── ... (Steps 02 to 60)
│
└── tests/                          # Automated unit & integration tests
    ├── __init__.py
    ├── test_anilist_client.py
    ├── test_document_builder.py
    ├── test_query_parser.py
    ├── test_guardrails.py
    └── test_api_endpoints.py
```

---

## 3. Deliverables & Acceptance Criteria
- [x] Standardized naming conventions established across all modules.
- [x] Clear encapsulation of data, retrieval, generation, API, UI, and test domains.
