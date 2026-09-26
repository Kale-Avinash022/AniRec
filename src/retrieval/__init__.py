"""AniRec retrieval module.

Phase 4: Embeddings & Vector Database Indexing
  Step 21 — embedder.py        : BaseEmbedder implementations (local + cloud)
  Step 22 — vectorstore.py     : AnimeVectorStore (ChromaDB persistent client)
  Step 23 — indexer.py         : Batch indexing pipeline
  Step 24 — (verify_index.py in scripts/)

Phase 5: Hybrid Retrieval, Reranking & Diversity
  Step 25 — query_parser.py    : Rules-based regex filter parser
  Step 26 — llm_query_parser.py: LLM structured filter extraction (+ fallback)
  Step 27 — search.py          : Semantic vector search with metadata WHERE clauses
  Step 28 — relaxation.py      : Progressive filter relaxation strategy
  Step 29 — reranker.py        : Cross-Encoder reranker (ms-marco-MiniLM-L-6-v2)
  Step 30 — dedup.py           : Franchise deduplication  [TODO]
  Step 31 — mmr.py             : Maximal Marginal Relevance diversity [TODO]
"""

from src.retrieval.dedup import deduplicate_franchises
from src.retrieval.diversity import cosine_sim, mmr_diversity_filter
from src.retrieval.embedder import (
    OpenAIEmbedder,
    SentenceTransformerEmbedder,
    get_embedder,
)
from src.retrieval.indexer import build_vector_index
from src.retrieval.llm_query_parser import parse_query, parse_query_with_llm
from src.retrieval.query_parser import ExtractedFilters, ParsedQuery, parse_query_rules
from src.retrieval.relaxation import search_with_relaxation
from src.retrieval.reranker import Reranker
from src.retrieval.search import build_chroma_filter, search_candidates
from src.retrieval.vectorstore import COLLECTION_NAME, AnimeVectorStore

__all__ = [
    # Step 21 — Embedder
    "SentenceTransformerEmbedder",
    "OpenAIEmbedder",
    "get_embedder",
    # Step 22 — VectorStore
    "AnimeVectorStore",
    "COLLECTION_NAME",
    # Step 23 — Indexer
    "build_vector_index",
    # Step 25 — Rules Query Parser
    "ExtractedFilters",
    "ParsedQuery",
    "parse_query_rules",
    # Step 26 — LLM Query Parser
    "parse_query",
    "parse_query_with_llm",
    # Step 27 — Search
    "build_chroma_filter",
    "search_candidates",
    # Step 28 — Relaxation
    "search_with_relaxation",
    # Step 29 — Reranker
    "Reranker",
    # Step 30 — Deduplication
    "deduplicate_franchises",
    # Step 31 — Diversity
    "mmr_diversity_filter",
    "cosine_sim",
]
