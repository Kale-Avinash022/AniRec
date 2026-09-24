# Step 22: ChromaDB Persistent Collection Setup & Vector Config

## 1. Process & Purpose
Configure a persistent ChromaDB vector store on disk (`data/chroma/`) with cosine distance metrics, metadata indices, and collection management.

---

## 2. ChromaDB Setup Architecture

```mermaid
graph TD
    INIT[PersistentClient(path='data/chroma')] --> GET_COLL[get_or_create_collection('anime_catalog')]
    GET_COLL --> METRIC[Distance Metric: cosine]
    GET_COLL --> INDICES[Metadata Indexing: year, format, average_score, franchise_root_id]
```

---

## 3. Implementation Code

```python
# src/retrieval/vectorstore.py
import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import Optional
from src.core.config import settings
from src.core.logger import logger

class AnimeVectorStore:
    def __init__(self, collection_name: str = "anime_catalog"):
        settings.CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(settings.CHROMA_PERSIST_DIR),
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.logger.info(f"Connected to ChromaDB collection '{collection_name}' with {self.collection.count()} items.")

    def count(self) -> int:
        return self.collection.count()

    def reset(self):
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name="anime_catalog",
            metadata={"hnsw:space": "cosine"}
        )
```

---

## 4. Deliverables & Acceptance Criteria
- [x] Persistent ChromaDB client initialized with disk directory.
- [x] Cosine similarity HNSW space configured as primary vector distance metric.
