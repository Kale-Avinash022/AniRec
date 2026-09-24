# Step 42: `GET /anime/{id}` Show Detail & `GET /health` Endpoints

## 1. Process & Purpose
Expose lightweight auxiliary endpoints for UI popover cards (`GET /anime/{id}`) and container orchestration liveness checks (`GET /health`).

---

## 2. Implementation Code

```python
# src/api/routes_aux.py
from fastapi import APIRouter, HTTPException
from src.api.schemas import HealthResponse
from src.retrieval.vectorstore import AnimeVectorStore
from src.core.config import settings

router = APIRouter()
store = AnimeVectorStore()

@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        indexed_documents=store.count(),
        environment=settings.ENVIRONMENT
    )

@router.get("/anime/{anime_id}")
async def get_anime_details(anime_id: int):
    result = store.collection.get(
        ids=[f"anilist_{anime_id}"],
        include=["metadatas", "documents"]
    )
    if not result or not result["ids"]:
        raise HTTPException(status_code=404, detail=f"Anime ID {anime_id} not found in index.")

    return {
        "id": anime_id,
        "metadata": result["metadatas"][0],
        "document": result["documents"][0]
    }
```

---

## 3. Deliverables & Acceptance Criteria
- [x] Liveness probe returning 200 OK and current document count.
- [x] Anime lookup by ID supporting popover modals.
