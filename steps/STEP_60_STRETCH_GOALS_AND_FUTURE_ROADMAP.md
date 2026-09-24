# Step 60: Advanced Stretch Goals Roadmap (AniList Profile Sync, pgvector, Langfuse)

## 1. Process & Purpose
Outline high-leverage post-v1 extensions that take AniRec from a strong project to a senior-level portfolio centerpiece.

---

## 2. Stretch Goal Feature Matrix

| Feature | Technical Implementation | Portfolio / Hiring Wow-Factor |
|---|---|---|
| **1. AniList Profile Import** | User inputs their AniList username $\rightarrow$ API fetches `MediaListGroup` watched IDs $\rightarrow$ injected into `exclude_ids` | **Live Demo Wow Factor**: Interviewer pastes their own username and instantly gets novel shows they haven't seen |
| **2. pgvector Migration** | Migrate Chroma to PostgreSQL with `pgvector` extension + HNSW index | Proves SQL, relational database management, and enterprise data modeling skills |
| **3. Langfuse / Tracing Dashboard** | Integrate `@observe` decorator to log LLM latency, token counts, and step traces to self-hosted Langfuse | Enterprise-grade observability and production AI monitoring |
| **4. Automated Weekly Catalog Cron** | GitHub Actions workflow executing weekly ingestion to harvest newly airing seasonal anime | Proves automated CI/CD and data pipeline maintenance thinking |
| **5. Preference Fine-Tuned Cross-Encoder** | Fine-tune cross-encoder reranker on collected `feedback.jsonl` thumbs-up/down data using pairwise contrastive loss | Machine Learning / RLHF practical modeling expertise |

---

## 3. AniList Profile Ingestion Code Preview

```python
# src/ingest/user_profile.py
import httpx

USER_WATCHLIST_QUERY = """
query ($userName: String) {
  MediaListCollection(userName: $userName, type: ANIME, status: COMPLETED) {
    lists {
      entries {
        mediaId
      }
    }
  }
}
"""

async def fetch_user_watched_ids(username: str) -> list[int]:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://graphql.anilist.co",
            json={"query": USER_WATCHLIST_QUERY, "variables": {"userName": username}}
        )
        if resp.status_code != 200:
            return []
        data = resp.json()
        lists = data.get("data", {}).get("MediaListCollection", {}).get("lists", [])
        watched_ids = []
        for l in lists:
            for entry in l.get("entries", []):
                watched_ids.append(entry.get("mediaId"))
        return watched_ids
```

---

## 4. Deliverables & Acceptance Criteria
- [x] Clear technical roadmap for senior-level post-v1 extensions.
- [x] Profile import GraphQL snippet ready for quick integration.
