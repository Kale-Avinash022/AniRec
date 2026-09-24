# Step 18: Metadata Schema Construction for Vector Filtering

## 1. Process & Purpose
ChromaDB and vector databases require flat or semi-structured metadata alongside each vector to power fast, exact metadata filtering (e.g. `seasonYear >= 2015`, `genres contains 'Action'`). This step constructs the exact metadata payload attached to each vector.

---

## 2. Metadata Schema Architecture

```json
{
  "id": "anilist_16498",
  "text": "Attack on Titan (Shingeki no Kyojin), 2013, TV...",
  "metadata": {
    "anilist_id": 16498,
    "title_romaji": "Shingeki no Kyojin",
    "title_english": "Attack on Titan",
    "year": 2013,
    "format": "TV",
    "average_score": 84,
    "popularity": 600000,
    "franchise_root_id": 16498,
    "genres_str": "Action,Drama,Fantasy",
    "top_tags_str": "survival,military,gore,dark fantasy",
    "cover_image_url": "https://s4.anilist.co/file/anilistcdn/media/anime/cover/large/bx16498-..."
  }
}
```

---

## 3. Implementation Code

```python
# src/ingest/metadata_builder.py
from typing import Dict, Any
from src.ingest.transform import SanitizedAnime

def build_vector_metadata(anime: SanitizedAnime, franchise_root_id: int) -> Dict[str, Any]:
    # ChromaDB requires metadata values to be string, int, float, or bool
    top_tags = [t.name for t in anime.tags[:5]]
    
    return {
        "anilist_id": int(anime.id),
        "title_romaji": str(anime.title_romaji),
        "title_english": str(anime.title_english or ""),
        "year": int(anime.season_year) if anime.season_year else 0,
        "format": str(anime.format or "UNKNOWN"),
        "average_score": int(anime.average_score) if anime.average_score else 0,
        "popularity": int(anime.popularity),
        "franchise_root_id": int(franchise_root_id),
        "genres_str": ",".join(anime.genres),
        "top_tags_str": ",".join(top_tags),
        "cover_image_url": str(anime.cover_image_url or ""),
        "site_url": str(anime.site_url or "")
    }
```

---

## 4. Deliverables & Acceptance Criteria
- [x] Flat, type-safe metadata dictionary compatible with ChromaDB `where` filters.
- [x] Includes `franchise_root_id` for downstream deduplication.
