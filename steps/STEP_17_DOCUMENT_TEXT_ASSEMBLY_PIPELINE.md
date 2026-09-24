# Step 17: Document Text Formatter & Context Assembly

## 1. Process & Purpose
Combine all extracted fields (titles, release year, format, episode count, genres, weighted tags, studio, synopsis, and scores) into a single dense, information-rich document string tailored for semantic embedding models.

---

## 2. Document Template Specification

```
{Primary Title} ({Alternate Title}), {Year}, {Format}, {Episodes} episodes.
Genres: {Comma-separated genres}.
Tags: {Top weighted ranked tags}.
Studio: {Main studio}.
Description: {Cleaned synopsis}.
Average score: {Score}/100. Popularity: {Popularity}.
```

---

## 3. Concrete Example
```text
Attack on Titan (Shingeki no Kyojin), 2013, TV, 25 episodes.
Genres: Action, Drama, Fantasy.
Tags: survival (90), military (86), gore (80), dark fantasy (78), shounen (75).
Studio: Wit Studio.
Description: Several hundred years ago, humans were nearly exterminated by titans. Titans are typically several stories tall, seem to have no intelligence, devour human beings and, worst of all, seem to do it for the pleasure rather than as a food source...
Average score: 84/100. Popularity: 600000.
```

---

## 4. Implementation Code

```python
# src/ingest/document_builder.py
from typing import Dict, Any
from src.ingest.transform import SanitizedAnime
from src.ingest.tag_weighting import format_weighted_tags

def build_embedding_document(anime: SanitizedAnime) -> str:
    # Title line
    title_parts = [anime.title_romaji]
    if anime.title_english and anime.title_english.lower() != anime.title_romaji.lower():
        title_parts.append(f"({anime.title_english})")
    title_header = " ".join(title_parts)
    
    year_str = str(anime.season_year) if anime.season_year else "Unknown Year"
    format_str = anime.format or "Unknown Format"
    episodes_str = f"{anime.episodes} episodes" if anime.episodes else "Unknown episodes"
    
    genres_str = ", ".join(anime.genres) if anime.genres else "General"
    tags_str = format_weighted_tags(anime.tags, min_rank=40, max_tags=10)
    studios_str = ", ".join(anime.studios) if anime.studios else "Unknown Studio"
    score_str = f"{anime.average_score}/100" if anime.average_score else "Unrated"
    
    doc = (
        f"{title_header}, {year_str}, {format_str}, {episodes_str}.\n"
        f"Genres: {genres_str}.\n"
        f"Tags: {tags_str}.\n"
        f"Studio: {studios_str}.\n"
        f"Description: {anime.description_clean}\n"
        f"Average score: {score_str}. Popularity: {anime.popularity}."
    )
    return doc.strip()
```

---

## 5. Deliverables & Acceptance Criteria
- [x] Deterministic document assembly function accepting `SanitizedAnime`.
- [x] Output string validated to be within token limits of target embedding models (< 512 tokens).
