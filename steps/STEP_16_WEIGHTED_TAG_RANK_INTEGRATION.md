# Step 16: Weighted Tag Rank Integration Engine

## 1. Process & Purpose
AniList tags carry community-voted relevance ranks from $0$ to $100$ (e.g. *military (86)*, *gore (80)*, *survival (90)*). A flat list of tags discards this signal. This step builds an algorithmic tag synthesizer that weights, filters, and formats tags into the embedding text according to their community rank.

---

## 2. Tag Weighting Architecture

```mermaid
graph LR
    ALL_TAGS[Raw Community Tags: 30+ items] --> FILTER_MIN[Filter Rank >= 40 & isSpoiler == False]
    FILTER_MIN --> TOP_N[Select Top 10 Ranked Tags]
    TOP_N --> SYNTH[Format as 'tag_name (rank)']
    SYNTH --> OUT["survival (90), military (86), gore (80), dark fantasy (78)"]
```

---

## 3. Implementation Code

```python
# src/ingest/tag_weighting.py
from typing import List
from src.ingest.transform import SanitizedTag

def format_weighted_tags(tags: List[SanitizedTag], min_rank: int = 40, max_tags: int = 10) -> str:
    """
    Filters and formats top-ranked non-spoiler tags with their weights.
    Example output: 'survival (90), military (86), gore (80), dark fantasy (78)'
    """
    valid_tags = [
        t for t in tags 
        if not t.is_spoiler and t.rank >= min_rank and t.name.strip()
    ]
    # Sort descending by rank
    valid_tags.sort(key=lambda x: x.rank, reverse=True)
    selected = valid_tags[:max_tags]
    
    if not selected:
        return "None"
        
    formatted_parts = [f"{t.name.lower()} ({t.rank})" for t in selected]
    return ", ".join(formatted_parts)
```

---

## 4. Deliverables & Acceptance Criteria
- [x] Ranks $< 40$ filtered to eliminate low-confidence community tags.
- [x] Spoiler tags strictly excluded from embedding generation.
- [x] Weighted formatted string output verified.
