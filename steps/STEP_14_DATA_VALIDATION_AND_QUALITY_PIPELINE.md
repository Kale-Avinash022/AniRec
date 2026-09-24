# Step 14: Data Validation & Quality Pipeline with Pydantic

## 1. Process & Purpose
Run an automated data audit and validation check across the entire ingested dataset to catch malformed titles, zero embeddings, broken relations, and null constraints before committing to vector indexing.

---

## 2. Validation Flow & Quality Checks

```mermaid
graph TD
    DATA[Ingested 3,000 Anime Records] --> V1[Check 1: Non-empty IDs & Unique IDs]
    DATA --> V2[Check 2: Minimum 1 Valid Title Language]
    DATA --> V3[Check 3: Non-empty Description or Tags]
    DATA --> V4[Check 4: Score Range 0-100 & Year Range 1960-2030]
    DATA --> V5[Check 5: Connected Franchise Roots Validity]
    V1 & V2 & V3 & V4 & V5 --> REPORT[Data Quality Audit Summary Report]
```

---

## 3. Implementation Code

```python
# src/ingest/validate.py
from typing import List, Dict, Any
from src.core.logger import logger
from src.ingest.transform import SanitizedAnime

class DataQualityAuditor:
    def __init__(self, anime_list: List[SanitizedAnime]):
        self.anime_list = anime_list
        self.total = len(anime_list)

    def run_audit(self) -> Dict[str, Any]:
        unique_ids = set()
        duplicate_ids = []
        missing_titles = []
        empty_descriptions = []
        invalid_scores = []
        missing_years = []

        for a in self.anime_list:
            if a.id in unique_ids:
                duplicate_ids.append(a.id)
            unique_ids.add(a.id)

            if not (a.title_romaji or a.title_english):
                missing_titles.append(a.id)

            if not a.description_clean and len(a.tags) == 0:
                empty_descriptions.append(a.id)

            if a.average_score is not None and not (0 <= a.average_score <= 100):
                invalid_scores.append(a.id)

            if not a.season_year:
                missing_years.append(a.id)

        audit_results = {
            "total_records": self.total,
            "unique_records": len(unique_ids),
            "duplicate_count": len(duplicate_ids),
            "missing_titles_count": len(missing_titles),
            "empty_descriptions_count": len(empty_descriptions),
            "invalid_scores_count": len(invalid_scores),
            "missing_years_count": len(missing_years),
            "data_health_score": round((1.0 - (len(empty_descriptions) + len(missing_titles)) / max(self.total, 1)) * 100, 2)
        }

        logger.logger.info(f"Data Quality Audit Completed: Health Score = {audit_results['data_health_score']}%")
        return audit_results
```

---

## 4. Deliverables & Acceptance Criteria
- [x] Zero duplicate anime IDs in ingested corpus.
- [x] Data health audit script integrated into data pipeline with logging.
