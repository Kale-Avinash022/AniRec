# Step 49: Golden Dataset Authoring (50 Domain-Expert Q&A Pairs)

## 1. Process & Purpose
Author a robust, ground-truth evaluation dataset consisting of 50 carefully constructed question and expected-answer pairs across 5 distinct query categories. This benchmark serves as the objective measuring tape for all retrieval and generation experiments.

---

## 2. Category Distribution Table

| Category | Description | Target Count | Example Query | Expected Shows (IDs / Titles) |
|---|---|---|---|---|
| **Mood / Semantic** | Complex thematic, psychological, or aesthetic descriptions | 20 | *"FMAB's politics but darker"* | Vinland Saga, Code Geass, Shinsekai Yori |
| **Constraint-Heavy** | Strict filters on year, rating, and format | 10 | *"2018+, romance, score >= 75, TV only"* | Kaguya-sama: Love Is War, Horimiya |
| **Hybrid** | Mixed thematic intent and hard attributes | 10 | *"like Death Note — thriller, but completed and short"* | Erased (Boku dake ga Inai Machi), The Promised Neverland S1 |
| **Negative / Adversarial** | Queries asking for bad shows or nonsensical combinations | 5 | *"recommend a terrible anime similar to Steins;Gate"* | Rejection / Graceful explanation |
| **Franchise Edge Cases** | Requests looking for standalone shows without sequel spam | 5 | *"like Attack on Titan but not the sequels or OVAs"* | Kabaneri of the Iron Fortress, 86 Eighty-Six |

---

## 3. Golden Set JSON Schema (`evals/golden_set.json`)

```json
[
  {
    "id": "eval_01",
    "category": "mood_semantic",
    "query": "I want FMAB's politics and worldbuilding but darker and more morally grey",
    "expected_anilist_ids": [101348, 1575, 13125],
    "expected_titles": ["Vinland Saga", "Code Geass", "Shinsekai Yori"],
    "required_filters": {}
  },
  {
    "id": "eval_02",
    "category": "constraint_heavy",
    "query": "2018+, romance comedy, score at least 75, TV series only",
    "expected_anilist_ids": [101921, 102288],
    "expected_titles": ["Kaguya-sama: Love Is War", "Horimiya"],
    "required_filters": {
      "year_min": 2018,
      "format": "TV",
      "min_score": 75,
      "genres": ["Romance", "Comedy"]
    }
  }
]
```

---

## 4. Deliverables & Acceptance Criteria
- [x] Complete 50-pair golden benchmark dataset created in `evals/golden_set.json`.
- [x] Balanced coverage across all 5 query archetypes.
