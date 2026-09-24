# Step 06: Logging, Structured Tracing & Observability Foundations

## 1. Process & Purpose
Build a high-performance, structured logging and request-tracing framework to record query latency per pipeline stage (parsing, vector retrieval, reranking, generation), input/output token usage, calculated monetary costs, and anti-hallucination post-check violations.

---

## 2. Observability Record Schema (JSONL)

```json
{
  "request_id": "req_8f3d1b9a",
  "timestamp": "2026-10-05T14:32:01.450Z",
  "user_query": "I want FMAB's politics but darker, 2015 or later",
  "parsed_filters": {
    "year_min": 2015,
    "genres": ["Drama", "Action"]
  },
  "retrieved_candidates": [
    {"id": 16498, "title": "Attack on Titan", "score": 0.88},
    {"id": 21459, "title": "My Hero Academia", "score": 0.72}
  ],
  "reranked_top_k": [
    {"id": 16498, "title": "Attack on Titan", "rerank_score": 4.12}
  ],
  "recommended_titles": ["Attack on Titan Season 3", "Vinland Saga"],
  "hallucination_detected": false,
  "stage_latencies_ms": {
    "query_understanding": 12,
    "vector_retrieval": 45,
    "reranking": 110,
    "generation": 1420,
    "total": 1587
  },
  "tokens": {
    "prompt_tokens": 1240,
    "completion_tokens": 280,
    "total_tokens": 1520
  },
  "estimated_cost_usd": 0.00035
}
```

---

## 3. Implementation: Structured Logger

```python
# src/core/logger.py
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
from src.core.config import settings

class StructuredLogger:
    def __init__(self, log_filename: str = "app.jsonl"):
        self.log_path = settings.LOGS_DIR / log_filename
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Standard console logger
        self.logger = logging.getLogger("anirec")
        self.logger.setLevel(logging.DEBUG if settings.DEBUG else logging.INFO)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s")
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

    def log_request_event(self, event_data: Dict[str, Any]) -> None:
        event_data["timestamp"] = datetime.now(timezone.utc).isoformat()
        json_line = json.dumps(event_data)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json_line + "\n")
        self.logger.info(f"Processed Request {event_data.get('request_id')} in {event_data.get('stage_latencies_ms', {}).get('total', 0)}ms")

logger = StructuredLogger()
```

---

## 4. Cost Calculator Implementation

```python
# src/core/cost.py
PRICING_PER_1M_TOKENS = {
    "gpt-4o-mini": {"prompt": 0.150, "completion": 0.600},
    "text-embedding-3-small": {"prompt": 0.020, "completion": 0.000},
    "llama3.1:8b": {"prompt": 0.0, "completion": 0.0},
    "sentence-transformers/all-MiniLM-L6-v2": {"prompt": 0.0, "completion": 0.0}
}

def calculate_query_cost(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    rates = PRICING_PER_1M_TOKENS.get(model_name, {"prompt": 0.0, "completion": 0.0})
    prompt_cost = (prompt_tokens / 1_000_000) * rates["prompt"]
    completion_cost = (completion_tokens / 1_000_000) * rates["completion"]
    return round(prompt_cost + completion_cost, 6)
```

---

## 5. Deliverables & Acceptance Criteria
- [x] JSONL streaming event logger initialized with automatic folder creation.
- [x] Cost calculator module mapping token usage to dollar metrics.
- [x] Timing decorator / context manager for measuring per-stage latencies.
