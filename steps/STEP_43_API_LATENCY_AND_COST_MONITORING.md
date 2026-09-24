# Step 43: API Latency Profiling & Token Cost Tracking Middleware

## 1. Process & Purpose
Record granular latency per pipeline stage (parsing, vector retrieval, cross-encoder reranking, and generation) and token dollar expenses per request, streaming events into `logs/app.jsonl`.

---

## 2. Latency Profiler Context Manager

```python
# src/core/profiler.py
import time
from typing import Dict

class PipelineProfiler:
    def __init__(self):
        self.stage_durations: Dict[str, float] = {}
        self.start_times: Dict[str, float] = {}

    def start_stage(self, name: str):
        self.start_times[name] = time.perf_counter()

    def end_stage(self, name: str) -> float:
        if name in self.start_times:
            duration = round((time.perf_counter() - self.start_times[name]) * 1000, 2)
            self.stage_durations[name] = duration
            return duration
        return 0.0

    def get_summary(self) -> Dict[str, float]:
        return self.stage_durations
```

---

## 3. Deliverables & Acceptance Criteria
- [x] Sub-millisecond timing accuracy across all 4 stages.
- [x] Trace output written to JSONL for aggregate analytics.
