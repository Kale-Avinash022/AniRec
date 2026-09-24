# Step 48: User Feedback Thumbs-Up/Down Logger

## 1. Process & Purpose
Capture explicit end-user feedback (thumbs up / down) on generated recommendation sets to enable RLHF-style offline data collection and reranking fine-tuning.

---

## 2. Implementation Code

```python
# src/ui/feedback.py
import json
from datetime import datetime, timezone
import streamlit as st
from src.core.config import settings

def render_feedback_buttons(turn_id: str, query: str, response_text: str):
    col1, col2, _ = st.columns([1, 1, 8])
    with col1:
        if st.button("👍 Helpful", key=f"up_{turn_id}"):
            log_user_feedback(turn_id, query, response_text, rating="positive")
            st.toast("Thank you for your feedback!")
    with col2:
        if st.button("👎 Poor", key=f"down_{turn_id}"):
            log_user_feedback(turn_id, query, response_text, rating="negative")
            st.toast("Feedback recorded!")

def log_user_feedback(turn_id: str, query: str, response_text: str, rating: str):
    feedback_file = settings.LOGS_DIR / "feedback.jsonl"
    record = {
        "turn_id": turn_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "response": response_text,
        "rating": rating
    }
    with open(feedback_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
```

---

## 3. Deliverables & Acceptance Criteria
- [x] Thumbs up/down interactive UI buttons integrated per chat turn.
- [x] Persisted feedback logs saved to `logs/feedback.jsonl`.
