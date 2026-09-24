# Step 47: "Why This Recommendation?" Explainability Expander Panel

## 1. Process & Purpose
RAG engines in interviews and portfolios stand out when the inner retrieval mechanics are made fully visible. This step implements an expandable UI panel beneath every recommendation turn, detailing candidate similarity scores, reranker logits, and matching community tags.

---

## 2. Explainability Panel Component

```python
# src/ui/explainability.py
import streamlit as st
from typing import List, Dict, Any

def render_provenance_expander(candidates: List[Dict[str, Any]]):
    if not candidates:
        return

    with st.expander("🔍 Why were these shows selected? (Retrieval Provenance)"):
        st.markdown("**Top Retrieved & Rescored Candidates from Vector Store:**")
        for i, c in enumerate(candidates):
            title = c.get("title", "Unknown")
            sim = c.get("similarity_score", 0.0)
            rerank = c.get("rerank_score")
            tags = c.get("matched_tags", "")

            rerank_badge = f" | **Cross-Encoder Score:** `{rerank:.2f}`" if rerank is not None else ""
            st.markdown(f"- **{i+1}. {title}** — *Vector Sim:* `{sim:.2f}`{rerank_badge}")
            if tags:
                st.caption(f"Matched Tags: {tags}")
```

---

## 3. Deliverables & Acceptance Criteria
- [x] Transparent explainability expander rendered per assistant response.
- [x] Clear visualization of cosine similarity and cross-encoder scores.
