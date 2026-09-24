# Step 44: Streamlit App Initialization & Session State Architecture

## 1. Process & Purpose
Initialize the frontend application with modern layout configurations, reactive session state management for message history, active filter selections, and candidate provenance inspection tabs.

---

## 2. Session State Model

```python
# State fields initialized in st.session_state:
# - messages: List[Dict[str, str]] (role, content)
# - provenance_map: Dict[int, List[Dict]] (turn_id -> retrieved shows)
# - active_filters: Dict (genres, year_range, min_score, format)
# - user_feedback: Dict (turn_id -> "up" | "down")
```

---

## 3. Implementation Code

```python
# src/ui/state.py
import streamlit as st

def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "👋 Hi! I'm AniRec. Tell me what kind of anime you're in the mood for (e.g. *'dark fantasy like Attack on Titan'* or *'90s cyberpunk'*), and I'll find the best matches!"}
        ]
    if "provenance_map" not in st.session_state:
        st.session_state.provenance_map = {}
    if "feedback_history" not in st.session_state:
        st.session_state.feedback_history = {}
```

---

## 4. Deliverables & Acceptance Criteria
- [x] Streamlit page configuration configured (`page_title="AniRec"`, wide layout).
- [x] Session state variables initialized safely on page refresh.
