# Step 46: Interactive Sidebar Filters (Genres, Year Slider, Score, Format)

## 1. Process & Purpose
Provide an intuitive sidebar filter panel allowing users to manually customize hard constraints alongside their natural language prompts.

---

## 2. Sidebar Components

```python
# src/ui/sidebar.py
import streamlit as st

def render_sidebar_filters():
    st.sidebar.header("🎯 Recommendation Filters")

    # Genre Multi-select
    all_genres = [
        "Action", "Adventure", "Comedy", "Drama", "Fantasy", "Horror",
        "Mecha", "Mystery", "Psychological", "Romance", "Sci-Fi", 
        "Slice of Life", "Sports", "Supernatural", "Thriller"
    ]
    selected_genres = st.sidebar.multiselect("Genres", all_genres)

    # Year Range Slider
    year_range = st.sidebar.slider("Release Year Range", 1970, 2026, (2000, 2026))

    # Min Score Slider
    min_score = st.sidebar.slider("Minimum AniList Score", 0, 100, 70, step=5)

    # Format Checkboxes
    formats = st.sidebar.multiselect("Format", ["TV", "MOVIE", "OVA"], default=["TV", "MOVIE"])

    return {
        "genres": selected_genres,
        "year_min": year_range[0],
        "year_max": year_range[1],
        "min_score": min_score if min_score > 0 else None,
        "format": formats[0] if len(formats) == 1 else None
    }
```

---

## 3. Deliverables & Acceptance Criteria
- [x] Multi-attribute sidebar filtering UI integrated.
- [x] Filter states bound to query execution payload.
