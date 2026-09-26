"""UI Components & Explainability Panels (Step 47).

Provides rich UI cards and explainability expanders for:
  - "Why were these shows selected?" candidate provenance inspector.
  - Recommended show display cards with cover images, score badges, and tag pills.
  - Relaxations audit banner when filters were loosened.
"""

from __future__ import annotations

from typing import Any


def render_provenance_expander(
    st_module: Any,
    candidates: list[dict[str, Any]],
    relaxations: list[str] | None = None,
) -> None:
    """Render the 'Why were these shows selected?' explainability expander."""
    if not candidates and not relaxations:
        return

    with st_module.expander("🔍 **Why were these shows selected? (Retrieval Provenance)**", expanded=False):
        if relaxations:
            st_module.info("⚡ **Search Relaxations Applied:**\n" + "\n".join(f"- {r}" for r in relaxations))

        st_module.markdown("#### 🎯 Candidate Scoring & Ranking Pipeline")
        st_module.caption(
            "Shows retrieved by ChromaDB vector similarity and re-ranked using "
            "`cross-encoder/ms-marco-MiniLM-L-6-v2` cross-attention."
        )

        for i, c in enumerate(candidates):
            title = c.get("title") or c.get("title_romaji") or f"Anime #{c.get('id', i+1)}"
            sim = c.get("similarity_score") or c.get("similarity", 0.0)
            rerank = c.get("rerank_score")
            tags = c.get("top_tags") or c.get("matched_tags") or c.get("genres", "")
            aid = c.get("anilist_id") or c.get("id")

            rerank_str = f" | ⚡ **Rerank Logit:** `{rerank:.2f}`" if rerank is not None else ""
            st_module.markdown(
                f"**{i+1}. [{title}](https://anilist.co/anime/{aid})** — "
                f"*Vector Similarity:* `{float(sim):.3f}`{rerank_str}"
            )
            if tags:
                st_module.caption(f"🏷️ *Attributes:* `{tags}`")


def render_recommendation_card(
    st_module: Any,
    show: dict[str, Any],
) -> None:
    """Render a single formatted recommendation card with cover image & justification."""
    title = show.get("title", "Unknown Title")
    year = show.get("year")
    format_type = show.get("format")
    score = show.get("score")
    why = show.get("why", "")
    matched_attrs = show.get("matched_attributes", [])
    cover_url = show.get("cover_image_url")
    site_url = show.get("site_url") or f"https://anilist.co/anime/{show.get('anilist_id', '')}"

    col1, col2 = st_module.columns([1, 4])

    with col1:
        if cover_url:
            st_module.image(cover_url, use_container_width=True)
        else:
            st_module.markdown("🎬 **[No Image]**")

    with col2:
        header_parts = [f"### [{title}]({site_url})"]
        meta_sub = []
        if year:
            meta_sub.append(f"📅 {year}")
        if format_type:
            meta_sub.append(f"🎞️ {format_type}")
        if score:
            meta_sub.append(f"⭐ **{score}/100**")

        st_module.markdown(" ".join(header_parts))
        if meta_sub:
            st_module.caption(" • ".join(meta_sub))

        st_module.write(why)

        if matched_attrs:
            badges = " ".join([f"`{attr}`" for attr in matched_attrs])
            st_module.markdown(f"🏷️ **Matched:** {badges}")

    st_module.divider()
