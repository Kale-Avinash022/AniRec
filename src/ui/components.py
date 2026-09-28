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


# Default high-contrast anime placeholder poster when image is unavailable
FALLBACK_COVER_URL = "https://placehold.co/230x320/1f2937/f9fafb?text=AniRec"


def render_cover_cards_grid(
    st_module: Any,
    candidates: list[dict[str, Any]],
    cols_per_row: int = 5,
) -> None:
    """Render a responsive cover-image grid for a list of suggested anime."""
    if not candidates:
        return

    st_module.markdown("")
    actual_cols = min(max(1, len(candidates)), cols_per_row)

    for row_start in range(0, len(candidates), actual_cols):
        row = candidates[row_start : row_start + actual_cols]
        cols = st_module.columns(len(row))
        for col, show in zip(cols, row, strict=False):
            with col:
                title = show.get("title") or show.get("title_romaji") or "Unknown Anime"
                year = show.get("year") or show.get("season_year") or ""
                fmt = show.get("format") or ""
                score = show.get("score") or show.get("average_score")
                aid = show.get("anilist_id") or show.get("id") or ""
                site_url = show.get("site_url") or (f"https://anilist.co/anime/{aid}" if aid else "")
                cover = show.get("cover_image_url") or FALLBACK_COVER_URL

                st_module.image(cover, use_container_width=True)
                meta_parts = [p for p in [str(year) if year else "", str(fmt) if fmt else ""] if p]
                score_str = f" · ⭐{score}" if score else ""
                meta_line = f"{' · '.join(meta_parts)}{score_str}"

                link_target = site_url if site_url else "#"
                st_module.markdown(
                    f"**[{title}]({link_target})**  \n"
                    f"<span style='font-size:0.75rem;color:#888'>{meta_line}</span>",
                    unsafe_allow_html=True,
                )


def render_aligned_recommendation_card(
    st_module: Any,
    rank: int,
    show: dict[str, Any],
    description_override: str | None = None,
) -> None:
    """Render a single suggestion with its poster picture aligned on the left and full 100-150 word description on the right."""
    title = show.get("title") or show.get("title_romaji") or "Unknown Anime"
    year = show.get("year") or show.get("season_year") or ""
    fmt = show.get("format") or ""
    score = show.get("score") or show.get("average_score")
    aid = show.get("anilist_id") or show.get("id") or ""
    site_url = show.get("site_url") or (f"https://anilist.co/anime/{aid}" if aid else "")
    cover = show.get("cover_image_url") or FALLBACK_COVER_URL
    genres = show.get("genres") or ""
    tags = show.get("tags") or show.get("matched_tags") or show.get("top_tags") or ""

    desc = description_override or show.get("why") or show.get("synopsis") or ""

    col1, col2 = st_module.columns([1, 4])

    with col1:
        st_module.image(cover, use_container_width=True)

    with col2:
        meta_parts = []
        if year:
            meta_parts.append(f"📅 **{year}**")
        if fmt:
            meta_parts.append(f"🎞️ **{fmt}**")
        if score:
            meta_parts.append(f"⭐ **{score}/100**")

        st_module.markdown(f"### {rank}. [{title}]({site_url})")
        if meta_parts:
            st_module.markdown(
                f"<span style='font-size:0.85rem;color:#6c757d'>{' · '.join(meta_parts)}</span>",
                unsafe_allow_html=True,
            )

        matched_items = []
        if tags:
            matched_items.extend([t.strip() for t in str(tags).split(",")[:4] if t.strip()])
        elif genres:
            matched_items.extend([g.strip() for g in str(genres).split(",")[:4] if g.strip()])

        if matched_items:
            tag_badges = " ".join([f"`{item}`" for item in matched_items])
            st_module.markdown(f"🏷️ {tag_badges}")

        st_module.write(desc)

    st_module.divider()


def parse_streamed_suggestions(content: str) -> tuple[str, dict[int, str]]:
    """Extract introductory message and individual numbered recommendation descriptions."""
    import re

    pattern = (
        r"(?:^|\n)\s*(\d{1,2})\.\s+(?:\*\*(.*?)\*\*|([^*\n—–]+?))\s*(?:\(\d{4}\))?\s*[—–-]\s*(.*?)"
        r"(?=(?:\n\s*\d{1,2}\.\s+)|$)"
    )
    matches = list(re.finditer(pattern, content, re.DOTALL))

    if not matches:
        return content, {}

    first_match_start = matches[0].start()
    intro_text = content[:first_match_start].strip()

    descriptions_by_rank: dict[int, str] = {}
    for m in matches:
        rank = int(m.group(1))
        desc = m.group(4).strip()
        descriptions_by_rank[rank] = desc

    return intro_text, descriptions_by_rank


def render_recommendation_card(
    st_module: Any,
    show: dict[str, Any],
) -> None:
    """Render a single formatted recommendation card with cover image & justification."""
    render_aligned_recommendation_card(st_module, rank=1, show=show)


def render_comparison_view(
    st_module: Any,
    candidates: list[dict[str, Any]],
    comparison_markdown: str,
) -> None:
    """Render a head-to-head comparison view with side-by-side anime poster cards and analytical breakdown."""
    if len(candidates) >= 2:
        c1, c2 = candidates[0], candidates[1]

        # Display side-by-side comparison poster cards
        col1, col_vs, col2 = st_module.columns([4, 1, 4])

        with col1:
            title1 = c1.get("title") or c1.get("title_romaji") or "Show A"
            aid1 = c1.get("anilist_id") or c1.get("id") or ""
            site_url1 = c1.get("site_url") or (f"https://anilist.co/anime/{aid1}" if aid1 else "")
            cover1 = c1.get("cover_image_url") or FALLBACK_COVER_URL
            score1 = c1.get("score") or c1.get("average_score")
            year1 = c1.get("year") or c1.get("season_year") or ""
            fmt1 = c1.get("format") or "TV"
            genres1 = c1.get("genres") or ""

            st_module.image(cover1, use_container_width=True)
            meta1 = f"📅 {year1} · 🎞️ {fmt1}" + (f" · ⭐ {score1}/100" if score1 else "")
            st_module.markdown(
                f"#### [{title1}]({site_url1})\n<span style='font-size:0.85rem;color:#6c757d'>{meta1}</span>",
                unsafe_allow_html=True,
            )
            if genres1:
                g_pills = " ".join([f"`{g.strip()}`" for g in str(genres1).split(",")[:3] if g.strip()])
                st_module.markdown(f"🏷️ {g_pills}")

        with col_vs:
            st_module.markdown(
                "<div style='text-align:center;padding-top:80px;font-size:1.8rem;font-weight:bold;color:#FF4B4B;'>⚔️<br>VS</div>",
                unsafe_allow_html=True,
            )

        with col2:
            title2 = c2.get("title") or c2.get("title_romaji") or "Show B"
            aid2 = c2.get("anilist_id") or c2.get("id") or ""
            site_url2 = c2.get("site_url") or (f"https://anilist.co/anime/{aid2}" if aid2 else "")
            cover2 = c2.get("cover_image_url") or FALLBACK_COVER_URL
            score2 = c2.get("score") or c2.get("average_score")
            year2 = c2.get("year") or c2.get("season_year") or ""
            fmt2 = c2.get("format") or "TV"
            genres2 = c2.get("genres") or ""

            st_module.image(cover2, use_container_width=True)
            meta2 = f"📅 {year2} · 🎞️ {fmt2}" + (f" · ⭐ {score2}/100" if score2 else "")
            st_module.markdown(
                f"#### [{title2}]({site_url2})\n<span style='font-size:0.85rem;color:#6c757d'>{meta2}</span>",
                unsafe_allow_html=True,
            )
            if genres2:
                g_pills2 = " ".join([f"`{g.strip()}`" for g in str(genres2).split(",")[:3] if g.strip()])
                st_module.markdown(f"🏷️ {g_pills2}")

        st_module.markdown("---")

    # Render the full comparative breakdown
    st_module.markdown(comparison_markdown)

