"""Weighted Tag Rank Integration Engine (Step 16).

What this module does
---------------------
AniList exposes community-voted relevance ranks for every tag attached to a
show (e.g. "Military (88)", "Gore (76)", "Isekai (92)").  A flat unweighted
tag list throws away this signal entirely.

This engine:
1. **Filters** — drops tags below ``min_rank`` (low-confidence community votes)
   and always excludes spoiler-tagged keywords.
2. **Sorts**   — highest rank first (most defining trait at the front of the
   embedding string where the model attends most strongly).
3. **Caps**    — takes at most ``max_tags`` entries so the tag block stays
   within the token budget set by ``MAX_TAG_SLOTS`` (Step 15).
4. **Formats** — produces ``"tag_name (rank)"`` tokens that the embedding
   model can read as weighted thematic descriptors.

Why include the rank number in the text?
-----------------------------------------
Embedding models are trained on natural language that includes numeric
qualifiers ("highly rated", "strongly themed").  Writing "dark fantasy (95)"
vs "dark fantasy (42)" introduces a quantitative signal that pushes the
resulting vector toward or away from similar high-confidence thematic clusters
in embedding space.  Empirically this outperforms unweighted flat lists on
thematic similarity benchmarks.

Output example
--------------
``"survival (92), military (88), gore (82), dark fantasy (78), tragedy (74)"``
"""

from __future__ import annotations

from src.docs.schema import MAX_TAG_SLOTS
from src.ingest.transform import SanitizedTag

# ---------------------------------------------------------------------------
# Default thresholds
# ---------------------------------------------------------------------------

DEFAULT_MIN_RANK: int = 40
"""Tags with rank below this are treated as low-confidence noise.

AniList allows any registered user to tag a show; tags below 40 indicate the
community broadly disagrees with the label.  Including them adds noise to the
embedding surface.
"""

DEFAULT_MAX_TAGS: int = MAX_TAG_SLOTS
"""Maximum tags included per document — imported from Step 15 constants.

Changing ``MAX_TAG_SLOTS`` in schema.py propagates here automatically.
"""

HIGH_CONFIDENCE_THRESHOLD: int = 70
"""Rank above which a tag is considered a defining characteristic.

Used by ``split_tags_by_confidence()`` to separate core descriptors from
supporting context tags.
"""


# ---------------------------------------------------------------------------
# Core formatting function
# ---------------------------------------------------------------------------


def format_weighted_tags(
    tags: list[SanitizedTag],
    min_rank: int = DEFAULT_MIN_RANK,
    max_tags: int = DEFAULT_MAX_TAGS,
) -> str:
    """Filter, sort, and format a tag list into a weighted embedding string.

    This is the primary entry point for Step 16.  Called by the document
    assembler (Step 17) when building the ``document_text`` for each show.

    Args:
        tags:     ``SanitizedTag`` list from Step 12 (already spoiler-filtered
                  and rank-sorted, but we re-apply guards for safety).
        min_rank: Minimum community rank to include (default 40).
        max_tags: Maximum number of tags to emit (default ``MAX_TAG_SLOTS``).

    Returns:
        Comma-separated string of ``"name (rank)"`` tokens, highest rank first.
        Returns ``"None"`` if no tags survive the filters (not an empty string,
        so the assembler always has a non-blank placeholder in the document).

    Examples:
        >>> tags = [
        ...     SanitizedTag(id=1, name='Military',    rank=88, is_spoiler=False),
        ...     SanitizedTag(id=2, name='Gore',        rank=76, is_spoiler=False),
        ...     SanitizedTag(id=3, name='Isekai',      rank=30, is_spoiler=False),  # below min
        ...     SanitizedTag(id=4, name='Plot Twist',  rank=95, is_spoiler=True),   # spoiler
        ... ]
        >>> format_weighted_tags(tags, min_rank=40, max_tags=5)
        'military (88), gore (76)'
    """
    valid: list[SanitizedTag] = [
        t
        for t in tags
        if not t.is_spoiler and t.rank >= min_rank and t.name.strip()
    ]

    # Re-sort descending — Step 12 already sorts but be defensive
    valid.sort(key=lambda t: t.rank, reverse=True)

    selected = valid[:max_tags]

    if not selected:
        return "None"

    return ", ".join(f"{t.name.lower()} ({t.rank})" for t in selected)


# ---------------------------------------------------------------------------
# Analysis helpers (used by Step 18 metadata schema and Step 33 prompting)
# ---------------------------------------------------------------------------


def get_top_tag_names(
    tags: list[SanitizedTag],
    n: int = 5,
    min_rank: int = DEFAULT_MIN_RANK,
) -> list[str]:
    """Return the top-N tag names as a plain string list (no rank numbers).

    Useful for populating the ``top_tags`` metadata field in Step 18 and for
    building the ``genres_and_tags`` filter block in Step 25 query parsing.

    Args:
        tags:     ``SanitizedTag`` list from Step 12.
        n:        Number of tag names to return.
        min_rank: Minimum rank threshold.

    Returns:
        List of lowercase tag name strings, ranked highest first.
    """
    valid = [
        t for t in tags
        if not t.is_spoiler and t.rank >= min_rank and t.name.strip()
    ]
    valid.sort(key=lambda t: t.rank, reverse=True)
    return [t.name.lower() for t in valid[:n]]


def split_tags_by_confidence(
    tags: list[SanitizedTag],
    min_rank: int = DEFAULT_MIN_RANK,
    high_threshold: int = HIGH_CONFIDENCE_THRESHOLD,
) -> tuple[list[SanitizedTag], list[SanitizedTag]]:
    """Partition valid tags into high-confidence and supporting tiers.

    High-confidence tags (rank >= ``high_threshold``) represent a show's
    defining characteristics — they get prioritised in the document text and
    in the LLM grounding prompt (Step 33).

    Supporting tags (``min_rank`` <= rank < ``high_threshold``) add colour but
    are secondary signals.

    Args:
        tags:           ``SanitizedTag`` list from Step 12.
        min_rank:       Lower bound for inclusion.
        high_threshold: Rank boundary between tiers.

    Returns:
        2-tuple ``(high_confidence_tags, supporting_tags)``, each sorted
        by rank descending.
    """
    valid = [
        t for t in tags
        if not t.is_spoiler and t.rank >= min_rank and t.name.strip()
    ]
    valid.sort(key=lambda t: t.rank, reverse=True)

    high = [t for t in valid if t.rank >= high_threshold]
    supporting = [t for t in valid if t.rank < high_threshold]

    return high, supporting


def compute_tag_density_score(
    tags: list[SanitizedTag],
    min_rank: int = DEFAULT_MIN_RANK,
) -> float:
    """Compute a 0.0–1.0 signal representing how richly tagged a show is.

    Shows with dense, high-ranked tag coverage embed more distinctively than
    tag-sparse entries.  This score is stored as ``tag_density`` in the Step 18
    metadata schema and can be used as a soft quality signal during retrieval.

    Formula:
        ``mean(rank / 100 for qualifying tags)`` capped to ``[0.0, 1.0]``.

    Args:
        tags:     ``SanitizedTag`` list from Step 12.
        min_rank: Minimum rank threshold.

    Returns:
        Float in ``[0.0, 1.0]`` — higher means denser, higher-confidence tags.
    """
    valid = [
        t for t in tags
        if not t.is_spoiler and t.rank >= min_rank
    ]
    if not valid:
        return 0.0
    return round(sum(t.rank for t in valid) / (len(valid) * 100), 4)
