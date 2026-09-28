"""Query Understanding — Rules-Based / Regex Filter Parser (Step 25).

Fast, zero-cost first-pass parser that extracts structured metadata filters
from natural language anime queries using deterministic regex patterns.

Examples handled:
    "psychological thrillers 2018 or later TV only"
        → semantic_text="psychological thrillers"
          filters=ExtractedFilters(year_min=2018, format="TV", genres=["Psychological"])

    "action anime after 2015 score >= 80"
        → semantic_text="action anime"
          filters=ExtractedFilters(year_min=2015, min_score=80, genres=["Action"])

    "90s romance movie"
        → semantic_text="romance"
          filters=ExtractedFilters(year_max=1999, format="MOVIE", genres=["Romance"])
"""

from __future__ import annotations

import re
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ExtractedFilters(BaseModel):
    """Structured metadata filters extracted from a user query."""

    year_min: Optional[int] = None   # e.g. 2015 → season_year >= 2015
    year_max: Optional[int] = None   # e.g. 1999 → season_year <= 1999
    format: Optional[str] = None     # "TV" | "MOVIE" | "OVA" | "ONA" | "SPECIAL"
    min_score: Optional[int] = None  # e.g. 80 → average_score >= 80
    genres: list[str] = Field(default_factory=list)

    def is_empty(self) -> bool:
        """Return True when no constraints have been extracted."""
        return (
            self.year_min is None
            and self.year_max is None
            and self.format is None
            and self.min_score is None
            and not self.genres
        )


class ParsedQuery(BaseModel):
    """Output of the rules-based query parser."""

    raw_query: str
    semantic_text: str       # cleaned text sent to the embedding model
    filters: ExtractedFilters
    is_comparison: bool = False
    comparison_titles: list[str] = Field(default_factory=list)
    intent: str = "recommendation"  # "recommendation" | "comparison" | "qa"



# ---------------------------------------------------------------------------
# Genre vocabulary (AniList canonical names)
# ---------------------------------------------------------------------------

KNOWN_GENRES: list[str] = [
    "Action",
    "Adventure",
    "Comedy",
    "Drama",
    "Ecchi",
    "Fantasy",
    "Horror",
    "Mahou Shoujo",
    "Mecha",
    "Music",
    "Mystery",
    "Psychological",
    "Romance",
    "Sci-Fi",
    "Slice of Life",
    "Sports",
    "Supernatural",
    "Thriller",
]

# ---------------------------------------------------------------------------
# Compiled regex patterns (compiled once at import time)
# ---------------------------------------------------------------------------

# Year-min: "2015+", ">2015", ">= 2015", "after 2015", "2015 or later", "from 2015"
_RE_YEAR_MIN = re.compile(
    r"(?:>=|>)\s*(20\d\d|19\d\d)|\b(20\d\d|19\d\d)\s*\+|\bafter\s+(20\d\d|19\d\d)|\b(20\d\d|19\d\d)\s+or\s+later|\bfrom\s+(20\d\d|19\d\d)",
    re.IGNORECASE,
)

# Year-max: "<2010", "<= 2005", "before 2010", "up to 2005", "until 2000"
_RE_YEAR_MAX = re.compile(
    r"(?:<=|<)\s*(20\d\d|19\d\d)|\bbefore\s+(20\d\d|19\d\d|\d{2}s?)|\bup\s+to\s+(20\d\d|19\d\d)|\buntil\s+(20\d\d|19\d\d)",
    re.IGNORECASE,
)

# Decade: "90s", "1990s", "80s"
_RE_DECADE = re.compile(r"\b(1?[6-9]0)s\b|\b(19[6-9]0)s\b|\b(20[0-2]0)s\b", re.IGNORECASE)

# Format matchers
_RE_TV = re.compile(r"\b(tv(?:\s+series)?|anime\s+series)\b", re.IGNORECASE)
_RE_MOVIE = re.compile(r"\b(movie|film|feature)\b", re.IGNORECASE)
_RE_OVA = re.compile(r"\bova\b", re.IGNORECASE)
_RE_ONA = re.compile(r"\bona\b", re.IGNORECASE)
_RE_SHORT = re.compile(r"\bshort\s+(?:episodes?|series)\b", re.IGNORECASE)

# Min score: "score >= 80", "score > 75", "score:75+", "80+ score", "rating >= 80"
_RE_SCORE = re.compile(
    r"(?:(?:score|rating)\s*(?:>=|>|above|over|:)?\s*(\d{2,3})\+?)|(?:(\d{2,3})\s*\+\s*(?:score|rating))",
    re.IGNORECASE,
)

# Episode count: "12 episodes", "short 12 ep", "under 25 episodes"
_RE_EPISODES = re.compile(r"\b(\d+)\s+ep(?:isodes?)?\b", re.IGNORECASE)


# Comparison regex patterns
_RE_COMPARE_PREFIX = re.compile(
    r"^(?:please\s+)?(?:can\s+you\s+)?(?:compare|give\s+a\s+comparison\s+(?:of|between)|difference\s+between|differences\s+between)\s+",
    re.IGNORECASE,
)
_RE_COMPARE_VS = re.compile(
    r"['\"`]?([^'\"`]+?)['\"`]?\s+(?:\bvs\.?\b|\bversus\b|\bcompared\s+to\b)\s+['\"`]?([^'\"`]+?)['\"`]?$",
    re.IGNORECASE,
)
_RE_COMPARE_AND = re.compile(
    r"['\"`]?([^'\"`]+?)['\"`]?\s+(?:\band\b|\bwith\b)\s+['\"`]?([^'\"`]+?)['\"`]?$",
    re.IGNORECASE,
)
_RE_WHICH_BETTER = re.compile(
    r"['\"`]?([^'\"`]+?)['\"`]?\s+(?:\bor\b)\s+['\"`]?([^'\"`]+?)['\"`]?\s*[,?]?\s*(?:which|what)\s+(?:is|one|should|better|to watch)\b",
    re.IGNORECASE,
)
_RE_IS_BETTER = re.compile(
    r"^(?:is\s+)?['\"`]?([^'\"`]+?)['\"`]?\s+(?:better\s+than|similar\s+to|different\s+from)\s+['\"`]?([^'\"`]+?)['\"`]?$",
    re.IGNORECASE,
)


def _clean_title(t: str) -> str:
    """Clean extracted anime title candidate."""
    t = re.sub(r"^(?:the\s+anime|the\s+show|anime)\s+", "", t.strip(), flags=re.IGNORECASE)
    t = re.sub(r"\s+(?:the\s+anime|the\s+show|anime)$", "", t.strip(), flags=re.IGNORECASE)
    return t.strip().strip("'\"`?.!,:;")


def extract_comparison_intent(query: str) -> tuple[bool, list[str]]:
    """Detect if query is a head-to-head anime comparison and extract target titles.

    Returns:
        (is_comparison, [title1, title2])
    """
    clean_q = query.strip().rstrip("?.!")

    # 1. "compare X and Y" / "difference between X and Y"
    m_prefix = _RE_COMPARE_PREFIX.match(clean_q)
    if m_prefix:
        rest = clean_q[m_prefix.end():].strip()
        m_and = _RE_COMPARE_AND.search(rest) or _RE_COMPARE_VS.search(rest)
        if m_and:
            t1, t2 = _clean_title(m_and.group(1)), _clean_title(m_and.group(2))
            if len(t1) >= 2 and len(t2) >= 2:
                return True, [t1, t2]

    # 2. Direct "vs" / "versus" pattern (e.g. "Eden Zero vs Fairy Tail", "Death Note versus Code Geass")
    m = _RE_COMPARE_VS.search(clean_q)
    if m:
        t1, t2 = _clean_title(m.group(1)), _clean_title(m.group(2))
        if len(t1) >= 2 and len(t2) >= 2:
            return True, [t1, t2]

    # 3. "X or Y, which is better?"
    m_or = _RE_WHICH_BETTER.search(clean_q)
    if m_or:
        t1, t2 = _clean_title(m_or.group(1)), _clean_title(m_or.group(2))
        if len(t1) >= 2 and len(t2) >= 2:
            return True, [t1, t2]

    # 4. "is X better than Y"
    m_better = _RE_IS_BETTER.search(clean_q)
    if m_better:
        t1, t2 = _clean_title(m_better.group(1)), _clean_title(m_better.group(2))
        if len(t1) >= 2 and len(t2) >= 2:
            return True, [t1, t2]

    return False, []


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------


def parse_query_rules(query: str) -> ParsedQuery:
    """Parse a natural language query into semantic text + structured filters.

    Args:
        query: Raw user input string.

    Returns:
        :class:`ParsedQuery` with ``semantic_text``, ``filters``, and comparison intent.
    """
    is_comp, comp_titles = extract_comparison_intent(query)
    intent = "comparison" if is_comp else "recommendation"

    text = query  # working copy that we'll strip tokens from
    filters = ExtractedFilters()

    # ── 1. Year-min ───────────────────────────────────────────────────────
    m = _RE_YEAR_MIN.search(text)
    if m:
        year_str = next(g for g in m.groups() if g)
        filters.year_min = int(year_str.strip("+"))
        text = text[: m.start()] + text[m.end():]

    # ── 2. Year-max ───────────────────────────────────────────────────────
    m = _RE_YEAR_MAX.search(text)
    if m:
        year_str = next(g for g in m.groups() if g)
        # Handle abbreviated decade ("before 90s" → before 1990 → max 1989)
        if len(year_str) == 3 and year_str.endswith("s"):
            decade = int(year_str[:2]) * 10
            filters.year_max = 1900 + decade - 1 if decade < 100 else decade - 1
        else:
            filters.year_max = int(year_str) - 1
        text = text[: m.start()] + text[m.end():]

    # ── 3. Decade shorthand (e.g. "90s anime") ───────────────────────────
    if filters.year_min is None and filters.year_max is None:
        m = _RE_DECADE.search(text)
        if m:
            decade_str = next(g for g in m.groups() if g)
            decade_int = int(decade_str)
            if decade_int < 100:
                decade_int = 1900 + decade_int
            filters.year_min = decade_int
            filters.year_max = decade_int + 9
            text = text[: m.start()] + text[m.end():]

    # ── 4. Format ─────────────────────────────────────────────────────────
    if _RE_TV.search(text):
        filters.format = "TV"
    elif _RE_MOVIE.search(text):
        filters.format = "MOVIE"
    elif _RE_OVA.search(text):
        filters.format = "OVA"
    elif _RE_ONA.search(text):
        filters.format = "ONA"

    # Strip format keywords from semantic text (avoid polluting embedding)
    for pat in (_RE_TV, _RE_MOVIE, _RE_OVA, _RE_ONA, _RE_SHORT):
        text = pat.sub("", text)

    # ── 5. Min score ──────────────────────────────────────────────────────
    m = _RE_SCORE.search(text)
    if m:
        score_val = m.group(1) or m.group(2)
        if score_val:
            filters.min_score = int(score_val)
            text = text[: m.start()] + text[m.end():]

    # ── 6. Genres ─────────────────────────────────────────────────────────
    for genre in KNOWN_GENRES:
        if re.search(rf"\b{re.escape(genre)}\b", query, re.IGNORECASE):
            if genre not in filters.genres:
                filters.genres.append(genre)

    # ── 7. Clean up residual whitespace ───────────────────────────────────
    semantic_text = re.sub(r"\s{2,}", " ", text).strip()
    if not semantic_text:
        semantic_text = query  # full fallback — never send empty string to embedder

    return ParsedQuery(
        raw_query=query,
        semantic_text=semantic_text,
        filters=filters,
        is_comparison=is_comp,
        comparison_titles=comp_titles,
        intent=intent,
    )

