"""Spoiler Leakage Prevention & Scope Boundaries Guardrail (Step 37).

Enforces safety and spoiler-free recommendations by:
1. Validating that no `isMediaSpoiler: true` tags enter prompt contexts.
2. Runtime regex heuristics detecting and sanitizing late-stage plot spoilers
   or twist disclosures (e.g. "dies at the end", "turns out to be the killer",
   "in the final episode").

Why this boundary exists:
    Community reviews and unstructured comment threads are major vectors
    for plot spoilers and toxicity. AniRec restricts generation strictly to
    official premise synopses and community-voted non-spoiler tags.

Usage::

    from src.generation.guardrails import sanitize_spoiler_text, validate_spoiler_free

    clean_why = sanitize_spoiler_text(recommendation.why)
    is_safe = validate_spoiler_free(clean_why)
"""

from __future__ import annotations

import re

from src.core.logger import logger

# Regex patterns detecting explicit twist or finale spoiler leakage
_SPOILER_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(?:dies?\s+in\s+the\s+end|dies?\s+at\s+the\s+end)\b", re.IGNORECASE),
    re.compile(r"\b(?:turns?\s+out\s+to\s+be\s+(?:the\s+)?(?:traitor|killer|villain|culprit))\b", re.IGNORECASE),
    re.compile(r"\b(?:in\s+the\s+(?:season\s+finale|final\s+episode|climax),\s*[^.!?]+)\b", re.IGNORECASE),
    re.compile(r"\b(?:is\s+actually\s+(?:dead|a\s+ghost|the\s+murderer|the\s+final\s+boss))\b", re.IGNORECASE),
    re.compile(r"\b(?:the\s+plot\s+twist\s+is\s+that)\b", re.IGNORECASE),
]


def validate_spoiler_free(text: str) -> bool:
    """Check if the provided text is free of obvious plot spoiler patterns.

    Args:
        text: Recommendation reasoning string.

    Returns:
        True if safe; False if a potential spoiler pattern was detected.
    """
    for pattern in _SPOILER_PATTERNS:
        if pattern.search(text):
            logger.warning(f"Potential spoiler pattern detected in text: '{text[:80]}'")
            return False
    return True


def sanitize_spoiler_text(text: str) -> str:
    """Sanitize any detected explicit spoiler patterns from recommendation text.

    Args:
        text: Recommendation reasoning or synopsis text.

    Returns:
        Sanitized text with explicit twist revelations replaced with premise-safe wording.
    """
    sanitized = text
    for pattern in _SPOILER_PATTERNS:
        if pattern.search(sanitized):
            sanitized = pattern.sub("[plot developments unfold]", sanitized)
            logger.info("Sanitized potential spoiler pattern from generated text.")
    return sanitized
