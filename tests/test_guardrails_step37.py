"""Tests for Spoiler Prevention & Safety Guardrails (Step 37)."""

import pytest
from src.generation.guardrails import validate_spoiler_free, sanitize_spoiler_text


def test_validate_spoiler_free_premise():
    safe_text = "A high school student discovers a notebook with supernatural powers."
    assert validate_spoiler_free(safe_text) is True


def test_validate_spoiler_free_detects_spoilers():
    spoiler_1 = "He dies in the end after sacrificing himself."
    spoiler_2 = "The mentor turns out to be the traitor in the final episode."
    spoiler_3 = "The plot twist is that the protagonist was a robot all along."

    assert validate_spoiler_free(spoiler_1) is False
    assert validate_spoiler_free(spoiler_2) is False
    assert validate_spoiler_free(spoiler_3) is False


def test_sanitize_spoiler_text():
    raw_text = "He dies at the end of the battle."
    sanitized = sanitize_spoiler_text(raw_text)
    assert "dies at the end" not in sanitized
    assert "[plot developments unfold]" in sanitized
