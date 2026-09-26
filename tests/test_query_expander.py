"""Unit tests for Domain-Aware Query Expander (query_expander.py)."""

import pytest
from src.retrieval.query_expander import expand_query, EXPANSION_TABLE


def test_no_trigger_returns_original():
    query = "xyz random nonsense"
    assert expand_query(query) == query


def test_dark_political_fantasy_expansion():
    query = "dark political fantasy with military strategy and war"
    expanded = expand_query(query)
    assert "dystopian" in expanded or "tragedy" in expanded
    assert "military" in expanded or "empire" in expanded
    assert query in expanded


def test_cyberpunk_expansion():
    query = "cyberpunk thriller"
    expanded = expand_query(query)
    assert "futuristic" in expanded or "sci-fi" in expanded or "neon" in expanded


def test_max_expansion_terms():
    query = "dark political fantasy military sci-fi romance comedy space"
    expanded = expand_query(query, max_expansion_terms=5)
    added_terms = expanded.replace(query, "").strip().split()
    assert len(added_terms) <= 5


def test_regex_patterns_valid():
    import re
    for pattern, terms in EXPANSION_TABLE:
        re.compile(pattern, re.IGNORECASE)
        assert len(terms) > 0
