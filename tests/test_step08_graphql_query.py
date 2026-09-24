"""Unit tests for Step 08: AniList GraphQL Schema Discovery & Query Crafting."""

import pytest

from src.ingest import (
    ANILIST_GRAPHQL_URL,
    ANILIST_MEDIA_PAGE_QUERY,
    build_page_variables,
)


def test_anilist_graphql_url() -> None:
    """Verify the AniList GraphQL URL is correct and secure."""
    assert ANILIST_GRAPHQL_URL == "https://graphql.anilist.co"


def test_anilist_query_structure_and_mandatory_fields() -> None:
    """Verify all critical fields required for RAG document engineering exist in query."""
    query = ANILIST_MEDIA_PAGE_QUERY

    assert "query ($page: Int, $perPage: Int)" in query
    assert "Page(page: $page, perPage: $perPage)" in query
    assert "pageInfo" in query
    assert "hasNextPage" in query
    assert "currentPage" in query

    # Media identification
    assert "media(type: ANIME, sort: POPULARITY_DESC, isAdult: false)" in query
    assert "id" in query
    assert "idMal" in query

    # Multi-lingual titles
    assert "romaji" in query
    assert "english" in query
    assert "native" in query

    # Content & weighted tags
    assert "description(asHtml: false)" in query
    assert "genres" in query
    assert "tags {" in query
    assert "rank" in query
    assert "isMediaSpoiler" in query

    # Filtering metadata
    assert "averageScore" in query
    assert "popularity" in query
    assert "seasonYear" in query
    assert "format" in query
    assert "status" in query

    # Studios & Franchise relations
    assert "studios(isMain: true)" in query
    assert "relations {" in query
    assert "relationType" in query


def test_build_page_variables_defaults() -> None:
    """Verify default variables builder configuration."""
    vars_payload = build_page_variables()
    assert vars_payload == {"page": 1, "perPage": 50}


def test_build_page_variables_custom() -> None:
    """Verify custom variables configuration."""
    vars_payload = build_page_variables(page=5, per_page=25)
    assert vars_payload == {"page": 5, "perPage": 25}


def test_build_page_variables_validation() -> None:
    """Verify invalid inputs raise ValueError with appropriate bounds."""
    with pytest.raises(ValueError, match="Page must be >= 1"):
        build_page_variables(page=0)

    with pytest.raises(ValueError, match="per_page must be between 1 and 50"):
        build_page_variables(per_page=0)

    with pytest.raises(ValueError, match="per_page must be between 1 and 50"):
        build_page_variables(per_page=51)
