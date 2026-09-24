"""AniList GraphQL query definitions and query variable builders.

Step 08: AniList GraphQL Schema Discovery & Query Crafting.
Provides production-grade GraphQL queries for paginated extraction of anime
metadata, weighted tags, media metrics, and franchise relationship graphs.
"""

from typing import Any

ANILIST_GRAPHQL_URL: str = "https://graphql.anilist.co"

ANILIST_MEDIA_PAGE_QUERY: str = """
query ($page: Int, $perPage: Int) {
  Page(page: $page, perPage: $perPage) {
    pageInfo {
      hasNextPage
      currentPage
      lastPage
      total
    }
    media(type: ANIME, sort: POPULARITY_DESC, isAdult: false) {
      id
      idMal
      title {
        romaji
        english
        native
      }
      description(asHtml: false)
      genres
      tags {
        id
        name
        description
        rank
        isMediaSpoiler
      }
      averageScore
      meanScore
      popularity
      favourites
      seasonYear
      season
      format
      episodes
      duration
      status
      studios(isMain: true) {
        nodes {
          id
          name
        }
      }
      coverImage {
        extraLarge
        large
        medium
      }
      bannerImage
      siteUrl
      relations {
        edges {
          relationType
          node {
            id
            title {
              romaji
              english
            }
            format
            type
          }
        }
      }
    }
  }
}
""".strip()


def build_page_variables(page: int = 1, per_page: int = 50) -> dict[str, Any]:
    """Build standardized GraphQL variables payload for AniList page harvesting.

    Args:
        page: The 1-indexed page number to request.
        per_page: Number of items per page (AniList max is 50).

    Returns:
        Dictionary mapping variable names to their typed values.
    """
    if page < 1:
        raise ValueError(f"Page must be >= 1, got {page}")
    if not (1 <= per_page <= 50):
        raise ValueError(f"per_page must be between 1 and 50, got {per_page}")

    return {
        "page": page,
        "perPage": per_page,
    }
