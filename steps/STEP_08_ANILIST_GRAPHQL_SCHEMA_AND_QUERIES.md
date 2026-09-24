# Step 08: AniList GraphQL Schema Discovery & Query Crafting

## 1. Process & Purpose
Craft the core GraphQL query to harvest the top 3,000 anime from AniList (`https://graphql.anilist.co`). The query extracts comprehensive metadata including titles in multiple languages, descriptions, scores, popularity, format, studios, community weighted tags, and franchise relations.

---

## 2. AniList GraphQL Query Specification

```graphql
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
```

---

## 3. Field Breakdown & Architectural Justification

| Field Name | Downstream Architectural Purpose |
|---|---|
| `media.id` | Unique primary key across vector store and metadata cache |
| `media.title { romaji, english }` | Multi-lingual title resolution and UI display |
| `media.tags { name, rank }` | Weighted signal for document text assembly (community rank 0–100) |
| `media.averageScore` / `popularity` | Numerical metadata filters & popularity bias calibration |
| `media.seasonYear` / `format` | Hard metadata filters (e.g. `seasonYear >= 2018`, `format == "TV"`) |
| `media.relations.edges` | Crucial for franchise lineage graphs and deduplication |
| `media.isAdult: false` | Default safety filter to exclude adult content from general corpus |

---

## 4. Deliverables & Acceptance Criteria
- [x] GraphQL query string defined as a reusable constant in `src/ingest/anilist_client.py`.
- [x] Validation against live AniList endpoint confirming schema compatibility.
