"""Unit tests for Step 11: Fault-Tolerant Pagination & Retry Engine."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.ingest import AniListClient, save_page_snapshot


@pytest.fixture
def sample_page_payload() -> dict:
    """Fixture providing a valid AniList GraphQL response structure."""
    return {
        "data": {
            "Page": {
                "pageInfo": {
                    "hasNextPage": False,
                    "currentPage": 1,
                    "lastPage": 1,
                    "total": 1,
                },
                "media": [
                    {
                        "id": 1,
                        "title": {"romaji": "Cowboy Bebop", "english": "Cowboy Bebop"},
                        "description": "Bounty hunters in space.",
                        "averageScore": 89,
                    }
                ],
            }
        }
    }


@pytest.mark.asyncio
async def test_fetch_page_from_cache(tmp_path: Path, sample_page_payload: dict) -> None:
    """Verify that if snapshot exists on disk, no HTTP call is triggered."""
    save_page_snapshot(1, sample_page_payload, data_dir=tmp_path)

    client = AniListClient(data_dir=tmp_path)

    with patch("httpx.AsyncClient.post") as mock_post:
        data = await client.fetch_page(page=1)
        assert data == sample_page_payload
        mock_post.assert_not_called()


@pytest.mark.asyncio
async def test_fetch_page_from_network_and_caches(
    tmp_path: Path, sample_page_payload: dict
) -> None:
    """Verify network fetch on cache miss and subsequent disk persistence."""
    client = AniListClient(data_dir=tmp_path)

    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = sample_page_payload
    mock_resp.headers = {"X-RateLimit-Remaining": "85"}

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp

        data = await client.fetch_page(page=1, use_cache=True)
        assert data == sample_page_payload
        mock_post.assert_called_once()

        # Check that file was written to disk
        snapshot_file = tmp_path / "page_001.json"
        assert snapshot_file.exists()


@pytest.mark.asyncio
async def test_fetch_page_retry_on_429(
    tmp_path: Path, sample_page_payload: dict
) -> None:
    """Verify exponential backoff retry on HTTP 429 rate limit error."""
    client = AniListClient(data_dir=tmp_path)

    resp_429 = MagicMock(spec=httpx.Response)
    resp_429.status_code = 429
    resp_429.headers = {"Retry-After": "0"}

    resp_200 = MagicMock(spec=httpx.Response)
    resp_200.status_code = 200
    resp_200.json.return_value = sample_page_payload
    resp_200.headers = {}

    with (
        patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post,
        patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
    ):
        # First attempt returns 429, second attempt succeeds with 200
        mock_post.side_effect = [resp_429, resp_200]

        data = await client.fetch_page(page=2, use_cache=False)
        assert data == sample_page_payload
        assert mock_post.call_count == 2
        mock_sleep.assert_called()


@pytest.mark.asyncio
async def test_fetch_page_max_retries_failure(tmp_path: Path) -> None:
    """Verify RuntimeError is raised when max retries are exceeded."""
    client = AniListClient(data_dir=tmp_path)

    resp_500 = MagicMock(spec=httpx.Response)
    resp_500.status_code = 500
    resp_500.headers = {}

    with (
        patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post,
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        mock_post.return_value = resp_500

        with pytest.raises(
            RuntimeError, match="Failed to fetch page 1 after 3 attempts"
        ):
            await client.fetch_page(page=1, max_retries=3, use_cache=False)


@pytest.mark.asyncio
async def test_ingest_catalog_multi_page(tmp_path: Path) -> None:
    """Verify catalog iteration aggregates media and halts when hasNextPage is False."""
    client = AniListClient(data_dir=tmp_path)

    page1_data = {
        "data": {
            "Page": {
                "pageInfo": {"hasNextPage": True, "currentPage": 1},
                "media": [{"id": 1, "title": {"english": "Show 1"}}],
            }
        }
    }
    page2_data = {
        "data": {
            "Page": {
                "pageInfo": {"hasNextPage": False, "currentPage": 2},
                "media": [{"id": 2, "title": {"english": "Show 2"}}],
            }
        }
    }

    save_page_snapshot(1, page1_data, data_dir=tmp_path)
    save_page_snapshot(2, page2_data, data_dir=tmp_path)

    all_shows = await client.ingest_catalog(max_pages=5, use_cache=True)
    assert len(all_shows) == 2
    assert all_shows[0]["id"] == 1
    assert all_shows[1]["id"] == 2
