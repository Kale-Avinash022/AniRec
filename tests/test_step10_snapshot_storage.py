"""Unit tests for Step 10: Raw Data Disk Snapshot Storage."""

from pathlib import Path

import pytest

from src.ingest import (
    get_snapshot_path,
    list_existing_snapshots,
    load_page_snapshot,
    save_page_snapshot,
)


def test_get_snapshot_path_naming(tmp_path: Path) -> None:
    """Verify standard zero-padded 3-digit naming pattern."""
    path_1 = get_snapshot_path(1, data_dir=tmp_path)
    assert path_1.name == "page_001.json"

    path_42 = get_snapshot_path(42, data_dir=tmp_path)
    assert path_42.name == "page_042.json"

    with pytest.raises(ValueError, match="page_number must be >= 1"):
        get_snapshot_path(0, data_dir=tmp_path)


def test_save_and_load_page_snapshot(tmp_path: Path) -> None:
    """Verify atomic write and subsequent retrieval of raw snapshot JSON."""
    test_data = {
        "data": {
            "Page": {
                "pageInfo": {"currentPage": 1, "hasNextPage": True},
                "media": [{"id": 1, "title": {"romaji": "Cowboy Bebop"}}],
            }
        }
    }

    # Before saving, should return None
    assert load_page_snapshot(1, data_dir=tmp_path) is None

    # Save snapshot
    saved_path = save_page_snapshot(1, test_data, data_dir=tmp_path)
    assert saved_path.exists()
    assert not saved_path.with_suffix(".tmp").exists()

    # Load back and verify data integrity
    loaded_data = load_page_snapshot(1, data_dir=tmp_path)
    assert loaded_data == test_data


def test_list_existing_snapshots(tmp_path: Path) -> None:
    """Verify listing and sorting of available snapshot page numbers."""
    assert list_existing_snapshots(data_dir=tmp_path) == []

    # Create dummy snapshots for pages 3, 1, 10
    save_page_snapshot(3, {"page": 3}, data_dir=tmp_path)
    save_page_snapshot(1, {"page": 1}, data_dir=tmp_path)
    save_page_snapshot(10, {"page": 10}, data_dir=tmp_path)

    available = list_existing_snapshots(data_dir=tmp_path)
    assert available == [1, 3, 10]


def test_atomic_overwrite(tmp_path: Path) -> None:
    """Verify saving to an existing page safely updates content."""
    initial_data = {"version": 1}
    updated_data = {"version": 2}

    save_page_snapshot(5, initial_data, data_dir=tmp_path)
    assert load_page_snapshot(5, data_dir=tmp_path) == initial_data

    save_page_snapshot(5, updated_data, data_dir=tmp_path)
    assert load_page_snapshot(5, data_dir=tmp_path) == updated_data
