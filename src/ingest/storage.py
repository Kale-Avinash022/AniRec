"""Raw Data Disk Snapshot Storage Engine (Step 10).

Manages persistent JSON snapshots in `data/raw/` with atomic writes and
instant cache hits to minimize redundant API requests.
"""

import json
import os
from pathlib import Path
from typing import Any

from src.core.config import settings


def get_snapshot_path(page_number: int, data_dir: Path | None = None) -> Path:
    """Get the target filepath for a specific raw page snapshot.

    Args:
        page_number: 1-indexed page number (e.g. 1 -> page_001.json).
        data_dir: Optional override directory; defaults to settings.RAW_DATA_DIR.

    Returns:
        Path object pointing to the snapshot file.
    """
    if page_number < 1:
        raise ValueError(f"page_number must be >= 1, got {page_number}")

    target_dir = data_dir if data_dir is not None else settings.RAW_DATA_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / f"page_{page_number:03d}.json"


def load_page_snapshot(
    page_number: int, data_dir: Path | None = None
) -> dict[str, Any] | None:
    """Load cached raw API snapshot for a page if present on disk.

    Args:
        page_number: 1-indexed page number.
        data_dir: Optional override directory.

    Returns:
        Parsed dictionary if file exists, else None.
    """
    path = get_snapshot_path(page_number, data_dir=data_dir)
    if not path.exists():
        return None

    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_page_snapshot(
    page_number: int,
    data: dict[str, Any],
    data_dir: Path | None = None,
) -> Path:
    """Atomically write API snapshot payload to disk.

    Writes to an adjacent .tmp file first, then uses atomic filesystem rename
    to guarantee file integrity against unexpected process termination.

    Args:
        page_number: 1-indexed page number.
        data: AniList JSON response data dictionary.
        data_dir: Optional override directory.

    Returns:
        Path of the persisted snapshot.
    """
    target_path = get_snapshot_path(page_number, data_dir=data_dir)
    temp_path = target_path.with_suffix(".tmp")

    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Atomic rename replaces existing target safely
    os.replace(temp_path, target_path)
    return target_path


def list_existing_snapshots(data_dir: Path | None = None) -> list[int]:
    """Scan the snapshot directory and return all saved page numbers in ascending order.

    Args:
        data_dir: Optional override directory.

    Returns:
        List of integer page numbers already present on disk.
    """
    target_dir = data_dir if data_dir is not None else settings.RAW_DATA_DIR
    if not target_dir.exists():
        return []

    page_numbers: list[int] = []
    for file_path in target_dir.glob("page_*.json"):
        stem = file_path.stem  # e.g. "page_001"
        parts = stem.split("_")
        if len(parts) == 2 and parts[1].isdigit():
            page_numbers.append(int(parts[1]))

    return sorted(page_numbers)
