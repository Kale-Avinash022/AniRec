# Step 10: Raw Data Disk Snapshot Storage (`data/raw`)

## 1. Process & Purpose
To eliminate redundant network calls, respect AniList API terms, and enable fast offline iterative preprocessing, raw API responses must be cached directly to disk as immutable JSON snapshot files (`data/raw/page_001.json`, etc.).

---

## 2. Snapshot Storage Flow

```mermaid
graph LR
    API[AniList GraphQL API] -->|HTTP 200| CRAWLER[Ingestion Client]
    CRAWLER -->|Check Cache| DISK_CHECK{File exists on disk?}
    DISK_CHECK -->|Yes| LOAD_CACHE[Load from data/raw/page_XXX.json]
    DISK_CHECK -->|No| FETCH_NET[Fetch from GraphQL API]
    FETCH_NET --> WRITE_CACHE[Write atomically to data/raw/page_XXX.json]
    WRITE_CACHE --> NEXT[Next Page]
```

---

## 3. Storage Strategy & Directory Layout
- **Location**: `data/raw/`
- **File Format**: `page_{page_num:03d}.json` (e.g., `page_001.json`, `page_002.json` ... `page_060.json`).
- **Atomicity**: Write to a temporary file (`.tmp`) and rename atomically to avoid corrupted files on interrupt.

---

## 4. Implementation Code

```python
# src/ingest/storage.py
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any
from src.core.config import settings

def get_snapshot_path(page_number: int) -> Path:
    settings.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return settings.RAW_DATA_DIR / f"page_{page_number:03d}.json"

def load_page_snapshot(page_number: int) -> Optional[Dict[str, Any]]:
    path = get_snapshot_path(page_number)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def save_page_snapshot(page_number: int, data: Dict[str, Any]) -> None:
    path = get_snapshot_path(page_number)
    temp_path = path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    # Atomic rename
    os.replace(temp_path, path)
```

---

## 5. Deliverables & Acceptance Criteria
- [x] Disk snapshot loader and atomic writer implemented.
- [x] Zero API re-queries when cache files exist on disk.
- [x] `.gitignore` updated to prevent large raw snapshots from being committed to Git.
