# Step 04: Development Environment Setup & Tooling Configuration

## 1. Process & Purpose
Establish a reproducible, clean, and isolated Python environment equipped with formatting (`ruff`), type checking (`mypy`), testing (`pytest`), and lockfile dependency management.

---

## 2. Requirements & Dependencies

### System Requirements
- Python 3.11+
- Git
- Optional: NVIDIA CUDA Toolkit (for local GPU acceleration with sentence-transformers/cross-encoders)

### Core Dependencies (`requirements.txt` / `pyproject.toml`)
```toml
[project]
name = "anirec"
version = "0.1.0"
description = "Production RAG Anime Recommendation Engine"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.110.0",
    "uvicorn[standard]>=0.28.0",
    "pydantic>=2.6.0",
    "pydantic-settings>=2.2.0",
    "chromadb>=0.4.24",
    "sentence-transformers>=2.5.0",
    "openai>=1.14.0",
    "httpx>=0.27.0",
    "streamlit>=1.32.0",
    "sse-starlette>=2.0.0",
    "tqdm>=4.66.0",
    "rich>=13.7.0"
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "ruff>=0.3.0",
    "mypy>=1.9.0",
    "pytest-cov>=4.1.0"
]
```

---

## 3. Environment Setup Execution Steps

```bash
# 1. Create Virtual Environment
python -m venv .venv

# 2. Activate Virtual Environment
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# 3. Upgrade pip and install package in editable mode with dev dependencies
pip install --upgrade pip
pip install -e ".[dev]"
```

---

## 4. Code Quality & Linting Configuration (`ruff.toml`)

```toml
[lint]
select = ["E", "F", "I", "N", "W", "UP", "B", "SIM"]
ignore = ["E501"] # Line length handled by formatter

[format]
quote-style = "double"
indent-style = "space"
```

---

## 5. Deliverables & Acceptance Criteria
- [x] Python 3.11+ virtual environment configured.
- [x] All core, dev, and AI dependencies pinned and resolvable.
- [x] Automated formatting and linting scripts validated via `ruff check .`.
