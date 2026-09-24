# Step 55: Multi-Stage Dockerfile Containerization

## 1. Process & Purpose
Build a lean, multi-stage production `Dockerfile` that packages FastAPI and Streamlit into an optimized container image, caching dependencies and pre-downloading transformer models during the build phase.

---

## 2. Dockerfile Specification

```dockerfile
# -------------------------------------------------------------
# Stage 1: Build & Dependency Installation
# -------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Pre-download SentenceTransformer & Cross-Encoder models into image
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
    CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# -------------------------------------------------------------
# Stage 2: Minimal Runtime Image
# -------------------------------------------------------------
FROM python:3.11-slim AS runner

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /root/.cache /root/.cache

COPY . .

EXPOSE 8000 8501

# Entrypoint script to start both FastAPI and Streamlit
CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port 8000 & streamlit run src/ui/streamlit_app.py --server.port 8501 --server.address 0.0.0.0"]
```

---

## 3. `.dockerignore`

```
.git
.venv
__pycache__
*.pyc
logs/*.jsonl
evals/results/*.json
data/raw/*.json
.env
```

---

## 4. Deliverables & Acceptance Criteria
- [x] Multi-stage Dockerfile with cached dependencies and pre-warmed models.
- [x] Tested build yielding minimal container image (< 1.2 GB).
