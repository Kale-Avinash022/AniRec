# Step 56: Docker Compose Multi-Service Orchestration

## 1. Process & Purpose
Orchestrate local and staging deployments using `docker-compose.yml`, managing service networking between the FastAPI backend, Streamlit frontend, and optional local Ollama / Postgres services.

---

## 2. Docker Compose Specification (`docker-compose.yml`)

```yaml
version: "3.8"

services:
  anirec-api:
    build:
      context: .
      dockerfile: Dockerfile
    command: uvicorn src.api.main:app --host 0.0.0.0 --port 8000
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    env_file:
      - .env
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  anirec-ui:
    build:
      context: .
      dockerfile: Dockerfile
    command: streamlit run src/ui/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
    ports:
      - "8501:8501"
    depends_on:
      anirec-api:
        condition: service_healthy
    environment:
      - API_BASE_URL=http://anirec-api:8000
    restart: unless-stopped
```

---

## 3. Execution Commands
```bash
# Start all services in background
docker compose up -d --build

# Inspect logs
docker compose logs -f

# Teardown
docker compose down
```

---

## 4. Deliverables & Acceptance Criteria
- [x] Docker Compose configured with health checks and network dependencies.
- [x] Volume mounts established for persistent Chroma index and logs.
