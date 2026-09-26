"""
Centralized Configuration Management and Secrets Architecture (Step 05).
Uses pydantic-settings to load configuration from environment variables and .env files.
"""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central Application Settings singleton model."""

    # Application Settings
    APP_NAME: str = "AniRec"
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000


    # Path Settings (Dynamically resolved relative to project root)
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    CHROMA_PERSIST_DIR: Path = DATA_DIR / "chroma"
    LOGS_DIR: Path = BASE_DIR / "logs"

    # AI / Model Provider Settings
    EMBEDDING_PROVIDER: Literal["local", "openai"] = "local"
    LOCAL_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

    LLM_PROVIDER: Literal["openai", "ollama"] = "openai"
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.1:8b"

    # Ingestion & Crawler Settings
    ANILIST_API_URL: str = "https://graphql.anilist.co"
    ANILIST_MAX_PAGES: int = 60  # 60 pages * 50 per page = 3,000 shows
    ANILIST_PER_PAGE: int = 50
    ANILIST_RATE_LIMIT_DELAY_SECONDS: float = 0.8

    # Retrieval Settings
    TOP_K_RETRIEVAL: int = 20
    TOP_K_RERANKED: int = 5
    CROSS_ENCODER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def ensure_directories(self) -> None:
        """Ensure all runtime directories exist on disk."""
        for path in [
            self.DATA_DIR,
            self.RAW_DATA_DIR,
            self.PROCESSED_DATA_DIR,
            self.CHROMA_PERSIST_DIR,
            self.LOGS_DIR,
        ]:
            path.mkdir(parents=True, exist_ok=True)


# Global singleton settings instance
settings = Settings()
