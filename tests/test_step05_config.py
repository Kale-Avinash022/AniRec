"""
Unit Tests for Step 05: Configuration Management & Secrets Architecture.
"""

import unittest

from src.core.config import Settings, settings


class TestStep05Configuration(unittest.TestCase):
    def test_default_settings_values(self):
        """Verify standard baseline configuration values."""
        self.assertEqual(settings.APP_NAME, "AniRec")
        self.assertEqual(settings.ENVIRONMENT, "development")
        self.assertFalse(settings.DEBUG)
        self.assertEqual(settings.EMBEDDING_PROVIDER, "local")
        self.assertEqual(settings.LLM_PROVIDER, "openai")
        self.assertEqual(settings.TOP_K_RETRIEVAL, 20)
        self.assertEqual(settings.TOP_K_RERANKED, 5)

    def test_dynamic_paths_resolution(self):
        """Verify dynamic paths resolve correctly relative to repository root."""
        root = settings.BASE_DIR
        self.assertTrue((root / "pyproject.toml").exists())
        self.assertEqual(settings.DATA_DIR, root / "data")
        self.assertEqual(settings.RAW_DATA_DIR, root / "data" / "raw")
        self.assertEqual(settings.PROCESSED_DATA_DIR, root / "data" / "processed")
        self.assertEqual(settings.CHROMA_PERSIST_DIR, root / "data" / "chroma")
        self.assertEqual(settings.LOGS_DIR, root / "logs")

    def test_ensure_directories_creation(self):
        """Verify ensure_directories safely creates necessary directory paths."""
        settings.ensure_directories()
        self.assertTrue(settings.DATA_DIR.exists())
        self.assertTrue(settings.RAW_DATA_DIR.exists())
        self.assertTrue(settings.PROCESSED_DATA_DIR.exists())
        self.assertTrue(settings.CHROMA_PERSIST_DIR.exists())
        self.assertTrue(settings.LOGS_DIR.exists())

    def test_env_override(self):
        """Verify environment variables take precedence over default settings."""
        custom_env = {
            "APP_NAME": "AniRec-Custom",
            "ENVIRONMENT": "production",
            "DEBUG": "true",
            "EMBEDDING_PROVIDER": "openai",
            "LLM_PROVIDER": "ollama",
            "TOP_K_RETRIEVAL": "30",
        }
        test_settings = Settings(**custom_env)  # type: ignore[arg-type]
        self.assertEqual(test_settings.APP_NAME, "AniRec-Custom")
        self.assertEqual(test_settings.ENVIRONMENT, "production")
        self.assertTrue(test_settings.DEBUG)
        self.assertEqual(test_settings.EMBEDDING_PROVIDER, "openai")
        self.assertEqual(test_settings.LLM_PROVIDER, "ollama")
        self.assertEqual(test_settings.TOP_K_RETRIEVAL, 30)

    def test_dotenv_example_exists(self):
        """Verify .env.example exists and documents essential variables."""
        example_path = settings.BASE_DIR / ".env.example"
        self.assertTrue(example_path.exists())
        content = example_path.read_text(encoding="utf-8")
        self.assertIn("OPENAI_API_KEY", content)
        self.assertIn("EMBEDDING_PROVIDER", content)
        self.assertIn("LLM_PROVIDER", content)


if __name__ == "__main__":
    unittest.main()
