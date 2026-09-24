"""
Unit Tests for Step 04: Development Environment Setup & Tooling Configuration.
Verifies project metadata, pyproject.toml, ruff.toml, Python version, and dependency availability.
"""

import importlib.util
import sys
import tomllib
import unittest
from pathlib import Path


class TestStep04DevEnvironmentAndTooling(unittest.TestCase):
    def setUp(self):
        self.root_dir = Path(__file__).resolve().parent.parent

    def test_python_version_compatibility(self):
        """Verify Python runtime meets minimum requirement (>= 3.11)."""
        self.assertGreaterEqual(
            sys.version_info[:2],
            (3, 11),
            f"Python >= 3.11 is required, currently running {sys.version}",
        )

    def test_ruff_toml_configuration(self):
        """Verify ruff.toml exists and contains required lint/format rules."""
        ruff_path = self.root_dir / "ruff.toml"
        self.assertTrue(ruff_path.exists(), "ruff.toml must exist in project root")

        with open(ruff_path, "rb") as f:
            config = tomllib.load(f)

        self.assertIn("lint", config)
        self.assertIn("format", config)

        lint_rules = config["lint"].get("select", [])
        for expected_rule in ["E", "F", "I", "N", "W", "UP", "B", "SIM"]:
            self.assertIn(expected_rule, lint_rules)

        self.assertEqual(config["format"].get("quote-style"), "double")
        self.assertEqual(config["format"].get("indent-style"), "space")

    def test_pyproject_toml_configuration(self):
        """Verify pyproject.toml exists and defines dependencies and build metadata."""
        pyproject_path = self.root_dir / "pyproject.toml"
        self.assertTrue(
            pyproject_path.exists(), "pyproject.toml must exist in project root"
        )

        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)

        self.assertIn("project", data)
        project = data["project"]
        self.assertEqual(project.get("name"), "anirec")
        self.assertEqual(project.get("requires-python"), ">=3.11")

        deps = project.get("dependencies", [])
        dep_names = [d.split(">=")[0].split("[")[0].strip() for d in deps]
        for expected_dep in [
            "fastapi",
            "uvicorn",
            "pydantic",
            "pydantic-settings",
            "httpx",
            "streamlit",
            "sse-starlette",
            "tqdm",
            "rich",
            "sentence-transformers",
            "chromadb",
            "openai",
        ]:
            self.assertIn(expected_dep, dep_names)

        dev_deps = project.get("optional-dependencies", {}).get("dev", [])
        dev_dep_names = [d.split(">=")[0].strip() for d in dev_deps]
        for expected_dev in ["pytest", "pytest-asyncio", "ruff", "mypy"]:
            self.assertIn(expected_dev, dev_dep_names)

    def test_core_packages_installed(self):
        """Verify that core third-party packages are installed in environment."""
        packages = [
            "fastapi",
            "pydantic",
            "pydantic_settings",
            "chromadb",
            "sentence_transformers",
            "openai",
            "streamlit",
            "pytest",
            "ruff",
        ]
        for pkg in packages:
            spec = importlib.util.find_spec(pkg)
            self.assertIsNotNone(spec, f"Package '{pkg}' must be installed in .venv")


if __name__ == "__main__":
    unittest.main()
