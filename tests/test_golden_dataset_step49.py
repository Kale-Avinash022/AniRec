"""Unit tests for Step 49: Golden Evaluation Dataset authoring and schemas."""

import json
from pathlib import Path
import pytest
from pydantic import ValidationError

from src.eval.dataset import (
    GoldenDataset,
    GoldenQueryItem,
    load_golden_set,
    save_golden_set,
)


def test_load_golden_set_default_path():
    """Test loading the golden set from default location."""
    items = load_golden_set()
    assert len(items) == 50
    assert all(isinstance(it, GoldenQueryItem) for it in items)


def test_golden_set_category_distribution():
    """Verify exact category counts match the Step 49 roadmap specification."""
    items = load_golden_set()
    dataset = GoldenDataset(items=items)

    mood_items = dataset.by_category("mood_semantic")
    constraint_items = dataset.by_category("constraint_heavy")
    hybrid_items = dataset.by_category("hybrid")
    neg_items = dataset.by_category("negative_adversarial")
    franchise_items = dataset.by_category("franchise_edge")

    assert len(mood_items) == 20
    assert len(constraint_items) == 10
    assert len(hybrid_items) == 10
    assert len(neg_items) == 5
    assert len(franchise_items) == 5
    assert len(dataset) == 50


def test_golden_set_unique_ids_and_queries():
    """Verify that all item IDs and queries are unique and well-formed."""
    items = load_golden_set()
    ids = [item.id for item in items]
    queries = [item.query for item in items]

    assert len(ids) == len(set(ids)) == 50
    assert len(queries) == len(set(queries)) == 50

    for item in items:
        assert item.id.startswith("eval_")
        assert len(item.query.strip()) > 5
        assert item.difficulty in {"easy", "medium", "hard"}


def test_corpus_ground_truth_id_integrity():
    """Verify that all non-empty expected anime IDs exist in the processed anime corpus."""
    corpus_path = Path("data/processed/anime_corpus.json")
    if not corpus_path.exists():
        pytest.skip("Processed anime corpus not found at data/processed/anime_corpus.json")

    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    corpus_ids = {c["anime_id"] for c in corpus}

    items = load_golden_set()
    for item in items:
        for expected_id in item.expected_anilist_ids:
            assert expected_id in corpus_ids, (
                f"Evaluation item {item.id} has expected ID {expected_id} not in corpus!"
            )


def test_pydantic_schema_validation_error():
    """Verify Pydantic raises validation error on invalid input."""
    with pytest.raises(ValidationError):
        GoldenQueryItem.model_validate({
            "id": "eval_invalid",
            "category": "invalid_category",  # not in Literal
            "query": "Test query"
        })

    with pytest.raises(ValidationError):
        GoldenQueryItem.model_validate({
            # missing id and query
            "category": "mood_semantic"
        })


def test_golden_dataset_roundtrip_save(tmp_path: Path):
    """Verify saving and reloading dataset from a custom path."""
    items = load_golden_set()
    custom_path = tmp_path / "custom_golden_set.json"

    saved_path = save_golden_set(items, custom_path)
    assert saved_path.exists()

    reloaded = load_golden_set(saved_path)
    assert len(reloaded) == 50
    assert reloaded[0].id == items[0].id
    assert reloaded[0].query == items[0].query
