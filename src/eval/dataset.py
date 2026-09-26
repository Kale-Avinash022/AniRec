"""Dataset schemas and loader for AniRec golden evaluation benchmark."""

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
import json
from pydantic import BaseModel, Field


EvalCategory = Literal[
    "mood_semantic",
    "constraint_heavy",
    "hybrid",
    "negative_adversarial",
    "franchise_edge",
]


class GoldenQueryItem(BaseModel):
    """A single domain-expert annotated evaluation query pair."""

    id: str = Field(..., description="Unique evaluation identifier, e.g. eval_01")
    category: EvalCategory = Field(
        ..., description="Evaluation category archetype"
    )
    query: str = Field(..., description="User query text")
    expected_anilist_ids: List[int] = Field(
        default_factory=list,
        description="Ground-truth relevant AniList IDs in corpus",
    )
    expected_titles: List[str] = Field(
        default_factory=list,
        description="Ground-truth canonical titles for reference",
    )
    required_filters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Explicit metadata constraints parsed from query if applicable",
    )
    difficulty: Literal["easy", "medium", "hard"] = Field(
        default="medium",
        description="Difficulty tier of the evaluation sample",
    )
    explanation: Optional[str] = Field(
        default=None,
        description="Domain-expert rationale for ground-truth relevance",
    )


class GoldenDataset(BaseModel):
    """Collection wrapper for golden evaluation items."""

    items: List[GoldenQueryItem] = Field(
        ..., description="List of annotated golden test items"
    )

    def by_category(self, category: str) -> List[GoldenQueryItem]:
        """Filter items by evaluation category."""
        return [item for item in self.items if item.category == category]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> GoldenQueryItem:
        return self.items[index]


def load_golden_set(filepath: Optional[str | Path] = None) -> List[GoldenQueryItem]:
    """Load and validate the golden evaluation dataset JSON file.

    Args:
        filepath: Optional custom path to golden_set.json. If None, checks default locations.

    Returns:
        List of validated GoldenQueryItem instances.
    """
    if filepath is None:
        candidates = [
            Path("evals/golden_set.json"),
            Path("data/eval/golden_dataset.json"),
            Path(__file__).parent.parent.parent / "evals" / "golden_set.json",
        ]
        chosen = None
        for cand in candidates:
            if cand.exists():
                chosen = cand
                break
        if chosen is None:
            raise FileNotFoundError(
                "Could not locate golden evaluation dataset at evals/golden_set.json or data/eval/golden_dataset.json"
            )
        filepath = chosen
    else:
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Golden dataset file not found at: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if isinstance(raw_data, dict) and "items" in raw_data:
        raw_items = raw_data["items"]
    elif isinstance(raw_data, list):
        raw_items = raw_data
    else:
        raise ValueError(f"Invalid format for golden dataset: expected list or dict with 'items', got {type(raw_data)}")

    return [GoldenQueryItem.model_validate(item) for item in raw_items]


def save_golden_set(items: List[GoldenQueryItem] | List[Dict[str, Any]], filepath: str | Path) -> Path:
    """Serialize and save golden evaluation items to a JSON file.

    Args:
        items: List of GoldenQueryItem instances or raw dicts.
        filepath: Target destination path.

    Returns:
        Path to the written file.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    serialized = []
    for it in items:
        if isinstance(it, GoldenQueryItem):
            serialized.append(it.model_dump())
        elif isinstance(it, dict):
            # Validate through model
            validated = GoldenQueryItem.model_validate(it)
            serialized.append(validated.model_dump())
        else:
            raise TypeError(f"Unsupported item type: {type(it)}")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(serialized, f, indent=2, ensure_ascii=False)

    return path
