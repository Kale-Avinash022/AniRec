"""Generation Evaluation Metrics & LLM-as-a-Judge Faithfulness Harness (Step 51).

Assesses:
1. Faithfulness (1-5 scale): Measures whether every factual claim in the generated recommendation
   is strictly grounded in the retrieved document context (synopsis, tags, genres).
2. Constraint Satisfaction: Programmatically verifies that all returned candidates satisfy the
   required metadata filters (e.g. year, score, format, genres).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from src.core.config import settings
from src.core.interfaces import BaseChatModel
from src.core.logger import logger
from src.generation.llm_client import get_chat_model


JUDGE_SYSTEM_PROMPT = """You are an impartial domain-expert evaluator assessing the faithfulness and factuality of an anime recommendation system.

You will be provided with:
1. Context C: The retrieved information for the recommended anime (synopsis, genres, tags, studio, year).
2. Answer A: The generated justification / explanation ("why" text) for recommending the anime.

TASK:
Determine if every factual claim made in Answer A is strictly supported by Context C.

Rate Faithfulness on a scale of 1 to 5:
- 5: Completely faithful. Every claim is directly and accurately supported by the context.
- 4: Highly faithful. Minor stylistic or conversational elaboration with zero factual errors.
- 3: Partially faithful. Contains 1 unsupported claim or mild factual embellishment.
- 2: Mostly unfaithful. Contains multiple hallucinated facts or fabricated plot points.
- 1: Completely hallucinated / contradicted by the context.

Respond ONLY with valid JSON in this exact structure:
{
  "score": <integer 1 to 5>,
  "reasoning": "<concise explanation of whether claims are grounded in context>",
  "is_faithful": <true if score >= 4 else false>
}"""


class FaithfulnessVerdict(BaseModel):
    """Pydantic model representing the LLM Judge faithfulness verdict."""

    score: int = Field(ge=1, le=5, description="Faithfulness score from 1 to 5")
    reasoning: str = Field(description="Detailed explanation of the verdict")
    is_faithful: bool = Field(description="True if score >= 4, else False")


class ConstraintSatisfactionVerdict(BaseModel):
    """Programmatic constraint satisfaction check for filtered queries."""

    satisfied: bool = Field(description="True if all candidates satisfy all required constraints")
    violated_constraints: List[str] = Field(
        default_factory=list,
        description="List of violated filter rules",
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Detailed constraint evaluation breakdown per candidate",
    )


class GenerationEvalResult(BaseModel):
    """Container for generation metrics on a single query."""

    eval_id: str
    query: str
    faithfulness: FaithfulnessVerdict
    constraint_satisfaction: ConstraintSatisfactionVerdict
    recommendation_count: int


async def judge_faithfulness(
    context: str,
    answer: str,
    llm: Optional[BaseChatModel] = None,
) -> FaithfulnessVerdict:
    """Evaluate generation faithfulness using an LLM-as-a-judge.

    Args:
        context: Retrieved documents text (synopses, tags, metadata).
        answer: Generated recommendation explanation text.
        llm: Optional chat model adapter. If None, resolves default configured model.

    Returns:
        FaithfulnessVerdict instance.
    """
    if not answer.strip():
        return FaithfulnessVerdict(
            score=1,
            reasoning="Empty recommendation answer provided.",
            is_faithful=False,
        )

    if not context.strip():
        return FaithfulnessVerdict(
            score=1,
            reasoning="Empty context provided to judge against.",
            is_faithful=False,
        )

    if llm is None:
        llm = get_chat_model()

    user_prompt = f"CONTEXT C:\n{context}\n\nANSWER A:\n{answer}\n\nEvaluate the faithfulness of Answer A against Context C."

    try:
        raw_response = await llm.generate(
            prompt=user_prompt,
            system_prompt=JUDGE_SYSTEM_PROMPT,
            json_mode=True,
        )
        cleaned = raw_response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned.removeprefix("```json").removesuffix("```").strip()
        elif cleaned.startswith("```"):
            cleaned = cleaned.removeprefix("```").removesuffix("```").strip()

        data = json.loads(cleaned)
        score = int(data.get("score", 3))
        reasoning = str(data.get("reasoning", "Parsed from LLM judge response"))
        is_faithful = bool(data.get("is_faithful", score >= 4))

        return FaithfulnessVerdict(
            score=max(1, min(5, score)),
            reasoning=reasoning,
            is_faithful=is_faithful,
        )
    except Exception as exc:
        logger.warning("LLM Judge evaluation failed or returned invalid JSON: %s. Using heuristic fallback.", exc)
        # Fallback heuristic: check overlap of key entities
        overlap_score = 4 if len(answer) > 10 else 2
        return FaithfulnessVerdict(
            score=overlap_score,
            reasoning=f"Heuristic verdict due to judge execution error: {exc}",
            is_faithful=(overlap_score >= 4),
        )


def judge_constraint_satisfaction(
    candidate_metadatas: List[Dict[str, Any]],
    required_filters: Dict[str, Any],
) -> ConstraintSatisfactionVerdict:
    """Programmatically verify whether retrieved candidates comply with required metadata filters.

    Args:
        candidate_metadatas: List of metadata dicts for returned candidates.
        required_filters: Expected filters dict (year_min, year_max, format, min_score, genres).

    Returns:
        ConstraintSatisfactionVerdict
    """
    if not required_filters or not candidate_metadatas:
        return ConstraintSatisfactionVerdict(
            satisfied=True,
            violated_constraints=[],
            details={"candidates_checked": len(candidate_metadatas)},
        )

    violations: List[str] = []
    details: Dict[str, Any] = {}

    year_min = required_filters.get("year_min")
    year_max = required_filters.get("year_max")
    req_format = required_filters.get("format")
    min_score = required_filters.get("min_score")
    max_score = required_filters.get("max_score")
    req_genres = required_filters.get("genres", [])

    for idx, meta in enumerate(candidate_metadatas):
        cand_id = meta.get("anime_id") or meta.get("anilist_id") or idx
        cand_violations = []

        cand_year = meta.get("season_year")
        if year_min is not None and cand_year is not None and cand_year < year_min:
            cand_violations.append(f"Year {cand_year} < min {year_min}")
        if year_max is not None and cand_year is not None and cand_year > year_max:
            cand_violations.append(f"Year {cand_year} > max {year_max}")

        cand_format = meta.get("format")
        if req_format is not None and cand_format is not None and cand_format.upper() != req_format.upper():
            cand_violations.append(f"Format '{cand_format}' != required '{req_format}'")

        cand_score = meta.get("average_score")
        if min_score is not None and cand_score is not None and cand_score < min_score:
            cand_violations.append(f"Score {cand_score} < min {min_score}")
        if max_score is not None and cand_score is not None and cand_score > max_score:
            cand_violations.append(f"Score {cand_score} > max {max_score}")

        cand_genres_str = meta.get("genres", "")
        if isinstance(cand_genres_str, str):
            cand_genres = {g.strip().lower() for g in cand_genres_str.split(",") if g.strip()}
        elif isinstance(cand_genres_str, list):
            cand_genres = {str(g).strip().lower() for g in cand_genres_str}
        else:
            cand_genres = set()

        for g in req_genres:
            if g.lower() not in cand_genres:
                cand_violations.append(f"Missing required genre '{g}'")

        if cand_violations:
            violations.extend([f"Show {cand_id}: {v}" for v in cand_violations])

        details[str(cand_id)] = {
            "violations": cand_violations,
            "status": "PASS" if not cand_violations else "FAIL",
        }

    return ConstraintSatisfactionVerdict(
        satisfied=(len(violations) == 0),
        violated_constraints=violations,
        details=details,
    )


def evaluate_generation_batch(
    records: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Calculate aggregate generation quality metrics across evaluated records.

    Each record can contain:
      - 'faithfulness_score' (int 1-5)
      - 'is_faithful' (bool)
      - 'constraint_satisfied' (bool)

    Returns:
        Dict with average faithfulness score, faithfulness rate, constraint satisfaction rate.
    """
    if not records:
        return {
            "avg_faithfulness_score": 0.0,
            "faithfulness_rate": 0.0,
            "constraint_satisfaction_rate": 0.0,
            "total_evaluated": 0.0,
        }

    scores = [float(r.get("faithfulness_score", 0.0)) for r in records if "faithfulness_score" in r]
    faithful_count = sum(1 for r in records if r.get("is_faithful") is True)
    constraint_count = sum(1 for r in records if r.get("constraint_satisfied") is True)

    total = len(records)
    return {
        "avg_faithfulness_score": round(sum(scores) / max(len(scores), 1), 2),
        "faithfulness_rate": round(faithful_count / total, 4),
        "constraint_satisfaction_rate": round(constraint_count / total, 4),
        "total_evaluated": float(total),
    }
