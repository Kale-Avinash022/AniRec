"""AniRec generation module.

Phase 6: Generation Layer, Grounding & Guardrails
  Step 32 — llm_client.py    : Abstract LLM provider interface & factory
  Step 33 — prompts.py       : Grounded system prompt & few-shot design
  Step 34 — contract.py      : Strict Pydantic JSON Mode output contract
  Step 35 — verification.py  : Anti-hallucination verification gate
  Step 36 — pipeline.py      : Generation self-correction & fallback pipeline
  Step 37 — guardrails.py    : Spoiler prevention & safety boundary checks
"""

from src.generation.contract import (
    RecommendationOutput,
    RecommendedShow,
    extract_json_payload,
    parse_llm_json,
)
from src.generation.guardrails import (
    sanitize_spoiler_text,
    validate_spoiler_free,
)
from src.generation.llm_client import (
    BaseChatModel,
    MockChatModel,
    OllamaChatModel,
    OpenAIChatModel,
    get_chat_model,
)
from src.generation.pipeline import (
    GroundedRecommender,
    build_deterministic_fallback,
)
from src.generation.prompts import (
    SYSTEM_CHAT_PROMPT,
    SYSTEM_RECOMMENDER_PROMPT,
    format_candidate_context,
    format_user_prompt,
)
from src.generation.verification import (
    build_candidate_whitelist,
    verify_recommendations_against_candidates,
)

__all__ = [
    "BaseChatModel",
    "OpenAIChatModel",
    "OllamaChatModel",
    "MockChatModel",
    "get_chat_model",
    "SYSTEM_RECOMMENDER_PROMPT",
    "SYSTEM_CHAT_PROMPT",
    "format_candidate_context",
    "format_user_prompt",
    "RecommendedShow",
    "RecommendationOutput",
    "extract_json_payload",
    "parse_llm_json",
    "build_candidate_whitelist",
    "verify_recommendations_against_candidates",
    "GroundedRecommender",
    "build_deterministic_fallback",
    "validate_spoiler_free",
    "sanitize_spoiler_text",
]
