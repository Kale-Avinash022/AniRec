"""
Token and Query Monetary Cost Calculation Module (Step 06).
Computes exact dollar expenditure based on provider model token usage.
"""

PRICING_PER_1M_TOKENS: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"prompt": 0.150, "completion": 0.600},
    "text-embedding-3-small": {"prompt": 0.020, "completion": 0.000},
    "llama3.1:8b": {"prompt": 0.0, "completion": 0.0},
    "sentence-transformers/all-MiniLM-L6-v2": {"prompt": 0.0, "completion": 0.0},
}


def calculate_query_cost(
    model_name: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> float:
    """
    Calculate total monetary cost (in USD) for a model invocation.

    Args:
        model_name: Name of the model (e.g., 'gpt-4o-mini', 'text-embedding-3-small').
        prompt_tokens: Number of input / prompt tokens.
        completion_tokens: Number of output / completion tokens.

    Returns:
        Total calculated cost rounded to 6 decimal places.
    """
    rates = PRICING_PER_1M_TOKENS.get(model_name, {"prompt": 0.0, "completion": 0.0})
    prompt_cost = (prompt_tokens / 1_000_000.0) * rates["prompt"]
    completion_cost = (completion_tokens / 1_000_000.0) * rates["completion"]
    return round(prompt_cost + completion_cost, 6)
