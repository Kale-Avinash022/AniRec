"""SSE Streaming Chat Consumer & Markdown Renderer (Step 45).

Consumes the ``POST /chat`` Server-Sent Events stream from the FastAPI backend,
streaming tokens directly into Streamlit's chat containers and extracting
retrieved candidate metadata for provenance expanders.

Usage::

    from src.ui.chat import stream_chat_response

    for token, candidates in stream_chat_response(
        api_base_url="http://localhost:8000",
        message="dark political fantasy",
        history=[...],
    ):
        print(token, end="")
"""

from __future__ import annotations

import json
from collections.abc import Generator
from typing import Any

import httpx

from src.core.logger import logger


def stream_chat_response(
    api_base_url: str,
    message: str,
    history: list[dict[str, str]] | None = None,
    active_filters: dict[str, Any] | None = None,
    timeout: float = 60.0,
) -> Generator[tuple[str, list[dict[str, Any]]], None, None]:
    """Consume SSE stream from the AniRec backend /chat endpoint.

    Yields:
        2-tuples of (token_chunk: str, retrieved_candidates: list[dict]).
    """
    url = f"{api_base_url.rstrip('/')}/chat"
    payload = {
        "message": message,
        "history": history or [],
        "active_filters": active_filters,
    }

    retrieved_candidates: list[dict[str, Any]] = []
    current_event = "token"

    try:
        with httpx.stream(
            "POST",
            url,
            json=payload,
            timeout=timeout,
        ) as response:
            if response.status_code != 200:
                error_detail = response.read().decode("utf-8", errors="ignore")
                logger.error(f"Chat stream HTTP {response.status_code}: {error_detail}")
                yield (
                    f"⚠️ Error from recommendation server ({response.status_code}). Please try again.",
                    [],
                )
                return

            for line in response.iter_lines():
                if not line:
                    continue

                if line.startswith("event: "):
                    current_event = line[7:].strip()
                    continue

                if line.startswith("data:"):
                    data_str = line[6:] if line.startswith("data: ") else line[5:]
                    if data_str.strip() == "[DONE]":
                        break

                    if current_event == "context":
                        try:
                            retrieved_candidates = json.loads(data_str)
                        except Exception as exc:
                            logger.warning(f"Failed to parse candidate context event: {exc}")
                        continue

                    elif current_event == "error":
                        logger.error(f"SSE server error: {data_str}")
                        yield (f"\n\n⚠️ *Server error*: {data_str.strip()}", retrieved_candidates)
                        break

                    else:  # token chunk
                        try:
                            payload_json = json.loads(data_str)
                            if isinstance(payload_json, dict) and "text" in payload_json:
                                token_val = payload_json["text"]
                            elif isinstance(payload_json, str):
                                token_val = payload_json
                            else:
                                token_val = data_str
                        except Exception:
                            token_val = data_str
                        yield (token_val, retrieved_candidates)

    except Exception as exc:
        logger.error(f"Chat stream connection failure: {exc}")
        yield (
            f"⚠️ Could not connect to AniRec backend at {api_base_url}. Ensure the FastAPI server is running.",
            [],
        )
