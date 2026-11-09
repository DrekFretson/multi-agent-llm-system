"""Manually run one live OpenRouter request against each selected model.

Run with ``python -m tests.integration.test_openrouter``. This module does not
send requests when collected by pytest; live calls only run from ``main``.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

from dotenv import load_dotenv

from src.llm import LLMClient, LLMClientError


TIMEOUT_SECONDS = 90.0
GENERATION_PARAMS = {"temperature": 0.2, "max_tokens": 128}
MESSAGES = [
    {
        "role": "user",
        "content": "In one short paragraph, explain what an LLM API client does.",
    }
]
MODELS = (
    (
        "Orchestrator — NVIDIA Nemotron 3 Ultra",
        "OPENROUTER_ORCHESTRATOR_MODEL",
        "nvidia/nemotron-3-ultra-550b-a55b:free",
    ),
    (
        "Coding Agent — Cohere North Mini Code",
        "OPENROUTER_CODING_MODEL",
        "cohere/north-mini-code:free",
    ),
)


def _model_config() -> list[tuple[str, str]]:
    """Return display names and validated model IDs from environment/defaults."""
    configured = []
    for display_name, env_name, default_id in MODELS:
        model_id = os.getenv(env_name, default_id).strip()
        if not model_id:
            raise ValueError(f"{env_name} must contain a model ID.")
        configured.append((display_name, model_id))
    return configured


def _run_checks(client: LLMClient, models: list[tuple[str, str]]) -> list[dict[str, Any]]:
    """Call each configured model once and collect comparable result data."""
    results = []
    for display_name, model_id in models:
        started = time.perf_counter()
        try:
            response = client.generate(
                model=model_id,
                messages=MESSAGES,
                **GENERATION_PARAMS,
            )
        except Exception as error:
            if isinstance(error, LLMClientError):
                error_message = f"{type(error).__name__}: {error}"
            else:
                error_message = (
                    f"{type(error).__name__}: unexpected error; details suppressed"
                )
            results.append(
                {
                    "model_name": display_name,
                    "configured_model_id": model_id,
                    "actual_model_id": None,
                    "content": None,
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None,
                    "finish_reason": None,
                    "error": error_message,
                }
            )
        else:
            results.append(
                {
                    "model_name": display_name,
                    "configured_model_id": model_id,
                    "actual_model_id": response.model,
                    "content": response.content,
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                    "prompt_tokens": response.prompt_tokens,
                    "completion_tokens": response.completion_tokens,
                    "total_tokens": response.total_tokens,
                    "finish_reason": response.finish_reason,
                    "error": None,
                }
            )
    return results


def main() -> int:
    """Validate configuration, run the two live calls, and print JSON results."""
    load_dotenv()
    if not os.getenv("OPENROUTER_API_KEY"):
        print(
            "OPENROUTER_API_KEY is not set. Configure it in the environment or .env.",
            file=sys.stderr,
        )
        return 2

    try:
        models = _model_config()
        with LLMClient(timeout=TIMEOUT_SECONDS) as client:
            results = _run_checks(client, models)
    except Exception as error:
        print(
            f"Integration check could not start: {type(error).__name__}; details suppressed.",
            file=sys.stderr,
        )
        return 2

    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0 if all(result["error"] is None for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
