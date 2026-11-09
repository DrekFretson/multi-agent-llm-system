"""OpenRouter client built on top of the OpenAI Python SDK."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError as OpenAIAuthenticationError,
    OpenAI,
    RateLimitError as OpenAIRateLimitError,
)


class LLMClientError(Exception):
    """Base exception for errors raised by :class:`LLMClient`."""


class MissingAPIKeyError(LLMClientError):
    """Raised when the OpenRouter API key is not configured."""


class AuthenticationError(LLMClientError):
    """Raised when OpenRouter rejects the configured API key."""


class RateLimitError(LLMClientError):
    """Raised when the request exceeds the provider's rate limit."""


class LLMTimeoutError(LLMClientError):
    """Raised when a request to OpenRouter times out."""


class NetworkError(LLMClientError):
    """Raised when the client cannot connect to OpenRouter."""


class LLMAPIError(LLMClientError):
    """Raised for other errors returned by the OpenRouter API."""


@dataclass(frozen=True)
class LLMResponse:
    """Normalized chat completion result and optional usage metadata."""

    content: str | None
    model: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    finish_reason: str | None


class LLMClient:
    """Send chat completion requests to OpenRouter using the OpenAI SDK."""

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, timeout: float = 60.0) -> None:
        """Create a client, loading a local ``.env`` file when available."""
        try:
            from dotenv import load_dotenv
        except ImportError:
            pass
        else:
            load_dotenv()

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise MissingAPIKeyError(
                "OPENROUTER_API_KEY is not set. Configure it in the environment or .env."
            )

        self._client = OpenAI(
            api_key=api_key,
            base_url=self.BASE_URL,
            timeout=timeout,
            max_retries=0,
        )

    def close(self) -> None:
        """Close the underlying HTTP client and release its resources."""
        self._client.close()

    def __enter__(self) -> LLMClient:
        """Return this client for use in a ``with`` statement."""
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Close the underlying HTTP client when leaving a ``with`` block."""
        self.close()

    def generate(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **generation_params: Any,
    ) -> LLMResponse:
        """Generate a completion and normalize its content and usage data.

        Additional keyword arguments are forwarded to the OpenAI chat
        completions endpoint, for example ``temperature`` and ``max_tokens``.
        """
        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=messages,
                **generation_params,
            )
        except OpenAIAuthenticationError:
            raise AuthenticationError("OpenRouter rejected the API credentials.") from None
        except OpenAIRateLimitError:
            raise RateLimitError("OpenRouter rate limit was exceeded.") from None
        except APITimeoutError:
            raise LLMTimeoutError("The OpenRouter request timed out.") from None
        except APIConnectionError:
            raise NetworkError("Could not connect to OpenRouter.") from None
        except APIError:
            raise LLMAPIError("The OpenRouter API request failed.") from None

        choices = getattr(response, "choices", None) or []
        choice = choices[0] if choices else None
        message = getattr(choice, "message", None)
        usage = getattr(response, "usage", None)

        return LLMResponse(
            content=getattr(message, "content", None),
            model=getattr(response, "model", None),
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
            finish_reason=getattr(choice, "finish_reason", None),
        )
