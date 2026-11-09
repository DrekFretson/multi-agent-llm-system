"""OpenRouter client built on top of the OpenAI Python SDK."""

from __future__ import annotations

import os
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError as OpenAIAuthenticationError,
    OpenAI,
    RateLimitError as OpenAIRateLimitError,
)

from src.logging import Logger, LoggerError


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

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.request_id = request_id


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

    def __init__(
        self,
        timeout: float = 60.0,
        logger: Logger | None = None,
    ) -> None:
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

        self._api_key = api_key
        self._client = OpenAI(
            api_key=api_key,
            base_url=self.BASE_URL,
            timeout=timeout,
            max_retries=0,
        )
        self._logger = logger if logger is not None else Logger()

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
        *,
        agent_id: str | None = None,
        **generation_params: Any,
    ) -> LLMResponse:
        """Generate a completion and normalize its content and usage data.

        Additional keyword arguments are forwarded to the OpenAI chat
        completions endpoint, for example ``temperature`` and ``max_tokens``.
        """
        request_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        timestamp = timestamp.replace("+00:00", "Z")
        started = time.perf_counter()

        try:
            raw_response = self._client.chat.completions.create(
                model=model,
                messages=messages,
                **generation_params,
            )
            response = self._normalize_response(raw_response)
        except Exception as error:
            duration_ms = (time.perf_counter() - started) * 1000
            converted_error = self._convert_error(error)
            try:
                self._logger.log_event(
                    "llm.request",
                    **self._request_log_fields(
                        timestamp=timestamp,
                        request_id=request_id,
                        agent_id=agent_id,
                        model_id=model,
                        messages=messages,
                        response=None,
                        response_model=None,
                        duration_ms=duration_ms,
                        prompt_tokens=None,
                        completion_tokens=None,
                        total_tokens=None,
                        finish_reason=None,
                        success=False,
                        error_type=type(error).__name__,
                        error_message=str(error),
                    ),
                )
            except LoggerError as logging_error:
                converted_error.add_note(f"Request logging failed: {logging_error}")

            if converted_error is error:
                raise
            raise converted_error from None

        duration_ms = (time.perf_counter() - started) * 1000
        self._logger.log_event(
            "llm.request",
            **self._request_log_fields(
                timestamp=timestamp,
                request_id=request_id,
                agent_id=agent_id,
                model_id=model,
                messages=messages,
                response=response.content,
                response_model=response.model,
                duration_ms=duration_ms,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                total_tokens=response.total_tokens,
                finish_reason=response.finish_reason,
                success=True,
                error_type=None,
                error_message=None,
            ),
        )
        return response

    @staticmethod
    def _normalize_response(response: Any) -> LLMResponse:
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

    def _convert_error(self, error: Exception) -> Exception:
        if isinstance(error, OpenAIAuthenticationError):
            return AuthenticationError("OpenRouter rejected the API credentials.")
        if isinstance(error, OpenAIRateLimitError):
            return RateLimitError("OpenRouter rate limit was exceeded.")
        if isinstance(error, APITimeoutError):
            return LLMTimeoutError("The OpenRouter request timed out.")
        if isinstance(error, APIConnectionError):
            return NetworkError("Could not connect to OpenRouter.")
        if isinstance(error, APIError):
            status_code = getattr(error, "status_code", None)
            request_id = getattr(error, "request_id", None)
            detail = self._sanitize_error_detail(str(error))
            status = f" (HTTP {status_code})" if status_code is not None else ""
            request = f" [request ID: {request_id}]" if request_id else ""
            message = detail or "The provider returned an API error."
            return LLMAPIError(
                f"OpenRouter API error{status}{request}: {message}",
                status_code=status_code,
                request_id=request_id,
            )
        return error

    def _request_log_fields(self, **fields: Any) -> dict[str, Any]:
        fields["duration_ms"] = round(fields["duration_ms"], 3)
        return self._sanitize_log_value(fields)

    def _sanitize_log_value(self, value: Any) -> Any:
        if isinstance(value, str):
            return self._sanitize_error_detail(value)
        if isinstance(value, dict):
            return {
                self._sanitize_error_detail(str(key)): self._sanitize_log_value(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [self._sanitize_log_value(item) for item in value]
        return value

    def _sanitize_error_detail(self, detail: str) -> str:
        """Retain useful provider diagnostics while redacting credential text."""
        if self._api_key:
            detail = detail.replace(self._api_key, "[REDACTED]")
        detail = re.sub(
            r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,;]+",
            r"\1[REDACTED]",
            detail,
        )
        detail = re.sub(r"(?i)\bsk-or-v1-[A-Za-z0-9_-]+", "[REDACTED]", detail)
        detail = re.sub(r"\bsk-[A-Za-z0-9_-]{12,}\b", "[REDACTED]", detail)
        return detail[:1000]
