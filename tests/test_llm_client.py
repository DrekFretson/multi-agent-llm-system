"""Mock tests for the OpenRouter LLM client."""

from types import SimpleNamespace
from unittest.mock import Mock

import openai
import pytest

from src.llm import (
    AuthenticationError,
    LLMAPIError,
    LLMClient,
    LLMResponse,
    LLMTimeoutError,
    MissingAPIKeyError,
    NetworkError,
    RateLimitError,
)
from src.llm import client as client_module


@pytest.fixture
def fake_sdk(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Provide a mocked SDK client and prevent loading a local secret."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr("dotenv.load_dotenv", lambda: None)
    sdk = Mock()
    monkeypatch.setattr(client_module, "OpenAI", Mock(return_value=sdk))
    return sdk


def make_client(fake_sdk: Mock) -> LLMClient:
    """Build a client with the mocked SDK constructor."""
    return LLMClient()


def test_successful_generation_returns_normalized_response(fake_sdk: Mock) -> None:
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="Hello back!"),
                finish_reason="stop",
            )
        ],
        model="provider/model-id",
        usage=SimpleNamespace(
            prompt_tokens=4,
            completion_tokens=3,
            total_tokens=7,
        ),
    )
    fake_sdk.chat.completions.create.return_value = response

    result = make_client(fake_sdk).generate(
        model="provider/model-id",
        messages=[{"role": "user", "content": "Hello!"}],
        temperature=0.7,
        max_tokens=500,
    )

    assert result == LLMResponse(
        content="Hello back!",
        model="provider/model-id",
        prompt_tokens=4,
        completion_tokens=3,
        total_tokens=7,
        finish_reason="stop",
    )
    fake_sdk.chat.completions.create.assert_called_once_with(
        model="provider/model-id",
        messages=[{"role": "user", "content": "Hello!"}],
        temperature=0.7,
        max_tokens=500,
    )


def test_timeout_is_passed_to_sdk_constructor(fake_sdk: Mock) -> None:
    LLMClient(timeout=12.5)
    client_module.OpenAI.assert_called_with(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        timeout=12.5,
        max_retries=0,
    )


def test_missing_api_key_raises_clear_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda: None)

    with pytest.raises(MissingAPIKeyError, match="OPENROUTER_API_KEY"):
        LLMClient()


def sdk_exception(exception_type: type[Exception]) -> Exception:
    """Construct SDK exceptions with mock request metadata."""
    request = Mock()
    response = Mock(status_code=401)
    if exception_type in (openai.AuthenticationError, openai.RateLimitError):
        return exception_type("provider error", response=response, body=None)
    if exception_type is openai.APITimeoutError:
        return exception_type(request=request)
    if exception_type is openai.APIConnectionError:
        return exception_type(request=request)
    if exception_type is openai.APIError:
        return exception_type("provider error", request=request, body=None)
    raise AssertionError("Unsupported SDK exception type")


@pytest.mark.parametrize(
    ("sdk_error", "expected_error"),
    [
        (openai.AuthenticationError, AuthenticationError),
        (openai.RateLimitError, RateLimitError),
        (openai.APITimeoutError, LLMTimeoutError),
        (openai.APIConnectionError, NetworkError),
        (openai.APIError, LLMAPIError),
    ],
)
def test_api_failures_are_mapped_to_client_errors(
    fake_sdk: Mock,
    sdk_error: type[Exception],
    expected_error: type[Exception],
) -> None:
    fake_sdk.chat.completions.create.side_effect = sdk_exception(sdk_error)
    client = make_client(fake_sdk)

    with pytest.raises(expected_error):
        client.generate("provider/model-id", [{"role": "user", "content": "Hi"}])


def test_api_error_details_and_credentials_are_not_exposed(fake_sdk: Mock) -> None:
    fake_sdk.chat.completions.create.side_effect = sdk_exception(openai.AuthenticationError)
    client = make_client(fake_sdk)

    with pytest.raises(AuthenticationError) as error:
        client.generate("provider/model-id", [{"role": "user", "content": "Hi"}])

    assert "provider error" not in str(error.value)
    assert "test-key" not in str(error.value)


def test_missing_usage_and_optional_fields_are_supported(fake_sdk: Mock) -> None:
    fake_sdk.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Hi"))],
        model=None,
    )

    result = make_client(fake_sdk).generate(
        "provider/model-id", [{"role": "user", "content": "Hi"}]
    )

    assert result == LLMResponse(
        content="Hi",
        model=None,
        prompt_tokens=None,
        completion_tokens=None,
        total_tokens=None,
        finish_reason=None,
    )


def test_context_manager_closes_sdk_client(fake_sdk: Mock) -> None:
    with LLMClient():
        pass

    fake_sdk.close.assert_called_once_with()
