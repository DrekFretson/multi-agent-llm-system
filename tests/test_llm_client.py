"""Mock tests for the OpenRouter LLM client."""

from types import SimpleNamespace
from unittest.mock import Mock
import json
import uuid

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
from src.logging import Logger


@pytest.fixture
def fake_sdk(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Provide a mocked SDK client and prevent loading a local secret."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr("dotenv.load_dotenv", lambda: None)
    sdk = Mock()
    monkeypatch.setattr(client_module, "OpenAI", Mock(return_value=sdk))
    monkeypatch.setattr(client_module, "Logger", Mock(return_value=Mock()))
    return sdk


def make_client(fake_sdk: Mock, logger: Logger | None = None) -> LLMClient:
    """Build a client with the mocked SDK constructor."""
    return LLMClient(logger=logger)


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
    api_error = openai.APIError(
        "Upstream rejected token test-key",
        request=Mock(),
        body=None,
    )
    api_error.status_code = 422
    api_error.request_id = "req_mock_123"
    fake_sdk.chat.completions.create.side_effect = api_error
    client = make_client(fake_sdk)

    with pytest.raises(LLMAPIError) as error:
        client.generate("provider/model-id", [{"role": "user", "content": "Hi"}])

    assert "Upstream rejected token" in str(error.value)
    assert "HTTP 422" in str(error.value)
    assert error.value.status_code == 422
    assert error.value.request_id == "req_mock_123"
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


def test_successful_generate_writes_complete_jsonl_event(
    fake_sdk: Mock,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log_file = tmp_path / "logs" / "llm_requests.jsonl"
    logger = Logger(log_file)
    fake_sdk.chat.completions.create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="Hello from the model"),
                finish_reason="stop",
            )
        ],
        model="provider/model-id",
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=4, total_tokens=7),
    )
    times = iter((100.0, 100.025))
    monkeypatch.setattr(client_module.time, "perf_counter", lambda: next(times))

    response = make_client(fake_sdk, logger).generate(
        model="provider/model-id",
        messages=[{"role": "user", "content": "Hello!"}],
        agent_id="research-1",
        temperature=0.3,
        max_tokens=50,
    )

    lines = log_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["event_type"] == "llm.request"
    assert event["timestamp"].endswith("Z")
    assert str(uuid.UUID(event["request_id"])) == event["request_id"]
    assert event["agent_id"] == "research-1"
    assert event["model_id"] == "provider/model-id"
    assert event["messages"] == [{"role": "user", "content": "Hello!"}]
    assert event["response"] == response.content
    assert event["response_model"] == response.model
    assert event["duration_ms"] == 25.0
    assert event["prompt_tokens"] == 3
    assert event["completion_tokens"] == 4
    assert event["total_tokens"] == 7
    assert event["finish_reason"] == "stop"
    assert event["success"] is True
    assert event["error_type"] is None
    assert event["error_message"] is None


def test_api_error_is_logged_with_original_type_and_message(
    fake_sdk: Mock,
    tmp_path,
) -> None:
    log_file = tmp_path / "llm_requests.jsonl"
    api_error = openai.APIError(
        "provider capacity error for test-key",
        request=Mock(),
        body=None,
    )
    api_error.status_code = 503
    fake_sdk.chat.completions.create.side_effect = api_error

    with pytest.raises(LLMAPIError):
        make_client(fake_sdk, Logger(log_file)).generate(
            model="provider/model-id",
            messages=[{"role": "user", "content": "Hello!"}],
            agent_id="coding-1",
        )

    event = json.loads(log_file.read_text(encoding="utf-8").splitlines()[0])
    assert event["agent_id"] == "coding-1"
    assert event["success"] is False
    assert event["error_type"] == "APIError"
    assert "capacity error" in event["error_message"]
    assert "test-key" not in log_file.read_text(encoding="utf-8")
    assert event["duration_ms"] >= 0
    assert event["response"] is None
    assert event["prompt_tokens"] is None
    assert event["completion_tokens"] is None
    assert event["total_tokens"] is None
    assert event["finish_reason"] is None


def test_request_messages_and_response_do_not_log_api_key(
    fake_sdk: Mock,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api_key = "sk-or-v1-unit-test-secret"
    monkeypatch.setenv("OPENROUTER_API_KEY", api_key)
    fake_sdk.chat.completions.create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=f"echo: {api_key}"),
                finish_reason="stop",
            )
        ],
        model="provider/model-id",
        usage=None,
    )
    log_file = tmp_path / "llm_requests.jsonl"

    make_client(fake_sdk, Logger(log_file)).generate(
        model="provider/model-id",
        messages=[{"role": "user", "content": f"include {api_key}"}],
    )

    contents = log_file.read_text(encoding="utf-8")
    assert api_key not in contents
    event = json.loads(contents)
    assert "[REDACTED]" in event["messages"][0]["content"]
    assert "[REDACTED]" in event["response"]
