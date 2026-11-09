"""Mock-based tests for the Tavily-backed provider-neutral search tool."""

from unittest.mock import Mock

import pytest

from src.tools import (
    SearchAccessError,
    SearchAPIError,
    SearchAuthenticationError,
    SearchConfigurationError,
    SearchNetworkError,
    SearchQuotaExceededError,
    SearchRateLimitError,
    SearchResponseError,
    SearchRequestError,
    SearchResult,
    SearchTimeoutError,
    SearchTool,
)


def make_client(response: object | None = None) -> Mock:
    client = Mock()
    client.search.return_value = response or {"results": []}
    return client


def make_tool(client: Mock, **kwargs: object) -> SearchTool:
    return SearchTool(client=client, logger=Mock(), **kwargs)


def result(
    url: str = "https://example.org/article",
    *,
    title: str = "Example article",
    content: str = "Useful excerpt",
    score: float = 0.9,
) -> dict[str, object]:
    return {"title": title, "url": url, "content": content, "score": score}


def test_search_sends_query_limit_and_timeout_and_parses_result() -> None:
    client = make_client({"results": [result()]})
    tool = make_tool(client, timeout=8.5)

    found = tool.search("  RAG architecture  ", max_results=3)

    assert found == [SearchResult("Example article", "https://example.org/article", "Useful excerpt", 0.9)]
    client.search.assert_called_once_with(
        query="RAG architecture",
        max_results=3,
        timeout=8.5,
        include_answer=False,
        include_raw_content=False,
    )


def test_search_handles_empty_results() -> None:
    assert make_tool(make_client({"results": []})).search("no matches") == []


def test_search_rejects_missing_api_key_without_constructing_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "")

    with pytest.raises(SearchConfigurationError, match="TAVILY_API_KEY is not set"):
        SearchTool(logger=Mock())


@pytest.mark.parametrize(
    ("exception_type", "expected_error"),
    [
        (type("InvalidAPIKeyError", (Exception,), {}), SearchAuthenticationError),
        (type("UsageLimitExceededError", (Exception,), {}), SearchRateLimitError),
        (type("TimeoutError", (Exception,), {}), SearchTimeoutError),
        (type("ConnectionError", (Exception,), {}), SearchNetworkError),
        (type("UnexpectedProviderError", (Exception,), {}), SearchAPIError),
    ],
)
def test_search_maps_sdk_errors_without_echoing_provider_message(
    exception_type: type[Exception],
    expected_error: type[Exception],
) -> None:
    client = make_client()
    client.search.side_effect = exception_type("private provider detail")
    tool = make_tool(client)

    with pytest.raises(expected_error) as error:
        tool.search("test query")

    assert "private provider detail" not in str(error.value)


def test_search_rejects_malformed_provider_payload() -> None:
    tool = make_tool(make_client({"results": {"unexpected": "mapping"}}))

    with pytest.raises(SearchResponseError, match="invalid search response"):
        tool.search("test query")


def test_search_skips_invalid_urls_and_deduplicates_normalized_urls() -> None:
    client = make_client(
        {
            "results": [
                result("javascript:alert(1)"),
                result("https://Example.org/article/#section"),
                result("https://example.org/article/", title="Duplicate"),
            ]
        }
    )

    found = make_tool(client).search("deduplicate")

    assert len(found) == 1
    assert found[0].title == "Example article"


def test_search_many_caps_queries_and_total_results_and_deduplicates() -> None:
    client = make_client()
    client.search.side_effect = [
        {"results": [result()]},
        {"results": [result(), result("https://example.org/other")]},
        {"results": [result("https://example.org/third")]},
    ]
    tool = make_tool(client)

    found = tool.search_many(
        ["one", "two", "three", "four", "five"],
        max_results_per_query=2,
        max_total_results=3,
    )

    assert len(found) == 3
    assert client.search.call_count == 3
    assert client.search.call_args_list[0].kwargs["query"] == "one"
    assert client.search.call_args_list[1].kwargs["query"] == "two"
    assert client.search.call_args_list[2].kwargs["query"] == "three"


def test_search_validates_query_and_result_limit() -> None:
    tool = make_tool(make_client())

    with pytest.raises(ValueError, match="non-empty"):
        tool.search(" ")
    with pytest.raises(ValueError, match="max_results"):
        tool.search("query", max_results=SearchTool.MAX_RESULTS + 1)


def _status_error(name: str, status_code: int) -> Exception:
    error_type = type(name, (Exception,), {})
    error = error_type("provider detail must not be logged")
    error.status_code = status_code
    return error


def test_search_tool_starts_enabled_and_logs_success() -> None:
    client = make_client({"results": [result()]})
    logger = Mock()
    tool = SearchTool(client=client, logger=logger, agent_id="research-test")

    assert tool.enabled is True
    assert tool.disabled_reason is None
    assert len(tool.search_many(["query"])) == 1
    assert tool.last_search_status == "completed"
    event_names = [call.args[0] for call in logger.log_event.call_args_list]
    assert "tavily_search_started" in event_names
    assert "tavily_search_succeeded" in event_names
    assert all(call.kwargs.get("agent_id") == "research-test" for call in logger.log_event.call_args_list)


@pytest.mark.parametrize("status_code", [432, 433])
def test_confirmed_quota_exhaustion_latches_circuit_until_reset(status_code: int) -> None:
    client = make_client()
    client.search.side_effect = _status_error("ForbiddenError", status_code)
    logger = Mock()
    tool = SearchTool(client=client, logger=logger)

    with pytest.raises(SearchQuotaExceededError):
        tool.search_many(["first", "second"])
    assert tool.enabled is False
    assert tool.disabled_reason == "quota_exhausted"
    assert tool.last_status_code == status_code
    assert tool.search_many(["second"]) == []
    assert client.search.call_count == 1
    assert "tavily_quota_exhausted" in [call.args[0] for call in logger.log_event.call_args_list]

    client.search.side_effect = None
    client.search.return_value = {"results": [result()]}
    tool.reset()
    assert tool.enabled is True
    assert tool.disabled_reason is None
    assert len(tool.search_many(["after reset"])) == 1
    assert client.search.call_count == 2


def test_http_429_is_transient_and_does_not_disable_search() -> None:
    client = make_client()
    client.search.side_effect = _status_error("UsageLimitExceededError", 429)
    tool = make_tool(client)

    with pytest.raises(SearchRateLimitError):
        tool.search_many(["query"])
    assert tool.enabled is True
    assert tool.disabled_reason is None
    assert tool.last_search_status == "rate_limited"
    assert client.search.call_count == 1


def test_ambiguous_forbidden_without_status_is_not_called_quota() -> None:
    client = make_client()
    client.search.side_effect = type("ForbiddenError", (Exception,), {})("restricted")
    tool = make_tool(client)

    with pytest.raises(SearchAccessError):
        tool.search("query")
    assert tool.enabled is False
    assert tool.disabled_reason == "access_denied"
    assert tool.last_status_code is None
    assert not isinstance(tool._translate_error(client.search.side_effect), SearchQuotaExceededError)


def test_pinned_tavily_sdk_adapter_preserves_discarded_http_status() -> None:
    from unittest.mock import Mock

    tool = SearchTool(api_key="test-key-never-sent", logger=Mock())
    response = Mock()
    response.status_code = 432
    response.json.return_value = {"detail": {"error": "plan limit"}}

    with pytest.raises(Exception) as caught:
        tool._client._handle_error_response(response)

    assert type(caught.value).__name__ == "ForbiddenError"
    assert caught.value.status_code == 432
    assert isinstance(tool._translate_error(caught.value), SearchQuotaExceededError)


def test_http_401_disables_as_auth_error_without_claiming_quota() -> None:
    client = make_client()
    client.search.side_effect = _status_error("InvalidAPIKeyError", 401)
    tool = make_tool(client)

    with pytest.raises(SearchAuthenticationError):
        tool.search("query")
    assert tool.enabled is False
    assert tool.disabled_reason == "authentication_error"
    assert tool.last_search_status == "authentication_error"


@pytest.mark.parametrize(
    ("error", "expected", "status"),
    [
        (type("TimeoutError", (Exception,), {})("timeout"), SearchTimeoutError, "timeout"),
        (type("ConnectionError", (Exception,), {})("network"), SearchNetworkError, "network_error"),
        (_status_error("HTTPError", 500), SearchAPIError, "server_error"),
        (_status_error("HTTPError", 502), SearchAPIError, "server_error"),
        (_status_error("HTTPError", 503), SearchAPIError, "server_error"),
        (_status_error("HTTPError", 504), SearchAPIError, "server_error"),
        (_status_error("UsageLimitExceededError", 429), SearchRateLimitError, "rate_limited"),
        (_status_error("ForbiddenError", 403), SearchAccessError, "access_denied"),
        (_status_error("BadRequestError", 400), SearchRequestError, "request_error"),
        (type("UnknownError", (Exception,), {})("unknown"), SearchAPIError, "search_error"),
    ],
)
def test_non_quota_failures_keep_their_category_without_quota_circuit(
    error: Exception,
    expected: type[Exception],
    status: str,
) -> None:
    client = make_client()
    client.search.side_effect = error
    tool = make_tool(client)

    with pytest.raises(expected):
        tool.search("query")
    assert tool.last_search_status == status
    if status != "access_denied":
        assert tool.enabled is True
    assert tool.disabled_reason != "quota_exhausted"


def test_unconfigured_deferred_search_latches_and_prevents_repeated_calls(monkeypatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "")
    tool = SearchTool(allow_unconfigured=True, logger=Mock())
    assert tool.enabled is True

    with pytest.raises(SearchConfigurationError):
        tool.search_many(["query"])
    assert tool.enabled is False
    assert tool.disabled_reason == "configuration_error"
    assert tool.search_many(["query again"]) == []


def test_logger_failure_does_not_break_success_or_circuit_transition() -> None:
    logger = Mock()
    logger.log_event.side_effect = OSError("disk full")
    tool = SearchTool(client=make_client({"results": [result()]}), logger=logger)
    assert len(tool.search("query")) == 1

    client = make_client()
    client.search.side_effect = _status_error("ForbiddenError", 432)
    quota_tool = SearchTool(client=client, logger=logger)
    with pytest.raises(SearchQuotaExceededError):
        quota_tool.search("query")
    assert quota_tool.enabled is False


def test_logs_never_contain_api_key() -> None:
    client = make_client()
    client.search.side_effect = _status_error("ForbiddenError", 432)
    logger = Mock()
    tool = SearchTool(api_key="synthetic-key-marker", client=client, logger=logger)

    with pytest.raises(SearchQuotaExceededError):
        tool.search("query")

    assert "synthetic-key-marker" not in repr(logger.log_event.call_args_list)
