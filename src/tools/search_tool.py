"""Provider-independent search interface backed by Tavily's official SDK."""

from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass
from typing import Any, Iterable
from urllib.parse import urlsplit, urlunsplit

from src.logging import Logger


class SearchToolError(Exception):
    """Base error for search configuration, transport, and response failures."""

    def __init__(self, *args: Any, status_code: int | None = None) -> None:
        super().__init__(*args)
        self.status_code = status_code


class SearchConfigurationError(SearchToolError):
    """Raised when the search provider is not configured or installed."""


class SearchAuthenticationError(SearchToolError):
    """Raised when Tavily rejects the configured credentials."""


class SearchRateLimitError(SearchToolError):
    """Raised for an HTTP 429 rate limit, which may be temporary."""


class SearchQuotaExceededError(SearchToolError):
    """Raised when Tavily confirms a plan or pay-as-you-go quota limit."""


class SearchAccessError(SearchToolError):
    """Raised when credentials or access policy forbid the request."""


class SearchRequestError(SearchToolError):
    """Raised when Tavily rejects invalid request parameters (HTTP 400)."""


class SearchTimeoutError(SearchToolError):
    """Raised when a Tavily request exceeds its timeout."""


class SearchNetworkError(SearchToolError):
    """Raised when the search provider cannot be reached."""


class SearchAPIError(SearchToolError):
    """Raised for provider errors that do not have a more specific category."""


class SearchResponseError(SearchToolError):
    """Raised when Tavily returns data that does not match its search schema."""


@dataclass(frozen=True)
class SearchResult:
    """A validated search result; ``snippet`` is provider-returned page context."""

    title: str
    url: str
    snippet: str
    relevance_score: float | None = None


class SearchTool:
    """Search the web through Tavily while keeping provider details replaceable."""

    MAX_RESULTS = 20
    MAX_QUERIES = 4
    MAX_TOTAL_RESULTS = 20

    def __init__(
        self,
        api_key: str | None = None,
        *,
        timeout: float = 15.0,
        client: Any | None = None,
        logger: Any | None = None,
        agent_id: str | None = None,
        allow_unconfigured: bool = False,
    ) -> None:
        """Create a search tool, or inject a compatible client for tests.

        The Tavily SDK is imported only when a real provider client is needed,
        so importing the project and running mock-based tests does not require
        the optional integration to be installed first.
        """
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("timeout must be a positive number.")
        self.timeout = float(timeout)
        self._logger = logger if logger is not None else Logger()
        if not callable(getattr(self._logger, "log_event", None)):
            raise TypeError("logger must provide log_event().")
        self.agent_id = agent_id
        self._lock = threading.RLock()
        self._enabled = True
        self._disabled_reason: str | None = None
        self.last_search_status = "ready"
        self.last_error_type: str | None = None
        self.last_status_code: int | None = None
        self._allow_unconfigured = allow_unconfigured
        self._api_key = api_key
        self._client = client
        if client is not None:
            return

        try:
            from dotenv import load_dotenv
        except ImportError:
            pass
        else:
            load_dotenv()

        resolved_api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not resolved_api_key:
            if allow_unconfigured:
                self._api_key = None
                return
            raise SearchConfigurationError(
                "TAVILY_API_KEY is not set. Configure it in the environment or .env."
            )
        self._api_key = resolved_api_key
        if not allow_unconfigured:
            self._client = self._create_client(resolved_api_key)

    @property
    def enabled(self) -> bool:
        """Whether new Tavily requests may be attempted."""
        with self._lock:
            return self._enabled

    @property
    def disabled_reason(self) -> str | None:
        """Return the stable, secret-free reason for a latched disable state."""
        with self._lock:
            return self._disabled_reason

    def reset(self) -> None:
        """Manually re-enable Tavily after checking quota, access, or config."""
        with self._lock:
            self._enabled = True
            self._disabled_reason = None
            self.last_search_status = "ready"
            self.last_error_type = None
            self.last_status_code = None
        self.log_event("tavily_manually_reenabled")

    def set_agent_id(self, agent_id: str | None) -> None:
        """Attach an agent ID to subsequent search log events."""
        if agent_id is not None and (not isinstance(agent_id, str) or not agent_id.strip()):
            raise ValueError("agent_id must be a non-empty string or None.")
        self.agent_id = agent_id

    def log_event(self, event_type: str, **fields: Any) -> None:
        """Write a best-effort structured search event without affecting work."""
        record = dict(fields)
        if self.agent_id is not None:
            record.setdefault("agent_id", self.agent_id)
        try:
            self._logger.log_event(event_type, **record)
        except Exception:
            return

    @staticmethod
    def _create_client(api_key: str) -> Any:
        try:
            from tavily import TavilyClient
        except ImportError as error:
            raise SearchConfigurationError(
                "The Tavily SDK is unavailable. Install project dependencies first."
            ) from error

        class StatusAwareTavilyClient(TavilyClient):
            """Preserve HTTP codes discarded by the pinned Tavily SDK errors."""

            def _handle_error_response(self, response: Any) -> None:
                try:
                    super()._handle_error_response(response)
                except Exception as error:
                    try:
                        error.status_code = response.status_code
                    except Exception:
                        pass
                    raise

        try:
            return StatusAwareTavilyClient(api_key=api_key)
        except Exception:
            raise SearchConfigurationError(
                "Could not initialize the Tavily search client."
            ) from None

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        if not self._api_key:
            raise SearchConfigurationError(
                "TAVILY_API_KEY is not set. Configure it in the environment or .env."
            )
        self._client = self._create_client(self._api_key)

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """Return unique, validated results for one search query."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")
        if (
            isinstance(max_results, bool)
            or not isinstance(max_results, int)
            or not 1 <= max_results <= self.MAX_RESULTS
        ):
            raise ValueError(f"max_results must be an integer from 1 to {self.MAX_RESULTS}.")

        if not self.enabled:
            self.last_search_status = "disabled"
            return []
        self.log_event("tavily_search_started", query_count=1)

        try:
            self._ensure_client()
            response = self._client.search(
                query=query.strip(),
                max_results=max_results,
                timeout=self.timeout,
                include_answer=False,
                include_raw_content=False,
            )
            if not isinstance(response, dict) or not isinstance(response.get("results"), list):
                raise SearchResponseError("Tavily returned an invalid search response.")

            results: list[SearchResult] = []
            seen_urls: set[str] = set()
            for item in response["results"]:
                parsed = self._parse_result(item)
                if parsed is None:
                    continue
                normalized_url = self._normalize_url(parsed.url)
                if normalized_url in seen_urls:
                    continue
                seen_urls.add(normalized_url)
                results.append(parsed)
                if len(results) >= max_results:
                    break
        except Exception as error:
            translated = error if isinstance(error, SearchToolError) else self._translate_error(error)
            self._record_failure(translated)
            raise translated from None

        self.last_search_status = "completed" if results else "no_results"
        self.log_event("tavily_search_succeeded", result_count=len(results), query_count=1)
        return results

    def search_many(
        self,
        queries: Iterable[str],
        *,
        max_results_per_query: int = 5,
        max_total_results: int = 12,
        max_queries: int = MAX_QUERIES,
    ) -> list[SearchResult]:
        """Search multiple queries and deduplicate/cap their combined sources."""
        if (
            isinstance(max_total_results, bool)
            or not isinstance(max_total_results, int)
            or not 1 <= max_total_results <= self.MAX_TOTAL_RESULTS
        ):
            raise ValueError(
                f"max_total_results must be an integer from 1 to {self.MAX_TOTAL_RESULTS}."
            )
        if (
            isinstance(max_queries, bool)
            or not isinstance(max_queries, int)
            or not 1 <= max_queries <= self.MAX_QUERIES
        ):
            raise ValueError(f"max_queries must be between 1 and {self.MAX_QUERIES}.")

        query_list = list(queries)
        if any(not isinstance(query, str) or not query.strip() for query in query_list):
            raise ValueError("queries must contain only non-empty strings.")

        if not self.enabled:
            self.last_search_status = "disabled"
            self.log_event("tavily_search_disabled", reason=self.disabled_reason,
                           query_count=min(len(query_list), max_queries))
            return []
        self.log_event("tavily_search_started", query_count=min(len(query_list), max_queries))

        combined: list[SearchResult] = []
        seen_urls: set[str] = set()
        for query in query_list[:max_queries]:
            remaining = max_total_results - len(combined)
            if remaining <= 0:
                break
            per_query = min(max_results_per_query, remaining)
            for result in self.search(query, max_results=per_query):
                normalized_url = self._normalize_url(result.url)
                if normalized_url in seen_urls:
                    continue
                seen_urls.add(normalized_url)
                combined.append(result)
                if len(combined) >= max_total_results:
                    break
        self.last_search_status = "completed" if combined else "no_results"
        self.log_event("tavily_search_succeeded", result_count=len(combined),
                       query_count=min(len(query_list), max_queries))
        return combined

    def _record_failure(self, error: SearchToolError) -> None:
        status_code = getattr(error, "status_code", None)
        self.last_error_type = type(error).__name__
        self.last_status_code = status_code if isinstance(status_code, int) else None
        self.last_search_status = self._status_for_error(error)
        self.log_event(
            "tavily_request_failed",
            error_type=self.last_error_type,
            status_code=self.last_status_code,
            search_status=self.last_search_status,
        )
        if isinstance(error, SearchQuotaExceededError):
            self._disable("quota_exhausted", event_type="tavily_quota_exhausted")
        elif isinstance(error, (SearchAuthenticationError, SearchAccessError, SearchConfigurationError)):
            self._disable(self.last_search_status, event_type="tavily_disabled")

    def _disable(self, reason: str, *, event_type: str) -> None:
        with self._lock:
            self._enabled = False
            self._disabled_reason = reason
        self.log_event(event_type, reason=reason, status_code=self.last_status_code,
                       error_type=self.last_error_type)

    @staticmethod
    def _status_for_error(error: SearchToolError) -> str:
        if isinstance(error, SearchQuotaExceededError):
            return "quota_exhausted"
        if isinstance(error, SearchAuthenticationError):
            return "authentication_error"
        if isinstance(error, SearchAccessError):
            return "access_denied"
        if isinstance(error, SearchConfigurationError):
            return "configuration_error"
        if isinstance(error, SearchRateLimitError):
            return "rate_limited"
        if isinstance(error, SearchTimeoutError):
            return "timeout"
        if isinstance(error, SearchNetworkError):
            return "network_error"
        if isinstance(error, SearchRequestError):
            return "request_error"
        if isinstance(error, SearchAPIError) and getattr(error, "status_code", None) in {500, 502, 503, 504}:
            return "server_error"
        return "search_error"

    @classmethod
    def _parse_result(cls, item: Any) -> SearchResult | None:
        if not isinstance(item, dict):
            raise SearchResponseError("Tavily returned a malformed search result.")
        title = item.get("title")
        url = item.get("url")
        snippet = item.get("content", item.get("snippet"))
        if not isinstance(title, str) or not title.strip():
            raise SearchResponseError("A Tavily result is missing a valid title.")
        if not isinstance(url, str) or not isinstance(snippet, str):
            raise SearchResponseError("A Tavily result is missing a valid URL or snippet.")
        if not snippet.strip():
            raise SearchResponseError("A Tavily result contains an empty snippet.")
        if not cls._is_http_url(url):
            return None

        score = item.get("score")
        if score is not None:
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise SearchResponseError("A Tavily relevance score must be numeric or null.")
            if not math.isfinite(score):
                raise SearchResponseError("A Tavily relevance score must be finite.")
            score = float(score)
        return SearchResult(
            title=title.strip(),
            url=url.strip(),
            snippet=snippet.strip(),
            relevance_score=score,
        )

    @staticmethod
    def _is_http_url(url: str) -> bool:
        try:
            parsed = urlsplit(url.strip())
            return (
                parsed.scheme in {"http", "https"}
                and bool(parsed.hostname)
                and (parsed.port is None or 0 < parsed.port <= 65535)
            )
        except ValueError:
            return False

    @staticmethod
    def _normalize_url(url: str) -> str:
        parsed = urlsplit(url.strip())
        host = (parsed.hostname or "").lower()
        if parsed.port:
            host = f"{host}:{parsed.port}"
        path = parsed.path.rstrip("/") or "/"
        return urlunsplit((parsed.scheme.lower(), host, path, parsed.query, ""))

    @staticmethod
    def _translate_error(error: Exception) -> SearchToolError:
        """Map SDK/transport exceptions without echoing provider error bodies."""
        error_name = type(error).__name__.lower()
        status_code = SearchTool._extract_status_code(error)
        if "timeout" in error_name:
            return SearchTimeoutError("The Tavily search request timed out.", status_code=status_code)
        if status_code in {432, 433}:
            return SearchQuotaExceededError(
                "Tavily confirmed that the configured quota is exhausted.",
                status_code=status_code,
            )
        if status_code == 401 or "invalidapikey" in error_name or "missingapikey" in error_name:
            return SearchAuthenticationError(
                "Tavily rejected the configured API credentials.", status_code=status_code
            )
        if status_code == 403 or "forbidden" in error_name:
            return SearchAccessError("Tavily denied access to this search request.", status_code=status_code)
        if status_code == 400 or "badrequest" in error_name:
            return SearchRequestError("Tavily rejected the search request parameters.", status_code=status_code)
        if status_code == 429 or "ratelimit" in error_name or "usagelimit" in error_name:
            return SearchRateLimitError("Tavily temporarily rate limited the request.", status_code=status_code)
        if status_code is not None and status_code >= 500:
            return SearchAPIError("The Tavily search service returned a server error.", status_code=status_code)
        if any(token in error_name for token in ("connection", "network", "requestexception")):
            return SearchNetworkError("Could not connect to the Tavily search API.", status_code=status_code)
        return SearchAPIError("The Tavily search API returned an error.", status_code=status_code)

    @staticmethod
    def _extract_status_code(error: Exception) -> int | None:
        values = (
            getattr(error, "status_code", None),
            getattr(error, "status", None),
            getattr(getattr(error, "response", None), "status_code", None),
        )
        for value in values:
            if isinstance(value, int) and not isinstance(value, bool):
                return value
        return None
