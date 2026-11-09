"""External tools that can be injected into agents."""

from .math_tool import (
    MathComputationError,
    MathExpressionError,
    MathInputError,
    MathTool,
    MathToolError,
    MathToolResult,
    OptimizationError,
)
from .critic_tool import CriticCheck, CriticFinding, CriticTool, CriticToolResult
from .search_tool import (
    SearchAPIError,
    SearchAccessError,
    SearchAuthenticationError,
    SearchConfigurationError,
    SearchNetworkError,
    SearchQuotaExceededError,
    SearchRateLimitError,
    SearchRequestError,
    SearchResponseError,
    SearchResult,
    SearchTimeoutError,
    SearchTool,
    SearchToolError,
)

__all__ = [
    "CriticCheck",
    "CriticFinding",
    "CriticTool",
    "CriticToolResult",
    "MathComputationError",
    "MathExpressionError",
    "MathInputError",
    "MathTool",
    "MathToolError",
    "MathToolResult",
    "OptimizationError",
    "SearchAPIError",
    "SearchAccessError",
    "SearchAuthenticationError",
    "SearchConfigurationError",
    "SearchNetworkError",
    "SearchQuotaExceededError",
    "SearchRateLimitError",
    "SearchRequestError",
    "SearchResponseError",
    "SearchResult",
    "SearchTimeoutError",
    "SearchTool",
    "SearchToolError",
]
