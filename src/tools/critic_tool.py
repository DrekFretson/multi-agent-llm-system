"""Non-executing, deterministic checks used to support CriticAgent reviews."""

from __future__ import annotations

import ast
import re
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

from src.tools.math_tool import MathTool, MathToolError


@dataclass(frozen=True)
class CriticCheck:
    """A deterministic check and its explicit status."""

    name: str
    status: str
    details: str


@dataclass(frozen=True)
class CriticFinding:
    """A concrete problem found by a deterministic check."""

    category: str
    severity: str
    description: str
    location: str | None
    recommendation: str


@dataclass(frozen=True)
class CriticToolResult:
    """Results from local checks, kept separate from LLM judgments."""

    checks: list[CriticCheck]
    findings: list[CriticFinding]
    limitations: list[str]


class CriticTool:
    """Run safe syntax, structure, citation, and supported math checks.

    This tool parses submitted Python with :mod:`ast`; it never imports or
    executes submitted code. Math expressions are delegated to the existing
    restricted-grammar :class:`MathTool`.
    """

    _URL_PATTERN = re.compile(r"https?://[^\s\]\[()<>\"']+", re.IGNORECASE)
    _DANGEROUS_CALLS = {"eval", "exec", "compile", "__import__"}

    def __init__(self, math_tool: MathTool | None = None) -> None:
        self.math_tool = math_tool or MathTool()

    def check_python(self, code: str, *, task: str = "") -> CriticToolResult:
        """Parse Python and flag a few concrete static hazards without running it."""
        if not isinstance(code, str) or not code.strip():
            return CriticToolResult(
                checks=[CriticCheck("python_ast", "unavailable", "No Python source was provided.")],
                findings=[],
                limitations=["Python syntax and static checks could not run because code was absent."],
            )
        try:
            tree = ast.parse(code)
        except SyntaxError as error:
            location = f"line {error.lineno}" if error.lineno else None
            return CriticToolResult(
                checks=[CriticCheck("python_ast", "failed", f"SyntaxError: {error.msg}")],
                findings=[CriticFinding(
                    "CORRECTNESS", "CRITICAL", f"Python source has a syntax error: {error.msg}.",
                    location, "Fix the syntax error and parse the complete source again.",
                )],
                limitations=[],
            )

        findings: list[CriticFinding] = []
        checks = [CriticCheck("python_ast", "passed", "Python source parsed successfully.")]
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in self._DANGEROUS_CALLS:
                    findings.append(CriticFinding(
                        "SECURITY", "MAJOR",
                        f"The source calls {node.func.id}(), which can execute dynamically supplied code.",
                        f"line {node.lineno}",
                        "Remove dynamic code execution and use explicit, validated operations.",
                    ))
        if findings:
            checks.append(CriticCheck("dangerous_dynamic_execution", "failed", "Dynamic execution primitive found."))
        else:
            checks.append(CriticCheck("dangerous_dynamic_execution", "passed", "No eval/exec/compile/__import__ call found."))

        if self._is_binary_search_task(task):
            zero_returns = []
            for function in ast.walk(tree):
                if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if "binary" not in function.name.lower() or "search" not in function.name.lower():
                    continue
                loop_positions = [
                    index for index, statement in enumerate(function.body)
                    if isinstance(statement, (ast.For, ast.AsyncFor, ast.While))
                ]
                for index, statement in enumerate(function.body):
                    is_zero_return = (
                        isinstance(statement, ast.Return)
                        and isinstance(statement.value, ast.Constant)
                        and statement.value.value == 0
                    )
                    # Catch a bare always-zero function or a literal fallback
                    # after the search loop, without flagging a valid early
                    # return for an item at index zero.
                    if is_zero_return and (len(function.body) == 1 or (loop_positions and index > max(loop_positions))):
                        zero_returns.append(statement)
            if zero_returns and re.search(r"(?:return|returns?|вернуть|возвращать)\s+(-1|None|null)", task, re.I):
                findings.append(CriticFinding(
                    "TASK_COMPLIANCE", "MAJOR",
                    "The binary-search function has a `return 0` path although the task requires a -1/None not-found sentinel.",
                    f"line {zero_returns[0].lineno}",
                    "Return the specified not-found sentinel on every unsuccessful search path.",
                ))
                checks.append(CriticCheck("binary_search_not_found_contract", "failed", "A zero return conflicts with the requested not-found sentinel."))
        return CriticToolResult(checks, findings, [])

    def check_structure(
        self,
        result: Any,
        *,
        required_elements: Sequence[str] = (),
    ) -> CriticToolResult:
        """Check that a structured result includes requested non-empty fields."""
        payload = self.to_plain_data(result)
        if not isinstance(payload, Mapping):
            return CriticToolResult(
                [CriticCheck("result_structure", "unavailable", "Result is not a mapping or dataclass.")],
                [],
                ["Structural field checks require a mapping or dataclass result."],
            )
        missing = [name for name in required_elements if name not in payload or payload[name] in (None, "", [], {})]
        checks = [CriticCheck(
            "result_structure", "failed" if missing else "passed",
            "Missing or empty required fields: " + ", ".join(missing) if missing else "All requested fields are present.",
        )]
        findings = [CriticFinding(
            "COMPLETENESS", "MAJOR", f"Required result element {name!r} is missing or empty.", name,
            f"Provide a non-empty {name!r} field that satisfies the task.",
        ) for name in missing]
        return CriticToolResult(checks, findings, [])

    def check_source_urls(
        self,
        text: str,
        provided_sources: Sequence[str] | Sequence[Mapping[str, Any]],
    ) -> CriticToolResult:
        """Ensure URLs cited in text were included in the supplied source set."""
        if not isinstance(text, str):
            return CriticToolResult(
                [CriticCheck("source_url_match", "unavailable", "No textual report was provided.")],
                [], ["Citation checks require a textual report."],
            )
        allowed: set[str] = set()
        for source in provided_sources:
            value = source if isinstance(source, str) else source.get("url") if isinstance(source, Mapping) else None
            if isinstance(value, str) and value.strip():
                allowed.add(self._normalize_url(value))
        cited = {self._normalize_url(url.rstrip(".,;:")) for url in self._URL_PATTERN.findall(text)}
        unmatched = sorted(cited - allowed)
        finding_list = [CriticFinding(
            "EVIDENCE", "MAJOR", f"Cited URL {url!r} is not present in the provided source list.", url,
            "Remove the citation or include and verify the corresponding source material.",
        ) for url in unmatched]
        status = "unavailable" if not allowed else "failed" if unmatched else "passed"
        limitation = ["No source URLs were supplied, so URL-to-source matching was not possible."] if not allowed else []
        detail = "Unmatched URLs: " + ", ".join(unmatched) if unmatched else "Every cited URL matches a provided source URL." if allowed else "No source list available."
        return CriticToolResult([CriticCheck("source_url_match", status, detail)], finding_list, limitation)

    def check_math_value(
        self,
        *,
        expression: str,
        variable: str,
        reported_point: float,
        reported_value: float,
        check_stationary: bool = True,
        tolerance: float = 1e-7,
    ) -> CriticToolResult:
        """Independently substitute a reported point/value using safe MathTool operations."""
        try:
            actual = self.math_tool.evaluate_expression(expression, {variable: reported_point})
            actual_value = float(actual.result)
            derivative = self.math_tool.derivative(expression, variable)
            derivative_expr = derivative.result["derivative"]
            slope = float(self.math_tool.evaluate_expression(derivative_expr, {variable: reported_point}).result)
        except (MathToolError, KeyError, TypeError, ValueError, OverflowError) as error:
            return CriticToolResult(
                [CriticCheck("math_substitution", "unavailable", f"Could not verify supplied math: {error}")],
                [], ["Independent mathematical substitution could not be completed."],
            )
        findings: list[CriticFinding] = []
        if abs(actual_value - reported_value) > tolerance * max(1.0, abs(actual_value)):
            findings.append(CriticFinding(
                "CORRECTNESS", "CRITICAL",
                f"At {variable}={reported_point}, the expression evaluates to {actual_value}, not the reported {reported_value}.",
                "reported value", "Recalculate the objective value by substituting the reported point into the original expression.",
            ))
        if check_stationary and abs(slope) > tolerance:
            findings.append(CriticFinding(
                "CORRECTNESS", "MAJOR",
                f"The derivative at the reported minimum point is {slope}, not approximately zero.",
                "reported point", "Re-evaluate the stationary point and verify that it satisfies necessary optimality conditions.",
            ))
        ok = not findings
        return CriticToolResult(
            [CriticCheck("math_substitution", "passed" if ok else "failed",
                         f"Recomputed f({variable}={reported_point})={actual_value}; derivative={slope}.")],
            findings,
            [] if ok else ["This local check verifies the reported value and first-order stationarity only; it does not establish global optimality."],
        )

    @staticmethod
    def to_plain_data(value: Any) -> Any:
        """Convert common report dataclasses and nested containers to JSON data."""
        if is_dataclass(value) and not isinstance(value, type):
            return CriticTool.to_plain_data(asdict(value))
        if isinstance(value, Mapping):
            return {str(key): CriticTool.to_plain_data(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [CriticTool.to_plain_data(item) for item in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if hasattr(value, "__dict__"):
            return CriticTool.to_plain_data(vars(value))
        return str(value)

    @staticmethod
    def _normalize_url(url: str) -> str:
        parts = urlsplit(url.strip())
        return f"{parts.scheme.lower()}://{parts.netloc.lower()}{parts.path.rstrip('/')}" + (f"?{parts.query}" if parts.query else "")

    @staticmethod
    def _is_binary_search_task(task: str) -> bool:
        return "binary" in task.lower() and "search" in task.lower() or "бинарн" in task.lower() and "поиск" in task.lower()
