"""Deterministic, non-executing CriticTool tests."""

from src.tools import CriticTool


def test_python_ast_accepts_valid_code_without_executing_it() -> None:
    result = CriticTool().check_python("def add(a, b):\n    return a + b\n")
    assert result.checks[0].status == "passed"
    assert not result.findings


def test_python_ast_reports_syntax_error() -> None:
    result = CriticTool().check_python("def broken(:\n    pass\n")
    assert result.checks[0].status == "failed"
    assert result.findings[0].severity == "CRITICAL"


def test_structure_reports_missing_required_elements() -> None:
    result = CriticTool().check_structure({"code": "print('ok')"}, required_elements=("code", "tests"))
    assert result.checks[0].status == "failed"
    assert [finding.location for finding in result.findings] == ["tests"]


def test_structure_reports_unavailable_for_unstructured_text() -> None:
    result = CriticTool().check_structure("plain text", required_elements=("code",))
    assert result.checks[0].status == "unavailable"
    assert result.limitations


def test_source_url_mismatch_is_reported() -> None:
    result = CriticTool().check_source_urls(
        "RAG помогает обновлять знания (https://unknown.example/article).",
        ["https://known.example/article"],
    )
    assert result.checks[0].status == "failed"
    assert result.findings[0].category == "EVIDENCE"
    assert "unknown.example" in result.findings[0].description


def test_source_url_match_passes_but_does_not_claim_source_truth() -> None:
    result = CriticTool().check_source_urls(
        "Источник: https://known.example/article", ["https://known.example/article/"]
    )
    assert result.checks[0].status == "passed"
    assert not result.findings


def test_math_substitution_detects_wrong_value_and_nonstationary_point() -> None:
    result = CriticTool().check_math_value(
        expression="x**2 - 6*x + 11",
        variable="x",
        reported_point=2,
        reported_value=3,
    )
    assert result.checks[0].status == "failed"
    assert len(result.findings) == 1
    assert result.findings[0].severity == "MAJOR"
    assert "derivative" in result.findings[0].description


def test_python_checker_flags_dynamic_execution_without_running_code() -> None:
    result = CriticTool().check_python("def f(value):\n    return eval(value)\n")
    assert any(f.category == "SECURITY" for f in result.findings)


def test_binary_search_not_found_contract_is_checked_statically() -> None:
    result = CriticTool().check_python(
        "def binary_search(items, target):\n    return 0\n",
        task="Implement binary search; return -1 when the item is not found.",
    )
    assert any("not-found sentinel" in f.description for f in result.findings)


def test_binary_search_valid_index_zero_return_is_not_misidentified_as_fallback() -> None:
    result = CriticTool().check_python(
        "def binary_search(items, target):\n"
        "    left, right = 0, len(items) - 1\n"
        "    while left <= right:\n"
        "        mid = (left + right) // 2\n"
        "        if items[mid] == target:\n"
        "            return mid\n"
        "        if items[mid] < target:\n"
        "            left = mid + 1\n"
        "        else:\n"
        "            right = mid - 1\n"
        "    return -1\n"
        "def at_zero():\n"
        "    return 0\n",
        task="Binary search returns -1 when not found.",
    )
    assert not any("not-found sentinel" in f.description for f in result.findings)
