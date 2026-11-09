"""Mocked tests for CriticAgent orchestration and verdict enforcement."""

import json
from unittest.mock import Mock

import pytest

from src.agents import (
    AgentConfigRegistry,
    CriticAgent,
    CriticLLMError,
    CriticReport,
    CriticReportParseError,
)
from src.llm import LLMAPIError, LLMClient, LLMResponse
from src.tools import CriticTool


@pytest.fixture
def client() -> Mock:
    return Mock(spec=LLMClient)


@pytest.fixture
def agent(client: Mock) -> CriticAgent:
    return CriticAgent(
        client=client,
        config=AgentConfigRegistry().get_agent("critic"),
        agent_id="critic-test",
    )


def payload(*, verdict: str = "APPROVED", score: int = 94, issues: list | None = None) -> str:
    return json.dumps({
        "verdict": verdict,
        "score": score,
        "issues": issues or [],
        "strengths": ["Ответ сопоставлен с задачей."],
        "summary": "Результат оценён по задаче и применимым критериям.",
        "criteria": {
            "correctness": "Нет выявленных ошибок.",
            "task_compliance": "Требования выполнены.",
            "completeness": "Обязательные элементы присутствуют.",
            "logical_consistency": "Внутренних противоречий не найдено.",
            "evidence_and_verification": "Ограничения проверки указаны.",
        },
        "verification_limits": [],
    })


def response(content: str) -> LLMResponse:
    return LLMResponse(content, "mock/reasoning", 10, 20, 30, "stop")


def test_initialization_uses_configured_critic_role(agent: CriticAgent) -> None:
    assert agent.config.name == "critic"
    assert agent.config.logical_model_id == "reasoning"
    assert "независимый Critic Agent" in agent.system_prompt


def test_review_sends_task_result_criteria_and_returns_report(client: Mock, agent: CriticAgent) -> None:
    client.generate.return_value = response(payload())
    report = agent.review(
        "Напиши функцию сложения", {"code": "def add(a, b): return a + b", "tests": "assert add(1, 2) == 3"},
        result_type="coding", agent_name="Coding Agent",
    )
    assert isinstance(report, CriticReport)
    assert report.verdict == "APPROVED"
    sent = client.generate.call_args.kwargs["messages"][1]["content"]
    assert "Напиши функцию сложения" in sent
    assert "assert add(1, 2) == 3" in sent
    assert "deterministic_checks" in sent
    assert client.generate.call_args.kwargs["agent_id"] == "critic-test"


def test_valid_coding_result_can_be_approved(agent: CriticAgent, client: Mock) -> None:
    client.generate.return_value = response(payload())
    report = agent.review(
        "Напиши функцию add", {"code": "def add(a, b):\n    return a + b", "tests": "test_add"},
        result_type="coding",
    )
    assert report.verdict == "APPROVED"
    assert report.checks[0]["status"] == "passed"


def test_critical_binary_search_bug_overrides_high_score_approval(agent: CriticAgent, client: Mock) -> None:
    client.generate.return_value = response(payload(verdict="APPROVED", score=99))
    report = agent.review(
        "Implement binary search and return -1 when item is absent.",
        {"code": "def binary_search(items, target):\n    return 0", "tests": ""},
        result_type="coding",
    )
    assert report.verdict == "REJECTED"
    assert any(issue.category == "TASK_COMPLIANCE" for issue in report.issues)


def test_unprovided_research_url_is_rejected(agent: CriticAgent, client: Mock) -> None:
    client.generate.return_value = response(payload())
    report = agent.review(
        "Compare RAG and fine-tuning.",
        "RAG is more current (https://fake.example/rag).",
        result_type="research",
        additional_materials={"sources": [{"url": "https://real.example/rag", "snippet": "RAG retrieves context"}]},
    )
    assert report.verdict == "REJECTED"
    assert any(issue.category == "EVIDENCE" for issue in report.issues)


def test_no_search_research_report_discloses_unavailable_external_verification(
    agent: CriticAgent,
    client: Mock,
) -> None:
    client.generate.return_value = response(payload())
    result = {
        "summary": "Осторожный ответ из общих знаний.",
        "key_findings": [{"claim": "Непроверенное утверждение.", "source_urls": []}],
        "sources": [],
        "limitations": ["Актуальные источники не проверялись."],
        "search_performed": False,
        "search_status": "quota_exhausted",
        "search_unavailable_reason": "quota_exhausted",
    }

    review = agent.review("Question", result, result_type="research")

    sent = client.generate.call_args.kwargs["messages"][1]["content"]
    assert '"external_fact_checking": "not_performed"' in sent
    assert any(check["name"] == "external_fact_verification"
               and check["status"] == "unavailable" for check in review.checks)
    assert not any(issue.category == "EVIDENCE" for issue in review.issues)
    assert any("external sources" in item for item in review.verification_limits)


def test_math_error_is_independently_recomputed_and_rejected(agent: CriticAgent, client: Mock) -> None:
    client.generate.return_value = response(payload(verdict="APPROVED", score=90))
    report = agent.review(
        "Find minimum of f(x)=x**2-6*x+11.",
        {"expression": "x**2 - 6*x + 11", "variable": "x", "point": 2, "value": 3},
        result_type="math",
    )
    assert report.verdict == "REJECTED"
    assert any(issue.category == "CORRECTNESS" for issue in report.issues)
    assert "math_substitution" in [check["name"] for check in report.checks]


def test_llm_critical_issue_cannot_be_approved_with_high_score(agent: CriticAgent, client: Mock) -> None:
    issue = {
        "category": "CORRECTNESS", "severity": "CRITICAL", "description": "Incorrect answer.",
        "location": None, "recommendation": "Recalculate the answer.",
    }
    client.generate.return_value = response(payload(verdict="APPROVED", score=99, issues=[issue]))
    report = agent.review("Task", "Answer", result_type="general")
    assert report.verdict == "REJECTED"


def test_malformed_json_and_llm_errors_are_distinguished(agent: CriticAgent, client: Mock) -> None:
    client.generate.return_value = response("not json")
    with pytest.raises(CriticReportParseError):
        agent.review("Task", "Answer")
    client.generate.side_effect = LLMAPIError("provider failed")
    with pytest.raises(CriticLLMError):
        agent.review("Task", "Answer")


def test_unknown_result_type_discloses_limits_and_still_uses_general_review(agent: CriticAgent, client: Mock) -> None:
    client.generate.return_value = response(payload())
    report = agent.review("Review this", "Some prose", result_type="unknown-specialized-agent")
    assert report.result_type == "unknown-specialized-agent"
    assert report.verdict == "APPROVED"
    assert report.checks == []


def test_agent_handles_required_fields_and_rejects_missing_ones(agent: CriticAgent, client: Mock) -> None:
    client.generate.return_value = response(payload())
    report = agent.review("Return code and tests", {"code": "print(1)"}, required_elements=("code", "tests"))
    assert report.verdict == "REJECTED"
    assert any(issue.category == "COMPLETENESS" for issue in report.issues)


def test_rejects_invalid_input_before_calling_llm(agent: CriticAgent, client: Mock) -> None:
    from src.agents import AgentInputError
    with pytest.raises(AgentInputError):
        agent.review(" ", "result")
    client.generate.assert_not_called()
