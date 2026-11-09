"""Mocked planning and sequential execution tests for Orchestrator."""

import json
from dataclasses import dataclass
from unittest.mock import Mock

import pytest

from src.agents import (
    AgentConfigRegistry,
    AgentResult,
    CriticReport,
    CodingSolution,
    Orchestrator,
    OrchestratorPlan,
    OrchestratorPlanError,
    OrchestratorPlanParseError,
    Subtask,
)
from src.llm import LLMAPIError, LLMClient, LLMResponse


@dataclass
class SimpleReport:
    summary: str


@pytest.fixture
def client() -> Mock:
    return Mock(spec=LLMClient)


def llm_response(content: str) -> LLMResponse:
    return LLMResponse(content, "mock/reasoning", 5, 10, 15, "stop")


def plan_json(*subtasks: dict) -> str:
    return json.dumps({"subtasks": list(subtasks)})


def task(identifier: str, agent: str, depends_on: list[str] | None = None, description: str | None = None) -> dict:
    return {
        "id": identifier,
        "description": description or f"Выполни этап {identifier}",
        "agent": agent,
        "depends_on": depends_on or [],
    }


def create_orchestrator(client: Mock, **agents: Mock) -> Orchestrator:
    return Orchestrator(
        client=client,
        agents=agents,
        config=AgentConfigRegistry().get_agent("orchestrator"),
        agent_id="orchestrator-test",
    )


def test_simple_coding_request_uses_only_coding_agent(client: Mock) -> None:
    coding = Mock()
    solution = CodingSolution("Python", "Binary search", "def f(): pass", "assert True", [], "pytest")
    coding.run.return_value = AgentResult("coding", '{"ok":true}', "mock/code", True, None, 1, 1)
    coding.parse_solution.return_value = solution
    client.generate.side_effect = [
        llm_response(plan_json(task("code", "coding"))),
        llm_response("Функция бинарного поиска и тесты приведены ниже."),
    ]
    orchestrator = create_orchestrator(client, coding=coding)

    result = orchestrator.run("Напиши функцию бинарного поиска на Python")

    assert result.status == "COMPLETED"
    assert result.execution_order == ["code"]
    assert result.subtask_results["code"] == solution
    coding.run.assert_called_once()
    coding.parse_solution.assert_called_once()
    assert "функция бинарного поиска" in result.final_answer.lower()


def test_simple_general_question_can_be_answered_without_specialized_agents(client: Mock) -> None:
    client.generate.side_effect = [
        llm_response(plan_json()),
        llm_response("JSONL хранит по одному JSON-объекту в каждой строке."),
    ]
    orchestrator = create_orchestrator(client)

    result = orchestrator.run("Что такое JSONL?")

    assert result.status == "COMPLETED"
    assert result.plan.subtasks == []
    assert result.execution_order == []
    assert "JSONL" in result.final_answer


def test_research_request_calls_only_research_and_builds_final_answer(client: Mock) -> None:
    research = Mock()
    report = SimpleReport("RAG combines retrieval with generation.")
    research.run_research.return_value = report
    client.generate.side_effect = [
        llm_response(plan_json(task("research", "research"))),
        llm_response("RAG retrieves context before generating a response."),
    ]
    orchestrator = create_orchestrator(client, research=research)

    result = orchestrator.run("Исследуй RAG и его преимущества и недостатки")

    assert result.status == "COMPLETED"
    assert result.subtask_results["research"] == report
    research.run_research.assert_called_once()
    assert "RAG" in client.generate.call_args_list[1].kwargs["messages"][1]["content"]


def test_complex_plan_runs_agents_sequentially_and_passes_dependency_results(client: Mock) -> None:
    events: list[str] = []
    research = Mock()
    research.run_research.side_effect = lambda _: (events.append("research") or SimpleReport("Research findings"))
    math = Mock()
    math.solve.side_effect = lambda prompt: (events.append("math") or SimpleReport("Mathematical model"))
    coding = Mock()
    coding.run.side_effect = lambda *args, **kwargs: (
        events.append("coding") or AgentResult("coding", "{}", "mock/code", True, None, 1, 2)
    )
    solution = CodingSolution("Python", "Route optimizer", "def optimize(): pass", "assert True", [], "pytest")
    coding.parse_solution.return_value = solution
    critic = Mock()
    critic.review.side_effect = lambda **kwargs: (
        events.append("critic") or CriticReport("APPROVED", 95, [], ["Reviewed"], "Looks correct", "coding")
    )
    client.generate.side_effect = [
        llm_response(plan_json(
            task("r", "research"),
            task("m", "math", ["r"]),
            task("c", "coding", ["r", "m"]),
            task("check", "critic", ["c"]),
        )),
        llm_response("Research, mathematical formulation, implementation, and review are complete."),
    ]
    orchestrator = create_orchestrator(client, research=research, math=math, coding=coding, critic=critic)

    result = orchestrator.run("Исследуй, формализуй, реализуй и проверь оптимизацию маршрута")

    assert result.status == "COMPLETED"
    assert events == ["research", "math", "coding", "critic"]
    assert result.execution_order == ["r", "m", "c", "check"]
    coding_context = coding.run.call_args.kwargs["additional_context"]
    assert "Research findings" in coding_context
    assert "Mathematical model" in coding_context
    critic.review.assert_called_once()
    assert critic.review.call_args.kwargs["result"] == solution
    assert "original_request" not in critic.review.call_args.kwargs


@pytest.mark.parametrize(
    ("subtasks", "message"),
    [
        ([task("x", "orchestrator")], "unsupported agent"),
        ([task("x", "coding"), task("x", "math")], "Duplicate subtask"),
        ([task("x", "coding", ["missing"])], "unknown dependency"),
        ([task("x", "coding", ["y"]), task("y", "math", ["x"])], "cyclic dependency"),
    ],
)
def test_invalid_plans_are_rejected_before_executor_calls(
    client: Mock, subtasks: list[dict], message: str
) -> None:
    coding = Mock()
    math = Mock()
    orchestrator = create_orchestrator(client, coding=coding, math=math)
    plan = OrchestratorPlan("request", [Subtask(**item) for item in subtasks])
    with pytest.raises(OrchestratorPlanError, match=message):
        orchestrator.validate_plan(plan)
    coding.run.assert_not_called()
    math.solve.assert_not_called()


def test_missing_injected_agent_is_rejected_by_plan_validation(client: Mock) -> None:
    orchestrator = create_orchestrator(client, coding=Mock())
    with pytest.raises(OrchestratorPlanError, match="uninjected agent 'math'"):
        orchestrator.validate_plan(OrchestratorPlan("request", [Subtask("m", "math work", "math", [])]))


def test_agent_error_skips_dependents_but_runs_independent_work(client: Mock) -> None:
    research = Mock()
    research.run_research.side_effect = RuntimeError("search provider down")
    math = Mock()
    math.solve.return_value = SimpleReport("independent math result")
    client.generate.side_effect = [
        llm_response(plan_json(
            task("r", "research"), task("dependent", "coding", ["r"]), task("m", "math")
        )),
        llm_response("Math completed; research and dependent coding did not."),
    ]
    coding = Mock()
    coding.parse_solution.return_value = CodingSolution("Python", "", "", "", [], "")
    orchestrator = create_orchestrator(client, research=research, coding=coding, math=math)

    result = orchestrator.run("Исследуй и независимо посчитай")

    status = {item.id: item.status for item in result.plan.subtasks}
    assert status == {"r": "FAILED", "dependent": "SKIPPED", "m": "COMPLETED"}
    assert result.status == "PARTIAL"
    assert "RuntimeError: search provider down" in result.plan.subtasks[0].error
    coding.run.assert_not_called()
    math.solve.assert_called_once()
    assert result.subtask_results["m"].summary == "independent math result"


def test_critic_rejection_does_not_become_success(client: Mock) -> None:
    coding = Mock()
    coding.run.return_value = AgentResult("coding", "{}", "mock/code", True, None, 1, 1)
    solution = CodingSolution("Python", "bad code", "return 0", "", [], "")
    coding.parse_solution.return_value = solution
    critic = Mock()
    rejected = CriticReport("REJECTED", 90, [], [], "Incorrect result", "coding")
    critic.review.return_value = rejected
    client.generate.side_effect = [
        llm_response(plan_json(task("c", "coding"), task("review", "critic", ["c"]))),
        llm_response("The implementation was rejected by Critic; see details."),
    ]
    orchestrator = create_orchestrator(client, coding=coding, critic=critic)

    result = orchestrator.run("Implement and review a binary search")

    assert result.status == "REJECTED"
    assert result.plan.subtasks[1].status == "FAILED"
    assert result.plan.subtasks[1].result == rejected
    assert "Critic rejected" in result.plan.subtasks[1].error
    assert "REJECTED" in result.final_answer or "REJECTED" in result.final_answer.upper()


def test_partial_successful_results_survive_later_failure(client: Mock) -> None:
    research = Mock()
    report = SimpleReport("saved research")
    research.run_research.return_value = report
    math = Mock()
    math.solve.side_effect = ValueError("malformed math task")
    client.generate.side_effect = [
        llm_response(plan_json(task("r", "research"), task("m", "math", ["r"]))),
        llm_response("Research is complete, but the math stage failed."),
    ]
    orchestrator = create_orchestrator(client, research=research, math=math)

    result = orchestrator.run("Research then calculate")

    assert result.status == "PARTIAL"
    assert result.subtask_results["r"] == report
    assert result.subtask_results["m"] is None
    assert result.plan.subtasks[1].status == "FAILED"
    assert result.errors and "ValueError" in result.errors[0]


def test_invalid_planning_json_is_reported_without_retry(client: Mock) -> None:
    client.generate.return_value = llm_response("not json")
    orchestrator = create_orchestrator(client, coding=Mock())

    result = orchestrator.run("Write a small function")

    assert result.status == "FAILED"
    assert result.plan is None
    assert "OrchestratorPlanParseError" in result.errors[0]
    assert client.generate.call_count == 1


def test_explicit_plan_parse_error_type(client: Mock) -> None:
    orchestrator = create_orchestrator(client, coding=Mock())
    with pytest.raises(OrchestratorPlanParseError):
        orchestrator._parse_plan("{broken", "request")


def test_final_answer_includes_every_subtask_result_and_status(client: Mock) -> None:
    coding = Mock()
    solution = CodingSolution("Python", "A simple addition function", "def add(a,b): return a+b", "assert add(1,2)==3", [], "pytest")
    coding.run.return_value = AgentResult("coding", "{}", "mock/code", True, None, 1, 1)
    coding.parse_solution.return_value = solution
    client.generate.side_effect = [
        llm_response(plan_json(task("code", "coding"))),
        llm_response("Функция add возвращает сумму аргументов."),
    ]
    orchestrator = create_orchestrator(client, coding=coding)

    result = orchestrator.run("Напиши функцию сложения")

    final_prompt = client.generate.call_args_list[1].kwargs["messages"][1]["content"]
    assert "A simple addition function" in final_prompt
    assert "COMPLETED" in final_prompt
    assert result.final_answer == "Функция add возвращает сумму аргументов."
