"""End-to-end adaptive orchestration tests with all model calls mocked."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.agents import AgentConfigRegistry, AgentResult, Orchestrator, OrchestratorPlanError
from src.coordination import ExecutionBudgetExceeded, TaskStore
from src.evaluation import MetricsCollector
from src.llm import LLMClient, LLMResponse
from src.llm import client as llm_client_module


def llm_response(content):
    return LLMResponse(content, "mock/orchestrator", 8, 15, 23, "stop")


def make_orchestrator(client, coding, critic):
    return Orchestrator(
        client=client,
        agents={"coding": coding, "critic": critic},
        config=AgentConfigRegistry().get_agent("orchestrator"),
    )


def test_adaptive_run_decomposes_executes_reviews_and_synthesizes_final_answer():
    client = Mock(spec=LLMClient)
    client.generate.side_effect = [
        llm_response(json.dumps({"subtasks": [{
            "id": "implement", "description": "Build a small function",
            "task_type": "CODING", "assigned_agent": "coding",
            "priority": "HIGH", "dependencies": [],
        }]})),
        llm_response("Final response grounded in reviewed output."),
    ]
    coding = Mock()
    coding.run.return_value = AgentResult("coding", "{}", "mock/coding", True, None, 12, 20)
    coding.parse_solution.return_value = {"code": "def answer(): return 42", "tests": "assert answer() == 42"}
    critic = Mock()
    critic.review.return_value = SimpleNamespace(verdict="APPROVED", summary="The code meets the request.")
    orchestrator = make_orchestrator(client, coding, critic)
    store = TaskStore()

    result = orchestrator.run_adaptive("Напиши функцию и проверь её", task_store=store)

    assert result.status == "COMPLETED"
    assert "reviewed output" in result.final_answer
    assert result.execution_metrics["review_status"] == "APPROVED"
    assert len(result.plan.subtasks) == 2
    assert all(task.status == "COMPLETED" for task in result.plan.subtasks)
    coding.run.assert_called_once()
    critic.review.assert_called_once()
    assert "def answer" in str(critic.review.call_args.kwargs["result"])
    root = store.get_task(result.plan.root_task_id)
    assert root.result["final_status"] == "COMPLETED"
    assert root.result["history"]
    assert result.execution_metrics["run_id"]
    assert result.execution_metrics["metrics_snapshot"]["run_id"] == result.execution_metrics["run_id"]
    assert result.execution_metrics["metrics_snapshot"]["task_count"] == 2


def test_adaptive_run_never_reports_success_when_critic_rejects():
    client = Mock(spec=LLMClient)
    client.generate.side_effect = [
        llm_response(json.dumps({"subtasks": [{
            "id": "implement", "description": "Build function",
            "task_type": "CODING", "assigned_agent": "coding",
            "priority": "HIGH", "dependencies": [],
        }]})),
        llm_response("A partial response."),
    ]
    coding = Mock()
    coding.run.return_value = AgentResult("coding", "{}", "mock/coding", True, None, 1, 1)
    coding.parse_solution.return_value = {"code": "incorrect"}
    critic = Mock()
    critic.review.return_value = SimpleNamespace(
        verdict="REJECTED", summary="The result is wrong.",
        issues=[{"severity": "MAJOR", "description": "The expected output is incorrect."}],
    )
    orchestrator = make_orchestrator(client, coding, critic)

    result = orchestrator.run_adaptive("Напиши функцию")

    assert result.status == "PARTIAL"
    assert result.execution_metrics["review_status"] == "REJECTED"
    assert "не завершено успешно" in result.final_answer.lower()
    critic_task = next(task for task in result.plan.subtasks if task.agent == "critic")
    assert "expected output is incorrect" in str(critic_task.result)


def test_adaptive_run_requires_critic_for_verified_success():
    client = Mock(spec=LLMClient)
    orchestrator = Orchestrator(
        client=client,
        config=AgentConfigRegistry().get_agent("orchestrator"),
    )

    with pytest.raises(OrchestratorPlanError, match="Critic Agent"):
        orchestrator.run_adaptive("Запрос")


def test_adaptive_run_can_disable_critic_for_paired_experiments():
    client = Mock(spec=LLMClient)
    client.generate.side_effect = [
        llm_response(json.dumps({"subtasks": [{
            "id": "implement", "description": "Build a small function",
            "task_type": "CODING", "assigned_agent": "coding",
            "priority": "HIGH", "dependencies": [],
        }]})),
        llm_response("Unreviewed output."),
    ]
    coding = Mock()
    coding.run.return_value = AgentResult("coding", "{}", "mock/coding", True, None, 1, 1)
    coding.parse_solution.return_value = {"code": "def value(): return 1"}
    critic = Mock()
    orchestrator = make_orchestrator(client, coding, critic)

    result = orchestrator.run_adaptive("Build a function", critic_enabled=False)

    assert result.status == "COMPLETED"
    assert result.execution_metrics["review_status"] == "NOT_REQUIRED"
    assert result.execution_metrics["critic_enabled"] is False
    assert not any(task.agent == "critic" for task in result.plan.subtasks)
    assert "def value(): return 1" in result.final_answer
    critic.review.assert_not_called()


def test_adaptive_run_collects_llm_task_agent_and_message_metrics():
    client = object.__new__(LLMClient)
    client._api_key = "test-secret"
    planned = json.dumps({"needs_input": False, "clarification_question": None, "subtasks": [{
        "id": "implement", "description": "Build a small function",
        "task_type": "CODING", "assigned_agent": "coding",
        "priority": "HIGH", "depends_on": [],
    }]})
    responses = [planned, "Final answer"]
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=Mock(side_effect=lambda **_: SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=responses.pop(0)), finish_reason="stop")],
            model="nvidia/nemotron-3-ultra-550b-a55b:free",
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )))))
    client._logger = Mock()
    client._request_guard = None
    client._response_observer = None
    coding = Mock()
    coding.run.return_value = AgentResult("coding", "{}", "mock/coding", True, None, 3, 4)
    coding.parse_solution.return_value = {"code": "def answer(): return 42"}
    orchestrator = make_orchestrator(client, coding, Mock())
    metrics = MetricsCollector()

    result = orchestrator.run_adaptive("Build a function", critic_enabled=False,
        metrics_collector=metrics, metrics_output_dir=None)

    assert result.status == "COMPLETED"
    snapshot = result.execution_metrics["metrics_snapshot"]
    assert snapshot["run_id"] == metrics.run_id
    assert snapshot["api_calls"] == 1
    assert snapshot["prompt_tokens"] == 10
    assert snapshot["completion_tokens"] == 5
    assert snapshot["task_count"] == 1
    assert any(row["agent_role"] == "coding" for row in snapshot["agents"])
    assert snapshot["message_count"] > 0
    assert all(call["run_id"] == metrics.run_id for call in snapshot["llm_calls"])


def test_resource_budget_failure_during_planning_is_returned_as_failed_result():
    client = Mock(spec=LLMClient)
    client.generate.side_effect = ExecutionBudgetExceeded(
        "LLM request blocked: call budget is exhausted.",
        details={"limit": "calls", "max_calls": 0},
    )
    coding = Mock()
    critic = Mock()
    orchestrator = make_orchestrator(client, coding, critic)

    result = orchestrator.run_adaptive("Build a function", max_calls=0)

    assert result.status == "FAILED"
    assert result.execution_history[0]["event"] == "resource_limit_exceeded"
    assert result.execution_history[0]["budget_details"]["max_calls"] == 0
    assert result.plan is None
    coding.run.assert_not_called()
    critic.review.assert_not_called()


def test_adaptive_run_returns_structured_clarification_without_executing_tasks():
    client = Mock(spec=LLMClient)
    client.generate.return_value = llm_response(json.dumps({
        "needs_input": True,
        "clarification_question": "Какую версию Python нужно поддержать?",
        "subtasks": [],
    }))
    coding = Mock()
    critic = Mock()
    orchestrator = make_orchestrator(client, coding, critic)
    store = TaskStore()

    result = orchestrator.run_adaptive("Сделай приложение", task_store=store)

    assert result.status == "NEEDS_INPUT"
    assert "Какую версию Python" in result.final_answer
    assert result.plan.needs_input is True
    assert result.execution_history[0]["failure_category"] == "insufficient_context"
    root = store.get_task(result.plan.root_task_id)
    assert root.status.value == "NEEDS_INPUT"
    assert root.result["clarification_question"] == "Какую версию Python нужно поддержать?"
    coding.run.assert_not_called()
    critic.review.assert_not_called()


@pytest.mark.parametrize(("max_calls", "expected_status", "expected_calls"), [
    (2, "COMPLETED", 2),
    (1, "COMPLETED", 1),
])
def test_adaptive_synthesis_respects_budget_and_falls_back_to_saved_result(
    monkeypatch, max_calls, expected_status, expected_calls
):
    monkeypatch.setenv("OPENROUTER_API_KEY", "unit-test-key")
    monkeypatch.setattr("dotenv.load_dotenv", lambda: None)
    sdk = Mock()
    planner_json = json.dumps({"subtasks": [{
        "id": "implement", "description": "Build a small function",
        "task_type": "CODING", "assigned_agent": "coding",
        "priority": "HIGH", "dependencies": [],
    }]})
    sdk.chat.completions.create.side_effect = [
        SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=planner_json), finish_reason="stop")],
            model="planner", usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        ),
        SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="reviewed final"), finish_reason="stop")],
            model="planner", usage=SimpleNamespace(prompt_tokens=30, completion_tokens=40, total_tokens=70),
        ),
    ]
    monkeypatch.setattr(llm_client_module, "OpenAI", Mock(return_value=sdk))
    monkeypatch.setattr(llm_client_module, "Logger", Mock(return_value=Mock()))
    client = LLMClient()
    coding = Mock()
    coding.run.return_value = AgentResult("coding", "{}", "mock/coding", True, None, 1, 1)
    coding.parse_solution.return_value = {"code": "return 42"}
    critic = Mock()
    critic.review.return_value = SimpleNamespace(verdict="APPROVED", summary="ok")
    orchestrator = make_orchestrator(client, coding, critic)
    store = TaskStore()

    result = orchestrator.run_adaptive("Напиши функцию", task_store=store, max_calls=max_calls)

    assert result.status == expected_status
    assert result.execution_metrics["calls"] == expected_calls
    assert result.execution_metrics["prompt_tokens"] == (40 if expected_calls == 2 else 10)
    assert client._request_guard is None
    assert store.get_task(result.plan.root_task_id).status.value == (
        "COMPLETED" if expected_status == "COMPLETED" else "FAILED"
    )
    if max_calls == 1:
        assert "Дополнительный LLM-синтез пропущен" in result.final_answer
        assert "return 42" in result.final_answer


def test_final_synthesis_payload_is_bounded_and_omits_internal_history():
    from src.agents.orchestrator import MAX_FINAL_SYNTHESIS_BYTES

    payload = {
        "original_request": "Write a small implementation",
        "execution_status": "COMPLETED",
        "review_status": "APPROVED",
        "history": [{"debug": "internal history" * 100_000}],
        "tasks": [{
            "task_id": "coding-1", "description": "write code", "task_type": "CODING",
            "assigned_agent": "coding", "status": "COMPLETED", "error": None,
            "result": {"code": "x" * 100_000, "tests": "y" * 100_000},
        }],
    }

    bounded = Orchestrator._bounded_synthesis_payload(payload)
    serialized = json.dumps(bounded, ensure_ascii=False, separators=(",", ":"))

    assert len(serialized.encode("utf-8")) <= MAX_FINAL_SYNTHESIS_BYTES
    assert "history" not in bounded
    assert "internal history" not in serialized
    assert "обрезано для экономии контекста" in serialized


def test_stable_algorithm_request_does_not_run_unrequested_research():
    client = Mock(spec=LLMClient)
    client.generate.side_effect = [
        llm_response(json.dumps({"subtasks": [
            {"id": "research", "description": "Explain binary search background",
             "task_type": "RESEARCH", "assigned_agent": "research", "priority": "MEDIUM", "dependencies": []},
            {"id": "code", "description": "Implement binary search with tests",
             "task_type": "CODING", "assigned_agent": "coding", "priority": "HIGH", "dependencies": []},
        ]})),
        llm_response("Final answer from coding results."),
    ]
    research = Mock()
    research.run_research.return_value = "unneeded web research"
    coding = Mock()
    coding.run.return_value = AgentResult("coding", "{}", "mock/coding", True, None, 1, 1)
    coding.parse_solution.return_value = {"code": "def binary_search(): pass", "tests": "assert True"}
    critic = Mock()
    critic.review.return_value = SimpleNamespace(verdict="APPROVED", summary="Tests look reasonable.")
    orchestrator = Orchestrator(
        client=client,
        agents={"research": research, "coding": coding, "critic": critic},
        config=AgentConfigRegistry().get_agent("orchestrator"),
    )

    result = orchestrator.run_adaptive(
        "Объясни бинарный поиск, напиши реализацию на Python и тесты для граничных случаев."
    )

    assert result.status == "COMPLETED"
    assert all(task.agent != "research" for task in result.plan.subtasks)
    research.run_research.assert_not_called()
    coding.run.assert_called_once()

