"""Concurrent task execution tests using mocked agents and local metadata."""

import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import yaml
import pytest

from src.agents import AgentResult, AgentConfigRegistry, BaseAgent
from src.communication import MessageBus, MessageType
from src.coordination import ExecutionBudgetExceeded, Task, TaskExecutor, TaskPriority, TaskStatus, TaskStore, TaskType
from src.llm import LLMClient, LLMResponse
from src.llm import client as llm_client_module
from src.llm.model_registry import ModelRegistry
from src.routing import AdaptiveRouter


def make_store(*specs):
    store = TaskStore()
    records = [Task("root", "original request")]
    for task_id, task_type, agent, dependencies in specs:
        records.append(Task(
            task_id, f"work {task_id}", task_type=task_type,
            priority=TaskPriority.MEDIUM, status=TaskStatus.ASSIGNED,
            parent_task_id="root", assignee=agent, dependencies=dependencies,
        ))
    store.create_tasks(records)
    return store


def agent_result(agent_id="orchestrator", content="done"):
    return AgentResult(agent_id, content, "model-from-mock", True, None, 4, 6)


def test_independent_tasks_execute_in_parallel_and_history_is_persisted():
    store = make_store(
        ("one", TaskType.GENERAL, "orchestrator", []),
        ("two", TaskType.GENERAL, "orchestrator", []),
    )
    barrier = threading.Barrier(2)
    orchestrator_agent = Mock()
    def synchronized_run(_prompt):
        barrier.wait(timeout=2)
        return agent_result()
    orchestrator_agent.run.side_effect = synchronized_run
    bus = MessageBus()
    executor = TaskExecutor(
        store,
        AdaptiveRouter(available_agents={"orchestrator"}),
        {"orchestrator": orchestrator_agent},
        message_bus=bus,
        max_workers=2,
        max_retries=0,
    )

    report = executor.execute("root")

    assert report.status == "COMPLETED"
    assert report.calls == 2
    assert report.prompt_tokens == 8
    assert report.completion_tokens == 12
    assert all(store.get_task(task_id).status is TaskStatus.COMPLETED for task_id in ("one", "two"))
    assert store.get_task("root").result["history"]
    assert sum(1 for event in report.history if event.event == "completed") == 2
    assert bus.receive("orchestrator").message_type is MessageType.TASK_ASSIGN


def test_dependencies_run_in_order_and_results_are_passed_forward():
    store = make_store(
        ("first", TaskType.CODING, "coding", []),
        ("second", TaskType.GENERAL, "orchestrator", ["first"]),
    )
    coding = Mock()
    coding.run.return_value = agent_result("coding", "solution payload")
    coding.parse_solution.return_value = {"code": "solution payload"}
    orchestrator_agent = Mock()
    orchestrator_agent.run.return_value = agent_result(content="summarized")
    executor = TaskExecutor(
        store,
        AdaptiveRouter(available_agents={"coding", "orchestrator"}),
        {"coding": coding, "orchestrator": orchestrator_agent},
        max_workers=2,
        max_retries=0,
    )

    report = executor.execute("root")

    assert report.status == "COMPLETED"
    assert report.task_statuses == {"first": "COMPLETED", "second": "COMPLETED"}
    prompt = orchestrator_agent.run.call_args.args[0]
    assert "solution payload" in prompt
    assert report.history.index(next(event for event in report.history if event.task_id == "first")) < report.history.index(
        next(event for event in report.history if event.task_id == "second")
    )


def test_failed_dependency_is_skipped_and_execution_is_not_reported_successful():
    store = make_store(
        ("broken", TaskType.GENERAL, "orchestrator", []),
        ("blocked", TaskType.GENERAL, "orchestrator", ["broken"]),
    )
    agent = Mock()
    agent.run.side_effect = RuntimeError("agent unavailable")
    report = TaskExecutor(
        store,
        AdaptiveRouter(available_agents={"orchestrator"}),
        {"orchestrator": agent},
        max_retries=0,
    ).execute("root")

    assert report.status == "FAILED"
    assert report.task_statuses == {"broken": "FAILED", "blocked": "SKIPPED"}
    assert store.get_task("blocked").status is TaskStatus.SKIPPED


def test_needs_input_is_persisted_and_blocks_dependent_tasks():
    store = make_store(
        ("clarify", TaskType.GENERAL, "orchestrator", []),
        ("dependent", TaskType.CODING, "coding", ["clarify"]),
    )
    agent = Mock()
    agent.run.return_value = AgentResult(
        "orchestrator", "", "mock/model", True, None, 4, 2,
        task_completed=False,
        needs_input=True,
        clarification_question="Какой формат вывода требуется?",
    )
    coding = Mock()
    report = TaskExecutor(
        store,
        AdaptiveRouter(available_agents={"orchestrator", "coding"}),
        {"orchestrator": agent, "coding": coding},
        max_retries=1,
    ).execute("root")

    assert report.status == "NEEDS_INPUT"
    assert report.task_statuses == {"clarify": "NEEDS_INPUT", "dependent": "SKIPPED"}
    assert store.get_task("clarify").result["clarification_question"] == "Какой формат вывода требуется?"
    assert "clarify" in store.get_task("dependent").error
    assert [event.event for event in report.history if event.task_id == "clarify"] == ["needs_input"]
    skipped = next(event for event in report.history if event.task_id == "dependent")
    assert skipped.failure_category == "dependency_not_completed"
    assert coding.run.call_count == 0


def test_coding_agent_structured_clarification_reaches_task_status(monkeypatch):
    import json
    from src.agents import CodingAgent

    monkeypatch.setenv("OPENROUTER_API_KEY", "unit-test-key")
    monkeypatch.setattr("dotenv.load_dotenv", lambda: None)
    sdk = Mock()
    clarification = json.dumps({
        "language": "python", "summary": "Need one implementation detail.",
        "code": "", "tests": "", "dependencies": [], "run_instructions": "",
        "task_completed": False, "needs_input": True,
        "clarification_question": "Should the parser accept CSV or JSON?",
    })
    sdk.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=clarification), finish_reason="stop")],
        model="mock/coding", usage=SimpleNamespace(prompt_tokens=10, completion_tokens=12, total_tokens=22),
    )
    monkeypatch.setattr(llm_client_module, "OpenAI", Mock(return_value=sdk))
    monkeypatch.setattr(llm_client_module, "Logger", Mock(return_value=Mock()))
    coding = CodingAgent(client=LLMClient())
    store = make_store(("implement", TaskType.CODING, "coding", []))

    report = TaskExecutor(
        store, AdaptiveRouter(available_agents={"coding"}), {"coding": coding}, max_retries=1,
    ).execute("root")

    assert report.status == "NEEDS_INPUT"
    assert store.get_task("implement").status is TaskStatus.NEEDS_INPUT
    assert report.history[0].clarification_question == "Should the parser accept CSV or JSON?"
    assert sdk.chat.completions.create.call_count == 1


def test_task_completed_false_without_clarification_is_failed_not_completed():
    store = make_store(("incomplete", TaskType.GENERAL, "orchestrator", []))
    agent = Mock()
    agent.run.return_value = AgentResult(
        "orchestrator", "Some partial output", "mock/model", True, None, 1, 1,
        task_completed=False,
    )
    report = TaskExecutor(
        store, AdaptiveRouter(available_agents={"orchestrator"}),
        {"orchestrator": agent}, max_retries=1,
    ).execute("root")

    assert report.status == "FAILED"
    assert store.get_task("incomplete").status is TaskStatus.FAILED
    event = next(event for event in report.history if event.task_id == "incomplete")
    assert event.event == "result_incomplete"
    assert event.failure_category == "incomplete_result"
    assert event.retry_allowed is False


def test_incomplete_successful_coding_result_is_not_fallback_or_completed():
    from src.agents import CodingSolutionValidationError

    store = make_store(("implement", TaskType.CODING, "coding", []))
    coding = Mock()
    coding.run.return_value = agent_result("coding", "valid JSON")
    coding.parse_solution.side_effect = CodingSolutionValidationError("placeholder code")
    fallback = Mock()
    report = TaskExecutor(
        store,
        AdaptiveRouter(available_agents={"coding", "orchestrator"}),
        {"coding": coding, "orchestrator": fallback},
        max_retries=2,
    ).execute("root")

    assert report.status == "FAILED"
    assert store.get_task("implement").status is TaskStatus.FAILED
    assert fallback.run.call_count == 0
    failure = next(event for event in report.history if event.task_id == "implement")
    assert failure.event == "format_repair_failed"
    assert failure.failure_category == "format"
    assert failure.retry_allowed is False


def test_research_format_repair_failure_does_not_fallback_to_orchestrator():
    from src.agents import ResearchAgent
    from src.tools import SearchTool

    client = Mock(spec=LLMClient)
    client.generate.side_effect = [
        LLMResponse("not json", "mock/research", 2, 1, 3, "stop"),
        LLMResponse("still not json", "mock/research", 3, 1, 4, "stop"),
    ]
    search_tool = Mock(spec=SearchTool)
    search_tool.enabled = True
    research = ResearchAgent(
        client=client, search_tool=search_tool,
        config=AgentConfigRegistry().get_agent("research"),
    )
    fallback = Mock()
    fallback.run.return_value = agent_result(content="fallback must not run")
    store = make_store(("research", TaskType.RESEARCH, "research", []))

    report = TaskExecutor(
        store,
        AdaptiveRouter(available_agents={"research", "orchestrator"}),
        {"research": research, "orchestrator": fallback},
        max_retries=2,
    ).execute("root")

    assert report.status == "FAILED"
    assert client.generate.call_count == 2
    assert fallback.run.call_count == 0
    event = next(event for event in report.history if event.task_id == "research")
    assert event.event == "format_repair_failed"
    assert event.format_repair_attempts == 1
    assert event.retry_allowed is False
    assert "ResearchPlanParseError" in event.error


def test_agent_failure_retries_with_router_fallback_and_tracks_history():
    store = make_store(("research", TaskType.RESEARCH, "research", []))
    research = Mock()
    research.run_research.side_effect = RuntimeError("search temporarily failed")
    fallback = Mock()
    fallback.run.return_value = agent_result()
    report = TaskExecutor(
        store,
        AdaptiveRouter(available_agents={"research", "orchestrator"}),
        {"research": research, "orchestrator": fallback},
        max_retries=1,
    ).execute("root")

    assert report.status == "COMPLETED"
    assert report.calls == 2
    assert [event.event for event in report.history if event.task_id == "research"] == ["attempt_failed", "completed"]
    assert store.get_task("research").assignee == "orchestrator"
    fallback_event = next(event for event in report.history if event.task_id == "research" and event.event == "completed")
    assert fallback_event.fallback is True
    assert "Preferred executor" in fallback_event.route_reason


def test_call_budget_is_not_bypassed_by_router_fallback():
    store = make_store(("research", TaskType.RESEARCH, "research", []))
    research = Mock()
    research.run_research.side_effect = RuntimeError("temporary failure")
    fallback = Mock()
    report = TaskExecutor(
        store,
        AdaptiveRouter(available_agents={"research", "orchestrator"}),
        {"research": research, "orchestrator": fallback},
        max_calls=1,
        max_retries=2,
    ).execute("root")

    assert report.calls == 1
    assert research.run_research.call_count == 1
    fallback.run.assert_not_called()
    assert any(event.event == "budget_rejected" for event in report.history)


def test_real_baseagent_fallback_does_not_reenter_orchestrator_run_pipeline():
    store = make_store(("research", TaskType.RESEARCH, "research", []))
    client = Mock(spec=LLMClient)
    client.generate.return_value = LLMResponse(
        '{"task_completed":true,"needs_input":false,"clarification_question":null,"result":"direct fallback"}',
        "reasoning-model", 3, 4, 7, "stop",
    )
    fallback = BaseAgent(
        "orchestrator", "Orchestrator executor", "Coordinator prompt", client,
        AgentConfigRegistry().get_agent("orchestrator"),
    )
    report = TaskExecutor(
        store,
        AdaptiveRouter(available_agents={"orchestrator"}),
        {"orchestrator": fallback},
        max_retries=0,
    ).execute("root")

    assert report.status == "COMPLETED"
    assert store.get_task("research").assignee == "orchestrator"
    assert client.generate.call_args.kwargs["messages"][0]["content"].startswith("Ты универсальный")
    assert "task_completed" in client.generate.call_args.kwargs["messages"][0]["content"]


def test_generic_baseagent_prose_cannot_be_marked_completed_without_status_envelope():
    store = make_store(("research", TaskType.RESEARCH, "research", []))
    client = Mock(spec=LLMClient)
    client.generate.return_value = LLMResponse("Please provide more context.", "model", 2, 3, 5, "stop")
    fallback = BaseAgent(
        "orchestrator", "Orchestrator executor", "Coordinator prompt", client,
        AgentConfigRegistry().get_agent("orchestrator"),
    )
    report = TaskExecutor(
        store, AdaptiveRouter(available_agents={"orchestrator"}),
        {"orchestrator": fallback}, max_retries=0,
    ).execute("root")

    assert report.status == "FAILED"
    assert store.get_task("research").status is TaskStatus.FAILED
    event = next(event for event in report.history if event.task_id == "research")
    assert event.event == "format_repair_failed"
    assert event.failure_category == "format"
    assert "JSON status envelope" in event.error


def test_generic_baseagent_repairs_invalid_status_envelope_once_within_retry_budget():
    store = make_store(("research", TaskType.RESEARCH, "research", []))
    client = Mock(spec=LLMClient)
    client.generate.side_effect = [
        LLMResponse("Фотоэлементы преобразуют свет в электричество.", "model", 10, 12, 22, "stop"),
        LLMResponse(
            '{"task_completed":true,"needs_input":false,'
            '"clarification_question":null,"result":"Фотоэлементы преобразуют свет в электричество."}',
            "model", 20, 15, 35, "stop",
        ),
    ]
    fallback = BaseAgent(
        "orchestrator", "Orchestrator executor", "Coordinator prompt", client,
        AgentConfigRegistry().get_agent("orchestrator"),
    )

    report = TaskExecutor(
        store,
        AdaptiveRouter(available_agents={"orchestrator"}),
        {"orchestrator": fallback},
        max_retries=1,
    ).execute("root")

    assert report.status == "COMPLETED"
    assert store.get_task("research").result == "Фотоэлементы преобразуют свет в электричество."
    assert client.generate.call_count == 2
    assert report.retries == 1
    completed = next(event for event in report.history if event.task_id == "research")
    assert completed.format_repair_attempts == 1
    assert completed.prompt_tokens == 30
    assert completed.completion_tokens == 27


def test_task_executor_default_timeout_allows_slow_provider_responses():
    executor = TaskExecutor(
        make_store(("research", TaskType.RESEARCH, "research", [])),
        AdaptiveRouter(available_agents={"research"}),
        {"research": Mock()},
    )

    assert executor.task_timeout == 180.0


def test_executor_routes_around_non_injected_specialist_without_spending_retry():
    store = make_store(("research", TaskType.RESEARCH, "research", []))
    fallback = Mock()
    fallback.run.return_value = agent_result()
    report = TaskExecutor(
        store,
        AdaptiveRouter(),  # The registry knows the role, but this runtime has no Research Agent.
        {"orchestrator": fallback},
        max_retries=0,
    ).execute("root")

    assert report.status == "COMPLETED"
    assert report.calls == 1
    assert store.get_task("research").assignee == "orchestrator"


def test_critic_rejection_marks_review_failed_and_root_not_successful():
    store = make_store(
        ("result", TaskType.GENERAL, "orchestrator", []),
        ("review", TaskType.REVIEW, "critic", ["result"]),
    )
    orchestrator_agent = Mock()
    orchestrator_agent.run.return_value = agent_result()
    critic = Mock()
    critic.review.return_value = SimpleNamespace(verdict="REJECTED", summary="A problem was found")
    report = TaskExecutor(
        store,
        AdaptiveRouter(available_agents={"orchestrator", "critic"}),
        {"orchestrator": orchestrator_agent, "critic": critic},
        max_retries=0,
    ).execute("root")

    assert report.status == "PARTIAL"
    assert report.review_status == "REJECTED"
    assert store.get_task("review").status is TaskStatus.FAILED


def test_critic_feedback_revises_original_executor_and_rechecks_revised_result():
    store = make_store(
        ("implementation", TaskType.CODING, "coding", []),
        ("review", TaskType.REVIEW, "critic", ["implementation"]),
    )
    coding = Mock()
    coding.run.side_effect = [agent_result("coding", "initial"), agent_result("coding", "revised")]
    coding.parse_solution.side_effect = [{"code": "initial"}, {"code": "fixed"}]
    critic = Mock()
    critic.review.side_effect = [
        SimpleNamespace(
            verdict="REJECTED", summary="Return the correct value.",
            issues=[{"category": "CORRECTNESS", "severity": "MAJOR",
                     "description": "Wrong result", "location": "code",
                     "recommendation": "Fix the return value."}],
        ),
        SimpleNamespace(verdict="APPROVED", summary="The revised result is correct.", issues=[]),
    ]

    report = TaskExecutor(
        store,
        AdaptiveRouter(available_agents={"coding", "critic"}),
        {"coding": coding, "critic": critic},
        max_critic_cycles=1,
        max_retries=0,
    ).execute("root")

    assert report.status == "COMPLETED"
    assert report.review_status == "APPROVED"
    assert coding.run.call_count == 2
    assert critic.review.call_count == 2
    assert store.get_task("implementation").result == {"code": "fixed"}
    review_result = store.get_task("review").result
    assert review_result.cycle_count == 1
    assert review_result.iterations[0]["review"]["verdict"] == "REJECTED"
    assert review_result.iterations[1]["revisions"][0]["previous_result"] == {"code": "initial"}
    assert any(event.event == "critic_refinement" for event in report.history)


def test_critic_feedback_targets_the_named_original_executor_only():
    store = make_store(
        ("research_notes", TaskType.RESEARCH, "research", []),
        ("implementation", TaskType.CODING, "coding", []),
        ("review", TaskType.REVIEW, "critic", ["research_notes", "implementation"]),
    )
    research = Mock()
    research.run_research.return_value = SimpleNamespace(summary="Research is complete.")
    coding = Mock()
    coding.run.side_effect = [agent_result("coding", "first"), agent_result("coding", "second")]
    coding.parse_solution.side_effect = [{"code": "first"}, {"code": "fixed"}]
    critic = Mock()
    critic.review.side_effect = [
        SimpleNamespace(verdict="REJECTED", summary="Fix the implementation.", issues=[{
            "category": "CORRECTNESS", "severity": "MAJOR", "description": "Wrong code",
            "location": "code", "recommendation": "Correct the implementation.",
            "target_task_id": "implementation",
        }]),
        SimpleNamespace(verdict="APPROVED", summary="The fix is correct.", issues=[]),
    ]

    report = TaskExecutor(
        store,
        AdaptiveRouter(available_agents={"research", "coding", "critic"}),
        {"research": research, "coding": coding, "critic": critic},
        max_workers=1,
        max_critic_cycles=1,
        max_retries=0,
    ).execute("root")

    assert report.status == "COMPLETED"
    assert research.run_research.call_count == 1
    assert coding.run.call_count == 2
    reviewed_results = critic.review.call_args_list[1].kwargs["result"]
    assert any(item == {"code": "fixed"} for item in reviewed_results)


def test_critic_feedback_cycle_limit_preserves_rejection_without_false_success():
    store = make_store(
        ("implementation", TaskType.CODING, "coding", []),
        ("review", TaskType.REVIEW, "critic", ["implementation"]),
    )
    coding = Mock()
    coding.run.return_value = agent_result("coding", "initial")
    coding.parse_solution.return_value = {"code": "initial"}
    critic = Mock()
    critic.review.return_value = SimpleNamespace(
        verdict="REJECTED", summary="A concrete correction is required.",
        issues=[{"category": "CORRECTNESS", "severity": "MAJOR",
                 "description": "Wrong result", "location": "code",
                 "recommendation": "Fix the return value."}],
    )

    report = TaskExecutor(
        store,
        AdaptiveRouter(available_agents={"coding", "critic"}),
        {"coding": coding, "critic": critic},
        max_critic_cycles=0,
        max_retries=0,
    ).execute("root")

    assert report.status == "PARTIAL"
    assert report.review_status == "REJECTED"
    assert coding.run.call_count == 1
    assert critic.review.call_count == 1
    assert store.get_task("review").status is TaskStatus.FAILED


def test_critic_can_request_input_instead_of_rejecting_or_approving():
    store = make_store(
        ("implementation", TaskType.CODING, "coding", []),
        ("review", TaskType.REVIEW, "critic", ["implementation"]),
    )
    coding = Mock()
    coding.run.return_value = agent_result("coding", "initial")
    coding.parse_solution.return_value = {"code": "initial"}
    critic = Mock()
    critic.review.return_value = SimpleNamespace(
        verdict="NEEDS_INPUT", summary="Нужно уточнить ожидаемую версию Python.",
        clarification_question="Какую версию Python поддерживать?", issues=[],
    )

    report = TaskExecutor(
        store, AdaptiveRouter(available_agents={"coding", "critic"}),
        {"coding": coding, "critic": critic}, max_retries=0,
    ).execute("root")

    assert report.status == "NEEDS_INPUT"
    assert report.review_status == "NEEDS_INPUT"
    assert store.get_task("review").status is TaskStatus.NEEDS_INPUT
    assert store.get_task("review").result["clarification_question"] == "Какую версию Python поддерживать?"


def test_global_retry_limit_bounds_fallback_attempts():
    store = make_store(("work", TaskType.GENERAL, "orchestrator", []))
    primary = Mock()
    primary.run.side_effect = RuntimeError("primary failed")
    fallback = Mock()
    fallback.run.side_effect = RuntimeError("fallback failed")
    report = TaskExecutor(
        store,
        AdaptiveRouter(available_agents={"orchestrator"}),
        {"orchestrator": primary},
        max_retries=3,
        max_total_retries=0,
    ).execute("root")
    assert report.status == "FAILED"
    assert report.retries == 0
    assert primary.run.call_count == 1
    assert not any(event.retry_allowed for event in report.history if event.event == "attempt_failed")


def test_authentication_failure_is_not_retried_or_routed_to_fallback():
    store = make_store(("work", TaskType.CODING, "coding", []))
    coding = Mock()
    coding.run.return_value = AgentResult(
        "coding", "", "cohere/north-mini-code:free", False,
        "AuthenticationError: invalid key", None, None,
        error_type="AuthenticationError",
    )
    fallback = Mock()
    report = TaskExecutor(
        store,
        AdaptiveRouter(available_agents={"coding", "orchestrator"}),
        {"coding": coding, "orchestrator": fallback},
        max_retries=2,
    ).execute("root")

    assert report.status == "FAILED"
    assert coding.run.call_count == 1
    fallback.run.assert_not_called()
    failure = next(event for event in report.history if event.event == "attempt_failed")
    assert failure.failure_category == "authentication"
    assert failure.retry_allowed is False


def test_rate_limit_fallback_requires_a_different_model():
    store = make_store(("research", TaskType.RESEARCH, "research", []))
    research = Mock()
    research.run_research.side_effect = RuntimeError("RateLimitError: free model limit")
    same_model_fallback = Mock()
    report = TaskExecutor(
        store,
        AdaptiveRouter(available_agents={"research", "orchestrator"}),
        {"research": research, "orchestrator": same_model_fallback},
        max_retries=2,
    ).execute("root")

    assert report.status == "FAILED"
    same_model_fallback.run.assert_not_called()
    failure = next(event for event in report.history if event.event == "attempt_failed")
    assert failure.failure_category == "rate_limit"
    assert failure.retry_allowed is False


def test_rate_limit_can_fallback_to_a_different_configured_model():
    store = make_store(("implementation", TaskType.CODING, "coding", []))
    coding = Mock()
    coding.run.return_value = AgentResult(
        "coding", "", "cohere/north-mini-code:free", False,
        "RateLimitError: model limited", None, None,
        error_type="RateLimitError",
    )
    fallback = Mock()
    fallback.run.return_value = agent_result()
    report = TaskExecutor(
        store,
        AdaptiveRouter(available_agents={"coding", "orchestrator"}),
        {"coding": coding, "orchestrator": fallback},
        max_retries=1,
    ).execute("root")

    assert report.status == "COMPLETED"
    assert coding.run.call_count == 1
    fallback.run.assert_called_once()
    fallback_event = next(event for event in report.history if event.event == "completed")
    assert fallback_event.fallback is True
    assert fallback_event.model_id == "nvidia/nemotron-3-ultra-550b-a55b:free"


def test_parallel_tasks_cannot_race_past_shared_api_call_limit():
    store = make_store(
        ("one", TaskType.GENERAL, "orchestrator", []),
        ("two", TaskType.GENERAL, "orchestrator", []),
    )
    agent = Mock()
    def run(_prompt):
        time.sleep(0.02)
        return agent_result()
    agent.run.side_effect = run
    report = TaskExecutor(
        store,
        AdaptiveRouter(available_agents={"orchestrator"}),
        {"orchestrator": agent},
        max_workers=2,
        max_calls=1,
        max_retries=0,
    ).execute("root")

    assert report.calls == 1
    assert agent.run.call_count == 1
    assert sum(event.event == "completed" for event in report.history) == 1
    assert sum(event.event == "budget_rejected" for event in report.history) == 1


def test_max_execution_time_cancels_late_result_without_overwriting_status():
    store = make_store(("slow", TaskType.GENERAL, "orchestrator", []))
    agent = Mock()
    def slow_run(_prompt):
        time.sleep(0.06)
        return agent_result()
    agent.run.side_effect = slow_run
    report = TaskExecutor(
        store,
        AdaptiveRouter(available_agents={"orchestrator"}),
        {"orchestrator": agent},
        task_timeout=1.0,
        max_execution_time=0.01,
        max_retries=0,
    ).execute("root")
    assert report.status == "FAILED"
    assert store.get_task("slow").status is TaskStatus.CANCELLED
    assert any(event.event == "execution_deadline" for event in report.history)
    time.sleep(0.08)
    assert store.get_task("slow").status is TaskStatus.CANCELLED
    assert store.get_task("slow").result is None


def test_critic_parse_failure_is_terminal_and_never_falls_back():
    from src.agents import CriticReportParseError

    store = make_store(
        ("output", TaskType.GENERAL, "orchestrator", []),
        ("review", TaskType.REVIEW, "critic", ["output"]),
    )
    orchestrator = Mock()
    orchestrator.run.return_value = agent_result()
    critic = Mock()
    critic.review.side_effect = CriticReportParseError("Issue severity is unsupported.")
    report = TaskExecutor(
        store,
        AdaptiveRouter(available_agents={"orchestrator", "critic"}),
        {"orchestrator": orchestrator, "critic": critic},
        max_retries=2,
    ).execute("root")

    assert report.status == "PARTIAL"
    assert report.review_status == "FAILED"
    assert critic.review.call_count == 1
    event = next(event for event in report.history if event.task_id == "review")
    assert event.event == "format_repair_failed"
    assert event.retry_allowed is False
    assert event.review_performed is False
    assert not any(event.event == "no_route" for event in report.history)


def test_transient_critic_failure_retries_same_critic_without_router_exclusion():
    from src.agents import CriticLLMError

    store = make_store(
        ("output", TaskType.GENERAL, "orchestrator", []),
        ("review", TaskType.REVIEW, "critic", ["output"]),
    )
    orchestrator = Mock()
    orchestrator.run.return_value = agent_result()
    critic = Mock()
    critic.review.side_effect = [
        CriticLLMError("temporary provider failure"),
        SimpleNamespace(verdict="APPROVED", summary="reviewed"),
    ]
    report = TaskExecutor(
        store,
        AdaptiveRouter(available_agents={"orchestrator", "critic"}),
        {"orchestrator": orchestrator, "critic": critic},
        max_retries=1,
    ).execute("root")

    assert report.status == "COMPLETED"
    assert report.review_status == "APPROVED"
    assert critic.review.call_count == 2
    assert orchestrator.run.call_count == 1
    first = next(event for event in report.history if event.task_id == "review")
    assert first.retry_allowed is True
    assert "same Critic" in first.retry_reason


@pytest.mark.parametrize(("max_calls", "expected_calls", "expected_status"), [
    (2, 2, "COMPLETED"),
    (1, 1, "PARTIAL"),
])
def test_critic_format_repair_calls_and_usage_are_budgeted(
    monkeypatch, max_calls, expected_calls, expected_status,
):
    import json
    from src.agents import CriticAgent

    monkeypatch.setenv("OPENROUTER_API_KEY", "unit-test-key")
    monkeypatch.setattr("dotenv.load_dotenv", lambda: None)
    valid_report = json.dumps({
        "verdict": "APPROVED", "score": 90, "issues": [],
        "strengths": ["Evidence reviewed."], "summary": "Looks good.",
        "criteria": {
            "correctness": "ok", "task_compliance": "ok", "completeness": "ok",
            "logical_consistency": "ok", "evidence_and_verification": "ok",
        },
        "verification_limits": [],
    })
    sdk = Mock()
    sdk.chat.completions.create.side_effect = [
        SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="not json"), finish_reason="stop")],
            model="mock/critic", usage=SimpleNamespace(prompt_tokens=3, completion_tokens=5, total_tokens=8),
        ),
        SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=valid_report), finish_reason="stop")],
            model="mock/critic", usage=SimpleNamespace(prompt_tokens=7, completion_tokens=11, total_tokens=18),
        ),
    ]
    monkeypatch.setattr(llm_client_module, "OpenAI", Mock(return_value=sdk))
    monkeypatch.setattr(llm_client_module, "Logger", Mock(return_value=Mock()))
    llm_client = LLMClient()
    critic = CriticAgent(client=llm_client)
    orchestrator = Mock()
    orchestrator.run.return_value = agent_result(content="result")
    store = make_store(
        ("output", TaskType.GENERAL, "orchestrator", []),
        ("review", TaskType.REVIEW, "critic", ["output"]),
    )

    report = TaskExecutor(
        store,
        AdaptiveRouter(available_agents={"orchestrator", "critic"}),
        {"orchestrator": orchestrator, "critic": critic},
        max_calls=max_calls,
        max_retries=1,
    ).execute("root")

    assert report.calls == expected_calls
    assert report.retries == 1
    assert report.status == expected_status
    assert report.prompt_tokens == (10 if expected_calls == 2 else 3)
    assert report.completion_tokens == (16 if expected_calls == 2 else 5)
    assert sdk.chat.completions.create.call_count == expected_calls
    if expected_calls == 2:
        completed = next(event for event in report.history if event.task_id == "review" and event.event == "completed")
        assert completed.format_repair_attempts == 1
        assert completed.review_performed is True
    else:
        rejected = next(event for event in report.history if event.task_id == "review")
        assert rejected.event == "budget_rejected"
        assert rejected.retry_allowed is False


def test_budget_stops_calls_before_agent_invocation():
    store = make_store(("one", TaskType.GENERAL, "orchestrator", []))
    agent = Mock()
    report = TaskExecutor(
        store,
        AdaptiveRouter(available_agents={"orchestrator"}),
        {"orchestrator": agent},
        max_calls=0,
        max_retries=0,
    ).execute("root")

    assert report.calls == 0
    assert report.status == "FAILED"
    assert "ExecutionBudgetExceeded" in report.errors[0]
    agent.run.assert_not_called()


def test_token_budget_is_checked_against_configured_output_limit():
    store = make_store(("one", TaskType.GENERAL, "orchestrator", []))
    agent = Mock()
    report = TaskExecutor(
        store,
        AdaptiveRouter(available_agents={"orchestrator"}),
        {"orchestrator": agent},
        max_tokens=100,
        max_retries=0,
    ).execute("root")

    assert report.calls == 0
    assert store.get_task("one").status is TaskStatus.FAILED
    agent.run.assert_not_called()


def test_real_request_caps_output_to_remaining_budget_and_updates_estimate(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "unit-test-key")
    monkeypatch.setattr("dotenv.load_dotenv", lambda: None)
    sdk = Mock()
    sdk.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"), finish_reason="stop")],
        model="nvidia/nemotron-3-ultra-550b-a55b:free",
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=1684, total_tokens=1689),
    )
    monkeypatch.setattr(llm_client_module, "OpenAI", Mock(return_value=sdk))
    monkeypatch.setattr(llm_client_module, "Logger", Mock(return_value=Mock()))
    client = LLMClient()
    store = make_store(("one", TaskType.GENERAL, "orchestrator", []))
    executor = TaskExecutor(
        store, AdaptiveRouter(available_agents={"orchestrator"}),
        {"orchestrator": Mock()}, max_tokens=20_000,
    )
    executor._budget_tokens_used = 18_296
    prompt = "x"
    message = [{"role": "user", "content": prompt}]
    reservation = executor._guard_llm_request(
        "nvidia/nemotron-3-ultra-550b-a55b:free", message, {"max_tokens": 4_000}
    )

    assert reservation["generation_params"] == {"max_tokens": 1_702}
    assert executor._reserved_tokens == 1_704
    client.generate(
        "nvidia/nemotron-3-ultra-550b-a55b:free", message,
        **reservation["generation_params"],
    )
    # Directly observe the reserved request as LLMClient hooks normally do.
    executor._observe_llm_response(
        "nvidia/nemotron-3-ultra-550b-a55b:free",
        LLMResponse("ok", "model", 5, 1684, 1689, "stop"), 1.0, None, reservation,
    )
    assert sdk.chat.completions.create.call_args.kwargs["max_tokens"] == 1_702
    assert executor._budget_tokens_used == 19_985
    assert executor.metrics_snapshot()["estimated_tokens"] == 0
    assert executor.metrics_snapshot()["budget_violation"] is None


def test_parallel_llm_requests_reserve_shared_token_budget_atomically():
    executor = TaskExecutor(
        make_store(("work", TaskType.GENERAL, "orchestrator", [])),
        AdaptiveRouter(available_agents={"orchestrator"}),
        {"orchestrator": Mock()},
        max_tokens=200,
    )
    barrier = threading.Barrier(2)
    results = []
    errors = []
    def reserve():
        barrier.wait(timeout=2)
        try:
            results.append(executor._guard_llm_request(
                "nvidia/nemotron-3-ultra-550b-a55b:free",
                [{"role": "user", "content": "x"}],
                {"max_tokens": 100},
            ))
        except Exception as error:
            errors.append(error)
    threads = [threading.Thread(target=reserve) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert len(results) == 1
    assert len(errors) == 1
    assert isinstance(errors[0], ExecutionBudgetExceeded)
    assert executor._calls == 1
    assert executor._reserved_tokens == 102


def test_partial_usage_estimates_only_the_missing_usage_field():
    store = make_store(("one", TaskType.GENERAL, "orchestrator", []))
    executor = TaskExecutor(
        store, AdaptiveRouter(available_agents={"orchestrator"}),
        {"orchestrator": Mock()}, max_tokens=1_000,
    )
    reservation = executor._guard_llm_request(
        "unknown/model", [{"role": "user", "content": "short prompt"}],
        {"max_tokens": 100},
    )
    executor._observe_llm_response(
        "unknown/model", SimpleNamespace(prompt_tokens=9, completion_tokens=None),
        2.0, None, reservation,
    )

    metrics = executor.metrics_snapshot()
    assert metrics["prompt_tokens"] == 9
    assert metrics["completion_tokens"] == 0
    assert metrics["estimated_tokens"] == 100
    assert executor._budget_tokens_used == 109


def test_budget_rejection_reason_is_saved_in_execution_history(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "unit-test-key")
    monkeypatch.setattr("dotenv.load_dotenv", lambda: None)
    sdk = Mock()
    monkeypatch.setattr(llm_client_module, "OpenAI", Mock(return_value=sdk))
    monkeypatch.setattr(llm_client_module, "Logger", Mock(return_value=Mock()))
    client = LLMClient()

    class CallingAgent:
        def __init__(self):
            self.client = client

        def run(self, _prompt):
            return self.client.generate(
                "nvidia/nemotron-3-ultra-550b-a55b:free",
                [{"role": "user", "content": "x"}], max_tokens=100,
            )

    store = make_store(("one", TaskType.GENERAL, "orchestrator", []))
    executor = TaskExecutor(
        store, AdaptiveRouter(available_agents={"orchestrator"}),
        {"orchestrator": CallingAgent()}, max_tokens=200, max_retries=1,
    )
    executor._budget_tokens_used = 150
    report = executor.execute("root")

    rejected = next(event for event in report.history if event.event == "budget_rejected")
    assert rejected.budget_details["used_tokens"] == 150
    assert rejected.budget_details["max_tokens"] == 200
    assert "prompt_upper_bound" in rejected.error
    assert store.get_task("root").result["history"][-1]["budget_details"] == rejected.budget_details
    assert sdk.chat.completions.create.call_count == 0


def test_empty_response_metadata_is_retained_and_retry_is_bounded_with_fallback():
    store = make_store(("coding", TaskType.CODING, "coding", []))
    empty = AgentResult(
        "coding", "", "coding-model", False, "empty", 8, 0,
        total_tokens=8, finish_reason="stop", error_type="EmptyModelResponseError",
    )
    coding = Mock()
    coding.run.side_effect = [empty, empty]
    fallback = Mock()
    fallback.run.return_value = agent_result("orchestrator", "fallback completed")

    report = TaskExecutor(
        store,
        AdaptiveRouter(available_agents={"coding", "orchestrator"}),
        {"coding": coding, "orchestrator": fallback},
        max_retries=2,
    ).execute("root")

    events = [event for event in report.history if event.task_id == "coding"]
    assert [event.event for event in events] == [
        "empty_response_retry", "attempt_failed", "completed",
    ]
    assert events[0].finish_reason == "stop"
    assert events[0].total_tokens == 8
    assert coding.run.call_count == 2
    assert fallback.run.call_count == 1
    assert events[-1].fallback is True
    assert report.prompt_tokens == 20


def test_empty_truncated_response_retries_same_coding_agent_before_fallback():
    store = make_store(("coding", TaskType.CODING, "coding", []))
    empty_truncated = AgentResult(
        "coding", "", "coding-model", False, "empty", 8, 0,
        total_tokens=8, finish_reason="length", error_type="EmptyModelResponseError",
    )
    complete = AgentResult(
        "coding", '{"solution":"ready"}', "coding-model", True, None, 10, 5,
        total_tokens=15, finish_reason="stop",
    )
    coding = Mock()
    coding.run.side_effect = [empty_truncated, complete]
    coding.parse_solution.return_value = {"solution": "ready"}
    fallback = Mock()

    report = TaskExecutor(
        store,
        AdaptiveRouter(available_agents={"coding", "orchestrator"}),
        {"coding": coding, "orchestrator": fallback},
        max_retries=1,
    ).execute("root")

    events = [event for event in report.history if event.task_id == "coding"]
    assert [event.event for event in events] == ["empty_response_retry", "completed"]
    assert events[0].finish_reason == "length"
    assert events[0].retry_reason == "Retry the same agent once after an empty completion."
    assert coding.run.call_count == 2
    fallback.run.assert_not_called()
    assert report.status == "COMPLETED"


def test_empty_response_does_not_retry_when_finish_reason_is_not_retryable():
    store = make_store(("coding", TaskType.CODING, "coding", []))
    coding = Mock()
    coding.run.return_value = AgentResult(
        "coding", "", "coding-model", False, "empty", 8, 0,
        total_tokens=8, finish_reason="content_filter", error_type="EmptyModelResponseError",
    )
    fallback = Mock()
    fallback.run.return_value = agent_result("orchestrator", "fallback completed")
    report = TaskExecutor(
        store, AdaptiveRouter(available_agents={"coding", "orchestrator"}),
        {"coding": coding, "orchestrator": fallback}, max_retries=1,
    ).execute("root")

    event = next(event for event in report.history if event.event == "attempt_failed")
    assert event.finish_reason == "content_filter"
    assert coding.run.call_count == 1
    assert fallback.run.call_count == 1


def test_cost_budget_uses_configured_model_prices(tmp_path):
    models_path = tmp_path / "models.yaml"
    agents_path = tmp_path / "agents.yaml"
    models_path.write_text(yaml.safe_dump({"models": {
        "priced": {
            "provider": "test",
            "model_id": "test/paid-model",
            "cost": {"input_per_1m_tokens": 10.0, "output_per_1m_tokens": 10.0},
            "context_window": 8192,
            "capabilities": ["text_generation"],
        },
    }}), encoding="utf-8")
    agents_path.write_text(yaml.safe_dump({"agents": {
        "orchestrator": {"model": "priced", "generation": {"max_tokens": 100}},
    }}), encoding="utf-8")
    registry = AgentConfigRegistry(agents_path, ModelRegistry(models_path))
    router = AdaptiveRouter(agent_config_registry=registry, available_agents={"orchestrator"})
    store = make_store(("one", TaskType.GENERAL, "orchestrator", []))
    agent = Mock()

    report = TaskExecutor(
        store, router, {"orchestrator": agent}, max_cost=0.0001, max_retries=0
    ).execute("root")

    assert report.calls == 0
    assert report.status == "FAILED"
    assert "cost" in report.errors[0]
    agent.run.assert_not_called()


def test_real_llmclient_hooks_enforce_request_count_and_track_actual_usage(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "unit-test-key")
    monkeypatch.setattr("dotenv.load_dotenv", lambda: None)
    sdk = Mock()
    sdk.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"), finish_reason="stop")],
        model="nvidia/nemotron-3-ultra-550b-a55b:free",
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2, total_tokens=5),
    )
    monkeypatch.setattr(llm_client_module, "OpenAI", Mock(return_value=sdk))
    monkeypatch.setattr(llm_client_module, "Logger", Mock(return_value=Mock()))
    client = LLMClient()

    class DoubleCallingExecutorAgent:
        def __init__(self, llm_client):
            self.client = llm_client

        def run(self, _prompt):
            self.client.generate(
                "nvidia/nemotron-3-ultra-550b-a55b:free",
                [{"role": "user", "content": "first"}],
                max_tokens=5,
            )
            self.client.generate(
                "nvidia/nemotron-3-ultra-550b-a55b:free",
                [{"role": "user", "content": "second"}],
                max_tokens=5,
            )

    agent = DoubleCallingExecutorAgent(client)
    store = make_store(("one", TaskType.GENERAL, "orchestrator", []))
    report = TaskExecutor(
        store,
        AdaptiveRouter(available_agents={"orchestrator"}),
        {"orchestrator": agent},
        max_calls=1,
        max_retries=0,
    ).execute("root")

    assert report.calls == 1
    assert report.prompt_tokens == 3
    assert report.completion_tokens == 2
    assert report.status == "FAILED"
    assert sdk.chat.completions.create.call_count == 1
    assert client._request_guard is None  # Hooks are restored after execution.


def test_timeout_marks_task_failed_and_ignores_late_result():
    store = make_store(("slow", TaskType.GENERAL, "orchestrator", []))
    agent = Mock()
    def slow_run(_prompt):
        time.sleep(0.08)
        return agent_result()
    agent.run.side_effect = slow_run

    report = TaskExecutor(
        store,
        AdaptiveRouter(available_agents={"orchestrator"}),
        {"orchestrator": agent},
        max_retries=0,
        task_timeout=0.01,
    ).execute("root")
    time.sleep(0.12)  # Let the uninterruptible fake finish and verify it cannot overwrite the timeout.

    assert report.status == "FAILED"
    assert store.get_task("slow").status is TaskStatus.FAILED
    assert store.get_task("slow").result is None
    assert any(event.event == "timeout" for event in report.history)
