"""Rule-based router tests using local model metadata only."""

import pytest

from src.coordination import Task, TaskType
from src.routing import AdaptiveRouter, NoAvailableRouteError


def make_task(kind=TaskType.CODING, assignee=None, description="Implement a parser"):
    return Task("task", description, task_type=kind, assignee=assignee)


def test_router_uses_configured_specialist_and_model_registry_metadata():
    decision = AdaptiveRouter(available_agents={"coding"}).route(make_task())
    assert decision.assigned_agent == "coding"
    assert decision.logical_model_id == "coding"
    assert decision.model_id
    assert decision.context_limit == 256000
    assert decision.estimated_cost == 0.0


def test_router_falls_back_to_configured_orchestrator_when_specialist_model_is_down():
    router = AdaptiveRouter(
        available_agents={"coding", "orchestrator"},
        model_available=lambda model_id: model_id != "cohere/north-mini-code:free",
    )
    decision = router.route(make_task())
    assert decision.assigned_agent == "orchestrator"
    assert decision.fallback is True
    assert decision.logical_model_id == "reasoning"


def test_router_obeys_context_window_and_exclusions():
    router = AdaptiveRouter(available_agents={"coding", "orchestrator"})
    task = make_task(description="x" * 800_000)
    decision = router.route(task)
    assert decision.assigned_agent == "orchestrator"
    with pytest.raises(NoAvailableRouteError, match="No route"):
        router.route(task, exclude_agents={"coding", "orchestrator"})


def test_router_keeps_unknown_latency_unknown_and_checks_measured_limit():
    decision = AdaptiveRouter(available_agents={"math"}).route(make_task(TaskType.MATH, "math"))
    assert decision.estimated_latency_ms is None
    measured = AdaptiveRouter(
        available_agents={"math"}, measured_latency_ms={"reasoning": 25.0}
    )
    with pytest.raises(NoAvailableRouteError, match="latency"):
        measured.route(make_task(TaskType.MATH, "math"), max_latency_ms=10)
    with pytest.raises(NoAvailableRouteError, match="unknown latency"):
        AdaptiveRouter(available_agents={"math"}).route(
            make_task(TaskType.MATH, "math"), max_latency_ms=100
        )


def test_router_does_not_replace_critic_with_unverified_general_executor():
    router = AdaptiveRouter(
        available_agents={"critic", "orchestrator"},
        model_available=lambda _model_id: False,
    )
    with pytest.raises(NoAvailableRouteError, match="critic: configured model is unavailable"):
        router.route(make_task(TaskType.REVIEW, "critic"))


def test_router_rejects_non_task_inputs():
    with pytest.raises(TypeError, match="Task"):
        AdaptiveRouter().route({"description": "no"})
