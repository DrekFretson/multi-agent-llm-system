"""Mocked plan decomposition, validation, and atomic persistence tests."""

import json
from unittest.mock import Mock

import pytest

from src.agents import (
    AgentConfigRegistry,
    AgentResult,
    Orchestrator,
    OrchestratorPlan,
    OrchestratorPlanError,
    Subtask,
)
from src.coordination import (
    Task,
    TaskAlreadyExistsError,
    TaskPriority,
    TaskStatus,
    TaskStore,
    TaskType,
)
from src.llm import LLMClient, LLMResponse


def make_orchestrator(client: Mock, **agents) -> Orchestrator:
    return Orchestrator(
        client=client,
        agents=agents,
        config=AgentConfigRegistry().get_agent("orchestrator"),
    )


def response(payload: dict) -> LLMResponse:
    return LLMResponse(json.dumps(payload), "mock/planner", 10, 20, 30, "stop")


def item(identifier: str, agent: str, *, dependencies=None, task_type=None, priority=None, description=None):
    value = {
        "id": identifier,
        "description": description or f"Выполнить этап {identifier}",
        "assigned_agent": agent,
        "dependencies": dependencies or [],
    }
    if task_type is not None:
        value["task_type"] = task_type
    if priority is not None:
        value["priority"] = priority
    return value


def test_decompose_classifies_tasks_and_keeps_independent_branches() -> None:
    llm = Mock(spec=LLMClient)
    llm.generate.return_value = response({"subtasks": [
        item("research", "research", task_type="RESEARCH", priority="MEDIUM"),
        item("math", "math", task_type="MATH", priority="LOW"),
        item("code", "coding", dependencies=["research"], task_type="CODING", priority="HIGH"),
    ]})
    orchestrator = make_orchestrator(llm)

    plan = orchestrator.decompose("Исследуй подход и реализуй решение")

    assert len(plan.plan_id) > 0 and len(plan.root_task_id) > 0
    assert plan.created_at.tzinfo is not None
    assert [subtask.task_type for subtask in plan.subtasks] == [
        TaskType.RESEARCH, TaskType.MATH, TaskType.CODING,
    ]
    assert [subtask.assigned_agent for subtask in plan.subtasks] == ["research", "math", "coding"]
    assert plan.subtasks[0].dependencies == []
    assert plan.subtasks[2].depends_on == ["research"]
    assert plan.subtasks[2].dependencies == [plan.subtasks[0].task_id]
    assert plan.subtasks[0].parent_task_id == plan.root_task_id
    llm.generate.assert_called_once()


def test_decompose_infers_type_and_differentiated_priorities() -> None:
    llm = Mock(spec=LLMClient)
    llm.generate.return_value = response({"subtasks": [
        item("inspect", "research"),
        item("implement", "coding", dependencies=["inspect"], description="Реализовать выбранное решение"),
        item("optional", "critic", description="Необязательная проверка, если останется время"),
    ]})

    plan = make_orchestrator(llm).decompose("Разработай решение")

    assert [task.task_type for task in plan.subtasks] == [TaskType.RESEARCH, TaskType.CODING, TaskType.REVIEW]
    assert [task.priority for task in plan.subtasks] == [
        TaskPriority.MEDIUM, TaskPriority.HIGH, TaskPriority.LOW,
    ]


@pytest.mark.parametrize("subtasks", [
    [],
    [item("x", "coding"), item("x", "coding")],
    [item("x", "coding", task_type="UNKNOWN")],
    [item("x", "coding", priority="URGENT")],
    [item("x", "missing")],
    [item("x", "coding", dependencies=["missing"])],
    [item("x", "coding", dependencies=["x"])],
    [item("x", "coding", dependencies=["y"]), item("y", "math", dependencies=["x"])],
    [item("x", "coding", description=" ")],
    [{"id": "x", "description": "missing role and deps"}],
])
def test_decompose_rejects_invalid_llm_plans(subtasks) -> None:
    llm = Mock(spec=LLMClient)
    llm.generate.return_value = response({"subtasks": subtasks})
    orchestrator = make_orchestrator(llm)
    store = TaskStore()

    with pytest.raises(OrchestratorPlanError):
        plan = orchestrator.decompose("Сформируй план")
        orchestrator.save_plan(plan, store)
    assert store.get_all_tasks() == []


def test_plan_and_save_persists_root_subtasks_and_dependencies_without_execution() -> None:
    llm = Mock(spec=LLMClient)
    llm.generate.return_value = response({"subtasks": [
        item("research", "research", task_type="RESEARCH", priority="HIGH"),
        item("code", "coding", dependencies=["research"], task_type="CODING", priority="MEDIUM"),
    ]})
    research = Mock()
    research.run_research.return_value = AgentResult("research", "should not run", "mock", True, None, 1, 1)
    coding = Mock()
    coding.run.return_value = AgentResult("coding", "should not run", "mock", True, None, 1, 1)
    coding.parse_solution.return_value = object()
    orchestrator = make_orchestrator(llm, research=research, coding=coding)
    store = TaskStore()

    plan = orchestrator.plan_and_save("Исследуй и реализуй", store)

    root = store.get_task(plan.root_task_id)
    children = store.get_subtasks(plan.root_task_id)
    assert root.description == "Исследуй и реализуй"
    assert root.task_type is TaskType.GENERAL
    assert {task.assignee for task in children} == {"research", "coding"}
    assert {task.status for task in children} == {TaskStatus.ASSIGNED}
    assert {task.task_type for task in children} == {TaskType.RESEARCH, TaskType.CODING}
    by_description = {task.description: task for task in children}
    assert by_description["Выполнить этап code"].dependencies == [by_description["Выполнить этап research"].task_id]
    research.run_research.assert_not_called()
    coding.run.assert_not_called()


def test_invalid_plan_does_not_mutate_store_and_repeated_plans_have_unique_ids() -> None:
    llm = Mock(spec=LLMClient)
    llm.generate.side_effect = [
        response({"subtasks": [item("one", "coding")]}),
        response({"subtasks": [item("two", "coding")]}),
    ]
    orchestrator = make_orchestrator(llm)
    store = TaskStore()
    first = orchestrator.plan_and_save("Первая задача", store)
    second = orchestrator.plan_and_save("Вторая задача", store)

    assert first.plan_id != second.plan_id
    assert first.root_task_id != second.root_task_id
    assert len(store.get_all_tasks()) == 4


def test_task_store_conflict_does_not_leave_partial_plan() -> None:
    llm = Mock(spec=LLMClient)
    llm.generate.return_value = response({"subtasks": [item("one", "coding")]})
    orchestrator = make_orchestrator(llm)
    plan = orchestrator.decompose("Сохрани задачу")
    store = TaskStore()
    store.create_task(Task(task_id=plan.subtasks[0].task_id, description="existing"))

    with pytest.raises(OrchestratorPlanError, match="atomically save"):
        orchestrator.save_plan(plan, store)
    assert [task.task_id for task in store.get_all_tasks()] == [plan.subtasks[0].task_id]


def test_mapping_rejects_duplicate_actual_task_ids() -> None:
    llm = Mock(spec=LLMClient)
    orchestrator = make_orchestrator(llm)
    common = {"task_id": "same-task"}
    plan = OrchestratorPlan("request", [
        Subtask("one", "one", "coding", [], **common),
        Subtask("two", "two", "math", [], **common),
    ])

    with pytest.raises(OrchestratorPlanError, match="Duplicate task_id"):
        orchestrator.validate_plan(plan, require_injected_agents=False, require_subtasks=True, allow_orchestrator=True)
