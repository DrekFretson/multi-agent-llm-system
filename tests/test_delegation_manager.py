"""Unit tests for bounded delegation over the local TaskStore and MessageBus."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.communication import Message, MessageBus, MessageType
from src.coordination import (
    DelegationCycleError,
    DelegationDeliveryError,
    DelegationLimitError,
    DelegationManager,
    DelegationPermissionError,
    DelegationResultError,
    DelegationTaskError,
    DuplicateResultError,
    InvalidHelpRequestError,
    Task,
    TaskStatus,
    TaskStore,
)


class FakeAgent:
    def __init__(self, agent_id: str, output: object = "done") -> None:
        self.agent_id = agent_id
        self.output = output
        self.calls: list[tuple[str, object]] = []

    def run(self, task: str, *, additional_context: str | None = None) -> object:
        self.calls.append((task, additional_context))
        return self.output


class RaisingLogger:
    def log_event(self, event_type: str, **fields: object) -> None:
        raise OSError("log is unavailable")


@dataclass
class FakeAgentResult:
    success: bool
    content: str = ""
    error: str | None = None


@pytest.fixture
def system():
    store = TaskStore()
    bus = MessageBus()
    for agent_id in ("research_agent", "coding_agent", "math_agent"):
        bus.register_agent(agent_id)
    agents = {
        "research_agent": FakeAgent("research_agent"),
        "coding_agent": FakeAgent("coding_agent"),
        "math_agent": FakeAgent("math_agent"),
    }
    manager = DelegationManager(store, bus, agents=agents)
    root = store.create_task(Task(task_id="root", description="root request"))
    store.assign_task(root.task_id, "research_agent")
    return store, bus, agents, manager


def _request(manager: DelegationManager, **overrides: object) -> Message:
    args: dict[str, object] = {
        "parent_task_id": "root",
        "initiator_id": "research_agent",
        "recipient_id": "coding_agent",
        "description": "write an implementation",
    }
    args.update(overrides)
    return manager.request_help(**args)  # type: ignore[arg-type]


def _complete_help(manager: DelegationManager, bus: MessageBus, recipient: str) -> Message:
    help_message = bus.receive(recipient)
    assert help_message is not None
    result_message = manager.handle_help_request(help_message)
    assert result_message.message_type is MessageType.RESULT
    return result_message


def test_research_delegates_to_coding_and_keeps_parent_active(system):
    store, bus, _, manager = system
    request = _request(manager, context={"source": "validated"})
    child = store.get_task(request.task_id)
    assert request.message_type is MessageType.HELP_REQUEST
    assert child.parent_task_id == "root"
    assert child.assignee == "coding_agent"
    assert child.status is TaskStatus.WAITING
    assert store.get_task("root").status is TaskStatus.ASSIGNED
    assert bus.receive("coding_agent").message_id == request.message_id


def test_coding_delegates_to_math_and_math_result_returns_to_coding(system):
    store, bus, agents, manager = system
    first = _request(manager)
    child = store.get_task(first.task_id)
    store.assign_task(child.task_id, "coding_agent")

    second = manager.request_help(
        parent_task_id=child.task_id,
        initiator_id="coding_agent",
        recipient_id="math_agent",
        description="calculate 6 * 7",
    )
    returned = _complete_help(manager, bus, "math_agent")
    assert returned.receiver == "coding_agent"
    assert returned.task_id == second.task_id
    manager.handle_result(returned)
    assert store.get_task(second.task_id).status is TaskStatus.COMPLETED
    assert store.get_task(child.task_id).status is TaskStatus.ASSIGNED
    assert agents["math_agent"].calls == [("calculate 6 * 7", None)]


def test_successful_result_is_stored_and_parent_is_not_completed(system):
    store, bus, _, manager = system
    request = _request(manager)
    result = _complete_help(manager, bus, "coding_agent")
    completed = manager.handle_result(result)
    assert completed.status is TaskStatus.COMPLETED
    assert completed.result == "done"
    assert store.get_task("root").status is TaskStatus.ASSIGNED


def test_agent_execution_receives_description_and_context(system):
    _, bus, agents, manager = system
    _request(manager, context="project context")
    _complete_help(manager, bus, "coding_agent")
    assert agents["coding_agent"].calls == [("write an implementation", "project context")]


def test_math_agent_returns_result_to_coding_agent(system):
    store, bus, _, manager = system
    manager.request_help(parent_task_id="root", initiator_id="research_agent",
                         recipient_id="coding_agent", description="code")
    coding_task = store.get_subtasks("root")[0]
    store.assign_task(coding_task.task_id, "coding_agent")
    math_help = manager.request_help(parent_task_id=coding_task.task_id,
                                     initiator_id="coding_agent", recipient_id="math_agent",
                                     description="compute 1+1")
    result2 = _complete_help(manager, bus, "math_agent")
    assert result2.task_id == math_help.task_id
    assert result2.receiver == "coding_agent"


def test_forbidden_direction_is_rejected_before_child_creation(system):
    store, _, _, manager = system
    with pytest.raises(DelegationPermissionError):
        _request(manager, recipient_id="math_agent")
    assert store.get_subtasks("root") == []


def test_missing_parent_and_wrong_initiator_are_rejected(system):
    store, _, _, manager = system
    with pytest.raises(DelegationTaskError):
        _request(manager, parent_task_id="missing")
    with pytest.raises(DelegationPermissionError):
        _request(manager, initiator_id="coding_agent")
    assert store.get_subtasks("root") == []


def test_depth_three_allowed_and_depth_four_rejected():
    store, bus = TaskStore(), MessageBus()
    ids = ["a", "b", "c", "d", "e"]
    for agent_id in ids:
        bus.register_agent(agent_id)
    rules = {"a": {"b"}, "b": {"c"}, "c": {"d"}, "d": {"e"}}
    manager = DelegationManager(store, bus, delegation_rules=rules)
    root = store.create_task(Task(task_id="r", description="root"))
    store.assign_task("r", "a")
    parent_id, initiator = "r", "a"
    for depth, recipient in enumerate(ids[1:4], start=1):
        help_message = manager.request_help(parent_task_id=parent_id, initiator_id=initiator,
                                            recipient_id=recipient, description=f"depth {depth}")
        parent_id, initiator = help_message.task_id, recipient
        store.update_status(parent_id, TaskStatus.RUNNING)
    assert len(store.get_all_tasks()) == 4  # root plus child at depth 3
    with pytest.raises(DelegationLimitError):
        manager.request_help(parent_task_id=parent_id, initiator_id=initiator,
                            recipient_id="e", description="depth 4")
    assert len(store.get_all_tasks()) == 4


def test_fifth_handoff_is_allowed_and_sixth_rejected(system):
    store, _, _, manager = system
    for index in range(5):
        _request(manager, request_id=f"handoff-{index}")
    assert len(store.get_subtasks("root")) == 5
    with pytest.raises(DelegationLimitError):
        _request(manager, request_id="handoff-5")
    assert len(store.get_subtasks("root")) == 5


def test_independent_roots_have_independent_handoff_counters(system):
    store, _, _, manager = system
    _request(manager)
    other = store.create_task(Task(task_id="other", description="independent root"))
    store.assign_task(other.task_id, "research_agent")
    for index in range(5):
        manager.request_help(parent_task_id="other", initiator_id="research_agent",
                             recipient_id="coding_agent", description=f"task {index}")
    assert len(store.get_subtasks("other")) == 5


def test_repeated_request_id_returns_original_message_and_counts_once(system):
    store, bus, _, manager = system
    first = _request(manager, request_id="same-request")
    second = _request(manager, request_id="same-request")
    assert first.message_id == second.message_id
    assert first.task_id == second.task_id
    assert len(store.get_subtasks("root")) == 1
    assert bus.receive("coding_agent").message_id == first.message_id
    assert bus.receive("coding_agent") is None


def test_repeated_agent_in_same_ancestry_is_rejected():
    store, bus = TaskStore(), MessageBus()
    for agent_id in ("a", "b"):
        bus.register_agent(agent_id)
    manager = DelegationManager(store, bus, delegation_rules={"a": {"b"}, "b": {"a"}})
    store.create_task(Task(task_id="root", description="root"))
    store.assign_task("root", "a")
    child = manager.request_help(parent_task_id="root", initiator_id="a", recipient_id="b",
                                 description="branch")
    store.assign_task(child.task_id, "b")
    with pytest.raises(DelegationCycleError):
        manager.request_help(parent_task_id=child.task_id, initiator_id="b", recipient_id="a",
                            description="cycle")


def test_same_agent_can_be_reused_in_independent_branches(system):
    store, _, _, manager = system
    first = _request(manager, request_id="branch-1")
    second = _request(manager, request_id="branch-2")
    assert first.task_id != second.task_id
    assert len(store.get_subtasks("root")) == 2


def test_new_child_branch_cannot_bypass_chain_cycle_guard():
    store, bus = TaskStore(), MessageBus()
    for agent_id in ("a", "b", "c"):
        bus.register_agent(agent_id)
    rules = {"a": {"b", "c"}, "b": {"c"}, "c": {"b"}}
    manager = DelegationManager(store, bus, delegation_rules=rules)
    store.create_task(Task(task_id="root", description="root"))
    store.assign_task("root", "a")
    child_b = manager.request_help(parent_task_id="root", initiator_id="a", recipient_id="b",
                                   description="b branch")
    store.assign_task(child_b.task_id, "b")
    child_c = manager.request_help(parent_task_id=child_b.task_id, initiator_id="b",
                                   recipient_id="c", description="c branch")
    store.assign_task(child_c.task_id, "c")
    with pytest.raises(DelegationCycleError):
        manager.request_help(parent_task_id=child_c.task_id, initiator_id="c", recipient_id="b",
                            description="repeat b deeper")


def test_wrong_result_sender_and_receiver_are_rejected(system):
    _, bus, _, manager = system
    help_message = _request(manager)
    valid = _complete_help(manager, bus, "coding_agent")
    bad_sender = Message(sender="research_agent", receiver="research_agent",
                         message_type=MessageType.RESULT, task_id=valid.task_id,
                         parent_task_id="root", payload=valid.payload)
    with pytest.raises(DelegationResultError):
        manager.handle_result(bad_sender)
    bad_receiver = Message(sender="coding_agent", receiver="math_agent",
                           message_type=MessageType.RESULT, task_id=valid.task_id,
                           parent_task_id="root", payload=valid.payload)
    with pytest.raises(DelegationResultError):
        manager.handle_result(bad_receiver)
    assert help_message.task_id == valid.task_id


def test_duplicate_result_is_rejected_without_second_update(system):
    store, bus, _, manager = system
    _request(manager)
    result = _complete_help(manager, bus, "coding_agent")
    manager.handle_result(result)
    with pytest.raises(DuplicateResultError):
        manager.handle_result(result)
    assert store.get_task(result.task_id).status is TaskStatus.COMPLETED


def test_unsuccessful_agent_result_marks_child_failed(system):
    store, bus, agents, manager = system
    agents["coding_agent"].output = FakeAgentResult(False, error="execution failed")
    request = _request(manager)
    result = _complete_help(manager, bus, "coding_agent")
    assert result.payload["success"] is False
    failed = manager.handle_result(result)
    assert failed.status is TaskStatus.FAILED
    assert failed.error == "execution failed"
    assert store.get_task(request.task_id).status is TaskStatus.FAILED


def test_execution_exception_is_returned_as_failed_result(system):
    store, bus, agents, manager = system
    agents["coding_agent"].run = lambda task, **kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
    request = _request(manager)
    result = _complete_help(manager, bus, "coding_agent")
    assert result.payload["success"] is False
    failed = manager.handle_result(result)
    assert failed.status is TaskStatus.FAILED
    assert "RuntimeError: boom" in failed.error
    assert store.get_task("root").status is TaskStatus.ASSIGNED
    assert request.task_id == failed.task_id


def test_delivery_error_marks_created_child_failed(system, monkeypatch):
    store, bus, _, manager = system
    monkeypatch.setattr(bus, "send", lambda message: (_ for _ in ()).throw(RuntimeError("offline")))
    with pytest.raises(DelegationDeliveryError):
        _request(manager)
    children = store.get_subtasks("root")
    assert len(children) == 1
    assert children[0].status is TaskStatus.FAILED


def test_result_delivery_error_marks_running_child_failed(system, monkeypatch):
    store, bus, _, manager = system
    _request(manager)
    monkeypatch.setattr(
        bus, "send",
        lambda message: (_ for _ in ()).throw(RuntimeError("offline"))
        if message.message_type is MessageType.RESULT else None,
    )
    help_message = bus.receive("coding_agent")
    assert help_message is not None
    with pytest.raises(DelegationDeliveryError):
        manager.handle_help_request(help_message)
    assert store.get_task(help_message.task_id).status is TaskStatus.FAILED


def test_missing_executor_is_reported_as_invalid_help_request(system):
    store, bus, agents, manager = system
    message = _request(manager)
    # The receiver remains on the same manager, but its executor is unavailable.
    del manager.agents["coding_agent"]
    with pytest.raises(InvalidHelpRequestError):
        manager.handle_help_request(message)


def test_malformed_help_request_is_rejected(system):
    _, _, _, manager = system
    malformed = Message(sender="research_agent", receiver="coding_agent",
                        message_type=MessageType.HELP_REQUEST, task_id="missing",
                        parent_task_id="root", payload={})
    with pytest.raises(InvalidHelpRequestError):
        manager.handle_help_request(malformed)


def test_missing_parent_and_unknown_agents_fail_without_creating_child(system):
    store, bus, _, manager = system
    with pytest.raises(DelegationTaskError):
        manager.request_help(parent_task_id="missing", initiator_id="research_agent",
                             recipient_id="coding_agent", description="work")
    with pytest.raises(DelegationPermissionError):
        manager.request_help(parent_task_id="root", initiator_id="unknown",
                             recipient_id="coding_agent", description="work")
    assert store.get_subtasks("root") == []


def test_logger_failure_does_not_break_delegation(system):
    store, bus, agents, _ = system
    manager = DelegationManager(store, bus, agents=agents, logger=RaisingLogger())
    message = _request(manager)
    assert message.message_type is MessageType.HELP_REQUEST
    assert store.get_task(message.task_id).status is TaskStatus.WAITING


def test_duplicate_help_message_does_not_execute_twice(system):
    store, bus, agents, manager = system
    _request(manager)
    message = bus.receive("coding_agent")
    assert message is not None
    manager.handle_help_request(message)
    with pytest.raises(InvalidHelpRequestError):
        manager.handle_help_request(message)
    assert len(agents["coding_agent"].calls) == 1
    assert store.get_task(message.task_id).status is TaskStatus.RUNNING

