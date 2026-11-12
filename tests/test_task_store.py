"""Validation, lifecycle, dependency, and isolation tests for TaskStore."""

from datetime import datetime, timedelta, timezone

import pytest

from src.coordination import (
    DependencyError,
    InvalidTaskStatusError,
    Task,
    TaskAlreadyExistsError,
    TaskNotFoundError,
    TaskPriority,
    TaskStatus,
    TaskStore,
    TaskType,
    TaskValidationError,
)


def make_task(task_id: str, **overrides) -> Task:
    values = {"task_id": task_id, "description": f"Description for {task_id}"}
    values.update(overrides)
    return Task(**values)


def test_task_defaults_and_utc_timestamps() -> None:
    task = make_task("task-a")
    assert task.task_type is TaskType.GENERAL
    assert task.priority is TaskPriority.MEDIUM
    assert task.status is TaskStatus.NEW
    assert task.parent_task_id is None
    assert task.assignee is None
    assert task.dependencies == []
    assert task.deadline is None
    assert task.result is None
    assert task.error is None
    assert task.created_at.tzinfo is timezone.utc
    assert task.updated_at.tzinfo is timezone.utc


def test_task_explicit_values_are_preserved_and_datetimes_normalized_to_utc() -> None:
    local_time = datetime(2025, 1, 1, 3, tzinfo=timezone(timedelta(hours=3)))
    task = make_task(
        "task-a",
        task_type=TaskType.CODING,
        priority=TaskPriority.HIGH,
        status=TaskStatus.READY,
        assignee="coding-agent",
        dependencies=["dep-a"],
        deadline=local_time,
        created_at=local_time,
        updated_at=local_time,
    )
    assert task.task_type is TaskType.CODING
    assert task.priority is TaskPriority.HIGH
    assert task.status is TaskStatus.READY
    assert task.deadline.tzinfo is timezone.utc
    assert task.deadline.hour == 0
    assert task.created_at == local_time.astimezone(timezone.utc)


@pytest.mark.parametrize(
    ("field", "value", "error_text"),
    [
        ("task_id", "", "task_id"),
        ("task_id", 2, "task_id"),
        ("description", "  ", "description"),
        ("description", None, "description"),
        ("task_type", "CODING", "task_type"),
        ("priority", "HIGH", "priority"),
        ("status", "NEW", "status"),
        ("parent_task_id", "", "parent_task_id"),
        ("assignee", "", "assignee"),
        ("dependencies", "not-a-list", "dependencies"),
        ("dependencies", ["dep", "dep"], "duplicate"),
        ("dependencies", ["task-a"], "depend on itself"),
        ("error", 5, "error"),
        ("deadline", "tomorrow", "deadline"),
        ("created_at", datetime(2025, 1, 1), "created_at"),
        ("updated_at", datetime(2025, 1, 1), "updated_at"),
    ],
)
def test_task_rejects_invalid_values(field, value, error_text) -> None:
    with pytest.raises(TaskValidationError, match=error_text):
        Task(**{"task_id": "task-a", "description": "Description", field: value})


def test_task_rejects_itself_as_parent() -> None:
    with pytest.raises(TaskValidationError, match="own parent"):
        make_task("task-a", parent_task_id="task-a")


def test_dependency_defaults_are_not_shared_between_tasks() -> None:
    first = make_task("task-a")
    second = make_task("task-b")
    first.dependencies.append("other")
    assert second.dependencies == []


def test_task_id_cannot_be_changed_after_creation() -> None:
    task = make_task("task-a")
    with pytest.raises(AttributeError, match="immutable"):
        task.task_id = "task-b"


def test_create_and_get_task_return_equivalent_detached_values() -> None:
    store = TaskStore()
    original = make_task("task-a", dependencies=[], result={"items": [1, 2]})
    created = store.create_task(original)
    fetched = store.get_task("task-a")
    assert created == original
    assert fetched == original
    assert created is not original
    assert fetched is not created


def test_create_only_accepts_task_instance() -> None:
    with pytest.raises(TaskValidationError, match="Task instance"):
        TaskStore().create_task({"task_id": "task-a"})


def test_duplicate_task_id_is_rejected() -> None:
    store = TaskStore()
    store.create_task(make_task("task-a"))
    with pytest.raises(TaskAlreadyExistsError, match="already exists"):
        store.create_task(make_task("task-a", description="another"))


def test_create_tasks_atomically_stores_related_batch() -> None:
    store = TaskStore()
    created = store.create_tasks([
        make_task("root"),
        make_task("first", parent_task_id="root"),
        make_task("second", parent_task_id="root", dependencies=["first"]),
    ])

    assert [task.task_id for task in created] == ["root", "first", "second"]
    assert store.get_task("second").dependencies == ["first"]
    assert [task.task_id for task in store.get_subtasks("root")] == ["first", "second"]


def test_create_tasks_does_not_partially_store_invalid_batch() -> None:
    store = TaskStore()
    with pytest.raises(DependencyError, match="was not found"):
        store.create_tasks([
            make_task("root"),
            make_task("child", parent_task_id="root", dependencies=["missing"]),
        ])
    assert store.get_all_tasks() == []


def test_create_tasks_rejects_duplicate_ids_without_partial_storage() -> None:
    store = TaskStore()
    with pytest.raises(TaskAlreadyExistsError, match="duplicate task IDs"):
        store.create_tasks([make_task("same"), make_task("same")])
    assert store.get_all_tasks() == []


def test_get_missing_task_raises_not_found() -> None:
    with pytest.raises(TaskNotFoundError, match="not found"):
        TaskStore().get_task("missing")


def test_task_id_lookup_validation() -> None:
    with pytest.raises(TaskValidationError, match="task_id"):
        TaskStore().get_task(" ")


def test_create_child_requires_and_stores_existing_parent() -> None:
    store = TaskStore()
    store.create_task(make_task("parent"))
    child = store.create_task(make_task("child", parent_task_id="parent"))
    assert child.parent_task_id == "parent"
    assert [item.task_id for item in store.get_subtasks("parent")] == ["child"]


def test_create_child_with_missing_parent_is_rejected() -> None:
    with pytest.raises(TaskNotFoundError, match="Parent task"):
        TaskStore().create_task(make_task("child", parent_task_id="missing"))


def test_create_task_requires_existing_dependencies() -> None:
    with pytest.raises(DependencyError, match="was not found"):
        TaskStore().create_task(make_task("task-a", dependencies=["missing"]))


def test_successful_atomic_update_changes_only_allowed_fields_and_timestamp() -> None:
    store = TaskStore()
    created = store.create_task(make_task("task-a"))
    updated = store.update_task("task-a", description="Updated", priority=TaskPriority.HIGH)
    assert updated.description == "Updated"
    assert updated.priority is TaskPriority.HIGH
    assert updated.updated_at > created.updated_at
    assert store.get_task("task-a").description == "Updated"


def test_update_missing_task_raises_not_found() -> None:
    with pytest.raises(TaskNotFoundError):
        TaskStore().update_task("missing", description="new")


@pytest.mark.parametrize("field", ["task_id", "created_at", "updated_at", "status", "assignee"])
def test_update_task_rejects_reserved_fields(field) -> None:
    store = TaskStore()
    store.create_task(make_task("task-a"))
    with pytest.raises(TaskValidationError, match="cannot be updated"):
        if field == "task_id":
            store.update_task("task-a", task_id="invalid")
        else:
            store.update_task("task-a", **{field: "invalid"})


def test_update_task_rejects_unknown_fields() -> None:
    store = TaskStore()
    store.create_task(make_task("task-a"))
    with pytest.raises(TaskValidationError, match="Unsupported task field"):
        store.update_task("task-a", unknown_field=1)


def test_failed_update_does_not_partially_change_stored_task() -> None:
    store = TaskStore()
    store.create_task(make_task("task-a", description="Original"))
    with pytest.raises(DependencyError):
        store.update_task("task-a", description="Changed", dependencies=["missing"])
    stored = store.get_task("task-a")
    assert stored.description == "Original"
    assert stored.dependencies == []


def test_update_rejects_parent_cycle_without_mutation() -> None:
    store = TaskStore()
    store.create_task(make_task("root"))
    store.create_task(make_task("child", parent_task_id="root"))
    with pytest.raises(TaskValidationError, match="Parent task relationships"):
        store.update_task("root", parent_task_id="child")
    assert store.get_task("root").parent_task_id is None


@pytest.mark.parametrize(
    ("current", "target"),
    [
        (TaskStatus.NEW, TaskStatus.READY),
        (TaskStatus.NEW, TaskStatus.ASSIGNED),
        (TaskStatus.NEW, TaskStatus.FAILED),
        (TaskStatus.NEW, TaskStatus.CANCELLED),
        (TaskStatus.READY, TaskStatus.ASSIGNED),
        (TaskStatus.READY, TaskStatus.FAILED),
        (TaskStatus.READY, TaskStatus.CANCELLED),
        (TaskStatus.ASSIGNED, TaskStatus.READY),
        (TaskStatus.ASSIGNED, TaskStatus.RUNNING),
        (TaskStatus.ASSIGNED, TaskStatus.WAITING),
        (TaskStatus.ASSIGNED, TaskStatus.FAILED),
        (TaskStatus.ASSIGNED, TaskStatus.CANCELLED),
        (TaskStatus.RUNNING, TaskStatus.WAITING),
        (TaskStatus.RUNNING, TaskStatus.REVIEW),
        (TaskStatus.RUNNING, TaskStatus.COMPLETED),
        (TaskStatus.RUNNING, TaskStatus.FAILED),
        (TaskStatus.RUNNING, TaskStatus.CANCELLED),
        (TaskStatus.WAITING, TaskStatus.READY),
        (TaskStatus.WAITING, TaskStatus.ASSIGNED),
        (TaskStatus.WAITING, TaskStatus.RUNNING),
        (TaskStatus.WAITING, TaskStatus.FAILED),
        (TaskStatus.WAITING, TaskStatus.CANCELLED),
        (TaskStatus.REVIEW, TaskStatus.READY),
        (TaskStatus.REVIEW, TaskStatus.COMPLETED),
        (TaskStatus.REVIEW, TaskStatus.FAILED),
        (TaskStatus.REVIEW, TaskStatus.CANCELLED),
        (TaskStatus.NEEDS_INPUT, TaskStatus.READY),
        (TaskStatus.NEEDS_INPUT, TaskStatus.FAILED),
        (TaskStatus.NEEDS_INPUT, TaskStatus.CANCELLED),
        (TaskStatus.FAILED, TaskStatus.READY),
        (TaskStatus.FAILED, TaskStatus.ASSIGNED),
        (TaskStatus.FAILED, TaskStatus.CANCELLED),
    ],
)
def test_allowed_status_transitions(current: TaskStatus, target: TaskStatus) -> None:
    store = TaskStore()
    store.create_task(make_task("task-a", status=current))
    updated = store.update_status("task-a", target)
    assert updated.status is target


@pytest.mark.parametrize(
    ("current", "target"),
    [
        (TaskStatus.NEW, TaskStatus.RUNNING),
        (TaskStatus.COMPLETED, TaskStatus.RUNNING),
        (TaskStatus.CANCELLED, TaskStatus.RUNNING),
        (TaskStatus.FAILED, TaskStatus.COMPLETED),
        (TaskStatus.REVIEW, TaskStatus.ASSIGNED),
        (TaskStatus.NEEDS_INPUT, TaskStatus.COMPLETED),
    ],
)
def test_disallowed_status_transitions_do_not_mutate_task(current, target) -> None:
    store = TaskStore()
    store.create_task(make_task("task-a", status=current))
    old_updated_at = store.get_task("task-a").updated_at
    with pytest.raises(InvalidTaskStatusError):
        store.update_status("task-a", target)
    assert store.get_task("task-a").status is current
    assert store.get_task("task-a").updated_at == old_updated_at


def test_invalid_status_type_and_missing_task() -> None:
    store = TaskStore()
    store.create_task(make_task("task-a"))
    with pytest.raises(InvalidTaskStatusError):
        store.update_status("task-a", "READY")
    with pytest.raises(TaskNotFoundError):
        store.update_status("missing", TaskStatus.READY)


def test_assign_task_sets_assignee_and_assigned_status() -> None:
    store = TaskStore()
    store.create_task(make_task("task-a"))
    assigned = store.assign_task("task-a", "agent-1")
    assert assigned.assignee == "agent-1"
    assert assigned.status is TaskStatus.ASSIGNED


def test_assign_task_supports_retry_after_failure() -> None:
    store = TaskStore()
    store.create_task(make_task("task-a", status=TaskStatus.FAILED))
    reassigned = store.assign_task("task-a", "replacement-agent")
    assert reassigned.assignee == "replacement-agent"
    assert reassigned.status is TaskStatus.ASSIGNED


@pytest.mark.parametrize("agent_id", ["", "  ", None, 2])
def test_assign_task_rejects_invalid_agent_id(agent_id) -> None:
    store = TaskStore()
    store.create_task(make_task("task-a"))
    with pytest.raises(TaskValidationError, match="agent_id"):
        store.assign_task("task-a", agent_id)


def test_assign_nonexistent_task_and_terminal_task_are_rejected() -> None:
    store = TaskStore()
    with pytest.raises(TaskNotFoundError):
        store.assign_task("missing", "agent")
    store.create_task(make_task("done", status=TaskStatus.COMPLETED))
    with pytest.raises(InvalidTaskStatusError):
        store.assign_task("done", "agent")


def test_add_dependency_and_get_dependencies() -> None:
    store = TaskStore()
    store.create_task(make_task("base"))
    store.create_task(make_task("middle"))
    store.create_task(make_task("top"))
    updated = store.add_dependency("top", "base")
    store.add_dependency("top", "middle")
    assert updated.dependencies == ["base"]
    assert [task.task_id for task in store.get_dependencies("top")] == ["base", "middle"]


def test_dependencies_complete_only_when_all_are_completed() -> None:
    store = TaskStore()
    store.create_task(make_task("dep-a"))
    store.create_task(make_task("dep-b"))
    store.create_task(make_task("target", dependencies=["dep-a", "dep-b"]))
    assert not store.are_dependencies_completed("target")
    for task_id in ("dep-a",):
        store.update_status(task_id, TaskStatus.READY)
        store.update_status(task_id, TaskStatus.ASSIGNED)
        store.update_status(task_id, TaskStatus.RUNNING)
        store.update_status(task_id, TaskStatus.COMPLETED)
    assert not store.are_dependencies_completed("target")
    store.update_status("dep-b", TaskStatus.READY)
    store.update_status("dep-b", TaskStatus.ASSIGNED)
    store.update_status("dep-b", TaskStatus.RUNNING)
    store.update_status("dep-b", TaskStatus.COMPLETED)
    assert store.are_dependencies_completed("target")


def test_task_without_dependencies_is_ready_from_dependency_perspective() -> None:
    store = TaskStore()
    store.create_task(make_task("task-a"))
    assert store.are_dependencies_completed("task-a")


def test_add_dependency_rejects_duplicate_self_and_missing_ids() -> None:
    store = TaskStore()
    store.create_task(make_task("task-a"))
    store.create_task(make_task("task-b"))
    store.add_dependency("task-a", "task-b")
    with pytest.raises(DependencyError, match="already depends"):
        store.add_dependency("task-a", "task-b")
    with pytest.raises(DependencyError, match="itself"):
        store.add_dependency("task-a", "task-a")
    with pytest.raises(TaskNotFoundError):
        store.add_dependency("missing", "task-b")
    with pytest.raises(TaskNotFoundError):
        store.add_dependency("task-a", "missing")


def test_add_dependency_rejects_direct_cycle_atomically() -> None:
    store = TaskStore()
    store.create_task(make_task("a"))
    store.create_task(make_task("b", dependencies=["a"]))
    with pytest.raises(DependencyError, match="cycle"):
        store.add_dependency("a", "b")
    assert store.get_task("a").dependencies == []


def test_add_dependency_rejects_cycle_of_three_or_more_tasks() -> None:
    store = TaskStore()
    store.create_task(make_task("a"))
    store.create_task(make_task("b", dependencies=["a"]))
    store.create_task(make_task("c", dependencies=["b"]))
    store.create_task(make_task("d", dependencies=["c"]))
    with pytest.raises(DependencyError, match="cycle"):
        store.add_dependency("a", "d")
    assert store.get_task("a").dependencies == []


def test_update_task_dependency_cycle_is_rejected_atomically() -> None:
    store = TaskStore()
    store.create_task(make_task("a"))
    store.create_task(make_task("b", dependencies=["a"]))
    with pytest.raises(DependencyError, match="cycle"):
        store.update_task("a", description="changed", dependencies=["b"])
    assert store.get_task("a").description == "Description for a"
    assert store.get_task("a").dependencies == []


def test_get_subtasks_requires_existing_parent_and_returns_only_direct_children() -> None:
    store = TaskStore()
    store.create_task(make_task("root"))
    store.create_task(make_task("child", parent_task_id="root"))
    store.create_task(make_task("grandchild", parent_task_id="child"))
    assert [task.task_id for task in store.get_subtasks("root")] == ["child"]
    with pytest.raises(TaskNotFoundError):
        store.get_subtasks("missing")


def test_get_tasks_by_status_and_all_tasks_return_copies() -> None:
    store = TaskStore()
    store.create_task(make_task("new-task"))
    store.create_task(make_task("running-task", status=TaskStatus.RUNNING))
    running = store.get_tasks_by_status(TaskStatus.RUNNING)
    assert [task.task_id for task in running] == ["running-task"]
    all_tasks = store.get_all_tasks()
    assert [task.task_id for task in all_tasks] == ["new-task", "running-task"]
    running[0].description = "external mutation"
    running[0].dependencies.append("new-task")
    running[0].result = {"changed": True}
    assert store.get_task("running-task").description == "Description for running-task"
    assert store.get_task("running-task").dependencies == []
    assert store.get_task("running-task").result is None


def test_dependencies_and_subtasks_are_detached_copies() -> None:
    store = TaskStore()
    store.create_task(make_task("parent"))
    store.create_task(make_task("dep"))
    store.create_task(make_task("child", parent_task_id="parent", dependencies=["dep"]))
    dependencies = store.get_dependencies("child")
    subtasks = store.get_subtasks("parent")
    dependencies[0].description = "mutated dependency"
    subtasks[0].dependencies.clear()
    assert store.get_task("dep").description == "Description for dep"
    assert store.get_task("child").dependencies == ["dep"]


def test_status_filter_rejects_non_enum() -> None:
    with pytest.raises(InvalidTaskStatusError):
        TaskStore().get_tasks_by_status("NEW")
