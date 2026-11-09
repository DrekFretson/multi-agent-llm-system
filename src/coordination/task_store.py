"""In-memory task records and dependency-aware task state management."""

from __future__ import annotations

import copy
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class TaskType(str, Enum):
    """Domain category for a task; it does not determine its executor."""

    RESEARCH = "RESEARCH"
    CODING = "CODING"
    MATH = "MATH"
    REVIEW = "REVIEW"
    GENERAL = "GENERAL"


class TaskPriority(str, Enum):
    """Relative scheduling importance assigned to a task."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class TaskStatus(str, Enum):
    """Lifecycle states supported by the in-memory task store."""

    NEW = "NEW"
    READY = "READY"
    ASSIGNED = "ASSIGNED"
    RUNNING = "RUNNING"
    WAITING = "WAITING"
    REVIEW = "REVIEW"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class TaskStoreError(Exception):
    """Base class for task storage and validation errors."""


class TaskNotFoundError(TaskStoreError, LookupError):
    """Raised when a requested task or relationship target does not exist."""


class TaskAlreadyExistsError(TaskStoreError):
    """Raised when a task ID is already present in the store."""


class TaskValidationError(TaskStoreError, ValueError):
    """Raised when task data or an update violates the task schema."""


class InvalidTaskStatusError(TaskStoreError, ValueError):
    """Raised for an invalid status value or a forbidden state transition."""


class DependencyError(TaskStoreError, ValueError):
    """Raised when task dependency relationships are invalid or cyclic."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc_datetime(value: Any, field_name: str, *, optional: bool = False) -> datetime | None:
    if value is None and optional:
        return None
    if not isinstance(value, datetime):
        raise TaskValidationError(f"{field_name} must be a timezone-aware datetime.")
    if value.tzinfo is None or value.utcoffset() is None:
        raise TaskValidationError(f"{field_name} must be timezone-aware.")
    return value.astimezone(timezone.utc)


@dataclass
class Task:
    """A validated task record with identity that cannot be reassigned."""

    task_id: str
    description: str
    task_type: TaskType = TaskType.GENERAL
    priority: TaskPriority = TaskPriority.MEDIUM
    status: TaskStatus = TaskStatus.NEW
    parent_task_id: str | None = None
    assignee: str | None = None
    dependencies: list[str] = field(default_factory=list)
    deadline: datetime | None = None
    result: Any = None
    error: str | None = None
    created_at: datetime = field(default_factory=_utc_now)
    updated_at: datetime = field(default_factory=_utc_now)
    _identity_locked: bool = field(default=False, init=False, repr=False, compare=False)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "task_id" and getattr(self, "_identity_locked", False):
            raise AttributeError("task_id is immutable after Task creation.")
        object.__setattr__(self, name, value)

    def __post_init__(self) -> None:
        """Validate task values and normalize temporal values to UTC."""
        if not isinstance(self.task_id, str) or not self.task_id.strip():
            raise TaskValidationError("task_id must be a non-empty string.")
        if not isinstance(self.description, str) or not self.description.strip():
            raise TaskValidationError("description must be a non-empty string.")
        if not isinstance(self.task_type, TaskType):
            raise TaskValidationError("task_type must be a TaskType value.")
        if not isinstance(self.priority, TaskPriority):
            raise TaskValidationError("priority must be a TaskPriority value.")
        if not isinstance(self.status, TaskStatus):
            raise TaskValidationError("status must be a TaskStatus value.")
        if self.parent_task_id is not None:
            if not isinstance(self.parent_task_id, str) or not self.parent_task_id.strip():
                raise TaskValidationError("parent_task_id must be a non-empty string or None.")
            if self.parent_task_id == self.task_id:
                raise TaskValidationError("A task cannot be its own parent.")
        if self.assignee is not None and (
            not isinstance(self.assignee, str) or not self.assignee.strip()
        ):
            raise TaskValidationError("assignee must be a non-empty string or None.")
        if not isinstance(self.dependencies, list):
            raise TaskValidationError("dependencies must be a list of task IDs.")
        if any(not isinstance(item, str) or not item.strip() for item in self.dependencies):
            raise TaskValidationError("Every dependency ID must be a non-empty string.")
        if len(set(self.dependencies)) != len(self.dependencies):
            raise TaskValidationError("dependencies cannot contain duplicate task IDs.")
        if self.task_id in self.dependencies:
            raise TaskValidationError("A task cannot depend on itself.")
        if self.error is not None and not isinstance(self.error, str):
            raise TaskValidationError("error must be a string or None.")

        deadline = _as_utc_datetime(self.deadline, "deadline", optional=True)
        created_at = _as_utc_datetime(self.created_at, "created_at")
        updated_at = _as_utc_datetime(self.updated_at, "updated_at")
        object.__setattr__(self, "deadline", deadline)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "updated_at", updated_at)
        object.__setattr__(self, "_identity_locked", True)


class TaskStore:
    """Manage in-memory task records and validate their relationships."""

    # State changes are explicit. Failed work can be readied and assigned again;
    # terminal tasks cannot silently return to active execution.
    ALLOWED_STATUS_TRANSITIONS: dict[TaskStatus, frozenset[TaskStatus]] = {
        TaskStatus.NEW: frozenset({TaskStatus.READY, TaskStatus.ASSIGNED, TaskStatus.FAILED, TaskStatus.CANCELLED}),
        TaskStatus.READY: frozenset({TaskStatus.ASSIGNED, TaskStatus.FAILED, TaskStatus.CANCELLED}),
        TaskStatus.ASSIGNED: frozenset({TaskStatus.READY, TaskStatus.RUNNING, TaskStatus.WAITING, TaskStatus.FAILED, TaskStatus.CANCELLED}),
        TaskStatus.RUNNING: frozenset({TaskStatus.WAITING, TaskStatus.REVIEW, TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}),
        TaskStatus.WAITING: frozenset({TaskStatus.READY, TaskStatus.ASSIGNED, TaskStatus.RUNNING, TaskStatus.FAILED, TaskStatus.CANCELLED}),
        TaskStatus.REVIEW: frozenset({TaskStatus.READY, TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}),
        TaskStatus.COMPLETED: frozenset(),
        TaskStatus.FAILED: frozenset({TaskStatus.READY, TaskStatus.ASSIGNED, TaskStatus.CANCELLED}),
        TaskStatus.CANCELLED: frozenset(),
    }

    _UPDATABLE_FIELDS = frozenset({
        "description", "task_type", "priority", "parent_task_id", "dependencies",
        "deadline", "result", "error",
    })

    def __init__(self) -> None:
        """Create an empty store; all data remains local to this instance."""
        self._tasks: dict[str, Task] = {}
        self._lock = threading.RLock()

    def create_task(self, task: Task) -> Task:
        """Validate and store a task, returning a detached copy."""
        if not isinstance(task, Task):
            raise TaskValidationError("task must be a Task instance.")
        candidate = self._copy_task(task)
        with self._lock:
            if candidate.task_id in self._tasks:
                raise TaskAlreadyExistsError(f"Task {candidate.task_id!r} already exists.")
            self._validate_relationships(candidate)
            self._tasks[candidate.task_id] = candidate
            return self._copy_task(candidate)

    def get_task(self, task_id: str) -> Task:
        """Return a detached copy of one task."""
        task_id = self._validate_task_id(task_id)
        with self._lock:
            try:
                return self._copy_task(self._tasks[task_id])
            except KeyError:
                raise TaskNotFoundError(f"Task {task_id!r} was not found.") from None

    def update_task(self, identifier: str | None = None, **fields: Any) -> Task:
        """Atomically update allowed task fields and advance ``updated_at``."""
        # Accept task_id=... as the lookup argument when it is the only task_id
        # supplied. A positional identifier plus task_id=... is treated as an
        # attempt to mutate the immutable identity and receives validation error.
        if identifier is None:
            identifier = fields.pop("task_id", None)
        task_id = self._validate_task_id(identifier)
        if not fields:
            raise TaskValidationError("At least one task field must be supplied for update.")
        forbidden = {"task_id", "created_at", "updated_at", "status", "assignee"} & fields.keys()
        if forbidden:
            raise TaskValidationError(
                "Fields cannot be updated through update_task: " + ", ".join(sorted(forbidden)) + "."
            )
        unknown = fields.keys() - self._UPDATABLE_FIELDS
        if unknown:
            raise TaskValidationError("Unsupported task field(s): " + ", ".join(sorted(unknown)) + ".")

        with self._lock:
            current = self._get_internal(task_id)
            candidate = self._copy_task(current)
            try:
                for field_name, value in fields.items():
                    setattr(candidate, field_name, copy.deepcopy(value))
                candidate.__post_init__()
            except TaskValidationError:
                raise
            except Exception as error:
                raise TaskValidationError("Task update values could not be copied or validated.") from error
            self._validate_relationships(candidate)
            candidate.updated_at = self._next_timestamp(current.updated_at)
            self._tasks[task_id] = candidate
            return self._copy_task(candidate)

    def update_status(self, task_id: str, status: TaskStatus) -> Task:
        """Apply one explicitly permitted lifecycle transition."""
        task_id = self._validate_task_id(task_id)
        with self._lock:
            current = self._get_internal(task_id)
            if not isinstance(status, TaskStatus):
                raise InvalidTaskStatusError("status must be a TaskStatus value.")
            if status not in self.ALLOWED_STATUS_TRANSITIONS[current.status]:
                raise InvalidTaskStatusError(
                    f"Transition from {current.status.value} to {status.value} is not allowed."
                )
            candidate = self._copy_task(current)
            candidate.status = status
            candidate.updated_at = self._next_timestamp(current.updated_at)
            self._tasks[task_id] = candidate
            return self._copy_task(candidate)

    def assign_task(self, task_id: str, agent_id: str) -> Task:
        """Set an executor ID and move an assignable task to ASSIGNED."""
        task_id = self._validate_task_id(task_id)
        if not isinstance(agent_id, str) or not agent_id.strip():
            raise TaskValidationError("agent_id must be a non-empty string.")
        with self._lock:
            current = self._get_internal(task_id)
            if current.status not in {
                TaskStatus.NEW,
                TaskStatus.READY,
                TaskStatus.ASSIGNED,
                TaskStatus.WAITING,
                TaskStatus.FAILED,
            }:
                raise InvalidTaskStatusError(
                    f"Task in status {current.status.value} cannot be assigned."
                )
            candidate = self._copy_task(current)
            candidate.assignee = agent_id
            candidate.status = TaskStatus.ASSIGNED
            candidate.updated_at = self._next_timestamp(current.updated_at)
            self._tasks[task_id] = candidate
            return self._copy_task(candidate)

    def add_dependency(self, task_id: str, dependency_id: str) -> Task:
        """Add a dependency if both tasks exist and the graph remains acyclic."""
        task_id = self._validate_task_id(task_id)
        dependency_id = self._validate_task_id(dependency_id)
        with self._lock:
            current = self._get_internal(task_id)
            self._get_internal(dependency_id)
            if task_id == dependency_id:
                raise DependencyError("A task cannot depend on itself.")
            if dependency_id in current.dependencies:
                raise DependencyError(
                    f"Task {task_id!r} already depends on {dependency_id!r}."
                )
            dependencies = [*current.dependencies, dependency_id]
            if self._would_create_dependency_cycle(task_id, dependencies):
                raise DependencyError("Adding this dependency would create a cycle.")
            candidate = self._copy_task(current)
            candidate.dependencies = dependencies
            candidate.__post_init__()
            candidate.updated_at = self._next_timestamp(current.updated_at)
            self._tasks[task_id] = candidate
            return self._copy_task(candidate)

    def get_dependencies(self, task_id: str) -> list[Task]:
        """Return detached dependency records in the task's declared order."""
        task = self.get_task(task_id)
        with self._lock:
            return [self._copy_task(self._get_internal(dep_id)) for dep_id in task.dependencies]

    def are_dependencies_completed(self, task_id: str) -> bool:
        """Return True only when every dependency is completed; empty is ready."""
        task = self.get_task(task_id)
        with self._lock:
            return all(
                self._get_internal(dep_id).status is TaskStatus.COMPLETED
                for dep_id in task.dependencies
            )

    def get_subtasks(self, parent_task_id: str) -> list[Task]:
        """Return all direct children of an existing parent task."""
        parent_task_id = self._validate_task_id(parent_task_id)
        with self._lock:
            self._get_internal(parent_task_id)
            return [
                self._copy_task(task)
                for task in self._tasks.values()
                if task.parent_task_id == parent_task_id
            ]

    def get_tasks_by_status(self, status: TaskStatus) -> list[Task]:
        """Return detached task copies matching one lifecycle status."""
        if not isinstance(status, TaskStatus):
            raise InvalidTaskStatusError("status must be a TaskStatus value.")
        with self._lock:
            return [self._copy_task(task) for task in self._tasks.values() if task.status is status]

    def get_all_tasks(self) -> list[Task]:
        """Return detached copies of all stored tasks in insertion order."""
        with self._lock:
            return [self._copy_task(task) for task in self._tasks.values()]

    def _validate_relationships(self, task: Task) -> None:
        if task.parent_task_id is not None:
            if task.parent_task_id not in self._tasks:
                raise TaskNotFoundError(f"Parent task {task.parent_task_id!r} was not found.")
            parent_id = task.parent_task_id
            visited = {task.task_id}
            while parent_id is not None:
                if parent_id in visited:
                    raise TaskValidationError("Parent task relationships cannot contain a cycle.")
                visited.add(parent_id)
                parent = self._tasks.get(parent_id)
                if parent is None:
                    raise TaskNotFoundError(f"Parent task {parent_id!r} was not found.")
                parent_id = parent.parent_task_id

        for dependency_id in task.dependencies:
            if dependency_id not in self._tasks:
                raise DependencyError(f"Dependency task {dependency_id!r} was not found.")
        if self._would_create_dependency_cycle(task.task_id, task.dependencies):
            raise DependencyError("Task dependencies cannot contain a cycle.")

    def _would_create_dependency_cycle(self, task_id: str, dependencies: list[str]) -> bool:
        pending = list(dependencies)
        visited: set[str] = set()
        while pending:
            current_id = pending.pop()
            if current_id == task_id:
                return True
            if current_id in visited:
                continue
            visited.add(current_id)
            current = self._tasks.get(current_id)
            if current is not None:
                pending.extend(current.dependencies)
        return False

    def _get_internal(self, task_id: str) -> Task:
        try:
            return self._tasks[task_id]
        except KeyError:
            raise TaskNotFoundError(f"Task {task_id!r} was not found.") from None

    @staticmethod
    def _validate_task_id(task_id: str) -> str:
        if not isinstance(task_id, str) or not task_id.strip():
            raise TaskValidationError("task_id must be a non-empty string.")
        return task_id

    @staticmethod
    def _copy_task(task: Task) -> Task:
        try:
            duplicate = copy.deepcopy(task)
            duplicate.__post_init__()
            return duplicate
        except TaskValidationError:
            raise
        except Exception as error:
            raise TaskValidationError("Task values must support safe copying.") from error

    @staticmethod
    def _next_timestamp(previous: datetime) -> datetime:
        now = _utc_now()
        if now <= previous:
            from datetime import timedelta

            return previous + timedelta(microseconds=1)
        return now


__all__ = [
    "DependencyError",
    "InvalidTaskStatusError",
    "Task",
    "TaskAlreadyExistsError",
    "TaskNotFoundError",
    "TaskPriority",
    "TaskStatus",
    "TaskStore",
    "TaskStoreError",
    "TaskType",
    "TaskValidationError",
]
