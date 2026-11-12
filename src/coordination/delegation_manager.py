"""Policy and message handling for bounded, task-scoped agent delegation."""

from __future__ import annotations

import copy
import threading
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Mapping
from uuid import uuid4

from src.communication import Message, MessageBus, MessageType
from src.coordination.task_store import (
    InvalidTaskStatusError,
    Task,
    TaskAlreadyExistsError,
    TaskNotFoundError,
    TaskPriority,
    TaskStatus,
    TaskStore,
    TaskStoreError,
    TaskType,
)
from src.logging import Logger


class DelegationError(Exception):
    """Base error for policy, execution, and delivery failures."""


class DelegationPermissionError(DelegationError):
    """Raised when an agent or direction is not authorized to delegate."""


class DelegationTaskError(DelegationError):
    """Raised when a parent or child task cannot be resolved or updated."""


class DelegationLimitError(DelegationError):
    """Raised when delegation depth or handoff count has been exhausted."""


class DelegationCycleError(DelegationError):
    """Raised when delegation would repeat an agent in the current ancestry."""


class InvalidHelpRequestError(DelegationError, ValueError):
    """Raised when a HELP_REQUEST does not match its assigned task."""


class DelegationResultError(DelegationError, ValueError):
    """Raised when a RESULT is not from the expected executor/requester pair."""


class DuplicateResultError(DelegationResultError):
    """Raised when a result for the same delegated task was already processed."""


class DelegationDeliveryError(DelegationError):
    """Raised when a delegation or result message cannot be delivered."""


class DelegationExecutionError(DelegationError):
    """Raised when the assigned agent cannot execute a help request."""


DEFAULT_DELEGATION_RULES: dict[str, frozenset[str]] = {
    "research_agent": frozenset({"coding_agent"}),
    "coding_agent": frozenset({"math_agent"}),
    "math_agent": frozenset(),
}

_ROLE_ALIASES = {
    "research": "research_agent",
    "research_agent": "research_agent",
    "coding": "coding_agent",
    "coding_agent": "coding_agent",
    "math": "math_agent",
    "math_agent": "math_agent",
}


class DelegationManager:
    """Create delegated child tasks and route HELP_REQUEST/RESULT messages.

    The manager is synchronous and process-local. It deliberately leaves parent
    task status unchanged when a child is delegated or completes.
    """

    MAX_DELEGATION_DEPTH = 3
    MAX_HANDOFFS = 5

    def __init__(
        self,
        task_store: TaskStore,
        message_bus: MessageBus,
        *,
        agents: Mapping[str, Any] | None = None,
        logger: Logger | Any | None = None,
        delegation_rules: Mapping[str, set[str] | frozenset[str]] | None = None,
        max_delegation_depth: int = MAX_DELEGATION_DEPTH,
        max_handoffs: int = MAX_HANDOFFS,
    ) -> None:
        """Initialize with shared task/message infrastructure and agent adapters."""
        if not isinstance(task_store, TaskStore):
            raise TypeError("task_store must be a TaskStore instance.")
        if not isinstance(message_bus, MessageBus):
            raise TypeError("message_bus must be a MessageBus instance.")
        if logger is not None and not callable(getattr(logger, "log_event", None)):
            raise TypeError("logger must provide log_event() or be None.")
        for name, value in (("max_delegation_depth", max_delegation_depth), ("max_handoffs", max_handoffs)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        self.task_store = task_store
        self.message_bus = message_bus
        self.logger = logger
        self.max_delegation_depth = max_delegation_depth
        self.max_handoffs = max_handoffs
        self.agents = dict(agents or {})
        raw_rules = delegation_rules or DEFAULT_DELEGATION_RULES
        self.delegation_rules = {
            self._role(source): frozenset(self._role(target) for target in targets)
            for source, targets in raw_rules.items()
        }
        self._lock = threading.RLock()
        self._handoffs_by_root: dict[str, int] = {}
        self._delegations: dict[str, dict[str, Any]] = {}
        self._request_keys: dict[str, str] = {}
        self._processed_help: set[str] = set()
        self._processed_results: set[str] = set()
        self._completed_result_tasks: set[str] = set()
        self._agent_roles: dict[str, str] = {}
        for role, agent in self.agents.items():
            canonical_role = self._role(role)
            agent_id = getattr(agent, "agent_id", None)
            if isinstance(agent_id, str) and agent_id.strip():
                self._agent_roles[agent_id] = canonical_role

    def request_help(
        self,
        *,
        parent_task_id: str,
        initiator_id: str,
        recipient_id: str,
        description: str,
        context: Any = None,
        request_id: str | None = None,
    ) -> Message:
        """Create and send one HELP_REQUEST for a child of ``parent_task_id``.

        ``request_id`` is an optional idempotency key. Reusing it for the same
        parent returns the original message without creating or sending again.
        """
        self._validate_nonempty(parent_task_id, "parent_task_id")
        self._validate_nonempty(initiator_id, "initiator_id")
        self._validate_nonempty(recipient_id, "recipient_id")
        self._validate_nonempty(description, "description")
        if request_id is not None:
            self._validate_nonempty(request_id, "request_id")
        root_id = ""
        child_id = str(uuid4())
        self._log("delegation_attempt", parent_task_id=parent_task_id,
                   initiator_id=initiator_id, recipient_id=recipient_id)
        with self._lock:
            dedupe_key = f"{parent_task_id}:{request_id}" if request_id is not None else None
            if dedupe_key is not None and dedupe_key in self._request_keys:
                child_id = self._request_keys[dedupe_key]
                existing = self._delegations[child_id]
                self._log("delegation_duplicate_request", parent_task_id=parent_task_id,
                          child_task_id=child_id, request_id=request_id)
                return copy.deepcopy(existing["help_message"])
            try:
                parent = self.task_store.get_task(parent_task_id)
                if parent.assignee != initiator_id:
                    raise DelegationPermissionError(
                        f"Agent {initiator_id!r} is not the assignee of parent task {parent_task_id!r}."
                    )
                if parent.status not in {
                    TaskStatus.ASSIGNED,
                    TaskStatus.RUNNING,
                    TaskStatus.WAITING,
                    TaskStatus.REVIEW,
                }:
                    raise DelegationPermissionError(
                        f"Task {parent_task_id!r} in status {parent.status.value} cannot delegate."
                    )
                self._require_registered(initiator_id, "initiator")
                self._require_registered(recipient_id, "recipient")
                source_role = self._agent_role(initiator_id)
                target_role = self._agent_role(recipient_id)
                if target_role not in self.delegation_rules.get(source_role, frozenset()):
                    raise DelegationPermissionError(
                        f"Delegation from {initiator_id!r} to {recipient_id!r} is not allowed."
                    )
                root_id, depth, ancestry = self._task_lineage(parent)
                child_depth = depth + 1
                if child_depth > self.max_delegation_depth:
                    raise DelegationLimitError(
                        f"Delegation depth {child_depth} exceeds maximum {self.max_delegation_depth}."
                    )
                handoffs = max(
                    self._handoffs_by_root.get(root_id, 0),
                    self._count_delegated_descendants(root_id),
                )
                if handoffs >= self.max_handoffs:
                    raise DelegationLimitError(
                        f"Root task {root_id!r} has reached {self.max_handoffs} handoffs."
                    )
                if target_role in ancestry or recipient_id in ancestry:
                    raise DelegationCycleError(
                        f"Agent {recipient_id!r} already appears in this task's delegation chain."
                    )

                child = Task(
                    task_id=child_id,
                    description=description.strip(),
                    task_type=self._task_type(target_role),
                    priority=parent.priority,
                    parent_task_id=parent_task_id,
                    delegation_depth=child_depth,
                )
                self.task_store.create_task(child)
                self._log("delegation_child_created", root_task_id=root_id,
                          parent_task_id=parent_task_id, child_task_id=child_id)
                self.task_store.assign_task(child_id, recipient_id)
                self.task_store.update_status(child_id, TaskStatus.WAITING)
                payload = {
                    "description": description.strip(),
                    "context": copy.deepcopy(context),
                    "parent_task_id": parent_task_id,
                    "child_task_id": child_id,
                    "initiator_id": initiator_id,
                    "request_id": request_id,
                }
                message = Message(
                    sender=initiator_id,
                    receiver=recipient_id,
                    message_type=MessageType.HELP_REQUEST,
                    task_id=child_id,
                    parent_task_id=parent_task_id,
                    payload=payload,
                )
                record = {
                    "root_task_id": root_id,
                    "parent_task_id": parent_task_id,
                    "initiator_id": initiator_id,
                    "recipient_id": recipient_id,
                    "help_message": message,
                    "request_id": request_id,
                }
                self._delegations[child_id] = record
                if dedupe_key is not None:
                    self._request_keys[dedupe_key] = child_id
                try:
                    self.message_bus.send(message)
                except Exception as error:
                    self._fail_child(child_id, error)
                    self._log("delegation_delivery_failed", child_task_id=child_id,
                              error_type=type(error).__name__, error_message=str(error))
                    raise DelegationDeliveryError(
                        f"Could not deliver HELP_REQUEST for child task {child_id!r}."
                    ) from error
                self._handoffs_by_root[root_id] = self._handoffs_by_root.get(root_id, 0) + 1
                self._log("help_request_sent", root_task_id=root_id,
                          parent_task_id=parent_task_id, child_task_id=child_id,
                          initiator_id=initiator_id, recipient_id=recipient_id)
                self._log("delegation_succeeded", root_task_id=root_id,
                          parent_task_id=parent_task_id, child_task_id=child_id,
                          depth=child_depth, handoffs=self._handoffs_by_root[root_id])
                return copy.deepcopy(message)
            except (DelegationError, TaskNotFoundError) as error:
                self._log("delegation_rejected", parent_task_id=parent_task_id,
                          initiator_id=initiator_id, recipient_id=recipient_id,
                          error_type=type(error).__name__, error_message=str(error))
                if isinstance(error, TaskNotFoundError):
                    raise DelegationTaskError(str(error)) from error
                raise
            except TaskStoreError as error:
                self._log("delegation_failed", parent_task_id=parent_task_id,
                          error_type=type(error).__name__, error_message=str(error))
                if child_id in self._delegations:
                    self._fail_child(child_id, error)
                raise DelegationError(f"Could not create delegated task: {error}") from error

    def receive_and_handle(self, agent_id: str) -> Message | None:
        """Receive one inbox message and handle HELP_REQUEST or RESULT."""
        message = self.message_bus.receive(agent_id)
        if message is None:
            return None
        return self.handle_message(message)

    def handle_message(self, message: Message) -> Message | None:
        """Dispatch one inbox message to its supported delegation handler."""
        if not isinstance(message, Message):
            raise TypeError("message must be a Message instance.")
        if message.message_type is MessageType.HELP_REQUEST:
            return self.handle_help_request(message)
        if message.message_type is MessageType.RESULT:
            self.handle_result(message)
            return None
        raise DelegationError(f"Unsupported delegation message type {message.message_type.value!r}.")

    def handle_help_request(self, message: Message) -> Message:
        """Execute an assigned child task and send its outcome as RESULT."""
        self._validate_help_message(message)
        with self._lock:
            if message.message_id in self._processed_help:
                self._log("duplicate_help_request", message_id=message.message_id,
                          child_task_id=message.task_id)
                raise InvalidHelpRequestError("HELP_REQUEST has already been processed.")
            task = self.task_store.get_task(message.task_id)
            record = self._delegations.get(task.task_id)
            if record is None:
                raise InvalidHelpRequestError("HELP_REQUEST does not reference a registered delegation.")
            if record["recipient_id"] != message.receiver or record["initiator_id"] != message.sender:
                raise InvalidHelpRequestError("HELP_REQUEST sender or receiver does not match delegation.")
            if task.assignee != message.receiver:
                raise InvalidHelpRequestError("HELP_REQUEST receiver is not the assigned executor.")
            if task.parent_task_id != message.parent_task_id:
                raise InvalidHelpRequestError("HELP_REQUEST parent_task_id does not match the child task.")
            if task.status is not TaskStatus.WAITING:
                raise InvalidHelpRequestError("Delegated task is not waiting for execution.")
            if message.sender != self.task_store.get_task(task.parent_task_id or "").assignee:
                raise InvalidHelpRequestError("HELP_REQUEST sender is not the parent task assignee.")
            agent = self._get_agent(message.receiver)
            self._processed_help.add(message.message_id)
            self.task_store.update_status(task.task_id, TaskStatus.RUNNING)
            self._log("help_request_received", message_id=message.message_id,
                      child_task_id=task.task_id, receiver=message.receiver)

        try:
            result_value = self._execute_agent(agent, message.payload["description"],
                                               message.payload.get("context"))
            success = self._execution_succeeded(result_value)
            error_text = self._execution_error(result_value) if not success else None
            result_payload = {
                "success": success,
                "result": self._json_safe(result_value) if success else None,
                "error": error_text,
                "child_task_id": task.task_id,
                "parent_task_id": task.parent_task_id,
                "initiator_id": message.sender,
            }
        except Exception as error:
            result_payload = {
                "success": False,
                "result": None,
                "error": f"{type(error).__name__}: {error}",
                "child_task_id": task.task_id,
                "parent_task_id": task.parent_task_id,
                "initiator_id": message.sender,
            }
            self._log("delegated_execution_failed", child_task_id=task.task_id,
                      error_type=type(error).__name__, error_message=str(error))

        result_message = Message(
            sender=message.receiver,
            receiver=message.sender,
            message_type=MessageType.RESULT,
            task_id=task.task_id,
            parent_task_id=task.parent_task_id,
            payload=result_payload,
        )
        try:
            self.message_bus.send(result_message)
        except Exception as error:
            self._fail_child(task.task_id, error)
            self._log("result_delivery_failed", child_task_id=task.task_id,
                      error_type=type(error).__name__, error_message=str(error))
            raise DelegationDeliveryError(
                f"Could not deliver RESULT for child task {task.task_id!r}."
            ) from error
        self._log("result_sent", child_task_id=task.task_id,
                  sender=message.receiver, receiver=message.sender,
                  success=result_payload["success"])
        return copy.deepcopy(result_message)

    def handle_result(self, message: Message) -> Task:
        """Validate a RESULT, update the child, and retain it for its parent."""
        if not isinstance(message, Message) or message.message_type is not MessageType.RESULT:
            raise DelegationResultError("message must be a RESULT Message.")
        if not isinstance(message.payload, dict) or not isinstance(message.payload.get("success"), bool):
            raise DelegationResultError("RESULT payload must contain a boolean success field.")
        with self._lock:
            if message.message_id in self._processed_results or message.task_id in self._completed_result_tasks:
                self._log("duplicate_result", message_id=message.message_id,
                          child_task_id=message.task_id)
                raise DuplicateResultError(f"RESULT for task {message.task_id!r} was already processed.")
            try:
                child = self.task_store.get_task(message.task_id)
            except TaskNotFoundError as error:
                raise DelegationResultError("RESULT references an unknown child task.") from error
            parent_id = child.parent_task_id
            if parent_id is None or message.parent_task_id != parent_id:
                raise DelegationResultError("RESULT parent_task_id does not match the delegated task.")
            try:
                parent = self.task_store.get_task(parent_id)
            except TaskNotFoundError as error:
                raise DelegationResultError("RESULT parent task no longer exists.") from error
            if child.assignee != message.sender:
                raise DelegationResultError("RESULT sender is not the assigned child executor.")
            record = self._delegations.get(child.task_id)
            if record is None:
                raise DelegationResultError("RESULT task was not created by this DelegationManager.")
            if record["recipient_id"] != message.sender:
                raise DelegationResultError("RESULT sender does not match the registered delegation.")
            if record["initiator_id"] != message.receiver:
                raise DelegationResultError("RESULT receiver is not the original delegating agent.")
            if parent.assignee != message.receiver:
                raise DelegationResultError("RESULT receiver is not the delegating parent executor.")
            if message.payload.get("initiator_id") != message.receiver:
                raise DelegationResultError("RESULT initiator does not match its receiver.")
            if message.payload.get("child_task_id", message.task_id) != message.task_id:
                raise DelegationResultError("RESULT child_task_id does not match task_id.")
            if child.status is not TaskStatus.RUNNING:
                raise DelegationResultError("Delegated task is not running and cannot accept a result.")
            success = message.payload["success"]
            result = copy.deepcopy(message.payload.get("result"))
            error_text = message.payload.get("error")
            try:
                self.task_store.update_task(
                    child.task_id,
                    result=result,
                    error=None if success else str(error_text or "Delegated task failed."),
                )
                updated = self.task_store.update_status(
                    child.task_id,
                    TaskStatus.COMPLETED if success else TaskStatus.FAILED,
                )
            except TaskStoreError as error:
                raise DelegationResultError(f"Could not update delegated task: {error}") from error
            self._processed_results.add(message.message_id)
            self._completed_result_tasks.add(child.task_id)
            self._log("result_received", message_id=message.message_id,
                      child_task_id=child.task_id, sender=message.sender,
                      receiver=message.receiver, success=success)
            if success:
                self._log("delegated_task_completed", child_task_id=child.task_id,
                          parent_task_id=parent_id)
            else:
                self._log("delegated_task_failed", child_task_id=child.task_id,
                          parent_task_id=parent_id, error_message=error_text)
            return updated

    def _validate_help_message(self, message: Message) -> None:
        if not isinstance(message, Message) or message.message_type is not MessageType.HELP_REQUEST:
            raise InvalidHelpRequestError("message must be a HELP_REQUEST Message.")
        payload = message.payload
        if not isinstance(payload, dict):
            raise InvalidHelpRequestError("HELP_REQUEST payload must be a dictionary.")
        required = ("description", "parent_task_id", "child_task_id", "initiator_id")
        if any(not isinstance(payload.get(field), str) or not payload[field].strip() for field in required):
            raise InvalidHelpRequestError("HELP_REQUEST payload is missing required fields.")
        if payload["child_task_id"] != message.task_id:
            raise InvalidHelpRequestError("HELP_REQUEST child_task_id does not match task_id.")
        if payload["parent_task_id"] != message.parent_task_id:
            raise InvalidHelpRequestError("HELP_REQUEST parent_task_id does not match the envelope.")
        if payload["initiator_id"] != message.sender:
            raise InvalidHelpRequestError("HELP_REQUEST initiator_id does not match sender.")

    def _task_lineage(self, parent: Task) -> tuple[str, int, set[str]]:
        chain: list[Task] = [parent]
        visited = {parent.task_id}
        current = parent
        while current.parent_task_id is not None:
            parent_id = current.parent_task_id
            if parent_id in visited:
                raise DelegationCycleError("Task parent chain contains a cycle.")
            visited.add(parent_id)
            current = self.task_store.get_task(parent_id)
            chain.append(current)
        ancestry: set[str] = set()
        for item in chain:
            if item.assignee:
                ancestry.add(item.assignee)
                ancestry.add(self._agent_role(item.assignee))
        return current.task_id, max(len(chain) - 1, parent.delegation_depth), ancestry

    def _count_delegated_descendants(self, root_id: str) -> int:
        """Count persisted delegated tasks so a new manager cannot reset handoff caps."""
        count = 0
        for task in self.task_store.get_all_tasks():
            if task.delegation_depth <= 0:
                continue
            current = task
            visited: set[str] = set()
            belongs_to_root = False
            while current.task_id not in visited:
                if current.task_id == root_id:
                    belongs_to_root = True
                    break
                visited.add(current.task_id)
                if current.parent_task_id is None:
                    break
                try:
                    current = self.task_store.get_task(current.parent_task_id)
                except TaskNotFoundError:
                    break
            if belongs_to_root:
                count += 1
        return count

    def _require_registered(self, agent_id: str, label: str) -> None:
        if agent_id not in self.message_bus.get_registered_agents():
            raise DelegationPermissionError(f"Unknown {label} agent {agent_id!r}.")

    def _agent_role(self, agent_id: str) -> str:
        if agent_id in self._agent_roles:
            return self._agent_roles[agent_id]
        return self._role(agent_id)

    def _get_agent(self, agent_id: str) -> Any:
        if agent_id in self.agents:
            return self.agents[agent_id]
        for key, agent in self.agents.items():
            if getattr(agent, "agent_id", None) == agent_id:
                return agent
            if self._role(key) == self._agent_role(agent_id):
                return agent
        raise InvalidHelpRequestError(f"No executor is registered for agent {agent_id!r}.")

    @staticmethod
    def _role(value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Agent roles and IDs must be non-empty strings.")
        return _ROLE_ALIASES.get(value.strip(), value.strip())

    @staticmethod
    def _task_type(role: str) -> TaskType:
        return {
            "research_agent": TaskType.RESEARCH,
            "coding_agent": TaskType.CODING,
            "math_agent": TaskType.MATH,
        }.get(role, TaskType.GENERAL)

    @staticmethod
    def _validate_nonempty(value: str, field: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a non-empty string.")

    @staticmethod
    def _execute_agent(agent: Any, description: str, context: Any) -> Any:
        if hasattr(agent, "run_research"):
            task = description if context is None else f"{description}\n\nКонтекст:\n{context}"
            return agent.run_research(task)
        if hasattr(agent, "solve"):
            task = description if context is None else f"{description}\n\nКонтекст:\n{context}"
            return agent.solve(task)
        run = getattr(agent, "run", None)
        if not callable(run):
            raise TypeError("Agent must provide run_research(), solve(), or run().")
        if context is None:
            return run(description)
        return run(description, additional_context=str(context))

    @staticmethod
    def _execution_succeeded(result: Any) -> bool:
        return getattr(result, "success", True) is not False

    @staticmethod
    def _execution_error(result: Any) -> str:
        return str(getattr(result, "error", None) or "Agent reported unsuccessful execution.")

    @classmethod
    def _json_safe(cls, value: Any) -> Any:
        if is_dataclass(value) and not isinstance(value, type):
            return cls._json_safe(asdict(value))
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, dict):
            return {str(key): cls._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [cls._json_safe(item) for item in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return str(value)

    def _fail_child(self, child_id: str, error: Exception) -> None:
        try:
            child = self.task_store.get_task(child_id)
            if child.status not in {TaskStatus.FAILED, TaskStatus.COMPLETED, TaskStatus.CANCELLED}:
                self.task_store.update_task(child_id, error=f"{type(error).__name__}: {error}")
                self.task_store.update_status(child_id, TaskStatus.FAILED)
        except (TaskStoreError, InvalidTaskStatusError) as update_error:
            self._log("delegation_failure_state_error", child_task_id=child_id,
                      error_type=type(update_error).__name__, error_message=str(update_error))

    def _log(self, event_type: str, **fields: Any) -> None:
        if self.logger is None:
            return
        try:
            self.logger.log_event(event_type, **fields)
        except Exception:
            # Logger failures are deliberately best-effort, as in MessageBus.
            return


__all__ = [
    "DEFAULT_DELEGATION_RULES",
    "DelegationCycleError",
    "DelegationDeliveryError",
    "DelegationError",
    "DelegationExecutionError",
    "DelegationLimitError",
    "DelegationPermissionError",
    "DelegationResultError",
    "DelegationTaskError",
    "DelegationManager",
    "DuplicateResultError",
    "InvalidHelpRequestError",
]
