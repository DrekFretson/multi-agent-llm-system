"""In-memory synchronous message delivery between registered agent IDs."""

from __future__ import annotations

import threading
from collections import deque
from typing import Protocol

from src.communication.message import Message
from src.logging import Logger


class MessageBusError(Exception):
    """Base exception for local Message Bus errors."""


class AgentRegistrationError(MessageBusError):
    """Raised when agent registration, removal, or inbox access is invalid."""


class UnknownSenderError(MessageBusError):
    """Raised when a message sender is not registered on the bus."""


class UnknownReceiverError(MessageBusError):
    """Raised when a message receiver is not registered on the bus."""


class MessageDeliveryError(MessageBusError):
    """Raised when a technical error prevents adding a message to an inbox."""


class MessageAuthorizationError(MessageBusError):
    """Raised when an optional authorization policy blocks message delivery."""


class _EventLogger(Protocol):
    def log_event(self, event_type: str, **fields: object) -> object: ...


class MessageBus:
    """Register agent inboxes and deliver messages directly in FIFO order.

    The bus is process-local and synchronous. If an agent's inbox is not empty,
    unregistering it is rejected so queued messages cannot be silently lost.
    Logging is best-effort: a logger failure is recorded on
    :attr:`last_logging_error` but never changes delivery status or triggers a
    retry. Pass the existing project :class:`Logger` to persist event metadata;
    without one, delivery remains available and no event file is written.
    """

    def __init__(self, logger: _EventLogger | None = None, *, authorizer: object | None = None) -> None:
        """Create a bus with optional event logging and a message policy hook."""
        if logger is not None and not callable(getattr(logger, "log_event", None)):
            raise TypeError("logger must provide a callable log_event() method or be None.")
        self._logger = logger
        if authorizer is not None and not callable(authorizer):
            raise TypeError("authorizer must be callable or None.")
        self._authorizer = authorizer
        self._inboxes: dict[str, deque[Message]] = {}
        self._lock = threading.RLock()
        self._last_logging_error: Exception | None = None

    @property
    def last_logging_error(self) -> Exception | None:
        """Return the most recent swallowed logger failure, if any."""
        return self._last_logging_error

    def register_agent(self, agent_id: str) -> None:
        """Register an agent and create its empty incoming FIFO queue."""
        self._validate_agent_id(agent_id)
        with self._lock:
            if agent_id in self._inboxes:
                raise AgentRegistrationError(f"Agent {agent_id!r} is already registered.")
            self._inboxes[agent_id] = deque()

    def unregister_agent(self, agent_id: str) -> None:
        """Remove an agent only when its inbox is empty.

        Pending messages are retained by refusing removal; callers must receive
        or otherwise handle each message before unregistering the agent.
        """
        self._validate_agent_id(agent_id)
        with self._lock:
            if agent_id not in self._inboxes:
                raise AgentRegistrationError(f"Agent {agent_id!r} is not registered.")
            if self._inboxes[agent_id]:
                raise AgentRegistrationError(
                    f"Agent {agent_id!r} has unread messages; receive them before unregistering."
                )
            del self._inboxes[agent_id]

    def get_registered_agents(self) -> list[str]:
        """Return registered identifiers in registration order."""
        with self._lock:
            return list(self._inboxes)

    def send(self, message: Message) -> None:
        """Deliver one message to its addressed receiver, without broadcasting."""
        if not isinstance(message, Message):
            raise MessageBusError("message must be a Message instance.")
        # Message is mutable, so revalidate before it enters the queue.
        message.__post_init__()

        if self._authorizer is not None:
            try:
                self._authorizer(message)
            except Exception as cause:
                blocked = MessageAuthorizationError(
                    f"Message {message.message_id!r} was blocked by authorization policy."
                )
                self._log_delivery(message, "blocked", blocked, cause)
                raise blocked from cause

        failure: MessageBusError | None = None
        delivery_cause: Exception | None = None
        with self._lock:
            if message.sender not in self._inboxes:
                failure = UnknownSenderError(
                    f"Message sender {message.sender!r} is not registered."
                )
            elif message.receiver not in self._inboxes:
                failure = UnknownReceiverError(
                    f"Message receiver {message.receiver!r} is not registered."
                )
            else:
                try:
                    self._inboxes[message.receiver].append(message)
                except Exception as error:
                    delivery_cause = error
                    failure = MessageDeliveryError(
                        f"Could not deliver message {message.message_id!r} to {message.receiver!r}."
                    )

        if failure is not None:
            self._log_delivery(message, "failed", failure, delivery_cause)
            if delivery_cause is not None:
                raise failure from delivery_cause
            raise failure

        # The message is already in the recipient's queue. A logger failure is
        # intentionally isolated and must never turn this into a failed send.
        self._log_delivery(message, "delivered", None)

    def receive(self, agent_id: str) -> Message | None:
        """Return and remove the oldest queued message, or None immediately."""
        self._validate_agent_id(agent_id)
        with self._lock:
            try:
                inbox = self._inboxes[agent_id]
            except KeyError:
                raise AgentRegistrationError(f"Agent {agent_id!r} is not registered.") from None
            return inbox.popleft() if inbox else None

    @staticmethod
    def _validate_agent_id(agent_id: str) -> None:
        if not isinstance(agent_id, str) or not agent_id.strip():
            raise AgentRegistrationError("agent_id must be a non-empty string.")

    def _log_delivery(
        self,
        message: Message,
        delivery_status: str,
        error: MessageBusError | None,
        cause: Exception | None = None,
    ) -> None:
        try:
            from src.evaluation.metrics import record_current_message

            record_current_message(
                task_id=message.task_id, message_id=message.message_id,
                sender=message.sender, receiver=message.receiver,
                message_type=message.message_type.value, delivery_status=delivery_status,
                error_type=type(error).__name__ if error is not None else None,
                error_message=str(error) if error is not None else None,
            )
        except Exception:
            # Metrics are observational and cannot affect message delivery.
            pass
        if self._logger is None:
            return
        fields: dict[str, object] = {
            "message_id": message.message_id,
            "sender": message.sender,
            "receiver": message.receiver,
            "message_type": message.message_type.value,
            "task_id": message.task_id,
            "parent_task_id": message.parent_task_id,
            "delivery_status": delivery_status,
            "error_type": type(error).__name__ if error is not None else None,
            "error_message": str(error) if error is not None else None,
            "cause_type": type(cause).__name__ if cause is not None else None,
            "cause_message": str(cause) if cause is not None else None,
        }
        try:
            from src.evaluation.metrics import current_metric_context

            context = current_metric_context()
            if context is not None:
                fields.update({"run_id": context.run_id,
                               "experiment_id": context.experiment_id,
                               "agent_role": context.agent_role})
        except Exception:
            pass
        try:
            self._logger.log_event("message_delivery", **fields)
        except Exception as logging_error:
            self._last_logging_error = logging_error
        else:
            self._last_logging_error = None


__all__ = [
    "AgentRegistrationError",
    "MessageAuthorizationError",
    "MessageBus",
    "MessageBusError",
    "MessageDeliveryError",
    "UnknownReceiverError",
    "UnknownSenderError",
]
