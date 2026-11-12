"""Policy checks for message authority, task actions, and instruction conflicts.

Structural validation remains the responsibility of ``Message``. This policy
adds principal/role permissions, payload contract validation, action scopes,
and a deterministic instruction precedence rule.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from src.communication import Message, MessageType


class SecurityPolicyError(Exception):
    """Base class for denied or malformed security-policy decisions."""


class InstructionConflictError(SecurityPolicyError):
    """Raised when equally trusted instruction sources contradict each other."""


@dataclass(frozen=True)
class PolicyDecision:
    """One explicit authorization outcome with a machine-readable reason."""

    allowed: bool
    actor_id: str
    action: str
    task_id: str
    reason: str


@dataclass(frozen=True)
class SecurityIncident:
    """A minimal incident record that avoids storing untrusted content bodies."""

    incident_type: str
    actor_id: str | None
    task_id: str | None
    reason: str
    blocked: bool


@dataclass(frozen=True)
class InstructionResolution:
    """Winning instruction and provenance for suppressed lower-priority text."""

    text: str
    source: str
    suppressed_sources: tuple[str, ...]


class SecurityPolicy:
    """Enforce allowlists and fixed source precedence without interpreting LLM text."""

    INSTRUCTION_PRECEDENCE = {
        "system": 0,
        "security_policy": 1,
        "task_owner": 2,
        "delegated_context": 3,
        "agent_output": 4,
        "untrusted_data": 5,
    }

    def __init__(
        self,
        *,
        principals: Mapping[str, str],
        allowed_message_types: Mapping[str, Iterable[MessageType | str]],
        allowed_recipient_roles: Mapping[str, Iterable[str]],
        payload_schemas: Mapping[MessageType | str, Mapping[str, type]] | None = None,
        allowed_actions: Mapping[str, Iterable[str]],
        task_scopes: Mapping[str, Iterable[str]] | None = None,
        max_payload_bytes: int = 16_384,
        logger: Any | None = None,
    ) -> None:
        if not isinstance(principals, Mapping) or not principals:
            raise ValueError("principals must map agent IDs to roles.")
        self.principals = dict(principals)
        if any(not isinstance(agent, str) or not agent.strip() or
               not isinstance(role, str) or not role.strip()
               for agent, role in self.principals.items()):
            raise ValueError("Principal IDs and roles must be non-empty strings.")
        self.allowed_message_types = {
            role: frozenset(self._message_type(value) for value in values)
            for role, values in allowed_message_types.items()
        }
        self.allowed_recipient_roles = {
            role: frozenset(values) for role, values in allowed_recipient_roles.items()
        }
        self.allowed_actions = {role: frozenset(values) for role, values in allowed_actions.items()}
        self.task_scopes = {task: frozenset(actions) for task, actions in (task_scopes or {}).items()}
        self.payload_schemas = {
            self._message_type(key): dict(value) for key, value in (payload_schemas or {}).items()
        }
        if isinstance(max_payload_bytes, bool) or not isinstance(max_payload_bytes, int) or max_payload_bytes < 1:
            raise ValueError("max_payload_bytes must be a positive integer.")
        if logger is not None and not callable(getattr(logger, "log_event", None)):
            raise TypeError("logger must provide log_event() or be None.")
        self.max_payload_bytes = max_payload_bytes
        self.logger = logger
        self._incidents: list[SecurityIncident] = []
        self._lock = threading.RLock()

    @property
    def incidents(self) -> tuple[SecurityIncident, ...]:
        """Return a consistent snapshot of recorded policy incidents."""
        with self._lock:
            return tuple(self._incidents)

    def authorize_message(self, message: Message) -> None:
        """Validate structural type, principal authority, destination and payload."""
        if not isinstance(message, Message):
            self._deny("invalid_message", None, None, "Expected a structurally validated Message.")
        sender_role = self.principals.get(message.sender)
        receiver_role = self.principals.get(message.receiver)
        if sender_role is None:
            self._deny("unknown_principal", message.sender, message.task_id, "Sender has no configured principal role.")
        if receiver_role is None:
            self._deny("unknown_principal", message.sender, message.task_id, "Receiver has no configured principal role.")
        if message.message_type not in self.allowed_message_types.get(sender_role, frozenset()):
            self._deny("message_authority", message.sender, message.task_id,
                       f"Role {sender_role!r} may not send {message.message_type.value}.")
        if receiver_role not in self.allowed_recipient_roles.get(sender_role, frozenset()):
            self._deny("recipient_authority", message.sender, message.task_id,
                       f"Role {sender_role!r} may not send to role {receiver_role!r}.")
        schema = self.payload_schemas.get(message.message_type)
        if schema is not None:
            for field, expected_type in schema.items():
                value = message.payload.get(field)
                if not isinstance(value, expected_type) or isinstance(value, bool) and expected_type is int:
                    self._deny("payload_contract", message.sender, message.task_id,
                               f"Payload field {field!r} is missing or has the wrong type.")
        if message.payload.get("initiator_id") not in (None, message.sender):
            self._deny("payload_identity_mismatch", message.sender, message.task_id,
                       "Payload initiator does not match the authenticated sender.")
        if message.payload.get("task_id") not in (None, message.task_id):
            self._deny("payload_task_mismatch", message.sender, message.task_id,
                       "Payload task_id does not match the message envelope.")
        try:
            serialized = json.dumps(message.payload, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError):
            self._deny("payload_serialization", message.sender, message.task_id,
                       "Payload is not safe JSON data.")
        if len(serialized.encode("utf-8")) > self.max_payload_bytes:
            self._deny("payload_size", message.sender, message.task_id,
                       "Payload exceeds the configured size limit.")

    def authorize_action(self, *, agent_id: str, task_id: str, action: str) -> PolicyDecision:
        """Check an action against both the agent role and the task's explicit scope."""
        role = self.principals.get(agent_id)
        if role is None:
            return self._deny("unknown_principal", agent_id, task_id,
                              "Agent has no configured principal role.")
        if action not in self.allowed_actions.get(role, frozenset()):
            return self._deny("action_authority", agent_id, task_id,
                              f"Role {role!r} is not allowed to perform action {action!r}.")
        scope = self.task_scopes.get(task_id)
        if scope is not None and action not in scope:
            return self._deny("task_scope", agent_id, task_id,
                              f"Action {action!r} is outside the task's authorized scope.")
        return PolicyDecision(True, agent_id, action, task_id, "role and task scope allow the action")

    def resolve_instructions(self, instructions: Sequence[Mapping[str, str]], *,
                             actor_id: str | None = None,
                             task_id: str | None = None) -> InstructionResolution:
        """Choose by fixed provenance hierarchy; reject conflicts at equal trust."""
        if not instructions:
            raise ValueError("instructions must contain at least one entry.")
        normalized: list[tuple[int, int, str, str]] = []
        for index, instruction in enumerate(instructions):
            source, text = instruction.get("source"), instruction.get("text")
            if (not isinstance(source, str) or source not in self.INSTRUCTION_PRECEDENCE
                    or not isinstance(text, str) or not text.strip()):
                self._deny("instruction_contract", actor_id, task_id,
                           "Instruction source or text is invalid.")
            normalized.append((self.INSTRUCTION_PRECEDENCE[source], index, source, text.strip()))
        normalized.sort()
        top_priority = normalized[0][0]
        highest = [item for item in normalized if item[0] == top_priority]
        distinct = {item[3] for item in highest}
        if len(distinct) > 1:
            self._record_incident("instruction_conflict", actor_id, task_id,
                                   "Equally trusted instruction sources disagree.", blocked=True)
            raise InstructionConflictError("Conflicting instructions have equal precedence.")
        winner = highest[0]
        if any(item[3] != winner[3] for item in normalized if item[0] > top_priority):
            self._record_incident("instruction_conflict", actor_id, task_id,
                "A lower-precedence instruction contradicted the selected trusted instruction.",
                blocked=True)
        return InstructionResolution(winner[3], winner[2],
            tuple(item[2] for item in normalized if item[1] != winner[1]))

    def _deny(self, incident_type: str, actor_id: str | None,
              task_id: str | None, reason: str) -> Any:
        self._record_incident(incident_type, actor_id, task_id, reason, blocked=True)
        raise SecurityPolicyError(reason)

    def _record_incident(self, incident_type: str, actor_id: str | None,
                         task_id: str | None, reason: str, *, blocked: bool) -> None:
        incident = SecurityIncident(incident_type, actor_id, task_id, reason, blocked)
        with self._lock:
            self._incidents.append(incident)
        if self.logger is not None:
            try:
                self.logger.log_event("security_incident", incident_type=incident_type,
                    actor_id=actor_id, task_id=task_id, reason=reason, blocked=blocked)
            except Exception:
                pass

    @staticmethod
    def _message_type(value: MessageType | str) -> MessageType:
        if isinstance(value, MessageType):
            return value
        try:
            return MessageType(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Unknown message type {value!r} in security policy.") from error


__all__ = ["InstructionConflictError", "InstructionResolution", "PolicyDecision",
           "SecurityIncident", "SecurityPolicy", "SecurityPolicyError"]
