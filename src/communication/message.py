"""Typed message envelope shared by future multi-agent communication layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


class MessageType(str, Enum):
    """Supported message kinds exchanged by system components."""

    TASK_ASSIGN = "TASK_ASSIGN"
    TASK_ACCEPT = "TASK_ACCEPT"
    TASK_REJECT = "TASK_REJECT"
    HELP_REQUEST = "HELP_REQUEST"
    RESULT = "RESULT"
    REVIEW_REQUEST = "REVIEW_REQUEST"
    REVIEW_FEEDBACK = "REVIEW_FEEDBACK"
    TASK_UPDATE = "TASK_UPDATE"
    TASK_CANCEL = "TASK_CANCEL"
    TASK_BID = "TASK_BID"
    AUCTION_CALL = "AUCTION_CALL"
    AUCTION_AWARD = "AUCTION_AWARD"
    CONTRACT_AWARD = "CONTRACT_AWARD"
    NEGOTIATION_PROPOSAL = "NEGOTIATION_PROPOSAL"
    NEGOTIATION_COUNTER = "NEGOTIATION_COUNTER"
    NEGOTIATION_ACCEPT = "NEGOTIATION_ACCEPT"
    NEGOTIATION_REJECT = "NEGOTIATION_REJECT"
    ARBITRATION_DECISION = "ARBITRATION_DECISION"
    SECURITY_INCIDENT = "SECURITY_INCIDENT"
    ERROR = "ERROR"


class MessageValidationError(ValueError):
    """Raised when a message or its serialized representation is invalid."""


@dataclass
class Message:
    """A validated, timestamped envelope for one task-related message."""

    sender: str
    receiver: str
    message_type: MessageType
    task_id: str
    payload: dict[str, Any]
    message_id: str = field(default_factory=lambda: str(uuid4()))
    parent_task_id: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        """Validate values supplied directly or restored from serialized data."""
        for field_name in ("message_id", "sender", "receiver", "task_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise MessageValidationError(f"{field_name} must be a non-empty string.")
        if not isinstance(self.message_type, MessageType):
            raise MessageValidationError("message_type must be a MessageType value.")
        if self.parent_task_id is not None and (
            not isinstance(self.parent_task_id, str) or not self.parent_task_id.strip()
        ):
            raise MessageValidationError("parent_task_id must be a non-empty string or None.")
        if not isinstance(self.payload, dict):
            raise MessageValidationError("payload must be a dictionary.")
        if not isinstance(self.timestamp, datetime):
            raise MessageValidationError("timestamp must be a datetime instance.")
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise MessageValidationError("timestamp must be timezone-aware.")

    def to_dict(self) -> dict[str, Any]:
        """Return all message fields in a JSON-compatible representation."""
        return {
            "message_id": self.message_id,
            "sender": self.sender,
            "receiver": self.receiver,
            "message_type": self.message_type.value,
            "task_id": self.task_id,
            "parent_task_id": self.parent_task_id,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        """Restore and validate a message from its serialized dictionary."""
        if not isinstance(data, dict):
            raise MessageValidationError("Serialized message must be a dictionary.")
        required_fields = {
            "message_id",
            "sender",
            "receiver",
            "message_type",
            "task_id",
            "parent_task_id",
            "payload",
            "timestamp",
        }
        missing = required_fields - data.keys()
        if missing:
            raise MessageValidationError(
                "Serialized message is missing required field(s): "
                + ", ".join(sorted(missing))
                + "."
            )

        raw_type = data["message_type"]
        if isinstance(raw_type, MessageType):
            message_type = raw_type
        elif isinstance(raw_type, str):
            try:
                message_type = MessageType(raw_type)
            except ValueError as error:
                raise MessageValidationError(
                    f"Unknown message_type value {raw_type!r}."
                ) from error
        else:
            raise MessageValidationError("message_type must be a MessageType or string value.")

        raw_timestamp = data["timestamp"]
        if isinstance(raw_timestamp, datetime):
            timestamp = raw_timestamp
        elif isinstance(raw_timestamp, str):
            try:
                timestamp = datetime.fromisoformat(raw_timestamp)
            except ValueError as error:
                raise MessageValidationError("timestamp must be a valid ISO 8601 datetime string.") from error
        else:
            raise MessageValidationError("timestamp must be an ISO 8601 string or datetime.")

        return cls(
            message_id=data["message_id"],
            sender=data["sender"],
            receiver=data["receiver"],
            message_type=message_type,
            task_id=data["task_id"],
            parent_task_id=data["parent_task_id"],
            payload=data["payload"],
            timestamp=timestamp,
        )


__all__ = ["Message", "MessageType", "MessageValidationError"]
