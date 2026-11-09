"""Validation and serialization tests for communication messages."""

from datetime import datetime, timedelta, timezone

import pytest

from src.communication import Message, MessageType, MessageValidationError


def make_message(**overrides) -> Message:
    values = {
        "sender": "agent-a",
        "receiver": "agent-b",
        "message_type": MessageType.TASK_ASSIGN,
        "task_id": "task-1",
        "payload": {"instruction": "summarize", "attempt": 1},
    }
    values.update(overrides)
    return Message(**values)


def test_create_valid_message_generates_uuid_and_aware_utc_timestamp() -> None:
    before = datetime.now(timezone.utc)
    message = make_message(parent_task_id=None)
    after = datetime.now(timezone.utc)

    assert message.message_id
    assert message.timestamp.tzinfo is timezone.utc
    assert before <= message.timestamp <= after
    assert message.parent_task_id is None


def test_each_message_gets_a_unique_id() -> None:
    first = make_message()
    second = make_message()
    assert first.message_id != second.message_id


def test_explicit_id_and_timestamp_are_preserved() -> None:
    timestamp = datetime(2025, 2, 3, 4, 5, tzinfo=timezone(timedelta(hours=3)))
    message = make_message(message_id="explicit-id", timestamp=timestamp)
    assert message.message_id == "explicit-id"
    assert message.timestamp is timestamp


@pytest.mark.parametrize("message_type", list(MessageType))
def test_all_message_types_are_supported_and_roundtrip(message_type: MessageType) -> None:
    message = make_message(message_type=message_type)
    restored = Message.from_dict(message.to_dict())
    assert restored.message_type is message_type
    assert restored.to_dict() == message.to_dict()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("message_id", "", "message_id"),
        ("message_id", 123, "message_id"),
        ("sender", "  ", "sender"),
        ("sender", None, "sender"),
        ("receiver", "", "receiver"),
        ("receiver", False, "receiver"),
        ("message_type", "TASK_ASSIGN", "message_type"),
        ("message_type", None, "message_type"),
        ("task_id", "", "task_id"),
        ("task_id", 5, "task_id"),
        ("parent_task_id", " ", "parent_task_id"),
        ("parent_task_id", 7, "parent_task_id"),
        ("payload", [], "payload"),
        ("payload", None, "payload"),
        ("timestamp", "2025-01-01T00:00:00+00:00", "timestamp"),
        ("timestamp", None, "timestamp"),
    ],
)
def test_invalid_constructor_fields_raise_validation_error(field, value, message) -> None:
    with pytest.raises(MessageValidationError, match=message):
        make_message(**{field: value})


def test_naive_datetime_is_rejected() -> None:
    with pytest.raises(MessageValidationError, match="timezone-aware"):
        make_message(timestamp=datetime(2025, 1, 1))


def test_to_dict_has_all_fields_and_serialized_enum_and_timestamp() -> None:
    message = make_message(message_id="m-1", parent_task_id="parent-1")
    result = message.to_dict()
    assert set(result) == {
        "message_id", "sender", "receiver", "message_type", "task_id",
        "parent_task_id", "payload", "timestamp",
    }
    assert result["message_type"] == "TASK_ASSIGN"
    assert result["timestamp"] == message.timestamp.isoformat()
    assert result["payload"] == message.payload


def test_message_to_dict_to_message_preserves_all_fields() -> None:
    original = make_message(
        message_id="roundtrip-id",
        message_type=MessageType.REVIEW_FEEDBACK,
        parent_task_id="parent-task",
        timestamp=datetime(2024, 6, 7, 8, 9, 10, tzinfo=timezone.utc),
    )
    restored = Message.from_dict(original.to_dict())
    assert restored == original


def test_from_dict_accepts_iso_timestamp_and_enum_string() -> None:
    message = make_message()
    data = message.to_dict()
    restored = Message.from_dict(data)
    assert restored.message_type == MessageType.TASK_ASSIGN
    assert restored.timestamp == message.timestamp


def test_from_dict_missing_fields_are_reported() -> None:
    with pytest.raises(MessageValidationError, match="missing required field"):
        Message.from_dict({"sender": "agent-a"})


def test_from_dict_unknown_message_type_is_reported() -> None:
    data = make_message().to_dict()
    data["message_type"] = "UNKNOWN"
    with pytest.raises(MessageValidationError, match="Unknown message_type"):
        Message.from_dict(data)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("timestamp", "not-a-date", "ISO 8601"),
        ("timestamp", 123, "timestamp"),
        ("timestamp", "2025-01-01T00:00:00", "timezone-aware"),
        ("message_type", 1, "MessageType or string"),
        ("payload", "not-a-dict", "payload"),
        ("sender", [], "sender"),
    ],
)
def test_from_dict_invalid_values_raise_validation_error(field, value, message) -> None:
    data = make_message().to_dict()
    data[field] = value
    with pytest.raises(MessageValidationError, match=message):
        Message.from_dict(data)


def test_from_dict_requires_dictionary() -> None:
    with pytest.raises(MessageValidationError, match="must be a dictionary"):
        Message.from_dict([("sender", "agent-a")])
