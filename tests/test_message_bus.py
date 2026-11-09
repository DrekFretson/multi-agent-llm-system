"""Synchronous MessageBus tests using temporary logs and fake failures."""

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.communication import (
    AgentRegistrationError,
    Message,
    MessageBus,
    MessageBusError,
    MessageDeliveryError,
    MessageType,
    UnknownReceiverError,
    UnknownSenderError,
)
from src.logging import Logger


def make_message(
    sender: str = "agent-a",
    receiver: str = "agent-b",
    *,
    message_id: str = "message-1",
    payload: dict | None = None,
) -> Message:
    return Message(
        message_id=message_id,
        sender=sender,
        receiver=receiver,
        message_type=MessageType.TASK_ASSIGN,
        task_id="task-1",
        parent_task_id=None,
        payload=payload if payload is not None else {"instruction": "do work"},
        timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )


def registered_bus(logger=None) -> MessageBus:
    bus = MessageBus(logger=logger)
    bus.register_agent("agent-a")
    bus.register_agent("agent-b")
    return bus


def test_register_one_and_multiple_agents_in_registration_order() -> None:
    bus = MessageBus()
    bus.register_agent("agent-a")
    bus.register_agent("agent-b")
    assert bus.get_registered_agents() == ["agent-a", "agent-b"]


@pytest.mark.parametrize("agent_id", ["", "  ", None, 3, False])
def test_register_invalid_id_raises(agent_id) -> None:
    with pytest.raises(AgentRegistrationError, match="non-empty string"):
        MessageBus().register_agent(agent_id)


def test_duplicate_registration_is_rejected() -> None:
    bus = MessageBus()
    bus.register_agent("agent-a")
    with pytest.raises(AgentRegistrationError, match="already registered"):
        bus.register_agent("agent-a")


def test_unregister_empty_registered_agent() -> None:
    bus = MessageBus()
    bus.register_agent("agent-a")
    bus.unregister_agent("agent-a")
    assert bus.get_registered_agents() == []


def test_unregister_unknown_agent_raises() -> None:
    with pytest.raises(AgentRegistrationError, match="not registered"):
        MessageBus().unregister_agent("missing")


def test_unregister_refuses_to_drop_unread_messages() -> None:
    bus = registered_bus()
    message = make_message()
    bus.send(message)
    with pytest.raises(AgentRegistrationError, match="unread messages"):
        bus.unregister_agent("agent-b")
    assert bus.get_registered_agents() == ["agent-a", "agent-b"]
    assert bus.receive("agent-b") is message
    bus.unregister_agent("agent-b")


def test_send_delivers_only_to_addressed_registered_receiver() -> None:
    bus = registered_bus()
    bus.register_agent("agent-c")
    message = make_message()
    bus.send(message)
    assert bus.receive("agent-b") is message
    assert bus.receive("agent-c") is None


def test_send_rejects_non_message_argument() -> None:
    with pytest.raises(MessageBusError, match="Message instance"):
        registered_bus().send({"receiver": "agent-b"})


def test_unknown_sender_is_not_delivered() -> None:
    bus = MessageBus()
    bus.register_agent("agent-b")
    with pytest.raises(UnknownSenderError, match="sender"):
        bus.send(make_message(sender="missing"))
    assert bus.receive("agent-b") is None


def test_unknown_receiver_is_not_technical_delivery_error() -> None:
    bus = MessageBus()
    bus.register_agent("agent-a")
    with pytest.raises(UnknownReceiverError) as error:
        bus.send(make_message(receiver="missing"))
    assert not isinstance(error.value, MessageDeliveryError)


def test_multiple_messages_are_received_fifo_and_only_once() -> None:
    bus = registered_bus()
    messages = [make_message(message_id=f"message-{index}") for index in range(3)]
    for message in messages:
        bus.send(message)
    assert [bus.receive("agent-b") for _ in messages] == messages
    assert bus.receive("agent-b") is None


def test_empty_queue_returns_none_without_blocking() -> None:
    assert registered_bus().receive("agent-b") is None


def test_receive_from_unknown_agent_raises() -> None:
    with pytest.raises(AgentRegistrationError, match="not registered"):
        MessageBus().receive("missing")


def test_send_does_not_mutate_original_message() -> None:
    bus = registered_bus()
    message = make_message(payload={"nested": {"value": 2}})
    before = message.to_dict()
    bus.send(message)
    assert message.to_dict() == before
    assert bus.receive("agent-b") is message


def test_self_delivery_is_allowed() -> None:
    bus = MessageBus()
    bus.register_agent("same-agent")
    message = make_message(sender="same-agent", receiver="same-agent")
    bus.send(message)
    assert bus.receive("same-agent") is message


def test_successful_delivery_is_logged_after_enqueue_with_metadata(tmp_path: Path) -> None:
    log_file = tmp_path / "events.jsonl"
    bus = registered_bus(Logger(log_file))
    message = make_message(payload={"secret": "do-not-log-this"})
    bus.send(message)

    event = json.loads(log_file.read_text(encoding="utf-8").splitlines()[0])
    assert event["event_type"] == "message_delivery"
    assert event["message_id"] == message.message_id
    assert event["sender"] == "agent-a"
    assert event["receiver"] == "agent-b"
    assert event["message_type"] == "TASK_ASSIGN"
    assert event["task_id"] == "task-1"
    assert event["parent_task_id"] is None
    assert event["delivery_status"] == "delivered"
    assert event["error_type"] is None
    assert event["timestamp"].endswith("Z")
    assert "do-not-log-this" not in log_file.read_text(encoding="utf-8")
    assert bus.receive("agent-b") is message


def test_logger_observes_message_already_queued() -> None:
    bus = registered_bus()
    class InspectingLogger:
        called = False

        def log_event(self, _event_type, **_fields):
            self.called = True
            assert len(bus._inboxes["agent-b"]) == 1

    logger = InspectingLogger()
    bus._logger = logger
    bus.send(make_message())
    assert logger.called


def test_failed_unknown_receiver_is_logged_before_error(tmp_path: Path) -> None:
    log_file = tmp_path / "events.jsonl"
    bus = MessageBus(Logger(log_file))
    bus.register_agent("agent-a")
    message = make_message(receiver="missing")

    with pytest.raises(UnknownReceiverError):
        bus.send(message)

    event = json.loads(log_file.read_text(encoding="utf-8").splitlines()[0])
    assert event["delivery_status"] == "failed"
    assert event["error_type"] == "UnknownReceiverError"
    assert "not registered" in event["error_message"]


def test_logger_failure_after_delivery_does_not_fail_or_duplicate_send() -> None:
    logger = Mock()
    logger.log_event.side_effect = OSError("disk unavailable")
    bus = registered_bus(logger)
    message = make_message()

    assert bus.send(message) is None
    assert isinstance(bus.last_logging_error, OSError)
    assert bus.receive("agent-b") is message
    assert bus.receive("agent-b") is None
    logger.log_event.assert_called_once()


def test_logger_failure_does_not_mask_unknown_receiver_error() -> None:
    logger = Mock()
    logger.log_event.side_effect = RuntimeError("logger down")
    bus = MessageBus(logger)
    bus.register_agent("agent-a")

    with pytest.raises(UnknownReceiverError):
        bus.send(make_message(receiver="missing"))
    assert isinstance(bus.last_logging_error, RuntimeError)


def test_technical_append_failure_is_chained_and_logged(tmp_path: Path) -> None:
    class BrokenInbox:
        def append(self, _message):
            raise OSError("queue storage failed")

        def __bool__(self):
            return False

    log_file = tmp_path / "events.jsonl"
    bus = registered_bus(Logger(log_file))
    bus._inboxes["agent-b"] = BrokenInbox()
    with pytest.raises(MessageDeliveryError) as error:
        bus.send(make_message())

    assert isinstance(error.value.__cause__, OSError)
    event = json.loads(log_file.read_text(encoding="utf-8").splitlines()[0])
    assert event["delivery_status"] == "failed"
    assert event["error_type"] == "MessageDeliveryError"
    assert event["cause_type"] == "OSError"
    assert event["cause_message"] == "queue storage failed"


def test_delivery_can_work_without_a_logger() -> None:
    bus = registered_bus()
    message = make_message()
    bus.send(message)
    assert bus.last_logging_error is None
    assert bus.receive("agent-b") is message


def test_non_callable_logger_is_rejected() -> None:
    with pytest.raises(TypeError, match="log_event"):
        MessageBus(logger=object())
