"""Tests for the project JSONL event logger."""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.logging import Logger, LoggerSerializationError, LoggerWriteError


def test_logger_writes_jsonl_and_creates_parent_directories(tmp_path: Path) -> None:
    log_file = tmp_path / "nested" / "logs" / "events.jsonl"
    logger = Logger(log_file)

    event = logger.log_event(
        "message.received",
        sender="agent-a",
        content="Привет",
    )

    lines = log_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == event
    assert event["event_type"] == "message.received"
    assert event["sender"] == "agent-a"
    assert event["content"] == "Привет"
    timestamp = datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00"))
    assert timestamp.tzinfo == timezone.utc


def test_logger_appends_multiple_events_as_separate_lines(tmp_path: Path) -> None:
    log_file = tmp_path / "events.jsonl"
    logger = Logger(log_file)

    logger.log_event("llm.request", request_id="one", success=True)
    logger.log_event("message.sent", sender="agent-a", recipient="agent-b")

    lines = log_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["request_id"] == "one"
    assert json.loads(lines[1])["event_type"] == "message.sent"


def test_logger_reports_non_json_values(tmp_path: Path) -> None:
    logger = Logger(tmp_path / "events.jsonl")

    with pytest.raises(LoggerSerializationError, match="cannot be serialized"):
        logger.log_event("bad.event", unsupported=object())


def test_logger_reports_file_write_errors(tmp_path: Path) -> None:
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("file blocks directory creation", encoding="utf-8")
    logger = Logger(blocker / "events.jsonl")

    with pytest.raises(LoggerWriteError, match="Could not append event"):
        logger.log_event("test.event", value=1)


def test_project_package_does_not_replace_standard_library_logging() -> None:
    project_root = Path(__file__).resolve().parents[1]
    script = f"""\
import sys
sys.path.insert(0, {str(project_root)!r})
sys.path.insert(0, {str(project_root / 'src')!r})

import logging
import logging.handlers
import sysconfig
from pathlib import Path
from src.logging import Logger

expected = (Path(sysconfig.get_path("stdlib")) / "logging" / "__init__.py").resolve()
assert Path(logging.__file__).resolve() == expected
assert callable(logging.getLogger)
assert Logger is not logging.Logger
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(project_root)
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=project_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
