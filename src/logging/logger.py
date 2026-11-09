"""Small dependency-free JSON Lines event logger."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class LoggerError(Exception):
    """Base exception for structured logger failures."""


class LoggerSerializationError(LoggerError):
    """Raised when an event cannot be represented as JSON."""


class LoggerWriteError(LoggerError):
    """Raised when the JSONL file or its parent directory cannot be written."""


class Logger:
    """Append structured events as one UTF-8 JSON object per line."""

    def __init__(self, log_file: str | Path | None = None) -> None:
        """Create a logger targeting ``logs/llm_requests.jsonl`` by default."""
        if log_file is None:
            log_file = (
                Path(__file__).resolve().parents[2]
                / "logs"
                / "llm_requests.jsonl"
            )
        self.log_file = Path(log_file)
        self._lock = threading.Lock()

    def log_event(self, event_type: str, **fields: Any) -> dict[str, Any]:
        """Write one structured event and return the record that was written."""
        if not isinstance(event_type, str) or not event_type.strip():
            raise ValueError("event_type must be a non-empty string.")

        record = dict(fields)
        timestamp = record.get("timestamp")
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
            timestamp = timestamp.replace("+00:00", "Z")
        record["timestamp"] = timestamp
        record["event_type"] = event_type

        try:
            line = json.dumps(
                record,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as error:
            raise LoggerSerializationError(
                f"Event {event_type!r} cannot be serialized as JSON."
            ) from error

        try:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            with self._lock, self.log_file.open("a", encoding="utf-8") as output:
                output.write(line + "\n")
        except OSError as error:
            raise LoggerWriteError(
                f"Could not append event to log file {self.log_file}."
            ) from error

        return record
