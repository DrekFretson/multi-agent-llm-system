"""Thread-safe collection and export of pipeline execution metrics."""

from __future__ import annotations

import csv
import json
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping
from uuid import uuid4

from src.llm.model_registry import ModelRegistry


class MetricsError(Exception):
    """Base error for metrics collection and export."""


class MetricsExportError(MetricsError):
    """Raised when metrics cannot be exported."""


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class MetricContext:
    """Execution identity propagated through async context and worker threads."""

    collector: "MetricsCollector"
    run_id: str
    experiment_id: str | None = None
    task_id: str | None = None
    agent_role: str | None = None


_ACTIVE_CONTEXT: ContextVar[MetricContext | None] = ContextVar("active_metrics_context", default=None)


def current_metric_context() -> MetricContext | None:
    """Return the metrics context active in this execution context, if any."""
    return _ACTIVE_CONTEXT.get()


def activate_metrics(collector: "MetricsCollector", *, task_id: str | None = None,
                     agent_role: str | None = None) -> Token:
    """Bind a collector to the current context and return a reset token."""
    return _ACTIVE_CONTEXT.set(MetricContext(
        collector, collector.run_id, collector.experiment_id, task_id, agent_role
    ))


def reset_metrics(token: Token) -> None:
    """Restore the metrics context active before :func:`activate_metrics`."""
    _ACTIVE_CONTEXT.reset(token)


@dataclass
class MetricsCollector:
    """Collect one run's LLM, task, agent, message, error and quality records.

    Records are protected by a lock so independent task workers can append
    without races. Durations should be measured by callers with perf_counter.
    """

    experiment_id: str | None = None
    run_id: str = field(default_factory=lambda: str(uuid4()))
    model_registry: ModelRegistry = field(default_factory=ModelRegistry, repr=False)
    _started: float = field(default_factory=time.perf_counter, repr=False)
    _started_at: str = field(default_factory=_timestamp)
    _ended: float | None = field(default=None, repr=False)
    _ended_at: str | None = None
    _llm_calls: list[dict[str, Any]] = field(default_factory=list, repr=False)
    _external_calls: list[dict[str, Any]] = field(default_factory=list, repr=False)
    _tasks: list[dict[str, Any]] = field(default_factory=list, repr=False)
    _agents: list[dict[str, Any]] = field(default_factory=list, repr=False)
    _messages: list[dict[str, Any]] = field(default_factory=list, repr=False)
    _errors: list[dict[str, Any]] = field(default_factory=list, repr=False)
    _security_incidents: list[dict[str, Any]] = field(default_factory=list, repr=False)
    _quality: dict[str, Any] = field(default_factory=dict, repr=False)
    _retry_count: int = field(default=0, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    @contextmanager
    def activate(self, *, task_id: str | None = None,
                 agent_role: str | None = None) -> Iterator["MetricsCollector"]:
        """Temporarily make this collector available to pipeline components."""
        token = activate_metrics(self, task_id=task_id, agent_role=agent_role)
        try:
            yield self
        finally:
            reset_metrics(token)

    @contextmanager
    def task_context(self, task_id: str, agent_role: str | None = None) -> Iterator[None]:
        """Set task and optional role identity for calls made in this task."""
        token = activate_metrics(self, task_id=task_id, agent_role=agent_role)
        try:
            yield
        finally:
            reset_metrics(token)

    def record_llm_call(self, *, model_id: str, agent_id: str | None,
                        task_id: str | None, agent_role: str | None,
                        request_id: str | None = None,
                        duration_ms: float, prompt_tokens: int | None,
                        completion_tokens: int | None, total_tokens: int | None,
                        finish_reason: str | None, success: bool,
                        error_type: str | None = None,
                        error_message: str | None = None) -> dict[str, Any]:
        """Record one actual provider request; missing usage remains explicit."""
        prompt_tokens = prompt_tokens if _is_token_count(prompt_tokens) else None
        completion_tokens = completion_tokens if _is_token_count(completion_tokens) else None
        total_tokens = total_tokens if _is_token_count(total_tokens) else None
        model = next((item for item in self.model_registry.models.values()
                      if item.model_id == model_id), None)
        price_known = bool(model is not None and model.cost.input_per_1m_tokens is not None
                           and model.cost.output_per_1m_tokens is not None)
        usage_known = prompt_tokens is not None and completion_tokens is not None
        cost: float | None = None
        if price_known and usage_known:
            assert model is not None
            cost = (prompt_tokens * model.cost.input_per_1m_tokens
                    + completion_tokens * model.cost.output_per_1m_tokens) / 1_000_000
        row = {
            "timestamp": _timestamp(), "run_id": self.run_id,
            "experiment_id": self.experiment_id, "task_id": task_id,
            "request_id": request_id,
            "agent_id": agent_id, "agent_role": agent_role,
            "model_id": model_id, "duration_ms": round(max(0.0, duration_ms), 3),
            "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "total_tokens": total_tokens, "usage_source": "provider" if usage_known else "missing",
            "finish_reason": finish_reason, "estimated_cost": cost,
            "cost_status": ("known" if cost is not None else
                            "missing_usage" if price_known else "unknown_price"),
            "success": bool(success), "error_type": error_type,
            "error_message": error_message,
        }
        with self._lock:
            self._llm_calls.append(row)
            if not success:
                self._errors.append({key: row[key] for key in (
                    "timestamp", "run_id", "experiment_id", "task_id", "agent_id",
                    "model_id", "error_type", "error_message"
                )})
        return dict(row)

    def record_task(self, *, task_id: str, agent_role: str | None, status: str,
                    duration_ms: float, error: str | None = None) -> None:
        """Record a task's elapsed execution duration and final status."""
        with self._lock:
            row = {"timestamp": _timestamp(), "run_id": self.run_id,
                "experiment_id": self.experiment_id, "task_id": task_id,
                "agent_role": agent_role, "status": status,
                "duration_ms": round(max(0.0, duration_ms), 3), "error": error}
            self._tasks.append(row)
            if error is not None:
                self._errors.append({"timestamp": row["timestamp"], "run_id": self.run_id,
                    "experiment_id": self.experiment_id, "task_id": task_id,
                    "agent_role": agent_role, "error_type": "TaskExecutionError",
                    "error_message": error})

    def record_external_call(self, *, service: str, agent_id: str | None,
                             task_id: str | None, agent_role: str | None,
                             duration_ms: float, success: bool,
                             error_type: str | None = None,
                             error_message: str | None = None) -> None:
        """Record an actual non-LLM provider request, such as Tavily Search."""
        row = {"timestamp": _timestamp(), "run_id": self.run_id,
               "experiment_id": self.experiment_id, "task_id": task_id,
               "service": service, "agent_id": agent_id, "agent_role": agent_role,
               "duration_ms": round(max(0.0, duration_ms), 3),
               "success": bool(success), "error_type": error_type,
               "error_message": error_message}
        with self._lock:
            self._external_calls.append(row)
            if not success:
                self._errors.append({key: row[key] for key in (
                    "timestamp", "run_id", "experiment_id", "task_id", "agent_id",
                    "error_type", "error_message")})

    def record_agent(self, *, agent_id: str, agent_role: str, task_id: str,
                     status: str, duration_ms: float, model_id: str | None = None) -> None:
        """Record active time spent in an agent invocation, excluding queue wait."""
        with self._lock:
            self._agents.append({"timestamp": _timestamp(), "run_id": self.run_id,
                "experiment_id": self.experiment_id, "task_id": task_id,
                "agent_id": agent_id, "agent_role": agent_role,
                "model_id": model_id, "status": status,
                "duration_ms": round(max(0.0, duration_ms), 3)})

    def record_message(self, *, task_id: str | None, message_id: str,
                       sender: str, receiver: str, message_type: str,
                       delivery_status: str, error_type: str | None = None,
                       error_message: str | None = None) -> None:
        """Record one Message Bus delivery attempt without message contents."""
        with self._lock:
            timestamp = _timestamp()
            self._messages.append({"timestamp": timestamp, "run_id": self.run_id,
                "experiment_id": self.experiment_id, "task_id": task_id,
                "message_id": message_id, "sender": sender, "receiver": receiver,
                "message_type": message_type, "delivery_status": delivery_status,
                "error_type": error_type, "error_message": error_message})
            if delivery_status != "delivered":
                self._errors.append({"timestamp": timestamp, "run_id": self.run_id,
                    "experiment_id": self.experiment_id, "task_id": task_id,
                    "agent_role": None, "error_type": error_type or "MessageDeliveryError",
                    "error_message": error_message})

    def set_quality(self, **quality: Any) -> None:
        """Attach explicitly measured quality/status fields to this run."""
        with self._lock:
            self._quality.update(quality)

    def record_retry(self, count: int = 1) -> None:
        """Increment the number of execution-level retry attempts."""
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("count must be a non-negative integer.")
        with self._lock:
            self._retry_count += count

    def record_security_incident(self, *, incident_type: str, actor_id: str | None,
                                 task_id: str | None, reason: str, blocked: bool) -> None:
        """Record an authorization or integrity incident without raw payloads."""
        with self._lock:
            row = {"timestamp": _timestamp(), "run_id": self.run_id,
                   "experiment_id": self.experiment_id, "incident_type": incident_type,
                   "actor_id": actor_id, "task_id": task_id, "reason": reason,
                   "blocked": bool(blocked)}
            self._security_incidents.append(row)
            self._errors.append({"timestamp": row["timestamp"], "run_id": self.run_id,
                "experiment_id": self.experiment_id, "task_id": task_id,
                "agent_id": actor_id, "error_type": "SecurityIncident",
                "error_message": reason})

    def finish(self, *, status: str | None = None) -> dict[str, Any]:
        """Mark the run complete and return its aggregate metric snapshot."""
        with self._lock:
            if self._ended is None:
                self._ended = time.perf_counter()
                self._ended_at = _timestamp()
            if status is not None:
                self._quality["execution_status"] = status
            return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        """Return a consistent export-ready snapshot of all run records."""
        with self._lock:
            duration = ((self._ended if self._ended is not None else time.perf_counter()) - self._started) * 1000
            calls = list(self._llm_calls)
            external_calls = list(self._external_calls)
            prompt_values = [c["prompt_tokens"] for c in calls]
            completion_values = [c["completion_tokens"] for c in calls]
            total_values = [c["total_tokens"] for c in calls]
            costs = [c["estimated_cost"] for c in calls]
            all_have = lambda values: bool(calls) and all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in values
            )
            return {
                "timestamp": self._started_at, "finished_at": self._ended_at,
                "run_id": self.run_id, "experiment_id": self.experiment_id,
                "duration_ms": round(duration, 3), "api_calls": len(calls) + len(external_calls),
                "llm_api_calls": len(calls), "external_api_calls": len(external_calls),
                "prompt_tokens": _sum_known(prompt_values) if all_have(prompt_values) else None,
                "completion_tokens": _sum_known(completion_values) if all_have(completion_values) else None,
                "total_tokens": _sum_known(total_values) if all_have(total_values) else None,
                "estimated_cost": _sum_known(costs) if all_have(costs) else None,
                "known_prompt_tokens_subtotal": _sum_known(prompt_values),
                "known_completion_tokens_subtotal": _sum_known(completion_values),
                "known_cost_subtotal": _sum_known(costs),
                "unknown_cost_calls": sum(c["cost_status"] == "unknown_price" for c in calls),
                "incomplete_cost_calls": sum(c["cost_status"] != "known" for c in calls),
                "missing_usage_calls": sum(c["usage_source"] == "missing" for c in calls),
                "error_count": len(self._errors), "message_count": len(self._messages),
                "incident_count": len(self._security_incidents),
                "task_count": len(self._tasks), "agent_invocations": len(self._agents),
                "retry_count": self._retry_count,
                "critic_calls": sum(1 for call in calls if call.get("agent_role") == "critic"),
                "critic_cycles": self._quality.get("critic_cycles"),
                "quality": dict(self._quality), "llm_calls": calls,
                "external_calls": external_calls,
                "security_incidents": list(self._security_incidents),
                "tasks": list(self._tasks), "agents": list(self._agents),
                "messages": list(self._messages), "errors": list(self._errors),
            }

    def export(self, directory: str | Path, *, prefix: str | None = None) -> dict[str, Path]:
        """Write one JSON document and normalized CSV files for each record type."""
        target = Path(directory)
        stem = prefix or self.run_id
        snapshot = self.snapshot()
        paths = {"json": target / f"{stem}.json"}
        rows = {
            "run": [{key: value for key, value in snapshot.items()
                if key not in {"llm_calls", "external_calls", "tasks", "agents", "messages", "errors", "quality", "security_incidents"}}],
            "llm_calls": snapshot["llm_calls"], "external_calls": snapshot["external_calls"], "tasks": snapshot["tasks"],
            "agents": snapshot["agents"], "messages": snapshot["messages"],
            "errors": snapshot["errors"],
            "security_incidents": snapshot["security_incidents"],
            "quality": [{"run_id": self.run_id, "experiment_id": self.experiment_id, **snapshot["quality"]}],
        }
        paths.update({name: target / f"{stem}_{name}.csv" for name in rows})
        try:
            target.mkdir(parents=True, exist_ok=True)
            paths["json"].write_text(json.dumps(snapshot, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            for name, records in rows.items():
                _write_csv(paths[name], records)
        except (OSError, TypeError, ValueError, csv.Error) as error:
            raise MetricsExportError(f"Could not export metrics for run {self.run_id}.") from error
        return paths


def _sum_known(values: Any) -> int | float | None:
    found = [value for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
    return sum(found) if found else None


def _is_token_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    if not columns:
        columns = ["run_id", "experiment_id"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def record_current_message(**fields: Any) -> None:
    """Record a bus event when a pipeline metrics context is active."""
    context = current_metric_context()
    if context is not None:
        context.collector.record_message(task_id=fields.get("task_id"),
            message_id=fields["message_id"], sender=fields["sender"],
            receiver=fields["receiver"], message_type=fields["message_type"],
            delivery_status=fields["delivery_status"], error_type=fields.get("error_type"),
            error_message=fields.get("error_message"))


def record_current_external_call(*, service: str, agent_id: str | None,
                                 duration_ms: float, success: bool,
                                 error_type: str | None = None,
                                 error_message: str | None = None) -> None:
    """Record a non-LLM provider request when a run context is active."""
    context = current_metric_context()
    if context is not None:
        context.collector.record_external_call(service=service, agent_id=agent_id,
            task_id=context.task_id, agent_role=context.agent_role,
            duration_ms=duration_ms, success=success, error_type=error_type,
            error_message=error_message)


__all__ = ["MetricsCollector", "MetricsError", "MetricsExportError",
           "activate_metrics", "current_metric_context", "record_current_external_call",
           "record_current_message", "reset_metrics"]
