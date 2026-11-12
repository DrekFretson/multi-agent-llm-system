"""Dependency-aware concurrent execution of tasks already stored in TaskStore."""

from __future__ import annotations

import json
import inspect
import re
import threading
import time
from contextvars import copy_context
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from datetime import datetime, timezone
from typing import Any, Mapping

from src.communication import Message, MessageBus, MessageType
from src.coordination.task_store import Task, TaskPriority, TaskStatus, TaskStore, TaskType


class TaskExecutionError(Exception):
    """Base class for saved-plan execution failures."""


class ExecutionBudgetExceeded(TaskExecutionError):
    """Raised internally when the remaining execution budget is insufficient."""

    def __init__(self, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = dict(details or {})


class EmptyModelResponseError(TaskExecutionError):
    """An agent received an empty completion, with provider metadata retained."""

    def __init__(self, result: AgentResult) -> None:
        self.finish_reason = result.finish_reason
        self.prompt_tokens = result.prompt_tokens
        self.completion_tokens = result.completion_tokens
        self.total_tokens = result.total_tokens
        self.model = result.model
        # An empty completion marked ``length`` usually means the provider used
        # the output budget before emitting visible content. Retry that same
        # route once; the executor's normal retry budget keeps this bounded.
        self.retryable = result.finish_reason in {None, "stop", "error", "length"}
        super().__init__(
            "The model returned an empty response "
            f"(model={result.model!r}, finish_reason={result.finish_reason!r}, "
            f"prompt_tokens={result.prompt_tokens!r}, "
            f"completion_tokens={result.completion_tokens!r})."
        )


class IncompleteTaskResultError(TaskExecutionError):
    """Raised when an agent call succeeded but the task was not completed."""


class AgentRequestFailed(TaskExecutionError):
    """An agent returned a normalized LLM failure with its type retained."""

    def __init__(self, result: AgentResult) -> None:
        self.error_type = result.error_type or "LLMClientError"
        self.retryable = self.error_type in {"NetworkError", "LLMTimeoutError", "RateLimitError"}
        self.repair_attempts = result.format_repair_attempts
        super().__init__(result.error or f"{self.error_type}: agent request failed.")


class AgentOutcomeParseError(TaskExecutionError):
    """Raised when a generic agent omits its required task-status envelope."""


class CriticRejectedError(TaskExecutionError):
    """Critic found a rejection and carries the complete review report."""

    def __init__(self, message: str, report: Any) -> None:
        super().__init__(message)
        self.report = report


@dataclass(frozen=True)
class CriticExecutionRecord:
    """Final Critic report and the bounded sequence of reviewed revisions."""

    report: Any
    iterations: list[dict[str, Any]]

    @property
    def verdict(self) -> str:
        return str(getattr(self.report, "verdict", "REJECTED"))

    @property
    def summary(self) -> str:
        return str(getattr(self.report, "summary", ""))

    @property
    def issues(self) -> Any:
        return getattr(self.report, "issues", [])

    @property
    def clarification_question(self) -> str | None:
        return getattr(self.report, "clarification_question", None)

    @property
    def cycle_count(self) -> int:
        return len({
            item.get("cycle") for item in self.iterations
            if item.get("cycle", 0) > 0 and "review" in item
        })


@dataclass(frozen=True)
class ExecutionEvent:
    """One immutable event in the in-memory execution history."""

    timestamp: str
    task_id: str
    event: str
    attempt: int | None = None
    agent: str | None = None
    model_id: str | None = None
    duration_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    estimated_cost: float | None = None
    error: str | None = None
    fallback: bool = False
    route_reason: str | None = None
    total_tokens: int | None = None
    finish_reason: str | None = None
    budget_details: dict[str, Any] | None = None
    clarification_question: str | None = None
    failure_category: str | None = None
    retry_allowed: bool | None = None
    format_repair_attempts: int = 0
    review_performed: bool | None = None
    retry_reason: str | None = None
    critic_cycle: int | None = None


@dataclass
class TaskExecutionReport:
    """Execution outcome and measured/estimated resource use for one root task."""

    root_task_id: str
    status: str
    task_results: dict[str, Any]
    task_statuses: dict[str, str]
    history: list[ExecutionEvent]
    calls: int
    prompt_tokens: int
    completion_tokens: int
    estimated_cost: float
    review_status: str
    errors: list[str] = field(default_factory=list)
    estimated_tokens: int = 0
    duration_ms: float = 0.0
    budget_violation: str | None = None
    unknown_cost_requests: int = 0
    retries: int = 0
    max_execution_time: float | None = None
    max_total_retries: int | None = None


@dataclass
class _AttemptResult:
    value: Any
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int | None = None
    finish_reason: str | None = None
    task_completed: bool | None = None
    needs_input: bool = False
    clarification_question: str | None = None
    format_repair_attempts: int = 0


class TaskExecutor:
    """Run a TaskStore plan by dependency-ready waves and bounded retries."""

    def __init__(
        self,
        task_store: TaskStore,
        router: Any,
        agents: Mapping[str, Any],
        *,
        message_bus: MessageBus | None = None,
        max_workers: int = 4,
        max_retries: int = 1,
        max_total_retries: int = 8,
        max_critic_cycles: int = 1,
        task_timeout: float = 180.0,
        max_execution_time: float | None = 600.0,
        max_latency_ms: float | None = None,
        max_calls: int | None = 30,
        max_tokens: int | None = 100_000,
        max_cost: float | None = None,
        coordinator_id: str = "task_executor",
    ) -> None:
        if not isinstance(task_store, TaskStore):
            raise TypeError("task_store must be a TaskStore instance.")
        from src.routing import AdaptiveRouter
        if not isinstance(router, AdaptiveRouter):
            raise TypeError("router must be an AdaptiveRouter instance.")
        if not isinstance(agents, Mapping):
            raise TypeError("agents must map configured roles to agent instances.")
        for name, value in (("max_workers", max_workers), ("max_retries", max_retries),
                            ("max_total_retries", max_total_retries),
                            ("max_critic_cycles", max_critic_cycles)):
            if isinstance(value, bool) or not isinstance(value, int) or value < (1 if name == "max_workers" else 0):
                raise ValueError(f"{name} has an invalid value.")
        if isinstance(task_timeout, bool) or not isinstance(task_timeout, (int, float)) or task_timeout <= 0:
            raise ValueError("task_timeout must be a positive number of seconds.")
        if max_execution_time is not None and (
            isinstance(max_execution_time, bool)
            or not isinstance(max_execution_time, (int, float))
            or max_execution_time <= 0
        ):
            raise ValueError("max_execution_time must be a positive number or None.")
        if max_latency_ms is not None and (
            isinstance(max_latency_ms, bool)
            or not isinstance(max_latency_ms, (int, float))
            or max_latency_ms <= 0
        ):
            raise ValueError("max_latency_ms must be a positive number or None.")
        for name, value in (("max_calls", max_calls), ("max_tokens", max_tokens)):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ValueError(f"{name} must be a non-negative integer or None.")
        if max_cost is not None and (isinstance(max_cost, bool) or not isinstance(max_cost, (int, float)) or max_cost < 0):
            raise ValueError("max_cost must be a non-negative number or None.")
        self.task_store = task_store
        self.router = router
        self.agents = dict(agents)
        self.message_bus = message_bus or MessageBus()
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.max_total_retries = max_total_retries
        self.max_critic_cycles = max_critic_cycles
        self.task_timeout = float(task_timeout)
        self.max_execution_time = float(max_execution_time) if max_execution_time is not None else None
        self._created_at = time.perf_counter()
        self._deadline = self._created_at + self.max_execution_time if self.max_execution_time is not None else None
        self.max_latency_ms = float(max_latency_ms) if max_latency_ms is not None else None
        self.max_calls = max_calls
        self.max_tokens = max_tokens
        self.max_cost = float(max_cost) if max_cost is not None else None
        self.coordinator_id = coordinator_id
        self._history: list[ExecutionEvent] = []
        self._errors: list[str] = []
        self._review_history: dict[str, list[dict[str, Any]]] = {}
        self._lock = threading.RLock()
        self._calls = 0
        self._retries = 0
        self._reserved_retries = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._estimated_tokens = 0
        self._budget_tokens_used = 0
        self._estimated_cost = 0.0
        self._reserved_tokens = 0
        self._reserved_cost = 0.0
        self._budget_violation: str | None = None
        self._unknown_cost_requests = 0
        from src.llm import LLMClient
        self._llm_clients = list({
            id(client): client
            for agent in self.agents.values()
            if isinstance((client := getattr(agent, "client", None)), LLMClient)
            and hasattr(client, "_client")
        }.values())
        self._hook_stack: list[Any] = []
        self._register_bus_agents()

    def execute(self, root_task_id: str, *, finalize_root: bool = True) -> TaskExecutionReport:
        """Execute all direct children of a stored root task."""
        execution_started = time.perf_counter()
        root = self.task_store.get_task(root_task_id)
        tasks = self.task_store.get_subtasks(root_task_id)
        if not tasks:
            raise TaskExecutionError(f"Root task {root_task_id!r} has no saved subtasks.")
        for task in tasks:
            if task.status is TaskStatus.RUNNING:
                message = "Recovered a task left in RUNNING state by an earlier interrupted execution."
                self.task_store.update_task(task.task_id, error=message)
                self.task_store.update_status(task.task_id, TaskStatus.FAILED)
                self._record(ExecutionEvent(_now(), task.task_id, "stale_task_recovered", error=message))
        by_id = {task.task_id: task for task in tasks}
        if root.status is TaskStatus.NEW:
            self.task_store.update_status(root_task_id, TaskStatus.READY)
            self.task_store.update_status(root_task_id, TaskStatus.ASSIGNED)
            self.task_store.update_status(root_task_id, TaskStatus.RUNNING)

        pending = set(by_id)
        executor = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="agent-task")
        self._install_request_hooks()
        try:
            while pending:
                if self._time_exceeded():
                    reason = self._time_limit_message()
                    for identifier in list(pending):
                        self._cancel_task(by_id[identifier], reason, event="execution_deadline")
                        pending.remove(identifier)
                    break
                self._skip_blocked_tasks(pending, by_id)
                ready = [
                    by_id[identifier]
                    for identifier in pending
                    if self._dependencies_completed(by_id[identifier])
                ]
                priority_rank = {TaskPriority.HIGH: 0, TaskPriority.MEDIUM: 1, TaskPriority.LOW: 2}
                ready.sort(key=lambda task: (priority_rank[task.priority], task.created_at))
                if not ready:
                    if pending:
                        # This should only happen if the saved graph was changed after validation.
                        for identifier in list(pending):
                            self._mark_skipped(by_id[identifier], "No runnable dependency order remained.")
                            pending.remove(identifier)
                    break
                futures = {}
                for task in ready:
                    cancellation = threading.Event()
                    context = copy_context()
                    futures[executor.submit(
                        context.run, self._execute_task, task, root, cancellation
                    )] = (task, cancellation)
                    pending.remove(task.task_id)
                remaining_time = self._remaining_time()
                wave_timeout = self.task_timeout if remaining_time is None else min(self.task_timeout, remaining_time)
                done, unfinished = wait(futures, timeout=wave_timeout)
                for future in done:
                    future.result()  # Worker records ordinary agent errors as task failures.
                for future in unfinished:
                    task, cancellation = futures[future]
                    cancellation.set()
                    future.cancel()
                    if self._time_exceeded():
                        self._cancel_task(task, self._time_limit_message(), event="execution_deadline")
                    else:
                        self._fail_running_task(task, "Task execution timed out.", event="timeout")
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
            self._restore_request_hooks()

        children = self.task_store.get_subtasks(root_task_id)
        statuses = {task.task_id: task.status.value for task in children}
        results = {task.task_id: task.result for task in children}
        review_status = self._review_status(children)
        failed = any(task.status in {TaskStatus.FAILED, TaskStatus.SKIPPED, TaskStatus.CANCELLED} for task in children)
        needs_input = any(task.status is TaskStatus.NEEDS_INPUT for task in children)
        complete = all(task.status is TaskStatus.COMPLETED for task in children)
        status = "NEEDS_INPUT" if needs_input else "COMPLETED" if complete and review_status in {"APPROVED", "NOT_REQUIRED"} else (
            "FAILED" if not any(task.status is TaskStatus.COMPLETED for task in children) else "PARTIAL"
        )
        if self._budget_violation is not None and status == "COMPLETED":
            status = "PARTIAL"
            self._errors.append(self._budget_violation)
        if failed and status == "COMPLETED":
            status = "PARTIAL"
        report = TaskExecutionReport(
            root_task_id=root_task_id,
            status=status,
            task_results=results,
            task_statuses=statuses,
            history=list(self._history),
            calls=self._calls,
            prompt_tokens=self._prompt_tokens,
            completion_tokens=self._completion_tokens,
            estimated_cost=round(self._estimated_cost, 9),
            review_status=review_status,
            errors=list(self._errors),
            estimated_tokens=self._estimated_tokens,
            duration_ms=round((time.perf_counter() - execution_started) * 1000, 3),
            budget_violation=self._budget_violation,
            unknown_cost_requests=self._unknown_cost_requests,
            retries=self._retries,
            max_execution_time=self.max_execution_time,
            max_total_retries=self.max_total_retries,
        )
        history_data = [asdict(event) for event in report.history]
        self.task_store.update_task(root_task_id, result={
            "status": report.status,
            "review_status": report.review_status,
            "task_statuses": report.task_statuses,
            "history": history_data,
            "metrics": {
                "calls": report.calls,
                "prompt_tokens": report.prompt_tokens,
                "completion_tokens": report.completion_tokens,
                "estimated_tokens": report.estimated_tokens,
                "estimated_cost": report.estimated_cost,
                "duration_ms": report.duration_ms,
                "budget_violation": report.budget_violation,
                "unknown_cost_requests": report.unknown_cost_requests,
                "retries": report.retries,
                "max_execution_time": report.max_execution_time,
                "max_total_retries": report.max_total_retries,
                "max_critic_cycles": self.max_critic_cycles,
            },
            "errors": report.errors,
        })
        current_root = self.task_store.get_task(root_task_id)
        if finalize_root and current_root.status is TaskStatus.RUNNING:
            self.task_store.update_status(
                root_task_id,
                TaskStatus.COMPLETED if status == "COMPLETED" else (
                    TaskStatus.NEEDS_INPUT if status == "NEEDS_INPUT" else TaskStatus.FAILED
                ),
            )
        return report

    def _execute_task(self, task: Task, root: Task, cancellation: threading.Event) -> None:
        """Execute a task and persist its wall-clock duration independently."""
        started = time.perf_counter()
        try:
            self._execute_task_impl(task, root, cancellation)
        finally:
            try:
                from src.evaluation.metrics import current_metric_context

                metric_context = current_metric_context()
                if metric_context is not None:
                    saved = self.task_store.get_task(task.task_id)
                    metric_context.collector.record_task(
                        task_id=task.task_id,
                        agent_role=saved.assignee or metric_context.agent_role,
                        status=saved.status.value,
                        duration_ms=(time.perf_counter() - started) * 1000,
                        error=saved.error,
                    )
            except Exception:
                pass

    def _execute_task_impl(self, task: Task, root: Task, cancellation: threading.Event) -> None:
        dependencies = [self.task_store.get_task(dep_id) for dep_id in task.dependencies]
        context = self._dependency_context(dependencies)
        attempted: set[str] = set()
        same_agent_empty_retry_used = False
        for attempt in range(1, self.max_retries + 2):
            if cancellation.is_set():
                return
            if self._time_exceeded():
                self._cancel_task(task, self._time_limit_message(), event="execution_deadline")
                return
            started = time.perf_counter()
            decision: RoutingDecision | None = None
            reserved = False
            try:
                decision = self.router.route(
                    task,
                    exclude_agents=attempted,
                    additional_context=context,
                    available_agents=self.agents.keys(),
                    max_cost=self.max_cost,
                    max_latency_ms=self.max_latency_ms,
                )
                if decision.assigned_agent not in self.agents:
                    raise NoAvailableRouteError(f"Executor {decision.assigned_agent!r} is not injected.")
                self._reserve_budget(decision)
                reserved = True
                current = self.task_store.get_task(task.task_id)
                if current.status is TaskStatus.FAILED:
                    self.task_store.update_status(task.task_id, TaskStatus.READY)
                    current = self.task_store.get_task(task.task_id)
                if current.status is TaskStatus.READY:
                    self.task_store.assign_task(task.task_id, decision.assigned_agent)
                elif current.status is TaskStatus.ASSIGNED and current.assignee != decision.assigned_agent:
                    self.task_store.assign_task(task.task_id, decision.assigned_agent)
                current = self.task_store.get_task(task.task_id)
                if current.status is TaskStatus.ASSIGNED:
                    self.task_store.update_status(task.task_id, TaskStatus.RUNNING)
                self._send(MessageType.TASK_ASSIGN, decision.assigned_agent, task, {
                    "description": task.description,
                    "dependencies": [dep.task_id for dep in dependencies],
                    "attempt": attempt,
                    "model_id": decision.model_id,
                })
                agent_started = time.perf_counter()
                outcome = None
                active_metric_context = None
                invocation_succeeded = False
                try:
                    from src.evaluation.metrics import current_metric_context

                    active_metric_context = current_metric_context()
                    if active_metric_context is None:
                        outcome = self._invoke(self.agents[decision.assigned_agent], decision.assigned_agent, task, root, dependencies, context)
                    else:
                        with active_metric_context.collector.task_context(task.task_id, decision.assigned_agent):
                            outcome = self._invoke(self.agents[decision.assigned_agent], decision.assigned_agent, task, root, dependencies, context)
                    invocation_succeeded = True
                finally:
                    try:
                        if active_metric_context is not None:
                            active_metric_context.collector.record_agent(
                                agent_id=getattr(self.agents[decision.assigned_agent], "agent_id", decision.assigned_agent),
                                agent_role=decision.assigned_agent, task_id=task.task_id,
                                status="RETURNED" if invocation_succeeded else "FAILED",
                                duration_ms=(time.perf_counter() - agent_started) * 1000,
                                model_id=decision.model_id,
                            )
                            if outcome is not None and outcome.format_repair_attempts:
                                active_metric_context.collector.record_retry(outcome.format_repair_attempts)
                    except Exception:
                        pass
                self._release_reservation(decision, outcome)
                reserved = False
                if decision.assigned_agent == "critic":
                    outcome = self._run_critic_feedback_loop(
                        task, root, dependencies, context, outcome
                    )
                if cancellation.is_set():
                    self._record(ExecutionEvent(
                        _now(), task.task_id, "late_result_ignored", attempt,
                        decision.assigned_agent, decision.model_id,
                        round((time.perf_counter() - started) * 1000, 3),
                        outcome.prompt_tokens, outcome.completion_tokens,
                        decision.estimated_cost,
                    ))
                    return
                if outcome.needs_input:
                    question = outcome.clarification_question or "Уточните требования, необходимые для продолжения задачи."
                    needs_input_result = {
                        "status": "NEEDS_INPUT", "clarification_question": question,
                    }
                    if decision.assigned_agent == "critic":
                        needs_input_result["review"] = self._plain(outcome.value)
                    self.task_store.update_task(task.task_id, result=needs_input_result, error=None)
                    self.task_store.update_status(task.task_id, TaskStatus.NEEDS_INPUT)
                    self._record(ExecutionEvent(
                        _now(), task.task_id, "needs_input", attempt, decision.assigned_agent,
                        decision.model_id, round((time.perf_counter() - started) * 1000, 3),
                        outcome.prompt_tokens, outcome.completion_tokens, decision.estimated_cost,
                        fallback=decision.fallback,
                        route_reason="; ".join(decision.reasons) or None,
                        total_tokens=outcome.total_tokens,
                        finish_reason=outcome.finish_reason,
                        clarification_question=question,
                        failure_category="insufficient_context",
                        retry_allowed=False,
                        format_repair_attempts=outcome.format_repair_attempts,
                        review_performed=decision.assigned_agent == "critic",
                        critic_cycle=getattr(outcome.value, "cycle_count", None),
                    ))
                    self._send(MessageType.ERROR, self.coordinator_id, task, {
                        "status": "NEEDS_INPUT", "clarification_question": question,
                    })
                    return
                if outcome.task_completed is False:
                    raise IncompleteTaskResultError("Agent returned a result marked as incomplete.")
                if cancellation.is_set():
                    self._record(ExecutionEvent(
                        _now(), task.task_id, "late_result_ignored", attempt,
                        decision.assigned_agent, decision.model_id,
                        round((time.perf_counter() - started) * 1000, 3),
                        outcome.prompt_tokens, outcome.completion_tokens,
                        decision.estimated_cost,
                    ))
                    return
                if self.max_tokens is not None and self._prompt_tokens + self._completion_tokens > self.max_tokens:
                    raise ExecutionBudgetExceeded("Measured token usage exceeded the configured budget.")
                duration = (time.perf_counter() - started) * 1000
                self.task_store.update_task(task.task_id, result=outcome.value, error=None)
                self.task_store.update_status(task.task_id, TaskStatus.COMPLETED)
                self._record(ExecutionEvent(
                    _now(), task.task_id, "completed", attempt, decision.assigned_agent,
                    decision.model_id, round(duration, 3), outcome.prompt_tokens,
                    outcome.completion_tokens, decision.estimated_cost,
                    fallback=decision.fallback,
                    route_reason="; ".join(decision.reasons) or None,
                    total_tokens=outcome.total_tokens,
                    finish_reason=outcome.finish_reason,
                    format_repair_attempts=outcome.format_repair_attempts,
                    review_performed=decision.assigned_agent == "critic",
                    critic_cycle=getattr(outcome.value, "cycle_count", None),
                ))
                self._send(MessageType.RESULT, self.coordinator_id, task, {
                    "status": "COMPLETED", "result": self._plain(outcome.value),
                    "attempt": attempt,
                })
                return
            except Exception as error:
                retry_same_agent = (
                    isinstance(error, EmptyModelResponseError)
                    and error.retryable
                    and not same_agent_empty_retry_used
                    and attempt <= self.max_retries
                )
                transient_critic_retry = (
                    decision is not None
                    and decision.assigned_agent == "critic"
                    and type(error).__name__ == "CriticLLMError"
                    and self._failure_category(error) not in {"authentication", "rate_limit", "invalid_request"}
                    and attempt <= self.max_retries
                )
                retry_same_agent = retry_same_agent or transient_critic_retry
                if retry_same_agent:
                    same_agent_empty_retry_used = True
                if decision is not None and reserved:
                    if not retry_same_agent:
                        attempted.add(decision.assigned_agent)
                    self._release_reservation(decision, None)
                    reserved = False
                    if not self._llm_clients:
                        self._record_failed_usage(error)
                elif decision is not None:
                    if not retry_same_agent:
                        attempted.add(decision.assigned_agent)
                duration = (time.perf_counter() - started) * 1000
                message = f"{type(error).__name__}: {error}"
                if cancellation.is_set():
                    self._record(ExecutionEvent(_now(), task.task_id, "late_failure_ignored", attempt, error=message))
                    return
                if isinstance(error, CriticRejectedError):
                    current = self.task_store.get_task(task.task_id)
                    self.task_store.update_task(task.task_id, result=error.report, error=message)
                    if current.status is TaskStatus.RUNNING:
                        self.task_store.update_status(task.task_id, TaskStatus.FAILED)
                    self._errors.append(f"{task.task_id}: {message}")
                    self._record(ExecutionEvent(
                        _now(), task.task_id, "critic_rejected", attempt,
                        decision.assigned_agent if decision else None,
                        decision.model_id if decision else None,
                        round(duration, 3), error=message,
                        review_performed=True,
                        critic_cycle=getattr(error.report, "cycle_count", None),
                    ))
                    self._send(MessageType.ERROR, self.coordinator_id, task, {
                        "status": "REJECTED", "error": message,
                        "review": self._plain(error.report),
                    })
                    return
                if decision is not None and decision.assigned_agent == "critic" and task.task_id in self._review_history:
                    self.task_store.update_task(task.task_id, result={
                        "status": "FAILED", "review_history": self._review_history[task.task_id],
                    })
                budget_rejection = isinstance(error, ExecutionBudgetExceeded)
                time_limit = budget_rejection and error.details.get("limit") == "execution_time"
                failure_category = self._failure_category(error)
                format_failure = type(error).__name__ in {
                    "ResearchPlanParseError", "ResearchReportParseError",
                    "ResearchReportValidationError", "CriticReportParseError",
                    "CodingSolutionParseError", "CodingSolutionValidationError",
                    "AgentOutcomeParseError",
                }
                incomplete_failure = isinstance(error, IncompleteTaskResultError)
                route_failure = type(error).__name__ == "NoAvailableRouteError"
                event_name = (
                    "execution_deadline" if time_limit else
                    "budget_rejected" if budget_rejection else
                    "format_repair_failed" if format_failure else
                    "result_incomplete" if incomplete_failure else
                    "no_route" if route_failure else
                    "empty_response_retry" if retry_same_agent and isinstance(error, EmptyModelResponseError) else
                    "attempt_failed"
                )
                retry_candidate = retry_same_agent or (
                    attempt <= self.max_retries
                    and decision is not None
                    and not budget_rejection
                    and not format_failure
                    and not incomplete_failure
                    and failure_category != "authentication"
                    and failure_category != "invalid_request"
                )
                if retry_candidate and failure_category == "rate_limit":
                    retry_candidate = decision is not None and self._has_alternate_model(
                        task, decision.assigned_agent, decision.model_id
                    )
                retry_allowed = retry_candidate and self._consume_retry()
                failed_repairs = getattr(error, "repair_attempts", 0)
                if failed_repairs:
                    try:
                        from src.evaluation.metrics import current_metric_context

                        metric_context = current_metric_context()
                        if metric_context is not None:
                            metric_context.collector.record_retry(failed_repairs)
                    except Exception:
                        pass
                if retry_same_agent:
                    retry_reason = (
                        "Retry the same Critic after a transient API error."
                        if transient_critic_retry
                        else "Retry the same agent once after an empty completion."
                    )
                elif format_failure:
                    retry_reason = "Format repair was exhausted or unsafe; task will fail without fallback."
                elif incomplete_failure:
                    retry_reason = "Successful API response did not complete the task; fallback is disabled."
                elif budget_rejection:
                    retry_reason = "Budget rejection is terminal for this execution."
                elif failure_category == "authentication":
                    retry_reason = "Authentication failures are terminal and cannot be fixed by fallback."
                elif failure_category == "rate_limit" and not retry_candidate:
                    retry_reason = "No eligible fallback uses a different model; avoid repeating the rate-limited request."
                elif retry_candidate and not retry_allowed:
                    retry_reason = "The global retry budget has been exhausted."
                elif retry_allowed:
                    retry_reason = "The configured retry policy permits another attempt via routing."
                elif route_failure:
                    retry_reason = "Router found no eligible executor; no task attempt can be retried."
                else:
                    retry_reason = "The configured max_retries limit has been exhausted."
                self._record(ExecutionEvent(
                    _now(), task.task_id,
                    event_name, attempt,
                    decision.assigned_agent if decision else None,
                    decision.model_id if decision else None,
                    round(duration, 3), error=message,
                    estimated_cost=decision.estimated_cost if decision else None,
                    fallback=decision.fallback if decision else False,
                    route_reason=("; ".join(decision.reasons) or None) if decision else str(error),
                    prompt_tokens=getattr(error, "prompt_tokens", None),
                    completion_tokens=getattr(error, "completion_tokens", None),
                    total_tokens=getattr(error, "total_tokens", None),
                    finish_reason=getattr(error, "finish_reason", None),
                    budget_details=(dict(error.details) if isinstance(error, ExecutionBudgetExceeded) else None),
                    failure_category=("execution_time" if time_limit else "format" if format_failure else "incomplete_result" if incomplete_failure else "route" if route_failure else "budget" if budget_rejection else failure_category),
                    retry_allowed=retry_allowed,
                    format_repair_attempts=getattr(error, "repair_attempts", 0),
                    review_performed=(
                        False if decision is not None and decision.assigned_agent == "critic" else None
                    ),
                    retry_reason=retry_reason,
                ))
                if retry_allowed:
                    current = self.task_store.get_task(task.task_id)
                    if current.status is TaskStatus.RUNNING:
                        self.task_store.update_task(task.task_id, error=message)
                        self.task_store.update_status(task.task_id, TaskStatus.FAILED)
                    continue
                self._fail_task(task, message)
                return

    def _invoke(self, agent: Any, role: str, task: Task, root: Task, dependencies: list[Task], context: str) -> _AttemptResult:
        from src.agents.base_agent import AgentResult, BaseAgent

        if role == "research":
            result = self._call_with_reserved_format_retry(
                agent.run_research,
                self._with_context(task.description, context),
            )
        elif role == "math":
            result = agent.solve(self._with_context(task.description, context))
        elif role == "coding":
            result = self._call_with_reserved_format_retry(
                agent.run, task.description, additional_context=context or None
            )
            if isinstance(result, AgentResult):
                usage = self._usage(result)
                total_tokens = result.total_tokens
                finish_reason = result.finish_reason
                if not result.success:
                    if result.error_type == "EmptyModelResponseError":
                        raise EmptyModelResponseError(result)
                    raise AgentRequestFailed(result)
                if not result.content.strip() and not result.needs_input and result.task_completed is not False:
                    raise EmptyModelResponseError(result)
                result = agent.parse_solution(result)
                return _AttemptResult(
                    result, *usage, total_tokens, finish_reason,
                    task_completed=getattr(result, "task_completed", None),
                    needs_input=bool(getattr(result, "needs_input", False)),
                    clarification_question=getattr(result, "clarification_question", None),
                    format_repair_attempts=getattr(result, "format_repair_attempts", 0),
                )
            return self._structured_attempt(result)
        elif role == "critic":
            if not dependencies:
                raise TaskExecutionError("Critic task must depend on a result to review.")
            result = self._call_with_reserved_format_retry(
                agent.review,
                task=root.description,
                result=[self._plain(dep.result) for dep in dependencies],
                result_type="multi_agent_execution",
                agent_name="orchestrator",
                additional_materials={
                    "dependency_task_ids": [dep.task_id for dep in dependencies],
                    "dependency_tasks": [
                        {"task_id": dep.task_id, "agent": dep.assignee, "description": dep.description}
                        for dep in dependencies
                    ],
                },
            )
        else:
            prompt = self._with_context(task.description, context)
            if isinstance(agent, BaseAgent):
                result = self._call_with_reserved_format_retry(
                    self._run_generic_baseagent,
                    agent,
                    prompt,
                )
            else:
                result = agent.run(prompt)
            if isinstance(result, AgentResult):
                usage = self._usage(result)
                if not result.success:
                    if result.error_type == "EmptyModelResponseError":
                        raise EmptyModelResponseError(result)
                    raise AgentRequestFailed(result)
                if not result.content.strip() and not result.needs_input and result.task_completed is not False:
                    raise EmptyModelResponseError(result)
                if isinstance(agent, BaseAgent):
                    payload = self._parse_generic_agent_envelope(result.content)
                    return _AttemptResult(
                        payload["result"], *usage, result.total_tokens, result.finish_reason,
                        task_completed=payload["task_completed"],
                        needs_input=payload["needs_input"],
                        clarification_question=payload["clarification_question"],
                        format_repair_attempts=result.format_repair_attempts,
                    )
                return _AttemptResult(
                    result, *usage, result.total_tokens, result.finish_reason,
                    task_completed=result.task_completed,
                    needs_input=result.needs_input,
                    clarification_question=result.clarification_question,
                )
        usage = self._usage(result)
        attempt_result = self._structured_attempt(result)
        attempt_result.prompt_tokens, attempt_result.completion_tokens = usage
        attempt_result.total_tokens = getattr(result, "total_tokens", None)
        attempt_result.finish_reason = getattr(result, "finish_reason", None)
        return attempt_result

    @staticmethod
    def _generic_agent_system_prompt() -> str:
        """Return the status-envelope contract used for general-agent execution."""
        return (
            "Ты универсальный исполнитель подзадачи. Выполни задачу напрямую, "
            "используя переданный контекст и не создавая новый план. Верни строго "
            "JSON envelope: {\"task_completed\":true,\"needs_input\":false,"
            "\"clarification_question\":null,\"result\":...}. Если без ответа "
            "пользователя задачу нельзя решить, верни task_completed=false, "
            "needs_input=true, конкретный clarification_question и result=null. "
            "Не выдавай запрос уточнения или заглушку за завершённую работу."
        )

    @staticmethod
    def _parse_generic_agent_envelope(content: str) -> dict[str, Any]:
        """Parse and validate the status envelope returned by a general agent."""
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as error:
            raise AgentOutcomeParseError(
                "Generic agent result must be a JSON status envelope."
            ) from error
        fields = {"task_completed", "needs_input", "clarification_question", "result"}
        if not isinstance(payload, dict) or payload.keys() != fields:
            raise AgentOutcomeParseError(
                "Generic agent envelope must contain task_completed, needs_input, "
                "clarification_question, and result."
            )
        if (
            not isinstance(payload["task_completed"], bool)
            or not isinstance(payload["needs_input"], bool)
            or (payload["clarification_question"] is not None
                and not isinstance(payload["clarification_question"], str))
        ):
            raise AgentOutcomeParseError("Generic agent envelope has invalid status field types.")
        if payload["needs_input"] and (
            payload["task_completed"]
            or not (payload["clarification_question"] or "").strip()
        ):
            raise AgentOutcomeParseError(
                "A needs_input envelope must be incomplete and include a question."
            )
        if not payload["needs_input"] and payload["clarification_question"] is not None:
            raise AgentOutcomeParseError(
                "clarification_question must be null unless needs_input is true."
            )
        if payload["task_completed"] and (
            payload["result"] is None
            or isinstance(payload["result"], str) and not payload["result"].strip()
        ):
            raise AgentOutcomeParseError("A completed generic result cannot be empty.")
        return payload

    def _run_generic_baseagent(
        self,
        agent: Any,
        prompt: str,
        *,
        max_format_retries: int = 0,
    ) -> Any:
        """Run a general task and perform at most one budgeted envelope repair."""
        from src.agents.base_agent import BaseAgent

        system_prompt = self._generic_agent_system_prompt()
        result = BaseAgent.run(agent, prompt, system_prompt=system_prompt)
        if not result.success:
            return result
        try:
            self._parse_generic_agent_envelope(result.content)
            return result
        except AgentOutcomeParseError as parse_error:
            if max_format_retries == 0:
                return result
            initial_error = parse_error

        repair_prompt = (
            f"Исходная задача:\n{prompt}\n\n"
            "Предыдущий ответ не прошёл проверку статуса. Исправь только формат: "
            "сохрани содержательный ответ и верни один JSON-объект строго по системной "
            "схеме. Если ответ требует уточнения, укажи это в полях needs_input и "
            "clarification_question. Не добавляй Markdown или текст вне JSON.\n\n"
            f"Ошибка проверки: {initial_error}\nПредыдущий ответ:\n{result.content}"
        )
        repaired = BaseAgent.run(agent, repair_prompt, system_prompt=system_prompt)
        if not repaired.success:
            error = AgentRequestFailed(repaired)
            error.repair_attempts = 1
            raise error
        try:
            self._parse_generic_agent_envelope(repaired.content)
        except AgentOutcomeParseError as repair_error:
            repair_error.repair_attempts = 1
            raise
        return replace(
            repaired,
            prompt_tokens=_sum_token_counts(result.prompt_tokens, repaired.prompt_tokens),
            completion_tokens=_sum_token_counts(
                result.completion_tokens, repaired.completion_tokens
            ),
            total_tokens=_sum_token_counts(result.total_tokens, repaired.total_tokens),
            format_repair_attempts=1,
        )
    def _run_critic_feedback_loop(
        self,
        review_task: Task,
        root: Task,
        dependencies: list[Task],
        context: str,
        initial: _AttemptResult,
    ) -> _AttemptResult:
        """Apply bounded Critic feedback to the reviewed executors and re-review."""
        report = initial.value
        history: list[dict[str, Any]] = [{"cycle": 0, "review": self._plain(report), "revisions": []}]
        self._review_history[review_task.task_id] = history
        prompt_tokens = initial.prompt_tokens
        completion_tokens = initial.completion_tokens
        repairs = initial.format_repair_attempts

        def terminal_record() -> CriticExecutionRecord:
            return CriticExecutionRecord(report=report, iterations=list(history))

        verdict = getattr(report, "verdict", None)
        if verdict == "REJECTED":
            self._record(ExecutionEvent(
                _now(), review_task.task_id, "critic_review_rejected", 1,
                agent="critic", review_performed=True, critic_cycle=0,
                retry_allowed=self.max_critic_cycles > 0,
                retry_reason=("The executor may revise this result."
                              if self.max_critic_cycles > 0 else
                              "The configured Critic cycle limit is zero."),
            ))
        if verdict == "NEEDS_INPUT":
            question = getattr(report, "clarification_question", None) or getattr(report, "summary", None)
            return _AttemptResult(
                terminal_record(), prompt_tokens, completion_tokens,
                initial.total_tokens, initial.finish_reason, needs_input=True,
                clarification_question=question, format_repair_attempts=repairs,
            )

        for cycle in range(1, self.max_critic_cycles + 1):
            if verdict != "REJECTED":
                break
            issues = getattr(report, "issues", None) or []
            if not issues:
                break
            if not self._consume_retry():
                history.append({
                    "cycle": cycle, "event": "retry_budget_exhausted",
                    "message": "The global retry budget is exhausted; no revision was requested.",
                })
                break

            feedback_items = []
            target_ids: set[str] = set()
            for issue in issues:
                item = self._plain(issue)
                if isinstance(item, Mapping):
                    target_id = item.get("target_task_id")
                    if isinstance(target_id, str) and target_id.strip():
                        target_ids.add(target_id)
                    feedback_items.append({
                        key: item.get(key)
                        for key in ("category", "severity", "description", "location", "recommendation", "target_task_id")
                        if item.get(key) is not None
                    })
                else:
                    feedback_items.append(str(item))
            feedback = (
                "Предыдущий результат был отклонён Critic. Исправь именно перечисленные "
                "проблемы, сохрани исходные требования и верни полный обновлённый результат.\n"
                "Замечания Critic:\n" + json.dumps(feedback_items, ensure_ascii=False, default=str)
                + "\nКраткий итог проверки:\n" + str(getattr(report, "summary", ""))
            )
            revisions: list[dict[str, Any]] = []
            any_changed = False
            valid_targets = {dependency.task_id for dependency in dependencies}
            if target_ids and not target_ids.issubset(valid_targets):
                history.append({
                    "cycle": cycle, "event": "invalid_repair_target",
                    "target_task_ids": sorted(target_ids),
                    "available_task_ids": sorted(valid_targets),
                })
                break
            selected_dependencies = [
                dependency for dependency in dependencies
                if not target_ids or dependency.task_id in target_ids
            ]
            for dependency in selected_dependencies:
                role = dependency.assignee
                agent = self.agents.get(role or "")
                if agent is None:
                    continue
                previous = dependency.result
                revision_context = (
                    "ТВОЯ ПРЕДЫДУЩАЯ ВЕРСИЯ (исправь её, не начинай с пустой заглушки):\n"
                    + json.dumps(self._plain(previous), ensure_ascii=False, default=str)
                    + "\n\n" + feedback
                )
                revision_task = Task(
                    task_id=dependency.task_id,
                    description=dependency.description,
                    task_type=dependency.task_type,
                    priority=dependency.priority,
                    status=TaskStatus.READY,
                    parent_task_id=dependency.parent_task_id,
                    assignee=role,
                    dependencies=list(dependency.dependencies),
                )
                decision = self.router.route(
                    revision_task,
                    additional_context=revision_context,
                    available_agents=self.agents.keys(),
                    max_cost=self.max_cost,
                    max_latency_ms=self.max_latency_ms,
                )
                self._reserve_budget(decision)
                revised: _AttemptResult | None = None
                try:
                    revised = self._invoke_with_metrics(
                        agent, role, revision_task, root, [], revision_context, decision.model_id
                    )
                finally:
                    self._release_reservation(decision, revised)
                if revised is None:
                    continue
                if revised.needs_input:
                    question = revised.clarification_question or "Уточните требования для доработки результата."
                    history.append({
                        "cycle": cycle, "event": "needs_input", "task_id": dependency.task_id,
                        "question": question, "revisions": revisions,
                    })
                    record = CriticExecutionRecord(report=report, iterations=list(history))
                    return _AttemptResult(
                        record, prompt_tokens + revised.prompt_tokens,
                        completion_tokens + revised.completion_tokens,
                        needs_input=True, clarification_question=question,
                    )
                if revised.task_completed is False:
                    raise IncompleteTaskResultError(
                        f"Executor {role!r} returned an incomplete revision for {dependency.task_id!r}."
                    )
                current_value = revised.value
                changed = self._plain(current_value) != self._plain(previous)
                any_changed = any_changed or changed
                revisions.append({
                    "task_id": dependency.task_id,
                    "agent": role,
                    "previous_result": self._plain(previous),
                    "revised_result": self._plain(current_value),
                    "changed": changed,
                })
                if changed:
                    self.task_store.update_task(dependency.task_id, result=current_value, error=None)
                prompt_tokens += revised.prompt_tokens
                completion_tokens += revised.completion_tokens
                repairs += revised.format_repair_attempts

            history.append({"cycle": cycle, "review": self._plain(report), "revisions": revisions})
            self._record(ExecutionEvent(
                _now(), review_task.task_id, "critic_refinement", cycle,
                agent="critic", review_performed=True, critic_cycle=cycle,
                retry_allowed=any_changed,
                retry_reason=(
                    "Revisions were returned; Critic will review the new versions."
                    if any_changed else
                    "No revised result changed; the same result will not be reviewed again."
                ),
            ))
            if not any_changed:
                history.append({
                    "cycle": cycle, "event": "unchanged_result",
                    "message": "The executor returned no changed result; Critic was not asked to review it again.",
                })
                break

            refreshed_dependencies = [
                self.task_store.get_task(dependency.task_id) for dependency in dependencies
            ]
            review_decision = self.router.route(
                review_task,
                additional_context=self._dependency_context(refreshed_dependencies),
                available_agents=self.agents.keys(),
                max_cost=self.max_cost,
                max_latency_ms=self.max_latency_ms,
            )
            self._reserve_budget(review_decision)
            reviewed: _AttemptResult | None = None
            try:
                reviewed = self._invoke_with_metrics(
                    self.agents[review_decision.assigned_agent],
                    review_decision.assigned_agent,
                    review_task,
                    root,
                    refreshed_dependencies,
                    self._dependency_context(refreshed_dependencies),
                    review_decision.model_id,
                )
            finally:
                self._release_reservation(review_decision, reviewed)
            if reviewed is None:
                break
            prompt_tokens += reviewed.prompt_tokens
            completion_tokens += reviewed.completion_tokens
            repairs += reviewed.format_repair_attempts
            report = reviewed.value
            verdict = getattr(report, "verdict", None)
            history.append({"cycle": cycle, "review": self._plain(report), "revisions": []})
            self._review_history[review_task.task_id] = history
            if verdict == "REJECTED":
                self._record(ExecutionEvent(
                    _now(), review_task.task_id, "critic_review_rejected", cycle + 1,
                    agent="critic", review_performed=True, critic_cycle=cycle,
                    retry_allowed=cycle < self.max_critic_cycles,
                    retry_reason=("Another bounded revision cycle remains."
                                  if cycle < self.max_critic_cycles else
                                  "The configured Critic cycle limit was reached."),
                ))
            if verdict == "NEEDS_INPUT":
                question = getattr(report, "clarification_question", None) or getattr(report, "summary", None)
                return _AttemptResult(
                    terminal_record(), prompt_tokens, completion_tokens,
                    needs_input=True, clarification_question=question,
                    format_repair_attempts=repairs,
                )

        self._review_history[review_task.task_id] = history

        record = terminal_record()
        if verdict == "REJECTED":
            raise CriticRejectedError(
                f"Critic rejected the result after {len(history) - 1} bounded feedback iteration(s).",
                record,
            )
        return _AttemptResult(
            record, prompt_tokens, completion_tokens,
            format_repair_attempts=repairs,
        )

    def _invoke_with_metrics(
        self, agent: Any, role: str, task: Task, root: Task,
        dependencies: list[Task], context: str, model_id: str | None,
    ) -> _AttemptResult:
        """Invoke a reworking/reviewing agent with its own active-time span."""
        from src.evaluation.metrics import current_metric_context

        active = current_metric_context()
        started = time.perf_counter()
        outcome: _AttemptResult | None = None
        try:
            if active is None:
                outcome = self._invoke(agent, role, task, root, dependencies, context)
            else:
                with active.collector.task_context(task.task_id, role):
                    outcome = self._invoke(agent, role, task, root, dependencies, context)
            return outcome
        finally:
            if active is not None:
                try:
                    active.collector.record_agent(
                        agent_id=getattr(agent, "agent_id", role), agent_role=role,
                        task_id=task.task_id,
                        status="RETURNED" if outcome is not None else "FAILED",
                        duration_ms=(time.perf_counter() - started) * 1000,
                        model_id=model_id,
                    )
                    if outcome is not None and outcome.format_repair_attempts:
                        active.collector.record_retry(outcome.format_repair_attempts)
                except Exception:
                    pass

    @classmethod
    def _structured_attempt(cls, result: Any) -> _AttemptResult:
        return _AttemptResult(
            value=result,
            task_completed=getattr(result, "task_completed", None),
            needs_input=bool(getattr(result, "needs_input", False)),
            clarification_question=getattr(result, "clarification_question", None),
            format_repair_attempts=getattr(result, "format_repair_attempts", 0),
        )

    @staticmethod
    def _call_with_format_retry_limit(method: Any, *args: Any, **kwargs: Any) -> Any:
        """Pass the retry budget to current agent APIs while preserving older adapters."""
        try:
            signature = inspect.signature(method)
        except (TypeError, ValueError):
            kwargs.pop("max_format_retries", None)
            return method(*args, **kwargs)
        accepts_keyword = "max_format_retries" in signature.parameters or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        if not accepts_keyword:
            kwargs.pop("max_format_retries", None)
        return method(*args, **kwargs)

    def _call_with_reserved_format_retry(self, method: Any, *args: Any, **kwargs: Any) -> Any:
        """Charge an agent-level format repair to the shared retry allowance."""
        retry_slot = self.max_retries > 0 and self._reserve_retry_slot()
        kwargs["max_format_retries"] = 1 if retry_slot else 0
        try:
            result = self._call_with_format_retry_limit(method, *args, **kwargs)
        except Exception as error:
            self._finish_retry_slot(retry_slot, getattr(error, "repair_attempts", 0) > 0)
            raise
        self._finish_retry_slot(
            retry_slot, getattr(result, "format_repair_attempts", 0) > 0
        )
        return result

    def _reserve_budget(self, decision: RoutingDecision) -> None:
        if self._llm_clients:
            # Real LLMClient requests receive exact per-request reservations below.
            return
        with self._lock:
            if self.max_calls is not None and self._calls >= self.max_calls:
                raise ExecutionBudgetExceeded(
                    f"Agent invocation blocked: calls_used={self._calls}, max_calls={self.max_calls}.",
                    details={"calls_used": self._calls, "max_calls": self.max_calls},
                )
            token_reservation = decision.context_estimate + decision.estimated_output_tokens
            token_used = self._prompt_tokens + self._completion_tokens
            if self.max_tokens is not None and token_used + self._reserved_tokens + token_reservation > self.max_tokens:
                raise ExecutionBudgetExceeded(
                    "Agent invocation blocked by estimated token budget: "
                    f"used={token_used}, reserved={self._reserved_tokens}, "
                    f"next_estimate={token_reservation}, max_tokens={self.max_tokens}.",
                    details={"used_tokens": token_used, "reserved_tokens": self._reserved_tokens,
                             "next_estimate": token_reservation, "max_tokens": self.max_tokens},
                )
            cost = decision.estimated_cost or 0.0
            if self.max_cost is not None and self._estimated_cost + self._reserved_cost + cost > self.max_cost:
                raise ExecutionBudgetExceeded("Known estimated cost budget would be exceeded.")
            self._calls += 1
            self._reserved_tokens += token_reservation
            self._reserved_cost += cost

    def _release_reservation(self, decision: RoutingDecision, outcome: _AttemptResult | None) -> None:
        if self._llm_clients:
            return
        with self._lock:
            self._reserved_tokens = max(
                0,
                self._reserved_tokens - decision.context_estimate - decision.estimated_output_tokens,
            )
            self._reserved_cost = max(0.0, self._reserved_cost - (decision.estimated_cost or 0.0))
            if outcome is not None:
                self._prompt_tokens += outcome.prompt_tokens
                self._completion_tokens += outcome.completion_tokens
            self._estimated_cost += decision.estimated_cost or 0.0
            if decision.estimated_cost is None:
                self._unknown_cost_requests += 1

    def _record_failed_usage(self, error: Exception) -> None:
        prompt = getattr(error, "prompt_tokens", None)
        completion = getattr(error, "completion_tokens", None)
        with self._lock:
            if isinstance(prompt, int) and not isinstance(prompt, bool):
                self._prompt_tokens += prompt
            else:
                prompt = 0
            if isinstance(completion, int) and not isinstance(completion, bool):
                self._completion_tokens += completion
            else:
                completion = 0
            self._budget_tokens_used += prompt + completion
            if self.max_tokens is not None and self._budget_tokens_used > self.max_tokens:
                self._budget_violation = "Measured failed-response token usage exceeded the configured budget."

    def _skip_blocked_tasks(self, pending: set[str], by_id: Mapping[str, Task]) -> None:
        changed = True
        while changed:
            changed = False
            for identifier in list(pending):
                task = by_id[identifier]
                deps = [self.task_store.get_task(dep_id) for dep_id in task.dependencies]
                failed = [dep for dep in deps if dep.status in {TaskStatus.FAILED, TaskStatus.SKIPPED, TaskStatus.CANCELLED, TaskStatus.NEEDS_INPUT}]
                if failed:
                    reason = "Blocked by dependencies without completed results: " + ", ".join(dep.task_id for dep in failed)
                    self._mark_skipped(task, reason)
                    pending.remove(identifier)
                    changed = True

    def _dependencies_completed(self, task: Task) -> bool:
        return all(self.task_store.get_task(dep_id).status is TaskStatus.COMPLETED for dep_id in task.dependencies)

    def _mark_skipped(self, task: Task, reason: str) -> None:
        current = self.task_store.get_task(task.task_id)
        self.task_store.update_task(task.task_id, error=reason)
        if current.status not in {TaskStatus.SKIPPED, TaskStatus.CANCELLED, TaskStatus.COMPLETED}:
            self.task_store.update_status(task.task_id, TaskStatus.SKIPPED)
        self._errors.append(f"{task.task_id}: {reason}")
        self._record(ExecutionEvent(
            _now(), task.task_id, "skipped", agent=task.assignee, error=reason,
            failure_category="dependency_not_completed",
            route_reason=reason,
            review_performed=False if task.task_type is TaskType.REVIEW else None,
            retry_allowed=False,
        ))
        self._send(MessageType.ERROR, self.coordinator_id, task, {"status": "SKIPPED", "error": reason})

    def _fail_task(self, task: Task, message: str) -> None:
        current = self.task_store.get_task(task.task_id)
        self.task_store.update_task(task.task_id, error=message)
        if current.status not in {TaskStatus.FAILED, TaskStatus.SKIPPED, TaskStatus.CANCELLED, TaskStatus.COMPLETED}:
            self.task_store.update_status(task.task_id, TaskStatus.FAILED)
        self._errors.append(f"{task.task_id}: {message}")
        self._send(MessageType.ERROR, self.coordinator_id, task, {"status": "FAILED", "error": message})

    def _fail_running_task(self, task: Task, message: str, *, event: str) -> None:
        self._fail_task(task, message)
        self._record(ExecutionEvent(_now(), task.task_id, event, error=message))

    def _cancel_task(self, task: Task, message: str, *, event: str) -> None:
        current = self.task_store.get_task(task.task_id)
        self.task_store.update_task(task.task_id, error=message)
        if current.status not in {TaskStatus.CANCELLED, TaskStatus.FAILED, TaskStatus.SKIPPED, TaskStatus.COMPLETED}:
            self.task_store.update_status(task.task_id, TaskStatus.CANCELLED)
        self._errors.append(f"{task.task_id}: {message}")
        self._record(ExecutionEvent(
            _now(), task.task_id, event, agent=current.assignee, error=message,
            failure_category="execution_time", retry_allowed=False,
            retry_reason="The execution deadline was reached.",
        ))

    def _consume_retry(self) -> bool:
        with self._lock:
            if self._retries + self._reserved_retries >= self.max_total_retries:
                return False
            self._retries += 1
        try:
            from src.evaluation.metrics import current_metric_context

            metric_context = current_metric_context()
            if metric_context is not None:
                metric_context.collector.record_retry()
        except Exception:
            pass
        return True

    @staticmethod
    def _failure_category(error: Exception) -> str:
        """Classify normalized and wrapped provider errors for retry policy/history."""
        error_type = str(getattr(error, "error_type", type(error).__name__)).lower()
        message = str(error).lower()
        combined = error_type + " " + message
        if "missingapikey" in combined or "authenticationerror" in combined or "unauthorized" in combined:
            return "authentication"
        if "ratelimiterror" in combined or "rate limit" in combined or "free-models-per-" in combined:
            return "rate_limit"
        if "timeout" in combined or "timed out" in combined:
            return "timeout"
        if "networkerror" in combined or "connectionerror" in combined or "could not connect" in combined:
            return "network"
        status_match = re.search(r"http\s+(\d{3})", combined)
        if status_match and 400 <= int(status_match.group(1)) < 500 and int(status_match.group(1)) not in {408, 429}:
            return "invalid_request"
        if "llmapierror" in combined or "api error" in combined:
            return "provider_api"
        return "provider_or_agent"

    def _has_alternate_model(self, task: Task, failed_agent: str, failed_model: str) -> bool:
        try:
            alternate = self.router.route(
                task,
                exclude_agents={failed_agent},
                available_agents=self.agents.keys(),
                max_cost=self.max_cost,
                max_latency_ms=self.max_latency_ms,
            )
        except Exception:
            return False
        return alternate.model_id != failed_model

    def _reserve_retry_slot(self) -> bool:
        with self._lock:
            if self._retries + self._reserved_retries >= self.max_total_retries:
                return False
            self._reserved_retries += 1
            return True

    def _finish_retry_slot(self, reserved: bool, used: bool) -> None:
        if not reserved:
            return
        with self._lock:
            self._reserved_retries = max(0, self._reserved_retries - 1)
            if used:
                self._retries += 1

    def _remaining_time(self) -> float | None:
        if self._deadline is None:
            return None
        return max(0.0, self._deadline - time.perf_counter())

    def _time_exceeded(self) -> bool:
        remaining = self._remaining_time()
        return remaining is not None and remaining <= 0

    def _time_limit_message(self) -> str:
        return f"Maximum execution time exceeded ({self.max_execution_time} seconds)."

    def _dependency_context(self, dependencies: list[Task]) -> str:
        if not dependencies:
            return ""
        payload = [{"task_id": dep.task_id, "description": dep.description, "result": self._plain(dep.result)} for dep in dependencies]
        return "Результаты зависимых задач (используй как входные данные):\n" + json.dumps(payload, ensure_ascii=False, default=str)

    def _send(self, message_type: MessageType, receiver: str, task: Task, payload: dict[str, Any]) -> None:
        try:
            self.message_bus.send(Message(
                sender=self.coordinator_id,
                receiver=receiver,
                message_type=message_type,
                task_id=task.task_id,
                parent_task_id=task.parent_task_id,
                payload=payload,
            ))
        except Exception as error:
            self._record(ExecutionEvent(_now(), task.task_id, "message_delivery_failed", error=f"{type(error).__name__}: {error}"))

    def _register_bus_agents(self) -> None:
        registered = set(self.message_bus.get_registered_agents())
        for agent_id in {self.coordinator_id, *self.agents.keys()}:
            if agent_id not in registered:
                self.message_bus.register_agent(agent_id)

    def _record(self, event: ExecutionEvent) -> None:
        with self._lock:
            self._history.append(event)

    def _install_request_hooks(self) -> None:
        if not self._llm_clients:
            return
        from src.llm.client import bind_request_hooks
        self._hook_stack.append(bind_request_hooks(self._guard_llm_request, self._observe_llm_response))

    def _restore_request_hooks(self) -> None:
        if not self._hook_stack:
            return
        from src.llm.client import reset_request_hooks
        reset_request_hooks(self._hook_stack.pop())

    def metrics_snapshot(self, *, duration_ms: float | None = None) -> dict[str, Any]:
        """Return current budget totals, including calls outside ``execute``."""
        with self._lock:
            return {
                "calls": self._calls,
                "call_count_mode": "api_requests" if self._llm_clients else "agent_invocation_estimate",
                "prompt_tokens": self._prompt_tokens,
                "completion_tokens": self._completion_tokens,
                "estimated_tokens": self._estimated_tokens,
                "estimated_cost": round(self._estimated_cost, 9),
                "duration_ms": round(duration_ms, 3) if duration_ms is not None else None,
                "budget_violation": self._budget_violation,
                "unknown_cost_requests": self._unknown_cost_requests,
                "retries": self._retries,
                "max_calls": self.max_calls,
                "max_tokens": self.max_tokens,
                "max_execution_time": self.max_execution_time,
                "max_total_retries": self.max_total_retries,
                "retry_budget_remaining": max(
                    0, self.max_total_retries - self._retries - self._reserved_retries
                ),
                "max_critic_cycles": self.max_critic_cycles,
                "remaining_execution_time": self._remaining_time(),
            }

    def _guard_llm_request(self, model_id: str, messages: list[dict[str, Any]], params: dict[str, Any]) -> dict[str, Any]:
        if self._time_exceeded():
            raise ExecutionBudgetExceeded(
                self._time_limit_message(),
                details={"limit": "execution_time", "max_execution_time": self.max_execution_time},
            )
        text = "\n".join(
            str(message.get("content", "")) for message in messages if isinstance(message, Mapping)
        )
        prompt_bytes = len(text.encode("utf-8"))
        # Provider usage in this project has been up to ~1.8 prompt tokens per
        # UTF-8 byte for the configured models. Keep a 2x safety margin because
        # there is no shared tokenizer across the OpenRouter model catalog.
        prompt_estimate = max(1, prompt_bytes * 2)
        output_key = "max_completion_tokens" if "max_completion_tokens" in params else "max_tokens"
        raw_output_limit = params.get(output_key, 0)
        requested_output = (
            raw_output_limit
            if isinstance(raw_output_limit, int) and not isinstance(raw_output_limit, bool)
            else 0
        )
        output_estimate = requested_output
        adjustments: dict[str, int] = {}
        model = next((model for model in self.router.model_registry.models.values() if model.model_id == model_id), None)
        cost = None
        input_rate = output_rate = None
        if model is not None:
            input_rate = model.cost.input_per_1m_tokens
            output_rate = model.cost.output_per_1m_tokens
        with self._lock:
            if self.max_calls is not None and self._calls >= self.max_calls:
                raise ExecutionBudgetExceeded(
                    f"LLM request blocked: call budget is exhausted ({self._calls}/{self.max_calls}).",
                    details={"calls_used": self._calls, "max_calls": self.max_calls},
                )
            if self.max_tokens is not None:
                used = self._budget_tokens_used
                reserved = self._reserved_tokens
                remaining = self.max_tokens - used - reserved
                available_output = remaining - prompt_estimate
                minimum_output = min(128, requested_output) if requested_output > 0 else 128
                if available_output < minimum_output:
                    raise ExecutionBudgetExceeded(
                        "LLM request blocked by token budget: "
                        f"used={used}, reserved={reserved}, prompt_upper_bound={prompt_estimate}, "
                        f"remaining={remaining}, requested_output={requested_output}, "
                        f"max_tokens={self.max_tokens}; at least {minimum_output} output tokens "
                        "cannot be safely reserved.",
                        details={
                            "used_tokens": used,
                            "reserved_tokens": reserved,
                            "prompt_upper_bound": prompt_estimate,
                            "remaining_tokens": remaining,
                            "requested_output_tokens": requested_output,
                            "available_output_tokens": max(0, available_output),
                            "max_tokens": self.max_tokens,
                        },
                    )
                if requested_output == 0 or requested_output > available_output:
                    output_estimate = available_output
                    adjustments[output_key] = output_estimate
            token_reservation = prompt_estimate + output_estimate
            if input_rate is not None and output_rate is not None:
                cost = (prompt_estimate * input_rate + output_estimate * output_rate) / 1_000_000
            if self.max_cost is not None and cost is None:
                raise ExecutionBudgetExceeded(
                    f"LLM request blocked: cannot enforce max_cost={self.max_cost}; "
                    f"price metadata for model {model_id!r} is unknown."
                )
            if self.max_cost is not None and self._estimated_cost + self._reserved_cost + cost > self.max_cost:
                raise ExecutionBudgetExceeded(
                    "LLM request blocked by cost budget: "
                    f"spent_or_reserved={self._estimated_cost + self._reserved_cost:.9f}, "
                    f"next_request_estimate={cost:.9f}, max_cost={self.max_cost:.9f}."
                )
            self._calls += 1
            if cost is None:
                self._unknown_cost_requests += 1
            self._reserved_tokens += token_reservation
            self._reserved_cost += cost or 0.0
        reservation = {
            "prompt_estimate": prompt_estimate,
            "prompt_bytes": prompt_bytes,
            "output_estimate": output_estimate,
            "token_reservation": token_reservation,
            "estimated_cost": cost,
            "input_rate": input_rate,
            "output_rate": output_rate,
        }
        if adjustments:
            reservation["generation_params"] = adjustments
        return reservation

    def _observe_llm_response(
        self,
        _model_id: str,
        response: Any,
        _duration_ms: float,
        error: Exception | None,
        reservation: Any,
    ) -> None:
        if not isinstance(reservation, Mapping):
            return
        with self._lock:
            self._reserved_tokens = max(0, self._reserved_tokens - reservation["token_reservation"])
            self._reserved_cost = max(0.0, self._reserved_cost - (reservation["estimated_cost"] or 0.0))
            prompt = getattr(response, "prompt_tokens", None) if response is not None else None
            completion = getattr(response, "completion_tokens", None) if response is not None else None
            prompt_actual = isinstance(prompt, int) and not isinstance(prompt, bool)
            completion_actual = isinstance(completion, int) and not isinstance(completion, bool)
            if prompt_actual:
                self._prompt_tokens += prompt
                prompt_budget = prompt
            else:
                prompt = None
                prompt_budget = reservation["prompt_estimate"]
                self._estimated_tokens += prompt_budget
            if completion_actual:
                self._completion_tokens += completion
                completion_budget = completion
            else:
                completion = None
                completion_budget = reservation["output_estimate"]
                self._estimated_tokens += completion_budget
            budget_tokens = prompt_budget + completion_budget
            self._budget_tokens_used += budget_tokens
            input_rate = reservation["input_rate"]
            output_rate = reservation["output_rate"]
            if error is not None or not prompt_actual or not completion_actual or input_rate is None or output_rate is None:
                cost = reservation["estimated_cost"] or 0.0
            else:
                cost = (prompt * input_rate + completion * output_rate) / 1_000_000
            self._estimated_cost += cost
            if self.max_tokens is not None and self._budget_tokens_used + self._reserved_tokens > self.max_tokens:
                self._budget_violation = "Measured token usage exceeded the configured budget."
            if self.max_cost is not None and self._estimated_cost + self._reserved_cost > self.max_cost:
                self._budget_violation = "Measured/estimated model cost exceeded the configured budget."

    @staticmethod
    def _usage(value: Any) -> tuple[int, int]:
        prompt = getattr(value, "prompt_tokens", 0) or 0
        completion = getattr(value, "completion_tokens", 0) or 0
        return (prompt if isinstance(prompt, int) else 0, completion if isinstance(completion, int) else 0)

    @classmethod
    def _plain(cls, value: Any) -> Any:
        if is_dataclass(value) and not isinstance(value, type):
            return cls._plain(asdict(value))
        if isinstance(value, Mapping):
            return {str(key): cls._plain(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [cls._plain(item) for item in value]
        if hasattr(value, "__dict__") and not isinstance(value, type):
            return cls._plain(vars(value))
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return str(value)

    @staticmethod
    def _with_context(description: str, context: str) -> str:
        return description + ("\n\n" + context if context else "")

    @staticmethod
    def _review_status(tasks: list[Task]) -> str:
        reviews = [task for task in tasks if task.task_type is TaskType.REVIEW]
        if not reviews:
            return "NOT_REQUIRED"
        if any(
            task.status is TaskStatus.FAILED
            and getattr(task.result, "verdict", None) == "REJECTED"
            for task in reviews
        ):
            return "REJECTED"
        if any(task.status is TaskStatus.FAILED for task in reviews):
            return "FAILED"
        if any(task.status is TaskStatus.NEEDS_INPUT for task in reviews):
            return "NEEDS_INPUT"
        if all(task.status is TaskStatus.COMPLETED for task in reviews):
            return "APPROVED"
        if any(task.status is TaskStatus.SKIPPED for task in reviews):
            return "SKIPPED"
        return "PENDING"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sum_token_counts(first: int | None, second: int | None) -> int | None:
    """Add token counts from bounded format-repair requests when known."""
    if first is None and second is None:
        return None
    return (first or 0) + (second or 0)


__all__ = [
    "CriticRejectedError", "EmptyModelResponseError", "ExecutionBudgetExceeded", "ExecutionEvent", "IncompleteTaskResultError", "TaskExecutionError",
    "TaskExecutionReport", "TaskExecutor",
]
