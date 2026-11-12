"""Sequential task planning and coordination across specialized agents."""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

from src.agents.base_agent import AgentInputError, AgentResult, BaseAgent
from src.agents.config import AgentConfig, AgentConfigRegistry
from src.coordination import (
    ExecutionBudgetExceeded,
    TaskExecutionReport,
    TaskExecutor,
    Task,
    TaskPriority,
    TaskStatus,
    TaskStore,
    TaskStoreError,
    TaskType,
)
from src.llm import LLMClient
from src.communication import MessageBus


ORCHESTRATOR_SYSTEM_PROMPT = """\
Ты Orchestrator многоагентной системы. Координируй работу, не выполняя вместо
специализированных агентов исследование, математические вычисления или написание
кода. Выбирай только нужные роли из списка research, coding, math, critic.

Разбей запрос на минимальный набор конкретных подзадач. Если результат одной
подзадачи нужен другой, укажи зависимость. Используй результаты предшественников
в следующих заданиях. Не назначай несуществующих исполнителей и не добавляй
проверку Critic без причины; включай её, когда задача просит проверить/оценить
результат или результат существенно важен. Допустим план с одним исполнителем.
Для простого общего вопроса, которому не нужны специализированные агенты,
верни пустой список subtasks: координатор ответит напрямую на этапе сборки. Если
существенных требований не хватает, верни needs_input=true, пустой subtasks и
один конкретный clarification_question. Иначе используй false и null.

Верни только JSON следующей формы:
{"needs_input":false,"clarification_question":null,"subtasks":[{"id":"task_1","description":"...","agent":"coding","depends_on":[]}]}
Для краткого запроса создавай одну подходящую подзадачу. Не используй агента
orchestrator и не добавляй поля.
"""

TASK_DECOMPOSITION_SYSTEM_PROMPT = """\
Ты формируешь план задач для Adaptive Multi-Agent LLM System. Обычно разбей запрос на
минимальный набор конкретных подзадач; если ключевых требований не хватает, запроси ввод.
Назначай только существующие роли: research, coding, math, critic или
orchestrator. Для общей задачи используй task_type GENERAL и assigned_agent
orchestrator. Соотнеси тип и роль: RESEARCH/research, CODING/coding, MATH/math,
REVIEW/critic, GENERAL/orchestrator.

Укажи зависимости только когда результат одной подзадачи нужен другой.
Независимые задачи оставляй без зависимости. ID должны быть короткими,
уникальными строками. Выбирай осмысленные приоритеты HIGH, MEDIUM или LOW:
учитывай срочность пользователя, критичность результата и влияние на
последующие задачи; не ставь всем задачам одинаковый приоритет автоматически.

План должен включать только реально необходимые роли. Не назначай Research для
общеизвестных стабильных сведений, объяснения стандартного алгоритма или задачи по
коду, если пользователь явно не просит веб-поиск, актуальные сведения или источники.
Если запрос просит объяснить алгоритм, написать его реализацию и тесты, объедини это
в одну подзадачу Coding Agent. Проверку граничных случаев включи в требования к
этой подзадаче; не создавай отдельные повторяющие её задачи.

Если для корректного плана не хватает существенных требований, не угадывай их.
Верни needs_input=true, пустой subtasks и один конкретный clarification_question.
Иначе верни needs_input=false, clarification_question=null и план.
Верни только JSON без Markdown:
{"needs_input":false,"clarification_question":null,"subtasks":[{"id":"task_1","description":"...","task_type":"CODING",
"assigned_agent":"coding","priority":"HIGH","depends_on":[]}]}
Не добавляй поля и не пиши код для выполнения.
"""

FINAL_ANSWER_SYSTEM_PROMPT = """\
Ты собираешь финальный ответ из фактически полученных результатов агентов.
Не выдумывай результаты и не скрывай ошибки, пропуски, ограничения или отклонение
Critic. Сохраняй существенные выводы исследования, математический результат,
код и замечания Critic. Различай утверждённые и отклонённые результаты. Если
выполнение неполное, прямо скажи об этом и перечисли невыполненные этапы.
Не утверждай, что код запускался, если этого нет в переданных результатах.
Пиши на языке исходного запроса.
"""

ALLOWED_AGENTS = frozenset({"research", "coding", "math", "critic"})
SUBTASK_STATUSES = frozenset({"NEW", "READY", "RUNNING", "COMPLETED", "FAILED", "SKIPPED", "NEEDS_INPUT"})
MAX_SUBTASKS = 12
MAX_FINAL_SYNTHESIS_BYTES = 24_000


class OrchestratorError(Exception):
    """Base exception for orchestration errors."""


class OrchestratorPlanError(OrchestratorError, ValueError):
    """Raised when a plan is malformed, unsafe, or cannot be executed."""


class OrchestratorPlanParseError(OrchestratorPlanError):
    """Raised when the planning LLM does not return valid JSON."""


class OrchestratorLLMError(OrchestratorError):
    """Raised when planning or final answer generation fails."""


@dataclass
class Subtask:
    """One planned unit of work with a specialized executor and dependencies."""

    id: str
    description: str
    agent: str
    depends_on: list[str]
    status: str = "NEW"
    result: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    started_at: str | None = None
    finished_at: str | None = None
    task_type: TaskType | str | None = None
    priority: TaskPriority | str | None = None
    task_id: str = field(default_factory=lambda: str(uuid4()))
    parent_task_id: str | None = None
    dependencies: list[str] = field(default_factory=list)

    @property
    def assigned_agent(self) -> str:
        """Return the configured logical role assigned to this subtask."""
        return self.agent

@dataclass
class OrchestratorPlan:
    """Validated plan for one original user request."""

    original_request: str
    subtasks: list[Subtask]
    execution_order: list[str] = field(default_factory=list)
    plan_id: str = field(default_factory=lambda: str(uuid4()))
    root_task_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    needs_input: bool = False
    clarification_question: str | None = None


@dataclass
class OrchestratorResult:
    """Execution report retaining the plan, per-task results, and final response."""

    original_request: str
    plan: OrchestratorPlan | None
    subtask_results: dict[str, Any]
    final_answer: str
    status: str
    errors: list[str]
    execution_order: list[str] = field(default_factory=list)
    execution_history: list[dict[str, Any]] = field(default_factory=list)
    execution_metrics: dict[str, Any] = field(default_factory=dict)


class Orchestrator(BaseAgent):
    """Plan and run specialized agents sequentially using dependency injection."""

    CONFIG_ID = "orchestrator"

    def __init__(
        self,
        client: LLMClient,
        agents: Mapping[str, Any] | None = None,
        config: AgentConfig | None = None,
        *,
        config_registry: AgentConfigRegistry | None = None,
        agent_id: str = "orchestrator",
        name: str = "Orchestrator",
        system_prompt: str = ORCHESTRATOR_SYSTEM_PROMPT,
    ) -> None:
        """Create an orchestrator with an injected LLM and optional executors."""
        self.agent_config_registry = config_registry or AgentConfigRegistry()
        if config is None:
            config = self.agent_config_registry.get_agent(self.CONFIG_ID)
        elif config.name != self.CONFIG_ID:
            raise ValueError(f"Orchestrator requires the agent config {self.CONFIG_ID!r}.")
        super().__init__(agent_id, name, system_prompt, client, config)
        if agents is None:
            agents = {}
        if not isinstance(agents, Mapping):
            raise TypeError("agents must be a mapping of allowed role names to agent instances.")
        unknown = set(agents) - ALLOWED_AGENTS
        if unknown:
            raise ValueError(f"Unsupported executor name(s): {', '.join(sorted(map(str, unknown)))}.")
        self.agents = dict(agents)
        self._validate_injected_agents()

    def plan(self, request: str) -> OrchestratorPlan:
        """Ask the LLM for a bounded plan and validate it before returning."""
        return self._generate_plan(
            request,
            system_prompt=self.system_prompt,
            require_injected_agents=True,
            require_subtasks=False,
            allow_orchestrator=False,
        )

    def decompose(self, request: str) -> OrchestratorPlan:
        """Decompose a request into a validated plan without executing agents."""
        return self._generate_plan(
            request,
            system_prompt=TASK_DECOMPOSITION_SYSTEM_PROMPT,
            require_injected_agents=False,
            require_subtasks=True,
            allow_orchestrator=True,
        )

    def plan_and_save(self, request: str, task_store: TaskStore) -> OrchestratorPlan:
        """Generate and atomically persist a task plan without executing it."""
        plan = self.decompose(request)
        self.save_plan(plan, task_store)
        return plan

    def run_adaptive(
        self,
        request: str,
        *,
        task_store: TaskStore | None = None,
        router: AdaptiveRouter | None = None,
        message_bus: MessageBus | None = None,
        max_workers: int = 4,
        max_retries: int = 1,
        max_total_retries: int = 8,
        max_critic_cycles: int = 1,
        critic_enabled: bool = True,
        task_timeout: float = 180.0,
        max_execution_time: float | None = 600.0,
        max_latency_ms: float | None = None,
        max_calls: int | None = 30,
        max_tokens: int | None = 100_000,
        max_cost: float | None = None,
        metrics_collector: Any | None = None,
        metrics_output_dir: str | None = str(Path(__file__).resolve().parents[2] / "experiments" / "results"),
    ) -> OrchestratorResult:
        """Run one bounded plan, optionally reviewing and revising its results.

        ``max_execution_time`` spans planning, task execution, feedback, and final
        synthesis. ``max_calls`` and ``max_tokens`` are shared across all LLM
        clients participating in the run. ``max_total_retries`` applies across
        fallbacks, agent retries, format repairs, and Critic feedback cycles.
        """
        if not isinstance(request, str) or not request.strip():
            raise AgentInputError("request must be a non-empty string.")
        if not isinstance(critic_enabled, bool):
            raise TypeError("critic_enabled must be a bool.")
        if critic_enabled and "critic" not in self.agents:
            raise OrchestratorPlanError("run_adaptive requires an injected Critic Agent for final review.")
        request = request.strip()
        from src.evaluation.metrics import MetricsCollector, activate_metrics, reset_metrics

        collector = metrics_collector or MetricsCollector()
        if not isinstance(collector, MetricsCollector):
            raise TypeError("metrics_collector must be a MetricsCollector or None.")
        metric_token = activate_metrics(collector, agent_role="orchestrator")

        def finalize_metrics(result: OrchestratorResult) -> OrchestratorResult:
            history = result.execution_history or []
            critic_issue_count = 0
            critical_issue_count = 0
            critic_cycle_count = 0
            for subtask in (result.plan.subtasks if result.plan else []):
                if subtask.agent != "critic":
                    continue
                review = getattr(subtask.result, "report", subtask.result)
                issues = getattr(review, "issues", None)
                if issues is None and isinstance(review, Mapping):
                    issues = review.get("issues", [])
                if not isinstance(issues, (list, tuple)):
                    issues = []
                critic_issue_count += len(issues)
                for issue in issues:
                    severity = getattr(issue, "severity", None)
                    if severity is None and isinstance(issue, Mapping):
                        severity = issue.get("severity")
                    if str(getattr(severity, "value", severity)).upper() == "CRITICAL":
                        critical_issue_count += 1
                critic_cycle_count += int(getattr(subtask.result, "cycle_count", 0) or 0)
            collector.set_quality(
                execution_status=result.status,
                critic_status=result.execution_metrics.get("review_status"),
                critic_issues=critic_issue_count,
                critical_critic_issues=critical_issue_count,
                successful_tasks=sum(1 for subtask in (result.plan.subtasks if result.plan else [])
                                     if subtask.status == "COMPLETED"),
                failed_tasks=sum(1 for subtask in (result.plan.subtasks if result.plan else [])
                                 if subtask.status == "FAILED"),
                skipped_tasks=sum(1 for subtask in (result.plan.subtasks if result.plan else [])
                                  if subtask.status == "SKIPPED"),
                needs_input_tasks=sum(1 for subtask in (result.plan.subtasks if result.plan else [])
                                      if subtask.status == "NEEDS_INPUT"),
                critic_cycles=max(critic_cycle_count, sum(1 for item in history
                                  if isinstance(item, Mapping) and item.get("event") == "critic_refinement")),
            )
            snapshot = collector.finish(status=result.status)
            result.execution_metrics.update({
                "run_id": collector.run_id,
                "api_calls": snapshot["api_calls"],
                "measured_prompt_tokens": snapshot["prompt_tokens"],
                "measured_completion_tokens": snapshot["completion_tokens"],
                "measured_total_tokens": snapshot["total_tokens"],
                "measured_cost": snapshot["estimated_cost"],
                "unknown_cost_calls": snapshot["unknown_cost_calls"],
                "metrics_snapshot": snapshot,
            })
            if metrics_output_dir is not None:
                try:
                    collector.export(metrics_output_dir)
                except Exception as export_error:
                    result.execution_metrics["metrics_export_error"] = (
                        f"{type(export_error).__name__}: {export_error}"
                    )
            if result.plan is not None:
                try:
                    root_task = store.get_task(result.plan.root_task_id)
                    if isinstance(root_task.result, dict):
                        saved_result = dict(root_task.result)
                        saved_result["metrics"] = dict(result.execution_metrics)
                        store.update_task(root_task.task_id, result=saved_result)
                except Exception:
                    pass
            return result

        store = task_store or TaskStore()
        runtime_agents = {**self.agents, "orchestrator": self}
        if not critic_enabled:
            runtime_agents.pop("critic", None)
        from src.routing import AdaptiveRouter
        active_router = router or AdaptiveRouter(
            agent_config_registry=self.agent_config_registry,
            available_agents=runtime_agents.keys(),
        )
        try:
            executor = TaskExecutor(
                store,
                active_router,
                runtime_agents,
                message_bus=message_bus,
                max_workers=max_workers,
                max_retries=max_retries,
                max_total_retries=max_total_retries,
                max_critic_cycles=max_critic_cycles,
                task_timeout=task_timeout,
                max_execution_time=max_execution_time,
                max_latency_ms=max_latency_ms,
                max_calls=max_calls,
                max_tokens=max_tokens,
                max_cost=max_cost,
                coordinator_id=self.agent_id,
            )
        except Exception:
            reset_metrics(metric_token)
            raise
        overall_started = time.perf_counter()
        plan: OrchestratorPlan | None = None
        plan_saved = False
        try:
            executor._install_request_hooks()
            plan = self.decompose(request)
            if plan.needs_input:
                question = plan.clarification_question or "Уточните требования к задаче."
                metrics = executor.metrics_snapshot(
                    duration_ms=(time.perf_counter() - overall_started) * 1000
                )
                store.create_task(Task(
                    task_id=plan.root_task_id,
                    description=request,
                    task_type=TaskType.GENERAL,
                    status=TaskStatus.NEEDS_INPUT,
                    result={"status": "NEEDS_INPUT", "clarification_question": question},
                ))
                event = {
                    "timestamp": self._now(),
                    "task_id": plan.root_task_id,
                    "event": "needs_input",
                    "clarification_question": question,
                    "failure_category": "insufficient_context",
                    "retry_allowed": False,
                }
                answer = f"Чтобы продолжить, уточните: {question}"
                result = OrchestratorResult(
                    original_request=request,
                    plan=plan,
                    subtask_results={},
                    final_answer=answer,
                    status="NEEDS_INPUT",
                    errors=[],
                    execution_history=[event],
                    execution_metrics={
                        **metrics,
                        "review_status": "NOT_RUN",
                        "critic_enabled": critic_enabled,
                        "max_critic_cycles": max_critic_cycles if critic_enabled else 0,
                    },
                )
                store.update_task(plan.root_task_id, result={
                    "status": "NEEDS_INPUT", "final_status": "NEEDS_INPUT",
                    "clarification_question": question, "history": [event],
                    "metrics": result.execution_metrics, "final_answer": answer,
                })
                return finalize_metrics(result)
            self._drop_unrequested_research(plan, request)
            review_tasks = [subtask for subtask in plan.subtasks if subtask.agent == "critic"]
            if not critic_enabled:
                review_ids = {subtask.id for subtask in review_tasks}
                dependents = [
                    subtask.id for subtask in plan.subtasks
                    if subtask.agent != "critic" and review_ids.intersection(subtask.depends_on)
                ]
                if dependents:
                    raise OrchestratorPlanError(
                        "Critic cannot be disabled because non-review tasks depend on its output: "
                        + ", ".join(dependents)
                    )
                plan.subtasks = [subtask for subtask in plan.subtasks if subtask.agent != "critic"]
                review_tasks = []
            review_dependencies = [subtask.id for subtask in plan.subtasks if subtask.agent != "critic"]
            if critic_enabled and not review_dependencies:
                raise OrchestratorPlanError("Adaptive execution needs at least one non-Critic result to review.")
            if critic_enabled and review_tasks:
                for review in review_tasks:
                    review.depends_on = list(review_dependencies)
            elif critic_enabled:
                plan.subtasks.append(Subtask(
                    id=f"final_critic_review_{uuid4().hex[:8]}",
                    description="Проверь результаты всех подзадач относительно исходного запроса.",
                    agent="critic",
                    depends_on=review_dependencies,
                    task_type=TaskType.REVIEW,
                    priority=TaskPriority.HIGH,
                ))
            self.validate_plan(plan, require_injected_agents=False,
                               require_subtasks=not plan.needs_input, allow_orchestrator=True)
            self.save_plan(plan, store)
            plan_saved = True
            report = executor.execute(plan.root_task_id, finalize_root=False)
            result = self._build_adaptive_result(request, plan, store, report)
            metrics = executor.metrics_snapshot(
                duration_ms=(time.perf_counter() - overall_started) * 1000
            )
            result.execution_metrics = {**result.execution_metrics, **metrics}
            result.execution_metrics["critic_enabled"] = critic_enabled
            result.execution_metrics["max_critic_cycles"] = max_critic_cycles if critic_enabled else 0
            if metrics.get("budget_violation") and result.status == "COMPLETED":
                result.status = "PARTIAL"
                result.final_answer += (
                    "\n\nВыполнение не завершено успешно. Статус: PARTIAL.\n"
                    + str(metrics["budget_violation"])
                )
            root = store.get_task(plan.root_task_id)
            stored_result = dict(root.result) if isinstance(root.result, dict) else {}
            stored_result["metrics"] = dict(result.execution_metrics)
            stored_result["final_status"] = result.status
            stored_result["final_answer"] = result.final_answer
            store.update_task(plan.root_task_id, result=stored_result)
            root = store.get_task(plan.root_task_id)
            if root.status is TaskStatus.RUNNING:
                store.update_status(
                    plan.root_task_id,
                    TaskStatus.COMPLETED if result.status == "COMPLETED" else (
                        TaskStatus.NEEDS_INPUT if result.status == "NEEDS_INPUT" else TaskStatus.FAILED
                    ),
                )
            return finalize_metrics(result)
        except Exception as error:
            collector.set_quality(execution_status="ERROR", error_type=type(error).__name__,
                                  error_message=str(error))
            if metrics_output_dir is not None and (
                not isinstance(error, ExecutionBudgetExceeded) or plan_saved
            ):
                try:
                    collector.finish(status="ERROR")
                    collector.export(metrics_output_dir)
                except Exception as export_error:
                    try:
                        error.add_note(f"Metrics export failed: {type(export_error).__name__}: {export_error}")
                    except Exception:
                        pass
            if isinstance(error, ExecutionBudgetExceeded) and not plan_saved:
                failure_status = "FAILED"
                message = f"{type(error).__name__}: {error}"
                metrics = executor.metrics_snapshot(
                    duration_ms=(time.perf_counter() - overall_started) * 1000
                )
                event = {
                    "timestamp": self._now(),
                    "task_id": plan.root_task_id if plan is not None else None,
                    "event": "resource_limit_exceeded",
                    "failure_category": error.details.get("limit", "budget"),
                    "error": message,
                    "budget_details": dict(error.details),
                    "retry_allowed": False,
                }
                root_id = plan.root_task_id if plan is not None else str(uuid4())
                if plan is None:
                    store.create_task(Task(
                        task_id=root_id,
                        description=request,
                        task_type=TaskType.GENERAL,
                        status=TaskStatus.FAILED,
                        result={"status": failure_status, "history": [event], "error": message},
                        error=message,
                    ))
                final = f"Не удалось начать выполнение: {message}"
                metrics["critic_enabled"] = critic_enabled
                metrics["max_critic_cycles"] = max_critic_cycles if critic_enabled else 0
                metrics["resource_limits"] = {
                    "max_calls": max_calls,
                    "max_tokens": max_tokens,
                    "max_execution_time": max_execution_time,
                    "max_total_retries": max_total_retries,
                }
                return finalize_metrics(OrchestratorResult(
                    original_request=request,
                    plan=plan,
                    subtask_results={},
                    final_answer=final,
                    status=failure_status,
                    errors=[message],
                    execution_history=[event],
                    execution_metrics=metrics,
                ))
            if plan_saved and plan is not None:
                try:
                    root = store.get_task(plan.root_task_id)
                    if root.status is TaskStatus.RUNNING:
                        store.update_task(plan.root_task_id, result={
                            "status": "FAILED",
                            "error": f"{type(error).__name__}: {error}",
                            "history": [],
                        })
                        store.update_status(plan.root_task_id, TaskStatus.FAILED)
                except Exception:
                    pass
            raise
        finally:
            executor._restore_request_hooks()
            reset_metrics(metric_token)

    def _build_adaptive_result(
        self,
        request: str,
        plan: OrchestratorPlan,
        task_store: TaskStore,
        report: TaskExecutionReport,
    ) -> OrchestratorResult:
        saved_tasks = {task.task_id: task for task in task_store.get_subtasks(plan.root_task_id)}
        logical_results: dict[str, Any] = {}
        for subtask in plan.subtasks:
            task = saved_tasks[subtask.task_id]
            subtask.status = task.status.value
            subtask.result = task.result
            subtask.error = task.error
            logical_results[subtask.id] = task.result
        logical_id_by_task = {subtask.task_id: subtask.id for subtask in plan.subtasks}
        plan.execution_order = [
            logical_id_by_task[event.task_id]
            for event in report.history
            if event.event == "completed" and event.task_id in logical_id_by_task
        ]
        failed = []
        for task in saved_tasks.values():
            if task.status not in {
                TaskStatus.FAILED, TaskStatus.SKIPPED, TaskStatus.CANCELLED, TaskStatus.NEEDS_INPUT,
            }:
                continue
            reason = task.error
            if task.status is TaskStatus.NEEDS_INPUT and isinstance(task.result, Mapping):
                reason = task.result.get("clarification_question", "User input is required.")
            failed.append(f"{task.description} ({task.status.value}): {reason}")
        task_payload = [
            {
                "task_id": task.task_id,
                "description": task.description,
                "task_type": task.task_type.value,
                "assigned_agent": task.assignee,
                "status": task.status.value,
                "result": self._plain_data(task.result),
                "error": task.error,
            }
            for task in saved_tasks.values()
        ]
        payload = {
            "original_request": request,
            "execution_status": report.status,
            "review_status": report.review_status,
            "tasks": task_payload,
        }
        payload = self._bounded_synthesis_payload(payload)
        clarification_questions = [
            event.clarification_question for event in report.history
            if event.clarification_question
        ]
        if report.status == "NEEDS_INPUT":
            question = clarification_questions[0] if clarification_questions else (
                "Уточните требования, необходимые для продолжения."
            )
            synthesis_success = True
            final_answer = f"Чтобы продолжить, уточните: {question}"
        elif (report.review_status == "NOT_REQUIRED" and report.status == "COMPLETED"
              and len(saved_tasks) == 1):
            # A single specialist result needs no second LLM call to repeat it.
            only_task = next(iter(saved_tasks.values()))
            synthesis_success = True
            final_answer = self._render_single_agent_result(only_task)
        else:
            try:
                response = super().run(
                    "Подготовь итоговый ответ по данным выполнения (JSON):\n"
                    + json.dumps(payload, ensure_ascii=False, default=str),
                    system_prompt=FINAL_ANSWER_SYSTEM_PROMPT,
                )
            except ExecutionBudgetExceeded as error:
                # Specialist results already exist. Do not turn a completed
                # task into a failed run because an optional synthesis call
                # cannot be reserved under the shared hard budget.
                synthesis_success = True
                final_answer = self._render_results_without_llm(
                    request, saved_tasks, budget_note=str(error)
                )
            else:
                synthesis_success = response.success
                if response.success:
                    final_answer = response.content.strip()
                else:
                    final_answer = f"Не удалось собрать итоговый ответ: {response.error}"
                    failed.append(final_answer)
        final_status = report.status if synthesis_success else (
            "PARTIAL" if logical_results else "FAILED"
        )
        if final_status != "COMPLETED":
            final_answer += "\n\nВыполнение не завершено успешно. Статус: " + final_status + "."
            if failed:
                final_answer += "\nПроблемные этапы:\n" + "\n".join(f"- {item}" for item in failed)
        root = task_store.get_task(plan.root_task_id)
        stored_result = dict(root.result) if isinstance(root.result, dict) else {}
        stored_result["final_answer"] = final_answer
        stored_result["final_status"] = final_status
        task_store.update_task(plan.root_task_id, result=stored_result)
        return OrchestratorResult(
            original_request=request,
            plan=plan,
            subtask_results=logical_results,
            final_answer=final_answer,
            status=final_status,
            errors=failed,
            execution_order=list(plan.execution_order),
            execution_history=[asdict(event) for event in report.history],
            execution_metrics={
                "calls": report.calls,
                "prompt_tokens": report.prompt_tokens,
                "completion_tokens": report.completion_tokens,
                "estimated_tokens": report.estimated_tokens,
                "estimated_cost": report.estimated_cost,
                "duration_ms": report.duration_ms,
                "review_status": report.review_status,
                "budget_violation": report.budget_violation,
                "unknown_cost_requests": report.unknown_cost_requests,
            },
        )

    @classmethod
    def _bounded_synthesis_payload(cls, payload: dict[str, Any]) -> dict[str, Any]:
        """Bound final-model context and omit internal event history."""
        bounded: dict[str, Any] = payload
        for limit in (8_000, 4_000, 2_000, 1_000, 500, 150):
            bounded = {
                "original_request": cls._limit_text(payload["original_request"], min(limit, 4_000)),
                "execution_status": payload["execution_status"],
                "review_status": payload["review_status"],
                "tasks": [
                    {
                        "task_id": task["task_id"],
                        "description": cls._limit_text(task["description"], min(limit, 1_500)),
                        "task_type": task["task_type"],
                        "assigned_agent": task["assigned_agent"],
                        "status": task["status"],
                        "result": cls._limit_value(task["result"], limit),
                        "error": cls._limit_text(task["error"], min(limit, 1_000)) if task["error"] else None,
                    }
                    for task in payload["tasks"]
                ],
            }
            serialized = json.dumps(bounded, ensure_ascii=False, separators=(",", ":"), default=str)
            if len(serialized.encode("utf-8")) <= MAX_FINAL_SYNTHESIS_BYTES:
                return bounded
        # In the extreme case of many oversized subtasks, retain only the
        # identifiers, statuses and errors. Raw outputs remain on the result.
        return {
            "original_request": bounded["original_request"],
            "execution_status": payload["execution_status"],
            "review_status": payload["review_status"],
            "tasks": [
                {key: task[key] for key in ("task_id", "assigned_agent", "status", "error")}
                for task in bounded["tasks"]
            ],
        }

    @classmethod
    def _limit_value(cls, value: Any, limit: int) -> Any:
        if isinstance(value, str):
            return cls._limit_text(value, limit)
        if isinstance(value, Mapping):
            return {str(key): cls._limit_value(item, limit)
                    for key, item in list(value.items())[:24]}
        if isinstance(value, (list, tuple)):
            return [cls._limit_value(item, min(limit, 1_000)) for item in value[:8]]
        return value

    @staticmethod
    def _limit_text(value: str, limit: int) -> str:
        if len(value) <= limit:
            return value
        return value[:limit] + "…[обрезано для экономии контекста]"

    @classmethod
    def _render_results_without_llm(
        cls, request: str, tasks: Mapping[str, Task], *, budget_note: str
    ) -> str:
        """Return existing agent outputs directly if final synthesis has no budget."""
        parts = [f"Запрос: {request}", "Результаты выполненных подзадач:"]
        for task in tasks.values():
            parts.extend((f"\n### {task.assignee or task.task_type.value} — {task.status.value}",
                          f"Задача: {task.description}"))
            if task.error:
                parts.append(f"Ошибка: {task.error}")
            result = cls._plain_data(task.result)
            if result is not None:
                parts.append(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        parts.append(
            "\nДополнительный LLM-синтез пропущен: общий лимит нельзя безопасно "
            f"зарезервировать. Результаты агентов сохранены. Детали: {budget_note}"
        )
        return "\n".join(parts)

    @classmethod
    def _render_single_agent_result(cls, task: Task) -> str:
        """Present one completed specialist result without a duplicate model call."""
        result = cls._plain_data(task.result)
        if not isinstance(result, Mapping):
            return str(result) if result is not None else "Агент не вернул содержимое."
        sections = []
        if result.get("summary"):
            sections.append(str(result["summary"]))
        language = str(result.get("language", ""))
        for key, title in (("code", "Код"), ("tests", "Тесты")):
            value = result.get(key)
            if isinstance(value, str) and value:
                sections.append(f"### {title}\n```{language}\n{value}\n```")
        for key, title in (("dependencies", "Зависимости"),
                           ("run_instructions", "Инструкция по запуску"),
                           ("clarification_question", "Уточнение")):
            value = result.get(key)
            if value:
                rendered = ", ".join(map(str, value)) if isinstance(value, list) else str(value)
                sections.append(f"### {title}\n{rendered}")
        if not sections:
            return json.dumps(result, ensure_ascii=False, indent=2, default=str)
        return "\n\n".join(sections)

    def save_plan(self, plan: OrchestratorPlan, task_store: TaskStore) -> list[Task]:
        """Validate and atomically store the plan root and its assigned subtasks."""
        if not isinstance(task_store, TaskStore):
            raise TypeError("task_store must be a TaskStore instance.")
        self.validate_plan(
            plan,
            require_injected_agents=False,
            require_subtasks=True,
            allow_orchestrator=True,
        )

        root_priority = (
            TaskPriority.HIGH
            if any(subtask.priority is TaskPriority.HIGH for subtask in plan.subtasks)
            else TaskPriority.MEDIUM
        )
        records = [Task(
            task_id=plan.root_task_id,
            description=plan.original_request,
            task_type=TaskType.GENERAL,
            priority=root_priority,
            assignee=self.agent_id,
        )]
        for subtask in plan.subtasks:
            records.append(Task(
                task_id=subtask.task_id,
                description=subtask.description.strip(),
                task_type=subtask.task_type,
                priority=subtask.priority,
                status=TaskStatus.ASSIGNED,
                parent_task_id=plan.root_task_id,
                assignee=subtask.assigned_agent,
                dependencies=list(subtask.dependencies),
            ))
        try:
            return task_store.create_tasks(records)
        except TaskStoreError as error:
            raise OrchestratorPlanError(
                f"Could not atomically save plan {plan.plan_id!r}: {error}"
            ) from error

    def _generate_plan(
        self,
        request: str,
        *,
        system_prompt: str,
        require_injected_agents: bool,
        require_subtasks: bool,
        allow_orchestrator: bool,
    ) -> OrchestratorPlan:
        if not isinstance(request, str) or not request.strip():
            raise AgentInputError("request must be a non-empty string.")
        response = super().run(
            "Составь минимальный план специализированного выполнения запроса:\n" + request.strip(),
            system_prompt=system_prompt,
        )
        if not response.success:
            raise OrchestratorLLMError(f"Could not plan the request: {response.error}")
        plan = self._parse_plan(response.content, request.strip())
        self.validate_plan(
            plan,
            require_injected_agents=require_injected_agents,
            require_subtasks=require_subtasks and not plan.needs_input,
            allow_orchestrator=allow_orchestrator,
        )
        return plan

    def validate_plan(
        self,
        plan: OrchestratorPlan | Mapping[str, Any],
        *,
        require_injected_agents: bool = True,
        require_subtasks: bool = False,
        allow_orchestrator: bool = False,
    ) -> OrchestratorPlan:
        """Validate schema, agent availability, dependencies, and acyclic ordering."""
        if isinstance(plan, Mapping):
            original_request = plan.get("original_request", plan.get("request"))
            raw_subtasks = plan.get("subtasks")
            if not isinstance(original_request, str) or not isinstance(raw_subtasks, list):
                raise OrchestratorPlanError("Plan must have a request and a subtasks list.")
            try:
                parsed = [self._subtask_from_mapping(item) for item in raw_subtasks]
                plan = OrchestratorPlan(
                    original_request=original_request,
                    subtasks=parsed,
                    plan_id=plan.get("plan_id", str(uuid4())),
                    root_task_id=plan.get("root_task_id", str(uuid4())),
                    created_at=plan.get("created_at", datetime.now(timezone.utc)),
                    needs_input=plan.get("needs_input", False),
                    clarification_question=plan.get("clarification_question"),
                )
            except (KeyError, TypeError, ValueError) as error:
                if isinstance(error, OrchestratorPlanError):
                    raise
                raise OrchestratorPlanError(f"Plan contains malformed fields: {error}") from error
        if not isinstance(plan, OrchestratorPlan):
            raise OrchestratorPlanError("plan must be an OrchestratorPlan or a plan mapping.")
        if not isinstance(plan.original_request, str) or not plan.original_request.strip():
            raise OrchestratorPlanError("Plan original_request must be a non-empty string.")
        for field_name, value in (("plan_id", plan.plan_id), ("root_task_id", plan.root_task_id)):
            if not isinstance(value, str) or not value.strip():
                raise OrchestratorPlanError(f"Plan {field_name} must be a non-empty string.")
        if not isinstance(plan.created_at, datetime) or plan.created_at.tzinfo is None or plan.created_at.utcoffset() is None:
            raise OrchestratorPlanError("Plan created_at must be a timezone-aware datetime.")
        if not isinstance(plan.subtasks, list):
            raise OrchestratorPlanError("Plan must contain a subtasks list.")
        if not isinstance(plan.needs_input, bool):
            raise OrchestratorPlanError("Plan needs_input must be a boolean.")
        if plan.needs_input:
            if plan.subtasks:
                raise OrchestratorPlanError("A plan awaiting user input cannot contain executable subtasks.")
            if not isinstance(plan.clarification_question, str) or not plan.clarification_question.strip():
                raise OrchestratorPlanError("A plan awaiting user input needs a concrete clarification_question.")
        elif plan.clarification_question is not None:
            raise OrchestratorPlanError("clarification_question must be null unless needs_input is true.")
        if require_subtasks and not plan.subtasks and not plan.needs_input:
            raise OrchestratorPlanError("A decomposition plan must contain at least one subtask.")
        if len(plan.subtasks) > MAX_SUBTASKS:
            raise OrchestratorPlanError(f"Plan cannot contain more than {MAX_SUBTASKS} subtasks.")

        identifiers: set[str] = set()
        task_identifiers = {plan.root_task_id}
        for subtask in plan.subtasks:
            if not isinstance(subtask, Subtask):
                raise OrchestratorPlanError("Every plan item must be a Subtask.")
            if not isinstance(subtask.id, str) or not subtask.id.strip():
                raise OrchestratorPlanError("Subtask ID must be a non-empty string.")
            if subtask.id in identifiers:
                raise OrchestratorPlanError(f"Duplicate subtask ID {subtask.id!r}.")
            identifiers.add(subtask.id)
            if not isinstance(subtask.description, str) or not subtask.description.strip():
                raise OrchestratorPlanError(f"Subtask {subtask.id!r} needs a non-empty description.")
            allowed_assignees = ALLOWED_AGENTS | ({"orchestrator"} if allow_orchestrator else set())
            if not isinstance(subtask.agent, str) or subtask.agent not in allowed_assignees:
                raise OrchestratorPlanError(f"Subtask {subtask.id!r} uses unsupported agent {subtask.agent!r}.")
            if not self.agent_config_registry.has_agent(subtask.agent):
                raise OrchestratorPlanError(f"Subtask {subtask.id!r} uses an unconfigured agent {subtask.agent!r}.")
            if require_injected_agents and subtask.agent not in self.agents:
                raise OrchestratorPlanError(f"Subtask {subtask.id!r} requires uninjected agent {subtask.agent!r}.")
            if not isinstance(subtask.task_id, str) or not subtask.task_id.strip():
                raise OrchestratorPlanError(f"Subtask {subtask.id!r} has an invalid task_id.")
            if subtask.task_id == plan.root_task_id:
                raise OrchestratorPlanError("A subtask task_id cannot equal the root_task_id.")
            if subtask.task_id in task_identifiers:
                raise OrchestratorPlanError(f"Duplicate task_id {subtask.task_id!r} in plan.")
            task_identifiers.add(subtask.task_id)
            if subtask.parent_task_id not in (None, plan.root_task_id):
                raise OrchestratorPlanError(
                    f"Subtask {subtask.id!r} parent_task_id must match the plan root_task_id."
                )
            subtask.parent_task_id = plan.root_task_id
            subtask.task_type = self._normalize_task_type(subtask.task_type, subtask.agent)
            subtask.priority = self._normalize_priority(subtask.priority)
            expected_type = self._task_type_for_agent(subtask.agent)
            if subtask.task_type is not expected_type:
                raise OrchestratorPlanError(
                    f"Subtask {subtask.id!r} task_type {subtask.task_type.value} does not match "
                    f"assigned agent {subtask.agent!r}."
                )
            if not isinstance(subtask.depends_on, list) or any(not isinstance(dep, str) for dep in subtask.depends_on):
                raise OrchestratorPlanError(f"Subtask {subtask.id!r} depends_on must be a list of IDs.")
            if not isinstance(subtask.status, str) or subtask.status not in SUBTASK_STATUSES:
                raise OrchestratorPlanError(f"Subtask {subtask.id!r} has unsupported status {subtask.status!r}.")
            if len(set(subtask.depends_on)) != len(subtask.depends_on):
                raise OrchestratorPlanError(f"Subtask {subtask.id!r} has duplicate dependencies.")
            if subtask.id in subtask.depends_on:
                raise OrchestratorPlanError(f"Subtask {subtask.id!r} cannot depend on itself.")

        by_id = {subtask.id: subtask for subtask in plan.subtasks}
        for subtask in plan.subtasks:
            missing = set(subtask.depends_on) - by_id.keys()
            if missing:
                raise OrchestratorPlanError(
                    f"Subtask {subtask.id!r} references unknown dependency/dependencies: {', '.join(sorted(missing))}."
                )

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(identifier: str) -> None:
            if identifier in visiting:
                raise OrchestratorPlanError("Plan contains a cyclic dependency.")
            if identifier in visited:
                return
            visiting.add(identifier)
            for dependency in by_id[identifier].depends_on:
                visit(dependency)
            visiting.remove(identifier)
            visited.add(identifier)

        for identifier in by_id:
            visit(identifier)

        for subtask in plan.subtasks:
            subtask.dependencies = [by_id[dependency].task_id for dependency in subtask.depends_on]

        # Fill omitted fields from the validated DAG while honoring explicit
        # LLM classifications and priorities.
        dependent_counts = {identifier: 0 for identifier in by_id}
        for subtask in plan.subtasks:
            for dependency in subtask.depends_on:
                dependent_counts[dependency] += 1
        for subtask in plan.subtasks:
            if subtask.priority is None:
                subtask.priority = self._infer_priority(
                    subtask, dependent_counts[subtask.id], plan.original_request
                )
        return plan

    def execute_plan(self, plan: OrchestratorPlan) -> OrchestratorPlan:
        """Execute ready subtasks one at a time and preserve all outcomes."""
        self.validate_plan(plan)
        completed: dict[str, Subtask] = {}
        pending = {subtask.id: subtask for subtask in plan.subtasks}
        execution_order: list[str] = []
        while pending:
            progress = False
            for identifier, subtask in list(pending.items()):
                dependency_states = [completed.get(dep) for dep in subtask.depends_on]
                failed_dependencies = [dep for dep in dependency_states if dep and dep.status in {"FAILED", "SKIPPED", "NEEDS_INPUT"}]
                if failed_dependencies:
                    subtask.status = "SKIPPED"
                    subtask.error = "Skipped because dependency/dependencies failed: " + ", ".join(
                        dep.id for dep in failed_dependencies
                    )
                    subtask.finished_at = self._now()
                    completed[identifier] = subtask
                    execution_order.append(identifier)
                    del pending[identifier]
                    progress = True
                    continue
                if any(dep is None or dep.status != "COMPLETED" for dep in dependency_states):
                    continue
                subtask.status = "READY"
                subtask.started_at = self._now()
                subtask.status = "RUNNING"
                try:
                    result = self._invoke_agent(subtask, plan.original_request, completed)
                    subtask.result = result
                    if subtask.agent == "critic" and getattr(result, "verdict", None) == "REJECTED":
                        subtask.status = "FAILED"
                        subtask.error = "Critic rejected the reviewed result."
                    elif getattr(result, "needs_input", False):
                        subtask.status = "NEEDS_INPUT"
                        subtask.error = getattr(result, "clarification_question", None)
                    elif getattr(result, "task_completed", True) is False:
                        subtask.status = "FAILED"
                        subtask.error = "Agent returned a result marked as incomplete."
                    else:
                        subtask.status = "COMPLETED"
                except Exception as error:
                    # The failure is preserved with its exception type and
                    # attributed to this subtask; dependent work will be skipped.
                    subtask.status = "FAILED"
                    subtask.error = f"{type(error).__name__}: {error}"
                finally:
                    subtask.finished_at = self._now()
                completed[identifier] = subtask
                execution_order.append(identifier)
                del pending[identifier]
                progress = True
            if not progress:
                # A validated DAG should always make progress; retain this guard
                # in case a caller mutates its plan while it is running.
                for identifier, subtask in pending.items():
                    subtask.status = "SKIPPED"
                    subtask.error = "No executable dependency order remained."
                    subtask.finished_at = self._now()
                    completed[identifier] = subtask
                    execution_order.append(identifier)
                pending.clear()
        plan.subtasks = [completed[subtask.id] for subtask in plan.subtasks]
        plan.execution_order = execution_order
        return plan

    def build_final_answer(self, request: str, plan: OrchestratorPlan) -> str:
        """Synthesize a user-facing response from actual subtask results."""
        if plan.needs_input:
            return f"Чтобы продолжить, уточните: {plan.clarification_question}"
        questions = [
            subtask.error for subtask in plan.subtasks
            if subtask.status == "NEEDS_INPUT" and subtask.error
        ]
        if questions:
            return "Чтобы продолжить, уточните: " + " ".join(questions)
        payload = {
            "request": request,
            "subtasks": [self._subtask_payload(subtask) for subtask in plan.subtasks],
            "completion_status": self._overall_status(plan),
        }
        response = super().run(
            "Собери итоговый ответ из фактически полученных результатов:\n"
            + json.dumps(payload, ensure_ascii=False, default=str),
            system_prompt=FINAL_ANSWER_SYSTEM_PROMPT,
        )
        if not response.success:
            raise OrchestratorLLMError(f"Could not build a final answer: {response.error}")
        answer = response.content.strip()
        status = self._overall_status(plan)
        if status != "COMPLETED":
            failures = [f"- {task.id} ({task.agent}): {task.error}" for task in plan.subtasks if task.status in {"FAILED", "SKIPPED"}]
            answer += "\n\nСтатус выполнения: " + status + "."
            if failures:
                answer += "\nПроблемные этапы:\n" + "\n".join(failures)
        return answer

    def run(self, request: str) -> OrchestratorResult:
        """Plan, execute, and summarize one request without parallel execution."""
        if not isinstance(request, str) or not request.strip():
            raise AgentInputError("request must be a non-empty string.")
        request = request.strip()
        errors: list[str] = []
        try:
            plan = self.plan(request)
        except (OrchestratorError, OrchestratorPlanError) as error:
            message = f"Planning failed: {type(error).__name__}: {error}"
            return OrchestratorResult(request, None, {}, message, "FAILED", [message], [])

        if plan.needs_input:
            question = plan.clarification_question or "Уточните требования к задаче."
            return OrchestratorResult(
                request, plan, {}, f"Чтобы продолжить, уточните: {question}",
                "NEEDS_INPUT", [], execution_history=[{
                    "event": "needs_input", "failure_category": "insufficient_context",
                    "clarification_question": question,
                }],
            )

        self.execute_plan(plan)
        errors.extend(
            f"Subtask {task.id} ({task.agent}) {task.status}: {task.error}"
            for task in plan.subtasks if task.error
        )
        try:
            final_answer = self.build_final_answer(request, plan)
        except OrchestratorLLMError as error:
            message = f"Final synthesis failed: {type(error).__name__}: {error}"
            errors.append(message)
            final_answer = message
            final_status = "PARTIAL" if any(task.status == "COMPLETED" for task in plan.subtasks) else "FAILED"
        else:
            final_status = self._overall_status(plan)
        return OrchestratorResult(
            original_request=request,
            plan=plan,
            subtask_results={subtask.id: subtask.result for subtask in plan.subtasks},
            final_answer=final_answer,
            status=final_status,
            errors=errors,
            execution_order=list(plan.execution_order),
        )

    def _invoke_agent(
        self,
        subtask: Subtask,
        original_request: str,
        completed: Mapping[str, Subtask],
    ) -> Any:
        """Adapt distinct public agent APIs to the shared sequential runner."""
        agent = self.agents[subtask.agent]
        dependencies = [completed[identifier] for identifier in subtask.depends_on]
        context = self._dependency_context(dependencies)
        if subtask.agent == "research":
            return agent.run_research(self._task_with_context(subtask.description, context))
        if subtask.agent == "math":
            return agent.solve(self._task_with_context(subtask.description, context))
        if subtask.agent == "coding":
            result = agent.run(subtask.description, additional_context=context or None)
            if isinstance(result, AgentResult):
                if not result.success:
                    raise OrchestratorError(result.error or "Coding Agent returned an unsuccessful result.")
                return agent.parse_solution(result)
            return result
        if subtask.agent == "critic":
            target = self._critic_target(dependencies)
            if target is None:
                raise OrchestratorPlanError("Critic subtask must depend on at least one result to review.")
            return agent.review(
                task=original_request,
                result=target.result,
                result_type=target.agent,
                agent_name=target.agent,
                additional_materials={"dependency_results": self._plain_data([self._subtask_payload(d) for d in dependencies])},
            )
        # validate_plan blocks every other name before execution.
        raise OrchestratorPlanError(f"Unsupported executor {subtask.agent!r}.")

    def _validate_injected_agents(self) -> None:
        method_by_agent = {
            "research": "run_research",
            "coding": "run",
            "math": "solve",
            "critic": "review",
        }
        for role, agent in self.agents.items():
            if not callable(getattr(agent, method_by_agent[role], None)):
                raise TypeError(f"Injected {role!r} agent must provide callable {method_by_agent[role]}().")
            if role == "coding" and not callable(getattr(agent, "parse_solution", None)):
                raise TypeError("Injected 'coding' agent must provide callable parse_solution().")

    @staticmethod
    def _subtask_from_mapping(item: Any) -> Subtask:
        if not isinstance(item, Mapping):
            raise OrchestratorPlanError("Every planned subtask must be an object.")
        dependency_keys = {"depends_on", "dependencies"} & set(item)
        if not {"id", "description"}.issubset(item) or len(dependency_keys) != 1:
            raise OrchestratorPlanError(
                "Every planned subtask needs id, description, and exactly one dependency list."
            )
        if ("agent" in item) == ("assigned_agent" in item):
            raise OrchestratorPlanError("Use exactly one of 'agent' or 'assigned_agent'.")
        allowed = {
            "id", "description", "depends_on", "dependencies", "agent", "assigned_agent",
            "task_type", "priority", "task_id", "parent_task_id",
        }
        extra = set(item) - allowed
        if extra:
            raise OrchestratorPlanError(
                "Planned subtask contains unknown field(s): " + ", ".join(sorted(extra)) + "."
            )
        return Subtask(
            id=item["id"],
            description=item["description"],
            agent=item.get("assigned_agent", item.get("agent")),
            depends_on=item[next(iter(dependency_keys))],
            task_type=item.get("task_type"),
            priority=item.get("priority"),
            task_id=item.get("task_id", str(uuid4())),
            parent_task_id=item.get("parent_task_id"),
        )

    @staticmethod
    def _task_type_for_agent(agent: str) -> TaskType:
        return {
            "research": TaskType.RESEARCH,
            "coding": TaskType.CODING,
            "math": TaskType.MATH,
            "critic": TaskType.REVIEW,
            "orchestrator": TaskType.GENERAL,
        }[agent]

    @classmethod
    def _normalize_task_type(cls, value: TaskType | str | None, agent: str) -> TaskType:
        if value is None:
            return cls._task_type_for_agent(agent)
        if isinstance(value, TaskType):
            return value
        if isinstance(value, str):
            try:
                return TaskType(value.strip().upper())
            except ValueError:
                pass
        raise OrchestratorPlanError(f"Unsupported task_type {value!r}.")

    @staticmethod
    def _normalize_priority(value: TaskPriority | str | None) -> TaskPriority | None:
        if value is None:
            return None
        if isinstance(value, TaskPriority):
            return value
        if isinstance(value, str):
            try:
                return TaskPriority(value.strip().upper())
            except ValueError:
                pass
        raise OrchestratorPlanError(f"Unsupported task priority {value!r}.")

    @staticmethod
    def _drop_unrequested_research(plan: OrchestratorPlan, request: str) -> None:
        """Skip independent web research for stable requests that do not ask for it."""
        if not any(task.agent != "research" for task in plan.subtasks):
            return
        request_text = request.casefold()
        no_research_markers = (
            "no web search", "don't search", "do not search", "без веб-поиска",
            "не используй веб-поиск", "не ищи в интернете", "поиск не нужен",
            "источники не нужны", "без источников",
        )
        if any(marker in request_text for marker in no_research_markers):
            return
        explicit_research_markers = (
            "research", "web search", "search the web", "find sources", "citations",
            "sources", "current", "latest", "today", "recent", "news",
            "исследуй", "исследован", "веб-поиск", "поиск в интернете",
            "найди источ", "источник", "источники", "цитат", "актуальн",
            "сегодня", "последние новости", "свежие данные",
        )
        if any(marker in request_text for marker in explicit_research_markers):
            return
        research_ids = {task.id for task in plan.subtasks if task.agent == "research"}
        removable = {
            task_id for task_id in research_ids
            if not any(task_id in other.depends_on for other in plan.subtasks if other.id not in research_ids)
        }
        if removable:
            plan.subtasks = [task for task in plan.subtasks if task.id not in removable]

    @staticmethod
    def _infer_priority(subtask: Subtask, dependent_count: int, request: str) -> TaskPriority:
        request_text = request.casefold()
        description = subtask.description.casefold()
        if any(word in request_text for word in ("срочно", "urgent", "asap", "немедленно")):
            return TaskPriority.HIGH
        if any(word in description for word in ("необязательно", "опционально", "optional", "если останется время")):
            return TaskPriority.LOW
        if dependent_count >= 2 or any(
            word in description
            for word in ("реализ", "implement", "реши", "solve", "подготовь итог", "final")
        ):
            return TaskPriority.HIGH
        return TaskPriority.MEDIUM

    @staticmethod
    def _task_with_context(description: str, context: str) -> str:
        return description + ("\n\nКонтекст и результаты предыдущих этапов:\n" + context if context else "")

    @classmethod
    def _parse_plan(cls, content: str, request: str) -> OrchestratorPlan:
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as error:
            raise OrchestratorPlanParseError(
                f"Planning response is invalid JSON at line {error.lineno}, column {error.colno}."
            ) from error
        allowed = {"subtasks", "needs_input", "clarification_question"}
        if not isinstance(payload, dict) or not {"subtasks"}.issubset(payload) or payload.keys() - allowed:
            raise OrchestratorPlanParseError("Planning response must contain 'subtasks' and only supported status fields.")
        if not isinstance(payload["subtasks"], list):
            raise OrchestratorPlanParseError("Planning response 'subtasks' must be a list.")
        needs_input = payload.get("needs_input", False)
        question = payload.get("clarification_question")
        if not isinstance(needs_input, bool):
            raise OrchestratorPlanParseError("Planning response needs_input must be a boolean.")
        if needs_input and (not isinstance(question, str) or not question.strip()):
            raise OrchestratorPlanParseError("A needs_input plan must include a concrete clarification_question.")
        if not needs_input and question is not None:
            raise OrchestratorPlanParseError("clarification_question must be null unless needs_input is true.")
        if needs_input and payload["subtasks"]:
            raise OrchestratorPlanParseError("A needs_input plan must not contain executable subtasks.")
        subtasks: list[Subtask] = []
        for item in payload["subtasks"]:
            try:
                subtasks.append(Orchestrator._subtask_from_mapping(item))
            except OrchestratorPlanError as error:
                raise OrchestratorPlanParseError(str(error)) from error
        return OrchestratorPlan(
            request, subtasks, needs_input=needs_input, clarification_question=question
        )

    @classmethod
    def _dependency_context(cls, dependencies: Sequence[Subtask]) -> str:
        if not dependencies:
            return ""
        payload = [cls._subtask_payload(item) for item in dependencies]
        return json.dumps(payload, ensure_ascii=False, default=str)

    @staticmethod
    def _critic_target(dependencies: Sequence[Subtask]) -> Subtask | None:
        for dependency in reversed(dependencies):
            if dependency.agent != "critic" and dependency.result is not None:
                return dependency
        return None

    @classmethod
    def _subtask_payload(cls, subtask: Subtask) -> dict[str, Any]:
        return {
            "id": subtask.id,
            "description": subtask.description,
            "agent": subtask.agent,
            "status": subtask.status,
            "result": cls._plain_data(subtask.result),
            "error": subtask.error,
            "metadata": subtask.metadata,
        }

    @classmethod
    def _plain_data(cls, value: Any) -> Any:
        if is_dataclass(value) and not isinstance(value, type):
            return cls._plain_data(asdict(value))
        if isinstance(value, Mapping):
            return {str(key): cls._plain_data(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._plain_data(item) for item in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if hasattr(value, "__dict__"):
            return cls._plain_data(vars(value))
        return str(value)

    @staticmethod
    def _overall_status(plan: OrchestratorPlan) -> str:
        if plan.needs_input or any(subtask.status == "NEEDS_INPUT" for subtask in plan.subtasks):
            return "NEEDS_INPUT"
        statuses = [subtask.status for subtask in plan.subtasks]
        if not statuses:
            return "COMPLETED"
        if any(
            subtask.agent == "critic" and subtask.status == "FAILED"
            and subtask.result is not None and getattr(subtask.result, "verdict", None) == "REJECTED"
            for subtask in plan.subtasks
        ):
            return "REJECTED"
        if statuses and all(status == "COMPLETED" for status in statuses):
            return "COMPLETED"
        if any(status == "COMPLETED" for status in statuses):
            return "PARTIAL"
        return "FAILED"

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


__all__ = [
    "ALLOWED_AGENTS",
    "FINAL_ANSWER_SYSTEM_PROMPT",
    "MAX_SUBTASKS",
    "ORCHESTRATOR_SYSTEM_PROMPT",
    "Orchestrator",
    "OrchestratorError",
    "OrchestratorLLMError",
    "OrchestratorPlan",
    "OrchestratorPlanError",
    "OrchestratorPlanParseError",
    "OrchestratorResult",
    "SUBTASK_STATUSES",
    "TASK_DECOMPOSITION_SYSTEM_PROMPT",
    "Subtask",
]
