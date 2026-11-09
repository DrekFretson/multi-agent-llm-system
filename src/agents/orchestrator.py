"""Sequential task planning and coordination across specialized agents."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from src.agents.base_agent import AgentInputError, AgentResult, BaseAgent
from src.agents.config import AgentConfig, AgentConfigRegistry
from src.llm import LLMClient


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
верни пустой список subtasks: координатор ответит напрямую на этапе сборки.

Верни только JSON следующей формы:
{"subtasks":[{"id":"task_1","description":"...","agent":"coding","depends_on":[]}]}
Для краткого запроса создавай одну подходящую подзадачу. Не используй агента
orchestrator и не добавляй поля.
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
SUBTASK_STATUSES = frozenset({"NEW", "READY", "RUNNING", "COMPLETED", "FAILED", "SKIPPED"})
MAX_SUBTASKS = 12


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


@dataclass
class OrchestratorPlan:
    """Validated plan for one original user request."""

    original_request: str
    subtasks: list[Subtask]
    execution_order: list[str] = field(default_factory=list)


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
        if config is None:
            config = (config_registry or AgentConfigRegistry()).get_agent(self.CONFIG_ID)
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
        if not isinstance(request, str) or not request.strip():
            raise AgentInputError("request must be a non-empty string.")
        response = super().run(
            "Составь минимальный план специализированного выполнения запроса:\n" + request.strip(),
            system_prompt=self.system_prompt,
        )
        if not response.success:
            raise OrchestratorLLMError(f"Could not plan the request: {response.error}")
        plan = self._parse_plan(response.content, request.strip())
        self.validate_plan(plan)
        return plan

    def validate_plan(self, plan: OrchestratorPlan | Mapping[str, Any]) -> OrchestratorPlan:
        """Validate schema, agent availability, dependencies, and acyclic ordering."""
        if isinstance(plan, Mapping):
            original_request = plan.get("original_request", plan.get("request"))
            raw_subtasks = plan.get("subtasks")
            if not isinstance(original_request, str) or not isinstance(raw_subtasks, list):
                raise OrchestratorPlanError("Plan must have a request and a subtasks list.")
            parsed: list[Subtask] = []
            for item in raw_subtasks:
                if not isinstance(item, Mapping) or set(item) != {"id", "description", "agent", "depends_on"}:
                    raise OrchestratorPlanError("Each planned subtask must contain exactly id, description, agent, and depends_on.")
                parsed.append(Subtask(
                    id=item["id"], description=item["description"], agent=item["agent"], depends_on=item["depends_on"]
                ))
            plan = OrchestratorPlan(original_request, parsed)
        if not isinstance(plan, OrchestratorPlan):
            raise OrchestratorPlanError("plan must be an OrchestratorPlan or a plan mapping.")
        if not isinstance(plan.original_request, str) or not plan.original_request.strip():
            raise OrchestratorPlanError("Plan original_request must be a non-empty string.")
        if not isinstance(plan.subtasks, list):
            raise OrchestratorPlanError("Plan must contain a subtasks list.")
        if len(plan.subtasks) > MAX_SUBTASKS:
            raise OrchestratorPlanError(f"Plan cannot contain more than {MAX_SUBTASKS} subtasks.")

        identifiers: set[str] = set()
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
            if not isinstance(subtask.agent, str) or subtask.agent not in ALLOWED_AGENTS:
                raise OrchestratorPlanError(f"Subtask {subtask.id!r} uses unsupported agent {subtask.agent!r}.")
            if subtask.agent not in self.agents:
                raise OrchestratorPlanError(f"Subtask {subtask.id!r} requires uninjected agent {subtask.agent!r}.")
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
                failed_dependencies = [dep for dep in dependency_states if dep and dep.status in {"FAILED", "SKIPPED"}]
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
        if not isinstance(payload, dict) or set(payload) != {"subtasks"}:
            raise OrchestratorPlanParseError("Planning response must contain exactly a 'subtasks' list.")
        if not isinstance(payload["subtasks"], list):
            raise OrchestratorPlanParseError("Planning response 'subtasks' must be a list.")
        subtasks: list[Subtask] = []
        for item in payload["subtasks"]:
            if not isinstance(item, dict) or set(item) != {"id", "description", "agent", "depends_on"}:
                raise OrchestratorPlanParseError("Every planned subtask must contain exactly id, description, agent, depends_on.")
            subtasks.append(Subtask(
                id=item["id"], description=item["description"], agent=item["agent"], depends_on=item["depends_on"]
            ))
        return OrchestratorPlan(request, subtasks)

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
    "Subtask",
]
