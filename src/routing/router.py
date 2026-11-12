"""Rule-based routing using configured agent and model metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

from src.agents.config import AgentConfigRegistry
from src.coordination import Task, TaskType
from src.llm.model_registry import ModelConfig, ModelRegistry


class RoutingError(Exception):
    """Base error raised when a task cannot be routed."""


class NoAvailableRouteError(RoutingError):
    """Raised when no configured, available executor can accept a task."""


@dataclass(frozen=True)
class RoutingDecision:
    """Selected executor and the configured model metadata for a task."""

    task_type: TaskType
    requested_agent: str | None
    assigned_agent: str
    logical_model_id: str
    model_id: str
    fallback: bool
    context_estimate: int
    estimated_output_tokens: int
    context_limit: int | None
    estimated_cost: float | None
    estimated_latency_ms: float | None
    reasons: tuple[str, ...]


class AdaptiveRouter:
    """Choose an available configured executor with deterministic rules.

    Availability and latency are supplied by the application from real health
    checks and measurements. Missing cost, latency, or context metadata remains
    unknown; the router never invents model characteristics.
    """

    SPECIALISTS: Mapping[TaskType, str] = {
        TaskType.RESEARCH: "research",
        TaskType.CODING: "coding",
        TaskType.MATH: "math",
        TaskType.REVIEW: "critic",
        TaskType.GENERAL: "orchestrator",
    }
    CAPABILITIES: Mapping[TaskType, str] = {
        TaskType.RESEARCH: "reasoning",
        TaskType.CODING: "code_generation",
        TaskType.MATH: "math",
        TaskType.REVIEW: "reasoning",
        TaskType.GENERAL: "text_generation",
    }

    def __init__(
        self,
        *,
        agent_config_registry: AgentConfigRegistry | None = None,
        model_registry: ModelRegistry | None = None,
        available_agents: Iterable[str] | None = None,
        model_available: Callable[[str], bool] | None = None,
        measured_latency_ms: Mapping[str, float] | None = None,
    ) -> None:
        self.agent_config_registry = agent_config_registry or AgentConfigRegistry()
        self.model_registry = model_registry or self.agent_config_registry.model_registry
        self.available_agents = frozenset(
            available_agents
            if available_agents is not None
            else self.agent_config_registry.agents.keys()
        )
        self.model_available = model_available or (lambda _model_id: True)
        self.measured_latency_ms = dict(measured_latency_ms or {})

    def route(
        self,
        task: Task,
        *,
        additional_context: str = "",
        available_agents: Iterable[str] | None = None,
        exclude_agents: Iterable[str] = (),
        max_cost: float | None = None,
        max_latency_ms: float | None = None,
    ) -> RoutingDecision:
        """Select an executor, respecting capability, context, availability and limits."""
        if not isinstance(task, Task):
            raise TypeError("task must be a Task instance.")
        excluded = set(exclude_agents)
        available = self.available_agents if available_agents is None else self.available_agents & frozenset(available_agents)
        request = task.assignee
        preferred = request if request in self.SPECIALISTS.values() else self.SPECIALISTS[task.task_type]
        candidates = [preferred]
        if preferred != "orchestrator" and task.task_type is not TaskType.REVIEW:
            candidates.append("orchestrator")

        if not isinstance(additional_context, str):
            raise TypeError("additional_context must be a string.")
        context_estimate = self._estimate_tokens(task.description + additional_context)
        reasons: list[str] = []
        rejected: list[str] = []
        for role in candidates:
            if role in excluded:
                rejected.append(f"{role}: excluded after a previous failed attempt")
                continue
            if role not in available or not self.agent_config_registry.has_agent(role):
                rejected.append(f"{role}: executor is not available")
                continue
            config = self.agent_config_registry.get_agent(role)
            model: ModelConfig = self.model_registry.get_model(config.logical_model_id)
            capability = self.CAPABILITIES[task.task_type]
            if capability not in model.capabilities:
                rejected.append(f"{role}: model lacks {capability} capability")
                continue
            generation = config.generation_params
            requested_output = generation.get("max_tokens", generation.get("max_completion_tokens", 0))
            if not isinstance(requested_output, int) or isinstance(requested_output, bool):
                requested_output = 0
            context_limit = model.context_window
            if context_limit is not None and context_estimate + requested_output > context_limit:
                rejected.append(f"{role}: estimated input and configured output exceed context window")
                continue
            if not self.model_available(model.model_id):
                rejected.append(f"{role}: configured model is unavailable")
                continue
            cost = self._estimated_cost(model, context_estimate, requested_output)
            latency = self.measured_latency_ms.get(config.logical_model_id)
            if max_cost is not None:
                if cost is None:
                    rejected.append(f"{role}: unknown model cost prevents enforcing the cost budget")
                    continue
                if cost > max_cost:
                    rejected.append(f"{role}: known estimated cost exceeds remaining budget")
                    continue
            if max_latency_ms is not None:
                if latency is None:
                    rejected.append(f"{role}: unknown latency prevents enforcing the latency limit")
                    continue
                if latency > max_latency_ms:
                    rejected.append(f"{role}: measured latency exceeds limit")
                    continue
            if role != preferred:
                reasons.append(f"Preferred executor {preferred!r} was unavailable or ineligible.")
            if model.context_window is None:
                reasons.append("Context window is unknown; only the conservative input estimate is available.")
            if model.cost.input_per_1m_tokens is None or model.cost.output_per_1m_tokens is None:
                reasons.append("Model cost is unknown; no cost estimate was applied.")
            if latency is None:
                reasons.append("No measured latency is available for this model.")
            return RoutingDecision(
                task_type=task.task_type,
                requested_agent=request,
                assigned_agent=role,
                logical_model_id=config.logical_model_id,
                model_id=model.model_id,
                fallback=role != preferred,
                context_estimate=context_estimate,
                estimated_output_tokens=requested_output,
                context_limit=context_limit,
                estimated_cost=cost,
                estimated_latency_ms=latency,
                reasons=tuple([*reasons, *rejected]),
            )
        detail = "; ".join(rejected) if rejected else "no executor candidates are configured"
        raise NoAvailableRouteError(f"No route for task {task.task_id!r}: {detail}.")

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Return a conservative rough estimate without a tokenizer dependency."""
        return max(1, (len(text) + 2) // 3)

    @staticmethod
    def _estimated_cost(model: ModelConfig, input_tokens: int, output_tokens: int) -> float | None:
        input_rate = model.cost.input_per_1m_tokens
        output_rate = model.cost.output_per_1m_tokens
        if input_rate is None or output_rate is None:
            return None
        return (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000


__all__ = ["AdaptiveRouter", "NoAvailableRouteError", "RoutingDecision", "RoutingError"]
