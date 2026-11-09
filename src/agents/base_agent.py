"""Reusable base class for agents that delegate generation to an LLMClient."""

from __future__ import annotations

from dataclasses import dataclass

from src.agents.config import AgentConfig
from src.llm import LLMClient, LLMClientError


class AgentInputError(ValueError):
    """Raised when an agent receives an invalid task."""


@dataclass(frozen=True)
class AgentResult:
    """Consistent result returned by :meth:`BaseAgent.run`."""

    agent_id: str
    content: str
    model: str
    success: bool
    error: str | None
    prompt_tokens: int | None
    completion_tokens: int | None


class BaseAgent:
    """Base implementation for agents with injected client and role config."""

    def __init__(
        self,
        agent_id: str,
        name: str,
        system_prompt: str,
        client: LLMClient,
        config: AgentConfig,
    ) -> None:
        """Create an agent using dependencies and instructions supplied by caller."""
        for field_name, value in (
            ("agent_id", agent_id),
            ("name", name),
            ("system_prompt", system_prompt),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string.")

        self.agent_id = agent_id
        self.name = name
        self.system_prompt = system_prompt
        self.client = client
        self.config = config

    def run(
        self,
        task: str,
        *,
        system_prompt: str | None = None,
    ) -> AgentResult:
        """Run a task with the configured model and return a normalized result.

        Invalid tasks raise :class:`AgentInputError`. Expected client failures
        become unsuccessful results; unexpected programming errors propagate.
        """
        if not isinstance(task, str) or not task.strip():
            raise AgentInputError("task must be a non-empty string.")

        active_system_prompt = self.system_prompt if system_prompt is None else system_prompt
        if not isinstance(active_system_prompt, str) or not active_system_prompt.strip():
            raise AgentInputError("system_prompt must be a non-empty string.")

        messages = [
            {"role": "system", "content": active_system_prompt},
            {"role": "user", "content": task},
        ]
        model_id = self.config.model_id

        try:
            response = self.client.generate(
                model=model_id,
                messages=messages,
                agent_id=self.agent_id,
                **self.config.generation_params,
            )
        except LLMClientError as error:
            return AgentResult(
                agent_id=self.agent_id,
                content="",
                model=model_id,
                success=False,
                error=f"{type(error).__name__}: {error}",
                prompt_tokens=None,
                completion_tokens=None,
            )

        response_model = response.model or model_id
        content = response.content
        if content is None or not content.strip():
            return AgentResult(
                agent_id=self.agent_id,
                content="",
                model=response_model,
                success=False,
                error="The model returned an empty response.",
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
            )

        return AgentResult(
            agent_id=self.agent_id,
            content=content,
            model=response_model,
            success=True,
            error=None,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
        )
