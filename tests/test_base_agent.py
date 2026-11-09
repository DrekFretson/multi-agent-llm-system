"""Mock tests for the reusable base agent."""

from unittest.mock import Mock

import pytest

from src.agents import (
    AgentConfig,
    AgentConfigRegistry,
    AgentInputError,
    AgentResult,
    BaseAgent,
)
from src.llm import LLMAPIError, LLMClient, LLMResponse


class ExampleAgent(BaseAgent):
    """Minimal concrete agent used to verify the inheritance interface."""


@pytest.fixture
def agent_config() -> AgentConfig:
    return AgentConfigRegistry().get_agent("coding")


@pytest.fixture
def client() -> Mock:
    return Mock(spec=LLMClient)


def make_agent(client: Mock, config: AgentConfig) -> ExampleAgent:
    return ExampleAgent(
        agent_id="coding-1",
        name="Coding Agent",
        system_prompt="You write clear, correct code.",
        client=client,
        config=config,
    )


def test_simple_base_agent_subclass_can_be_created(
    client: Mock,
    agent_config: AgentConfig,
) -> None:
    agent = make_agent(client, agent_config)

    assert agent.agent_id == "coding-1"
    assert agent.name == "Coding Agent"
    assert agent.system_prompt == "You write clear, correct code."
    assert agent.client is client
    assert agent.config is agent_config


def test_run_returns_agent_result_on_success(
    client: Mock,
    agent_config: AgentConfig,
) -> None:
    client.generate.return_value = LLMResponse(
        content="def add(a, b): return a + b",
        model="provider/served-model",
        prompt_tokens=12,
        completion_tokens=8,
        total_tokens=20,
        finish_reason="stop",
    )

    result = make_agent(client, agent_config).run("Write an add function.")

    assert result == AgentResult(
        agent_id="coding-1",
        content="def add(a, b): return a + b",
        model="provider/served-model",
        success=True,
        error=None,
        prompt_tokens=12,
        completion_tokens=8,
    )


def test_run_sends_system_and_user_messages(
    client: Mock,
    agent_config: AgentConfig,
) -> None:
    client.generate.return_value = LLMResponse(
        content="Done",
        model=None,
        prompt_tokens=None,
        completion_tokens=None,
        total_tokens=None,
        finish_reason=None,
    )

    make_agent(client, agent_config).run("Do this task.")

    assert client.generate.call_args.kwargs["messages"] == [
        {"role": "system", "content": "You write clear, correct code."},
        {"role": "user", "content": "Do this task."},
    ]


def test_run_allows_a_temporary_system_prompt_override(
    client: Mock,
    agent_config: AgentConfig,
) -> None:
    client.generate.return_value = LLMResponse(
        content="Done",
        model=None,
        prompt_tokens=1,
        completion_tokens=1,
        total_tokens=2,
        finish_reason="stop",
    )

    make_agent(client, agent_config).run("Plan searches.", system_prompt="Plan only.")

    assert client.generate.call_args.kwargs["messages"][0] == {
        "role": "system",
        "content": "Plan only.",
    }


def test_run_uses_resolved_model_and_generation_parameters(
    client: Mock,
    agent_config: AgentConfig,
) -> None:
    client.generate.return_value = LLMResponse(
        content="Done",
        model=None,
        prompt_tokens=1,
        completion_tokens=1,
        total_tokens=2,
        finish_reason="stop",
    )

    make_agent(client, agent_config).run("Do this task.")

    call = client.generate.call_args.kwargs
    assert call["model"] == agent_config.model_id
    assert call["agent_id"] == "coding-1"
    for key, value in agent_config.generation_params.items():
        assert call[key] == value
    assert call["messages"][0]["role"] == "system"
    assert call["messages"][1]["role"] == "user"


@pytest.mark.parametrize("task", ["", " \n\t ", None, 42])
def test_invalid_task_raises_input_error_without_calling_client(
    task: object,
    client: Mock,
    agent_config: AgentConfig,
) -> None:
    with pytest.raises(AgentInputError, match="non-empty string"):
        make_agent(client, agent_config).run(task)  # type: ignore[arg-type]

    client.generate.assert_not_called()


def test_llm_client_error_becomes_failed_result(
    client: Mock,
    agent_config: AgentConfig,
) -> None:
    client.generate.side_effect = LLMAPIError(
        "OpenRouter API error (HTTP 503): provider unavailable",
        status_code=503,
        request_id="req_test",
    )

    result = make_agent(client, agent_config).run("Do this task.")

    assert result == AgentResult(
        agent_id="coding-1",
        content="",
        model=agent_config.model_id,
        success=False,
        error="LLMAPIError: OpenRouter API error (HTTP 503): provider unavailable",
        prompt_tokens=None,
        completion_tokens=None,
    )


@pytest.mark.parametrize("content", [None, "", " \n "])
def test_empty_model_response_becomes_failed_result(
    content: str | None,
    client: Mock,
    agent_config: AgentConfig,
) -> None:
    client.generate.return_value = LLMResponse(
        content=content,
        model=None,
        prompt_tokens=5,
        completion_tokens=0,
        total_tokens=5,
        finish_reason="stop",
    )

    result = make_agent(client, agent_config).run("Do this task.")

    assert result.success is False
    assert result.content == ""
    assert result.error == "The model returned an empty response."
    assert result.model == agent_config.model_id
    assert result.prompt_tokens == 5
    assert result.completion_tokens == 0


def test_unexpected_programming_errors_are_not_swallowed(
    client: Mock,
    agent_config: AgentConfig,
) -> None:
    client.generate.side_effect = ValueError("unexpected bug")

    with pytest.raises(ValueError, match="unexpected bug"):
        make_agent(client, agent_config).run("Do this task.")
