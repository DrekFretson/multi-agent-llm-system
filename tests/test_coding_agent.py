"""Mock tests for CodingAgent and its structured response parser."""

import json
from unittest.mock import Mock

import pytest

from src.agents import (
    AgentConfigRegistry,
    AgentResult,
    CodingAgent,
    CodingSolution,
    CodingSolutionParseError,
    CodingSolutionValidationError,
)
from src.llm import LLMClient, LLMResponse


SOLUTION_PAYLOAD = {
    "language": "python",
    "summary": "Adds two integers.",
    "code": "def add(a, b):\n    return a + b",
    "tests": "def test_add():\n    assert add(2, 3) == 5",
    "dependencies": [],
    "run_instructions": "Run pytest tests.py.",
}


@pytest.fixture
def client() -> Mock:
    return Mock(spec=LLMClient)


@pytest.fixture
def config_registry() -> AgentConfigRegistry:
    return AgentConfigRegistry()


def prepare_response(client: Mock, payload: object = SOLUTION_PAYLOAD) -> None:
    content = payload if isinstance(payload, str) else json.dumps(payload)
    client.generate.return_value = LLMResponse(
        content=content,
        model="provider/served-model",
        prompt_tokens=20,
        completion_tokens=30,
        total_tokens=50,
        finish_reason="stop",
    )


def test_coding_agent_loads_its_role_config_and_has_coding_prompt(
    client: Mock,
    config_registry: AgentConfigRegistry,
) -> None:
    agent = CodingAgent(client=client, config_registry=config_registry)

    assert agent.config is config_registry.get_agent("coding")
    assert agent.config.logical_model_id == "coding"
    assert agent.name == "Coding Agent"
    assert "сложност" in agent.system_prompt.lower()
    assert "JSON" in agent.system_prompt
    assert "test" in agent.system_prompt.lower()
    assert "не утверждай" in agent.system_prompt.lower()
    assert "сверь каждый ожидаемый результат теста" in agent.system_prompt.lower()
    assert "преобразования входа" in agent.system_prompt.lower()
    assert "объём ответа под задачу" in agent.system_prompt.lower()


def test_run_passes_basic_task_and_returns_base_agent_result(client: Mock) -> None:
    prepare_response(client)
    agent = CodingAgent(client=client)

    result = agent.run("Write a Python function that adds two numbers.")

    assert isinstance(result, AgentResult)
    assert result.success is True
    assert result.model == "provider/served-model"
    assert json.loads(result.content) == SOLUTION_PAYLOAD
    call = client.generate.call_args.kwargs
    assert call["agent_id"] == "coding"
    assert call["model"] == agent.config.model_id
    assert call["messages"][0]["role"] == "system"
    assert call["messages"][0]["content"] == agent.system_prompt
    assert call["messages"][1] == {
        "role": "user",
        "content": "Техническое задание:\nWrite a Python function that adds two numbers.",
    }
    for key, value in agent.config.generation_params.items():
        assert call[key] == value


def test_run_includes_context_existing_code_and_test_requirements(client: Mock) -> None:
    prepare_response(client)
    agent = CodingAgent(client=client)

    agent.run(
        "Add input validation.",
        additional_context="This is part of a small calculator.",
        existing_code="def add(a, b):\n    return a + b",
        test_requirements="Cover non-numeric input and zero.",
    )

    user_content = client.generate.call_args.kwargs["messages"][1]["content"]
    assert "Техническое задание:\nAdd input validation." in user_content
    assert "Дополнительный контекст:\nThis is part of a small calculator." in user_content
    assert "Существующий код для доработки:" in user_content
    assert "def add(a, b):\n    return a + b" in user_content
    assert "Требования к тестам:\nCover non-numeric input and zero." in user_content


def test_run_works_without_optional_context(client: Mock) -> None:
    prepare_response(client)
    agent = CodingAgent(client=client)

    result = agent.run("Create a hello function.")

    user_content = client.generate.call_args.kwargs["messages"][1]["content"]
    assert result.success
    assert "Дополнительный контекст" not in user_content
    assert "Существующий код" not in user_content
    assert "Требования к тестам" not in user_content


def test_parse_solution_returns_typed_coding_solution(client: Mock) -> None:
    prepare_response(client)
    agent = CodingAgent(client=client)
    result = agent.run("Create an add function.")

    solution = agent.parse_solution(result)

    assert solution == CodingSolution(
        language="python",
        summary="Adds two integers.",
        code="def add(a, b):\n    return a + b",
        tests="def test_add():\n    assert add(2, 3) == 5",
        dependencies=[],
        run_instructions="Run pytest tests.py.",
    )


def test_parse_solution_reports_invalid_json(client: Mock) -> None:
    prepare_response(client, "This is not JSON")
    agent = CodingAgent(client=client)
    result = agent.run("Create an add function.", max_format_retries=0)

    with pytest.raises(CodingSolutionParseError, match="invalid JSON"):
        agent.parse_solution(result)


def test_parse_solution_reports_missing_required_fields(client: Mock) -> None:
    prepare_response(client, {"language": "python", "code": "print('hi')"})
    agent = CodingAgent(client=client)
    result = agent.run("Create a hello program.", max_format_retries=0)

    with pytest.raises(CodingSolutionValidationError, match="missing required field"):
        agent.parse_solution(result)


def test_parse_solution_rejects_obvious_placeholder_as_completed_code(client: Mock) -> None:
    placeholder = {**SOLUTION_PAYLOAD, "code": "def solve():\n    pass"}
    prepare_response(client, placeholder)
    agent = CodingAgent(client=client)
    result = agent.run("Implement the requested behavior.", max_format_retries=0)

    with pytest.raises(CodingSolutionValidationError, match="obvious placeholder"):
        agent.parse_solution(result)


def test_parse_solution_preserves_structured_need_for_user_input(client: Mock) -> None:
    clarification = {
        **SOLUTION_PAYLOAD,
        "code": "",
        "task_completed": False,
        "needs_input": True,
        "clarification_question": "Which file format should be supported?",
    }
    prepare_response(client, clarification)
    agent = CodingAgent(client=client)
    solution = agent.parse_solution(agent.run("Build an importer."))

    assert solution.task_completed is False
    assert solution.needs_input is True
    assert solution.clarification_question == "Which file format should be supported?"


def test_parse_solution_rejects_failed_agent_result(client: Mock) -> None:
    agent = CodingAgent(client=client)
    result = AgentResult(
        agent_id="coding",
        content="",
        model=agent.config.model_id,
        success=False,
        error="LLMAPIError: unavailable",
        prompt_tokens=None,
        completion_tokens=None,
    )

    with pytest.raises(CodingSolutionValidationError, match="unsuccessful"):
        agent.parse_solution(result)


def test_coding_agent_does_not_create_or_call_a_real_llm_client(client: Mock) -> None:
    prepare_response(client)
    agent = CodingAgent(client=client)

    result = agent.run("Create a small function.")

    client.generate.assert_called_once()
    assert result.success


def test_run_repairs_malformed_json_once_and_accounts_for_both_calls(client: Mock) -> None:
    malformed = LLMResponse(
        content='{"language": "python",\n  this is commentary',
        model="provider/model",
        prompt_tokens=10,
        completion_tokens=15,
        total_tokens=25,
        finish_reason="stop",
    )
    repaired = LLMResponse(
        content=json.dumps(SOLUTION_PAYLOAD),
        model="provider/model",
        prompt_tokens=20,
        completion_tokens=30,
        total_tokens=50,
        finish_reason="stop",
    )
    client.generate.side_effect = [malformed, repaired]
    agent = CodingAgent(client=client)

    result = agent.run("Create an add function.")

    assert result.success
    assert result.format_repair_attempts == 1
    assert result.prompt_tokens == 30
    assert result.completion_tokens == 45
    assert result.total_tokens == 75
    assert agent.parse_solution(result).code == SOLUTION_PAYLOAD["code"]
    assert client.generate.call_count == 2
    repair_task = client.generate.call_args.kwargs["messages"][1]["content"]
    assert "Предыдущий ответ" in repair_task


def test_run_repairs_truncated_placeholder_response_once(client: Mock) -> None:
    truncated_payload = {**SOLUTION_PAYLOAD, "code": "def analyze_csv(path):\n    pass"}
    truncated = LLMResponse(
        content=json.dumps(truncated_payload),
        model="provider/model",
        prompt_tokens=10,
        completion_tokens=5000,
        total_tokens=5010,
        finish_reason="length",
    )
    repaired = LLMResponse(
        content=json.dumps(SOLUTION_PAYLOAD),
        model="provider/model",
        prompt_tokens=20,
        completion_tokens=30,
        total_tokens=50,
        finish_reason="stop",
    )
    client.generate.side_effect = [truncated, repaired]
    agent = CodingAgent(client=client)

    result = agent.run("Implement a CSV analyzer.")

    assert result.success
    assert result.finish_reason == "stop"
    assert result.format_repair_attempts == 1
    assert agent.parse_solution(result).code == SOLUTION_PAYLOAD["code"]
    assert client.generate.call_count == 2


def test_run_repairs_schema_valid_response_marked_as_truncated(client: Mock) -> None:
    truncated = LLMResponse(
        content=json.dumps(SOLUTION_PAYLOAD),
        model="provider/model",
        prompt_tokens=10,
        completion_tokens=5000,
        total_tokens=5010,
        finish_reason="length",
    )
    repaired = LLMResponse(
        content=json.dumps(SOLUTION_PAYLOAD),
        model="provider/model",
        prompt_tokens=20,
        completion_tokens=30,
        total_tokens=50,
        finish_reason="stop",
    )
    client.generate.side_effect = [truncated, repaired]
    agent = CodingAgent(client=client)

    result = agent.run("Implement a CSV analyzer.")

    assert result.success
    assert result.finish_reason == "stop"
    assert result.format_repair_attempts == 1
    assert client.generate.call_count == 2


def test_run_does_not_repair_when_no_retry_slot_is_available(client: Mock) -> None:
    prepare_response(client, "not JSON")
    agent = CodingAgent(client=client)

    result = agent.run("Create an add function.", max_format_retries=0)

    with pytest.raises(CodingSolutionParseError):
        agent.parse_solution(result)
    client.generate.assert_called_once()
