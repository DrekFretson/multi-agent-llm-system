"""Unit tests for loading agent settings and resolving their models."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from src.agents import (
    AgentConfigError,
    AgentConfigFileNotFoundError,
    AgentConfigRegistry,
    AgentNotFoundError,
)
from src.llm import ModelRegistry


def write_model_config(tmp_path: Path, model_id: str = "provider/model") -> ModelRegistry:
    """Write a small local model catalog for agent configuration tests."""
    path = tmp_path / "models.yaml"
    path.write_text(
        f"""\
models:
  reasoning:
    provider: openrouter
    model_id: {model_id}
    cost:
      input_per_1m_tokens: null
      output_per_1m_tokens: null
    context_window: null
    capabilities: [text_generation, reasoning]
""",
        encoding="utf-8",
    )
    return ModelRegistry(path)


def write_agent_config(tmp_path: Path, content: str) -> Path:
    """Write agent YAML into a temporary file."""
    path = tmp_path / "agents.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def test_default_configuration_resolves_complete_agent_settings() -> None:
    agents = AgentConfigRegistry()

    coding = agents.get_agent("coding")

    assert coding.logical_model_id == "coding"
    assert coding.model_id == "cohere/north-mini-code:free"
    assert coding.model_config.provider == "openrouter"
    assert coding.generation_params == {"temperature": 0.05, "max_tokens": 3000}
    research = agents.get_agent("research")
    assert research.logical_model_id == "reasoning"
    assert research.generation_params == {"temperature": 0.1, "max_tokens": 3000}
    math_agent = agents.get_agent("math")
    assert math_agent.logical_model_id == "reasoning"
    assert math_agent.generation_params == {"temperature": 0.05, "max_tokens": 4000}
    assert agents.has_agent("research") is True
    assert agents.has_agent("not-configured") is False


def test_unknown_agent_raises_clear_error() -> None:
    with pytest.raises(AgentNotFoundError, match="unknown"):
        AgentConfigRegistry().get_agent("unknown")


def test_unknown_logical_model_reference_is_rejected_at_load(tmp_path: Path) -> None:
    config_path = write_agent_config(
        tmp_path,
        """\
agents:
  coding:
    model: missing
    generation: {temperature: 0.2, max_tokens: 100}
""",
    )

    with pytest.raises(AgentConfigError, match="unregistered model ID 'missing'"):
        AgentConfigRegistry(config_path, write_model_config(tmp_path))


def test_temperature_with_wrong_type_is_rejected(tmp_path: Path) -> None:
    config_path = write_agent_config(
        tmp_path,
        """\
agents:
  coding:
    model: reasoning
    generation: {temperature: warm, max_tokens: 100}
""",
    )

    with pytest.raises(AgentConfigError, match="temperature.*finite number"):
        AgentConfigRegistry(config_path, write_model_config(tmp_path))


def test_negative_max_tokens_is_rejected(tmp_path: Path) -> None:
    config_path = write_agent_config(
        tmp_path,
        """\
agents:
  coding:
    model: reasoning
    generation: {temperature: 0.2, max_tokens: -1}
""",
    )

    with pytest.raises(AgentConfigError, match="max_tokens.*positive integer"):
        AgentConfigRegistry(config_path, write_model_config(tmp_path))


def test_agent_configuration_file_must_exist(tmp_path: Path) -> None:
    with pytest.raises(AgentConfigFileNotFoundError, match="not found"):
        AgentConfigRegistry(tmp_path / "missing.yaml", write_model_config(tmp_path))


def test_malformed_yaml_is_rejected(tmp_path: Path) -> None:
    config_path = write_agent_config(tmp_path, "agents: [unfinished")

    with pytest.raises(AgentConfigError, match="Invalid YAML"):
        AgentConfigRegistry(config_path, write_model_config(tmp_path))


def test_replacing_model_reference_in_yaml_needs_no_code_change(tmp_path: Path) -> None:
    agents_path = write_agent_config(
        tmp_path,
        """\
agents:
  coding:
    model: coding
    generation: {temperature: 0.4, max_tokens: 250}
""",
    )

    agent = AgentConfigRegistry(agents_path, ModelRegistry()).get_agent("coding")

    assert agent.logical_model_id == "coding"
    assert agent.model_id == "cohere/north-mini-code:free"
    assert agent.generation_params == {"temperature": 0.4, "max_tokens": 250}


def test_agent_generation_parameters_are_forwarded_to_llm_client(tmp_path: Path) -> None:
    config_path = write_agent_config(
        tmp_path,
        """\
agents:
  coding:
    model: reasoning
    generation:
      temperature: 0.4
      max_tokens: 250
      top_p: 0.9
      seed: 42
""",
    )
    agent = AgentConfigRegistry(config_path, write_model_config(tmp_path)).get_agent("coding")
    mocked_client = Mock()
    messages = [{"role": "user", "content": "Write a function."}]

    mocked_client.generate(
        model=agent.model_id,
        messages=messages,
        **agent.generation_params,
    )

    mocked_client.generate.assert_called_once_with(
        model="provider/model",
        messages=messages,
        temperature=0.4,
        max_tokens=250,
        top_p=0.9,
        seed=42,
    )
