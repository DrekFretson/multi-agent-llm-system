"""Unit tests for the local YAML-backed model registry."""

from pathlib import Path

import pytest
import yaml

from src.llm import (
    ModelConfigError,
    ModelConfigFileNotFoundError,
    ModelNotFoundError,
    ModelRegistry,
)
from src.llm import client as client_module


VALID_MODELS_YAML = """\
models:
  reasoning:
    provider: openrouter
    model_id: provider/reasoner
    cost:
      input_per_1m_tokens: null
      output_per_1m_tokens: null
    context_window: null
    capabilities: [text_generation, reasoning]
  coding:
    provider: openrouter
    model_id: provider/coder
    cost:
      input_per_1m_tokens: 0.5
      output_per_1m_tokens: 1.25
    context_window: 64000
    capabilities: [text_generation, code_generation, tool_calling]
"""


def make_registry(tmp_path: Path, contents: str = VALID_MODELS_YAML) -> ModelRegistry:
    """Write test YAML to a temporary file and load it into a registry."""
    config_path = tmp_path / "models.yaml"
    config_path.write_text(contents, encoding="utf-8")
    return ModelRegistry(config_path)


def test_default_config_loads_multiple_models_and_null_values() -> None:
    registry = ModelRegistry()

    assert registry.has_model("reasoning")
    assert registry.has_model("coding")
    assert registry.has_model("math")
    assert registry.get_model("reasoning").context_window == 1_000_000
    assert registry.get_model("reasoning").cost.input_per_1m_tokens == 0.0
    assert registry.get_model("reasoning").capabilities == (
        "text_generation",
        "code_generation",
        "reasoning",
        "math",
        "tool_calling",
    )
    assert registry.get_model("coding").capabilities == (
        "text_generation",
        "code_generation",
        "reasoning",
        "tool_calling",
    )
    math_model = registry.get_model("math")
    assert math_model.model_id == "google/gemma-4-31b-it:free"
    assert math_model.context_window == 262144
    assert math_model.capabilities == ("text_generation", "reasoning", "math")


def test_loads_multiple_models_and_resolves_logical_id(tmp_path: Path) -> None:
    registry = make_registry(tmp_path)

    model = registry.get_model("coding")
    reasoning = registry.get_model("reasoning")

    assert model.provider == "openrouter"
    assert model.model_id == "provider/coder"
    assert model.cost.input_per_1m_tokens == 0.5
    assert model.cost.output_per_1m_tokens == 1.25
    assert model.context_window == 64000
    assert model.capabilities == ("text_generation", "code_generation", "tool_calling")
    assert len(registry.models) == 2
    assert reasoning.context_window is None
    assert reasoning.cost.output_per_1m_tokens is None


def test_get_model_for_unknown_logical_id_raises(tmp_path: Path) -> None:
    registry = make_registry(tmp_path)

    with pytest.raises(ModelNotFoundError, match="unknown"):
        registry.get_model("unknown")


def test_has_model_checks_only_local_registry(tmp_path: Path) -> None:
    registry = make_registry(tmp_path)

    assert registry.has_model("reasoning") is True
    assert registry.has_model("missing") is False


def test_missing_yaml_file_raises_specific_error(tmp_path: Path) -> None:
    with pytest.raises(ModelConfigFileNotFoundError, match="not found"):
        ModelRegistry(tmp_path / "missing.yaml")


def test_invalid_yaml_raises_configuration_error(tmp_path: Path) -> None:
    with pytest.raises(ModelConfigError, match="Invalid YAML"):
        make_registry(tmp_path, "models: [unfinished")


def test_invalid_root_structure_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ModelConfigError, match="'models' must be a mapping"):
        make_registry(tmp_path, "models: []")


def test_required_model_fields_are_validated(tmp_path: Path) -> None:
    incomplete = """\
models:
  coding:
    provider: openrouter
    model_id: provider/coder
"""

    with pytest.raises(ModelConfigError, match="missing required field"):
        make_registry(tmp_path, incomplete)


def test_unknown_model_fields_are_not_silently_ignored(tmp_path: Path) -> None:
    contents = VALID_MODELS_YAML.replace(
        "    model_id: provider/reasoner\n",
        "    model_id: provider/reasoner\n    unrecognized: true\n",
    )

    with pytest.raises(ModelConfigError, match="unknown field.*unrecognized"):
        make_registry(tmp_path, contents)


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ("provider: 12", "provider.*string"),
        ("context_window: true", "context_window.*integer"),
        ("capabilities: text_generation", "capabilities.*list"),
        ("input_per_1m_tokens: 'free'", "cost field.*number"),
    ],
)
def test_invalid_field_types_are_rejected(
    tmp_path: Path,
    replacement: str,
    message: str,
) -> None:
    contents = VALID_MODELS_YAML.replace("provider: openrouter", replacement, 1)
    if replacement.startswith("context_window:"):
        contents = VALID_MODELS_YAML.replace("context_window: null", replacement, 1)
    elif replacement.startswith("capabilities:"):
        contents = VALID_MODELS_YAML.replace(
            "capabilities: [text_generation, reasoning]", replacement, 1
        )
    elif replacement.startswith("input_per_1m_tokens:"):
        contents = VALID_MODELS_YAML.replace("input_per_1m_tokens: null", replacement, 1)

    with pytest.raises(ModelConfigError, match=message):
        make_registry(tmp_path, contents)


def test_unknown_capabilities_are_rejected(tmp_path: Path) -> None:
    contents = VALID_MODELS_YAML.replace(
        "capabilities: [text_generation, reasoning]",
        "capabilities: [text_generation, made_up_capability]",
    )

    with pytest.raises(ModelConfigError, match="unsupported capability"):
        make_registry(tmp_path, contents)


def test_duplicate_logical_ids_in_yaml_are_not_silently_overwritten(
    tmp_path: Path,
) -> None:
    contents = VALID_MODELS_YAML + "  coding:\n    provider: other\n"

    with pytest.raises(ModelConfigError, match="duplicate key 'coding'"):
        make_registry(tmp_path, contents)


def test_multiple_agents_can_reference_the_same_registered_model() -> None:
    registry = ModelRegistry()
    agents_path = Path(__file__).resolve().parents[1] / "config" / "agents.yaml"
    agents = yaml.safe_load(agents_path.read_text(encoding="utf-8"))["agents"]

    orchestrator_model = registry.get_model(agents["orchestrator"]["model"])
    research_model = registry.get_model(agents["research"]["model"])
    critic_model = registry.get_model(agents["critic"]["model"])

    assert orchestrator_model is research_model is critic_model
    assert registry.get_model(agents["coding"]["model"]).model_id == "cohere/north-mini-code:free"


def test_registry_does_not_construct_or_call_llm_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_client(*args: object, **kwargs: object) -> None:
        raise AssertionError("ModelRegistry must not construct an API client")

    monkeypatch.setattr(client_module, "OpenAI", unexpected_client)

    registry = make_registry(tmp_path)

    assert registry.has_model("coding")
