"""Load and validate local model configuration for LLM consumers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode


class ModelRegistryError(Exception):
    """Base exception for model registry errors."""


class ModelConfigError(ModelRegistryError):
    """Raised when the model configuration cannot be loaded or validated."""


class ModelConfigFileNotFoundError(ModelConfigError):
    """Raised when the configured YAML file does not exist."""


class ModelNotFoundError(ModelRegistryError):
    """Raised when a logical model identifier is not registered."""


SUPPORTED_CAPABILITIES = frozenset(
    {
        "text_generation",
        "code_generation",
        "reasoning",
        "math",
        "tool_calling",
        "vision",
    }
)


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate keys instead of overwriting."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    loader.flatten_mapping(node)
    result: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as error:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "mapping keys must be hashable",
                key_node.start_mark,
            ) from error
        if duplicate:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"duplicate key {key!r}",
                key_node.start_mark,
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


@dataclass(frozen=True)
class ModelCost:
    """Per-million-token input and output prices, or ``None`` if unknown."""

    input_per_1m_tokens: float | None
    output_per_1m_tokens: float | None


@dataclass(frozen=True)
class ModelConfig:
    """Validated configuration for one logical model identifier."""

    provider: str
    model_id: str
    cost: ModelCost
    context_window: int | None
    capabilities: tuple[str, ...]


class ModelRegistry:
    """Load model records from YAML and resolve their logical identifiers."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        """Load and validate a YAML model registry.

        When ``config_path`` is omitted, ``config/models.yaml`` at the project
        root is used. Loading is entirely local and makes no network requests.
        """
        if config_path is None:
            config_path = Path(__file__).resolve().parents[2] / "config" / "models.yaml"
        self.config_path = Path(config_path)
        self._models = self._load_models()

    @property
    def models(self) -> Mapping[str, ModelConfig]:
        """Return a read-only mapping of all logical IDs to model records."""
        return MappingProxyType(self._models)

    def get_model(self, logical_id: str) -> ModelConfig:
        """Return a model by logical ID, or raise :class:`ModelNotFoundError`."""
        try:
            return self._models[logical_id]
        except KeyError:
            raise ModelNotFoundError(
                f"Model with logical ID {logical_id!r} is not registered."
            ) from None

    def has_model(self, logical_id: str) -> bool:
        """Return whether the logical ID exists in the local registry."""
        return logical_id in self._models

    def _load_models(self) -> dict[str, ModelConfig]:
        try:
            with self.config_path.open("r", encoding="utf-8") as config_file:
                data = yaml.load(config_file, Loader=_UniqueKeyLoader)
        except FileNotFoundError as error:
            raise ModelConfigFileNotFoundError(
                f"Model configuration file was not found: {self.config_path}"
            ) from error
        except (OSError, UnicodeError) as error:
            raise ModelConfigError(
                f"Could not read model configuration file: {self.config_path}"
            ) from error
        except yaml.YAMLError as error:
            raise ModelConfigError(
                f"Invalid YAML in model configuration {self.config_path}: {error}"
            ) from error

        if not isinstance(data, dict):
            raise ModelConfigError("The YAML document must be a mapping with a 'models' key.")
        if set(data) != {"models"}:
            raise ModelConfigError("The YAML document must contain only the 'models' key.")

        raw_models = data["models"]
        if not isinstance(raw_models, dict):
            raise ModelConfigError("'models' must be a mapping of logical IDs to model records.")

        models: dict[str, ModelConfig] = {}
        for logical_id, raw_model in raw_models.items():
            if (
                not isinstance(logical_id, str)
                or not logical_id.strip()
                or logical_id != logical_id.strip()
            ):
                raise ModelConfigError("Every model logical ID must be a non-empty string.")
            if not isinstance(raw_model, dict):
                raise ModelConfigError(
                    f"Model {logical_id!r} must be defined as a mapping."
                )
            models[logical_id] = self._parse_model(logical_id, raw_model)
        return models

    @staticmethod
    def _parse_model(logical_id: str, raw_model: dict[object, object]) -> ModelConfig:
        required_fields = {
            "provider",
            "model_id",
            "cost",
            "context_window",
            "capabilities",
        }
        missing = required_fields - raw_model.keys()
        extra = raw_model.keys() - required_fields
        if missing:
            raise ModelConfigError(
                f"Model {logical_id!r} is missing required field(s): "
                f"{', '.join(sorted(map(str, missing)))}."
            )
        if extra:
            raise ModelConfigError(
                f"Model {logical_id!r} has unknown field(s): "
                f"{', '.join(sorted(map(str, extra)))}."
            )

        provider = raw_model["provider"]
        model_id = raw_model["model_id"]
        if not isinstance(provider, str) or not provider.strip():
            raise ModelConfigError(f"Model {logical_id!r}: 'provider' must be a non-empty string.")
        if not isinstance(model_id, str) or not model_id.strip():
            raise ModelConfigError(f"Model {logical_id!r}: 'model_id' must be a non-empty string.")
        if provider != provider.strip() or model_id != model_id.strip():
            raise ModelConfigError(
                f"Model {logical_id!r}: 'provider' and 'model_id' cannot have surrounding whitespace."
            )

        raw_cost = raw_model["cost"]
        cost_fields = {"input_per_1m_tokens", "output_per_1m_tokens"}
        if not isinstance(raw_cost, dict) or raw_cost.keys() != cost_fields:
            raise ModelConfigError(
                f"Model {logical_id!r}: 'cost' must contain exactly "
                "'input_per_1m_tokens' and 'output_per_1m_tokens'."
            )
        cost = ModelCost(
            input_per_1m_tokens=ModelRegistry._parse_cost(
                logical_id, "input_per_1m_tokens", raw_cost["input_per_1m_tokens"]
            ),
            output_per_1m_tokens=ModelRegistry._parse_cost(
                logical_id, "output_per_1m_tokens", raw_cost["output_per_1m_tokens"]
            ),
        )

        context_window = raw_model["context_window"]
        if context_window is not None and (
            isinstance(context_window, bool)
            or not isinstance(context_window, int)
            or context_window <= 0
        ):
            raise ModelConfigError(
                f"Model {logical_id!r}: 'context_window' must be a positive integer or null."
            )

        raw_capabilities = raw_model["capabilities"]
        if not isinstance(raw_capabilities, list) or any(
            not isinstance(capability, str) for capability in raw_capabilities
        ):
            raise ModelConfigError(
                f"Model {logical_id!r}: 'capabilities' must be a list of strings."
            )
        if len(raw_capabilities) != len(set(raw_capabilities)):
            raise ModelConfigError(f"Model {logical_id!r}: 'capabilities' contains duplicates.")
        unknown_capabilities = set(raw_capabilities) - SUPPORTED_CAPABILITIES
        if unknown_capabilities:
            raise ModelConfigError(
                f"Model {logical_id!r}: unsupported capability value(s): "
                f"{', '.join(sorted(unknown_capabilities))}."
            )

        return ModelConfig(
            provider=provider,
            model_id=model_id,
            cost=cost,
            context_window=context_window,
            capabilities=tuple(raw_capabilities),
        )

    @staticmethod
    def _parse_cost(logical_id: str, field: str, value: object) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ModelConfigError(
                f"Model {logical_id!r}: cost field {field!r} must be a non-negative number or null."
            )
        if not math.isfinite(value) or value < 0:
            raise ModelConfigError(
                f"Model {logical_id!r}: cost field {field!r} must be finite and non-negative."
            )
        return float(value)
