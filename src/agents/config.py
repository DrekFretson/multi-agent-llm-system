"""Load agent role settings and resolve their logical model references."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode

from src.llm.model_registry import ModelConfig, ModelNotFoundError, ModelRegistry


class AgentConfigError(Exception):
    """Raised when agent configuration is missing, invalid, or inconsistent."""


class AgentConfigFileNotFoundError(AgentConfigError):
    """Raised when the configured agents YAML file does not exist."""


class AgentNotFoundError(AgentConfigError):
    """Raised when a requested agent name is not configured."""


_GENERATION_PARAMETERS = frozenset(
    {
        "audio",
        "frequency_penalty",
        "function_call",
        "functions",
        "logit_bias",
        "logprobs",
        "max_completion_tokens",
        "max_tokens",
        "metadata",
        "modalities",
        "moderation",
        "n",
        "parallel_tool_calls",
        "prediction",
        "presence_penalty",
        "prompt_cache_key",
        "prompt_cache_options",
        "prompt_cache_retention",
        "reasoning",
        "reasoning_effort",
        "response_format",
        "safety_identifier",
        "seed",
        "service_tier",
        "stop",
        "store",
        "stream",
        "stream_options",
        "temperature",
        "tool_choice",
        "tools",
        "top_logprobs",
        "top_p",
        "user",
        "verbosity",
        "web_search_options",
    }
)


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


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
class AgentConfig:
    """Resolved role configuration with a concrete model and generation data."""

    name: str
    logical_model_id: str
    model_config: ModelConfig
    _generation_params: dict[str, Any] = field(repr=False)

    @property
    def model_id(self) -> str:
        """Return the concrete provider model ID resolved from the registry."""
        return self.model_config.model_id

    @property
    def generation_params(self) -> dict[str, Any]:
        """Return a copy of the generation parameters for a client call."""
        return deepcopy(self._generation_params)


class AgentConfigRegistry:
    """Load agent YAML and resolve each role through a :class:`ModelRegistry`."""

    def __init__(
        self,
        config_path: str | Path | None = None,
        model_registry: ModelRegistry | None = None,
    ) -> None:
        """Load and validate agents, using project defaults when omitted."""
        if config_path is None:
            config_path = Path(__file__).resolve().parents[2] / "config" / "agents.yaml"
        self.config_path = Path(config_path)
        self.model_registry = model_registry or ModelRegistry()
        self._agents = self._load_agents()

    def get_agent(self, name: str) -> AgentConfig:
        """Return the fully resolved configuration for an agent name."""
        try:
            return self._agents[name]
        except KeyError:
            raise AgentNotFoundError(f"Agent {name!r} is not configured.") from None

    def has_agent(self, name: str) -> bool:
        """Return whether an agent name is present in the local configuration."""
        return name in self._agents

    @property
    def agents(self) -> dict[str, AgentConfig]:
        """Return a copy of all resolved agent configurations."""
        return dict(self._agents)

    def _load_agents(self) -> dict[str, AgentConfig]:
        try:
            with self.config_path.open("r", encoding="utf-8") as config_file:
                data = yaml.load(config_file, Loader=_UniqueKeyLoader)
        except FileNotFoundError as error:
            raise AgentConfigFileNotFoundError(
                f"Agent configuration file was not found: {self.config_path}"
            ) from error
        except (OSError, UnicodeError) as error:
            raise AgentConfigError(
                f"Could not read agent configuration file: {self.config_path}"
            ) from error
        except yaml.YAMLError as error:
            raise AgentConfigError(
                f"Invalid YAML in agent configuration {self.config_path}: {error}"
            ) from error

        if not isinstance(data, dict) or set(data) != {"agents"}:
            raise AgentConfigError("The YAML document must contain only an 'agents' mapping.")
        raw_agents = data["agents"]
        if not isinstance(raw_agents, dict):
            raise AgentConfigError("'agents' must be a mapping of names to agent settings.")

        agents: dict[str, AgentConfig] = {}
        for name, raw_agent in raw_agents.items():
            if (
                not isinstance(name, str)
                or not name.strip()
                or name != name.strip()
            ):
                raise AgentConfigError("Every agent name must be a non-empty string without outer whitespace.")
            if not isinstance(raw_agent, dict):
                raise AgentConfigError(f"Agent {name!r} must be defined as a mapping.")
            agents[name] = self._parse_agent(name, raw_agent)
        return agents

    def _parse_agent(self, name: str, raw_agent: dict[object, object]) -> AgentConfig:
        required_fields = {"model", "generation"}
        missing = required_fields - raw_agent.keys()
        extra = raw_agent.keys() - required_fields
        if missing:
            raise AgentConfigError(
                f"Agent {name!r} is missing required field(s): "
                f"{', '.join(sorted(map(str, missing)))}."
            )
        if extra:
            raise AgentConfigError(
                f"Agent {name!r} has unknown field(s): "
                f"{', '.join(sorted(map(str, extra)))}."
            )

        logical_model_id = raw_agent["model"]
        if not isinstance(logical_model_id, str) or not logical_model_id.strip():
            raise AgentConfigError(f"Agent {name!r}: 'model' must be a non-empty logical ID.")
        if logical_model_id != logical_model_id.strip():
            raise AgentConfigError(f"Agent {name!r}: 'model' cannot have surrounding whitespace.")
        try:
            model_config = self.model_registry.get_model(logical_model_id)
        except ModelNotFoundError as error:
            raise AgentConfigError(
                f"Agent {name!r} references unregistered model ID {logical_model_id!r}."
            ) from error

        raw_generation = raw_agent["generation"]
        if not isinstance(raw_generation, dict):
            raise AgentConfigError(f"Agent {name!r}: 'generation' must be a mapping.")
        self._validate_generation_params(name, raw_generation)

        return AgentConfig(
            name=name,
            logical_model_id=logical_model_id,
            model_config=model_config,
            _generation_params=deepcopy(raw_generation),
        )

    @staticmethod
    def _validate_generation_params(name: str, params: dict[object, object]) -> None:
        if any(not isinstance(key, str) for key in params):
            raise AgentConfigError(f"Agent {name!r}: generation parameter names must be strings.")
        unknown = params.keys() - _GENERATION_PARAMETERS
        if unknown:
            raise AgentConfigError(
                f"Agent {name!r}: unsupported generation parameter(s): "
                f"{', '.join(sorted(unknown))}."
            )

        for key, value in params.items():
            if key == "temperature":
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or not 0 <= value <= 2
                ):
                    raise AgentConfigError(
                        f"Agent {name!r}: 'temperature' must be a finite number from 0 to 2."
                    )
            elif key in {"max_tokens", "max_completion_tokens", "n"}:
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise AgentConfigError(
                        f"Agent {name!r}: '{key}' must be a positive integer."
                    )
            elif key == "seed":
                if isinstance(value, bool) or not isinstance(value, int):
                    raise AgentConfigError(f"Agent {name!r}: 'seed' must be an integer.")
            elif key in {"top_p", "frequency_penalty", "presence_penalty"}:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise AgentConfigError(f"Agent {name!r}: '{key}' must be a number.")
                if not math.isfinite(value):
                    raise AgentConfigError(f"Agent {name!r}: '{key}' must be finite.")
                bounds = (0, 1) if key == "top_p" else (-2, 2)
                if not bounds[0] <= value <= bounds[1]:
                    raise AgentConfigError(
                        f"Agent {name!r}: '{key}' must be between {bounds[0]} and {bounds[1]}."
                    )
            elif key == "top_logprobs":
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or not 0 <= value <= 20
                ):
                    raise AgentConfigError(
                        f"Agent {name!r}: 'top_logprobs' must be an integer from 0 to 20."
                    )
            elif key in {"logprobs", "parallel_tool_calls", "store", "stream"}:
                if not isinstance(value, bool):
                    raise AgentConfigError(f"Agent {name!r}: '{key}' must be a boolean.")
            elif key in {
                "prompt_cache_key",
                "prompt_cache_retention",
                "reasoning_effort",
                "safety_identifier",
                "service_tier",
                "user",
                "verbosity",
            }:
                if not isinstance(value, str):
                    raise AgentConfigError(f"Agent {name!r}: '{key}' must be a string.")
            elif key in {"function_call", "tool_choice"}:
                if not isinstance(value, (str, dict)):
                    raise AgentConfigError(
                        f"Agent {name!r}: '{key}' must be a string or mapping."
                    )
            elif key == "stop":
                if not isinstance(value, str) and not (
                    isinstance(value, list)
                    and all(isinstance(item, str) for item in value)
                ):
                    raise AgentConfigError(
                        f"Agent {name!r}: 'stop' must be a string or list of strings."
                    )
            elif key in {"functions", "modalities", "tools"}:
                if not isinstance(value, list):
                    raise AgentConfigError(f"Agent {name!r}: '{key}' must be a list.")
                if key == "modalities" and any(not isinstance(item, str) for item in value):
                    raise AgentConfigError(
                        f"Agent {name!r}: 'modalities' entries must be strings."
                    )
                if key in {"functions", "tools"} and any(
                    not isinstance(item, dict) for item in value
                ):
                    raise AgentConfigError(
                        f"Agent {name!r}: '{key}' entries must be mappings."
                    )
            elif key == "logit_bias":
                if not isinstance(value, dict) or any(
                    not isinstance(token, str)
                    or isinstance(bias, bool)
                    or not isinstance(bias, (int, float))
                    for token, bias in value.items()
                ):
                    raise AgentConfigError(
                        f"Agent {name!r}: 'logit_bias' must map token IDs to numeric values."
                    )
            elif key == "metadata":
                if not isinstance(value, dict) or any(
                    not isinstance(meta_key, str) or not isinstance(meta_value, str)
                    for meta_key, meta_value in value.items()
                ):
                    raise AgentConfigError(
                        f"Agent {name!r}: 'metadata' must map strings to strings."
                    )
            elif key == "reasoning":
                if not isinstance(value, dict):
                    raise AgentConfigError(
                        f"Agent {name!r}: 'reasoning' must be a mapping."
                    )
                allowed_reasoning_fields = {"effort", "max_tokens", "enabled", "exclude"}
                unknown_reasoning_fields = value.keys() - allowed_reasoning_fields
                if unknown_reasoning_fields:
                    raise AgentConfigError(
                        f"Agent {name!r}: unsupported reasoning option(s): "
                        f"{', '.join(sorted(map(str, unknown_reasoning_fields)))}."
                    )
                if "effort" in value and not isinstance(value["effort"], str):
                    raise AgentConfigError(
                        f"Agent {name!r}: 'reasoning.effort' must be a string."
                    )
                if "max_tokens" in value and (
                    isinstance(value["max_tokens"], bool)
                    or not isinstance(value["max_tokens"], int)
                    or value["max_tokens"] <= 0
                ):
                    raise AgentConfigError(
                        f"Agent {name!r}: 'reasoning.max_tokens' must be a positive integer."
                    )
                if "effort" in value and "max_tokens" in value:
                    raise AgentConfigError(
                        f"Agent {name!r}: set either 'reasoning.effort' or "
                        "'reasoning.max_tokens', not both."
                    )
                if any(
                    field in value and not isinstance(value[field], bool)
                    for field in ("enabled", "exclude")
                ):
                    raise AgentConfigError(
                        f"Agent {name!r}: 'reasoning.enabled' and 'reasoning.exclude' "
                        "must be booleans."
                    )
            elif key in {
                "audio",
                "moderation",
                "prediction",
                "prompt_cache_options",
                "response_format",
                "stream_options",
                "web_search_options",
            } and not isinstance(value, dict):
                raise AgentConfigError(f"Agent {name!r}: '{key}' must be a mapping.")

        if "max_tokens" in params and "max_completion_tokens" in params:
            raise AgentConfigError(
                f"Agent {name!r}: set only one of 'max_tokens' or 'max_completion_tokens'."
            )
