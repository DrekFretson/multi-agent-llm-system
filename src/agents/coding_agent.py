"""Coding agent and parser for its structured solution response."""

from __future__ import annotations

import json
from dataclasses import dataclass

from src.agents.base_agent import AgentInputError, AgentResult, BaseAgent
from src.agents.config import AgentConfig, AgentConfigRegistry
from src.llm import LLMClient


CODING_SYSTEM_PROMPT = """\
Ты Coding Agent. Выполняй техническое задание и возвращай только один корректный
JSON-объект без Markdown-ограждений, вступления или текста после JSON.

Требования к решению:
- Пиши корректный, читаемый код и соблюдай требования задания.
- Если язык явно не задан, выбери язык, подходящий задаче, и укажи его в `language`.
- Учитывай временную и пространственную сложность, объясняя важные решения в `summary`.
- Обрабатывай граничные случаи и добавляй подходящие автоматические тесты.
- Перед ответом сверь каждый ожидаемый результат теста с реализацией и условием задачи;
  особенно проверь преобразования входа (например, удаление пробелов и знаков препинания).
- Не добавляй тесты с противоречивыми или неподтверждёнными ожиданиями; для каждого
  теста укажи вход, который действительно приводит к заявленному результату.
- Перечисляй только необходимые внешние зависимости в `dependencies`.
- Дай конкретные инструкции по запуску кода и тестов в `run_instructions`.
- Не утверждай и не подразумевай, что тесты запускались, если они фактически не запускались.
- Если предоставлен существующий код, предложи целостную доработанную версию.

Формат ответа — ровно следующие поля JSON:
{
  "language": "язык программирования",
  "summary": "краткое описание решения и принятых решений",
  "code": "полный исходный код",
  "tests": "исходный код тестов",
  "dependencies": ["имя зависимости"],
  "run_instructions": "как запустить код и тесты"
}

JSON должен быть синтаксически корректным: экранируй кавычки и переносы строк
внутри строк согласно JSON.
"""


class CodingSolutionError(ValueError):
    """Base exception for invalid structured coding solutions."""


class CodingSolutionParseError(CodingSolutionError):
    """Raised when a model response is not valid JSON."""


class CodingSolutionValidationError(CodingSolutionError):
    """Raised when decoded JSON does not match the solution schema."""


@dataclass(frozen=True)
class CodingSolution:
    """Structured code-generation result parsed from a successful agent result."""

    language: str
    summary: str
    code: str
    tests: str
    dependencies: list[str]
    run_instructions: str


class CodingAgent(BaseAgent):
    """Generate code and tests using the shared BaseAgent execution path."""

    CONFIG_ID = "coding"

    def __init__(
        self,
        client: LLMClient,
        config: AgentConfig | None = None,
        *,
        config_registry: AgentConfigRegistry | None = None,
        agent_id: str = "coding",
        name: str = "Coding Agent",
        system_prompt: str = CODING_SYSTEM_PROMPT,
    ) -> None:
        """Create a coding agent with an injected client and YAML role config.

        If ``config`` is omitted, the role is loaded from
        ``AgentConfigRegistry`` using the logical ID ``coding``.
        """
        if config is None:
            registry = config_registry or AgentConfigRegistry()
            config = registry.get_agent(self.CONFIG_ID)
        elif config.logical_model_id != self.CONFIG_ID:
            raise ValueError(
                f"CodingAgent requires the logical model config {self.CONFIG_ID!r}."
            )
        super().__init__(
            agent_id=agent_id,
            name=name,
            system_prompt=system_prompt,
            client=client,
            config=config,
        )

    def run(
        self,
        task: str,
        *,
        additional_context: str | None = None,
        existing_code: str | None = None,
        test_requirements: str | None = None,
    ) -> AgentResult:
        """Generate a solution, optionally including context and code to edit.

        The return type remains :class:`AgentResult`; call
        :meth:`parse_solution` to validate and unpack its JSON content.
        """
        for field_name, value in (
            ("additional_context", additional_context),
            ("existing_code", existing_code),
            ("test_requirements", test_requirements),
        ):
            if value is not None and not isinstance(value, str):
                raise AgentInputError(f"{field_name} must be a string or None.")

        request_parts = [f"Техническое задание:\n{task}"]
        if additional_context:
            request_parts.append(f"Дополнительный контекст:\n{additional_context}")
        if existing_code:
            request_parts.append(
                "Существующий код для доработки:\n"
                "```\n"
                f"{existing_code}\n"
                "```"
            )
        if test_requirements:
            request_parts.append(f"Требования к тестам:\n{test_requirements}")

        return super().run("\n\n".join(request_parts))

    def parse_solution(self, result: AgentResult) -> CodingSolution:
        """Parse and validate the JSON solution from an :class:`AgentResult`."""
        if not isinstance(result, AgentResult):
            raise TypeError("result must be an AgentResult instance.")
        if not result.success:
            raise CodingSolutionValidationError(
                f"Cannot parse an unsuccessful agent result: {result.error or 'unknown error'}."
            )

        try:
            payload = json.loads(result.content)
        except json.JSONDecodeError as error:
            raise CodingSolutionParseError(
                f"Coding Agent returned invalid JSON at line {error.lineno}, "
                f"column {error.colno}."
            ) from error

        if not isinstance(payload, dict):
            raise CodingSolutionValidationError("Solution JSON must be an object.")

        required_fields = {
            "language",
            "summary",
            "code",
            "tests",
            "dependencies",
            "run_instructions",
        }
        missing = required_fields - payload.keys()
        extra = payload.keys() - required_fields
        if missing:
            raise CodingSolutionValidationError(
                f"Solution JSON is missing required field(s): {', '.join(sorted(missing))}."
            )
        if extra:
            raise CodingSolutionValidationError(
                f"Solution JSON contains unknown field(s): {', '.join(sorted(extra))}."
            )

        for field_name in ("language", "summary", "code", "tests", "run_instructions"):
            if not isinstance(payload[field_name], str):
                raise CodingSolutionValidationError(
                    f"Solution field {field_name!r} must be a string."
                )
        if not payload["language"].strip():
            raise CodingSolutionValidationError("Solution field 'language' cannot be empty.")
        if not payload["summary"].strip():
            raise CodingSolutionValidationError("Solution field 'summary' cannot be empty.")
        if not payload["code"].strip():
            raise CodingSolutionValidationError("Solution field 'code' cannot be empty.")
        if not isinstance(payload["dependencies"], list) or any(
            not isinstance(dependency, str) or not dependency.strip()
            for dependency in payload["dependencies"]
        ):
            raise CodingSolutionValidationError(
                "Solution field 'dependencies' must be a list of non-empty strings."
            )

        return CodingSolution(
            language=payload["language"],
            summary=payload["summary"],
            code=payload["code"],
            tests=payload["tests"],
            dependencies=payload["dependencies"],
            run_instructions=payload["run_instructions"],
        )
