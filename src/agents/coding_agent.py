"""Coding agent and parser for its structured solution response."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from src.agents.base_agent import AgentInputError, AgentResult, BaseAgent
from src.agents.config import AgentConfig, AgentConfigRegistry
from src.llm import LLMClient


CODING_SYSTEM_PROMPT = """\
Ты Coding Agent. Выполняй техническое задание и возвращай только один корректный
JSON-объект без Markdown-ограждений, вступления или текста после JSON.

Требования к решению:
- Пиши корректный, читаемый код и соблюдай требования задания.
- Подбирай объём ответа под задачу: для небольшого стандартного алгоритма дай
  компактное объяснение, одну реализацию и только необходимые граничные тесты;
  не добавляй альтернативные реализации и избыточные пояснения.
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
- Если без конкретных требований пользователя нельзя выбрать корректное поведение,
  верни `task_completed: false`, `needs_input: true` и один конкретный вопрос;
  не подменяй решение демонстрационной заглушкой.
- Для обычного завершённого решения верни `task_completed: true`,
  `needs_input: false`, `clarification_question: null`.

Формат ответа — ровно следующие поля JSON:
{
  "language": "язык программирования",
  "summary": "краткое описание решения и принятых решений",
  "code": "полный исходный код",
  "tests": "исходный код тестов",
  "dependencies": ["имя зависимости"],
  "run_instructions": "как запустить код и тесты",
  "task_completed": true,
  "needs_input": false,
  "clarification_question": null
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
    task_completed: bool = True
    needs_input: bool = False
    clarification_question: str | None = None


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
        max_format_retries: int = 1,
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
        if isinstance(max_format_retries, bool) or not isinstance(max_format_retries, int) or max_format_retries < 0:
            raise AgentInputError("max_format_retries must be a non-negative integer.")

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

        request = "\n\n".join(request_parts)
        result = super().run(request)
        if not result.success:
            return result

        try:
            self.parse_solution(result)
            if result.finish_reason != "length":
                return result
            raise CodingSolutionValidationError(
                "Coding Agent response was truncated at the output token limit."
            )
        except CodingSolutionError as error:
            # A length-limited completion can leave placeholder fragments in an
            # otherwise useful answer. Give it the single bounded repair below.
            truncated = result.finish_reason == "length"
            if (
                isinstance(error, CodingSolutionValidationError)
                and "obvious placeholder" in str(error)
                and not truncated
            ):
                return result
            if max_format_retries == 0:
                return result
            initial_error = error

        repair_prompt = (
            "Исправь предыдущий ответ: он должен быть валидным JSON точной схемы, "
            "указанной в системных инструкциях. Верни только JSON-объект; не добавляй "
            "Markdown или пояснения. Экранируй все кавычки и переносы строк внутри "
            "JSON-строк. Сохрани решение, код и тесты по смыслу. Не утверждай, что "
            "тесты запускались.\n\n"
            f"Причина отклонения: {initial_error}\n\n"
            f"Предыдущий ответ:\n{result.content}"
        )
        repaired = super().run(request + "\n\n" + repair_prompt)
        if not repaired.success:
            initial_error.repair_attempts = 1
            raise initial_error
        try:
            self.parse_solution(repaired)
        except CodingSolutionError as repair_error:
            repair_error.repair_attempts = 1
            raise

        return AgentResult(
            agent_id=repaired.agent_id,
            content=repaired.content,
            model=repaired.model,
            success=True,
            error=None,
            prompt_tokens=_sum_optional(result.prompt_tokens, repaired.prompt_tokens),
            completion_tokens=_sum_optional(result.completion_tokens, repaired.completion_tokens),
            total_tokens=_sum_optional(result.total_tokens, repaired.total_tokens),
            finish_reason=repaired.finish_reason,
            format_repair_attempts=1,
        )
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
        status_fields = {"task_completed", "needs_input", "clarification_question"}
        extra = payload.keys() - required_fields - status_fields
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
        if not isinstance(payload["dependencies"], list) or any(
            not isinstance(dependency, str) or not dependency.strip()
            for dependency in payload["dependencies"]
        ):
            raise CodingSolutionValidationError(
                "Solution field 'dependencies' must be a list of non-empty strings."
            )

        task_completed = payload.get("task_completed", True)
        needs_input = payload.get("needs_input", False)
        clarification_question = payload.get("clarification_question")
        if not isinstance(task_completed, bool) or not isinstance(needs_input, bool):
            raise CodingSolutionValidationError("task_completed and needs_input must be booleans.")
        if clarification_question is not None and not isinstance(clarification_question, str):
            raise CodingSolutionValidationError("clarification_question must be a string or null.")
        if needs_input and (task_completed or not (clarification_question or "").strip()):
            raise CodingSolutionValidationError(
                "A needs_input solution must be incomplete and include a clarification question."
            )
        if not needs_input and clarification_question is not None:
            raise CodingSolutionValidationError(
                "clarification_question must be null unless needs_input is true."
            )
        if task_completed and not payload["code"].strip():
            raise CodingSolutionValidationError("A completed solution must include non-empty code.")
        if task_completed and _is_obvious_stub(payload["code"]):
            raise CodingSolutionValidationError(
                "Coding solution contains an obvious placeholder instead of an implementation."
            )

        return CodingSolution(
            language=payload["language"],
            summary=payload["summary"],
            code=payload["code"],
            tests=payload["tests"],
            dependencies=payload["dependencies"],
            run_instructions=payload["run_instructions"],
            task_completed=task_completed,
            needs_input=needs_input,
            clarification_question=clarification_question,
        )


def _is_obvious_stub(code: str) -> bool:
    """Reject unmistakable generated placeholders without executing code."""
    normalized = code.strip()
    lowered = normalized.casefold()
    return (
        normalized in {"...", "pass", "raise NotImplementedError"}
        or bool(re.fullmatch(r"(?:async\s+)?def\s+\w+\s*\([^)]*\)\s*:\s*(?:pass|\.\.\.)", normalized))
        or bool(re.search(r"(?im)^\s*(?:#|//)\s*TODO\b", code))
        or "notimplementederror" in lowered
        or normalized == "# implementation omitted"
    )


def _sum_optional(first: int | None, second: int | None) -> int | None:
    """Sum token counts when both calls provide them, retaining partial data."""
    if first is None and second is None:
        return None
    return (first or 0) + (second or 0)
