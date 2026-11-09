"""Math agent that formalizes problems and delegates calculations locally."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from src.agents.base_agent import AgentInputError, BaseAgent
from src.agents.config import AgentConfig, AgentConfigRegistry
from src.llm import LLMClient
from src.tools import MathTool, MathToolError, MathToolResult


MATH_FORMALIZATION_PROMPT = """\
Ты математический формализатор. Разбери задачу, не вычисляй ответ самостоятельно.
Определи тип задачи, переменные, исходные данные, ограничения, метод и
последовательность операций для предоставленного Python MathTool.

Правила:
- Если пользователь не указал язык, все текстовые пояснения пиши по-русски.
- Не придумывай данные и явно отделяй предположения.
- Для численных методов задай подходящее начальное приближение и при наличии
  укажи bounds/ограничения. Для ограничения x + y = c передай равенство как
  expression "x + y - c" типа "eq"; для x >= 0 передай bounds [0, null].
- Не называй численное решение глобальным оптимумом: SciPy сообщает результат
  локального численного метода и он может зависеть от начального приближения.
- Если условие неоднозначно или недоопределено, установи
  clarification_required=true, задай уточняющий вопрос и не планируй операций.
- Не генерируй Python-код, имена модулей, атрибуты или вызовы функций вне
  поддерживаемых MathTool операций.

Верни только корректный JSON, ровно такой структуры:
{
  "mathematical_formulation": "явная математическая постановка",
  "method": "выбранный математический метод",
  "assumptions": ["предположение или пустой список"],
  "clarification_required": false,
  "clarification_question": null,
  "operations": [
    {"operation": "solve_equation", "equation": "2*x + 5 = 15", "variable": "x"}
  ]
}

Поддерживаемые operations:
- simplify: expression
- solve_equation: equation, variable
- solve_system: equations (строки уравнений), variables
- evaluate: expression, values (словарь переменных и чисел)
- derivative: expression, variable, order (1 или 2)
- integrate: expression, variable; optional lower_bound и upper_bound вместе
- stationary_points: expression, variable
- evaluate_array: expression, variable, values (числовой список)
- minimize / maximize: objective, variables, initial_guess; optional bounds,
  constraints, method, tolerance, max_iterations. Constraint имеет ровно поля
  type (eq или ineq) и expression; ineq означает expression >= 0.

Не добавляй неизвестные поля. Каждая операция должна быть достаточной для
безопасного выполнения без дополнительного кода или догадок.
"""

MATH_EXPLANATION_PROMPT = """\
Ты объясняешь решение математической задачи по результатам Python-инструмента.
Пиши по-русски, если пользователь не запросил другой язык. Изложи решение
последовательно, поясни формулы и методы.

Результаты и их verification из MathTool являются единственными данными о
вычислениях. Не пересчитывай их мысленно, не меняй числовые значения и не
утверждай, что проверка прошла, если verified=false. Если SciPy использован для
оптимизации, укажи, что результат численный и зависит от начального приближения;
не называй его глобальным минимумом/максимумом, так как инструмент этого не
гарантирует. Отдельно упомяни неоднозначности и предположения.

Верни только JSON вида {"explanation": "пошаговое объяснение"}; без Markdown и
текста вне JSON.
"""


class MathAgentError(Exception):
    """Base error for mathematical task planning and reporting."""


class MathAgentLLMError(MathAgentError):
    """Raised when LLM formalization or explanation fails."""


class MathAgentFormulationError(MathAgentError):
    """Raised when the LLM returns a malformed or unsupported problem plan."""


class MathAgentToolError(MathAgentError):
    """Raised when MathTool cannot execute a validated operation."""


class MathAgentReportError(MathAgentError):
    """Raised when the final explanation response does not match its schema."""


@dataclass(frozen=True)
class MathReport:
    """Structured report separating tool computations from prose explanation."""

    problem: str
    mathematical_formulation: str
    method: str
    calculations: list[str]
    result: str
    verification: str
    explanation: str
    tool_results: list[MathToolResult] = field(default_factory=list)
    verified: bool = False
    assumptions: list[str] = field(default_factory=list)
    needs_clarification: bool = False
    clarification_question: str | None = None


class MathAgent(BaseAgent):
    """Formalize a math problem, use MathTool, and explain checked results."""

    CONFIG_ID = "math"
    MAX_OPERATIONS = 6

    def __init__(
        self,
        client: LLMClient,
        math_tool: MathTool | None = None,
        config: AgentConfig | None = None,
        *,
        config_registry: AgentConfigRegistry | None = None,
        agent_id: str = "math",
        name: str = "Math Agent",
        system_prompt: str = MATH_EXPLANATION_PROMPT,
    ) -> None:
        """Create the agent with shared LLM infrastructure and a MathTool."""
        if config is None:
            registry = config_registry or AgentConfigRegistry()
            config = registry.get_agent(self.CONFIG_ID)
        elif config.name != self.CONFIG_ID:
            raise ValueError(f"MathAgent requires the agent config {self.CONFIG_ID!r}.")
        super().__init__(
            agent_id=agent_id,
            name=name,
            system_prompt=system_prompt,
            client=client,
            config=config,
        )
        self.math_tool = math_tool if math_tool is not None else MathTool()

    def solve(self, problem: str) -> MathReport:
        """Formalize and solve one problem using validated MathTool operations."""
        if not isinstance(problem, str) or not problem.strip():
            raise AgentInputError("task must be a non-empty string.")
        problem = problem.strip()
        plan_result = super().run(
            "Формализуй задачу для MathTool, не вычисляя ответ самостоятельно:\n" + problem,
            system_prompt=MATH_FORMALIZATION_PROMPT,
        )
        if not plan_result.success:
            raise MathAgentLLMError(f"Could not formalize the math problem: {plan_result.error}")
        plan = self._parse_plan(plan_result.content)

        if plan["clarification_required"]:
            question = plan["clarification_question"]
            return MathReport(
                problem=problem,
                mathematical_formulation=plan["mathematical_formulation"],
                method=plan["method"],
                calculations=[],
                result="Решение не вычислялось: требуется уточнение условия.",
                verification="Проверка не выполнялась, поскольку MathTool не запускался.",
                explanation=question or "Условие задачи требует уточнения.",
                verified=False,
                assumptions=plan["assumptions"],
                needs_clarification=True,
                clarification_question=question,
            )

        tool_results: list[MathToolResult] = []
        for operation in plan["operations"]:
            try:
                tool_results.append(self._execute_operation(operation))
            except MathToolError as error:
                raise MathAgentToolError(
                    f"MathTool could not execute {operation['operation']!r}: {error}"
                ) from error

        results_payload = [self._result_payload(result) for result in tool_results]
        result_json = json.dumps(results_payload, ensure_ascii=False, allow_nan=False)
        explanation_result = super().run(
            "Математическая задача:\n"
            f"{problem}\n\nФормализация:\n{plan['mathematical_formulation']}\n\n"
            f"Метод:\n{plan['method']}\n\nПредположения:\n"
            f"{json.dumps(plan['assumptions'], ensure_ascii=False)}\n\n"
            "Авторитетные результаты MathTool (поле verified обязательно учитывай):\n"
            f"{result_json}",
            system_prompt=self.system_prompt,
        )
        if not explanation_result.success:
            raise MathAgentLLMError(
                f"Could not explain the computed math results: {explanation_result.error}"
            )
        explanation = self._parse_explanation(explanation_result.content)

        calculations = [
            f"{result.operation}: "
            f"{json.dumps(result.result, ensure_ascii=False, allow_nan=False, sort_keys=True)}"
            for result in tool_results
        ]
        verification = "\n".join(
            f"{result.operation}: {'Проверка пройдена' if result.verified else 'Проверка не пройдена'} — "
            f"{result.verification}"
            for result in tool_results
        )
        return MathReport(
            problem=problem,
            mathematical_formulation=plan["mathematical_formulation"],
            method=plan["method"],
            calculations=calculations,
            result=result_json,
            verification=verification,
            explanation=explanation,
            tool_results=tool_results,
            verified=bool(tool_results) and all(result.verified for result in tool_results),
            assumptions=plan["assumptions"],
        )

    @classmethod
    def _parse_plan(cls, content: str) -> dict[str, Any]:
        try:
            plan = json.loads(content)
        except json.JSONDecodeError as error:
            raise MathAgentFormulationError(
                f"Math formalization is not valid JSON at line {error.lineno}, column {error.colno}."
            ) from error
        fields = {
            "mathematical_formulation",
            "method",
            "assumptions",
            "clarification_required",
            "clarification_question",
            "operations",
        }
        if not isinstance(plan, dict) or plan.keys() != fields:
            raise MathAgentFormulationError("Math formalization has missing or unexpected fields.")
        for name in ("mathematical_formulation", "method"):
            if not isinstance(plan[name], str) or not plan[name].strip():
                raise MathAgentFormulationError(f"Formalization field {name!r} must be a non-empty string.")
        if not isinstance(plan["assumptions"], list) or any(
            not isinstance(item, str) for item in plan["assumptions"]
        ):
            raise MathAgentFormulationError("Formalization assumptions must be a list of strings.")
        if not isinstance(plan["clarification_required"], bool):
            raise MathAgentFormulationError("clarification_required must be a boolean.")
        question = plan["clarification_question"]
        if question is not None and (not isinstance(question, str) or not question.strip()):
            raise MathAgentFormulationError("clarification_question must be a non-empty string or null.")
        operations = plan["operations"]
        if not isinstance(operations, list) or len(operations) > cls.MAX_OPERATIONS:
            raise MathAgentFormulationError(
                f"operations must be a list of at most {cls.MAX_OPERATIONS} items."
            )
        if plan["clarification_required"]:
            if not question or operations:
                raise MathAgentFormulationError(
                    "A clarification plan needs a question and must not contain operations."
                )
        elif not operations:
            raise MathAgentFormulationError("A solvable plan must contain at least one operation.")
        for operation in operations:
            cls._validate_operation(operation)
        return plan

    @staticmethod
    def _validate_operation(operation: Any) -> None:
        if not isinstance(operation, dict) or not isinstance(operation.get("operation"), str):
            raise MathAgentFormulationError("Each MathTool operation must be a JSON object with an operation name.")
        name = operation["operation"]
        schemas: dict[str, tuple[set[str], set[str]]] = {
            "simplify": ({"expression"}, set()),
            "solve_equation": ({"equation", "variable"}, set()),
            "solve_system": ({"equations", "variables"}, set()),
            "evaluate": ({"expression", "values"}, set()),
            "derivative": ({"expression", "variable"}, {"order"}),
            "integrate": ({"expression", "variable"}, {"lower_bound", "upper_bound"}),
            "stationary_points": ({"expression", "variable"}, set()),
            "evaluate_array": ({"expression", "variable", "values"}, set()),
            "minimize": ({"objective", "variables", "initial_guess"},
                         {"bounds", "constraints", "method", "tolerance", "max_iterations"}),
            "maximize": ({"objective", "variables", "initial_guess"},
                         {"bounds", "constraints", "method", "tolerance", "max_iterations"}),
        }
        if name not in schemas:
            raise MathAgentFormulationError(f"Unsupported MathTool operation {name!r}.")
        required, optional = schemas[name]
        supplied = operation.keys() - {"operation"}
        if required - supplied or supplied - required - optional:
            raise MathAgentFormulationError(
                f"Operation {name!r} has missing or unexpected fields."
            )

    def _execute_operation(self, operation: dict[str, Any]) -> MathToolResult:
        name = operation["operation"]
        arguments = {key: value for key, value in operation.items() if key != "operation"}
        if name == "simplify":
            return self.math_tool.simplify(**arguments)
        if name == "solve_equation":
            return self.math_tool.solve_equation(**arguments)
        if name == "solve_system":
            return self.math_tool.solve_system(**arguments)
        if name == "evaluate":
            return self.math_tool.evaluate_expression(**arguments)
        if name == "derivative":
            return self.math_tool.derivative(**arguments)
        if name == "integrate":
            return self.math_tool.integrate(**arguments)
        if name == "stationary_points":
            return self.math_tool.stationary_points(**arguments)
        if name == "evaluate_array":
            return self.math_tool.evaluate_array(**arguments)
        if name in {"minimize", "maximize"}:
            return self.math_tool.optimize(maximize=name == "maximize", **arguments)
        raise MathAgentFormulationError(f"Unsupported MathTool operation {name!r}.")

    @staticmethod
    def _parse_explanation(content: str) -> str:
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as error:
            raise MathAgentReportError(
                f"Math explanation is not valid JSON at line {error.lineno}, column {error.colno}."
            ) from error
        if (
            not isinstance(payload, dict)
            or set(payload) != {"explanation"}
            or not isinstance(payload["explanation"], str)
            or not payload["explanation"].strip()
        ):
            raise MathAgentReportError("Math explanation must contain one non-empty 'explanation' string.")
        return payload["explanation"].strip()

    @staticmethod
    def _result_payload(result: MathToolResult) -> dict[str, Any]:
        return {
            "operation": result.operation,
            "expression": result.expression,
            "result": result.result,
            "verified": result.verified,
            "verification": result.verification,
            "diagnostics": result.diagnostics,
        }
