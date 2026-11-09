"""Mocked LLM orchestration tests for MathAgent."""

import json
from unittest.mock import Mock

import pytest

from src.agents import (
    AgentConfigRegistry,
    AgentInputError,
    MathAgent,
    MathAgentFormulationError,
    MathAgentLLMError,
    MathAgentReportError,
    MathAgentToolError,
    MathReport,
)
from src.llm import LLMAPIError, LLMClient, LLMResponse
from src.tools import MathInputError, MathTool, MathToolResult


@pytest.fixture
def client() -> Mock:
    return Mock(spec=LLMClient)


@pytest.fixture
def math_tool() -> Mock:
    return Mock(spec=MathTool)


@pytest.fixture
def agent(client: Mock, math_tool: Mock) -> MathAgent:
    return MathAgent(
        client=client,
        math_tool=math_tool,
        config=AgentConfigRegistry().get_agent("math"),
        agent_id="math-test",
    )


def response(content: str) -> LLMResponse:
    return LLMResponse(
        content=content,
        model="provider/reasoning-model",
        prompt_tokens=40,
        completion_tokens=60,
        total_tokens=100,
        finish_reason="stop",
    )


def plan(
    operations: list[dict[str, object]],
    *,
    clarification_required: bool = False,
    question: str | None = None,
) -> dict[str, object]:
    return {
        "mathematical_formulation": "2*x + 5 = 15",
        "method": "Алгебраическое решение уравнения",
        "assumptions": [],
        "clarification_required": clarification_required,
        "clarification_question": question,
        "operations": operations,
    }


def tool_result(
    operation: str = "solve_equation",
    *,
    value: object = None,
    verified: bool = True,
) -> MathToolResult:
    return MathToolResult(
        operation=operation,
        expression="2*x + 5 = 15",
        result=value if value is not None else {"variable": "x", "solutions": ["5"]},
        verified=verified,
        verification="Substitution residual is zero." if verified else "Check failed.",
        diagnostics={"residuals": ["0"]},
    )


def test_math_agent_uses_math_role_and_specialized_prompts(agent: MathAgent) -> None:
    assert agent.config.name == "math"
    assert agent.config.logical_model_id == "reasoning"
    assert agent.name == "Math Agent"
    assert "Python-инструмента" in agent.system_prompt


def test_solve_formalizes_executes_tool_and_returns_structured_report(
    client: Mock,
    math_tool: Mock,
    agent: MathAgent,
) -> None:
    valid_plan = plan([{"operation": "solve_equation", "equation": "2*x+5=15", "variable": "x"}])
    client.generate.side_effect = [
        response(json.dumps(valid_plan, ensure_ascii=False)),
        response(json.dumps({"explanation": "Вычтем 5 и разделим обе части на 2."}, ensure_ascii=False)),
    ]
    result = tool_result()
    math_tool.solve_equation.return_value = result

    report = agent.solve("Реши уравнение 2x + 5 = 15")

    assert isinstance(report, MathReport)
    assert report.problem == "Реши уравнение 2x + 5 = 15"
    assert report.mathematical_formulation == "2*x + 5 = 15"
    assert report.tool_results == [result]
    assert report.verified
    assert '"solutions": ["5"]' in report.result
    assert "Проверка пройдена" in report.verification
    assert report.explanation == "Вычтем 5 и разделим обе части на 2."
    math_tool.solve_equation.assert_called_once_with(equation="2*x+5=15", variable="x")
    assert client.generate.call_count == 2

    explanation_call = client.generate.call_args_list[1].kwargs
    assert explanation_call["agent_id"] == "math-test"
    assert explanation_call["model"] == agent.config.model_id
    assert '"verified": true' in explanation_call["messages"][1]["content"]


def test_math_agent_runs_real_local_calculation_with_mocked_llm(client: Mock) -> None:
    client.generate.side_effect = [
        response(
            json.dumps(
                plan(
                    [
                        {
                            "operation": "minimize",
                            "objective": "x**2 - 6*x + 11",
                            "variables": ["x"],
                            "initial_guess": [0],
                        }
                    ]
                )
            )
        ),
        response(json.dumps({"explanation": "SciPy нашёл стационарную точку."})),
    ]
    agent = MathAgent(client=client, config=AgentConfigRegistry().get_agent("math"))

    report = agent.solve("Найди минимум x**2 - 6*x + 11")

    assert report.verified
    assert report.tool_results[0].result["variables"]["x"] == pytest.approx(3.0, abs=1e-5)
    assert report.tool_results[0].result["objective_value"] == pytest.approx(2.0, abs=1e-8)
    assert report.tool_results[0].diagnostics["gradient_check_passed"] is True


def test_derivative_plan_runs_each_requested_operation(
    client: Mock,
    math_tool: Mock,
    agent: MathAgent,
) -> None:
    client.generate.side_effect = [
        response(json.dumps(plan([
            {"operation": "derivative", "expression": "x**3-3*x**2+2*x", "variable": "x", "order": 1},
            {"operation": "derivative", "expression": "x**3-3*x**2+2*x", "variable": "x", "order": 2},
        ]))),
        response(json.dumps({"explanation": "Первая и вторая производные найдены."})),
    ]
    math_tool.derivative.side_effect = [
        tool_result("derivative", value={"derivative": "3*x**2 - 6*x + 2"}),
        tool_result("derivative", value={"derivative": "6*x - 6"}),
    ]

    report = agent.solve("Найди первую и вторую производные.")

    assert len(report.tool_results) == 2
    assert report.verified
    assert math_tool.derivative.call_count == 2
    assert math_tool.derivative.call_args_list[0].kwargs["order"] == 1
    assert math_tool.derivative.call_args_list[1].kwargs["order"] == 2


def test_optimization_result_is_forwarded_and_not_called_global(
    client: Mock,
    math_tool: Mock,
    agent: MathAgent,
) -> None:
    optimization_result = tool_result(
        "minimize",
        value={
            "variables": {"x": 3.0},
            "objective_value": 2.0,
            "optimizer_success": True,
            "global_optimum_guaranteed": False,
        },
    )
    client.generate.side_effect = [
        response(json.dumps(plan([
            {
                "operation": "minimize",
                "objective": "x**2-6*x+11",
                "variables": ["x"],
                "initial_guess": [0],
            }
        ]))),
        response(json.dumps({"explanation": "Численный локальный поиск сошёлся."})),
    ]
    math_tool.optimize.return_value = optimization_result

    report = agent.solve("Найди минимум квадратичной функции.")

    assert report.verified
    assert report.tool_results[0].result["global_optimum_guaranteed"] is False
    math_tool.optimize.assert_called_once_with(
        objective="x**2-6*x+11",
        variables=["x"],
        initial_guess=[0],
        maximize=False,
    )
    assert "global_optimum_guaranteed" in client.generate.call_args_list[1].kwargs["messages"][1]["content"]


def test_unverified_tool_result_stays_unverified_in_report(
    client: Mock,
    math_tool: Mock,
    agent: MathAgent,
) -> None:
    client.generate.side_effect = [
        response(json.dumps(plan([{"operation": "simplify", "expression": "x+x"}]))),
        response(json.dumps({"explanation": "Инструмент не смог проверить результат."})),
    ]
    math_tool.simplify.return_value = tool_result("simplify", value="2*x", verified=False)

    report = agent.solve("Упрости x+x")

    assert not report.verified
    assert "Проверка не пройдена" in report.verification


def test_invalid_formalization_json_and_schema_are_rejected(
    client: Mock,
    agent: MathAgent,
) -> None:
    client.generate.return_value = response("not json")
    with pytest.raises(MathAgentFormulationError, match="not valid JSON"):
        agent.solve("Question")

    client.generate.return_value = response(json.dumps({"operations": []}))
    with pytest.raises(MathAgentFormulationError, match="missing or unexpected"):
        agent.solve("Question")


def test_unsupported_operation_is_rejected_before_math_tool_call(
    client: Mock,
    math_tool: Mock,
    agent: MathAgent,
) -> None:
    client.generate.return_value = response(
        json.dumps(plan([{"operation": "run_python", "code": "print(1)"}]))
    )

    with pytest.raises(MathAgentFormulationError, match="Unsupported MathTool operation"):
        agent.solve("Do math")
    math_tool.simplify.assert_not_called()
    math_tool.optimize.assert_not_called()


def test_math_tool_errors_are_wrapped(
    client: Mock,
    math_tool: Mock,
    agent: MathAgent,
) -> None:
    client.generate.return_value = response(
        json.dumps(plan([{"operation": "solve_equation", "equation": "x=1", "variable": "x"}]))
    )
    math_tool.solve_equation.side_effect = MathInputError("invalid equation")

    with pytest.raises(MathAgentToolError, match="invalid equation"):
        agent.solve("Solve x=1")


def test_llm_errors_are_reported_for_planning_and_explanation(
    client: Mock,
    math_tool: Mock,
    agent: MathAgent,
) -> None:
    client.generate.side_effect = LLMAPIError("provider unavailable")
    with pytest.raises(MathAgentLLMError, match="formalize"):
        agent.solve("Question")

    client.generate.side_effect = [
        response(json.dumps(plan([{"operation": "simplify", "expression": "x+x"}]))),
        LLMAPIError("provider unavailable"),
    ]
    math_tool.simplify.return_value = tool_result("simplify", value="2*x")
    with pytest.raises(MathAgentLLMError, match="explain"):
        agent.solve("Simplify x+x")


def test_ambiguous_problem_returns_clarification_without_tool_execution(
    client: Mock,
    math_tool: Mock,
    agent: MathAgent,
) -> None:
    client.generate.return_value = response(
        json.dumps(
            plan(
                [],
                clarification_required=True,
                question="Какая функция и на каком отрезке рассматривается?",
            ),
            ensure_ascii=False,
        )
    )

    report = agent.solve("Найди минимум")

    assert report.needs_clarification
    assert not report.verified
    assert "отрезке" in report.clarification_question
    assert "не вычислялось" in report.result
    math_tool.simplify.assert_not_called()
    math_tool.optimize.assert_not_called()
    client.generate.assert_called_once()


def test_malformed_explanation_response_is_rejected(client: Mock, agent: MathAgent) -> None:
    with pytest.raises(MathAgentReportError, match="not valid JSON"):
        agent._parse_explanation("not json")


def test_invalid_task_is_rejected_before_llm_call(client: Mock, agent: MathAgent) -> None:
    with pytest.raises(AgentInputError, match="non-empty string"):
        agent.solve("   ")
    client.generate.assert_not_called()
