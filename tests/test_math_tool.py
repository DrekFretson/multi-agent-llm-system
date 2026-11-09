"""Real local calculation tests for the safe MathTool grammar and operations."""

import math

import numpy as np
import pytest

from src.tools import (
    MathExpressionError,
    MathInputError,
    MathTool,
    MathToolError,
    OptimizationError,
)


@pytest.fixture
def tool() -> MathTool:
    return MathTool()


def test_simplify_expression(tool: MathTool) -> None:
    result = tool.simplify("x + x + 2 - 2")

    assert result.result == "2*x"
    assert result.verified


def test_solve_linear_and_nonlinear_equations(tool: MathTool) -> None:
    linear = tool.solve_equation("2*x + 5 = 15", "x")
    nonlinear = tool.solve_equation("x**2 - 4 = 0", "x")

    assert linear.result["solutions"] == ["5"]
    assert linear.verified
    assert nonlinear.result["solutions"] == ["-2", "2"]
    assert nonlinear.verified
    assert nonlinear.diagnostics["residuals"] == ["0", "0"]


def test_solve_system_and_verify_all_equations(tool: MathTool) -> None:
    result = tool.solve_system(["x + y = 3", "x - y = 1"], ["x", "y"])

    assert result.result["solutions"] == [{"x": "2", "y": "1"}]
    assert result.verified
    assert result.diagnostics["residuals"] == [["0", "0"]]


@pytest.mark.parametrize(
    ("order", "expected"),
    [(1, "3*x**2 - 6*x + 2"), (2, "6*(x - 1)")],
)
def test_derivative_and_second_derivative_are_numerically_checked(
    tool: MathTool,
    order: int,
    expected: str,
) -> None:
    result = tool.derivative("x**3 - 3*x**2 + 2*x", "x", order)

    assert result.result["derivative"] == expected
    assert result.verified
    assert result.diagnostics["finite_difference_errors"]


def test_indefinite_and_definite_integrals_are_verified(tool: MathTool) -> None:
    indefinite = tool.integrate("x**2", "x")
    definite = tool.integrate("x**2", "x", 0, 2)

    assert indefinite.result == "x**3/3"
    assert indefinite.verified
    assert definite.result == "2.66666666666667"
    assert definite.verified
    assert definite.diagnostics["absolute_error"] < 1e-10


def test_numeric_expression_and_array_evaluation_use_finite_values(tool: MathTool) -> None:
    scalar = tool.evaluate_expression("x**2 + 2*y", {"x": 3, "y": 4})
    array = tool.evaluate_array("x**2 + 1", "x", np.array([0, 1, 2]))

    assert scalar.result == 17
    assert scalar.verified
    assert array.result == [1.0, 2.0, 5.0]
    assert array.verified
    assert array.diagnostics["input_count"] == 3


def test_stationary_points_are_checked(tool: MathTool) -> None:
    result = tool.stationary_points("x**2 - 6*x + 11", "x")

    assert result.result["points"] == ["3"]
    assert result.verified


def test_minimization_of_single_variable_returns_diagnostics(tool: MathTool) -> None:
    result = tool.optimize("x**2 - 6*x + 11", ["x"], [0.0])

    assert result.operation == "minimize"
    assert result.verified
    assert result.result["optimizer_success"]
    assert result.result["variables"]["x"] == pytest.approx(3.0, abs=1e-5)
    assert result.result["objective_value"] == pytest.approx(2.0, abs=1e-8)
    assert result.result["iterations"] is not None
    assert result.result["global_optimum_guaranteed"] is False
    assert result.diagnostics["objective_recomputed"] is True
    assert result.diagnostics["gradient_norm"] < 1e-5
    assert result.diagnostics["gradient_check_applicable"] is True
    assert result.diagnostics["gradient_check_passed"] is True


def test_minimization_of_multiple_variables(tool: MathTool) -> None:
    result = tool.optimize("(x - 2)**2 + (y + 3)**2", ["x", "y"], [0.0, 0.0])

    assert result.verified
    assert result.result["variables"]["x"] == pytest.approx(2.0, abs=1e-5)
    assert result.result["variables"]["y"] == pytest.approx(-3.0, abs=1e-5)
    assert result.result["objective_value"] == pytest.approx(0.0, abs=1e-8)


def test_maximization_minimizes_negative_objective(tool: MathTool) -> None:
    result = tool.optimize("-(x - 2)**2 + 5", ["x"], [0.0], maximize=True)

    assert result.operation == "maximize"
    assert result.verified
    assert result.result["variables"]["x"] == pytest.approx(2.0, abs=1e-5)
    assert result.result["objective_value"] == pytest.approx(5.0, abs=1e-8)


def test_optimization_checks_bounds_and_equality_constraints(tool: MathTool) -> None:
    result = tool.optimize(
        "x**2 + y**2",
        ["x", "y"],
        [0.0, 2.0],
        bounds=[[0.0, None], [0.0, None]],
        constraints=[{"type": "eq", "expression": "x + y - 2"}],
    )

    assert result.verified
    assert result.result["variables"]["x"] == pytest.approx(1.0, abs=1e-5)
    assert result.result["variables"]["y"] == pytest.approx(1.0, abs=1e-5)
    assert result.result["objective_value"] == pytest.approx(2.0, abs=1e-8)
    assert result.result["bounds_satisfied"]
    assert result.result["constraints_satisfied"]
    assert result.result["constraint_values"] == pytest.approx([0.0], abs=1e-8)


@pytest.mark.parametrize(
    "expression",
    ["x +", "__import__('os').system('id')", "x.__class__", "[x for x in ()]", "x**1001"],
)
def test_unsupported_or_arbitrary_python_syntax_is_rejected(
    tool: MathTool,
    expression: str,
) -> None:
    with pytest.raises(MathToolError):
        tool.simplify(expression)


def test_division_by_zero_and_non_finite_values_are_rejected(tool: MathTool) -> None:
    with pytest.raises(MathExpressionError):
        tool.simplify("1 / 0")
    with pytest.raises(MathToolError, match="divided by zero|non-finite|non-real"):
        tool.evaluate_expression("1 / x", {"x": 0})


def test_invalid_optimizer_parameters_are_rejected(tool: MathTool) -> None:
    with pytest.raises(OptimizationError, match="initial_guess"):
        tool.optimize("x**2", ["x"], [])
    with pytest.raises(OptimizationError, match="bound"):
        tool.optimize("x**2", ["x"], [0], bounds=[[2, 1]])
    with pytest.raises(OptimizationError, match="Unsupported"):
        tool.optimize("x**2", ["x"], [0], method="arbitrary")
    with pytest.raises(OptimizationError, match="does not support"):
        tool.optimize(
            "x**2", ["x"], [0], constraints=[{"type": "eq", "expression": "x"}], method="BFGS"
        )


def test_array_size_and_numeric_inputs_are_validated(tool: MathTool) -> None:
    with pytest.raises(MathInputError, match="one-dimensional"):
        tool.evaluate_array("x", "x", [[1, 2]])
    with pytest.raises(MathInputError, match="finite"):
        tool.evaluate_array("x", "x", [math.inf])
    with pytest.raises(MathInputError, match="undeclared variable"):
        tool.evaluate_expression("x + y", {"x": 1})
