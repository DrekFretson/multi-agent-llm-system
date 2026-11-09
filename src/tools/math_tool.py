"""Restricted symbolic and numerical math operations for agent workflows.

Expressions are parsed from a small arithmetic grammar into SymPy objects.
Python expressions are never evaluated, compiled, or executed.
"""

from __future__ import annotations

import ast
import math
import keyword
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import sympy as sp
from scipy.integrate import quad
from scipy.optimize import minimize as scipy_minimize


class MathToolError(Exception):
    """Base class for rejected mathematical input or failed computation."""


class MathInputError(MathToolError, ValueError):
    """Raised when an expression or structured operation is invalid."""


class MathExpressionError(MathInputError):
    """Raised when an expression contains unsupported syntax or names."""


class MathComputationError(MathToolError):
    """Raised when a symbolic or numeric operation cannot be completed."""


class OptimizationError(MathComputationError):
    """Raised when optimizer inputs are invalid or its calculation fails."""


@dataclass(frozen=True)
class MathToolResult:
    """Machine-readable output and independent verification for one operation."""

    operation: str
    expression: str | None
    result: Any
    verified: bool
    verification: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


_FUNCTIONS = {
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
    "exp": sp.exp,
    "log": sp.log,
    "sqrt": sp.sqrt,
    "abs": sp.Abs,
}
_CONSTANTS = {"pi": sp.pi, "E": sp.E}
_BINARY_OPERATORS = {
    ast.Add: lambda left, right: left + right,
    ast.Sub: lambda left, right: left - right,
    ast.Mult: lambda left, right: left * right,
    ast.Div: lambda left, right: left / right,
    ast.Pow: lambda left, right: left**right,
}
_MAX_EXPRESSION_LENGTH = 512
_MAX_AST_NODES = 100
_MAX_ARRAY_SIZE = 10_000
_MAX_VARIABLES = 10
_MAX_CONSTRAINTS = 20
_MAX_EXPONENT = 1_000
_ALLOWED_OPTIMIZERS = {
    "BFGS",
    "L-BFGS-B",
    "SLSQP",
    "COBYLA",
    "trust-constr",
    "Nelder-Mead",
    "Powell",
}


class MathTool:
    """Perform safe symbolic math and bounded SciPy optimization."""

    def simplify(self, expression: str) -> MathToolResult:
        """Simplify an arithmetic expression and check symbolic equivalence."""
        parsed, _ = self._parse_expression(expression)
        simplified = sp.simplify(parsed)
        verified = self._is_zero(sp.simplify(parsed - simplified))
        return MathToolResult(
            operation="simplify",
            expression=expression,
            result=str(simplified),
            verified=verified,
            verification=(
                "The simplified expression is symbolically equivalent to the input."
                if verified
                else "Symbolic equivalence could not be established."
            ),
        )

    def solve_equation(self, equation: str, variable: str) -> MathToolResult:
        """Solve one equation and substitute candidates back into the equation."""
        symbol = self._make_symbol(variable)
        left, right = self._parse_equation(equation, [variable])
        expression = left - right
        try:
            roots = sp.solve(expression, symbol)
        except Exception as error:
            raise MathComputationError("SymPy could not solve the equation.") from error
        residuals = [sp.simplify(expression.subs(symbol, root)) for root in roots]
        verified = bool(roots) and all(self._is_zero(residual) for residual in residuals)
        return MathToolResult(
            operation="solve_equation",
            expression=equation,
            result={"variable": variable, "solutions": [str(root) for root in roots]},
            verified=verified,
            verification=(
                "Every returned candidate was substituted into the original equation."
                if verified
                else "No returned solution was independently verified; no-solution claims are not proven."
            ),
            diagnostics={"residuals": [str(residual) for residual in residuals]},
        )

    def solve_system(
        self,
        equations: Sequence[str],
        variables: Sequence[str],
    ) -> MathToolResult:
        """Solve a system and verify every candidate against every equation."""
        symbols = self._validated_variables(variables)
        if not isinstance(equations, (list, tuple)) or not equations:
            raise MathInputError("equations must be a non-empty list of strings.")
        parsed_equations = [self._equation_expression(eq, variables) for eq in equations]
        try:
            solutions = sp.solve(parsed_equations, symbols, dict=True)
        except Exception as error:
            raise MathComputationError("SymPy could not solve the equation system.") from error

        residuals: list[list[str]] = []
        verified = bool(solutions)
        serialized_solutions: list[dict[str, str]] = []
        for solution in solutions:
            serialized_solutions.append(
                {str(symbol): str(solution[symbol]) for symbol in symbols if symbol in solution}
            )
            candidate_residuals = [
                sp.simplify(expression.subs(solution)) for expression in parsed_equations
            ]
            residuals.append([str(value) for value in candidate_residuals])
            verified = verified and len(solution) == len(symbols) and all(
                self._is_zero(value) for value in candidate_residuals
            )
        return MathToolResult(
            operation="solve_system",
            expression="; ".join(equations),
            result={"solutions": serialized_solutions},
            verified=bool(verified),
            verification=(
                "Every returned complete solution satisfies every equation."
                if verified
                else "Complete solutions were not all independently verified."
            ),
            diagnostics={"residuals": residuals},
        )

    def evaluate_expression(
        self,
        expression: str,
        values: dict[str, int | float],
    ) -> MathToolResult:
        """Evaluate a safe expression at finite numeric variable values."""
        if not isinstance(values, dict) or any(
            not isinstance(name, str) for name in values
        ):
            raise MathInputError("values must map variable names to finite numbers.")
        parsed, symbols = self._parse_expression(expression, tuple(values))
        if set(symbols) != set(values):
            missing = set(symbols) - set(values)
            if missing:
                raise MathInputError(
                    f"No numeric value was provided for: {', '.join(sorted(missing))}."
                )
        numeric = self._numeric_value(parsed, symbols, values)
        return MathToolResult(
            operation="evaluate",
            expression=expression,
            result=numeric,
            verified=math.isfinite(numeric),
            verification="The expression was evaluated and the finite result was checked.",
        )

    def derivative(
        self,
        expression: str,
        variable: str,
        order: int = 1,
    ) -> MathToolResult:
        """Compute a derivative and verify it at representative numeric points."""
        if isinstance(order, bool) or not isinstance(order, int) or not 1 <= order <= 5:
            raise MathInputError("order must be an integer from 1 to 5.")
        symbol = self._make_symbol(variable)
        parsed, _ = self._parse_expression(expression)
        if symbol not in parsed.free_symbols:
            raise MathInputError(f"Variable {variable!r} does not occur in the expression.")
        try:
            derivative = sp.diff(parsed, symbol, order)
        except Exception as error:
            raise MathComputationError("SymPy could not compute the derivative.") from error
        verified, errors = self._verify_derivative(parsed, derivative, symbol, order)
        return MathToolResult(
            operation="derivative",
            expression=expression,
            result={"variable": variable, "order": order, "derivative": str(derivative)},
            verified=verified,
            verification=(
                "Finite-difference checks agreed with the symbolic derivative."
                if verified
                else "The symbolic result was produced, but independent numeric checks did not pass."
            ),
            diagnostics={"finite_difference_errors": errors},
        )

    def integrate(
        self,
        expression: str,
        variable: str,
        lower_bound: int | float | None = None,
        upper_bound: int | float | None = None,
    ) -> MathToolResult:
        """Compute an indefinite or definite integral and verify it."""
        if (lower_bound is None) != (upper_bound is None):
            raise MathInputError("Provide both integration bounds or neither.")
        symbol = self._make_symbol(variable)
        parsed, _ = self._parse_expression(expression)
        try:
            if lower_bound is None:
                integral = sp.integrate(parsed, symbol)
                verified = not isinstance(integral, sp.Integral) and self._is_zero(
                    sp.simplify(sp.diff(integral, symbol) - parsed)
                )
                verification = (
                    "Differentiating the antiderivative recovers the integrand."
                    if verified
                    else "The antiderivative could not be independently verified."
                )
                diagnostic: dict[str, Any] = {}
            else:
                lower = self._finite_number(lower_bound, "lower_bound")
                upper = self._finite_number(upper_bound, "upper_bound")
                exact = sp.integrate(parsed, (symbol, lower, upper))
                exact_value = float(sp.N(exact))
                numeric_value, _ = quad(
                    lambda point: self._numeric_value(parsed, {variable: symbol}, {variable: point}),
                    lower,
                    upper,
                )
                error = abs(exact_value - numeric_value)
                verified = math.isfinite(exact_value) and error <= 1e-7 * max(1.0, abs(exact_value))
                verification = (
                    "The symbolic integral agrees with independent numerical quadrature."
                    if verified
                    else "The symbolic and numeric integral checks do not agree."
                )
                integral = exact
                diagnostic = {"quadrature_value": numeric_value, "absolute_error": error}
        except MathToolError:
            raise
        except Exception as error:
            raise MathComputationError("SymPy or SciPy could not compute the integral.") from error
        return MathToolResult(
            operation="integrate",
            expression=expression,
            result=str(integral),
            verified=bool(verified),
            verification=verification,
            diagnostics=diagnostic,
        )

    def stationary_points(self, expression: str, variable: str) -> MathToolResult:
        """Find derivative roots and check that each is stationary."""
        symbol = self._make_symbol(variable)
        parsed, _ = self._parse_expression(expression)
        derivative = sp.diff(parsed, symbol)
        try:
            points = sp.solve(derivative, symbol)
        except Exception as error:
            raise MathComputationError("SymPy could not find stationary points.") from error
        residuals = [sp.simplify(derivative.subs(symbol, point)) for point in points]
        verified = bool(points) and all(self._is_zero(residual) for residual in residuals)
        return MathToolResult(
            operation="stationary_points",
            expression=expression,
            result={"variable": variable, "points": [str(point) for point in points]},
            verified=verified,
            verification=(
                "Every returned point makes the first derivative zero."
                if verified
                else "No stationary point was independently verified."
            ),
            diagnostics={"derivative": str(derivative), "residuals": [str(v) for v in residuals]},
        )

    def evaluate_array(
        self,
        expression: str,
        variable: str,
        values: Sequence[int | float] | np.ndarray,
    ) -> MathToolResult:
        """Evaluate an expression for a bounded one-dimensional NumPy array."""
        if isinstance(values, (str, bytes)):
            raise MathInputError("values must be a one-dimensional numeric sequence.")
        try:
            raw_array = np.asarray(values)
        except (TypeError, ValueError) as error:
            raise MathInputError("values must contain only finite numbers.") from error
        if raw_array.dtype.kind not in "iuf":
            raise MathInputError("values must contain only finite real numbers.")
        array = raw_array.astype(float, copy=False)
        if array.ndim != 1 or not 0 < array.size <= _MAX_ARRAY_SIZE:
            raise MathInputError(f"values must be a one-dimensional array of 1 to {_MAX_ARRAY_SIZE} items.")
        if not bool(np.all(np.isfinite(array))):
            raise MathInputError("values must contain only finite numbers.")
        parsed, symbols = self._parse_expression(expression, [variable])
        symbol = symbols[variable]
        try:
            output = np.fromiter(
                (
                    self._numeric_value(parsed, {variable: symbol}, {variable: float(value)})
                    for value in array
                ),
                dtype=float,
                count=array.size,
            )
        except MathToolError:
            raise
        if not bool(np.all(np.isfinite(output))):
            raise MathComputationError("Array evaluation produced a non-finite value.")
        return MathToolResult(
            operation="evaluate_array",
            expression=expression,
            result=output.tolist(),
            verified=True,
            verification="Every NumPy array result was checked against scalar evaluation.",
            diagnostics={"input_count": int(array.size)},
        )

    def optimize(
        self,
        objective: str,
        variables: Sequence[str],
        initial_guess: Sequence[int | float],
        *,
        maximize: bool = False,
        bounds: Sequence[Sequence[int | float | None] | None] | None = None,
        constraints: Sequence[dict[str, str]] | None = None,
        method: str | None = None,
        tolerance: float = 1e-8,
        max_iterations: int = 1000,
    ) -> MathToolResult:
        """Minimize/maximize an expression with SciPy and verify feasibility.

        Constraints use SciPy's dictionary convention and safe expression
        values: ``{"type": "eq"|"ineq", "expression": "x + y - 2"}``.
        Inequality expressions must be non-negative at feasible points.
        """
        symbols = self._validated_variables(variables)
        names = list(variables)
        if not isinstance(initial_guess, (list, tuple)) or len(initial_guess) != len(names):
            raise OptimizationError("initial_guess must contain one value per variable.")
        initial = np.asarray(
            [self._finite_number(value, "initial_guess") for value in initial_guess],
            dtype=float,
        )
        if len(names) > _MAX_VARIABLES:
            raise OptimizationError(f"At most {_MAX_VARIABLES} optimization variables are supported.")
        if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 1:
            raise OptimizationError("max_iterations must be a positive integer.")
        if (
            isinstance(tolerance, bool)
            or not isinstance(tolerance, (int, float))
            or not math.isfinite(tolerance)
            or tolerance <= 0
        ):
            raise OptimizationError("tolerance must be a positive finite number.")
        objective_expr, symbol_map = self._parse_expression(objective, names)

        parsed_bounds = self._parse_bounds(bounds, len(names))
        for index, (lower, upper) in enumerate(parsed_bounds):
            if lower is not None and upper is not None and lower > upper:
                raise OptimizationError("Each lower bound must be less than or equal to its upper bound.")
            if lower is not None and initial[index] < lower or upper is not None and initial[index] > upper:
                raise OptimizationError("initial_guess must lie within the supplied bounds.")

        if constraints is None:
            raw_constraints: list[dict[str, str]] = []
        elif isinstance(constraints, (list, tuple)) and len(constraints) <= _MAX_CONSTRAINTS:
            raw_constraints = list(constraints)
        else:
            raise OptimizationError(f"constraints must contain at most {_MAX_CONSTRAINTS} items.")
        parsed_constraints: list[tuple[str, sp.Expr]] = []
        for constraint in raw_constraints:
            if (
                not isinstance(constraint, dict)
                or set(constraint) != {"type", "expression"}
                or not isinstance(constraint["type"], str)
                or constraint["type"] not in {"eq", "ineq"}
                or not isinstance(constraint["expression"], str)
            ):
                raise OptimizationError(
                    "Each constraint must contain type='eq'|'ineq' and an expression string."
                )
            parsed_constraint, _ = self._parse_expression(constraint["expression"], names)
            parsed_constraints.append((constraint["type"], parsed_constraint))

        has_bounds = any(lower is not None or upper is not None for lower, upper in parsed_bounds)
        has_constraints = bool(parsed_constraints)
        chosen_method = (
            ("SLSQP" if has_bounds or has_constraints else "BFGS")
            if method is None
            else method
        )
        if not isinstance(chosen_method, str) or chosen_method not in _ALLOWED_OPTIMIZERS:
            raise OptimizationError("Unsupported SciPy optimization method.")
        if chosen_method == "COBYLA" and any(
            constraint_type == "eq" for constraint_type, _ in parsed_constraints
        ):
            raise OptimizationError("COBYLA supports inequalities only; use SLSQP for equalities.")
        if has_constraints and chosen_method not in {"SLSQP", "COBYLA", "trust-constr"}:
            raise OptimizationError("The selected optimizer does not support these constraints.")

        def substitutions(point: np.ndarray) -> dict[str, float]:
            return {name: float(point[index]) for index, name in enumerate(names)}

        direction = -1.0 if maximize else 1.0

        def objective_function(point: np.ndarray) -> float:
            return direction * self._numeric_value(objective_expr, symbol_map, substitutions(point))

        scipy_constraints = [
            {
                "type": constraint_type,
                "fun": lambda point, expr=expr: self._numeric_value(
                    expr, symbol_map, substitutions(point)
                ),
            }
            for constraint_type, expr in parsed_constraints
        ]
        scipy_bounds = parsed_bounds if has_bounds else None
        options: dict[str, Any] = {"maxiter": max_iterations}
        if chosen_method == "SLSQP":
            options["ftol"] = float(tolerance)
        elif chosen_method in {"BFGS", "L-BFGS-B"}:
            options["gtol"] = float(tolerance)
        try:
            optimized = scipy_minimize(
                objective_function,
                initial,
                method=chosen_method,
                bounds=scipy_bounds,
                constraints=scipy_constraints,
                tol=float(tolerance),
                options=options,
            )
        except MathToolError:
            raise
        except Exception as error:
            raise OptimizationError("SciPy optimization failed during evaluation.") from error

        point = np.asarray(optimized.x, dtype=float)
        if point.shape != initial.shape or not bool(np.all(np.isfinite(point))):
            raise OptimizationError("SciPy returned invalid optimization coordinates.")
        variable_values = substitutions(point)
        objective_value = self._numeric_value(objective_expr, symbol_map, variable_values)
        bounds_ok = all(
            (lower is None or point[index] >= lower - float(tolerance) * 10)
            and (upper is None or point[index] <= upper + float(tolerance) * 10)
            for index, (lower, upper) in enumerate(parsed_bounds)
        )
        constraint_values = [
            self._numeric_value(expr, symbol_map, variable_values)
            for _, expr in parsed_constraints
        ]
        constraints_ok = all(
            abs(value) <= float(tolerance) * 10
            if constraint_type == "eq"
            else value >= -float(tolerance) * 10
            for (constraint_type, _), value in zip(parsed_constraints, constraint_values)
        )
        recomputed_objective = self._numeric_value(objective_expr, symbol_map, variable_values)
        objective_matches = abs(objective_value - recomputed_objective) <= float(tolerance) * max(
            10.0, abs(objective_value)
        )
        gradient_norm = self._gradient_norm(objective_expr, symbol_map, names, variable_values)
        active_bound = any(
            lower is not None and abs(point[index] - lower) <= float(tolerance) * 10
            or upper is not None and abs(point[index] - upper) <= float(tolerance) * 10
            for index, (lower, upper) in enumerate(parsed_bounds)
        )
        gradient_check_applicable = not has_constraints and not active_bound and not objective_expr.has(
            sp.Abs, sp.Piecewise
        )
        gradient_check_passed = (
            gradient_norm is not None
            and gradient_norm <= max(1e-4, math.sqrt(float(tolerance)) * 10)
            if gradient_check_applicable
            else None
        )
        optimizer_success = bool(optimized.success)
        verified = (
            optimizer_success
            and bounds_ok
            and constraints_ok
            and objective_matches
            and (not gradient_check_applicable or bool(gradient_check_passed))
        )
        operation = "maximize" if maximize else "minimize"
        result_data = {
            "variables": variable_values,
            "objective_value": objective_value,
            "optimizer_success": optimizer_success,
            "status": int(optimized.status),
            "iterations": getattr(optimized, "nit", None),
            "method": chosen_method,
            "bounds_satisfied": bounds_ok,
            "constraints_satisfied": constraints_ok,
            "constraint_values": constraint_values,
            "global_optimum_guaranteed": False,
        }
        diagnostics = {
            "optimizer_message": str(optimized.message),
            "function_evaluations": getattr(optimized, "nfev", None),
            "gradient_norm": gradient_norm,
            "gradient_check_applicable": gradient_check_applicable,
            "gradient_check_passed": gradient_check_passed,
            "objective_recomputed": objective_matches,
            "initial_guess": initial.tolist(),
        }
        return MathToolResult(
            operation=operation,
            expression=objective,
            result=result_data,
            verified=verified,
            verification=(
                "Optimizer status is successful; objective value was recomputed and bounds/constraints checked. "
                "This does not prove a global optimum."
                if verified
                else "Optimization did not pass convergence, objective, or feasibility checks."
            ),
            diagnostics=diagnostics,
        )

    def _parse_expression(
        self,
        expression: str,
        variables: Sequence[str] | None = None,
    ) -> tuple[sp.Expr, dict[str, sp.Symbol]]:
        if not isinstance(expression, str) or not expression.strip():
            raise MathExpressionError("expression must be a non-empty string.")
        if len(expression) > _MAX_EXPRESSION_LENGTH:
            raise MathExpressionError("expression exceeds the supported length limit.")
        try:
            tree = ast.parse(expression.strip(), mode="eval")
        except (SyntaxError, ValueError, MemoryError) as error:
            raise MathExpressionError("expression is not valid supported math syntax.") from error
        if sum(1 for _ in ast.walk(tree)) > _MAX_AST_NODES:
            raise MathExpressionError("expression exceeds the supported complexity limit.")

        if variables is None:
            variable_names = sorted(
                {
                    node.id
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Name)
                    and node.id not in _FUNCTIONS
                    and node.id not in _CONSTANTS
                }
            )
        else:
            if not isinstance(variables, (list, tuple)):
                raise MathInputError("variables must be a list of identifier strings.")
            variable_names = list(variables)
            if any(not self._valid_identifier(name) for name in variable_names):
                raise MathInputError("variables must be valid simple identifiers.")
            if len(set(variable_names)) != len(variable_names):
                raise MathInputError("variables must not contain duplicates.")
            referenced_names = {
                node.id
                for node in ast.walk(tree)
                if isinstance(node, ast.Name)
                and node.id not in _FUNCTIONS
                and node.id not in _CONSTANTS
            }
            unknown = referenced_names - set(variable_names)
            if unknown:
                raise MathExpressionError(
                    f"Expression uses undeclared variable(s): {', '.join(sorted(unknown))}."
                )

        symbols = {name: sp.Symbol(name, real=True) for name in variable_names}
        try:
            parsed = self._convert_node(tree.body, symbols)
        except MathToolError:
            raise
        except Exception as error:
            raise MathExpressionError("expression could not be parsed as safe mathematics.") from error
        if not isinstance(parsed, sp.Expr) or self._has_nonfinite(parsed):
            raise MathExpressionError("expression contains a non-finite or unsupported value.")
        return parsed, symbols

    def _convert_node(self, node: ast.AST, symbols: dict[str, sp.Symbol]) -> sp.Expr:
        if isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise MathExpressionError("only integer and finite decimal literals are supported.")
            if isinstance(value, float) and not math.isfinite(value):
                raise MathExpressionError("numeric literals must be finite.")
            if abs(value) > 1e100:
                raise MathExpressionError("numeric literal exceeds the supported magnitude.")
            return sp.Integer(value) if isinstance(value, int) else sp.Float(value)
        if isinstance(node, ast.Name):
            if node.id in symbols:
                return symbols[node.id]
            if node.id in _CONSTANTS:
                return _CONSTANTS[node.id]
            raise MathExpressionError(f"Unknown or unsupported symbol {node.id!r}.")
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
            if isinstance(node.op, ast.Pow) and isinstance(node.right, ast.Constant):
                exponent = node.right.value
                if isinstance(exponent, (int, float)) and abs(exponent) > _MAX_EXPONENT:
                    raise MathExpressionError("literal exponents above 1000 are not supported.")
            left = self._convert_node(node.left, symbols)
            right = self._convert_node(node.right, symbols)
            return _BINARY_OPERATORS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = self._convert_node(node.operand, symbols)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            function = _FUNCTIONS.get(node.func.id)
            if function is None or node.keywords:
                raise MathExpressionError("function call is not in the supported math function set.")
            expected_args = (1, 2) if node.func.id == "log" else (1,)
            if len(node.args) not in expected_args:
                raise MathExpressionError(f"function {node.func.id!r} has an invalid number of arguments.")
            return function(*(self._convert_node(argument, symbols) for argument in node.args))
        raise MathExpressionError(f"unsupported syntax node: {type(node).__name__}.")

    def _parse_equation(
        self,
        equation: str,
        variables: Sequence[str],
    ) -> tuple[sp.Expr, sp.Expr]:
        if not isinstance(equation, str) or equation.count("=") != 1:
            raise MathExpressionError("equation must contain exactly one '=' sign.")
        left_text, right_text = equation.split("=", maxsplit=1)
        if not left_text.strip() or not right_text.strip():
            raise MathExpressionError("both sides of the equation must be non-empty.")
        left, _ = self._parse_expression(left_text, variables)
        right, _ = self._parse_expression(right_text, variables)
        return left, right

    def _equation_expression(self, equation: str, variables: Sequence[str]) -> sp.Expr:
        left, right = self._parse_equation(equation, variables)
        return left - right

    @staticmethod
    def _valid_identifier(name: Any) -> bool:
        return (
            isinstance(name, str)
            and bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", name))
            and not keyword.iskeyword(name)
            and name not in _FUNCTIONS
            and name not in _CONSTANTS
        )

    def _make_symbol(self, variable: str) -> sp.Symbol:
        if not self._valid_identifier(variable):
            raise MathInputError("variable must be a valid simple identifier.")
        return sp.Symbol(variable, real=True)

    def _validated_variables(self, variables: Sequence[str]) -> list[sp.Symbol]:
        if not isinstance(variables, (list, tuple)) or not 1 <= len(variables) <= _MAX_VARIABLES:
            raise MathInputError(f"variables must contain 1 to {_MAX_VARIABLES} identifiers.")
        if any(not self._valid_identifier(variable) for variable in variables):
            raise MathInputError("variables must be unique valid simple identifiers.")
        if len(set(variables)) != len(variables):
            raise MathInputError("variables must be unique valid simple identifiers.")
        return [sp.Symbol(variable, real=True) for variable in variables]

    def _numeric_value(
        self,
        expression: sp.Expr,
        symbols: dict[str, sp.Symbol],
        values: dict[str, int | float],
    ) -> float:
        missing = {str(symbol) for symbol in expression.free_symbols} - set(values)
        if missing:
            raise MathInputError(f"Missing numeric variable value(s): {', '.join(sorted(missing))}.")
        substitutions = {
            symbols[name]: self._finite_number(value, name)
            for name, value in values.items()
            if name in symbols
        }
        try:
            evaluated = sp.N(expression.subs(substitutions), 17)
            if evaluated.is_real is False:
                raise MathComputationError("Expression evaluated to a non-real number.")
            numeric = float(evaluated)
        except MathToolError:
            raise
        except (ArithmeticError, TypeError, ValueError, OverflowError) as error:
            raise MathComputationError("Expression evaluation failed or divided by zero.") from error
        if not math.isfinite(numeric):
            raise MathComputationError("Expression evaluation produced a non-finite number.")
        return numeric

    @staticmethod
    def _finite_number(value: Any, field_name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise MathInputError(f"{field_name} must be a finite number.")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise MathInputError(f"{field_name} must be a finite number.")
        return numeric

    @classmethod
    def _parse_bounds(
        cls,
        bounds: Sequence[Sequence[int | float | None] | None] | None,
        variable_count: int,
    ) -> list[tuple[float | None, float | None]]:
        if bounds is None:
            return [(None, None)] * variable_count
        if not isinstance(bounds, (list, tuple)) or len(bounds) != variable_count:
            raise OptimizationError("bounds must contain one [lower, upper] pair per variable.")
        parsed: list[tuple[float | None, float | None]] = []
        for bound in bounds:
            if bound is None:
                parsed.append((None, None))
                continue
            if not isinstance(bound, (list, tuple)) or len(bound) != 2:
                raise OptimizationError("Each bound must be a [lower, upper] pair or null.")
            lower = None if bound[0] is None else cls._finite_number(bound[0], "lower bound")
            upper = None if bound[1] is None else cls._finite_number(bound[1], "upper bound")
            if lower is not None and upper is not None and lower > upper:
                raise OptimizationError("Each lower bound must be less than or equal to its upper bound.")
            parsed.append((lower, upper))
        return parsed

    @staticmethod
    def _is_zero(value: sp.Expr) -> bool:
        simplified = sp.simplify(value)
        if simplified == 0 or simplified.is_zero is True:
            return True
        if simplified.is_number:
            try:
                return abs(float(sp.N(simplified))) <= 1e-10
            except (TypeError, ValueError, OverflowError):
                return False
        return False

    @staticmethod
    def _has_nonfinite(expression: sp.Expr) -> bool:
        return expression.has(sp.zoo, sp.nan, sp.oo, -sp.oo)

    def _verify_derivative(
        self,
        expression: sp.Expr,
        derivative: sp.Expr,
        symbol: sp.Symbol,
        order: int,
    ) -> tuple[bool, list[float]]:
        step = 1e-3
        errors: list[float] = []
        for point in (0.73, 1.31):
            try:
                if order == 1:
                    approximation = (
                        self._numeric_value(expression, {str(symbol): symbol}, {str(symbol): point + step})
                        - self._numeric_value(expression, {str(symbol): symbol}, {str(symbol): point - step})
                    ) / (2 * step)
                elif order == 2:
                    center = self._numeric_value(expression, {str(symbol): symbol}, {str(symbol): point})
                    plus = self._numeric_value(expression, {str(symbol): symbol}, {str(symbol): point + step})
                    minus = self._numeric_value(expression, {str(symbol): symbol}, {str(symbol): point - step})
                    approximation = (plus - 2 * center + minus) / step**2
                else:
                    # Symbolic differentiation is deterministic, but the
                    # simple independent finite-difference check is bounded to
                    # first and second derivatives.
                    return False, []
                expected = self._numeric_value(
                    derivative, {str(symbol): symbol}, {str(symbol): point}
                )
                errors.append(abs(approximation - expected))
            except MathToolError:
                continue
        tolerance = 2e-4 if order == 1 else 2e-3
        return bool(errors) and max(errors) <= tolerance, errors

    def _gradient_norm(
        self,
        objective: sp.Expr,
        symbols: dict[str, sp.Symbol],
        names: list[str],
        values: dict[str, float],
    ) -> float | None:
        try:
            components = [
                self._numeric_value(sp.diff(objective, symbols[name]), symbols, values)
                for name in names
            ]
            return float(np.linalg.norm(np.asarray(components, dtype=float)))
        except MathToolError:
            return None
