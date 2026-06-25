from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

_MAX_AST_NODES = 160
_MAX_SCALAR_ABS = 1e9


@dataclass(frozen=True, slots=True)
class VectorExpressionResult:
    expression: str
    label: str
    alias: str
    vector: np.ndarray
    magnitude: float
    referenced_aliases: tuple[str, ...]


def alias_for_index(index: int) -> str:
    """Return Excel-style aliases: A..Z, AA..AZ, BA..."""
    if index < 0:
        raise ValueError("Alias index must be non-negative.")
    value = index + 1
    chars: list[str] = []
    while value:
        value, remainder = divmod(value - 1, 26)
        chars.append(chr(ord("A") + remainder))
    return "".join(reversed(chars))


def alias_map(vectors: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(vectors, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("vectors must have shape [token_count, hidden_dim].")
    return {alias_for_index(index): values[index] for index in range(values.shape[0])}


@dataclass(slots=True)
class _Value:
    kind: str
    value: float | np.ndarray


def spherical_linear_interpolation(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Interpolate two vectors along the unit-hypersphere geodesic.

    Direction follows SLERP while magnitude is linearly interpolated. This
    preserves both endpoint vectors exactly and keeps the operation meaningful
    for embedding rows whose L2 norms differ.
    """
    first = np.asarray(a, dtype=np.float64)
    second = np.asarray(b, dtype=np.float64)
    fraction = float(t)
    if not math.isfinite(fraction) or fraction < 0.0 or fraction > 1.0:
        raise ValueError("SLERP t must be a finite scalar in the interval [0, 1].")
    norm_a = float(np.linalg.norm(first))
    norm_b = float(np.linalg.norm(second))
    if norm_a <= 1e-15 or norm_b <= 1e-15:
        raise ValueError("SLERP requires two non-zero vectors.")
    if fraction <= 0.0:
        return first.copy()
    if fraction >= 1.0:
        return second.copy()

    unit_a = first / norm_a
    unit_b = second / norm_b
    dot = float(np.clip(np.dot(unit_a, unit_b), -1.0, 1.0))

    if dot > 0.9995:
        direction = (1.0 - fraction) * unit_a + fraction * unit_b
        direction_norm = float(np.linalg.norm(direction))
        direction = direction / max(direction_norm, 1e-15)
    elif dot < -0.9995:
        axis_index = int(np.argmin(np.abs(unit_a)))
        basis = np.zeros_like(unit_a)
        basis[axis_index] = 1.0
        orthogonal = basis - float(np.dot(basis, unit_a)) * unit_a
        orthogonal /= max(float(np.linalg.norm(orthogonal)), 1e-15)
        direction = math.cos(math.pi * fraction) * unit_a + math.sin(math.pi * fraction) * orthogonal
    else:
        angle = math.acos(dot)
        sine = math.sin(angle)
        direction = (
            math.sin((1.0 - fraction) * angle) / sine * unit_a
            + math.sin(fraction * angle) / sine * unit_b
        )
        direction /= max(float(np.linalg.norm(direction)), 1e-15)

    magnitude = (1.0 - fraction) * norm_a + fraction * norm_b
    return direction * magnitude


class _SafeVectorEvaluator:
    def __init__(self, aliases: dict[str, np.ndarray]):
        self.aliases = {name.upper(): np.asarray(vector, dtype=np.float64) for name, vector in aliases.items()}
        self.referenced: set[str] = set()

    def evaluate(self, expression: str) -> np.ndarray:
        text = str(expression or "").strip()
        if not text:
            raise ValueError("Vector expression cannot be blank.")
        if len(text) > 500:
            raise ValueError("Vector expression is too long (maximum 500 characters).")
        try:
            tree = ast.parse(text, mode="eval")
        except SyntaxError as exc:
            message = exc.msg or "invalid syntax"
            raise ValueError(f"Invalid vector expression: {message}.") from exc
        if sum(1 for _ in ast.walk(tree)) > _MAX_AST_NODES:
            raise ValueError("Vector expression is too complex.")
        result = self._visit(tree.body)
        if result.kind != "vector":
            raise ValueError("The expression must evaluate to a vector, not a scalar.")
        vector = np.asarray(result.value, dtype=np.float64)
        if not np.all(np.isfinite(vector)):
            raise ValueError("The resultant vector contains NaN or infinite values.")
        return vector

    def _visit(self, node: ast.AST) -> _Value:
        if isinstance(node, ast.Name):
            name = node.id.upper()
            vector = self.aliases.get(name)
            if vector is None:
                available = ", ".join(list(self.aliases)[:12])
                suffix = "…" if len(self.aliases) > 12 else ""
                raise ValueError(f"Unknown vector alias {node.id!r}. Available aliases: {available}{suffix}.")
            self.referenced.add(name)
            return _Value("vector", vector.copy())

        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            scalar = float(node.value)
            if not math.isfinite(scalar) or abs(scalar) > _MAX_SCALAR_ABS:
                raise ValueError("Scalar constants must be finite and reasonably sized.")
            return _Value("scalar", scalar)

        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            operand = self._visit(node.operand)
            sign = 1.0 if isinstance(node.op, ast.UAdd) else -1.0
            return _Value(operand.kind, operand.value * sign)

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only mean(...) and slerp(A, B, t) vector functions are supported.")
            function_name = node.func.id.casefold()
            if node.keywords:
                raise ValueError("Vector functions accept positional arguments only.")
            if function_name in {"mean", "avg", "average"}:
                if not node.args:
                    raise ValueError(f"{node.func.id} requires at least one vector argument.")
                values = [self._visit(argument) for argument in node.args]
                if any(value.kind != "vector" for value in values):
                    raise ValueError(f"{node.func.id} accepts vector arguments only.")
                return _Value(
                    "vector",
                    np.mean(np.stack([np.asarray(value.value) for value in values], axis=0), axis=0),
                )
            if function_name != "slerp":
                raise ValueError("Only mean(...) and slerp(A, B, t) vector functions are supported.")
            if len(node.args) != 3:
                raise ValueError("slerp requires exactly three positional arguments: slerp(A, B, t).")
            first = self._visit(node.args[0])
            second = self._visit(node.args[1])
            fraction = self._visit(node.args[2])
            if first.kind != "vector" or second.kind != "vector" or fraction.kind != "scalar":
                raise ValueError("slerp requires two vectors followed by a scalar t in [0, 1].")
            return _Value(
                "vector",
                spherical_linear_interpolation(
                    np.asarray(first.value),
                    np.asarray(second.value),
                    float(fraction.value),
                ),
            )

        if isinstance(node, ast.BinOp):
            left = self._visit(node.left)
            right = self._visit(node.right)
            if isinstance(node.op, (ast.Add, ast.Sub)):
                if left.kind != right.kind:
                    raise ValueError("Addition and subtraction require two vectors or two scalars.")
                sign = 1.0 if isinstance(node.op, ast.Add) else -1.0
                return _Value(left.kind, left.value + sign * right.value)

            if isinstance(node.op, ast.Mult):
                if left.kind == "vector" and right.kind == "vector":
                    raise ValueError("Vector-by-vector multiplication is not supported; multiply by a scalar instead.")
                if left.kind == "scalar" and right.kind == "scalar":
                    return _Value("scalar", float(left.value) * float(right.value))
                if left.kind == "vector":
                    return _Value("vector", np.asarray(left.value) * float(right.value))
                return _Value("vector", np.asarray(right.value) * float(left.value))

            if isinstance(node.op, ast.Div):
                if right.kind != "scalar":
                    raise ValueError("Division is only supported by a scalar denominator.")
                denominator = float(right.value)
                if abs(denominator) <= 1e-15:
                    raise ValueError("Division by zero is not allowed.")
                if left.kind == "scalar":
                    return _Value("scalar", float(left.value) / denominator)
                return _Value("vector", np.asarray(left.value) / denominator)

        raise ValueError(
            "Unsupported expression syntax. Use vector aliases, numeric scalars, parentheses, +, -, *, /, mean(...), and slerp(A, B, t)."
        )


def evaluate_vector_expression(expression: str, aliases: dict[str, np.ndarray]) -> tuple[np.ndarray, tuple[str, ...]]:
    evaluator = _SafeVectorEvaluator(aliases)
    vector = evaluator.evaluate(expression)
    return vector, tuple(sorted(evaluator.referenced, key=lambda name: (len(name), name)))


def evaluate_vector_expressions(
    vectors: np.ndarray,
    expressions: Iterable[object],
) -> list[VectorExpressionResult]:
    """Evaluate request-like objects with ``expression`` and optional ``label`` attributes."""
    aliases = alias_map(vectors)
    results: list[VectorExpressionResult] = []
    for index, item in enumerate(expressions):
        expression = str(getattr(item, "expression", "") or "").strip()
        requested_label = str(getattr(item, "label", "") or "").strip()
        vector, referenced = evaluate_vector_expression(expression, aliases)
        magnitude = float(np.linalg.norm(vector))
        if not math.isfinite(magnitude):
            raise ValueError(f"Resultant {index + 1} has a non-finite magnitude.")
        if magnitude <= 1e-12:
            raise ValueError(
                f"Resultant {index + 1} is zero or near-zero and has no direction to project."
            )
        alias = f"R{index + 1}"
        results.append(
            VectorExpressionResult(
                expression=expression,
                label=requested_label or alias,
                alias=alias,
                vector=vector,
                magnitude=magnitude,
                referenced_aliases=referenced,
            )
        )
    return results
