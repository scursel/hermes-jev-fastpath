"""Bounded safe-AST calculator.

Every candidate expression is parsed with :mod:`ast`, every node is validated against an
allowlist, and every intermediate result is bounded BEFORE evaluation continues, so a
pathological payload can never allocate unbounded integers or execute anything.
"""

from __future__ import annotations

import ast
import operator
import re

MAX_EXPRESSION_CHARS = 160
MAX_AST_NODES = 64
MAX_INTEGER_DIGITS = 100
MAX_ABS_EXPONENT = 12

_CALC_PREFIXES = (
    "qual o resultado de", "qual e o resultado de", "quanto e o resultado de",
    "what is the result of", "quanto dá", "quanto da", "quanto é", "quanto e",
    "calculate", "calcule", "calcular", "calcula", "calculo de", "calculo",
    "compute", "resolve", "resolva", "calc", "what is", "what's", "whats",
)

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}

_ALLOWED_CHARS = re.compile(r"^[0-9+\-*/%().\s]+$")
_LETTERS = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]")


class ArithmeticRejected(ValueError):
    """The text is not a bounded, pure arithmetic request."""


def _strip_prefix(text: str) -> str:
    lowered = text.strip().lower()
    for prefix in _CALC_PREFIXES:
        if lowered.startswith(prefix):
            rest = text.strip()[len(prefix):]
            if rest.startswith(":"):
                rest = rest[1:]
            return rest.strip()
    return text.strip()


def extract_expression(text: str) -> str:
    """Remove only an allowlisted Portuguese/English calculator prefix and one terminal question mark."""
    if not isinstance(text, str):
        raise ArithmeticRejected("input must be text")
    candidate = _strip_prefix(text)
    if candidate.endswith("?") or candidate.endswith("="):
        candidate = candidate[:-1].strip()
    if not candidate:
        raise ArithmeticRejected("no expression found")
    if _LETTERS.search(candidate):
        raise ArithmeticRejected("expression contains words")
    return candidate


def _reject(message: str) -> "ArithmeticRejected":
    return ArithmeticRejected(message)


def _check_int(value: int) -> int:
    if len(str(abs(value))) > MAX_INTEGER_DIGITS:
        raise _reject("integer result exceeds the digit cap")
    return value


def _eval_node(node: ast.AST) -> int | float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise _reject("only numeric constants are allowed")
        if isinstance(node.value, int):
            return _check_int(node.value)
        value = float(node.value)
        if value != value or value in (float("inf"), float("-inf")):
            raise _reject("non-finite constant")
        return value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow):
            exponent = float(right)
            if not (exponent == exponent) or exponent in (float("inf"), float("-inf")):
                raise _reject("non-finite exponent")
            if abs(exponent) > MAX_ABS_EXPONENT:
                raise _reject("exponent out of range")
        try:
            result = _BIN_OPS[type(node.op)](left, right)
        except ZeroDivisionError as exc:
            raise _reject("division by zero") from exc
        except OverflowError as exc:
            raise _reject("result overflow") from exc
        if isinstance(result, complex):
            raise _reject("complex results are not supported")
        if isinstance(result, int):
            return _check_int(result)
        result = float(result)
        if result != result or result in (float("inf"), float("-inf")):
            raise _reject("non-finite result")
        return result
    raise _reject("unsupported syntax")


def evaluate_expression(expression: str) -> int | float:
    """Parse with ``ast.parse(..., mode='eval')``, validate every node, then evaluate recursively."""
    if not isinstance(expression, str) or not expression.strip():
        raise _reject("empty expression")
    if len(expression) > MAX_EXPRESSION_CHARS:
        raise _reject("expression too long")
    if not _ALLOWED_CHARS.match(expression):
        raise _reject("expression contains unsupported characters")
    if not re.search(r"\d", expression):
        raise _reject("expression has no numeric value")
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise _reject("expression failed to parse") from exc
    if len(list(ast.walk(tree))) > MAX_AST_NODES:
        raise _reject("expression too complex")
    return _eval_node(tree)


def format_number(value: int | float) -> str:
    """Deterministic, locale-independent rendering: integral floats collapse to integers."""
    if isinstance(value, int):
        return str(value)
    if value == int(value):
        return str(int(value))
    return repr(value)


def render_calculation(text: str) -> str:
    """Extract, evaluate, and render one bounded arithmetic answer."""
    expression = extract_expression(text)
    value = evaluate_expression(expression)
    return f"{expression} = {format_number(value)}"
