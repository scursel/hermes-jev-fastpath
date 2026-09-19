"""Bounded safe-AST calculator with exact rational semantics.

Every candidate expression is parsed with :mod:`ast`, every node is validated against an
allowlist, and every intermediate result is bounded BEFORE evaluation continues, so a
pathological payload can never allocate unbounded integers or execute anything.

Numbers are exact: integer literals stay integers, decimal literals are read as exact
decimals (never binary floats), and every operation runs over :class:`fractions.Fraction`.
Exact arithmetic cannot overflow, so the resource guard is the digit cap applied to every
intermediate result.
"""

from __future__ import annotations

import ast
import operator
import re
from decimal import Decimal
from fractions import Fraction

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


def _check_fraction(value: Fraction) -> Fraction:
    if len(str(abs(value.numerator))) > MAX_INTEGER_DIGITS or len(str(value.denominator)) > MAX_INTEGER_DIGITS:
        raise _reject("result exceeds the digit cap")
    return value


def _fraction_constant(value) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _reject("only numeric constants are allowed")
    if isinstance(value, int):
        return _check_fraction(Fraction(value))
    # Read the decimal the user actually wrote (repr round-trips), never the binary float.
    number = Decimal(repr(float(value)))
    if not number.is_finite():
        raise _reject("non-finite constant")
    return _check_fraction(Fraction(number))


def _eval_node(node: ast.AST) -> Fraction:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        return _fraction_constant(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow):
            if right.denominator != 1:
                raise _reject("exponent must be an integer")
            exponent = right.numerator
            if abs(exponent) > MAX_ABS_EXPONENT:
                raise _reject("exponent out of range")
        try:
            result = _BIN_OPS[type(node.op)](left, right)
        except ZeroDivisionError as exc:
            raise _reject("division by zero") from exc
        except OverflowError as exc:
            raise _reject("result overflow") from exc
        return _check_fraction(Fraction(result))
    raise _reject("unsupported syntax")


def has_binary_operator(expression: str) -> bool:
    """True when the parsed expression performs at least one binary operation.

    Bare numbers (phone numbers, OTP codes, menu replies) must never become calculator
    candidates, so an explicit operator is required before Jev ever sees the text.
    """
    if not isinstance(expression, str) or not expression.strip():
        return False
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError, RecursionError):
        return False
    return any(isinstance(node, ast.BinOp) for node in ast.walk(tree))


def evaluate_expression(expression: str) -> int | Fraction:
    """Parse with ``ast.parse(..., mode='eval')``, validate every node, then evaluate exactly."""
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


def format_number(value: int | Fraction) -> str:
    """Deterministic, locale-independent rendering of an exact result.

    Integers render as integers; terminating decimals render exactly (``0.1 + 0.2`` is
    ``0.3``); non-terminating rationals render as the exact fraction ``numerator/denominator``
    instead of a misleading rounded decimal.
    """
    if isinstance(value, int):
        return str(value)
    if not isinstance(value, Fraction):
        return str(value)
    if value.denominator == 1:
        return str(value.numerator)
    numerator, denominator = value.numerator, value.denominator
    twos, rest = 0, denominator
    while rest % 2 == 0:
        rest //= 2
        twos += 1
    fives, rest = 0, rest
    while rest % 5 == 0:
        rest //= 5
        fives += 1
    if rest != 1:
        # Non-terminating decimal (e.g. 1/3): render the exact fraction.
        return f"{numerator}/{denominator}"
    places = max(twos, fives)
    scaled = abs(numerator) * (2 ** (places - twos)) * (5 ** (places - fives))
    digits = str(scaled).zfill(places + 1)
    sign = "-" if numerator < 0 else ""
    text = f"{sign}{digits[:-places]}.{digits[-places:]}".rstrip("0").rstrip(".")
    return text or "0"


def render_calculation(text: str) -> str:
    """Extract, evaluate, and render one bounded arithmetic answer."""
    expression = extract_expression(text)
    value = evaluate_expression(expression)
    return f"{expression} = {format_number(value)}"
