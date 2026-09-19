"""Bounded safe-AST calculator: malicious payloads reject before evaluation."""

import pytest

from jev_fastpath.arithmetic import (
    ArithmeticRejected,
    evaluate_expression,
    extract_expression,
    format_number,
    render_calculation,
)


class TestExtractExpression:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("2 + 2", "2 + 2"),
            ("quanto é (17 * 9) - 4?", "(17 * 9) - 4"),
            ("calcule 12.5 / 5", "12.5 / 5"),
            ("What is 2 + 2?", "2 + 2"),
            ("qual o resultado de 3 ** 4?", "3 ** 4"),
            ("QUANTO É 7*6", "7*6"),
        ],
    )
    def test_prefixes_and_question_mark(self, text, expected):
        assert extract_expression(text) == expected

    def test_no_expression_rejects(self):
        for text in ("", "   ", "?", "quanto é?", "calcule", "hello world"):
            with pytest.raises(ArithmeticRejected):
                extract_expression(text)

    def test_leftover_words_reject(self):
        with pytest.raises(ArithmeticRejected):
            extract_expression("calcule o desconto ideal para meu negócio")

    def test_only_one_terminal_question_mark_stripped(self):
        assert extract_expression("quanto é (2+3)??") == "(2+3)?"


class TestEvaluateExpression:
    @pytest.mark.parametrize(
        ("expression", "expected"),
        [
            ("2 + 2", 4),
            ("(17 * 9) - 4", 149),
            ("12.5 / 5", 2.5),
            ("10 // 3", 3),
            ("10 % 3", 1),
            ("2 ** 10", 1024),
            ("-5 + 3", -2),
            ("+7", 7),
            ("2 + 3 * 4", 14),
            ("-(2 + 3)", -5),
            ("1.5 + 1.5", 3.0),
        ],
    )
    def test_valid_expressions(self, expression, expected):
        assert evaluate_expression(expression) == expected

    def test_decimal_literals_are_exact_decimal_not_binary_float(self):
        from fractions import Fraction

        assert evaluate_expression("0.1 + 0.2") == Fraction(3, 10)

    def test_large_division_is_exact(self):
        assert evaluate_expression("12345678901234567890 / 2") == 6172839450617283945

    def test_small_division_is_exact(self):
        from fractions import Fraction

        assert evaluate_expression("1 / 100000000") == Fraction(1, 10**8)

    def test_division_by_zero_rejects(self):
        with pytest.raises(ArithmeticRejected):
            evaluate_expression("1 / 0")
        with pytest.raises(ArithmeticRejected):
            evaluate_expression("5 % 0")
        with pytest.raises(ArithmeticRejected):
            evaluate_expression("5 // 0")

    def test_exponent_bounds(self):
        assert evaluate_expression("2 ** 12") == 4096
        with pytest.raises(ArithmeticRejected):
            evaluate_expression("2 ** 13")
        with pytest.raises(ArithmeticRejected):
            evaluate_expression("2 ** -13")

    def test_non_integer_exponent_rejects(self):
        with pytest.raises(ArithmeticRejected):
            evaluate_expression("2 ** 0.5")
        with pytest.raises(ArithmeticRejected):
            evaluate_expression("9 ** (1 / 2)")

    def test_literal_digit_cap(self):
        assert evaluate_expression("9" * 100) == int("9" * 100)
        with pytest.raises(ArithmeticRejected):
            evaluate_expression("9" * 101)

    def test_intermediate_results_never_grow_unbounded(self):
        assert evaluate_expression("(9 ** 9) ** 9") == 387420489 ** 9
        with pytest.raises(ArithmeticRejected):
            evaluate_expression("((9 ** 9) ** 9) ** 9")

    def test_expression_length_cap(self):
        assert evaluate_expression("8" * 100 + " + 1") == int("8" * 100) + 1
        with pytest.raises(ArithmeticRejected):
            evaluate_expression("9" * 100 + " + " + "9" * 100)

    def test_node_count_cap(self):
        assert evaluate_expression(" + ".join(["1"] * 20)) == 20
        with pytest.raises(ArithmeticRejected):
            evaluate_expression(" + ".join(["1"] * 30))

    @pytest.mark.parametrize(
        "expression",
        [
            "__import__('os').system('id')",
            "(lambda: 1)()",
            "[1, 2][0]",
            "True + 1",
            "'2' * 3",
            "2 if 1 else 3",
            "os.getcwd()",
            "().__class__",
            "f(1)",
        ],
    )
    def test_malicious_and_unsupported_shapes_reject(self, expression):
        with pytest.raises(ArithmeticRejected):
            evaluate_expression(expression)

    def test_non_finite_constants_reject(self):
        with pytest.raises(ArithmeticRejected):
            evaluate_expression("1e400")

    def test_oversized_results_reject(self):
        # Exact arithmetic cannot overflow, so the resource guard is the digit cap.
        with pytest.raises(ArithmeticRejected):
            evaluate_expression("1e308 * 10")

    def test_bool_constant_rejected_even_as_exponent(self):
        with pytest.raises(ArithmeticRejected):
            evaluate_expression("2 ** True")


class TestRendering:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("2 + 2", "2 + 2 = 4"),
            ("quanto é (17 * 9) - 4?", "(17 * 9) - 4 = 149"),
            ("calcule 12.5 / 5", "12.5 / 5 = 2.5"),
            ("what is 10 / 4?", "10 / 4 = 2.5"),
            # Exact decimal, not the binary-float artifact 6172839450617283944.
            ("12345678901234567890 / 2", "12345678901234567890 / 2 = 6172839450617283945"),
            # Exact decimal addition, not 0.30000000000000004.
            ("0.1 + 0.2", "0.1 + 0.2 = 0.3"),
            # Terminating decimal rendered exactly, without misleading rounding.
            ("1 / 100000000", "1 / 100000000 = 0.00000001"),
            # Non-terminating results render as the exact fraction (documented).
            ("1 / 3", "1 / 3 = 1/3"),
        ],
    )
    def test_render_calculation(self, text, expected):
        assert render_calculation(text) == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(4, "4"), (2.5, "2.5"), (149, "149"), (-2, "-2")],
    )
    def test_format_number_int_and_float(self, value, expected):
        assert format_number(value) == expected

    @pytest.mark.parametrize(
        ("numerator", "denominator", "expected"),
        [
            (149, 1, "149"),
            (5, 2, "2.5"),
            (3, 10, "0.3"),
            (1, 10**8, "0.00000001"),
            (-1, 2, "-0.5"),
            (3, 4, "0.75"),
            (1, 3, "1/3"),
            (-2, 3, "-2/3"),
        ],
    )
    def test_format_number_exact_rationals(self, numerator, denominator, expected):
        from fractions import Fraction

        assert format_number(Fraction(numerator, denominator)) == expected
