"""Safe arithmetic for the calculate tool."""

import math

import pytest

from tio2agent.agent import CalculationError, evaluate


@pytest.mark.parametrize(
    ("expression", "variables", "expected"),
    [
        ("a / b * 100", {"a": 65, "b": 167}, 65 / 167 * 100),
        ("(a - b) / b", {"a": 1.46, "b": 1.21}, (1.46 - 1.21) / 1.21),
        ("（a－b）×2÷4", {"a": 3, "b": 1}, 1.0),          # full-width operators
        ("avg(a, b, c)", {"a": 1.21, "b": 1.35, "c": 1.46}, (1.21 + 1.35 + 1.46) / 3),
        ("max(a, b) - min(a, b)", {"a": 20, "b": 25}, 5),
        ("round(a / b, 3)", {"a": 45, "b": 55}, 0.818),
        ("sqrt(a) ** 2", {"a": 2}, 2.0),
        ("-a + 10 % 3", {"a": 1}, 0),
        ("2.5", {}, 2.5),
    ],
)
def test_evaluate(expression: str, variables: dict, expected: float) -> None:
    assert math.isclose(evaluate(expression, variables), expected, rel_tol=1e-9)


@pytest.mark.parametrize(
    ("expression", "message"),
    [
        ("__import__('os').system('ls')", "不支持"),
        ("a.__class__", "不支持"),
        ("[1, 2]", "不支持"),
        ("x + 1", "未定义的变量：x"),
        ("1 / 0", "除数为 0"),
        ("2 ** 100", "指数不能超过"),
        ("1 +", "语法错误"),
        ("", "表达式为空"),
        ("1" * 400, "超过"),
        ("True + 1", "不支持"),
        ("min()", "参数不正确"),
    ],
)
def test_rejects(expression: str, message: str) -> None:
    with pytest.raises(CalculationError, match=message):
        evaluate(expression, {"a": 1})


def test_variable_names_are_checked() -> None:
    with pytest.raises(CalculationError, match="变量名不合法"):
        evaluate("1", {"max": 1})
    with pytest.raises(CalculationError, match="变量名不合法"):
        evaluate("1", {"F3.low": 1})
