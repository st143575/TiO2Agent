"""Safe arithmetic for the ``calculate`` tool.

The model writes an expression over named variables ("a / b * 100"); each variable
is bound to a source id (F3, C1, F2.low), so every number in a calculation comes from
the documents, and the formula is kept as evidence. Evaluation walks the AST with a
whitelist of nodes; ``eval`` is never called.
"""

import ast
import math
import operator
import re

MAX_EXPRESSION_CHARS = 300
MAX_EXPONENT = 10

_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _avg(*xs: float) -> float:
    if not xs:
        raise CalculationError("avg() 至少需要一个参数")
    return sum(xs) / len(xs)


FUNCTIONS = {
    "min": min,
    "max": max,
    "abs": abs,
    "round": round,
    "sqrt": math.sqrt,
    "avg": _avg,
    "sum": lambda *xs: sum(xs),
}

# Full-width and typographic operators models sometimes write in Chinese text.
_REPLACEMENTS = {
    "（": "(",
    "）": ")",
    "×": "*",
    "÷": "/",
    "－": "-",
    "＋": "+",
    "，": ",",
    "−": "-",
}
_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class CalculationError(ValueError):
    pass


def normalize_expression(expression: str) -> str:
    for a, b in _REPLACEMENTS.items():
        expression = expression.replace(a, b)
    return expression.strip()


def evaluate(expression: str, variables: dict[str, float] | None = None) -> float:
    """Evaluate ``expression``; names must be in ``variables`` or FUNCTIONS."""
    variables = variables or {}
    expression = normalize_expression(expression)
    if not expression:
        raise CalculationError("表达式为空")
    if len(expression) > MAX_EXPRESSION_CHARS:
        raise CalculationError(f"表达式超过 {MAX_EXPRESSION_CHARS} 个字符")
    for name in variables:
        if not _NAME.match(name) or name in FUNCTIONS:
            raise CalculationError(f"变量名不合法：{name}")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise CalculationError(f"表达式语法错误：{expression}") from exc
    result = _eval(tree.body, variables)
    if isinstance(result, bool) or not isinstance(result, int | float) or not math.isfinite(result):
        raise CalculationError("结果不是有限数值")
    return float(result)


def _eval(node: ast.AST, variables: dict[str, float]) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in variables:
            raise CalculationError(f"未定义的变量：{node.id}（请在 variables 中绑定来源 id）")
        return variables[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _eval(node.left, variables), _eval(node.right, variables)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise CalculationError(f"指数不能超过 {MAX_EXPONENT}")
        try:
            return _BINARY[type(node.op)](left, right)
        except ZeroDivisionError as exc:
            raise CalculationError("除数为 0") from exc
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand, variables))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FUNCTIONS and not node.keywords:
        args = [_eval(a, variables) for a in node.args]
        try:
            return FUNCTIONS[node.func.id](*args)
        except (TypeError, ValueError) as exc:
            raise CalculationError(f"{node.func.id}() 参数不正确：{exc}") from exc
    what = node.func.id if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) else type(node).__name__
    raise CalculationError(f"不支持的表达式成分：{what}（只允许数字、变量、+ - * / ** % 和 {'、'.join(FUNCTIONS)}）")
