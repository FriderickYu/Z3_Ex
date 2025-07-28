# 文件：distractor/utils.py
# 说明：干扰项生成器通用工具函数，支持变量提取、替换、表达式处理等

import z3
from typing import List, Dict, Set


def extract_variables(expr: z3.ExprRef) -> Set[z3.ExprRef]:
    """
    提取 Z3 表达式中所有的变量（Bool 类型）
    """
    vars = set()

    def _collect(e):
        if z3.is_const(e):
            # 检查是否为布尔常量且不是内置常量 (True/False)
            if z3.is_bool(e) and e.decl().kind() == z3.Z3_OP_UNINTERPRETED:
                vars.add(e)
        else:
            # 递归处理子表达式
            for child in e.children():
                _collect(child)

    _collect(expr)
    return vars


def extract_all_bool_vars(*exprs) -> List[z3.BoolRef]:
    """
    从多个表达式中提取所有布尔变量并去重
    """
    all_vars = set()
    for expr in exprs:
        if expr is not None:
            all_vars.update(extract_variables(expr))

    # 过滤出布尔类型的变量
    bool_vars = [var for var in all_vars if z3.is_bool(var)]
    return bool_vars


def replace_variables(expr: z3.ExprRef, var_mapping: Dict[z3.ExprRef, z3.ExprRef]) -> z3.ExprRef:
    """
    将表达式中的变量替换为映射表中的目标变量
    """
    if z3.is_const(expr):
        return var_mapping.get(expr, expr)
    else:
        new_children = [replace_variables(c, var_mapping) for c in expr.children()]
        return expr.decl()(*new_children)


def flatten_and_expr(expr: z3.ExprRef) -> List[z3.ExprRef]:
    """
    将嵌套的 And 表达式扁平化为 [A, B, C, ...] 形式
    """
    result = []

    def _flatten(e):
        if z3.is_and(e):
            for child in e.children():
                _flatten(child)
        else:
            result.append(e)

    _flatten(expr)
    return result


def flatten_or_expr(expr: z3.ExprRef) -> List[z3.ExprRef]:
    """
    将嵌套的 Or 表达式扁平化为 [A, B, C, ...] 形式
    """
    result = []

    def _flatten(e):
        if z3.is_or(e):
            for child in e.children():
                _flatten(child)
        else:
            result.append(e)

    _flatten(expr)
    return result


def expr_to_str(expr: z3.ExprRef) -> str:
    """
    安全地将 Z3 表达式转换为字符串形式
    """
    if expr is None:
        return "None"

    try:
        return str(expr.sexpr())
    except Exception:
        try:
            return str(expr)
        except Exception:
            return f"<unparseable Z3 expr: {type(expr)}>"


def is_atomic_formula(expr: z3.ExprRef) -> bool:
    """
    判断表达式是否为原子公式（不包含逻辑连接词）
    """
    if z3.is_const(expr):
        return True

    return not (z3.is_and(expr) or z3.is_or(expr) or z3.is_implies(expr) or z3.is_not(expr))


def count_logical_operators(expr: z3.ExprRef) -> Dict[str, int]:
    """
    统计表达式中各种逻辑操作符的数量
    """
    counts = {"and": 0, "or": 0, "not": 0, "implies": 0, "variables": 0}

    def _count(e):
        if z3.is_and(e):
            counts["and"] += 1
        elif z3.is_or(e):
            counts["or"] += 1
        elif z3.is_not(e):
            counts["not"] += 1
        elif z3.is_implies(e):
            counts["implies"] += 1
        elif z3.is_const(e) and z3.is_bool(e):
            counts["variables"] += 1

        for child in e.children():
            _count(child)

    _count(expr)
    return counts