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
        if z3.is_const(e) and e.decl().kind() == z3.Z3_OP_UNINTERPRETED:
            vars.add(e)
        else:
            for child in e.children():
                _collect(child)

    _collect(expr)
    return vars


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


def expr_to_str(expr: z3.ExprRef) -> str:
    """
    安全地将 Z3 表达式转换为字符串形式
    """
    try:
        return str(expr.sexpr())
    except Exception:
        return str(expr)