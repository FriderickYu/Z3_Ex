"""
等价规则
双重否定律
¬¬P ⟷ P
"""

import z3
import random
from utils.variable_manager import RuleVariableMixin


class DoubleNegationRule(RuleVariableMixin):
    """双重否定规则 (Double Negation)"""

    def __init__(self):
        self.name = "DoubleNegation"
        self.description = "双重否定律：¬¬P ≡ P"

    def num_premises(self):
        return 1

    def _z3_equal(self, expr1, expr2):
        try:
            return z3.simplify(expr1) == z3.simplify(expr2)
        except Exception:
            return str(expr1) == str(expr2)

    def can_apply(self, premises):
        return len(premises) == 1

    def construct_conclusion(self, premises):
        if len(premises) != 1:
            raise ValueError("DoubleNegation需要恰好1个前提")
        expr = premises[0]
        # 如果是 ¬¬P，则返回 P
        if z3.is_not(expr):
            inner = expr.arg(0)
            if z3.is_not(inner):
                return inner.arg(0)
        # 如果前提是布尔表达式但不是双重否定，则生成 ¬¬expr
        if isinstance(expr, z3.BoolRef):
            return z3.Not(z3.Not(expr))
        # 否则生成新变量
        return self.create_conclusion_variable()

    def generate_premises(self, conclusion_expr, max_premises=1):
        """根据结论生成双重否定前提。"""
        # 如果结论是双重否定形式，则返回内部命题
        if z3.is_not(conclusion_expr) and z3.is_not(conclusion_expr.arg(0)):
            return [conclusion_expr.arg(0).arg(0)]
        # 如果结论是普通布尔表达式，则生成其双重否定
        if isinstance(conclusion_expr, z3.BoolRef):
            return [z3.Not(z3.Not(conclusion_expr))]
        # 否则随机生成双重否定
        p = self.create_premise_variable()
        return [z3.Not(z3.Not(p))]

    def explain_step(self, premises, conclusion):
        premise = premises[0]
        return f"DoubleNegation: 根据双重否定律，将 {premise} 转换为 {conclusion}"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "¬¬P 等价于 P",
            "formal": "¬¬P ↔ P",
            "example": "不不天下雨 等价于 天下雨",
            "variables": ["命题P", "双重否定前提", "双重否定结论"]
        }