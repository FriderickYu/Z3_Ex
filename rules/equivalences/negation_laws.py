# NegationLaws（排中/矛盾）：`P∨¬P ⟷ ⊤；P∧¬P ⟷ ⊥`

import z3
import random
from utils.variable_manager import RuleVariableMixin


class NegationLawsRule(RuleVariableMixin):
    """否定律规则 (Negation Laws)"""

    def __init__(self):
        self.name = "NegationLaws"
        self.description = "否定律：P∧¬P ≡ false；P∨¬P ≡ true"

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
            raise ValueError("NegationLaws需要恰好1个前提")
        expr = premises[0]
        # P ∧ ¬P 或 ¬P ∧ P → false
        if z3.is_and(expr) and len(expr.children()) == 2:
            a, b = expr.children()
            if z3.is_not(a) and self._z3_equal(a.arg(0), b):
                return z3.BoolVal(False)
            if z3.is_not(b) and self._z3_equal(b.arg(0), a):
                return z3.BoolVal(False)
        # P ∨ ¬P 或 ¬P ∨ P → true
        if z3.is_or(expr) and len(expr.children()) == 2:
            a, b = expr.children()
            if z3.is_not(a) and self._z3_equal(a.arg(0), b):
                return z3.BoolVal(True)
            if z3.is_not(b) and self._z3_equal(b.arg(0), a):
                return z3.BoolVal(True)
        # 不匹配生成新变量
        return self.create_conclusion_variable()

    def generate_premises(self, conclusion_expr, max_premises=1):
        """根据结论生成否定律前提。"""
        # 如果结论为 false，则生成 P ∧ ¬P
        if self._z3_equal(conclusion_expr, z3.BoolVal(False)):
            p = self.create_premise_variable()
            if random.choice([True, False]):
                return [z3.And(p, z3.Not(p))]
            else:
                return [z3.And(z3.Not(p), p)]
        # 如果结论为 true，则生成 P ∨ ¬P
        if self._z3_equal(conclusion_expr, z3.BoolVal(True)):
            p = self.create_premise_variable()
            if random.choice([True, False]):
                return [z3.Or(p, z3.Not(p))]
            else:
                return [z3.Or(z3.Not(p), p)]
        # 否则随机生成否定律前提
        p = conclusion_expr if isinstance(conclusion_expr, z3.BoolRef) else self.create_premise_variable()
        if random.choice([True, False]):
            return [z3.And(p, z3.Not(p)) if random.choice([True, False]) else z3.And(z3.Not(p), p)]
        else:
            return [z3.Or(p, z3.Not(p)) if random.choice([True, False]) else z3.Or(z3.Not(p), p)]

    def explain_step(self, premises, conclusion):
        premise = premises[0]
        return f"NegationLaws: 根据否定律，将 {premise} 简化为 {conclusion}"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "P∧¬P 等价于 false；P∨¬P 等价于 true",
            "formal": "P∧¬P ↔ false；P∨¬P ↔ true",
            "example": "天在下雨且不天在下雨 等价于 假；天在下雨或不天在下雨 等价于 真",
            "variables": ["命题P", "否定前提", "否定结论"]
        }