# DominationLaws（支配/恒等式）：`P∨⊤ ⟷ ⊤；P∧⊥ ⟷ ⊥`

import z3
import random
from utils.variable_manager import RuleVariableMixin


class DominationLawsRule(RuleVariableMixin):
    """支配律规则 (Domination Laws)"""

    def __init__(self):
        self.name = "DominationLaws"
        self.description = "支配律：P∧false ≡ false；P∨true ≡ true"

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
            raise ValueError("DominationLaws需要恰好1个前提")
        expr = premises[0]
        # P ∧ false → false
        if z3.is_and(expr):
            for child in expr.children():
                if self._z3_equal(child, z3.BoolVal(False)):
                    return z3.BoolVal(False)
        # P ∨ true → true
        if z3.is_or(expr):
            for child in expr.children():
                if self._z3_equal(child, z3.BoolVal(True)):
                    return z3.BoolVal(True)
        # 不匹配返回新变量
        return self.create_conclusion_variable()

    def generate_premises(self, conclusion_expr, max_premises=1):
        """根据结论生成支配律前提。"""
        # 如果结论是 False，则生成 P ∧ false
        if self._z3_equal(conclusion_expr, z3.BoolVal(False)):
            p = self.create_premise_variable()
            if random.choice([True, False]):
                return [z3.And(p, z3.BoolVal(False))]
            else:
                return [z3.And(z3.BoolVal(False), p)]
        # 如果结论是 True，则生成 P ∨ true
        if self._z3_equal(conclusion_expr, z3.BoolVal(True)):
            p = self.create_premise_variable()
            if random.choice([True, False]):
                return [z3.Or(p, z3.BoolVal(True))]
            else:
                return [z3.Or(z3.BoolVal(True), p)]
        # 否则随机生成一个支配律形式
        p = conclusion_expr if isinstance(conclusion_expr, z3.BoolRef) else self.create_premise_variable()
        if random.choice([True, False]):
            return [z3.And(p, z3.BoolVal(False)) if random.choice([True, False]) else z3.And(z3.BoolVal(False), p)]
        else:
            return [z3.Or(p, z3.BoolVal(True)) if random.choice([True, False]) else z3.Or(z3.BoolVal(True), p)]

    def explain_step(self, premises, conclusion):
        premise = premises[0]
        return f"DominationLaws: 根据支配律，将 {premise} 简化为 {conclusion}"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "P∧false 等价于 false；P∨true 等价于 true",
            "formal": "P∧false ↔ false；P∨true ↔ true",
            "example": "天在下雨且假(总是假) 等价于 假；天在下雨或真(总是真) 等价于 真",
            "variables": ["命题P", "支配前提", "支配结论"]
        }