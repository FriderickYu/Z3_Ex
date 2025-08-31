"""
布尔代数律
同一律
P∧⊤ ⟷ P 和 P∨⊥ ⟷ P
"""

import z3
import random
from utils.variable_manager import RuleVariableMixin


class IdentityLawsRule(RuleVariableMixin):
    """同一律规则 (Identity Laws)"""

    def __init__(self):
        self.name = "IdentityLaws"
        self.description = "同一律：P∧true ≡ P；P∨false ≡ P"

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
            raise ValueError("IdentityLaws需要恰好1个前提")
        expr = premises[0]
        # P ∧ true → 去掉 true
        if z3.is_and(expr):
            children = list(expr.children())
            # 过滤掉 True 子项
            others = [c for c in children if not self._z3_equal(c, z3.BoolVal(True))]
            if len(others) < len(children):
                if not others:
                    # 如果全部是 True，返回 True
                    return z3.BoolVal(True)
                if len(others) == 1:
                    return others[0]
                else:
                    return z3.And(*others)
        # P ∨ false → 去掉 false
        if z3.is_or(expr):
            children = list(expr.children())
            others = [c for c in children if not self._z3_equal(c, z3.BoolVal(False))]
            if len(others) < len(children):
                if not others:
                    # 如果全部是 False，返回 False
                    return z3.BoolVal(False)
                if len(others) == 1:
                    return others[0]
                else:
                    return z3.Or(*others)
        # 不匹配返回新变量
        return self.create_conclusion_variable()

    def generate_premises(self, conclusion_expr, max_premises=1):
        """根据结论生成同一律前提。"""
        p = conclusion_expr if isinstance(conclusion_expr, z3.BoolRef) else self.create_premise_variable()
        # 随机选择合取或析取
        if random.choice([True, False]):
            # P ∧ true
            if random.choice([True, False]):
                return [z3.And(p, z3.BoolVal(True))]
            else:
                return [z3.And(z3.BoolVal(True), p)]
        else:
            # P ∨ false
            if random.choice([True, False]):
                return [z3.Or(p, z3.BoolVal(False))]
            else:
                return [z3.Or(z3.BoolVal(False), p)]

    def explain_step(self, premises, conclusion):
        premise = premises[0]
        return f"IdentityLaws: 根据同一律，将 {premise} 转换为 {conclusion}"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "P∧true 等价于 P；P∨false 等价于 P",
            "formal": "P∧true ↔ P；P∨false ↔ P",
            "example": "天在下雨且真(总是真) 等价于 天在下雨；天在下雨或假(总是假) 等价于 天在下雨",
            "variables": ["命题P", "同一律前提", "同一律结论"]
        }