# IdempotentLaws（幂等律）：`P∧P ⟷ P；P∨P ⟷ P`

import z3
import random
from utils.variable_manager import RuleVariableMixin


class IdempotentLawsRule(RuleVariableMixin):
    """幂等律规则 (Idempotent Laws)"""

    def __init__(self):
        self.name = "IdempotentLaws"
        self.description = "幂等律：P∧P ≡ P；P∨P ≡ P"

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
            raise ValueError("IdempotentLaws需要恰好1个前提")
        expr = premises[0]
        # 如果是合取或析取，寻找重复子项
        if z3.is_and(expr) or z3.is_or(expr):
            children = list(expr.children())
            # 检查前两个是否相等即可体现 P∧P 或 P∨P
            if len(children) >= 2:
                first = children[0]
                second = children[1]
                if self._z3_equal(first, second):
                    return first
        # 其他情况生成新变量
        return self.create_conclusion_variable()

    def generate_premises(self, conclusion_expr, max_premises=1):
        """根据结论反向生成幂等律的前提。"""
        p = conclusion_expr if isinstance(conclusion_expr, z3.BoolRef) else self.create_premise_variable()
        # 随机选择合取或析取并置入两个相同子项
        if random.choice([True, False]):
            if random.choice([True, False]):
                return [z3.And(p, p)]
            else:
                return [z3.And(p, p)]  # 方向无关，都相同
        else:
            if random.choice([True, False]):
                return [z3.Or(p, p)]
            else:
                return [z3.Or(p, p)]

    def explain_step(self, premises, conclusion):
        premise = premises[0]
        return f"IdempotentLaws: 根据幂等律，将 {premise} 简化为 {conclusion}"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "P∧P 等价于 P；P∨P 等价于 P",
            "formal": "P∧P ↔ P；P∨P ↔ P",
            "example": "天在下雨且天在下雨 等价于 天在下雨",
            "variables": ["命题P", "幂等前提", "幂等结论"]
        }