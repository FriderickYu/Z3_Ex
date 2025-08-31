"""
布尔代数律
交换律
P∧Q ⟷ Q∧P 和 P∨Q ⟷ Q∨P
"""

import z3
import random
from utils.variable_manager import RuleVariableMixin


class CommutativeLawsRule(RuleVariableMixin):
    """交换律规则 (Commutative Laws)"""

    def __init__(self):
        self.name = "CommutativeLaws"
        self.description = "交换律：P ∧ Q ≡ Q ∧ P；P ∨ Q ≡ Q ∨ P"

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
            raise ValueError("CommutativeLaws需要恰好1个前提")
        expr = premises[0]
        # 如果是合取或析取，打乱子项顺序
        if z3.is_and(expr) or z3.is_or(expr):
            children = list(expr.children())
            if len(children) < 2:
                return expr
            random.shuffle(children)
            if z3.is_and(expr):
                return z3.And(*children)
            else:
                return z3.Or(*children)
        # 不匹配时返回新的结论变量
        return self.create_conclusion_variable()

    def generate_premises(self, conclusion_expr, max_premises=1):
        """根据结论生成一个交换顺序的前提。"""
        # 如果结论是合取或析取，重新打乱子项顺序作为前提
        if z3.is_and(conclusion_expr) or z3.is_or(conclusion_expr):
            children = list(conclusion_expr.children())
            if len(children) < 2:
                return [conclusion_expr]
            random.shuffle(children)
            expr = z3.And(*children) if z3.is_and(conclusion_expr) else z3.Or(*children)
            return [expr]
        # 否则随机生成两个命题并构造合取或析取
        a = conclusion_expr if isinstance(conclusion_expr, z3.BoolRef) else self.create_premise_variable()
        b = self.create_premise_variable()
        if random.choice([True, False]):
            expr = z3.And(a, b) if random.choice([True, False]) else z3.And(b, a)
        else:
            expr = z3.Or(a, b) if random.choice([True, False]) else z3.Or(b, a)
        return [expr]

    def explain_step(self, premises, conclusion):
        premise = premises[0]
        return f"CommutativeLaws: 根据交换律，将 {premise} 重排为 {conclusion}"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "P ∧ Q 等价于 Q ∧ P；P ∨ Q 等价于 Q ∨ P",
            "formal": "P ∧ Q ↔ Q ∧ P；P ∨ Q ↔ Q ∨ P",
            "example": "天在下雨且风在刮 等价于 风在刮且天在下雨；天在下雨或风在刮 等价于 风在刮或天在下雨",
            "variables": ["命题P", "命题Q", "交换前提", "交换结论"]
        }