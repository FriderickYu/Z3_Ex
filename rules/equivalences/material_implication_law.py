# MaterialImplication（蕴含等价）：`P→Q ⟷ ¬P∨Q`

import z3
import random
from utils.variable_manager import RuleVariableMixin


class MaterialImplicationLawRule(RuleVariableMixin):
    """材料蕴含律规则 (Material Implication Law)"""

    def __init__(self):
        self.name = "MaterialImplicationLaw"
        self.description = "材料蕴含律：P→Q ≡ ¬P∨Q"

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
            raise ValueError("MaterialImplicationLaw需要恰好1个前提")
        expr = premises[0]
        # 如果前提是蕴含 P→Q
        if z3.is_implies(expr):
            p = expr.arg(0)
            q = expr.arg(1)
            return z3.Or(z3.Not(p), q)
        # 如果前提是 ¬P ∨ Q，则生成 P→Q
        if z3.is_or(expr) and len(expr.children()) == 2:
            a, b = expr.children()
            # a = ¬P, b = Q
            if z3.is_not(a):
                p = a.arg(0)
                q = b
                return z3.Implies(p, q)
            # b = ¬P, a = Q
            if z3.is_not(b):
                p = b.arg(0)
                q = a
                return z3.Implies(p, q)
        # 不匹配生成新变量
        return self.create_conclusion_variable()

    def generate_premises(self, conclusion_expr, max_premises=1):
        """根据结论生成材料蕴含律前提。"""
        # 如果结论是蕴含 P→Q，则生成 ¬P ∨ Q
        if z3.is_implies(conclusion_expr):
            p = conclusion_expr.arg(0)
            q = conclusion_expr.arg(1)
            return [z3.Or(z3.Not(p), q)]
        # 如果结论是 ¬P ∨ Q，则生成 P→Q
        if z3.is_or(conclusion_expr) and len(conclusion_expr.children()) == 2:
            a, b = conclusion_expr.children()
            if z3.is_not(a):
                p = a.arg(0)
                q = b
                return [z3.Implies(p, q)]
            if z3.is_not(b):
                p = b.arg(0)
                q = a
                return [z3.Implies(p, q)]
        # 默认随机生成蕴含或析取形式
        p = conclusion_expr if isinstance(conclusion_expr, z3.BoolRef) else self.create_premise_variable()
        q = self.create_premise_variable()
        if random.choice([True, False]):
            return [z3.Implies(p, q)]
        else:
            return [z3.Or(z3.Not(p), q)]

    def explain_step(self, premises, conclusion):
        premise = premises[0]
        return f"MaterialImplicationLaw: 根据材料蕴含律，将 {premise} 转换为 {conclusion}"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "P→Q 等价于 ¬P∨Q",
            "formal": "P→Q ↔ ¬P∨Q",
            "example": "如果天在下雨蕴含地面湿，则等价于 不天在下雨或地面湿",
            "variables": ["前件P", "后件Q", "蕴含前提", "析取结论"]
        }