"""
等价规则
德摩根律
¬(P∧Q) ⟷ (¬P∨¬Q) 和 ¬(P∨Q) ⟷ (¬P∧¬Q)
"""

import z3
import random
from utils.variable_manager import RuleVariableMixin


class DeMorganLawsRule(RuleVariableMixin):
    """德摩根律规则 (De Morgan's Laws)"""

    def __init__(self):
        self.name = "DeMorganLaws"
        self.description = "德摩根律：¬(P∧Q) ≡ ¬P∨¬Q；¬(P∨Q) ≡ ¬P∧¬Q"

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
            raise ValueError("DeMorganLaws需要恰好1个前提")
        expr = premises[0]
        # 处理 ¬(P ∧ Q) → ¬P ∨ ¬Q
        if z3.is_not(expr):
            inner = expr.arg(0)
            if z3.is_and(inner):
                children = list(inner.children())
                neg_children = [z3.Not(c) for c in children]
                return z3.Or(*neg_children)
            if z3.is_or(inner):
                children = list(inner.children())
                neg_children = [z3.Not(c) for c in children]
                return z3.And(*neg_children)
        # 处理 ¬P ∨ ¬Q → ¬(P ∧ Q)
        if z3.is_or(expr):
            children = list(expr.children())
            if all(z3.is_not(c) for c in children) and len(children) >= 2:
                inner_children = [c.arg(0) for c in children]
                return z3.Not(z3.And(*inner_children))
        # 处理 ¬P ∧ ¬Q → ¬(P ∨ Q)
        if z3.is_and(expr):
            children = list(expr.children())
            if all(z3.is_not(c) for c in children) and len(children) >= 2:
                inner_children = [c.arg(0) for c in children]
                return z3.Not(z3.Or(*inner_children))
        # 无法匹配则生成新的结论变量
        return self.create_conclusion_variable()

    def generate_premises(self, conclusion_expr, max_premises=1):
        """根据结论反向生成德摩根律的前提。"""
        # 如果结论是 Or 的 Not 子项，返回 Not(And(...))
        if z3.is_or(conclusion_expr):
            children = list(conclusion_expr.children())
            if all(z3.is_not(c) for c in children) and len(children) >= 2:
                inner_children = [c.arg(0) for c in children]
                return [z3.Not(z3.And(*inner_children))]
        # 如果结论是 And 的 Not 子项，返回 Not(Or(...))
        if z3.is_and(conclusion_expr):
            children = list(conclusion_expr.children())
            if all(z3.is_not(c) for c in children) and len(children) >= 2:
                inner_children = [c.arg(0) for c in children]
                return [z3.Not(z3.Or(*inner_children))]
        # 如果结论是 Not(expr)，尝试反转
        if z3.is_not(conclusion_expr):
            inner = conclusion_expr.arg(0)
            if z3.is_and(inner):
                # 结论是 ¬(P ∧ Q)，则可生成 ¬P ∨ ¬Q
                children = list(inner.children())
                neg_children = [z3.Not(c) for c in children]
                return [z3.Or(*neg_children)]
            if z3.is_or(inner):
                # 结论是 ¬(P ∨ Q)，则可生成 ¬P ∧ ¬Q
                children = list(inner.children())
                neg_children = [z3.Not(c) for c in children]
                return [z3.And(*neg_children)]
        # 默认随机生成 ¬(P ∧ Q) 或 ¬(P ∨ Q)
        p = conclusion_expr if isinstance(conclusion_expr, z3.BoolRef) else self.create_premise_variable()
        q = self.create_premise_variable()
        if random.choice([True, False]):
            return [z3.Not(z3.And(p, q))]
        else:
            return [z3.Not(z3.Or(p, q))]

    def explain_step(self, premises, conclusion):
        premise = premises[0]
        return f"DeMorganLaws: 根据德摩根律，将 {premise} 转换为 {conclusion}"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "¬(P∧Q) 等价于 ¬P∨¬Q；¬(P∨Q) 等价于 ¬P∧¬Q",
            "formal": "¬(P∧Q) ↔ ¬P∨¬Q；¬(P∨Q) ↔ ¬P∧¬Q",
            "example": "不(天在下雨且风在刮) 等价于 (不天在下雨)或(不风在刮)",
            "variables": ["命题P", "命题Q", "德摩根前提", "德摩根结论"]
        }