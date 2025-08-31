"""
布尔代数律
分配律
P∧(Q∨R) ⟷ (P∧Q)∨(P∧R) 和 P∨(Q∧R) ⟷ (P∨Q)∧(P∨R)
"""

import z3
import random
from utils.variable_manager import RuleVariableMixin


class DistributiveLawsRule(RuleVariableMixin):
    """分配律规则 (Distributive Laws)"""

    def __init__(self):
        self.name = "DistributiveLaws"
        self.description = "分配律：P∧(Q∨R) ≡ (P∧Q)∨(P∧R)；P∨(Q∧R) ≡ (P∨Q)∧(P∨R)"

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
            raise ValueError("DistributiveLaws需要恰好1个前提")
        expr = premises[0]
        # 处理 P ∧ (Q ∨ R)
        if z3.is_and(expr):
            children = list(expr.children())
            if len(children) == 2:
                a, b = children
                # a 是析取 b 不是
                if z3.is_or(a):
                    other = b
                    or_children = list(a.children())
                    if len(or_children) == 2:
                        q, r = or_children
                        return z3.Or(z3.And(other, q), z3.And(other, r))
                # b 是析取 a 不是
                if z3.is_or(b):
                    other = a
                    or_children = list(b.children())
                    if len(or_children) == 2:
                        q, r = or_children
                        return z3.Or(z3.And(other, q), z3.And(other, r))
        # 处理 P ∨ (Q ∧ R)
        if z3.is_or(expr):
            children = list(expr.children())
            if len(children) == 2:
                a, b = children
                if z3.is_and(a):
                    other = b
                    and_children = list(a.children())
                    if len(and_children) == 2:
                        q, r = and_children
                        return z3.And(z3.Or(other, q), z3.Or(other, r))
                if z3.is_and(b):
                    other = a
                    and_children = list(b.children())
                    if len(and_children) == 2:
                        q, r = and_children
                        return z3.And(z3.Or(other, q), z3.Or(other, r))
        # 不匹配则返回新的结论变量
        return self.create_conclusion_variable()

    def _factor_common(self, and1, and2):
        """从两个合取表达式中提取共同因子。如果存在公共因子，则返回 (factor, left, right)，否则返回 None。"""
        a_children = list(and1.children())
        b_children = list(and2.children())
        for p in a_children:
            for q in b_children:
                if self._z3_equal(p, q):
                    # 找到公共因子
                    others_a = [c for c in a_children if not self._z3_equal(c, p)]
                    others_b = [c for c in b_children if not self._z3_equal(c, p)]
                    # 如果没有其他元素，则用 True 占位
                    left = others_a[0] if others_a else z3.BoolVal(True)
                    right = others_b[0] if others_b else z3.BoolVal(True)
                    return p, left, right
        return None

    def _factor_common_or(self, or1, or2):
        """从两个析取表达式中提取公共因子。如果存在公共因子，则返回 (factor, left, right)。"""
        a_children = list(or1.children())
        b_children = list(or2.children())
        for p in a_children:
            for q in b_children:
                if self._z3_equal(p, q):
                    others_a = [c for c in a_children if not self._z3_equal(c, p)]
                    others_b = [c for c in b_children if not self._z3_equal(c, p)]
                    left = others_a[0] if others_a else z3.BoolVal(False)
                    right = others_b[0] if others_b else z3.BoolVal(False)
                    return p, left, right
        return None

    def generate_premises(self, conclusion_expr, max_premises=1):
        """根据结论反向生成分配律的前提。"""
        # (P ∧ Q) ∨ (P ∧ R) → P ∧ (Q ∨ R)
        if z3.is_or(conclusion_expr):
            children = list(conclusion_expr.children())
            if len(children) == 2 and all(z3.is_and(c) for c in children):
                a, b = children
                common = self._factor_common(a, b)
                if common:
                    p, q, r = common
                    return [z3.And(p, z3.Or(q, r))]
        # (P ∨ Q) ∧ (P ∨ R) → P ∨ (Q ∧ R)
        if z3.is_and(conclusion_expr):
            children = list(conclusion_expr.children())
            if len(children) == 2 and all(z3.is_or(c) for c in children):
                a, b = children
                common = self._factor_common_or(a, b)
                if common:
                    p, q, r = common
                    return [z3.Or(p, z3.And(q, r))]
        # 默认随机生成分配律前提
        p = conclusion_expr if isinstance(conclusion_expr, z3.BoolRef) else self.create_premise_variable()
        q = self.create_premise_variable()
        r = self.create_premise_variable()
        # 随机选择合取分配还是析取分配
        if random.choice([True, False]):
            # 生成 P ∧ (Q ∨ R)
            or_expr = z3.Or(q, r) if random.choice([True, False]) else z3.Or(r, q)
            return [z3.And(p, or_expr)]
        else:
            # 生成 P ∨ (Q ∧ R)
            and_expr = z3.And(q, r) if random.choice([True, False]) else z3.And(r, q)
            return [z3.Or(p, and_expr)]

    def explain_step(self, premises, conclusion):
        premise = premises[0]
        return f"DistributiveLaws: 根据分配律，将 {premise} 转换为 {conclusion}"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "P∧(Q∨R) 等价于 (P∧Q)∨(P∧R)；P∨(Q∧R) 等价于 (P∨Q)∧(P∨R)",
            "formal": "P∧(Q∨R) ↔ (P∧Q)∨(P∧R)；P∨(Q∧R) ↔ (P∨Q)∧(P∨R)",
            "example": "天在下雨且(风在刮或天气晴朗) 等价于 (天在下雨且风在刮) 或 (天在下雨且天气晴朗)",
            "variables": ["命题P", "命题Q", "命题R", "分配前提", "分配结论"]
        }