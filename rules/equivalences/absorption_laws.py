# AbsorptionLaws（吸收律）：`P∨(P∧Q) ⟷ P；P∧(P∨Q) ⟷ P`

import z3
import random
from utils.variable_manager import RuleVariableMixin


class AbsorptionLawsRule(RuleVariableMixin):
    """吸收律规则 (Absorption Laws)

    根据前提 P ∧ (P ∨ Q) 或 P ∨ (P ∧ Q)，推出结论 P。
    """

    def __init__(self):
        self.name = "AbsorptionLaws"
        self.description = "吸收律：P ∧ (P ∨ Q) ≡ P；P ∨ (P ∧ Q) ≡ P"

    def num_premises(self):
        return 1

    def _z3_equal(self, expr1, expr2):
        """判断两个 Z3 布尔表达式是否等价。"""
        try:
            return z3.simplify(expr1) == z3.simplify(expr2)
        except Exception:
            return str(expr1) == str(expr2)

    def _find_absorption(self, expr):
        """尝试从表达式中识别吸收律结构，并返回可吸收的公共部分。

        如果 expr 为 ``A ∧ (A ∨ B)`` 或 ``(A ∨ B) ∧ A``，则返回 ``A``；
        如果 expr 为 ``A ∨ (A ∧ B)`` 或 ``(A ∧ B) ∨ A``，则返回 ``A``。
        若无匹配，则返回 None。
        """
        # P ∧ (P ∨ Q)
        if z3.is_and(expr):
            children = list(expr.children())
            if len(children) == 2:
                left, right = children
                # 左侧是析取，右侧是公共元素
                if z3.is_or(left):
                    for child in left.children():
                        if self._z3_equal(child, right):
                            return right
                # 右侧是析取，左侧是公共元素
                if z3.is_or(right):
                    for child in right.children():
                        if self._z3_equal(child, left):
                            return left
        # P ∨ (P ∧ Q)
        if z3.is_or(expr):
            children = list(expr.children())
            if len(children) == 2:
                left, right = children
                # 左侧是合取，右侧是公共元素
                if z3.is_and(left):
                    for child in left.children():
                        if self._z3_equal(child, right):
                            return right
                # 右侧是合取，左侧是公共元素
                if z3.is_and(right):
                    for child in right.children():
                        if self._z3_equal(child, left):
                            return left
        return None

    def can_apply(self, premises):
        return len(premises) == 1

    def construct_conclusion(self, premises):
        if len(premises) != 1:
            raise ValueError("AbsorptionLaws需要恰好1个前提")
        premise = premises[0]
        result = self._find_absorption(premise)
        if result is not None:
            return result
        # 无法匹配吸收律则返回新的结论变量
        return self.create_conclusion_variable()

    def generate_premises(self, conclusion_expr, max_premises=1):
        """根据结论反向生成满足吸收律的前提。

        生成的前提形如 ``P ∧ (P ∨ Q)`` 或 ``P ∨ (P ∧ Q)``。
        """
        # 取结论作为公共部分 P
        p = conclusion_expr if isinstance(conclusion_expr, z3.BoolRef) else self.create_premise_variable()
        # 新变量 Q 不能与 P 相同
        q = self.create_premise_variable()
        if self._z3_equal(p, q):
            q = self.create_premise_variable()
        # 随机选择使用合取-析取结构还是析取-合取结构
        if random.choice([True, False]):
            # 构造 P ∧ (P ∨ Q)，顺序随机
            or_expr = z3.Or(p, q) if random.choice([True, False]) else z3.Or(q, p)
            and_expr = z3.And(p, or_expr) if random.choice([True, False]) else z3.And(or_expr, p)
            return [and_expr]
        else:
            # 构造 P ∨ (P ∧ Q)，顺序随机
            and_inner = z3.And(p, q) if random.choice([True, False]) else z3.And(q, p)
            or_expr = z3.Or(p, and_inner) if random.choice([True, False]) else z3.Or(and_inner, p)
            return [or_expr]

    def explain_step(self, premises, conclusion):
        premise = premises[0]
        return f"AbsorptionLaws: 根据吸收律，将 {premise} 简化为 {conclusion}"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "P ∧ (P ∨ Q) 等价于 P；P ∨ (P ∧ Q) 等价于 P",
            "formal": "P ∧ (P ∨ Q) ↔ P；P ∨ (P ∧ Q) ↔ P",
            "example": "如果天在下雨且（天在下雨或天气晴朗），则等价于天在下雨；类似地，天在下雨或（天在下雨且天气晴朗）也等价于天在下雨",
            "variables": ["命题P", "命题Q", "吸收前提P∧(P∨Q)/P∨(P∧Q)", "吸收结论P"]
        }