"""
命题逻辑规则
析取三段论
P∨Q, ¬P ⊢ Q
"""

import z3
import random
from utils.variable_manager import RuleVariableMixin


class DisjunctiveSyllogismRule(RuleVariableMixin):
    """析取三段论规则 (Disjunctive Syllogism)

    根据前提 P∨Q 和 ¬P，推出 Q。
    """

    def __init__(self):
        self.name = "DisjunctiveSyllogism"
        self.description = "析取三段论：P∨Q, ¬P ⊢ Q"

    def num_premises(self):
        """返回该规则需要的前提数量，析取消除需要两个前提。"""
        return 2

    def _z3_equal(self, expr1, expr2):
        try:
            return z3.simplify(expr1) == z3.simplify(expr2)
        except Exception:
            return str(expr1) == str(expr2)

    def can_apply(self, premises):
        """判断该规则是否可以应用。

        条件：两条前提，其中一条是析取表达式，另一条是该析取某一项的否定。
        """
        if len(premises) != 2:
            return False

        p1, p2 = premises

        # p1 为析取，p2 为否定
        if z3.is_or(p1) and z3.is_not(p2):
            disjuncts = list(p1.children())
            for d in disjuncts:
                if self._z3_equal(d, p2.arg(0)):
                    return True

        # p2 为析取，p1 为否定
        if z3.is_or(p2) and z3.is_not(p1):
            disjuncts = list(p2.children())
            for d in disjuncts:
                if self._z3_equal(d, p1.arg(0)):
                    return True

        return False

    def construct_conclusion(self, premises):
        """根据前提构造结论。如果存在某项被否定的析取，则返回另一项作为结论。"""
        if len(premises) != 2:
            raise ValueError("DisjunctiveSyllogism需要恰好2个前提")

        p1, p2 = premises

        # 确定析取表达式与否定表达式
        if z3.is_or(p1) and z3.is_not(p2):
            or_expr = p1
            neg_expr = p2
        elif z3.is_or(p2) and z3.is_not(p1):
            or_expr = p2
            neg_expr = p1
        else:
            # 模式不匹配时返回新的结论变量
            return self.create_conclusion_variable()

        disjuncts = list(or_expr.children())
        negated = neg_expr.arg(0)
        # 寻找与否定项不同的析取项
        others = [d for d in disjuncts if not self._z3_equal(d, negated)]

        if others:
            # 如果有多个其他项，随机选择其中之一作为结论
            return random.choice(others) if len(others) > 1 else others[0]
        else:
            # 理论上不会只有一项析取，此处为保险处理
            return self.create_conclusion_variable()

    def generate_premises(self, conclusion_expr, max_premises=2):
        """根据期望的结论反向生成前提。

        生成的前提包括一个包含结论与新变量的析取表达式，以及该新变量的否定。
        """
        # 创建与结论不同的新变量作为被排除项
        other = self.create_premise_variable()
        # 尽量避免与结论变量重名
        if isinstance(conclusion_expr, z3.BoolRef) and self._z3_equal(other, conclusion_expr):
            other = self.create_premise_variable()
        # 构造析取表达式（顺序随机）
        or_expr = z3.Or(conclusion_expr, other) if random.choice([True, False]) else z3.Or(other, conclusion_expr)
        not_expr = z3.Not(other)
        # 随机决定前提顺序
        return [or_expr, not_expr] if random.choice([True, False]) else [not_expr, or_expr]

    def explain_step(self, premises, conclusion):
        """生成该推理步骤的人类可读解释。"""
        if len(premises) == 2:
            return f"DisjunctiveSyllogism: 由于 {premises[0]} 和 {premises[1]}，因此可得 {conclusion}"
        return f"DisjunctiveSyllogism: 基于前提 {premises} 推出 {conclusion}"

    def get_rule_template(self):
        """返回规则模板信息。"""
        return {
            "name": self.name,
            "pattern": "如果 P 或 Q 成立，并且非P成立，那么 Q 成立",
            "formal": "P∨Q, ¬P ⊢ Q",
            "example": "如果天在下雨或者天气晴朗，并且天不在下雨，那么天气晴朗",
            "variables": ["析取项P", "析取项Q", "否定项¬P", "结论Q"]
        }