"""
命题逻辑规则
否定后件
¬Q, P→Q ⊢ ¬P
"""

import z3
import random
from utils.variable_manager import RuleVariableMixin

class ModusTollensRule(RuleVariableMixin):
    """否定后件规则 (Modus Tollens)

    根据前提 ¬Q 和 P→Q，推出结论 ¬P。
    """

    def __init__(self):
        self.name = "ModusTollens"
        self.description = "否定后件：¬Q, P→Q ⊢ ¬P"

    def num_premises(self):
        """返回该规则需要的前提数量。否定后件需要两个前提。"""
        return 2

    def _z3_equal(self, expr1, expr2):
        """判断两个 Z3 表达式是否等价，若简化失败则退化为字符串比较。"""
        try:
            return z3.simplify(expr1) == z3.simplify(expr2)
        except Exception:
            return str(expr1) == str(expr2)

    def can_apply(self, premises):
        """判断该规则是否可以应用于给定的前提集合。

        条件：恰好两个前提，并且一条是蕴含表达式，另一条是该蕴含后件的否定。
        """
        if len(premises) != 2:
            return False

        p1, p2 = premises

        # 情况1：p1 为蕴含，p2 为否定
        if z3.is_implies(p1) and z3.is_not(p2):
            consequent = p1.arg(1)
            negated = p2.arg(0)
            if self._z3_equal(consequent, negated):
                return True

        # 情况2：p2 为蕴含，p1 为否定
        if z3.is_implies(p2) and z3.is_not(p1):
            consequent = p2.arg(1)
            negated = p1.arg(0)
            if self._z3_equal(consequent, negated):
                return True

        return False

    def construct_conclusion(self, premises):
        """根据前提构造结论。若识别出 ¬Q 和 P→Q 的形式，则返回 ¬P；否则返回新的结论变量的否定。"""
        if len(premises) != 2:
            raise ValueError("ModusTollens需要恰好2个前提")

        p1, p2 = premises

        # 确定哪一个是蕴含，哪一个是否定
        if z3.is_implies(p1) and z3.is_not(p2):
            implication = p1
            negation = p2
        elif z3.is_implies(p2) and z3.is_not(p1):
            implication = p2
            negation = p1
        else:
            # 前提不满足形态，直接生成新的结论变量的否定
            return z3.Not(self.create_conclusion_variable())

        antecedent = implication.arg(0)
        consequent = implication.arg(1)

        # 若否定对象等于蕴含的后件，则结论为否定前件
        if self._z3_equal(negation.arg(0), consequent):
            return z3.Not(antecedent)
        else:
            # 否则生成新的结论变量的否定
            return z3.Not(self.create_conclusion_variable())

    def generate_premises(self, conclusion_expr, max_premises=2):
        """根据给定的结论反向生成前提。

        如果目标结论是 ¬P，则生成 ¬Q 和 P→Q，其中 Q 为新变量；否则随机生成两个变量形成前提。
        """
        # 尝试从结论中提取前件
        if z3.is_not(conclusion_expr):
            antecedent = conclusion_expr.arg(0)
            # 创建一个新的中间变量作为蕴含的后件
            intermediate = self.create_intermediate_variable()
            premise1 = z3.Not(intermediate)
            premise2 = z3.Implies(antecedent, intermediate)
            # 随机打乱前提顺序
            return [premise1, premise2] if random.choice([True, False]) else [premise2, premise1]
        else:
            # 结论不是简单的否定，生成随机前提
            P = self.create_premise_variable()
            Q = self.create_intermediate_variable()
            premise1 = z3.Not(Q)
            premise2 = z3.Implies(P, Q)
            return [premise1, premise2] if random.choice([True, False]) else [premise2, premise1]

    def explain_step(self, premises, conclusion):
        """生成该推理步骤的人类可读解释。"""
        if len(premises) == 2:
            return f"ModusTollens: 由于 {premises[0]} 和 {premises[1]}，可以推出 {conclusion}"
        return f"ModusTollens: 基于前提 {premises}，推出 {conclusion}"

    def get_rule_template(self):
        """返回规则的模板信息，供 UI 或文档使用。"""
        return {
            "name": self.name,
            "pattern": "如果非Q成立，且 P 蕴含 Q，那么非P成立",
            "formal": "¬Q, P→Q ⊢ ¬P",
            "example": "如果地面不湿，并且天下雨意味着地面湿，那么天下不雨",
            "variables": ["命题P", "命题Q", "蕴含关系P→Q", "否定后件¬Q", "结论¬P"]
        }