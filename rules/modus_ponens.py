# 文件：rules/modus_ponens.py（更新版）
# 说明：肯定前件推理规则 - 使用统一变量命名

import z3
from utils.variable_manager import RuleVariableMixin


class ModusPonensRule(RuleVariableMixin):
    """
    肯定前件推理规则 (Modus Ponens)
    规则形式：如果有 P 和 P → Q，则可以推出 Q
    """

    def __init__(self):
        self.name = "ModusPonens"
        self.description = "肯定前件：P, P→Q ⊢ Q"

    def num_premises(self):
        return 2

    def can_apply(self, premises):
        if len(premises) != 2:
            return False

        p1, p2 = premises

        # 情况1：p1是P，p2是P→Q
        if z3.is_implies(p2):
            antecedent = p2.arg(0)
            if self._z3_equal(p1, antecedent):
                return True

        # 情况2：p2是P，p1是P→Q
        if z3.is_implies(p1):
            antecedent = p1.arg(0)
            if self._z3_equal(p2, antecedent):
                return True

        return False

    def _z3_equal(self, expr1, expr2):
        try:
            return z3.simplify(expr1) == z3.simplify(expr2)
        except:
            return str(expr1) == str(expr2)

    def construct_conclusion(self, premises):
        if len(premises) != 2:
            raise ValueError("ModusPonens需要恰好2个前提")

        p1, p2 = premises

        # 找到蕴含关系和对应的前件
        if z3.is_implies(p1):
            implication = p1
            premise = p2
        elif z3.is_implies(p2):
            implication = p2
            premise = p1
        else:
            return self.create_conclusion_variable()

        antecedent = implication.arg(0)
        consequent = implication.arg(1)

        if self._z3_equal(premise, antecedent):
            return consequent
        else:
            return self.create_conclusion_variable()

    def generate_premises(self, conclusion_expr, max_premises=2):
        # 创建前提变量
        premise_var = self.create_premise_variable()
        # 构造蕴含关系
        implication = z3.Implies(premise_var, conclusion_expr)
        return [premise_var, implication]

    def explain_step(self, premises, conclusion):
        if len(premises) != 2:
            return f"ModusPonens: 无法解释，前提数量不正确"

        p1, p2 = premises
        if z3.is_implies(p1):
            return f"ModusPonens: 由于有 {p2} 且有 {p1}，因此可以推出 {conclusion}"
        elif z3.is_implies(p2):
            return f"ModusPonens: 由于有 {p1} 且有 {p2}，因此可以推出 {conclusion}"
        else:
            return f"ModusPonens: 基于前提 {premises}，推出 {conclusion}"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "如果 P 成立，并且 P 蕴含 Q，那么 Q 成立",
            "formal": "P, P → Q ⊢ Q",
            "example": "如果天下雨，并且天下雨意味着地面会湿，那么地面会湿",
            "variables": ["前提P", "蕴含关系P→Q", "结论Q"]
        }