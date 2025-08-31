"""
布尔代数律
后件强化
P→Q ⊢ P→(P∧Q)
"""

import z3
from utils.variable_manager import RuleVariableMixin

class ConsequentStrengtheningRule(RuleVariableMixin):
    """后件强化规则 (Consequent Strengthening)

    根据前提 P→Q，推出结论 P→(P∧Q)。
    """

    def __init__(self):
        self.name = "ConsequentStrengthening"
        self.description = "后件强化：P→Q ⊢ P→(P∧Q)"

    def num_premises(self):
        """后件强化仅需要一个前提。"""
        return 1

    def _z3_equal(self, expr1, expr2):
        try:
            return z3.simplify(expr1) == z3.simplify(expr2)
        except Exception:
            return str(expr1) == str(expr2)

    def can_apply(self, premises):
        """判断该规则是否可以应用。

        条件：恰好一个前提且该前提是蕴含表达式。
        """
        return len(premises) == 1 and z3.is_implies(premises[0])

    def construct_conclusion(self, premises):
        """根据前提构造结论。

        如果前提是 P→Q，则结论为 P→(P∧Q)；否则生成新的蕴含表达式。
        """
        if len(premises) != 1:
            raise ValueError("ConsequentStrengthening需要恰好1个前提")

        premise = premises[0]
        if z3.is_implies(premise):
            antecedent = premise.arg(0)
            consequent = premise.arg(1)
            # 后件强化：P→Q 推出 P→(P∧Q)
            return z3.Implies(antecedent, z3.And(antecedent, consequent))
        else:
            # 如果前提不是蕴含，生成新的随机蕴含表达式
            antecedent = self.create_premise_variable()
            conclusion_part = z3.And(self.create_premise_variable(), self.create_conclusion_variable())
            return z3.Implies(antecedent, conclusion_part)

    def generate_premises(self, conclusion_expr, max_premises=1):
        """根据期望的结论反向生成前提。

        如果结论形如 P→(P∧Q)，则生成 P→Q；否则生成随机的蕴含。"""
        # 检查结论是否为蕴含
        if z3.is_implies(conclusion_expr):
            antecedent = conclusion_expr.arg(0)
            consequent = conclusion_expr.arg(1)
            # 尝试识别后件是否为合取且包含前件
            if z3.is_and(consequent):
                children = list(consequent.children())
                q_part = None
                for child in children:
                    if not self._z3_equal(child, antecedent):
                        q_part = child
                        break
                if q_part is None and children:
                    q_part = children[0]
                if q_part is not None:
                    return [z3.Implies(antecedent, q_part)]
        # 默认生成随机蕴含作为前提
        P = self.create_premise_variable()
        Q = self.create_conclusion_variable()
        return [z3.Implies(P, Q)]

    def explain_step(self, premises, conclusion):
        """生成该推理步骤的人类可读解释。"""
        if premises:
            return f"ConsequentStrengthening: 由于 {premises[0]}，因此可以推出 {conclusion}（后件强化）"
        return f"ConsequentStrengthening: 基于前提 {premises}，推出 {conclusion}"

    def get_rule_template(self):
        """返回规则的模板信息。"""
        return {
            "name": self.name,
            "pattern": "如果 P 蕴含 Q，那么 P 蕴含 (P 且 Q)",
            "formal": "P→Q ⊢ P→(P∧Q)",
            "example": "如果天下雨意味着地面湿，那么天下雨意味着（天下雨且地面湿）",
            "variables": ["前件P", "后件Q", "蕴含关系P→Q", "强化结论P→(P∧Q)"]
        }