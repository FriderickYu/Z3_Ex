# Contraposition（对当等价）：`P→Q ⟷ ¬Q→¬P`

import z3
import random
from utils.variable_manager import RuleVariableMixin


class ContrapositionEquivRule(RuleVariableMixin):
    """逆否等价规则 (Contraposition Equivalence)"""

    def __init__(self):
        self.name = "ContrapositionEquiv"
        self.description = "逆否等价：P→Q ≡ ¬Q→¬P"

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
            raise ValueError("ContrapositionEquiv需要恰好1个前提")
        premise = premises[0]
        # 如果前提是蕴含 P→Q
        if z3.is_implies(premise):
            antecedent = premise.arg(0)
            consequent = premise.arg(1)
            return z3.Implies(z3.Not(consequent), z3.Not(antecedent))
        # 如果前提是 ¬Q→¬P，则返回 P→Q
        if z3.is_implies(premise) and z3.is_not(premise.arg(0)) and z3.is_not(premise.arg(1)):
            q_neg = premise.arg(0).arg(0)
            p_neg = premise.arg(1).arg(0)
            return z3.Implies(p_neg, q_neg)
        # 否则生成随机蕴含
        p = self.create_premise_variable()
        q = self.create_premise_variable()
        return z3.Implies(p, q)

    def generate_premises(self, conclusion_expr, max_premises=1):
        """根据结论生成逆否等价的前提。"""
        # 如果结论是蕴含，则返回其逆否表达式
        if z3.is_implies(conclusion_expr):
            antecedent = conclusion_expr.arg(0)
            consequent = conclusion_expr.arg(1)
            return [z3.Implies(z3.Not(consequent), z3.Not(antecedent))]
        # 如果结论是逆否形式，则返回原蕴含
        if z3.is_implies(conclusion_expr) and z3.is_not(conclusion_expr.arg(0)) and z3.is_not(conclusion_expr.arg(1)):
            q_neg = conclusion_expr.arg(0).arg(0)
            p_neg = conclusion_expr.arg(1).arg(0)
            return [z3.Implies(p_neg, q_neg)]
        # 否则生成随机蕴含
        p = conclusion_expr if isinstance(conclusion_expr, z3.BoolRef) else self.create_premise_variable()
        q = self.create_premise_variable()
        return [z3.Implies(p, q)]

    def explain_step(self, premises, conclusion):
        premise = premises[0]
        return f"ContrapositionEquiv: 根据逆否等价，将 {premise} 转换为 {conclusion}"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "P→Q 等价于 ¬Q→¬P",
            "formal": "P→Q ↔ ¬Q→¬P",
            "example": "如果天下雨蕴含地面湿，那么地面不湿蕴含天下不雨",
            "variables": ["前件P", "后件Q", "蕴含关系P→Q", "逆否形式¬Q→¬P"]
        }