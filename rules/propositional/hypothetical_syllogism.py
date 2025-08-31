"""
命题逻辑规则
假言三段论规则
P → Q, Q → R ⊢ P → R
"""

import z3
from utils.variable_manager import RuleVariableMixin


class HypotheticalSyllogismRule(RuleVariableMixin):
    """假言三段论规则 (Hypothetical Syllogism)

    根据前提 P→Q 和 Q→R 推出结论 P→R。
    此实现为现有规则的拷贝。
    """

    def __init__(self):
        self.name = "HypotheticalSyllogism"
        self.description = "假言三段论：P → Q, Q → R ⊢ P → R"

    def num_premises(self):
        return 2

    def _z3_equal(self, expr1, expr2):
        try:
            return z3.simplify(expr1) == z3.simplify(expr2)
        except Exception:
            return str(expr1) == str(expr2)

    def can_apply(self, premises):
        if len(premises) != 2:
            return False
        p1, p2 = premises
        if not (z3.is_implies(p1) and z3.is_implies(p2)):
            return False
        antecedent1 = p1.arg(0)
        consequent1 = p1.arg(1)
        antecedent2 = p2.arg(0)
        consequent2 = p2.arg(1)
        # 检查传递性连接
        if self._z3_equal(consequent1, antecedent2):
            return True
        if self._z3_equal(consequent2, antecedent1):
            return True
        return False

    def construct_conclusion(self, premises):
        if len(premises) != 2:
            raise ValueError("HypotheticalSyllogism需要恰好2个前提")
        p1, p2 = premises
        if not (z3.is_implies(p1) and z3.is_implies(p2)):
            return self._create_new_implication()
        antecedent1 = p1.arg(0)
        consequent1 = p1.arg(1)
        antecedent2 = p2.arg(0)
        consequent2 = p2.arg(1)
        # 情况1：p1: P → Q, p2: Q → R，结论：P → R
        if self._z3_equal(consequent1, antecedent2):
            return z3.Implies(antecedent1, consequent2)
        # 情况2：p1: Q → R, p2: P → Q，结论：P → R
        if self._z3_equal(consequent2, antecedent1):
            return z3.Implies(antecedent2, consequent1)
        return self._create_new_implication()

    def _create_new_implication(self):
        antecedent = self.create_premise_variable()
        consequent = self.create_conclusion_variable()
        return z3.Implies(antecedent, consequent)

    def generate_premises(self, conclusion_expr, max_premises=2):
        if not z3.is_implies(conclusion_expr):
            return self._generate_random_premises()
        antecedent = conclusion_expr.arg(0)
        consequent = conclusion_expr.arg(1)
        # 创建中间变量
        intermediate_var = self.create_intermediate_variable()
        # 构造前提：P → Q 和 Q → R
        premise1 = z3.Implies(antecedent, intermediate_var)
        premise2 = z3.Implies(intermediate_var, consequent)
        return [premise1, premise2]

    def _generate_random_premises(self):
        P = self.create_premise_variable()
        Q = self.create_intermediate_variable()
        R = self.create_conclusion_variable()
        return [z3.Implies(P, Q), z3.Implies(Q, R)]

    def explain_step(self, premises, conclusion):
        if len(premises) == 2:
            p1, p2 = premises
            return f"HypotheticalSyllogism: 由于有 {p1} 且有 {p2}，通过传递性可以推出 {conclusion}"
        else:
            return f"HypotheticalSyllogism: 基于前提 {premises}，通过传递性推出 {conclusion}"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "如果 P 蕴含 Q，且 Q 蕴含 R，那么 P 蕴含 R",
            "formal": "P → Q, Q → R ⊢ P → R",
            "example": "如果天下雨就会积水，积水就会影响交通，那么天下雨就会影响交通",
            "variables": ["前提P", "中间项Q", "结论R", "蕴含关系"],
            "logical_property": "传递性"
        }