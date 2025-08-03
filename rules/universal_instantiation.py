# 文件：rules/universal_instantiation.py
# 说明：全称量词实例化规则
# UniversalInstantiation: ∀x P(x) ⊢ P(a)

import z3
from utils.variable_manager import RuleVariableMixin


class UniversalInstantiationRule(RuleVariableMixin):
    """全称量词实例化规则 (Universal Instantiation)"""

    def __init__(self):
        self.name = "UniversalInstantiation"
        self.description = "全称实例化：∀x P(x) ⊢ P(a)"

    def num_premises(self):
        return 1

    def can_apply(self, premises):
        if len(premises) != 1:
            return False

        premise = premises[0]
        return z3.is_quantifier(premise) and premise.is_forall() if hasattr(premise, 'is_forall') else self._is_universal_like(premise)

    def _is_universal_like(self, expr):
        expr_str = str(expr).lower()
        return 'forall' in expr_str or 'all' in expr_str

    def construct_conclusion(self, premises):
        if len(premises) != 1:
            raise ValueError("UniversalInstantiation需要恰好1个前提")

        return self.create_conclusion_variable()

    def generate_premises(self, conclusion_expr, max_premises=1):
        universal_var = self.create_premise_variable()
        return [universal_var]

    def explain_step(self, premises, conclusion):
        premise = premises[0]
        return f"UniversalInstantiation: 由于 {premise} 对所有情况成立，因此 {conclusion} 对特定情况也成立"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "如果对所有x都有P(x)，那么对特定的a有P(a)",
            "formal": "∀x P(x) ⊢ P(a)",
            "example": "如果所有人都会死，那么苏格拉底会死",
            "variables": ["全称量词∀x", "谓词P(x)", "特定实例a"]
        }