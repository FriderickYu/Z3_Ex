"""
命题逻辑规则
合取消除
P∧Q ⊢ P 或 P∧Q ⊢ Q
"""

import z3
import random
from utils.variable_manager import RuleVariableMixin


class ConjunctionEliminationRule(RuleVariableMixin):
    """合取消除规则 (Conjunction Elimination)

    根据前提 P ∧ Q 推出 P（或 Q）。此实现与已有模块保持一致。
    """

    def __init__(self):
        self.name = "ConjunctionElimination"
        self.description = "合取消除：P ∧ Q ⊢ P"

    def num_premises(self):
        return 1

    def can_apply(self, premises):
        if len(premises) != 1:
            return False
        return z3.is_and(premises[0])

    def construct_conclusion(self, premises):
        if len(premises) != 1:
            raise ValueError("ConjunctionElimination需要恰好1个前提")
        premise = premises[0]
        if not z3.is_and(premise):
            # 如果前提不是合取，则生成新的结论变量
            return self.create_conclusion_variable()
        conjuncts = list(premise.children())
        if conjuncts:
            return random.choice(conjuncts)
        else:
            return premise.arg(0) if random.choice([True, False]) else premise.arg(1)

    def generate_premises(self, conclusion_expr, max_premises=1):
        num_additional = random.randint(1, 3)
        additional_terms = []
        for _ in range(num_additional):
            additional_terms.append(self.create_premise_variable())
        all_terms = [conclusion_expr] + additional_terms
        conjunction = z3.And(*all_terms)
        return [conjunction]

    def explain_step(self, premises, conclusion):
        premise = premises[0]
        return f"ConjunctionElimination: 由于 {premise} 成立，其中包含 {conclusion}，因此 {conclusion} 成立"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "如果 P 且 Q 成立，那么 P 成立（或 Q 成立）",
            "formal": "P ∧ Q ⊢ P",
            "example": "如果天在下雨且风在刮，那么天在下雨",
            "variables": ["合取前提P∧Q", "结论P"]
        }