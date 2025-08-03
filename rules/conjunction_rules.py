import z3
import random
from utils.variable_manager import RuleVariableMixin


class ConjunctionIntroductionRule(RuleVariableMixin):
    """合取引入规则 (Conjunction Introduction)"""

    def __init__(self):
        self.name = "ConjunctionIntroduction"
        self.description = "合取引入：P, Q ⊢ P ∧ Q"

    def num_premises(self):
        return random.randint(2, 4)

    def can_apply(self, premises):
        return len(premises) >= 2

    def construct_conclusion(self, premises):
        if len(premises) < 2:
            raise ValueError("ConjunctionIntroduction需要至少2个前提")

        if len(premises) == 2:
            return z3.And(premises[0], premises[1])
        else:
            return z3.And(*premises)

    def generate_premises(self, conclusion_expr, max_premises=4):
        if z3.is_and(conclusion_expr):
            return list(conclusion_expr.children())
        else:
            num_premises = random.randint(2, min(max_premises, 3))
            premises = [conclusion_expr]

            for i in range(num_premises - 1):
                premises.append(self.create_premise_variable())

            return premises

    def explain_step(self, premises, conclusion):
        premise_strs = [str(p) for p in premises]
        return f"ConjunctionIntroduction: 由于 {' 和 '.join(premise_strs)} 都成立，因此 {conclusion} 成立"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "如果 P 成立且 Q 成立，那么 P 且 Q 成立",
            "formal": "P, Q ⊢ P ∧ Q",
            "example": "如果天在下雨且风在刮，那么天在下雨且风在刮",
            "variables": ["前提P", "前提Q", "合取结论P∧Q"]
        }


class ConjunctionEliminationRule(RuleVariableMixin):
    """合取消除规则 (Conjunction Elimination)"""

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
            return self.create_conclusion_variable()

        conjuncts = list(premise.children())
        if conjuncts:
            return random.choice(conjuncts)
        else:
            return premise.arg(0) if random.choice([True, False]) else premise.arg(1)

    def generate_premises(self, conclusion_expr, max_premises=1):
        num_additional = random.randint(1, 3)
        additional_terms = []

        for i in range(num_additional):
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