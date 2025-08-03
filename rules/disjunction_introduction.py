# 文件：rules/disjunction_introduction.py
# 说明：析取引入规则
# DisjunctionIntroduction: P ⊢ P ∨ Q

import z3
import random
from utils.variable_manager import RuleVariableMixin


class DisjunctionIntroductionRule(RuleVariableMixin):
    """析取引入规则 (Disjunction Introduction)"""

    def __init__(self):
        self.name = "DisjunctionIntroduction"
        self.description = "析取引入：P ⊢ P ∨ Q"

    def num_premises(self):
        return 1

    def can_apply(self, premises):
        return len(premises) == 1

    def construct_conclusion(self, premises):
        if len(premises) != 1:
            raise ValueError("DisjunctionIntroduction需要恰好1个前提")

        premise = premises[0]
        additional_var = self.create_premise_variable()

        # 随机决定是 P ∨ Q 还是 Q ∨ P
        if random.choice([True, False]):
            return z3.Or(premise, additional_var)
        else:
            return z3.Or(additional_var, premise)

    def generate_premises(self, conclusion_expr, max_premises=1):
        if z3.is_or(conclusion_expr):
            disjuncts = list(conclusion_expr.children())
            if disjuncts:
                chosen_premise = random.choice(disjuncts)
                return [chosen_premise]
            else:
                if random.choice([True, False]):
                    return [conclusion_expr.arg(0)]
                else:
                    return [conclusion_expr.arg(1)]
        else:
            premise = self.create_premise_variable()
            return [premise]

    def explain_step(self, premises, conclusion):
        premise = premises[0]
        return f"DisjunctionIntroduction: 由于 {premise} 成立，因此 {conclusion} 也成立（析取弱化）"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "如果 P 成立，那么 P 或 Q 成立",
            "formal": "P ⊢ P ∨ Q",
            "example": "如果天在下雨，那么天在下雨或者天气晴朗",
            "variables": ["前提P", "附加变量Q", "析取结论P∨Q"],
            "note": "这是逻辑弱化的体现：从强的条件可以推出弱的条件"
        }