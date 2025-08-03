# 文件：rules/transitivity_rule.py
# 说明：传递性规则（算术逻辑）
# TransitivityRule: a R b, b R c ⊢ a R c

import z3
from utils.variable_manager import RuleVariableMixin


class TransitivityRule(RuleVariableMixin):
    """传递性规则 (Transitivity Rule)"""

    def __init__(self):
        self.name = "TransitivityRule"
        self.description = "传递性：aRb, bRc ⊢ aRc"

    def num_premises(self):
        return 2

    def can_apply(self, premises):
        if len(premises) != 2:
            return False

        p1_str = str(premises[0])
        p2_str = str(premises[1])
        return self._has_transitive_structure(p1_str, p2_str)

    def _has_transitive_structure(self, expr1_str, expr2_str):
        return any(op in expr1_str and op in expr2_str for op in ['>', '<', '=', '>=', '<='])

    def construct_conclusion(self, premises):
        if len(premises) != 2:
            raise ValueError("TransitivityRule需要恰好2个前提")

        return self.create_conclusion_variable()

    def generate_premises(self, conclusion_expr, max_premises=2):
        var1 = self.create_premise_variable()
        var2 = self.create_premise_variable()
        return [var1, var2]

    def explain_step(self, premises, conclusion):
        return f"TransitivityRule: 由于 {premises[0]} 和 {premises[1]} 具有传递性关系，因此 {conclusion} 成立"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "如果a关系b，且b关系c，那么a关系c",
            "formal": "aRb, bRc ⊢ aRc",
            "example": "如果A大于B，且B大于C，那么A大于C",
            "variables": ["元素a", "元素b", "元素c", "传递关系R"]
        }