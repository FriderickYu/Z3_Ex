# 文件：rules/biconditional_elimination.py
# 说明：双条件消除规则
# BiconditionalElimination: P ↔ Q ⊢ (P → Q) ∧ (Q → P)

import z3
from utils.variable_manager import RuleVariableMixin


class BiconditionalEliminationRule(RuleVariableMixin):
    """双条件消除规则 (Biconditional Elimination)"""

    def __init__(self):
        self.name = "BiconditionalElimination"
        self.description = "双条件消除：P ↔ Q ⊢ (P → Q) ∧ (Q → P)"

    def num_premises(self):
        return 1

    def can_apply(self, premises):
        if len(premises) != 1:
            return False
        premise = premises[0]
        return self._is_biconditional(premise)

    def _is_biconditional(self, expr):
        expr_str = str(expr).lower()
        return '↔' in expr_str or 'iff' in expr_str or self._is_equivalence(expr)

    def _is_equivalence(self, expr):
        try:
            return z3.is_eq(expr) and z3.is_bool(expr.arg(0))
        except:
            return False

    def construct_conclusion(self, premises):
        if len(premises) != 1:
            raise ValueError("BiconditionalElimination需要恰好1个前提")

        # 创建两个变量代表双条件的两边
        var_p = self.create_premise_variable()
        var_q = self.create_premise_variable()

        # 构造 (P → Q) ∧ (Q → P)
        implication1 = z3.Implies(var_p, var_q)
        implication2 = z3.Implies(var_q, var_p)

        return z3.And(implication1, implication2)

    def generate_premises(self, conclusion_expr, max_premises=1):
        bicond_var = self.create_premise_variable()
        return [bicond_var]

    def explain_step(self, premises, conclusion):
        premise = premises[0]
        return f"BiconditionalElimination: 由于 {premise} 是双条件关系，因此 {conclusion} 成立（双向蕴含）"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "如果P当且仅当Q，那么P蕴含Q且Q蕴含P",
            "formal": "P ↔ Q ⊢ (P → Q) ∧ (Q → P)",
            "example": "如果天下雨当且仅当地面湿，那么天下雨蕴含地面湿，且地面湿蕴含天下雨",
            "variables": ["命题P", "命题Q", "双条件关系↔", "蕴含关系→"]
        }