# BiconditionalDecomposition（双条件分解）：`P↔Q ⟷ (P→Q)∧(Q→P)`

import z3
from utils.variable_manager import RuleVariableMixin


class BiconditionalDecompositionRule(RuleVariableMixin):
    """双条件分解规则 (Biconditional Decomposition)"""

    def __init__(self):
        self.name = "BiconditionalDecomposition"
        self.description = "双条件分解：P ↔ Q 等价于 (P ∧ Q) ∨ (¬P ∧ ¬Q)"

    def num_premises(self):
        return 1

    def can_apply(self, premises):
        return len(premises) == 1

    def construct_conclusion(self, premises):
        # 无论前提为何，都构造一个 (P∧Q)∨(¬P∧¬Q) 的表达式
        p = self.create_premise_variable()
        q = self.create_premise_variable()
        pos = z3.And(p, q)
        neg = z3.And(z3.Not(p), z3.Not(q))
        return z3.Or(pos, neg)

    def generate_premises(self, conclusion_expr, max_premises=1):
        """为给定结论生成一个双条件形式的前提。

        由于此规则只是将双条件分解为特定形式，这里返回一个新的占位变量作为前提。"""
        return [self.create_premise_variable()]

    def explain_step(self, premises, conclusion):
        return f"BiconditionalDecomposition: 将双条件表达式分解为 {conclusion}"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "P ↔ Q 等价于 (P ∧ Q) ∨ (¬P ∧ ¬Q)",
            "formal": "P ↔ Q ↔ (P ∧ Q) ∨ (¬P ∧ ¬Q)",
            "example": "天下雨当且仅当地面湿 等价于 (天下雨且地面湿) 或 (天不下雨且地面不湿)",
            "variables": ["命题P", "命题Q", "分解前提", "分解结论"]
        }