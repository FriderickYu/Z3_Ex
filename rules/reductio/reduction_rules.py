# 归约推理

"""
* NegationIntroduction（否定引入）：`Γ ∪ {P} ⊢ ⊥ ⟹ Γ ⊢ ¬P`
* ProofByContradiction / RAA（反证法）：`Γ ∪ {¬P} ⊢ ⊥ ⟹ Γ ⊢ P`
* Explosion / EFQ（爆炸律，可选）：`Γ ⊢ ⊥ ⟹ Γ ⊢ φ`
"""

import z3
import random
from utils.variable_manager import RuleVariableMixin


class NegationIntroductionRule(RuleVariableMixin):
    """否定引入规则：Γ ∪ {P} ⊢ ⊥ ⟹ Γ ⊢ ¬P。

    在实现中，将前提视为蕴含 P→False，结论为 ¬P。
    """

    def __init__(self):
        self.name = "NegationIntroduction"
        self.description = "否定引入：P→⊥ ⊢ ¬P"

    def num_premises(self):
        return 1

    def can_apply(self, premises):
        return len(premises) == 1 and z3.is_implies(premises[0])

    def construct_conclusion(self, premises):
        if len(premises) != 1:
            raise ValueError("NegationIntroduction需要恰好1个前提")
        expr = premises[0]
        if z3.is_implies(expr):
            p = expr.arg(0)
            q = expr.arg(1)
            # 判断结论是 False
            try:
                if z3.is_false(q) or (hasattr(q, 'decl') and q.decl().kind() == z3.Z3_OP_FALSE):
                    return z3.Not(p)
            except Exception:
                pass
        # 默认生成新的否定
        a = self.create_premise_variable()
        return z3.Not(a)

    def generate_premises(self, conclusion_expr, max_premises=1):
        """从结论 ¬P 生成前提 P→⊥。"""
        if z3.is_not(conclusion_expr):
            p = conclusion_expr.arg(0)
            return [z3.Implies(p, z3.BoolVal(False))]
        # 默认随机生成
        a = self.create_premise_variable()
        return [z3.Implies(a, z3.BoolVal(False))]

    def explain_step(self, premises, conclusion):
        premise = premises[0] if premises else None
        return f"NegationIntroduction: 由 {premise} 推出 {conclusion}（假设推出矛盾，则否定假设）"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "如果假设 P 推出矛盾，则 ¬P 成立",
            "formal": "Γ∪{P}⊢⊥ ⟹ Γ⊢¬P",
            "example": "若假设下雨导致矛盾，则证明不上雨",
            "variables": ["假设P"]
        }


class ProofByContradictionRule(RuleVariableMixin):
    """反证法规则 (RAA)：Γ ∪ {¬P} ⊢ ⊥ ⟹ Γ ⊢ P。

    在实现中，将前提视为蕴含 ¬P→False，结论为 P。
    """

    def __init__(self):
        self.name = "ProofByContradiction"
        self.description = "反证法：¬P→⊥ ⊢ P"

    def num_premises(self):
        return 1

    def can_apply(self, premises):
        return len(premises) == 1 and z3.is_implies(premises[0])

    def construct_conclusion(self, premises):
        if len(premises) != 1:
            raise ValueError("ProofByContradiction需要恰好1个前提")
        expr = premises[0]
        if z3.is_implies(expr):
            antecedent = expr.arg(0)
            consequent = expr.arg(1)
            # antecedent 应为 ¬P，consequent 为 False
            if z3.is_not(antecedent):
                inner = antecedent.arg(0)
                try:
                    if z3.is_false(consequent) or (hasattr(consequent, 'decl') and consequent.decl().kind() == z3.Z3_OP_FALSE):
                        return inner
                except Exception:
                    pass
        # 默认生成新的命题
        return self.create_conclusion_variable()

    def generate_premises(self, conclusion_expr, max_premises=1):
        """从结论 P 生成前提 ¬P→⊥。"""
        # 结论如果是单个命题 p
        if isinstance(conclusion_expr, z3.BoolRef) and not z3.is_not(conclusion_expr):
            return [z3.Implies(z3.Not(conclusion_expr), z3.BoolVal(False))]
        # 默认生成随机前提
        a = self.create_premise_variable()
        return [z3.Implies(z3.Not(a), z3.BoolVal(False))]

    def explain_step(self, premises, conclusion):
        premise = premises[0] if premises else None
        return f"ProofByContradiction: 由 {premise} 推出 {conclusion}（反证法）"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "如果假设 ¬P 推出矛盾，则 P 成立",
            "formal": "Γ∪{¬P}⊢⊥ ⟹ Γ⊢P",
            "example": "若假设不下雨导致矛盾，则证明下雨",
            "variables": ["命题P"]
        }


class ExplosionRule(RuleVariableMixin):
    """爆炸律 (EFQ)：从矛盾可以推出任意命题。

    在实现中，前提为 False 或矛盾表达式，结论为任意布尔变量。
    """

    def __init__(self):
        self.name = "ExplosionRule"
        self.description = "爆炸律：⊥ ⊢ φ"

    def num_premises(self):
        return 1

    def can_apply(self, premises):
        return len(premises) == 1

    def _is_false(self, expr):
        try:
            return z3.is_false(expr) or (hasattr(expr, 'decl') and expr.decl().kind() == z3.Z3_OP_FALSE)
        except Exception:
            return False

    def construct_conclusion(self, premises):
        if len(premises) != 1:
            raise ValueError("ExplosionRule需要恰好1个前提")

        # 爆炸律的核心：从矛盾（任何前提）可以推出任意命题
        # 无论前提是什么，都生成一个新的任意结论
        # 这符合爆炸律的语义：⊥ ⊢ φ（从矛盾推出任何东西）
        return self.create_conclusion_variable()

    def generate_premises(self, conclusion_expr, max_premises=1):
        """从任意结论生成前提 False。"""
        return [z3.BoolVal(False)]

    def explain_step(self, premises, conclusion):
        premise = premises[0] if premises else None
        return f"Explosion: 由矛盾 {premise} 可推出任意命题 {conclusion}（爆炸律）"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "⊥ 推出 φ",
            "formal": "Γ⊢⊥ ⟹ Γ⊢φ",
            "example": "如果证明出了矛盾，那么任何命题都成立",
            "variables": ["任意命题φ"]
        }