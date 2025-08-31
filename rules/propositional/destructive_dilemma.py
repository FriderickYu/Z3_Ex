"""
命题逻辑规则
破坏两难
(P→Q)∧(R→S), ¬Q∨¬S ⊢ ¬P∨¬R
"""

import z3
import random
from utils.variable_manager import RuleVariableMixin


class DestructiveDilemmaRule(RuleVariableMixin):
    """破坏两难（Destructive Dilemma）

    根据前提 (P→Q)∧(R→S) 和 ¬Q∨¬S 推出结论 ¬P∨¬R。
    """

    def __init__(self):
        self.name = "DestructiveDilemma"
        self.description = "破坏两难：((P→Q)∧(R→S), ¬Q∨¬S ⊢ ¬P∨¬R)"

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
        # 一个前提是合取包含蕴含，另一个前提是析取的否定项
        # 简化检测：一个为合取，一个为析取
        return (z3.is_and(p1) and z3.is_or(p2)) or (z3.is_and(p2) and z3.is_or(p1))

    def _extract_implications(self, conj_expr):
        if not z3.is_and(conj_expr):
            return []
        return [c for c in conj_expr.children() if z3.is_implies(c)]

    def construct_conclusion(self, premises):
        if len(premises) != 2:
            raise ValueError("DestructiveDilemma需要恰好2个前提")
        # 确定合取和析取
        if z3.is_and(premises[0]) and z3.is_or(premises[1]):
            conj, disj = premises[0], premises[1]
        elif z3.is_and(premises[1]) and z3.is_or(premises[0]):
            conj, disj = premises[1], premises[0]
        else:
            return self.create_conclusion_variable()
        implications = self._extract_implications(conj)
        disjuncts = list(disj.children())
        negated_antecedents = []
        # 对于析取的每个否定项 ¬Q 或 ¬S，找到对应的蕴含的后件 Q 或 S，并取其前件 P 或 R
        for lit in disjuncts:
            if z3.is_not(lit):
                target = lit.arg(0)
                for imp in implications:
                    antecedent = imp.arg(0)
                    consequent = imp.arg(1)
                    if self._z3_equal(consequent, target):
                        negated_antecedents.append(z3.Not(antecedent))
                        break
        if not negated_antecedents:
            return self.create_conclusion_variable()
        if len(negated_antecedents) == 1:
            return negated_antecedents[0]
        else:
            return z3.Or(*negated_antecedents)

    def generate_premises(self, conclusion_expr, max_premises=2):
        """根据结论反向生成前提。"""
        # 如果结论是析取形式的否定，如 ¬P∨¬R
        if z3.is_or(conclusion_expr):
            parts = list(conclusion_expr.children())
            # 寻找两个否定项
            negs = [p for p in parts if z3.is_not(p)]
            if len(negs) >= 2:
                # 取前件变量
                p_neg = negs[0].arg(0)
                r_neg = negs[1].arg(0)
            elif len(negs) == 1:
                p_neg = negs[0].arg(0)
                r_neg = self.create_premise_variable()
            else:
                p_neg = self.create_premise_variable()
                r_neg = self.create_premise_variable()
        else:
            p_neg = self.create_premise_variable()
            r_neg = self.create_premise_variable()
        # 创建后件变量
        q_var = self.create_conclusion_variable()
        s_var = self.create_conclusion_variable()
        # 构造合取前提 (P→Q) ∧ (R→S)
        conj = z3.And(z3.Implies(p_neg, q_var), z3.Implies(r_neg, s_var))
        # 构造析取前提 ¬Q∨¬S
        disj = z3.Or(z3.Not(q_var), z3.Not(s_var))
        return [conj, disj] if random.choice([True, False]) else [disj, conj]

    def explain_step(self, premises, conclusion):
        return f"DestructiveDilemma: 由 {premises[0]} 和 {premises[1]} 推出 {conclusion}"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "如果 (P→Q) 且 (R→S) 成立，并且非Q或非S成立，那么非P或非R成立",
            "formal": "(P→Q)∧(R→S), ¬Q∨¬S ⊢ ¬P∨¬R",
            "example": "如果天在下雨意味着地面会湿，且天降雪意味着天气会冷，并且地面不湿或天气不冷，那么天不下雨或天不降雪",
            "variables": ["前件P", "后件Q", "前件R", "后件S", "蕴含关系", "否定析取"]
        }