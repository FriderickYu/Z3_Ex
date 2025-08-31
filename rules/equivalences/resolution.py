"""
布尔代数律
归结规则
P∨Q, ¬P∨R ⊢ Q∨R
"""

import z3
import random
from utils.variable_manager import RuleVariableMixin

class ResolutionRule(RuleVariableMixin):
    """归结规则 (Resolution Rule)

    根据前提 P∨Q 和 ¬P∨R 推出结论 Q∨R。
    """

    def __init__(self):
        self.name = "ResolutionRule"
        self.description = "归结：P∨Q, ¬P∨R ⊢ Q∨R"

    def num_premises(self):
        return 2

    def _z3_equal(self, expr1, expr2):
        try:
            return z3.simplify(expr1) == z3.simplify(expr2)
        except Exception:
            return str(expr1) == str(expr2)

    def _is_complementary(self, a, b):
        """判断两个字句是否互补（一个是另一个的否定）。"""
        if z3.is_not(a) and self._z3_equal(a.arg(0), b):
            return True
        if z3.is_not(b) and self._z3_equal(b.arg(0), a):
            return True
        return False

    def can_apply(self, premises):
        if len(premises) != 2:
            return False
        p1, p2 = premises
        if not (z3.is_or(p1) and z3.is_or(p2)):
            return False
        disj1 = list(p1.children())
        disj2 = list(p2.children())
        for d1 in disj1:
            for d2 in disj2:
                if self._is_complementary(d1, d2):
                    return True
        return False

    def construct_conclusion(self, premises):
        if len(premises) != 2:
            raise ValueError("ResolutionRule需要恰好2个前提")
        p1, p2 = premises
        if not (z3.is_or(p1) and z3.is_or(p2)):
            return self.create_conclusion_variable()
        disj1 = list(p1.children())
        disj2 = list(p2.children())
        # 搜索互补对，构造结论
        for d1 in disj1:
            for d2 in disj2:
                if self._is_complementary(d1, d2):
                    rest1 = [x for x in disj1 if not self._z3_equal(x, d1)]
                    rest2 = [x for x in disj2 if not self._z3_equal(x, d2)]
                    combined = rest1 + rest2
                    if not combined:
                        # 如果没有其余项，则返回新变量
                        return self.create_conclusion_variable()
                    if len(combined) == 1:
                        return combined[0]
                    else:
                        return z3.Or(*combined)
        return self.create_conclusion_variable()

    def generate_premises(self, conclusion_expr, max_premises=2):
        """根据结论反向生成前提。"""
        # 如果结论是析取，则取两个项作为 Q, R
        if z3.is_or(conclusion_expr):
            parts = list(conclusion_expr.children())
            if len(parts) >= 2:
                q_part, r_part = parts[0], parts[1]
            elif len(parts) == 1:
                q_part = parts[0]
                r_part = self.create_conclusion_variable()
            else:
                q_part = self.create_conclusion_variable()
                r_part = self.create_conclusion_variable()
        else:
            q_part = conclusion_expr
            r_part = self.create_conclusion_variable()
        # 创建一个新变量 P
        p_var = self.create_premise_variable()
        # 前提1：P ∨ Q；前提2：¬P ∨ R
        premise1 = z3.Or(p_var, q_part)
        premise2 = z3.Or(z3.Not(p_var), r_part)
        return [premise1, premise2] if random.choice([True, False]) else [premise2, premise1]

    def explain_step(self, premises, conclusion):
        return f"Resolution: 由 {premises[0]} 和 {premises[1]} 归结得到 {conclusion}"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "若有 P∨Q 和 ¬P∨R，则可归结出 Q∨R",
            "formal": "P∨Q, ¬P∨R ⊢ Q∨R",
            "example": "如果有命题a或b，并且有非a或c，那么可以推出b或c",
            "variables": ["字句P", "字句Q", "字句R"]
        }