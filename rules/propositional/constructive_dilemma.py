"""
命题逻辑规则
建构两难
(P→Q)∧(R→S), P∨R ⊢ Q∨S
"""

import z3
import random
from utils.variable_manager import RuleVariableMixin

class ConstructiveDilemmaRule(RuleVariableMixin):
    """建构两难（Constructive Dilemma）

    根据前提 (P→Q)∧(R→S) 和 P∨R 推出结论 Q∨S。
    """

    def __init__(self):
        self.name = "ConstructiveDilemma"
        self.description = "建构两难：((P→Q)∧(R→S), P∨R ⊢ Q∨S)"

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
        # 一个前提是合取包含蕴含，另一个前提是析取
        return (z3.is_and(p1) and z3.is_or(p2)) or (z3.is_and(p2) and z3.is_or(p1))

    def _extract_implications(self, conj_expr):
        """从合取表达式中提取所有蕴含项。"""
        if not z3.is_and(conj_expr):
            return []
        return [c for c in conj_expr.children() if z3.is_implies(c)]

    def construct_conclusion(self, premises):
        if len(premises) != 2:
            raise ValueError("ConstructiveDilemma需要恰好2个前提")
        # 找到合取前提和析取前提
        if z3.is_and(premises[0]) and z3.is_or(premises[1]):
            conj, disj = premises[0], premises[1]
        elif z3.is_and(premises[1]) and z3.is_or(premises[0]):
            conj, disj = premises[1], premises[0]
        else:
            return self.create_conclusion_variable()
        implications = self._extract_implications(conj)
        disjuncts = list(disj.children()) if z3.is_or(disj) else []
        consequents = []
        # 尝试匹配每个析取项与蕴含的前件，收集对应的后件
        for d in disjuncts:
            matched = False
            for imp in implications:
                antecedent = imp.arg(0)
                consequent = imp.arg(1)
                if self._z3_equal(d, antecedent):
                    consequents.append(consequent)
                    matched = True
                    break
            # 如果没有匹配项，则忽略该析取项
        # 构造结论
        if not consequents:
            return self.create_conclusion_variable()
        if len(consequents) == 1:
            return consequents[0]
        else:
            return z3.Or(*consequents)

    def generate_premises(self, conclusion_expr, max_premises=2):
        """根据目标结论反向生成前提。"""
        # 解析结论中的析取项作为后件
        if z3.is_or(conclusion_expr):
            parts = list(conclusion_expr.children())
            if len(parts) >= 2:
                q_part, s_part = parts[0], parts[1]
            elif len(parts) == 1:
                q_part = parts[0]
                s_part = self.create_conclusion_variable()
            else:
                q_part = self.create_conclusion_variable()
                s_part = self.create_conclusion_variable()
        else:
            # 结论不是析取，则生成两个新结论变量作为后件
            q_part = self.create_conclusion_variable()
            s_part = self.create_conclusion_variable()
        # 创建前件变量
        p_var = self.create_premise_variable()
        r_var = self.create_premise_variable()
        # 构造两个蕴含并合取
        conj = z3.And(z3.Implies(p_var, q_part), z3.Implies(r_var, s_part))
        # 构造析取前提
        disj = z3.Or(p_var, r_var)
        return [conj, disj] if random.choice([True, False]) else [disj, conj]

    def explain_step(self, premises, conclusion):
        return f"ConstructiveDilemma: 由 {premises[0]} 和 {premises[1]} 推出 {conclusion}"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "如果 (P→Q) 且 (R→S) 成立，并且 P 或 R 成立，那么 Q 或 S 成立",
            "formal": "(P→Q)∧(R→S), P∨R ⊢ Q∨S",
            "example": "如果下雨意味着湿润，且下雪意味着寒冷，并且下雨或下雪，那么湿润或寒冷",
            "variables": ["前件P", "后件Q", "前件R", "后件S", "蕴含关系", "析取关系"]
        }