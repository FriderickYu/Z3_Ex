# 蕴含类规则

"""
* ExportationRule（外延）：`P→(Q→R) ⊢ (P∧Q)→R`
* ImportationRule（内延）：`(P∧Q)→R ⊢ P→(Q→R)`
* ContrapositionRule：`P→Q ⊢ ¬Q→¬P`
* MaterialImplication：`P→Q ⊢ ¬P∨Q`
"""

import z3
import random
from utils.variable_manager import RuleVariableMixin


class ExportationRule(RuleVariableMixin):
    """外延规则 (Exportation): P→(Q→R) ⊢ (P∧Q)→R"""

    def __init__(self):
        self.name = "ExportationRule"
        self.description = "外延：P→(Q→R) ⊢ (P∧Q)→R"

    def num_premises(self):
        return 1

    def can_apply(self, premises):
        return len(premises) == 1 and z3.is_implies(premises[0])

    def construct_conclusion(self, premises):
        if len(premises) != 1:
            raise ValueError("ExportationRule需要恰好1个前提")
        expr = premises[0]
        if z3.is_implies(expr):
            p = expr.arg(0)
            inner = expr.arg(1)
            if z3.is_implies(inner):
                q = inner.arg(0)
                r = inner.arg(1)
                return z3.Implies(z3.And(p, q), r)
        # 默认返回新的蕴含表达式
        a = self.create_premise_variable()
        b = self.create_premise_variable()
        c = self.create_conclusion_variable()
        return z3.Implies(z3.And(a, b), c)

    def generate_premises(self, conclusion_expr, max_premises=1):
        """根据结论生成前提。从 (P∧Q)→R 得到 P→(Q→R)。"""
        if z3.is_implies(conclusion_expr):
            antecedent = conclusion_expr.arg(0)
            consequent = conclusion_expr.arg(1)
            if z3.is_and(antecedent) and len(antecedent.children()) == 2:
                p, q = antecedent.children()
                r = consequent
                return [z3.Implies(p, z3.Implies(q, r))]
        # 默认：生成随机的前提
        p = self.create_premise_variable()
        q = self.create_premise_variable()
        r = self.create_conclusion_variable()
        return [z3.Implies(p, z3.Implies(q, r))]

    def explain_step(self, premises, conclusion):
        premise = premises[0] if premises else None
        return f"Exportation: 由 {premise} 可推出 {conclusion}（外延规则，将嵌套蕴含转换为合取蕴含）"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "P→(Q→R) 推出 (P∧Q)→R",
            "formal": "P→(Q→R) ⊢ (P∧Q)→R",
            "example": "如果下雨意味着（打雷意味着停电），则（下雨且打雷）意味着停电",
            "variables": ["前件P", "前件Q", "后件R"]
        }


class ImportationRule(RuleVariableMixin):
    """内延规则 (Importation): (P∧Q)→R ⊢ P→(Q→R)"""

    def __init__(self):
        self.name = "ImportationRule"
        self.description = "内延：(P∧Q)→R ⊢ P→(Q→R)"

    def num_premises(self):
        return 1

    def can_apply(self, premises):
        return len(premises) == 1 and z3.is_implies(premises[0])

    def construct_conclusion(self, premises):
        if len(premises) != 1:
            raise ValueError("ImportationRule需要恰好1个前提")
        expr = premises[0]
        if z3.is_implies(expr):
            antecedent = expr.arg(0)
            consequent = expr.arg(1)
            if z3.is_and(antecedent) and len(antecedent.children()) == 2:
                p, q = antecedent.children()
                r = consequent
                return z3.Implies(p, z3.Implies(q, r))
        # 默认返回新的嵌套蕴含
        p = self.create_premise_variable()
        q = self.create_premise_variable()
        r = self.create_conclusion_variable()
        return z3.Implies(p, z3.Implies(q, r))

    def generate_premises(self, conclusion_expr, max_premises=1):
        """根据结论生成前提：从 P→(Q→R) 得到 (P∧Q)→R"""
        if z3.is_implies(conclusion_expr):
            p = conclusion_expr.arg(0)
            inner = conclusion_expr.arg(1)
            if z3.is_implies(inner):
                q = inner.arg(0)
                r = inner.arg(1)
                return [z3.Implies(z3.And(p, q), r)]
        # 默认生成随机前提
        p = self.create_premise_variable()
        q = self.create_premise_variable()
        r = self.create_conclusion_variable()
        return [z3.Implies(z3.And(p, q), r)]

    def explain_step(self, premises, conclusion):
        premise = premises[0] if premises else None
        return f"Importation: 由 {premise} 可推出 {conclusion}（内延规则，将合取蕴含转换为嵌套蕴含）"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "(P∧Q)→R 推出 P→(Q→R)",
            "formal": "(P∧Q)→R ⊢ P→(Q→R)",
            "example": "如果（下雨且打雷）意味着停电，则下雨意味着（打雷意味着停电）",
            "variables": ["前件P", "前件Q", "后件R"]
        }


class ContrapositionRule(RuleVariableMixin):
    """逆否规则 (Contraposition): P→Q ⊢ ¬Q→¬P"""

    def __init__(self):
        self.name = "ContrapositionRule"
        self.description = "逆否：P→Q ⊢ ¬Q→¬P"

    def num_premises(self):
        return 1

    def can_apply(self, premises):
        return len(premises) == 1 and z3.is_implies(premises[0])

    def construct_conclusion(self, premises):
        if len(premises) != 1:
            raise ValueError("ContrapositionRule需要恰好1个前提")
        expr = premises[0]
        if z3.is_implies(expr):
            p = expr.arg(0)
            q = expr.arg(1)
            return z3.Implies(z3.Not(q), z3.Not(p))
        # 默认返回随机逆否式
        a = self.create_premise_variable()
        b = self.create_conclusion_variable()
        return z3.Implies(z3.Not(b), z3.Not(a))

    def generate_premises(self, conclusion_expr, max_premises=1):
        """从结论 ¬Q→¬P 生成前提 P→Q"""
        if z3.is_implies(conclusion_expr):
            antecedent = conclusion_expr.arg(0)
            consequent = conclusion_expr.arg(1)
            if z3.is_not(antecedent) and z3.is_not(consequent):
                q = antecedent.arg(0)
                p = consequent.arg(0)
                return [z3.Implies(p, q)]
        # 默认生成随机前提
        a = self.create_premise_variable()
        b = self.create_conclusion_variable()
        return [z3.Implies(a, b)]

    def explain_step(self, premises, conclusion):
        premise = premises[0] if premises else None
        return f"Contraposition: 由 {premise} 可推出 {conclusion}（逆否规则）"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "P→Q 推出 ¬Q→¬P",
            "formal": "P→Q ⊢ ¬Q→¬P",
            "example": "如果下雨意味着地面湿，则地面不湿意味着不下雨",
            "variables": ["前件P", "后件Q"]
        }


class MaterialImplicationRule(RuleVariableMixin):
    """材料蕴含规则 (Material Implication): P→Q ⊢ ¬P∨Q"""

    def __init__(self):
        self.name = "MaterialImplication"
        self.description = "材料蕴含：P→Q ⊢ ¬P∨Q"

    def num_premises(self):
        return 1

    def can_apply(self, premises):
        return len(premises) == 1 and z3.is_implies(premises[0])

    def construct_conclusion(self, premises):
        if len(premises) != 1:
            raise ValueError("MaterialImplication需要恰好1个前提")
        expr = premises[0]
        if z3.is_implies(expr):
            p = expr.arg(0)
            q = expr.arg(1)
            return z3.Or(z3.Not(p), q)
        # 默认返回随机析取
        a = self.create_premise_variable()
        b = self.create_conclusion_variable()
        return z3.Or(z3.Not(a), b)

    def generate_premises(self, conclusion_expr, max_premises=1):
        """从结论 ¬P∨Q 生成前提 P→Q"""
        if z3.is_or(conclusion_expr) and len(conclusion_expr.children()) == 2:
            a, b = conclusion_expr.children()
            # 检查其中一个是¬P，一个是 Q
            if z3.is_not(a):
                p = a.arg(0)
                q = b
                return [z3.Implies(p, q)]
            if z3.is_not(b):
                p = b.arg(0)
                q = a
                return [z3.Implies(p, q)]
        # 默认生成随机前提
        a = self.create_premise_variable()
        b = self.create_conclusion_variable()
        return [z3.Implies(a, b)]

    def explain_step(self, premises, conclusion):
        premise = premises[0] if premises else None
        return f"MaterialImplication: 由 {premise} 可推出 {conclusion}（材料蕴含）"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "P→Q 推出 ¬P∨Q",
            "formal": "P→Q ⊢ ¬P∨Q",
            "example": "如果下雨意味着地面湿，则不是下雨或者地面湿",
            "variables": ["前件P", "后件Q"]
        }