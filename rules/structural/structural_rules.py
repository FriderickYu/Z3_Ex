# 结构类规则

"""
* CutRule（割）：`P⊢Q, Q⊢R ⟹ P⊢R`
* WeakeningRule（弱化）：`P⊢Q ⟹ P∧R⊢Q`
* ContractionRule（收缩）：`P∧P⊢Q ⟹ P⊢Q`
* ExchangeRule（交换）：`P∧Q⊢R ⟹ Q∧P⊢R`
"""

import z3
import random
from utils.variable_manager import RuleVariableMixin


class CutRule(RuleVariableMixin):
    """割规则：P⊢Q, Q⊢R ⟹ P⊢R。

    在实现中，前提视为两个蕴含表达式 P→Q 和 Q→R，结论为 P→R。
    """

    def __init__(self):
        self.name = "CutRule"
        self.description = "割：P→Q, Q→R ⊢ P→R"

    def num_premises(self):
        return 2

    def _z3_equal(self, expr1, expr2):
        """Z3 表达式等价性检查"""
        try:
            return z3.simplify(expr1) == z3.simplify(expr2)
        except Exception:
            return str(expr1) == str(expr2)

    def can_apply(self, premises):
        return len(premises) == 2 and z3.is_implies(premises[0]) and z3.is_implies(premises[1])

    def construct_conclusion(self, premises):
        if len(premises) != 2:
            raise ValueError("CutRule需要恰好2个前提")
        imp1, imp2 = premises
        if z3.is_implies(imp1) and z3.is_implies(imp2):
            p = imp1.arg(0)
            q1 = imp1.arg(1)  # imp1: P → Q1
            q2 = imp2.arg(0)  # imp2: Q2 → R
            r = imp2.arg(1)

            # 检查中间项是否匹配：Q1 与 Q2 相同
            if self._z3_equal(q1, q2):
                return z3.Implies(p, r)  # 返回 P → R

            # 也检查反向匹配的情况
            # imp1: P → Q1, imp2: Q2 → R
            # 可能的情况：imp2 的后件与 imp1 的前件匹配
            if self._z3_equal(r, p):
                return z3.Implies(q2, q1)  # Q2 → Q1

            # 或者 imp1 的后件与 imp2 的后件匹配，imp2 的前件与 imp1 的前件匹配
            if self._z3_equal(q1, r) and self._z3_equal(p, q2):
                return z3.Implies(p, p)  # P → P (自反)

        # 如果没有找到匹配的传递关系，生成默认的蕴含
        a = self.create_premise_variable()
        c = self.create_conclusion_variable()
        return z3.Implies(a, c)

    def generate_premises(self, conclusion_expr, max_premises=2):
        """根据结论生成前提：从 P→R 生成 P→Q 和 Q→R，其中 Q 为新引入命题。"""
        if z3.is_implies(conclusion_expr):
            p = conclusion_expr.arg(0)
            r = conclusion_expr.arg(1)
            q = self.create_premise_variable()
            imp1 = z3.Implies(p, q)
            imp2 = z3.Implies(q, r)
            return [imp1, imp2] if random.choice([True, False]) else [imp2, imp1]
        # 默认随机生成
        p = self.create_premise_variable()
        q = self.create_premise_variable()
        r = self.create_conclusion_variable()
        imp1 = z3.Implies(p, q)
        imp2 = z3.Implies(q, r)
        return [imp1, imp2] if random.choice([True, False]) else [imp2, imp1]

    def explain_step(self, premises, conclusion):
        if len(premises) == 2:
            return f"Cut: 由 {premises[0]} 和 {premises[1]} 推出 {conclusion}（割规则）"
        return f"Cut: 基于前提 {premises} 推出 {conclusion}"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "从 P→Q 和 Q→R 可以推出 P→R",
            "formal": "P⊢Q, Q⊢R ⟹ P⊢R",
            "example": "若从感冒可推出发烧，从发烧可推出住院，则从感冒可推出住院",
            "variables": ["命题P", "命题Q", "命题R"]
        }


class WeakeningRule(RuleVariableMixin):
    """弱化规则：P⊢Q ⟹ P∧R⊢Q。

    将已有的蕴含 P→Q 扩充前提为合取 P∧R，其中 R 为新命题。
    """

    def __init__(self):
        self.name = "WeakeningRule"
        self.description = "弱化：P→Q ⊢ (P∧R)→Q"

    def num_premises(self):
        return 1

    def can_apply(self, premises):
        return len(premises) == 1 and z3.is_implies(premises[0])

    def construct_conclusion(self, premises):
        if len(premises) != 1:
            raise ValueError("WeakeningRule需要恰好1个前提")
        imp = premises[0]
        if z3.is_implies(imp):
            p = imp.arg(0)
            q = imp.arg(1)
            r = self.create_premise_variable()
            return z3.Implies(z3.And(p, r), q)
        # 默认随机生成
        a = self.create_premise_variable()
        q = self.create_conclusion_variable()
        r = self.create_premise_variable()
        return z3.Implies(z3.And(a, r), q)

    def generate_premises(self, conclusion_expr, max_premises=1):
        """从 (P∧R)→Q 生成 P→Q。"""
        if z3.is_implies(conclusion_expr):
            antecedent = conclusion_expr.arg(0)
            consequent = conclusion_expr.arg(1)
            if z3.is_and(antecedent) and len(antecedent.children()) == 2:
                p, r = antecedent.children()
                return [z3.Implies(p, consequent)]
        # 默认生成随机前提
        a = self.create_premise_variable()
        q = self.create_conclusion_variable()
        return [z3.Implies(a, q)]

    def explain_step(self, premises, conclusion):
        premise = premises[0] if premises else None
        return f"Weakening: 由 {premise} 推出 {conclusion}（弱化规则）"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "P→Q 推出 (P∧R)→Q",
            "formal": "P⊢Q ⟹ P∧R⊢Q",
            "example": "若有雨则路滑，则（有雨且有雾）则路滑",
            "variables": ["前件P", "后件Q", "附加命题R"]
        }


class ContractionRule(RuleVariableMixin):
    """收缩规则：P∧P⊢Q ⟹ P⊢Q。

    对于蕴含 (P∧P)→Q，去掉重复的 P，得到 P→Q。
    """

    def __init__(self):
        self.name = "ContractionRule"
        self.description = "收缩：(P∧P)→Q ⊢ P→Q"

    def num_premises(self):
        return 1

    def _z3_equal(self, expr1, expr2):
        """Z3 表达式等价性检查"""
        try:
            return z3.simplify(expr1) == z3.simplify(expr2)
        except Exception:
            return str(expr1) == str(expr2)

    def can_apply(self, premises):
        return len(premises) == 1 and z3.is_implies(premises[0])

    def construct_conclusion(self, premises):
        if len(premises) != 1:
            raise ValueError("ContractionRule需要恰好1个前提")
        imp = premises[0]
        if z3.is_implies(imp):
            antecedent = imp.arg(0)
            consequent = imp.arg(1)

            # 检查 antecedent 是否为合取形式
            if z3.is_and(antecedent):
                conjuncts = list(antecedent.children()) if hasattr(antecedent, 'children') else [antecedent.arg(0), antecedent.arg(1)]

                # 检查是否所有合取项都相同（收缩的条件）
                if len(conjuncts) >= 2:
                    first = conjuncts[0]
                    # 检查是否所有项都与第一项相同
                    all_same = True
                    for conjunct in conjuncts[1:]:
                        if not self._z3_equal(first, conjunct):
                            all_same = False
                            break

                    if all_same:
                        # 所有合取项都相同，可以收缩为单个项
                        return z3.Implies(first, consequent)

                # 如果只有两个项且相同
                if len(conjuncts) == 2 and self._z3_equal(conjuncts[0], conjuncts[1]):
                    return z3.Implies(conjuncts[0], consequent)

        # 如果不符合收缩条件，生成默认蕴含
        a = self.create_premise_variable()
        q = self.create_conclusion_variable()
        return z3.Implies(a, q)

    def generate_premises(self, conclusion_expr, max_premises=1):
        """从 P→Q 生成 (P∧P)→Q。"""
        if z3.is_implies(conclusion_expr):
            p = conclusion_expr.arg(0)
            q = conclusion_expr.arg(1)
            return [z3.Implies(z3.And(p, p), q)]
        # 默认生成随机前提
        a = self.create_premise_variable()
        q = self.create_conclusion_variable()
        return [z3.Implies(z3.And(a, a), q)]

    def explain_step(self, premises, conclusion):
        premise = premises[0] if premises else None
        return f"Contraction: 由 {premise} 推出 {conclusion}（收缩规则）"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "(P∧P)→Q 推出 P→Q",
            "formal": "P∧P⊢Q ⟹ P⊢Q",
            "example": "如果两次下雨意味着路滑，则一次下雨意味着路滑",
            "variables": ["命题P", "命题Q"]
        }


class ExchangeRule(RuleVariableMixin):
    """交换规则：P∧Q⊢R ⟹ Q∧P⊢R。

    交换合取的顺序不影响结论。
    """

    def __init__(self):
        self.name = "ExchangeRule"
        self.description = "交换：(P∧Q)→R ⊢ (Q∧P)→R"

    def num_premises(self):
        return 1

    def can_apply(self, premises):
        return len(premises) == 1 and z3.is_implies(premises[0])

    def construct_conclusion(self, premises):
        if len(premises) != 1:
            raise ValueError("ExchangeRule需要恰好1个前提")
        imp = premises[0]
        if z3.is_implies(imp):
            antecedent = imp.arg(0)
            consequent = imp.arg(1)
            if z3.is_and(antecedent) and len(antecedent.children()) == 2:
                p, q = antecedent.children()
                return z3.Implies(z3.And(q, p), consequent)
        # 默认生成新的蕴含
        a = self.create_premise_variable()
        b = self.create_premise_variable()
        r = self.create_conclusion_variable()
        return z3.Implies(z3.And(b, a), r)

    def generate_premises(self, conclusion_expr, max_premises=1):
        """从 (Q∧P)→R 生成 (P∧Q)→R"""
        if z3.is_implies(conclusion_expr):
            antecedent = conclusion_expr.arg(0)
            consequent = conclusion_expr.arg(1)
            if z3.is_and(antecedent) and len(antecedent.children()) == 2:
                q, p = antecedent.children()
                return [z3.Implies(z3.And(p, q), consequent)]
        # 默认随机
        a = self.create_premise_variable()
        b = self.create_premise_variable()
        r = self.create_conclusion_variable()
        return [z3.Implies(z3.And(a, b), r)]

    def explain_step(self, premises, conclusion):
        premise = premises[0] if premises else None
        return f"Exchange: 由 {premise} 推出 {conclusion}（交换规则）"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "(P∧Q)→R 推出 (Q∧P)→R",
            "formal": "P∧Q⊢R ⟹ Q∧P⊢R",
            "example": "如果（下雨且有雾）意味着延误，则（有雾且下雨）也意味着延误",
            "variables": ["命题P", "命题Q", "命题R"]
        }