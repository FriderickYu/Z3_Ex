# 双条件类规则

"""
* BiconditionalIntroduction：`P→Q, Q→P ⊢ P↔Q`
* BiconditionalTransitivity：`P↔Q, Q↔R ⊢ P↔R`
"""

import z3
import random
from utils.variable_manager import RuleVariableMixin


class BiconditionalIntroductionRule(RuleVariableMixin):
    """双条件引入规则：P→Q, Q→P ⊢ P↔Q"""

    def __init__(self):
        self.name = "BiconditionalIntroduction"
        self.description = "双条件引入：P→Q, Q→P ⊢ P↔Q"

    def num_premises(self):
        return 2

    def _is_implies(self, expr):
        return z3.is_implies(expr)

    def _extract_implication(self, expr):
        """返回 (antecedent, consequent) 如果 expr 是蕴含，否则 None"""
        if not self._is_implies(expr):
            return None
        return expr.arg(0), expr.arg(1)

    def can_apply(self, premises):
        if len(premises) != 2:
            return False
        # 两个前提都应该是蕴含
        return self._is_implies(premises[0]) and self._is_implies(premises[1])

    def construct_conclusion(self, premises):
        if len(premises) != 2:
            raise ValueError("BiconditionalIntroduction需要恰好2个前提")
        imp1 = self._extract_implication(premises[0])
        imp2 = self._extract_implication(premises[1])
        if imp1 and imp2:
            p1, q1 = imp1
            p2, q2 = imp2
            # 寻找一个 p→q 和 q→p 结构
            # 情形1：premises[0]为 P→Q，premises[1]为 Q→P
            try:
                if z3.is_true(z3.simplify(q1 == p2)) and z3.is_true(z3.simplify(q2 == p1)):
                    # 返回双条件：P↔Q，可以用 And(Implies(p,q),Implies(q,p)) 表示
                    return z3.And(z3.Implies(p1, q1), z3.Implies(q1, p1))
                # 情形2：premises[0]为 Q→P，premises[1]为 P→Q
                if z3.is_true(z3.simplify(p1 == q2)) and z3.is_true(z3.simplify(p2 == q1)):
                    return z3.And(z3.Implies(p2, q2), z3.Implies(q2, p2))
            except Exception:
                pass
        # 默认返回新的结论变量
        return self.create_conclusion_variable()

    def generate_premises(self, conclusion_expr, max_premises=2):
        """根据结论反向生成前提。

        如果结论为 P↔Q，即形式为 And(Implies(P,Q),Implies(Q,P)) 或两个条件等价的布尔等式，生成这两条蕴含。
        """
        # 解析结论为双条件形式
        # 如果是 And 两个蕴含
        if z3.is_and(conclusion_expr) and len(conclusion_expr.children()) == 2:
            c1, c2 = conclusion_expr.children()
            if z3.is_implies(c1) and z3.is_implies(c2):
                return [c1, c2] if random.choice([True, False]) else [c2, c1]
        # 如果是布尔等式 P == Q
        if conclusion_expr.decl().kind() == z3.Z3_OP_EQ and conclusion_expr.num_args() == 2:
            p, q = conclusion_expr.arg(0), conclusion_expr.arg(1)
            imp1 = z3.Implies(p, q)
            imp2 = z3.Implies(q, p)
            return [imp1, imp2] if random.choice([True, False]) else [imp2, imp1]
        # 默认随机生成两个蕴含
        # 创建两个新的布尔前提变量
        p = self.create_premise_variable()
        q = self.create_premise_variable()
        return [z3.Implies(p, q), z3.Implies(q, p)] if random.choice([True, False]) else [z3.Implies(q, p), z3.Implies(p, q)]

    def explain_step(self, premises, conclusion):
        if len(premises) == 2:
            return f"BiconditionalIntroduction: 由 {premises[0]} 和 {premises[1]}，可引入双条件得到 {conclusion}"
        return f"BiconditionalIntroduction: 基于前提 {premises} 推出 {conclusion}"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "如果 P→Q 且 Q→P，则得到 P↔Q",
            "formal": "P→Q, Q→P ⊢ P↔Q",
            "example": "如果下雨意味着地面湿，且地面湿意味着下雨，则下雨当且仅当地面湿",
            "variables": ["前件P", "后件Q"],
        }


class BiconditionalTransitivityRule(RuleVariableMixin):
    """双条件传递规则：P↔Q, Q↔R ⊢ P↔R"""

    def __init__(self):
        self.name = "BiconditionalTransitivity"
        self.description = "双条件传递：P↔Q, Q↔R ⊢ P↔R"

    def num_premises(self):
        return 2

    def _extract_biconditional(self, expr):
        """尝试从双条件表达式中提取两个命题 P、Q。
        支持 And(Implies(P,Q), Implies(Q,P)) 或 p == q 的形式。
        如果不匹配，返回 None。
        """
        # 形式1：And(Implies(P,Q),Implies(Q,P))
        if z3.is_and(expr) and len(expr.children()) == 2:
            a, b = expr.children()
            if z3.is_implies(a) and z3.is_implies(b):
                p1, q1 = a.arg(0), a.arg(1)
                p2, q2 = b.arg(0), b.arg(1)
                # 判断是否互逆
                try:
                    if z3.is_true(z3.simplify(p1 == q2)) and z3.is_true(z3.simplify(q1 == p2)):
                        return p1, q1
                except Exception:
                    pass
        # 形式2：布尔相等 p == q
        if expr.decl().kind() == z3.Z3_OP_EQ and expr.num_args() == 2:
            return expr.arg(0), expr.arg(1)
        return None

    def can_apply(self, premises):
        if len(premises) != 2:
            return False
        return self._extract_biconditional(premises[0]) is not None and self._extract_biconditional(premises[1]) is not None

    def construct_conclusion(self, premises):
        if len(premises) != 2:
            raise ValueError("BiconditionalTransitivity需要恰好2个前提")
        pair1 = self._extract_biconditional(premises[0])
        pair2 = self._extract_biconditional(premises[1])
        if pair1 and pair2:
            p, q = pair1
            q2, r = pair2
            # 寻找匹配 q==q2
            try:
                if z3.is_true(z3.simplify(q == q2)):
                    # 结论为 P↔R
                    return z3.And(z3.Implies(p, r), z3.Implies(r, p))
                # 考虑顺序反转
                p, q = pair1
                r, q2 = pair2
                if z3.is_true(z3.simplify(q == q2)):
                    return z3.And(z3.Implies(p, r), z3.Implies(r, p))
            except Exception:
                pass
        return self.create_conclusion_variable()

    def generate_premises(self, conclusion_expr, max_premises=2):
        """根据结论生成前提：从 P↔R 推出 P↔Q 和 Q↔R，其中 Q 为新引入的命题。"""
        # 尝试解析结论
        pair = self._extract_biconditional(conclusion_expr)
        if pair:
            p, r = pair
            # 创建新的中间命题 Q
            q = self.create_premise_variable()
            bic1 = z3.And(z3.Implies(p, q), z3.Implies(q, p))
            bic2 = z3.And(z3.Implies(q, r), z3.Implies(r, q))
            return [bic1, bic2] if random.choice([True, False]) else [bic2, bic1]
        # 默认随机生成
        p = self.create_premise_variable()
        q = self.create_premise_variable()
        r = self.create_premise_variable()
        bic1 = z3.And(z3.Implies(p, q), z3.Implies(q, p))
        bic2 = z3.And(z3.Implies(q, r), z3.Implies(r, q))
        return [bic1, bic2] if random.choice([True, False]) else [bic2, bic1]

    def explain_step(self, premises, conclusion):
        if len(premises) == 2:
            return f"BiconditionalTransitivity: 由 {premises[0]} 和 {premises[1]} 可推出 {conclusion}（双条件传递）"
        return f"BiconditionalTransitivity: 基于前提 {premises} 推出 {conclusion}"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "如果 P↔Q 且 Q↔R，则 P↔R",
            "formal": "P↔Q, Q↔R ⊢ P↔R",
            "example": "若下雨当且仅当地面湿，且地面湿当且仅当天气阴，则下雨当且仅当天气阴",
            "variables": ["命题P", "命题Q", "命题R"],
        }