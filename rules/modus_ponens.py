# 文件：rules/modus_ponens.py
# 说明：肯定前件推理规则 (Modus Ponens)
# 形式：P, P → Q ⊢ Q

import z3
import random


class ModusPonensRule:
    """
    肯定前件推理规则 (Modus Ponens)

    规则形式：如果有 P 和 P → Q，则可以推出 Q
    这是最基本的推理规则之一
    """

    def __init__(self):
        self.name = "ModusPonens"
        self.description = "肯定前件：P, P→Q ⊢ Q"

    def num_premises(self):
        """该规则需要2个前提：P 和 P→Q"""
        return 2

    def can_apply(self, premises):
        """
        检查是否可以应用该规则

        Args:
            premises: 前提列表，应该包含 P 和 Implies(P, Q)

        Returns:
            bool: 是否可以应用该规则
        """
        if len(premises) != 2:
            return False

        # 检查是否有一个前提是另一个前提的蕴含关系
        p1, p2 = premises

        # 情况1：p1是P，p2是P→Q
        if z3.is_implies(p2):
            antecedent = p2.arg(0)  # P→Q 的前件
            if self._z3_equal(p1, antecedent):
                return True

        # 情况2：p2是P，p1是P→Q
        if z3.is_implies(p1):
            antecedent = p1.arg(0)  # P→Q 的前件
            if self._z3_equal(p2, antecedent):
                return True

        return False

    def _z3_equal(self, expr1, expr2):
        """检查两个Z3表达式是否相等"""
        try:
            # 使用Z3的简化器检查等价性
            return z3.simplify(expr1) == z3.simplify(expr2)
        except:
            return str(expr1) == str(expr2)

    def construct_conclusion(self, premises):
        """
        根据前提构造结论

        Args:
            premises: 前提列表 [P, P→Q] 或 [P→Q, P]

        Returns:
            z3.ExprRef: 结论 Q
        """
        if len(premises) != 2:
            raise ValueError("ModusPonens需要恰好2个前提")

        p1, p2 = premises

        # 找到蕴含关系和对应的前件
        if z3.is_implies(p1):
            implication = p1
            premise = p2
        elif z3.is_implies(p2):
            implication = p2
            premise = p1
        else:
            # 如果没有蕴含关系，创建一个新的结论变量
            return self._create_new_conclusion_var()

        antecedent = implication.arg(0)  # P→Q 的 P
        consequent = implication.arg(1)  # P→Q 的 Q

        # 验证前件匹配
        if self._z3_equal(premise, antecedent):
            return consequent  # 返回 Q
        else:
            # 如果不匹配，创建新的结论
            return self._create_new_conclusion_var()

    def _create_new_conclusion_var(self):
        """创建新的结论变量"""
        var_id = random.randint(1000, 9999)
        return z3.Bool(f"MP_Conclusion_{var_id}")

    def generate_premises(self, conclusion_expr, max_premises=2):
        """
        反向生成：给定结论，生成可能的前提

        Args:
            conclusion_expr: 结论表达式 Q
            max_premises: 最大前提数量

        Returns:
            list: 前提列表 [P, P→Q]
        """
        # 创建一个新的前提变量 P
        premise_var = z3.Bool(f"MP_Premise_{random.randint(1000, 9999)}")

        # 构造蕴含关系 P → Q
        implication = z3.Implies(premise_var, conclusion_expr)

        return [premise_var, implication]

    def explain_step(self, premises, conclusion):
        """
        解释推理步骤

        Args:
            premises: 前提列表
            conclusion: 结论

        Returns:
            str: 推理步骤的自然语言解释
        """
        if len(premises) != 2:
            return f"ModusPonens: 无法解释，前提数量不正确"

        p1, p2 = premises

        if z3.is_implies(p1):
            return f"ModusPonens: 由于有 {p2} 且有 {p1}，因此可以推出 {conclusion}"
        elif z3.is_implies(p2):
            return f"ModusPonens: 由于有 {p1} 且有 {p2}，因此可以推出 {conclusion}"
        else:
            return f"ModusPonens: 基于前提 {premises}，推出 {conclusion}"

    def get_rule_template(self):
        """
        获取规则模板，用于生成自然语言描述

        Returns:
            dict: 规则模板信息
        """
        return {
            "name": self.name,
            "pattern": "如果 P 成立，并且 P 蕴含 Q，那么 Q 成立",
            "formal": "P, P → Q ⊢ Q",
            "example": "如果天下雨，并且天下雨意味着地面会湿，那么地面会湿",
            "variables": ["前提P", "蕴含关系P→Q", "结论Q"]
        }