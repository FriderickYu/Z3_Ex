# 文件：rules/conjunction_rules.py
# 说明：合取引入和消除规则
# ConjunctionIntroduction: P, Q ⊢ P ∧ Q
# ConjunctionElimination: P ∧ Q ⊢ P (或 Q)

import z3
import random


class ConjunctionIntroductionRule:
    """
    合取引入规则 (Conjunction Introduction)

    规则形式：如果有 P 和 Q，则可以推出 P ∧ Q
    """

    def __init__(self):
        self.name = "ConjunctionIntroduction"
        self.description = "合取引入：P, Q ⊢ P ∧ Q"

    def num_premises(self):
        """该规则需要2个或更多前提"""
        return random.randint(2, 4)  # 支持2-4个前提的合取

    def can_apply(self, premises):
        """
        检查是否可以应用该规则

        Args:
            premises: 前提列表

        Returns:
            bool: 总是可以应用（任何前提都可以合取）
        """
        return len(premises) >= 2

    def construct_conclusion(self, premises):
        """
        根据前提构造合取结论

        Args:
            premises: 前提列表 [P, Q, R, ...]

        Returns:
            z3.ExprRef: 合取结论 P ∧ Q ∧ R ∧ ...
        """
        if len(premises) < 2:
            raise ValueError("ConjunctionIntroduction需要至少2个前提")

        # 构造合取表达式
        if len(premises) == 2:
            return z3.And(premises[0], premises[1])
        else:
            return z3.And(*premises)

    def generate_premises(self, conclusion_expr, max_premises=4):
        """
        反向生成：给定合取结论，分解为前提

        Args:
            conclusion_expr: 合取结论 P ∧ Q
            max_premises: 最大前提数量

        Returns:
            list: 前提列表
        """
        if z3.is_and(conclusion_expr):
            # 如果结论是合取，返回所有合取项
            return list(conclusion_expr.children())
        else:
            # 如果不是合取，创建随机前提
            num_premises = random.randint(2, min(max_premises, 3))
            premises = [conclusion_expr]  # 包含原结论

            for i in range(num_premises - 1):
                var_id = random.randint(1000, 9999)
                premises.append(z3.Bool(f"ConjPremise_{var_id}"))

            return premises

    def explain_step(self, premises, conclusion):
        """解释推理步骤"""
        premise_strs = [str(p) for p in premises]
        return f"ConjunctionIntroduction: 由于 {' 和 '.join(premise_strs)} 都成立，因此 {conclusion} 成立"

    def get_rule_template(self):
        """获取规则模板"""
        return {
            "name": self.name,
            "pattern": "如果 P 成立且 Q 成立，那么 P 且 Q 成立",
            "formal": "P, Q ⊢ P ∧ Q",
            "example": "如果天在下雨且风在刮，那么天在下雨且风在刮",
            "variables": ["前提P", "前提Q", "合取结论P∧Q"]
        }


class ConjunctionEliminationRule:
    """
    合取消除规则 (Conjunction Elimination)

    规则形式：如果有 P ∧ Q，则可以推出 P（或 Q）
    """

    def __init__(self):
        self.name = "ConjunctionElimination"
        self.description = "合取消除：P ∧ Q ⊢ P"

    def num_premises(self):
        """该规则需要1个前提（合取表达式）"""
        return 1

    def can_apply(self, premises):
        """
        检查是否可以应用该规则

        Args:
            premises: 前提列表，应该包含一个合取表达式

        Returns:
            bool: 是否可以应用
        """
        if len(premises) != 1:
            return False

        premise = premises[0]
        return z3.is_and(premise)

    def construct_conclusion(self, premises):
        """
        根据合取前提构造结论（选择其中一个合取项）

        Args:
            premises: 前提列表，包含一个合取表达式

        Returns:
            z3.ExprRef: 选择的合取项
        """
        if len(premises) != 1:
            raise ValueError("ConjunctionElimination需要恰好1个前提")

        premise = premises[0]

        if not z3.is_and(premise):
            # 如果不是合取，返回新变量
            var_id = random.randint(1000, 9999)
            return z3.Bool(f"ConjElim_Result_{var_id}")

        # 从合取项中随机选择一个
        conjuncts = list(premise.children())
        if conjuncts:
            return random.choice(conjuncts)
        else:
            # 二元合取的情况
            return premise.arg(0) if random.choice([True, False]) else premise.arg(1)

    def generate_premises(self, conclusion_expr, max_premises=1):
        """
        反向生成：给定结论，生成包含该结论的合取前提

        Args:
            conclusion_expr: 结论表达式
            max_premises: 最大前提数量

        Returns:
            list: 包含合取前提的列表
        """
        # 创建包含结论的合取表达式
        num_additional = random.randint(1, 3)
        additional_terms = []

        for i in range(num_additional):
            var_id = random.randint(1000, 9999)
            additional_terms.append(z3.Bool(f"ConjElim_Additional_{var_id}"))

        # 创建合取：conclusion ∧ additional_terms
        all_terms = [conclusion_expr] + additional_terms
        conjunction = z3.And(*all_terms)

        return [conjunction]

    def explain_step(self, premises, conclusion):
        """解释推理步骤"""
        premise = premises[0]
        return f"ConjunctionElimination: 由于 {premise} 成立，其中包含 {conclusion}，因此 {conclusion} 成立"

    def get_rule_template(self):
        """获取规则模板"""
        return {
            "name": self.name,
            "pattern": "如果 P 且 Q 成立，那么 P 成立（或 Q 成立）",
            "formal": "P ∧ Q ⊢ P",
            "example": "如果天在下雨且风在刮，那么天在下雨",
            "variables": ["合取前提P∧Q", "结论P"]
        }