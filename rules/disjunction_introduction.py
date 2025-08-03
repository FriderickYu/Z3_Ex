# 文件：rules/disjunction_introduction.py
# 说明：析取引入规则
# DisjunctionIntroduction: P ⊢ P ∨ Q

import z3
import random


class DisjunctionIntroductionRule:
    """
    析取引入规则 (Disjunction Introduction)

    规则形式：如果有 P，则可以推出 P ∨ Q（其中Q是任意命题）
    这个规则体现了逻辑中的弱化原理：从强的结论可以推出弱的结论
    """

    def __init__(self):
        self.name = "DisjunctionIntroduction"
        self.description = "析取引入：P ⊢ P ∨ Q"

    def num_premises(self):
        """该规则需要1个前提"""
        return 1

    def can_apply(self, premises):
        """
        检查是否可以应用该规则

        Args:
            premises: 前提列表

        Returns:
            bool: 总是可以应用（任何命题都可以析取引入）
        """
        return len(premises) == 1

    def construct_conclusion(self, premises):
        """
        根据前提构造析取结论

        Args:
            premises: 前提列表 [P]

        Returns:
            z3.ExprRef: 析取结论 P ∨ Q
        """
        if len(premises) != 1:
            raise ValueError("DisjunctionIntroduction需要恰好1个前提")

        premise = premises[0]

        # 创建一个新的变量作为析取的第二部分
        additional_var = self._create_additional_variable()

        # 随机决定是 P ∨ Q 还是 Q ∨ P
        if random.choice([True, False]):
            return z3.Or(premise, additional_var)
        else:
            return z3.Or(additional_var, premise)

    def _create_additional_variable(self):
        """创建析取中的附加变量"""
        var_id = random.randint(1000, 9999)
        return z3.Bool(f"DisjIntro_Additional_{var_id}")

    def construct_conclusion_with_target(self, premises, target_var):
        """
        构造包含特定目标变量的析取结论

        Args:
            premises: 前提列表 [P]
            target_var: 目标变量 Q

        Returns:
            z3.ExprRef: 析取结论 P ∨ Q
        """
        if len(premises) != 1:
            raise ValueError("DisjunctionIntroduction需要恰好1个前提")

        premise = premises[0]
        return z3.Or(premise, target_var)

    def generate_premises(self, conclusion_expr, max_premises=1):
        """
        反向生成：给定析取结论，提取其中一个析取项作为前提

        Args:
            conclusion_expr: 析取结论 P ∨ Q
            max_premises: 最大前提数量

        Returns:
            list: 前提列表
        """
        if z3.is_or(conclusion_expr):
            # 如果结论是析取，随机选择一个析取项作为前提
            disjuncts = list(conclusion_expr.children())
            if disjuncts:
                chosen_premise = random.choice(disjuncts)
                return [chosen_premise]
            else:
                # 二元析取的情况
                if random.choice([True, False]):
                    return [conclusion_expr.arg(0)]
                else:
                    return [conclusion_expr.arg(1)]
        else:
            # 如果结论不是析取，创建一个新的前提
            var_id = random.randint(1000, 9999)
            premise = z3.Bool(f"DisjIntro_Premise_{var_id}")
            return [premise]

    def explain_step(self, premises, conclusion):
        """解释推理步骤"""
        premise = premises[0]
        return f"DisjunctionIntroduction: 由于 {premise} 成立，因此 {conclusion} 也成立（析取弱化）"

    def get_rule_template(self):
        """获取规则模板"""
        return {
            "name": self.name,
            "pattern": "如果 P 成立，那么 P 或 Q 成立",
            "formal": "P ⊢ P ∨ Q",
            "example": "如果天在下雨，那么天在下雨或者天气晴朗",
            "variables": ["前提P", "附加变量Q", "析取结论P∨Q"],
            "note": "这是逻辑弱化的体现：从强的条件可以推出弱的条件"
        }

    def create_meaningful_disjunction(self, premise, context_vars):
        """
        创建有意义的析取结论

        Args:
            premise: 已知前提
            context_vars: 上下文中的其他变量

        Returns:
            z3.ExprRef: 有意义的析取结论
        """
        if context_vars:
            # 从上下文变量中选择一个作为析取项
            additional_var = random.choice(context_vars)
        else:
            # 创建新变量
            additional_var = self._create_additional_variable()

        return z3.Or(premise, additional_var)

    def get_logical_strength(self):
        """
        获取规则的逻辑强度信息

        Returns:
            dict: 逻辑强度信息
        """
        return {
            "type": "weakening",
            "direction": "premise_to_weaker_conclusion",
            "strength_change": "stronger → weaker",
            "description": "析取引入使结论变弱，因为增加了更多可能性"
        }