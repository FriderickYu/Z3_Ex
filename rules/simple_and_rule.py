# 文件：rules/simple_and_rule.py
# 说明：模拟规则 A ∧ B => And(A, B)，用于构建最基础的逻辑合取结构

import z3
import random

class SimpleAndRule:
    """
    模拟规则：A ∧ B => And(A, B)，用于构建复杂结论
    """

    def generate_premises(self, conclusion_expr, max_premises=2):
        """
        给定结论 And(A, B)，反向生成前提 A, B
        若结论非 And 类型，则生成随机布尔变量作为前提
        """
        if z3.is_and(conclusion_expr):
            return list(conclusion_expr.children())
        else:
            return [z3.Bool(f"Var_{random.randint(0, 100)}") for _ in range(2)]

    def construct_conclusion(self, premises):
        """
        构造简单合取结构：A, B => And(A, B)
        """
        if not premises:
            return z3.BoolVal(True)
        if len(premises) == 1:
            return premises[0]
        return z3.And(premises[0], premises[1])

    def num_premises(self):
        return 2