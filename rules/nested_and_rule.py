# 文件：rules/nested_and_rule.py
# 说明：嵌套多元合取规则 A ∧ B ∧ C ∧ D => 嵌套 And(...)

import z3
import random

class NestedAndRule:
    """
    多变量嵌套合取规则，例如 A ∧ B ∧ C => And(A, And(B, C))
    """
    def generate_premises(self, conclusion_expr, max_premises=5):
        """
        给定一个结论 And(...)，反向生成多个前提（≥3）
        如果不是 And 类型，返回 max_premises 个新变量
        """
        if z3.is_and(conclusion_expr):
            return list(conclusion_expr.children())
        else:
            n = random.randint(3, max_premises)
            return [z3.Bool(f"Var_{random.randint(0, 999)}") for _ in range(n)]

    def construct_conclusion(self, premises):
        """
        由多个前提构造嵌套的 And 合取结构
        """
        if not premises:
            return z3.BoolVal(True)
        expr = premises[0]
        for p in premises[1:]:
            expr = z3.And(expr, p)
        return expr

    def num_premises(self):
        return 4