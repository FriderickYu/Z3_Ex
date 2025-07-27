# 文件：rules/balanced_and_rule.py
# 说明：构造平衡结构的 And 合取规则，用于生成结构对称、深度受控的逻辑树

import z3
import random

class BalancedAndRule:
    """
    多前提合取规则：将多个前提平衡构造为 And 表达式树
    示例：A, B, C, D => And(And(A, B), And(C, D))
    """

    def generate_premises(self, conclusion_expr, max_premises=4):
        """
        给定一个结论 expr，反向生成若干前提（只适用于 And 类型）
        """
        if z3.is_and(conclusion_expr):
            return list(conclusion_expr.children())
        else:
            return [z3.Bool(f"Var_{random.randint(0, 999)}") for _ in range(2)]

    def construct_conclusion(self, premises):
        """
        由多个前提构造平衡二叉 And 合取表达式
        """
        if not premises:
            return z3.BoolVal(True)
        return self._build_balanced_and(premises)

    def _build_balanced_and(self, items):
        """
        递归构造平衡的 And 合取结构
        """
        if len(items) == 1:
            return items[0]
        mid = len(items) // 2
        left = self._build_balanced_and(items[:mid])
        right = self._build_balanced_and(items[mid:])
        return z3.And(left, right)

    def num_premises(self):
        return 4