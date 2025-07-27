# 文件：rules/multi_branch_and_rule.py
# 说明：支持多分支结构的 And 合取规则，用于控制逻辑图的分支数（branching）

import z3
import random

class MultiBranchAndRule:
    """
    构造可配置分支数的合取结论，例如：
    premises = [A, B, C, D] => And(And(A, B), And(C, D))
    支持最大分支控制，递归嵌套
    """

    def __init__(self, max_branching=3):
        self.max_branching = max_branching

    def generate_premises(self, conclusion_expr, max_premises=None):
        """
        从一个合取结论中拆解出前提（反向），默认拆分所有子句
        """
        if z3.is_and(conclusion_expr):
            return list(conclusion_expr.children())
        else:
            return [z3.Bool(f"Var_{random.randint(0, 999)}") for _ in range(2)]

    def construct_conclusion(self, premises):
        """
        将多个前提构造为具有指定分支数的嵌套合取结构
        """
        if not premises:
            return z3.BoolVal(True)

        # 控制每层合取的宽度（≤ max_branching）
        return self._build_nested_and(premises, self.max_branching)

    def _build_nested_and(self, items, max_branch):
        """
        将 items 分批打包为 And，每批最多 max_branch 个，递归构造
        """
        if len(items) <= max_branch:
            return z3.And(*items)

        grouped = [items[i:i + max_branch] for i in range(0, len(items), max_branch)]
        nested = [self._build_nested_and(group, max_branch) for group in grouped]
        return self._build_nested_and(nested, max_branch)

    def num_premises(self):
        return self.max_branching