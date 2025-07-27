
# 文件：distractor/strategies/logical_complexity.py
# 说明：逻辑复杂度控制型干扰项生成器（Logical Complexity Control）

import random
from typing import List, Dict
import z3


class LogicalComplexityDistractor:
    """
    构造逻辑结构更复杂（如嵌套否定、冗余括号）的干扰项
    """

    def __init__(self, available_vars: List[z3.BoolRef]):
        """
        参数：
            available_vars: 可供组合的布尔变量池
        """
        self.available_vars = available_vars

    def _add_negation_layers(self, expr: z3.BoolRef, layers: int = 1) -> z3.BoolRef:
        """
        添加层层否定，如 Not(Not(expr)) 增加逻辑复杂度
        """
        for _ in range(layers):
            expr = z3.Not(expr)
        return expr

    def generate(self, logical_steps: List[Dict], max_depth: int = 2) -> List[Dict]:
        """
        生成增加逻辑复杂度的干扰项
        参数：
            logical_steps: 原始推理步骤列表
            max_depth: 否定嵌套最大深度
        返回：
            distractors: 构造好的复杂干扰项
        """
        distractors = []

        for step in logical_steps:
            conclusion = step.get("conclusion_expr")
            rule = step.get("rule")
            if not conclusion:
                continue

            # 随机生成表达式（以And为主）并添加否定嵌套
            selected_vars = random.sample(self.available_vars, k=min(2, len(self.available_vars)))
            base_expr = z3.And(*selected_vars) if len(selected_vars) > 1 else selected_vars[0]

            nested_expr = self._add_negation_layers(base_expr, layers=random.randint(1, max_depth))

            distractors.append({
                "premises_expr": [nested_expr],
                "conclusion_expr": conclusion,
                "description": f"逻辑复杂度控制：增加逻辑嵌套层级 {nested_expr}",
                "strategy": "logical_complexity",
                "rule": rule,
                "perturbed_expr": nested_expr
            })

        return distractors