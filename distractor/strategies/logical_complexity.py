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
        # 过滤出简单的布尔变量，避免复杂表达式
        self.available_vars = [var for var in available_vars if self._is_simple_bool_var(var)]

        # 如果过滤后没有变量，创建一些默认变量
        if not self.available_vars:
            self.available_vars = [z3.Bool(f"ComplexVar_{i}") for i in range(3)]

    def _is_simple_bool_var(self, expr):
        """
        检查表达式是否为简单的布尔变量（不是复合表达式）
        """
        try:
            return (z3.is_const(expr) and
                    z3.is_bool(expr) and
                    expr.decl().kind() == z3.Z3_OP_UNINTERPRETED)
        except:
            return False

    def _add_negation_layers(self, expr: z3.BoolRef, layers: int = 1) -> z3.BoolRef:
        """
        添加层层否定，如 Not(Not(expr)) 增加逻辑复杂度
        """
        result = expr
        for _ in range(layers):
            result = z3.Not(result)
        return result

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
            if conclusion is None:
                continue

            try:
                # 确保有足够的变量可选择
                if len(self.available_vars) == 0:
                    continue

                num_to_select = min(2, len(self.available_vars))
                selected_vars = random.sample(self.available_vars, k=num_to_select)

                # 随机生成表达式（以And为主）
                if len(selected_vars) == 1:
                    base_expr = selected_vars[0]
                else:
                    base_expr = z3.And(*selected_vars)

                # 添加否定嵌套
                nested_layers = random.randint(1, max_depth)
                nested_expr = self._add_negation_layers(base_expr, layers=nested_layers)

                distractors.append({
                    "premises_expr": [nested_expr],
                    "conclusion_expr": conclusion,
                    "description": f"逻辑复杂度控制：增加逻辑嵌套层级 {nested_expr}",
                    "strategy": "logical_complexity",
                    "rule": rule,
                    "perturbed_expr": nested_expr
                })

            except Exception as e:
                # 如果生成失败，跳过这个干扰项
                print(f"[DEBUG] LogicalComplexityDistractor 生成失败: {e}")
                continue

        return distractors