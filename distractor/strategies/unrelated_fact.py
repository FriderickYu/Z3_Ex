# 文件：distractor/strategies/unrelated_fact.py
# 说明：无关事实型干扰项生成器（Unrelated Fact）

import random
from typing import List, Dict
import z3


class UnrelatedFactDistractor:
    """
    生成与主推理链无关的随机组合前提，用于构造干扰项
    """

    def __init__(self, available_vars: List[z3.BoolRef]):
        """
        参数：
            available_vars: 可供组合的布尔变量池，需来自主 DAG 构造过程
        """
        # 过滤出简单的布尔变量，避免复杂表达式
        self.available_vars = [var for var in available_vars if self._is_simple_bool_var(var)]

        # 如果过滤后没有变量，创建一些默认变量
        if not self.available_vars:
            self.available_vars = [z3.Bool(f"UnrelatedVar_{i}") for i in range(3)]

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

    def generate(self, logical_steps: List[Dict], num_facts: int = 1) -> List[Dict]:
        """
        生成若干条无关前提构成的干扰项
        每条干扰项结构为：
            {
                'premises_expr': [...],
                'conclusion_expr': 原始结论,
                'description': 干扰项自然语言说明,
                'strategy': 'unrelated_fact',
                ...
            }
        """
        distractors = []

        for step in logical_steps:
            conclusion = step.get("conclusion_expr")
            rule = step.get("rule")

            if conclusion is None:
                continue

            # 从 available_vars 中随机选择若干无关组合
            for _ in range(num_facts):
                try:
                    # 确保有足够的变量可选择
                    if len(self.available_vars) == 0:
                        continue

                    num_to_select = min(2, len(self.available_vars))
                    selected_vars = random.sample(self.available_vars, k=num_to_select)

                    # 构造无关表达式
                    if len(selected_vars) == 1:
                        unrelated_expr = selected_vars[0]
                    else:
                        unrelated_expr = z3.And(*selected_vars)

                    distractors.append({
                        "premises_expr": [unrelated_expr],
                        "conclusion_expr": conclusion,
                        "description": f"无关事实：与主推理链无关的事实组合 {unrelated_expr}",
                        "strategy": "unrelated_fact",
                        "rule": rule,
                        "perturbed_expr": unrelated_expr
                    })

                except Exception as e:
                    # 如果生成失败，跳过这个干扰项
                    print(f"[DEBUG] UnrelatedFactDistractor 生成失败: {e}")
                    continue

        return distractors