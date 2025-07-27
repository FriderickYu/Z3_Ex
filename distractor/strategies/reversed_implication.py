# 文件：distractors/reversed_implication.py
# 说明：逻辑方向反转型干扰项生成器（Reversed Implication）

from typing import List, Dict
import z3


class ReversedImplicationDistractor:
    """
    逻辑方向错置型干扰项生成器：
    - 将前提与结论的逻辑方向反转
    - 如 A ∧ B → C 替换为 C → A ∧ B
    """

    def generate(self, logical_steps: List[Dict]) -> List[Dict]:
        """
        针对每个推理步骤生成反向推理干扰项
        返回值：
            每条干扰项为 dict，包含：
                - premises_expr: 干扰前提列表
                - conclusion_expr: 原结论
                - description: 自然语言描述
                - strategy: 标记使用的策略
        """
        distractors = []

        for step in logical_steps:
            premises = step.get("premises_expr")
            conclusion = step.get("conclusion_expr")
            rule = step.get("rule")

            if not premises or conclusion is None:
                continue

            # 多前提暂不支持，默认只取第一个前提项
            premise_expr = premises[0]

            # 构造反向蕴含表达式：conclusion → premise_expr
            reversed_implication = z3.Implies(conclusion, premise_expr)

            distractors.append({
                "premises_expr": [reversed_implication],
                "conclusion_expr": conclusion,
                "description": "反向推理：将原推理方向 C ← A∧B 反转为 A∧B ← C",
                "strategy": "reversed_implication",
                "rule": rule,
                "original_expr": premise_expr,
                "perturbed_expr": reversed_implication
            })

        return distractors
