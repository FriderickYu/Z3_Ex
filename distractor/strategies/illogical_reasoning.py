# 文件：distractor/strategies/illogical_reasoning.py
# 说明：逻辑不充分型（Illogical Reasoning）干扰项生成器
# 策略：故意缺失一部分前提，造成逻辑链断裂，从而无法推出结论

from typing import List, Dict
import random
import z3


class IllogicalReasoningDistractor:
    """
    逻辑不充分型干扰项生成器：
    在给定推理链 logical_steps 中，故意删减部分前提生成无效推理路径。
    """

    def __init__(self, min_drop: int = 1, max_drop: int = 1):
        """
        参数：
            min_drop: 最少删减几个前提
            max_drop: 最多删减几个前提
        """
        self.min_drop = min_drop
        self.max_drop = max_drop

    def generate(self, logical_steps: List[Dict]) -> List[Dict]:
        """
        针对每个推理步骤，构造缺失部分前提的干扰项
        返回值：
            每条干扰项为 dict，包含：
                - premises_expr: 干扰前提列表（删减后）
                - conclusion_expr: 原结论
                - description: 自然语言描述
                - strategy: 标记使用的策略
        """
        distractors = []

        for step in logical_steps:
            original_premises = step.get("premises_expr", [])
            conclusion = step.get("conclusion_expr")
            rule = step.get("rule")

            if not original_premises or conclusion is None:
                continue

            try:
                # 确保有足够的前提可以删减
                if len(original_premises) <= 1:
                    # 如果只有1个或没有前提，跳过这个步骤
                    continue

                # 计算要删减的数量，确保至少留下1个前提
                max_droppable = len(original_premises) - 1
                actual_max_drop = min(self.max_drop, max_droppable)

                if actual_max_drop < self.min_drop:
                    # 如果无法满足最小删减要求，跳过
                    continue

                num_drop = random.randint(self.min_drop, actual_max_drop)

                # 随机选择要删减的前提
                dropped = random.sample(original_premises, num_drop)
                remaining = [p for p in original_premises if p not in dropped]

                distractors.append({
                    "premises_expr": remaining,
                    "conclusion_expr": conclusion,
                    "description": f"基于不完整前提{len(remaining)}项尝试推出结论",
                    "strategy": "illogical_reasoning",
                    "dropped": dropped,
                    "rule": rule
                })

            except Exception as e:
                # 如果生成失败，跳过这个干扰项
                print(f"[DEBUG] IllogicalReasoningDistractor 生成失败: {e}")
                continue

        return distractors