# 文件：distractor/strategies/adversarial_structure.py
# 说明：结构相似型（Adversarial Structure-Based）干扰项生成器
# 策略：扰动变量顺序、逻辑连接方式或推理方向，制造结构极相似但逻辑错误的干扰项

from typing import List, Dict
import random
import z3


class AdversarialStructureDistractor:
    """
    结构相似型干扰项生成器：
    - 使用相同变量构造结构相似但逻辑错误的前提
    - 如调换变量顺序、推理方向、连接方式等
    """

    def generate(self, logical_steps: List[Dict]) -> List[Dict]:
        """
        针对每个推理步骤生成结构扰动型干扰项。
        返回值：
            每条干扰项为 dict，包含：
                - premises_expr: 干扰前提列表
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
                # 只扰动第一个前提表达式（可扩展为多个）
                for expr in original_premises:
                    if z3.is_app_of(expr, z3.Z3_OP_AND) or z3.is_app_of(expr, z3.Z3_OP_OR):
                        children = list(expr.children())
                        if len(children) < 2:
                            continue

                        # 1. 对变量顺序打乱
                        shuffled = children[:]
                        random.shuffle(shuffled)

                        # 2. 构造扰动表达式（例如 AND -> OR 或反推）
                        if z3.is_app_of(expr, z3.Z3_OP_AND):
                            perturbed = z3.Or(*shuffled)
                        else:
                            perturbed = z3.And(*shuffled)

                        distractors.append({
                            "premises_expr": [perturbed],
                            "conclusion_expr": conclusion,
                            "description": "扰动结构：连接词或变量顺序被替换",
                            "strategy": "adversarial_structure",
                            "rule": rule,
                            "original_expr": expr,
                            "perturbed_expr": perturbed
                        })
                        break  # 每个步骤生成一个干扰项即可

            except Exception as e:
                # 如果生成失败，跳过这个干扰项
                print(f"[DEBUG] AdversarialStructureDistractor 生成失败: {e}")
                continue

        return distractors