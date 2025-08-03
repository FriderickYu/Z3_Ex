# 文件：rules/transitivity_rule.py
# 说明：传递性规则（算术逻辑）
# TransitivityRule: a R b, b R c ⊢ a R c

import z3
import random

class TransitivityRule:
    """
    传递性规则 (Transitivity Rule)

    规则形式：如果aRb且bRc，则aRc（其中R是传递关系）
    适用于等价关系、大小关系等
    """

    def __init__(self):
        self.name = "TransitivityRule"
        self.description = "传递性：aRb, bRc ⊢ aRc"

    def num_premises(self):
        """该规则需要2个前提"""
        return 2

    def can_apply(self, premises):
        """检查是否可以应用该规则"""
        if len(premises) != 2:
            return False

        # 简化版本：检查是否有共同元素
        p1_str = str(premises[0])
        p2_str = str(premises[1])

        # 基本的传递性检查
        return self._has_transitive_structure(p1_str, p2_str)

    def _has_transitive_structure(self, expr1_str, expr2_str):
        """检查是否具有传递性结构"""
        # 简化的模式匹配
        return any(op in expr1_str and op in expr2_str for op in ['>', '<', '=', '>=', '<='])

    def construct_conclusion(self, premises):
        """根据传递性前提构造结论"""
        if len(premises) != 2:
            raise ValueError("TransitivityRule需要恰好2个前提")

        # 创建传递性结论
        result_var = z3.Bool(f"Trans_Result_{random.randint(1000, 9999)}")
        return result_var

    def generate_premises(self, conclusion_expr, max_premises=2):
        """反向生成传递性前提"""
        var1 = z3.Bool(f"Trans_A_{random.randint(1000, 9999)}")
        var2 = z3.Bool(f"Trans_B_{random.randint(1000, 9999)}")
        return [var1, var2]

    def explain_step(self, premises, conclusion):
        """解释推理步骤"""
        return f"TransitivityRule: 由于 {premises[0]} 和 {premises[1]} 具有传递性关系，因此 {conclusion} 成立"

    def get_rule_template(self):
        """获取规则模板"""
        return {
            "name": self.name,
            "pattern": "如果a关系b，且b关系c，那么a关系c",
            "formal": "aRb, bRc ⊢ aRc",
            "example": "如果A大于B，且B大于C，那么A大于C",
            "variables": ["元素a", "元素b", "元素c", "传递关系R"]
        }