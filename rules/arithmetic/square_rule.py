# SquareNonnegativity（平方非负，ℝ）：`x ∈ ℝ ⊢ x² ≥ 0`

import z3
import random
from utils.variable_manager import RuleVariableMixin


class SquareNonnegativityRule(RuleVariableMixin):
    """平方非负：从 x∈ℝ 推出 x² ≥ 0"""

    def __init__(self):
        self.name = "SquareNonnegativity"
        self.description = "平方非负：x²≥0"

    def num_premises(self):
        # 允许 0 或 1 个前提
        return 1

    def can_apply(self, premises):
        # 接受 0 或 1 个前提
        return len(premises) <= 1

    def construct_conclusion(self, premises):
        # 若提供前提，则尝试从前提获取变量；否则生成新变量
        x = None
        if premises:
            p = premises[0]
            # 如果前提是一个布尔等式或不等式等，尝试从中提取变量
            # 尝试取第一个参数作为 x
            try:
                if p.num_args() >= 1:
                    candidate = p.arg(0)
                    if isinstance(candidate, z3.ArithRef):
                        x = candidate
            except Exception:
                pass
        if x is None:
            # 随机生成实数变量
            x = z3.Real(f"x{random.randint(1, 1_000_000)}")
        # 构造 x*x ≥ 0
        return x * x >= 0

    def generate_premises(self, conclusion_expr, max_premises=1):
        # 允许返回空前提或一个前提，例如 x∈ℝ
        return []

    def explain_step(self, premises, conclusion):
        if premises:
            return f"SquareNonnegativity: 由前提 {premises[0]} 可知 {conclusion}"
        return f"SquareNonnegativity: 根据平方非负性，可得 {conclusion}"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "x²≥0",
            "formal": "x∈ℝ ⊢ x²≥0",
            "example": "对于任意实数 x，总有 x²≥0",
            "variables": ["x"]
        }
