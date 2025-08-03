# 文件：rules/universal_instantiation.py
# 说明：全称量词实例化规则
# UniversalInstantiation: ∀x P(x) ⊢ P(a)

import z3
import random


class UniversalInstantiationRule:
    """
    全称量词实例化规则 (Universal Instantiation)

    规则形式：如果对所有x都有P(x)，则对特定的a有P(a)
    """

    def __init__(self):
        self.name = "UniversalInstantiation"
        self.description = "全称实例化：∀x P(x) ⊢ P(a)"

    def num_premises(self):
        """该规则需要1个前提（全称量化表达式）"""
        return 1

    def can_apply(self, premises):
        """检查是否可以应用该规则"""
        if len(premises) != 1:
            return False

        premise = premises[0]
        # 简化版本：检查是否包含"forall"关键字或者是否为全称表达式
        return z3.is_quantifier(premise) and premise.is_forall() if hasattr(premise,
                                                                            'is_forall') else self._is_universal_like(
            premise)

    def _is_universal_like(self, expr):
        """检查表达式是否类似全称量词"""
        expr_str = str(expr).lower()
        return 'forall' in expr_str or 'all' in expr_str

    def construct_conclusion(self, premises):
        """根据全称前提构造特定实例的结论"""
        if len(premises) != 1:
            raise ValueError("UniversalInstantiation需要恰好1个前提")

        premise = premises[0]

        # 创建特定实例
        instance_var = z3.Bool(f"UI_Instance_{random.randint(1000, 9999)}")
        return instance_var

    def generate_premises(self, conclusion_expr, max_premises=1):
        """反向生成：从特定实例生成全称前提"""
        universal_var = z3.Bool(f"UI_Universal_{random.randint(1000, 9999)}")
        return [universal_var]

    def explain_step(self, premises, conclusion):
        """解释推理步骤"""
        premise = premises[0]
        return f"UniversalInstantiation: 由于 {premise} 对所有情况成立，因此 {conclusion} 对特定情况也成立"

    def get_rule_template(self):
        """获取规则模板"""
        return {
            "name": self.name,
            "pattern": "如果对所有x都有P(x)，那么对特定的a有P(a)",
            "formal": "∀x P(x) ⊢ P(a)",
            "example": "如果所有人都会死，那么苏格拉底会死",
            "variables": ["全称量词∀x", "谓词P(x)", "特定实例a"]
        }