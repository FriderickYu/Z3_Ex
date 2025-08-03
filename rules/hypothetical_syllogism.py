# 文件：rules/hypothetical_syllogism.py
# 说明：假言三段论规则
# HypotheticalSyllogism: P → Q, Q → R ⊢ P → R

import z3
import random


class HypotheticalSyllogismRule:
    """
    假言三段论规则 (Hypothetical Syllogism)

    规则形式：如果有 P → Q 和 Q → R，则可以推出 P → R
    这是传递性推理的体现，是构建推理链的重要规则
    """

    def __init__(self):
        self.name = "HypotheticalSyllogism"
        self.description = "假言三段论：P → Q, Q → R ⊢ P → R"

    def num_premises(self):
        """该规则需要2个前提（两个蕴含关系）"""
        return 2

    def can_apply(self, premises):
        """
        检查是否可以应用该规则

        Args:
            premises: 前提列表，应该包含两个蕴含关系

        Returns:
            bool: 是否可以应用该规则
        """
        if len(premises) != 2:
            return False

        p1, p2 = premises

        # 检查两个前提都是蕴含关系
        if not (z3.is_implies(p1) and z3.is_implies(p2)):
            return False

        # 检查是否有共同的中间项
        # p1: A → B, p2: B → C 或 p1: B → C, p2: A → B
        antecedent1 = p1.arg(0)  # P → Q 的 P
        consequent1 = p1.arg(1)  # P → Q 的 Q
        antecedent2 = p2.arg(0)  # Q → R 的 Q
        consequent2 = p2.arg(1)  # Q → R 的 R

        # 情况1：p1的后件是p2的前件 (P → Q, Q → R)
        if self._z3_equal(consequent1, antecedent2):
            return True

        # 情况2：p2的后件是p1的前件 (Q → R, P → Q)
        if self._z3_equal(consequent2, antecedent1):
            return True

        return False

    def _z3_equal(self, expr1, expr2):
        """检查两个Z3表达式是否相等"""
        try:
            return z3.simplify(expr1) == z3.simplify(expr2)
        except:
            return str(expr1) == str(expr2)

    def construct_conclusion(self, premises):
        """
        根据前提构造结论

        Args:
            premises: 前提列表 [P → Q, Q → R] 或 [Q → R, P → Q]

        Returns:
            z3.ExprRef: 结论 P → R
        """
        if len(premises) != 2:
            raise ValueError("HypotheticalSyllogism需要恰好2个前提")

        p1, p2 = premises

        if not (z3.is_implies(p1) and z3.is_implies(p2)):
            # 如果不是蕴含关系，创建新的结论
            return self._create_new_implication()

        antecedent1 = p1.arg(0)
        consequent1 = p1.arg(1)
        antecedent2 = p2.arg(0)
        consequent2 = p2.arg(1)

        # 情况1：p1: P → Q, p2: Q → R，结论：P → R
        if self._z3_equal(consequent1, antecedent2):
            return z3.Implies(antecedent1, consequent2)

        # 情况2：p1: Q → R, p2: P → Q，结论：P → R
        if self._z3_equal(consequent2, antecedent1):
            return z3.Implies(antecedent2, consequent1)

        # 如果没有匹配的中间项，创建新的蕴含关系
        return self._create_new_implication()

    def _create_new_implication(self):
        """创建新的蕴含关系"""
        var_id1 = random.randint(1000, 9999)
        var_id2 = random.randint(1000, 9999)
        antecedent = z3.Bool(f"HS_Antecedent_{var_id1}")
        consequent = z3.Bool(f"HS_Consequent_{var_id2}")
        return z3.Implies(antecedent, consequent)

    def generate_premises(self, conclusion_expr, max_premises=2):
        """
        反向生成：给定结论 P → R，生成前提 P → Q 和 Q → R

        Args:
            conclusion_expr: 结论表达式 P → R
            max_premises: 最大前提数量

        Returns:
            list: 前提列表 [P → Q, Q → R]
        """
        if not z3.is_implies(conclusion_expr):
            # 如果结论不是蕴含关系，创建随机前提
            return self._generate_random_premises()

        antecedent = conclusion_expr.arg(0)  # P
        consequent = conclusion_expr.arg(1)  # R

        # 创建中间变量 Q
        intermediate_var = z3.Bool(f"HS_Intermediate_{random.randint(1000, 9999)}")

        # 构造前提：P → Q 和 Q → R
        premise1 = z3.Implies(antecedent, intermediate_var)
        premise2 = z3.Implies(intermediate_var, consequent)

        return [premise1, premise2]

    def _generate_random_premises(self):
        """生成随机的蕴含前提"""
        var_id1 = random.randint(1000, 9999)
        var_id2 = random.randint(1000, 9999)
        var_id3 = random.randint(1000, 9999)

        P = z3.Bool(f"HS_P_{var_id1}")
        Q = z3.Bool(f"HS_Q_{var_id2}")
        R = z3.Bool(f"HS_R_{var_id3}")

        return [z3.Implies(P, Q), z3.Implies(Q, R)]

    def explain_step(self, premises, conclusion):
        """解释推理步骤"""
        if len(premises) == 2:
            p1, p2 = premises
            return f"HypotheticalSyllogism: 由于有 {p1} 且有 {p2}，通过传递性可以推出 {conclusion}"
        else:
            return f"HypotheticalSyllogism: 基于前提 {premises}，通过传递性推出 {conclusion}"

    def get_rule_template(self):
        """获取规则模板"""
        return {
            "name": self.name,
            "pattern": "如果 P 蕴含 Q，且 Q 蕴含 R，那么 P 蕴含 R",
            "formal": "P → Q, Q → R ⊢ P → R",
            "example": "如果天下雨就会积水，积水就会影响交通，那么天下雨就会影响交通",
            "variables": ["前提P", "中间项Q", "结论R", "蕴含关系"],
            "logical_property": "传递性"
        }

    def extract_chain_elements(self, premises):
        """
        从前提中提取推理链的元素

        Args:
            premises: 前提列表

        Returns:
            dict: 包含推理链元素的字典
        """
        if len(premises) != 2:
            return None

        p1, p2 = premises

        if not (z3.is_implies(p1) and z3.is_implies(p2)):
            return None

        antecedent1 = p1.arg(0)
        consequent1 = p1.arg(1)
        antecedent2 = p2.arg(0)
        consequent2 = p2.arg(1)

        # 确定推理链的顺序
        if self._z3_equal(consequent1, antecedent2):
            # p1: P → Q, p2: Q → R
            return {
                "first": antecedent1,  # P
                "middle": consequent1,  # Q
                "last": consequent2,  # R
                "chain": f"{antecedent1} → {consequent1} → {consequent2}"
            }
        elif self._z3_equal(consequent2, antecedent1):
            # p1: Q → R, p2: P → Q
            return {
                "first": antecedent2,  # P
                "middle": consequent2,  # Q
                "last": consequent1,  # R
                "chain": f"{antecedent2} → {consequent2} → {consequent1}"
            }
        else:
            return None

    def is_valid_chain(self, premises):
        """检查前提是否构成有效的推理链"""
        return self.extract_chain_elements(premises) is not None

    def get_transitivity_info(self):
        """获取传递性相关信息"""
        return {
            "property": "transitivity",
            "domain": "implication",
            "formula": "∀P,Q,R: (P→Q ∧ Q→R) → (P→R)",
            "applications": [
                "因果关系推理",
                "逻辑链构建",
                "条件传递",
                "规则组合"
            ]
        }