# 除法

"""
* RealDivision（实数）：`a = b × c, c ≠ 0 ⊢ a / c = b`
* IntDivision（整数，SMT-LIB 欧几里得除法）：`a = b × c, c ≠ 0 ⊢ (a div c) = b`
"""

import z3
import random
from utils.variable_manager import RuleVariableMixin


class RealDivisionRule(RuleVariableMixin):
    """实数除法规则：从 a = b×c 且 c ≠ 0 推出 a/c = b"""

    def __init__(self):
        self.name = "RealDivision"
        self.description = "实数除法：a = b×c, c ≠ 0 ⊢ a/c = b"

    def num_premises(self):
        # 需要两个前提：一个等式和一个非零条件
        return 2

    def can_apply(self, premises):
        """
        检查是否有两个前提，其中包含一个等式和一个非零条件
        """
        if len(premises) != 2:
            return False

        has_eq = any(z3.is_eq(p) for p in premises)

        # 检查非零条件：通过结构检查而非字符串匹配
        has_nonzero = False
        for p in premises:
            # 检查 Not(x == 0) 形式
            if z3.is_not(p):
                inner = p.arg(0)
                if z3.is_eq(inner) and inner.num_args() == 2:
                    arg0, arg1 = inner.arg(0), inner.arg(1)
                    if (arg1.is_numeral() and arg1.as_long() == 0) or (arg0.is_numeral() and arg0.as_long() == 0):
                        has_nonzero = True
                        break
            # 检查 Distinct(x, 0) 形式
            elif z3.is_distinct(p) and p.num_args() == 2:
                arg0, arg1 = p.arg(0), p.arg(1)
                if (arg1.is_numeral() and arg1.as_long() == 0) or (arg0.is_numeral() and arg0.as_long() == 0):
                    has_nonzero = True
                    break

        return has_eq and has_nonzero

    def construct_conclusion(self, premises):
        """从前提中提取变量并构造除法结论"""
        if len(premises) != 2:
            raise ValueError("RealDivision需要恰好2个前提")

        # 找到等式前提和非零前提
        eq_premise = None
        nonzero_var = None

        for p in premises:
            if z3.is_eq(p):
                eq_premise = p
            else:
                # 从非零条件中提取变量
                if z3.is_not(p):
                    inner = p.arg(0)
                    if z3.is_eq(inner):
                        arg0, arg1 = inner.arg(0), inner.arg(1)
                        if arg1.is_numeral() and arg1.as_long() == 0:
                            nonzero_var = arg0
                        elif arg0.is_numeral() and arg0.as_long() == 0:
                            nonzero_var = arg1
                elif z3.is_distinct(p) and p.num_args() == 2:
                    arg0, arg1 = p.arg(0), p.arg(1)
                    if arg1.is_numeral() and arg1.as_long() == 0:
                        nonzero_var = arg0
                    elif arg0.is_numeral() and arg0.as_long() == 0:
                        nonzero_var = arg1

        if eq_premise is None:
            raise ValueError("未找到等式前提")

        # 解析等式 a = b*c，并确定哪个变量是非零的除数
        lhs = eq_premise.arg(0)
        rhs = eq_premise.arg(1)

        a, b, c = None, None, None

        # 检查 lhs = rhs 其中 rhs 是乘法
        if z3.is_mul(rhs) and rhs.num_args() == 2:
            a = lhs
            factor1, factor2 = rhs.arg(0), rhs.arg(1)
            # 确定哪个因子是非零变量c
            if nonzero_var is not None:
                if z3.eq(factor1, nonzero_var):
                    b, c = factor2, factor1
                elif z3.eq(factor2, nonzero_var):
                    b, c = factor1, factor2
                else:
                    b, c = factor1, factor2  # 默认选择
            else:
                b, c = factor1, factor2
        # 检查 lhs = rhs 其中 lhs 是乘法
        elif z3.is_mul(lhs) and lhs.num_args() == 2:
            a = rhs
            factor1, factor2 = lhs.arg(0), lhs.arg(1)
            if nonzero_var is not None:
                if z3.eq(factor1, nonzero_var):
                    b, c = factor2, factor1
                elif z3.eq(factor2, nonzero_var):
                    b, c = factor1, factor2
                else:
                    b, c = factor1, factor2
            else:
                b, c = factor1, factor2
        else:
            # 如果无法解析，使用默认变量
            a = z3.Real(f"a{random.randint(1, 1000000)}")
            b = z3.Real(f"b{random.randint(1, 1000000)}")
            c = z3.Real(f"c{random.randint(1, 1000000)}")

        # 构造 a/c = b
        return a / c == b

    def generate_premises(self, conclusion_expr=None, max_premises=2):
        """生成前提：a = b*c 和 c ≠ 0"""
        a = z3.Real(f"a{random.randint(1, 1000000)}")
        b = z3.Real(f"b{random.randint(1, 1000000)}")
        c = z3.Real(f"c{random.randint(1, 1000000)}")

        eq_premise = a == b * c
        nonzero_premise = c != 0

        # 随机排列前提顺序
        premises = [eq_premise, nonzero_premise]
        if random.choice([True, False]):
            premises.reverse()

        return premises

    def explain_step(self, premises, conclusion):
        return f"RealDivision: 由于 {premises[0]} 和 {premises[1]}，得出 {conclusion}"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "如果 a = b×c 且 c ≠ 0，那么 a/c = b",
            "formal": "a = b×c, c ≠ 0 ⊢ a/c = b",
            "example": "若 6 = 2×3 且 3 ≠ 0，则 6/3 = 2",
            "variables": ["a", "b", "c"]
        }


class IntDivisionRule(RuleVariableMixin):
    """整数除法规则：从 a = b×c 且 c ≠ 0 推出 (a div c) = b"""

    def __init__(self):
        self.name = "IntDivision"
        self.description = "整数除法：a = b×c, c ≠ 0 ⊢ (a div c) = b"

    def num_premises(self):
        return 2

    def can_apply(self, premises):
        """检查是否有两个前提，其中包含一个等式和一个非零条件"""
        if len(premises) != 2:
            return False

        has_eq = any(z3.is_eq(p) for p in premises)

        # 检查非零条件：通过结构检查而非字符串匹配
        has_nonzero = False
        for p in premises:
            # 检查 Not(x == 0) 形式
            if z3.is_not(p):
                inner = p.arg(0)
                if z3.is_eq(inner) and inner.num_args() == 2:
                    arg0, arg1 = inner.arg(0), inner.arg(1)
                    if (arg1.is_numeral() and arg1.as_long() == 0) or (arg0.is_numeral() and arg0.as_long() == 0):
                        has_nonzero = True
                        break
            # 检查 Distinct(x, 0) 形式
            elif z3.is_distinct(p) and p.num_args() == 2:
                arg0, arg1 = p.arg(0), p.arg(1)
                if (arg1.is_numeral() and arg1.as_long() == 0) or (arg0.is_numeral() and arg0.as_long() == 0):
                    has_nonzero = True
                    break

        return has_eq and has_nonzero

    def construct_conclusion(self, premises):
        """从前提中提取变量并构造整数除法结论"""
        if len(premises) != 2:
            raise ValueError("IntDivision需要恰好2个前提")

        # 找到等式前提和非零前提
        eq_premise = None
        nonzero_var = None

        for p in premises:
            if z3.is_eq(p):
                eq_premise = p
            else:
                # 从非零条件中提取变量
                if z3.is_not(p):
                    inner = p.arg(0)
                    if z3.is_eq(inner):
                        arg0, arg1 = inner.arg(0), inner.arg(1)
                        if arg1.is_numeral() and arg1.as_long() == 0:
                            nonzero_var = arg0
                        elif arg0.is_numeral() and arg0.as_long() == 0:
                            nonzero_var = arg1
                elif z3.is_distinct(p) and p.num_args() == 2:
                    arg0, arg1 = p.arg(0), p.arg(1)
                    if arg1.is_numeral() and arg1.as_long() == 0:
                        nonzero_var = arg0
                    elif arg0.is_numeral() and arg0.as_long() == 0:
                        nonzero_var = arg1

        if eq_premise is None:
            raise ValueError("未找到等式前提")

        # 解析等式 a = b*c，并确定哪个变量是非零的除数
        lhs = eq_premise.arg(0)
        rhs = eq_premise.arg(1)

        a, b, c = None, None, None

        if z3.is_mul(rhs) and rhs.num_args() == 2:
            a = lhs
            factor1, factor2 = rhs.arg(0), rhs.arg(1)
            # 确定哪个因子是非零变量c
            if nonzero_var is not None:
                if z3.eq(factor1, nonzero_var):
                    b, c = factor2, factor1
                elif z3.eq(factor2, nonzero_var):
                    b, c = factor1, factor2
                else:
                    b, c = factor1, factor2  # 默认选择
            else:
                b, c = factor1, factor2
        elif z3.is_mul(lhs) and lhs.num_args() == 2:
            a = rhs
            factor1, factor2 = lhs.arg(0), lhs.arg(1)
            if nonzero_var is not None:
                if z3.eq(factor1, nonzero_var):
                    b, c = factor2, factor1
                elif z3.eq(factor2, nonzero_var):
                    b, c = factor1, factor2
                else:
                    b, c = factor1, factor2
            else:
                b, c = factor1, factor2
        else:
            # 默认变量
            a = z3.Int(f"a{random.randint(1, 1000000)}")
            b = z3.Int(f"b{random.randint(1, 1000000)}")
            c = z3.Int(f"c{random.randint(1, 1000000)}")

        # 使用整数除法运算符
        # 在 Z3Py 中，对整数使用 / 会自动进行整数除法
        return a / c == b

    def generate_premises(self, conclusion_expr=None, max_premises=2):
        """生成前提：a = b*c 和 c ≠ 0"""
        a = z3.Int(f"a{random.randint(1, 1000000)}")
        b = z3.Int(f"b{random.randint(1, 1000000)}")
        c = z3.Int(f"c{random.randint(1, 1000000)}")

        eq_premise = a == b * c
        nonzero_premise = c != 0

        premises = [eq_premise, nonzero_premise]
        if random.choice([True, False]):
            premises.reverse()

        return premises

    def explain_step(self, premises, conclusion):
        return f"IntDivision: 由于 {premises[0]} 和 {premises[1]}，得出 {conclusion}"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "如果 a = b×c 且 c ≠ 0，那么 (a div c) = b",
            "formal": "a = b×c, c ≠ 0 ⊢ (a div c) = b",
            "example": "若 6 = 2×3 且 3 ≠ 0，则 (6 div 3) = 2",
            "variables": ["a", "b", "c"]
        }
