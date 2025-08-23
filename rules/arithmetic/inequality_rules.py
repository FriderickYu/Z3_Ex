# 不等式与等式推理

"""
* ArithmeticTransitivity（< 传递）：`a < b, b < c ⊢ a < c`
* AdditionPreservesEq（加法保等式）：`a = b ⊢ a + c = b + c`
* MultiplicationPreservesEq（乘法保等式）：`a = b ⊢ a × c = b × c（无需 c ≠ 0）`
* Antisymmetry（反对称）：`a ≤ b, b ≤ a ⊢ a = b`
* LinearInequality（线性不等式）：`a + b ≤ c + d, a ≥ c ⊢ b ≤ d`
"""

import z3
import random
from utils.variable_manager import RuleVariableMixin


class ArithmeticTransitivityRule(RuleVariableMixin):
    """不等式传递性：a<b, b<c ⊢ a<c"""

    def __init__(self):
        self.name = "ArithmeticTransitivity"
        self.description = "不等式传递：a<b, b<c ⊢ a<c"

    def num_premises(self):
        return 2

    def can_apply(self, premises):
        if len(premises) != 2:
            return False
        # 必须有两个严格小于
        return all(z3.is_lt(p) for p in premises)

    def construct_conclusion(self, premises):
        if len(premises) != 2:
            raise ValueError("ArithmeticTransitivity需要2个前提")
        p1, p2 = premises
        # 尝试匹配 a<b 和 b<c
        a1, b1 = p1.arg(0), p1.arg(1)
        a2, b2 = p2.arg(0), p2.arg(1)
        # 如果中间相等（b1==a2），则结论 a1<b2
        if z3.is_true(b1 == a2):
            return a1 < b2
        # 若 p2, p1 顺序相反
        if z3.is_true(b2 == a1):
            return a2 < b1
        # 默认：取第一个前提左侧与第二个前提右侧
        return a1 < b2

    def generate_premises(self, conclusion_expr, max_premises=2):
        # 从 a<c 生成 a<b 和 b<c
        if z3.is_lt(conclusion_expr):
            a, c = conclusion_expr.arg(0), conclusion_expr.arg(1)
            # 创建中间 b
            b = self.create_premise_variable()
            # b 的类型与 a、c 相同
            if hasattr(a, 'sort'):
                sort = a.sort()
                if sort == z3.IntSort():
                    b = z3.Int(f"b{random.randint(1, 1_000_000)}")
                elif sort == z3.RealSort():
                    b = z3.Real(f"b{random.randint(1, 1_000_000)}")
            return [a < b, b < c]
        # 默认生成随机不等式链
        a = z3.Int(f"a{random.randint(1, 1_000_000)}")
        b = z3.Int(f"b{random.randint(1, 1_000_000)}")
        c = z3.Int(f"c{random.randint(1, 1_000_000)}")
        return [a < b, b < c]

    def explain_step(self, premises, conclusion):
        return f"ArithmeticTransitivity: 由 {premises[0]} 和 {premises[1]} 得出 {conclusion}"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "如果 a<b 且 b<c，则 a<c",
            "formal": "a<b, b<c ⊢ a<c",
            "example": "若 1<2 且 2<5，则 1<5",
            "variables": ["a", "b", "c"]
        }


class AdditionPreservesEqRule(RuleVariableMixin):
    """加法保持等式：从 a=b 推出 a+c=b+c"""

    def __init__(self):
        self.name = "AdditionPreservesEq"
        self.description = "加法保等式：a=b ⊢ a+c=b+c"

    def num_premises(self):
        return 1

    def can_apply(self, premises):
        return len(premises) == 1 and z3.is_eq(premises[0])

    def construct_conclusion(self, premises):
        if len(premises) != 1:
            raise ValueError("AdditionPreservesEq需要1个前提")
        eq = premises[0]
        a, b = eq.arg(0), eq.arg(1)
        # 根据变量类型选择 c
        sort = a.sort() if hasattr(a, 'sort') else None
        if sort == z3.IntSort():
            c = z3.Int(f"c{random.randint(1, 1_000_000)}")
        elif sort == z3.RealSort():
            c = z3.Real(f"c{random.randint(1, 1_000_000)}")
        else:
            c = z3.Int(f"c{random.randint(1, 1_000_000)}")
        return a + c == b + c

    def generate_premises(self, conclusion_expr, max_premises=1):
        # 从 a+c=b+c 反推出 a=b
        if z3.is_eq(conclusion_expr):
            lhs, rhs = conclusion_expr.arg(0), conclusion_expr.arg(1)
            if z3.is_add(lhs) and z3.is_add(rhs) and lhs.num_args() == rhs.num_args():
                a = lhs.arg(0)
                b = rhs.arg(0)
                return [a == b]
        # 默认
        a = z3.Int(f"a{random.randint(1, 1_000_000)}")
        b = z3.Int(f"b{random.randint(1, 1_000_000)}")
        return [a == b]

    def explain_step(self, premises, conclusion):
        return f"AdditionPreservesEq: 由 {premises[0]} 可得 {conclusion}"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "如果 a=b，则 a+c=b+c",
            "formal": "a=b ⊢ a+c=b+c",
            "example": "若 3=3，则 3+4 = 3+4",
            "variables": ["a", "b", "c"]
        }


class MultiplicationPreservesEqRule(RuleVariableMixin):
    """乘法保持等式：从 a=b 推出 a×c=b×c"""

    def __init__(self):
        self.name = "MultiplicationPreservesEq"
        self.description = "乘法保等式：a=b ⊢ a×c=b×c"

    def num_premises(self):
        return 1

    def can_apply(self, premises):
        return len(premises) == 1 and z3.is_eq(premises[0])

    def construct_conclusion(self, premises):
        if len(premises) != 1:
            raise ValueError("MultiplicationPreservesEq需要1个前提")
        eq = premises[0]
        a, b = eq.arg(0), eq.arg(1)
        sort = a.sort() if hasattr(a, 'sort') else None
        if sort == z3.IntSort():
            c = z3.Int(f"c{random.randint(1, 1_000_000)}")
        elif sort == z3.RealSort():
            c = z3.Real(f"c{random.randint(1, 1_000_000)}")
        else:
            c = z3.Int(f"c{random.randint(1, 1_000_000)}")
        return a * c == b * c

    def generate_premises(self, conclusion_expr, max_premises=1):
        if z3.is_eq(conclusion_expr):
            lhs, rhs = conclusion_expr.arg(0), conclusion_expr.arg(1)
            if z3.is_mul(lhs) and z3.is_mul(rhs) and lhs.num_args() == rhs.num_args():
                a = lhs.arg(0)
                b = rhs.arg(0)
                return [a == b]
        a = z3.Int(f"a{random.randint(1, 1_000_000)}")
        b = z3.Int(f"b{random.randint(1, 1_000_000)}")
        return [a == b]

    def explain_step(self, premises, conclusion):
        return f"MultiplicationPreservesEq: 由 {premises[0]} 可得 {conclusion}"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "如果 a=b，则 a×c=b×c",
            "formal": "a=b ⊢ a×c=b×c",
            "example": "若 3=3，则 3×5 = 3×5",
            "variables": ["a", "b", "c"]
        }


class AntisymmetryRule(RuleVariableMixin):
    """不等式的反对称性：从 a≤b 与 b≤a 推出 a=b"""

    def __init__(self):
        self.name = "Antisymmetry"
        self.description = "反对称性：a≤b, b≤a ⊢ a=b"

    def num_premises(self):
        return 2

    def can_apply(self, premises):
        if len(premises) != 2:
            return False
        return all(z3.is_le(p) for p in premises)

    def construct_conclusion(self, premises):
        if len(premises) != 2:
            raise ValueError("Antisymmetry需要2个前提")
        p1, p2 = premises
        a1, b1 = p1.arg(0), p1.arg(1)
        a2, b2 = p2.arg(0), p2.arg(1)
        # 如果 a1=b2 且 b1=a2，则结论是 a1=b1
        # 这种情况下它们互为对称
        return a1 == b1 if z3.is_true(a1 == b2 and b1 == a2) else a1 == b1

    def generate_premises(self, conclusion_expr, max_premises=2):
        if z3.is_eq(conclusion_expr):
            a, b = conclusion_expr.arg(0), conclusion_expr.arg(1)
            return [a <= b, b <= a]
        # 默认
        a = z3.Int(f"a{random.randint(1, 1_000_000)}")
        b = z3.Int(f"b{random.randint(1, 1_000_000)}")
        return [a <= b, b <= a]

    def explain_step(self, premises, conclusion):
        return f"Antisymmetry: 由 {premises[0]} 和 {premises[1]} 得出 {conclusion}"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "如果 a≤b 且 b≤a，则 a=b",
            "formal": "a≤b, b≤a ⊢ a=b",
            "example": "若 2≤2 且 2≤2，则 2=2",
            "variables": ["a", "b"]
        }


class LinearInequalityRule(RuleVariableMixin):
    """线性不等式推理：a+b≤c+d, a≥c ⊢ b≤d"""

    def __init__(self):
        self.name = "LinearInequality"
        self.description = "线性不等式：a+b≤c+d, a≥c ⊢ b≤d"

    def num_premises(self):
        return 2

    def can_apply(self, premises):
        if len(premises) != 2:
            return False
        # 一个是 ≤ (两个加数), 一个是 ≥
        has_le = False
        has_ge = False
        for p in premises:
            if z3.is_le(p):
                has_le = True
            # z3 没有 is_ge, 用 <= 判断反转；若 p 是 ≥，形式为 b <= a
            if p.decl().kind() == z3.Z3_OP_LE:
                # 反向的 a≥c 写成 c <= a
                pass
            # 粗略标记 ge 条件
            if ">=" in str(p):
                has_ge = True
        return has_le and has_ge

    def construct_conclusion(self, premises):
        if len(premises) != 2:
            raise ValueError("LinearInequality需要2个前提")
        le = None
        ge = None
        for p in premises:
            # 识别形如 a+b ≤ c+d
            if z3.is_le(p):
                le = p
            else:
                ge = p
        # 默认变量
        b = z3.Int(f"b{random.randint(1, 1_000_000)}")
        d = z3.Int(f"d{random.randint(1, 1_000_000)}")
        if le is not None and ge is not None and z3.is_add(le.arg(0)) and z3.is_add(le.arg(1)):
            # le: a+b ≤ c+d
            a, b_var = le.arg(0).arg(0), le.arg(0).arg(1)
            c, d_var = le.arg(1).arg(0), le.arg(1).arg(1)
            # ge: a≥c 写作 c ≤ a 或 a >= c
            # 这里不严密检查，只要求变量一致
            return b_var <= d_var
        return b <= d

    def generate_premises(self, conclusion_expr, max_premises=2):
        # 从 b≤d 生成 a+b≤c+d 与 a≥c
        b = z3.Int(f"b{random.randint(1, 1_000_000)}")
        d = z3.Int(f"d{random.randint(1, 1_000_000)}")
        a = z3.Int(f"a{random.randint(1, 1_000_000)}")
        c = z3.Int(f"c{random.randint(1, 1_000_000)}")
        le = (a + b) <= (c + d)
        ge = a >= c
        return [le, ge]

    def explain_step(self, premises, conclusion):
        return f"LinearInequality: 由 {premises[0]} 和 {premises[1]} 推出 {conclusion}"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "如果 a+b≤c+d 且 a≥c，则 b≤d",
            "formal": "a+b≤c+d, a≥c ⊢ b≤d",
            "example": "若 1+4≤3+7 且 1≥3（假设），则 4≤7",
            "variables": ["a", "b", "c", "d"]
        }