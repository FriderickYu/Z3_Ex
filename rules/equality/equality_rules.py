# 等式推理

"""
* EqualitySubstitution（等式替换）：`a = b, P(a) ⊢ P(b)`
* ReflexivityRule（自反）：`⊢ a = a`
* SymmetryRule（对称）：`a = b ⊢ b = a`
* TransitivityRule（传递）：`a = b, b = c ⊢ a = c` **已存在**
"""

import z3
import random
from utils.variable_manager import RuleVariableMixin


class EqualitySubstitutionRule(RuleVariableMixin):
    """等式替换规则

    根据前提 a = b 和 P(a)，可以推出 P(b)。如果前提顺序相反或等号两边交换，也能识别并
    做出对应替换。仅处理一元谓词的情况。
    """

    def __init__(self):
        self.name = "EqualitySubstitution"
        self.description = "等式替换：a = b, P(a) ⊢ P(b)"

    def num_premises(self):
        return 2

    def _z3_equal(self, a, b):
        """尝试判断两个 Z3 表达式是否等价。"""
        try:
            return z3.is_true(z3.simplify(a == b))
        except Exception:
            return str(a) == str(b)

    def _find_equality_and_pred(self, premises):
        """从前提列表中分离出等式和谓词应用。如果不存在匹配，返回 None。"""
        eq = None
        pred = None
        for p in premises:
            # 判断等式：Z3中等式谓词为 BoolRef with decl().kind() == Z3_OP_EQ
            if (p.decl().kind() == z3.Z3_OP_EQ) and eq is None:
                eq = p
            elif z3.is_app(p) and p.decl().kind() != z3.Z3_OP_EQ and pred is None:
                pred = p
        if eq is not None and pred is not None and pred.num_args() == 1:
            return eq, pred
        # 也可能两个前提互换顺序
        return None

    def can_apply(self, premises):
        return self._find_equality_and_pred(premises) is not None

    def construct_conclusion(self, premises):
        if len(premises) != 2:
            raise ValueError("EqualitySubstitution需要恰好2个前提")
        found = self._find_equality_and_pred(premises)
        if not found:
            # 如果不匹配，返回一个新的结论变量
            return self.create_conclusion_variable()
        eq, pred = found
        lhs, rhs = eq.children()
        arg = pred.arg(0)
        func = pred.decl()
        # 如果谓词应用的参数与等式左侧相同，则替换为等式右侧
        if self._z3_equal(arg, lhs):
            return func(rhs)
        # 如果谓词应用的参数与等式右侧相同，则替换为等式左侧
        if self._z3_equal(arg, rhs):
            return func(lhs)
        # 如果两个参数都不匹配，保持原谓词并返回一个新的变量应用
        return func(arg)

    def generate_premises(self, conclusion_expr, max_premises=2):
        """根据结论反向生成前提。

        如果结论形如 P(b)，则生成两个前提：a = b 和 P(a)，其中 a 为同类型的新常量。
        若结论形如 P(a)，则生成 b = a 和 P(b)。否则随机生成等式和谓词前提。
        """
        # 尝试解析结论为一元谓词应用
        if z3.is_app(conclusion_expr) and conclusion_expr.num_args() == 1:
            try:
                func = conclusion_expr.decl()
                b = conclusion_expr.arg(0)
                # 创建一个新的常量 a 与 b 同类型
                a_name = f"a{random.randint(1, 1_000_000)}"
                a = None
                try:
                    sort = b.sort()
                    a = z3.Const(a_name, sort)
                except Exception:
                    # 默认使用整数类型
                    a = z3.Int(a_name)
                # 构造等式和谓词应用
                eq1 = a == b
                pred1 = func(a)
                # 随机调整前提顺序
                if random.choice([True, False]):
                    return [eq1, pred1]
                else:
                    return [pred1, eq1]
            except Exception:
                pass
        # 默认：生成随机等式和谓词前提
        # 创建两个新的常量 p, q
        sort = z3.IntSort()
        p = z3.Const(f"p{random.randint(1, 1_000_000)}", sort)
        q = z3.Const(f"q{random.randint(1, 1_000_000)}", sort)
        # 谓词函数
        pred = z3.Function(f"P{random.randint(1, 1_000_000)}", sort, z3.BoolSort())
        return [p == q, pred(p)] if random.choice([True, False]) else [pred(p), p == q]

    def explain_step(self, premises, conclusion):
        if len(premises) == 2:
            return f"EqualitySubstitution: 由于 {premises[0]} 和 {premises[1]}，可将等式一侧替换进谓词得到 {conclusion}"
        return f"EqualitySubstitution: 基于前提 {premises} 推出 {conclusion}"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "如果 a = b 且 P(a)，那么 P(b)",
            "formal": "a = b, P(a) ⊢ P(b)",
            "example": "若 2=3 且 f(2) 成立，则 f(3) 成立",
            "variables": ["等式左侧a", "等式右侧b", "谓词P", "前提P(a)", "结论P(b)"]
        }


class ReflexivityRule(RuleVariableMixin):
    """自反性规则

    不需要前提，可以直接推出 a = a，对任何项 a 都成立。这里通过创建一个新常量生成自反式。
    """

    def __init__(self):
        self.name = "ReflexivityRule"
        self.description = "自反：⊢ a = a"

    def num_premises(self):
        return 0

    def can_apply(self, premises):
        return len(premises) == 0

    def construct_conclusion(self, premises):
        if premises:
            raise ValueError("ReflexivityRule 不需要前提")
        # 生成一个新的常量 a 并返回 a = a
        a = z3.Const(f"a{random.randint(1, 1_000_000)}", z3.IntSort())
        return a == a

    def generate_premises(self, conclusion_expr, max_premises=0):
        # 自反规则不需要前提
        return []

    def explain_step(self, premises, conclusion):
        return f"Reflexivity: 任何对象都等于自身，因此 {conclusion}"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "每个对象都等于自身",
            "formal": "⊢ a = a",
            "example": "5 = 5",
            "variables": ["对象a"]
        }


class SymmetryRule(RuleVariableMixin):
    """对称性规则

    根据前提 a = b，可以推出 b = a。
    """

    def __init__(self):
        self.name = "SymmetryRule"
        self.description = "对称：a = b ⊢ b = a"

    def num_premises(self):
        return 1

    def _is_equality(self, expr):
        return expr.decl().kind() == z3.Z3_OP_EQ

    def can_apply(self, premises):
        return len(premises) == 1 and self._is_equality(premises[0])

    def construct_conclusion(self, premises):
        if len(premises) != 1:
            raise ValueError("SymmetryRule需要恰好1个前提")
        eq = premises[0]
        if not self._is_equality(eq):
            # 默认生成新的等式
            x = z3.Const(f"x{random.randint(1, 1_000_000)}", z3.IntSort())
            return x == x
        lhs, rhs = eq.children()
        return rhs == lhs

    def generate_premises(self, conclusion_expr, max_premises=1):
        # 如果结论是 b = a，则前提为 a = b
        if conclusion_expr.decl().kind() == z3.Z3_OP_EQ:
            lhs, rhs = conclusion_expr.children()
            return [rhs == lhs]
        # 默认生成随机等式
        x = z3.Const(f"x{random.randint(1, 1_000_000)}", z3.IntSort())
        y = z3.Const(f"y{random.randint(1, 1_000_000)}", z3.IntSort())
        return [x == y]

    def explain_step(self, premises, conclusion):
        premise = premises[0]
        return f"Symmetry: 由 {premise} 可交换两边得到 {conclusion}"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "如果 a = b，则 b = a",
            "formal": "a = b ⊢ b = a",
            "example": "若 2=3，则 3=2",
            "variables": ["左项a", "右项b"]
        }


class TransitivityRule(RuleVariableMixin):
    """传递性规则

    根据前提 a = b 和 b = c，可以推出 a = c。实现中尝试识别不同顺序和交换形式。
    """

    def __init__(self):
        self.name = "TransitivityRule"
        self.description = "传递：a = b, b = c ⊢ a = c"

    def num_premises(self):
        return 2

    def _is_equality(self, expr):
        return expr.decl().kind() == z3.Z3_OP_EQ

    def _z3_equal(self, a, b):
        """尝试判断两个 Z3 表达式是否等价。"""
        try:
            return z3.is_true(z3.simplify(a == b))
        except Exception:
            return str(a) == str(b)

    def can_apply(self, premises):
        if len(premises) != 2:
            return False
        return all(self._is_equality(p) for p in premises)

    def construct_conclusion(self, premises):
        if len(premises) != 2:
            raise ValueError("TransitivityRule需要恰好2个前提")
        if not all(self._is_equality(p) for p in premises):
            # 返回新的自反等式
            x = z3.Const(f"x{random.randint(1, 1_000_000)}", z3.IntSort())
            return x == x
        (e1, e2) = premises
        a1, b1 = e1.children()
        a2, b2 = e2.children()
        # 判断各种组合 a = b, b = c
        # e1: a1=b1, e2: a2=b2
        # 如果 b1 与 a2 相同，则 a1 = b2
        try:
            if self._z3_equal(b1, a2):
                return a1 == b2
            if self._z3_equal(a1, b2):
                return b1 == a2
            if self._z3_equal(a1, a2):
                return b1 == b2
            if self._z3_equal(b1, b2):
                return a1 == a2
        except Exception:
            pass
        # 尝试交换顺序再判断
        try:
            if self._z3_equal(a1, a2):
                return b1 == b2
            if self._z3_equal(b1, b2):
                return a1 == a2
            if self._z3_equal(a1, b2):
                return b1 == a2
            if self._z3_equal(b1, a2):
                return a1 == b2
        except Exception:
            pass
        # 默认返回自反等式
        x = z3.Const(f"x{random.randint(1, 1_000_000)}", z3.IntSort())
        return x == x

    def generate_premises(self, conclusion_expr, max_premises=2):
        # 如果结论是 a = c，生成 a = b 与 b = c
        if conclusion_expr.decl().kind() == z3.Z3_OP_EQ:
            a, c = conclusion_expr.children()
            # 创建新的中间常量 b
            sort = None
            try:
                sort = a.sort()
            except Exception:
                sort = z3.IntSort()
            b = z3.Const(f"b{random.randint(1, 1_000_000)}", sort)
            eq1 = a == b
            eq2 = b == c
            return [eq1, eq2] if random.choice([True, False]) else [eq2, eq1]
        # 默认随机生成
        sort = z3.IntSort()
        x = z3.Const(f"x{random.randint(1, 1_000_000)}", sort)
        y = z3.Const(f"y{random.randint(1, 1_000_000)}", sort)
        z = z3.Const(f"z{random.randint(1, 1_000_000)}", sort)
        eq1 = x == y
        eq2 = y == z
        return [eq1, eq2] if random.choice([True, False]) else [eq2, eq1]

    def explain_step(self, premises, conclusion):
        if len(premises) == 2:
            return f"Transitivity: 由 {premises[0]} 和 {premises[1]} 推出 {conclusion}（等式传递）"
        return f"Transitivity: 基于前提 {premises} 推出 {conclusion}"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "如果 a = b 且 b = c，那么 a = c",
            "formal": "a = b, b = c ⊢ a = c",
            "example": "若 2=3 且 3=4，则 2=4",
            "variables": ["a", "b", "c"]
        }