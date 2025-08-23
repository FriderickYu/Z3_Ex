"""
* ModularAddition（加法同余）：`a ≡ b (mod n), c ≡ d (mod n) ⊢ a + c ≡ b + d (mod n)`
* ModularTransitivity（同余传递）：`a ≡ b (mod n), b ≡ c (mod n) ⊢ a ≡ c (mod n)`
"""

import z3
import random
from utils.variable_manager import RuleVariableMixin


def _extract_congruence(eq_expr):
    """从等式表达式中提取同余三元组 (a, b, n)。

    支持两种形式：
    1. a % n == b % n
    2. (a - b) % n == 0
    返回 (a, b, n) 或 (None, None, None) 如果无法解析。
    """
    if not z3.is_eq(eq_expr):
        return None, None, None

    lhs, rhs = eq_expr.arg(0), eq_expr.arg(1)

    # 形式1：a % n == b % n
    if (z3.is_app(lhs) and lhs.decl().kind() == z3.Z3_OP_MOD and
            z3.is_app(rhs) and rhs.decl().kind() == z3.Z3_OP_MOD):
        a, n1 = lhs.arg(0), lhs.arg(1)
        b, n2 = rhs.arg(0), rhs.arg(1)
        # 检查模数是否相等（通过简化判断）
        if z3.simplify(n1 == n2).is_true():
            return a, b, n1

    # 形式2：(a - b) % n == 0
    elif (z3.is_app(lhs) and lhs.decl().kind() == z3.Z3_OP_MOD and
          rhs.is_numeral() and rhs.as_long() == 0):
        diff_expr = lhs.arg(0)
        n = lhs.arg(1)
        # 检查是否是减法表达式
        if (z3.is_app(diff_expr) and diff_expr.decl().kind() == z3.Z3_OP_SUB and
                diff_expr.num_args() == 2):
            a, b = diff_expr.arg(0), diff_expr.arg(1)
            return a, b, n

    return None, None, None


class ModularAdditionRule(RuleVariableMixin):
    """模加法规则：a≡b (mod n), c≡d (mod n) ⊢ a+c≡b+d (mod n)"""

    def __init__(self):
        self.name = "ModularAddition"
        self.description = "模加法：a≡b (mod n), c≡d (mod n) ⊢ a+c≡b+d (mod n)"

    def num_premises(self):
        return 2

    def can_apply(self, premises):
        if len(premises) != 2:
            return False
        # 两个前提必须都是同余等式并且模数相同
        a1, b1, n1 = _extract_congruence(premises[0])
        a2, b2, n2 = _extract_congruence(premises[1])

        if not all(v is not None for v in (a1, b1, n1, a2, b2, n2)):
            return False

        # 检查模数是否相同
        return z3.simplify(n1 == n2).is_true()

    def construct_conclusion(self, premises):
        if len(premises) != 2:
            raise ValueError("ModularAddition需要恰好2个前提")

        a1, b1, n1 = _extract_congruence(premises[0])
        a2, b2, n2 = _extract_congruence(premises[1])

        if (all(v is not None for v in (a1, b1, n1, a2, b2, n2)) and
                z3.simplify(n1 == n2).is_true()):
            # 构造 (a1 + a2) % n == (b1 + b2) % n
            return (a1 + a2) % n1 == (b1 + b2) % n1

        # 如果解析失败，则生成随机同余表达式
        a = z3.Int(f"a{random.randint(1, 1000000)}")
        b = z3.Int(f"b{random.randint(1, 1000000)}")
        c = z3.Int(f"c{random.randint(1, 1000000)}")
        d = z3.Int(f"d{random.randint(1, 1000000)}")
        n = z3.Int(f"n{random.randint(1, 1000000)}")
        return (a + c) % n == (b + d) % n

    def generate_premises(self, conclusion_expr=None, max_premises=2):
        # 从 (a+c) % n == (b+d) % n 反向生成 a % n == b % n, c % n == d % n
        if conclusion_expr and z3.is_eq(conclusion_expr):
            lhs, rhs = conclusion_expr.arg(0), conclusion_expr.arg(1)
            if (z3.is_app(lhs) and lhs.decl().kind() == z3.Z3_OP_MOD and
                    z3.is_app(rhs) and rhs.decl().kind() == z3.Z3_OP_MOD):
                # lhs: (e1) % n, rhs: (e2) % n
                e1, n = lhs.arg(0), lhs.arg(1)
                e2, _n = rhs.arg(0), rhs.arg(1)
                if (z3.is_add(e1) and z3.is_add(e2) and
                        e1.num_args() == 2 and e2.num_args() == 2):
                    a, c = e1.arg(0), e1.arg(1)
                    b, d = e2.arg(0), e2.arg(1)
                    p1 = a % n == b % n
                    p2 = c % n == d % n
                    return [p1, p2]

        # 默认生成随机前提
        a = z3.Int(f"a{random.randint(1, 1000000)}")
        b = z3.Int(f"b{random.randint(1, 1000000)}")
        c = z3.Int(f"c{random.randint(1, 1000000)}")
        d = z3.Int(f"d{random.randint(1, 1000000)}")
        n = z3.Int(f"n{random.randint(1, 1000000)}")
        p1 = a % n == b % n
        p2 = c % n == d % n
        return [p1, p2]

    def explain_step(self, premises, conclusion):
        return f"ModularAddition: 由 {premises[0]} 和 {premises[1]} 得出 {conclusion}"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "如果 a≡b(mod n) 且 c≡d(mod n)，则 a+c≡b+d(mod n)",
            "formal": "a≡b (mod n), c≡d (mod n) ⊢ a+c≡b+d (mod n)",
            "example": "若 2≡5 (mod 3) 且 7≡1 (mod 3)，则 2+7≡5+1 (mod 3)",
            "variables": ["a", "b", "c", "d", "n"]
        }


class ModularTransitivityRule(RuleVariableMixin):
    """模传递性规则：a≡b(mod n), b≡c(mod n) ⊢ a≡c(mod n)"""

    def __init__(self):
        self.name = "ModularTransitivity"
        self.description = "模传递性：a≡b(mod n), b≡c(mod n) ⊢ a≡c(mod n)"

    def num_premises(self):
        return 2

    def can_apply(self, premises):
        if len(premises) != 2:
            return False

        a1, b1, n1 = _extract_congruence(premises[0])
        a2, b2, n2 = _extract_congruence(premises[1])

        if not all(v is not None for v in (a1, b1, n1, a2, b2, n2)):
            return False

        # 检查传递性条件：b1 == a2 且模数相同
        return (z3.simplify(b1 == a2).is_true() and
                z3.simplify(n1 == n2).is_true())

    def construct_conclusion(self, premises):
        if len(premises) != 2:
            raise ValueError("ModularTransitivity需要恰好2个前提")

        a1, b1, n1 = _extract_congruence(premises[0])
        a2, b2, n2 = _extract_congruence(premises[1])

        if (all(v is not None for v in (a1, b1, n1, a2, b2, n2)) and
                z3.simplify(b1 == a2).is_true() and z3.simplify(n1 == n2).is_true()):
            # 构造 a1 % n == b2 % n
            return a1 % n1 == b2 % n1

        # 默认生成随机
        a = z3.Int(f"a{random.randint(1, 1000000)}")
        c = z3.Int(f"c{random.randint(1, 1000000)}")
        n = z3.Int(f"n{random.randint(1, 1000000)}")
        return a % n == c % n

    def generate_premises(self, conclusion_expr=None, max_premises=2):
        # 从 (a+c) % n == (b+d) % n 反向生成 a % n == b % n, c % n == d % n
        if conclusion_expr and z3.is_eq(conclusion_expr):
            lhs, rhs = conclusion_expr.arg(0), conclusion_expr.arg(1)
            if (z3.is_app(lhs) and lhs.decl().kind() == z3.Z3_OP_MOD and
                    z3.is_app(rhs) and rhs.decl().kind() == z3.Z3_OP_MOD):
                # lhs: (e1) % n, rhs: (e2) % n
                e1, n = lhs.arg(0), lhs.arg(1)
                e2, _n = rhs.arg(0), rhs.arg(1)
                if (z3.is_add(e1) and z3.is_add(e2) and
                        e1.num_args() == 2 and e2.num_args() == 2):
                    a, c = e1.arg(0), e1.arg(1)
                    b, d = e2.arg(0), e2.arg(1)
                    p1 = a % n == b % n
                    p2 = c % n == d % n
                    return [p1, p2]

        # 默认生成随机前提
        a = z3.Int(f"a{random.randint(1, 1000000)}")
        b = z3.Int(f"b{random.randint(1, 1000000)}")
        c = z3.Int(f"c{random.randint(1, 1000000)}")
        d = z3.Int(f"d{random.randint(1, 1000000)}")
        n = z3.Int(f"n{random.randint(1, 1000000)}")
        p1 = a % n == b % n
        p2 = c % n == d % n
        return [p1, p2]

    def explain_step(self, premises, conclusion):
        return f"ModularTransitivity: 由 {premises[0]} 和 {premises[1]} 得出 {conclusion}"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "如果 a≡b(mod n) 且 b≡c(mod n)，则 a≡c(mod n)",
            "formal": "a≡b (mod n), b≡c (mod n) ⊢ a≡c (mod n)",
            "example": "若 2≡5 (mod 3) 且 5≡8 (mod 3)，则 2≡8 (mod 3)",
            "variables": ["a", "b", "c", "n"]
        }