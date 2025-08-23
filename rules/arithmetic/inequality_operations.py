# 不等式运算规则

"""
* AddMonotone：`a < b ⊢ a + c < b + c`
* MulMonotonePos：`a < b, c > 0 ⊢ a×c < b×c`
* MulMonotoneNeg：`a < b, c < 0 ⊢ a×c > b×c`
* SumPositives：`x > 0, y > 0 ⊢ x + y > 0`
"""

import z3
import random
from utils.variable_manager import RuleVariableMixin


class AddMonotoneRule(RuleVariableMixin):
    """加法单调性：从 a < b 推出 a + c < b + c"""

    def __init__(self):
        self.name = "AddMonotone"
        self.description = "加法单调性：a<b ⊢ a+c < b+c"

    def num_premises(self):
        return 1

    def can_apply(self, premises):
        return len(premises) == 1 and z3.is_lt(premises[0])

    def construct_conclusion(self, premises):
        if len(premises) != 1:
            raise ValueError("AddMonotone需要恰好1个前提")
        lt = premises[0]
        a, b = lt.arg(0), lt.arg(1)
        # 新增变量 c 与 a、b 同类型
        sort = a.sort() if hasattr(a, 'sort') else None
        if sort == z3.IntSort():
            c = z3.Int(f"c{random.randint(1, 1_000_000)}")
        elif sort == z3.RealSort():
            c = z3.Real(f"c{random.randint(1, 1_000_000)}")
        else:
            # 默认整型
            c = z3.Int(f"c{random.randint(1, 1_000_000)}")
        return a + c < b + c

    def generate_premises(self, conclusion_expr, max_premises=1):
        # 若结论形如 a+c<b+c，则前提应为 a<b
        if z3.is_lt(conclusion_expr):
            lhs, rhs = conclusion_expr.arg(0), conclusion_expr.arg(1)
            # 试图拆分为 a+c 和 b+c
            if z3.is_add(lhs) and z3.is_add(rhs) and lhs.num_args() == rhs.num_args():
                # 取第一项作为 a，第二项作为 c
                a = lhs.arg(0)
                b = rhs.arg(0)
                return [a < b]
        # 默认生成随机 a<b
        a = z3.Int(f"a{random.randint(1, 1_000_000)}")
        b = z3.Int(f"b{random.randint(1, 1_000_000)}")
        return [a < b]

    def explain_step(self, premises, conclusion):
        return f"AddMonotone: 由 {premises[0]} 可得 {conclusion}"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "如果 a<b，则 a+c < b+c",
            "formal": "a<b ⊢ a+c < b+c",
            "example": "若 2<3，则 2+5 < 3+5",
            "variables": ["a", "b", "c"]
        }


class MulMonotonePosRule(RuleVariableMixin):
    """正数乘法单调性：从 a < b 且 c > 0 推出 a×c < b×c"""

    def __init__(self):
        self.name = "MulMonotonePos"
        self.description = "乘法单调性（正）：a<b, c>0 ⊢ a×c < b×c"

    def num_premises(self):
        return 2

    def can_apply(self, premises):
        if len(premises) != 2:
            return False
        # 一个前提应是严格小于，另一个应包含 >0
        has_lt = any(z3.is_lt(p) for p in premises)
        # 粗略检查 c>0：字符串中包含 ">0" 或 "0<"
        def _is_positive(p):
            s = str(p).replace(" ", "")
            return ">0" in s or "0<" in s
        has_pos = any(_is_positive(p) for p in premises)
        return has_lt and has_pos

    def construct_conclusion(self, premises):
        if len(premises) != 2:
            raise ValueError("MulMonotonePos需要恰好2个前提")
        lt = None
        pos = None
        for p in premises:
            if z3.is_lt(p):
                lt = p
            else:
                # 使用字符串匹配识别 c>0 或 0<c
                s = str(p).replace(" ", "")
                if ">0" in s or "0<" in s:
                    pos = p
        # 如果匹配失败则返回形式上表达式
        a = z3.Int(f"a{random.randint(1, 1_000_000)}")
        b = z3.Int(f"b{random.randint(1, 1_000_000)}")
        c = z3.Int(f"c{random.randint(1, 1_000_000)}")
        if lt is not None and pos is not None:
            a, b = lt.arg(0), lt.arg(1)
            # pos 可能是形如 c>0 或 0<c
            # 尝试从 pos 中获取变量
            try:
                left, right = pos.arg(0), pos.arg(1)
                if ">" in str(pos):
                    # 形式 a>0
                    c = left
                else:
                    # 形式 0<a
                    c = right
            except Exception:
                c = pos.arg(0)
            return a * c < b * c
        return a * c < b * c

    def generate_premises(self, conclusion_expr, max_premises=2):
        # 从 a*c<b*c 生成 a<b 和 c>0
        a = z3.Int(f"a{random.randint(1, 1_000_000)}")
        b = z3.Int(f"b{random.randint(1, 1_000_000)}")
        c = z3.Int(f"c{random.randint(1, 1_000_000)}")
        return [a < b, c > 0]

    def explain_step(self, premises, conclusion):
        return f"MulMonotonePos: 由 {premises[0]} 和 {premises[1]} 得出 {conclusion}"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "如果 a<b 且 c>0，则 a×c<b×c",
            "formal": "a<b, c>0 ⊢ a×c<b×c",
            "example": "若 2<3 且 4>0，则 2×4 < 3×4",
            "variables": ["a", "b", "c"]
        }


class MulMonotoneNegRule(RuleVariableMixin):
    """负数乘法单调性：从 a < b 且 c < 0 推出 a×c > b×c"""

    def __init__(self):
        self.name = "MulMonotoneNeg"
        self.description = "乘法单调性（负）：a<b, c<0 ⊢ a×c > b×c"

    def num_premises(self):
        return 2

    def can_apply(self, premises):
        if len(premises) != 2:
            return False
        has_lt = any(z3.is_lt(p) for p in premises)
        # c<0 检测，字符串中含有 "<0" 或 "0>"
        def _is_negative(p):
            s = str(p).replace(" ", "")
            return "<0" in s or "0>" in s
        has_neg = any(_is_negative(p) for p in premises)
        return has_lt and has_neg

    def construct_conclusion(self, premises):
        if len(premises) != 2:
            raise ValueError("MulMonotoneNeg需要恰好2个前提")
        lt = None
        neg = None
        for p in premises:
            if z3.is_lt(p) and neg is None:
                # 判断是否是 c<0
                s = str(p).replace(" ", "")
                if "<0" in s or "0>" in s:
                    neg = p
                else:
                    lt = p
            elif z3.is_lt(p):
                s = str(p).replace(" ", "")
                if "<0" in s or "0>" in s:
                    neg = p
        # 默认变量
        a = z3.Int(f"a{random.randint(1, 1_000_000)}")
        b = z3.Int(f"b{random.randint(1, 1_000_000)}")
        c = z3.Int(f"c{random.randint(1, 1_000_000)}")
        if lt is not None and neg is not None:
            a, b = lt.arg(0), lt.arg(1)
            c = neg.arg(0)
            return a * c > b * c
        return a * c > b * c

    def generate_premises(self, conclusion_expr, max_premises=2):
        a = z3.Int(f"a{random.randint(1, 1_000_000)}")
        b = z3.Int(f"b{random.randint(1, 1_000_000)}")
        c = z3.Int(f"c{random.randint(1, 1_000_000)}")
        return [a < b, c < 0]

    def explain_step(self, premises, conclusion):
        return f"MulMonotoneNeg: 由 {premises[0]} 和 {premises[1]} 得出 {conclusion}"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "如果 a<b 且 c<0，则 a×c > b×c",
            "formal": "a<b, c<0 ⊢ a×c > b×c",
            "example": "若 2<3 且 -1<0，则 2×(-1) > 3×(-1)",
            "variables": ["a", "b", "c"]
        }


class SumPositivesRule(RuleVariableMixin):
    """正数之和仍为正：若 x>0 且 y>0，则 x+y>0"""

    def __init__(self):
        self.name = "SumPositives"
        self.description = "正数之和：x>0, y>0 ⊢ x+y>0"

    def num_premises(self):
        return 2

    def can_apply(self, premises):
        if len(premises) != 2:
            return False
        # 两个前提都形如 x>0 或 0<x
        def _is_positive(p):
            s = str(p).replace(" ", "")
            return ">0" in s or "0<" in s
        return sum(1 for p in premises if _is_positive(p)) == 2

    def construct_conclusion(self, premises):
        if len(premises) != 2:
            raise ValueError("SumPositives需要恰好2个前提")
        vars_list = []
        for p in premises:
            s = str(p).replace(" ", "")
            try:
                left, right = p.arg(0), p.arg(1)
            except Exception:
                continue
            if ">0" in s:
                # 形式 x>0
                vars_list.append(left)
            elif "0<" in s:
                # 形式 0<x
                vars_list.append(right)
        # 如果提取到两个变量，则 x+y>0
        if len(vars_list) == 2:
            x, y = vars_list[0], vars_list[1]
            return x + y > 0
        # 默认生成随机变量
        x = z3.Int(f"x{random.randint(1, 1_000_000)}")
        y = z3.Int(f"y{random.randint(1, 1_000_000)}")
        return x + y > 0

    def generate_premises(self, conclusion_expr, max_premises=2):
        # 从 x+y>0 生成 x>0, y>0
        # 如果结论形如 x+y>0，则返回 x>0, y>0
        s = str(conclusion_expr).replace(" ", "")
        if ">0" in s and z3.is_add(conclusion_expr.arg(0)):
            lhs = conclusion_expr.arg(0)
            if lhs.num_args() == 2:
                x, y = lhs.arg(0), lhs.arg(1)
                return [x > 0, y > 0]
        x = z3.Int(f"x{random.randint(1, 1_000_000)}")
        y = z3.Int(f"y{random.randint(1, 1_000_000)}")
        return [x > 0, y > 0]

    def explain_step(self, premises, conclusion):
        return f"SumPositives: 由 {premises[0]} 和 {premises[1]} 可得 {conclusion}"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "如果 x>0 且 y>0，则 x+y>0",
            "formal": "x>0, y>0 ⊢ x+y>0",
            "example": "若 1>0 且 2>0，则 1+2>0",
            "variables": ["x", "y"]
        }