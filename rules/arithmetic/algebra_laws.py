# 代数基础律

"""
* AddComm（加法交换）：`a + b = b + a`
* MulComm（乘法交换）：`a × b = b × a`
* AddAssoc（加法结合）：`(a + b) + c = a + (b + c)`
* MulAssoc（乘法结合）：`(a × b) × c = a × (b × c)`
* LeftDistributive（左分配）：`a × (b + c) = a×b + a×c`
* RightDistributive（右分配）：`(a + b) × c = a×c + b×c`
* AddIdentity（加法恒等元）：`a + 0 = a`
* MulIdentity（乘法恒等元）：`a × 1 = a`
* MulZero（乘零归零）：`a × 0 = 0`
* AddInverse（加法逆元）：`a + (−a) = 0`
"""

import z3
import random
from utils.variable_manager import RuleVariableMixin


class AddCommRule(RuleVariableMixin):
    """加法交换律：a + b = b + a"""
    def __init__(self):
        self.name = "AddComm"
        self.description = "加法交换：a + b = b + a"

    def num_premises(self):
        return 0

    def can_apply(self, premises):
        return len(premises) == 0

    def construct_conclusion(self, premises):
        a = z3.Int(f"a{random.randint(1, 1_000_000)}")
        b = z3.Int(f"b{random.randint(1, 1_000_000)}")
        return a + b == b + a

    def generate_premises(self, conclusion_expr, max_premises=0):
        return []

    def explain_step(self, premises, conclusion):
        return f"AddComm: 由于加法交换律，得出 {conclusion}"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "a + b = b + a",
            "formal": "a+b = b+a",
            "example": "2+3=3+2",
            "variables": ["a", "b"]
        }


class MulCommRule(RuleVariableMixin):
    """乘法交换律：a × b = b × a"""
    def __init__(self):
        self.name = "MulComm"
        self.description = "乘法交换：a×b = b×a"

    def num_premises(self):
        return 0

    def can_apply(self, premises):
        return len(premises) == 0

    def construct_conclusion(self, premises):
        a = z3.Int(f"a{random.randint(1, 1_000_000)}")
        b = z3.Int(f"b{random.randint(1, 1_000_000)}")
        return a * b == b * a

    def generate_premises(self, conclusion_expr, max_premises=0):
        return []

    def explain_step(self, premises, conclusion):
        return f"MulComm: 由于乘法交换律，得出 {conclusion}"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "a×b = b×a",
            "formal": "a×b = b×a",
            "example": "2×3 = 3×2",
            "variables": ["a", "b"]
        }


class AddAssocRule(RuleVariableMixin):
    """加法结合律：(a + b) + c = a + (b + c)"""
    def __init__(self):
        self.name = "AddAssoc"
        self.description = "加法结合：(a+b)+c = a+(b+c)"

    def num_premises(self):
        return 0

    def can_apply(self, premises):
        return len(premises) == 0

    def construct_conclusion(self, premises):
        a = z3.Int(f"a{random.randint(1, 1_000_000)}")
        b = z3.Int(f"b{random.randint(1, 1_000_000)}")
        c = z3.Int(f"c{random.randint(1, 1_000_000)}")
        return (a + b) + c == a + (b + c)

    def generate_premises(self, conclusion_expr, max_premises=0):
        return []

    def explain_step(self, premises, conclusion):
        return f"AddAssoc: 由加法结合律得出 {conclusion}"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "(a+b)+c = a+(b+c)",
            "formal": "(a+b)+c = a+(b+c)",
            "example": "(1+2)+3 = 1+(2+3)",
            "variables": ["a", "b", "c"]
        }


class MulAssocRule(RuleVariableMixin):
    """乘法结合律：(a×b)×c = a×(b×c)"""
    def __init__(self):
        self.name = "MulAssoc"
        self.description = "乘法结合：(a×b)×c = a×(b×c)"

    def num_premises(self):
        return 0

    def can_apply(self, premises):
        return len(premises) == 0

    def construct_conclusion(self, premises):
        a = z3.Int(f"a{random.randint(1, 1_000_000)}")
        b = z3.Int(f"b{random.randint(1, 1_000_000)}")
        c = z3.Int(f"c{random.randint(1, 1_000_000)}")
        return (a * b) * c == a * (b * c)

    def generate_premises(self, conclusion_expr, max_premises=0):
        return []

    def explain_step(self, premises, conclusion):
        return f"MulAssoc: 由乘法结合律得出 {conclusion}"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "(a×b)×c = a×(b×c)",
            "formal": "(a×b)×c = a×(b×c)",
            "example": "(2×3)×4 = 2×(3×4)",
            "variables": ["a", "b", "c"]
        }


class LeftDistributiveRule(RuleVariableMixin):
    """左分配律：a×(b+c) = a×b + a×c"""
    def __init__(self):
        self.name = "LeftDistributive"
        self.description = "左分配：a×(b+c) = a×b + a×c"

    def num_premises(self):
        return 0

    def can_apply(self, premises):
        return len(premises) == 0

    def construct_conclusion(self, premises):
        a = z3.Int(f"a{random.randint(1, 1_000_000)}")
        b = z3.Int(f"b{random.randint(1, 1_000_000)}")
        c = z3.Int(f"c{random.randint(1, 1_000_000)}")
        return a * (b + c) == a * b + a * c

    def generate_premises(self, conclusion_expr, max_premises=0):
        return []

    def explain_step(self, premises, conclusion):
        return f"LeftDistributive: 根据分配律可知 {conclusion}"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "a×(b+c) = a×b + a×c",
            "formal": "a×(b+c) = a×b + a×c",
            "example": "2×(3+4) = 2×3 + 2×4",
            "variables": ["a", "b", "c"]
        }


class RightDistributiveRule(RuleVariableMixin):
    """右分配律：(a+b)×c = a×c + b×c"""
    def __init__(self):
        self.name = "RightDistributive"
        self.description = "右分配：(a+b)×c = a×c + b×c"

    def num_premises(self):
        return 0

    def can_apply(self, premises):
        return len(premises) == 0

    def construct_conclusion(self, premises):
        a = z3.Int(f"a{random.randint(1, 1_000_000)}")
        b = z3.Int(f"b{random.randint(1, 1_000_000)}")
        c = z3.Int(f"c{random.randint(1, 1_000_000)}")
        return (a + b) * c == a * c + b * c

    def generate_premises(self, conclusion_expr, max_premises=0):
        return []

    def explain_step(self, premises, conclusion):
        return f"RightDistributive: 根据分配律可知 {conclusion}"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "(a+b)×c = a×c + b×c",
            "formal": "(a+b)×c = a×c + b×c",
            "example": "(3+4)×2 = 3×2 + 4×2",
            "variables": ["a", "b", "c"]
        }


class AddIdentityRule(RuleVariableMixin):
    """加法恒等元：a + 0 = a"""
    def __init__(self):
        self.name = "AddIdentity"
        self.description = "加法恒等元：a+0=a"

    def num_premises(self):
        return 0

    def can_apply(self, premises):
        return len(premises) == 0

    def construct_conclusion(self, premises):
        a = z3.Int(f"a{random.randint(1, 1_000_000)}")
        return a + 0 == a

    def generate_premises(self, conclusion_expr, max_premises=0):
        return []

    def explain_step(self, premises, conclusion):
        return f"AddIdentity: 根据加法恒等律可知 {conclusion}"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "a+0 = a",
            "formal": "a+0 = a",
            "example": "5+0 = 5",
            "variables": ["a"]
        }


class MulIdentityRule(RuleVariableMixin):
    """乘法恒等元：a×1 = a"""
    def __init__(self):
        self.name = "MulIdentity"
        self.description = "乘法恒等元：a×1 = a"

    def num_premises(self):
        return 0

    def can_apply(self, premises):
        return len(premises) == 0

    def construct_conclusion(self, premises):
        a = z3.Int(f"a{random.randint(1, 1_000_000)}")
        return a * 1 == a

    def generate_premises(self, conclusion_expr, max_premises=0):
        return []

    def explain_step(self, premises, conclusion):
        return f"MulIdentity: 根据乘法恒等律可知 {conclusion}"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "a×1 = a",
            "formal": "a×1 = a",
            "example": "7×1 = 7",
            "variables": ["a"]
        }


class MulZeroRule(RuleVariableMixin):
    """乘零归零：a×0 = 0"""
    def __init__(self):
        self.name = "MulZero"
        self.description = "乘零归零：a×0 = 0"

    def num_premises(self):
        return 0

    def can_apply(self, premises):
        return len(premises) == 0

    def construct_conclusion(self, premises):
        a = z3.Int(f"a{random.randint(1, 1_000_000)}")
        return a * 0 == 0

    def generate_premises(self, conclusion_expr, max_premises=0):
        return []

    def explain_step(self, premises, conclusion):
        return f"MulZero: 根据乘零归零可知 {conclusion}"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "a×0 = 0",
            "formal": "a×0 = 0",
            "example": "4×0 = 0",
            "variables": ["a"]
        }


class AddInverseRule(RuleVariableMixin):
    """加法逆元：a + (−a) = 0"""
    def __init__(self):
        self.name = "AddInverse"
        self.description = "加法逆元：a+(−a)=0"

    def num_premises(self):
        return 0

    def can_apply(self, premises):
        return len(premises) == 0

    def construct_conclusion(self, premises):
        a = z3.Int(f"a{random.randint(1, 1_000_000)}")
        return a + (-a) == 0

    def generate_premises(self, conclusion_expr, max_premises=0):
        return []

    def explain_step(self, premises, conclusion):
        return f"AddInverse: 根据加法逆元可知 {conclusion}"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "a+(-a) = 0",
            "formal": "a+(-a) = 0",
            "example": "5+(-5) = 0",
            "variables": ["a"]
        }