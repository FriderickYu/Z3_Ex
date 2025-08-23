# 量词变换与分配

"""
* QuantifierNegation-1：`¬∀x P(x) ⟷ ∃x ¬P(x)`
* QuantifierNegation-2：`¬∃x P(x) ⟷ ∀x ¬P(x)`
* UniversalDistribution（对∧的分配）：`∀x (P(x)∧Q(x)) ⟷ (∀x P(x)) ∧ (∀x Q(x))`
* ExistentialDistribution（对∨的分配）：`∃x (P(x)∨Q(x)) ⟷ (∃x P(x)) ∨ (∃x Q(x))`
* ExistentialConjunction（一向）：`∃x (P(x)∧Q(x)) ⊢ (∃x P(x)) ∧ (∃x Q(x))`
"""

import z3
import random
from utils.variable_manager import RuleVariableMixin


def _fresh_var(name_prefix="v"):
    """生成一个新的整数变量。"""
    return z3.Int(f"{name_prefix}{random.randint(1, 1_000_000)}")


def _fresh_pred(name_prefix="P"):
    """生成一个新的一元谓词函数。"""
    return z3.Function(f"{name_prefix}{random.randint(1, 1_000_000)}", z3.IntSort(), z3.BoolSort())


class QuantifierNegation1Rule(RuleVariableMixin):
    """量词否定变换规则 1： ¬∀x P(x) ⟷ ∃x ¬P(x)。"""

    def __init__(self):
        self.name = "QuantifierNegation-1"
        self.description = "量词否定 1：¬∀x P(x) ⟷ ∃x ¬P(x)"

    def num_premises(self):
        return 1

    def can_apply(self, premises):
        return len(premises) == 1

    def construct_conclusion(self, premises):
        if len(premises) != 1:
            raise ValueError("QuantifierNegation-1需要恰好1个前提")
        expr = premises[0]
        # 情形1：前提为 ¬∀x P(x)，则结论为 ∃x ¬P(x)
        if z3.is_not(expr):
            inner = expr.arg(0)
            if z3.is_quantifier(inner) and inner.is_forall():
                try:
                    num_vars = inner.num_vars()
                    sorts = [inner.var_sort(i) for i in range(num_vars)]
                    # 获取体部
                    body = inner.body()
                    # 构造相同数量的受限变量
                    vars_list = [z3.Const(f"x{i}{random.randint(1, 1_000_000)}", s) for i, s in enumerate(sorts)]
                    # 构造 ∃x ¬P(x)
                    return z3.Exists(vars_list, z3.Not(body))
                except Exception:
                    pass
        # 情形2：前提为 ∃x ¬P(x)，则结论为 ¬∀x P(x)
        if z3.is_quantifier(expr) and expr.is_exists():
            try:
                num_vars = expr.num_vars()
                body = expr.body()
                if z3.is_not(body):
                    inner_body = body.arg(0)
                    sorts = [expr.var_sort(i) for i in range(num_vars)]
                    vars_list = [z3.Const(f"x{i}{random.randint(1, 1_000_000)}", s) for i, s in enumerate(sorts)]
                    return z3.Not(z3.ForAll(vars_list, inner_body))
            except Exception:
                pass
        # 默认：返回新的结论变量
        return self.create_conclusion_variable()

    def generate_premises(self, conclusion_expr, max_premises=1):
        """根据结论生成前提。"""
        # 如果结论是 ∃x ¬P(x)，则前提应为 ¬∀x P(x)
        if z3.is_quantifier(conclusion_expr) and conclusion_expr.is_exists():
            try:
                num_vars = conclusion_expr.num_vars()
                sorts = [conclusion_expr.var_sort(i) for i in range(num_vars)]
                body = conclusion_expr.body()
                if z3.is_not(body):
                    inner_body = body.arg(0)
                    vars_list = [z3.Const(f"x{i}{random.randint(1, 1_000_000)}", s) for i, s in enumerate(sorts)]
                    return [z3.Not(z3.ForAll(vars_list, inner_body))]
            except Exception:
                pass
        # 如果结论是 ¬∀x P(x)，则前提应为 ∃x ¬P(x)
        if z3.is_not(conclusion_expr):
            inner = conclusion_expr.arg(0)
            if z3.is_quantifier(inner) and inner.is_forall():
                try:
                    num_vars = inner.num_vars()
                    sorts = [inner.var_sort(i) for i in range(num_vars)]
                    body = inner.body()
                    vars_list = [z3.Const(f"x{i}{random.randint(1, 1_000_000)}", s) for i, s in enumerate(sorts)]
                    return [z3.Exists(vars_list, z3.Not(body))]
                except Exception:
                    pass
        # 默认随机：随机选择一个方向
        num_vars = 1
        sort = z3.IntSort()
        var = z3.Const(f"x{random.randint(1, 1_000_000)}", sort)
        pred = _fresh_pred("P")
        if random.choice([True, False]):
            return [z3.Not(z3.ForAll([var], pred(var)))]
        else:
            return [z3.Exists([var], z3.Not(pred(var)))]

    def explain_step(self, premises, conclusion):
        premise = premises[0]
        return f"QuantifierNegation-1: 根据 ¬∀x P(x) ↔ ∃x ¬P(x) 将 {premise} 转换为 {conclusion}"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "¬∀x P(x) 等价于 ∃x ¬P(x)",
            "formal": "¬∀x P(x) ↔ ∃x ¬P(x)",
            "example": "不存在所有人都高兴 等价于 存在某人不高兴",
            "variables": ["量化变量x", "谓词P", "前提", "结论"],
        }


class QuantifierNegation2Rule(RuleVariableMixin):
    """量词否定变换规则 2： ¬∃x P(x) ⟷ ∀x ¬P(x)。"""

    def __init__(self):
        self.name = "QuantifierNegation-2"
        self.description = "量词否定 2：¬∃x P(x) ⟷ ∀x ¬P(x)"

    def num_premises(self):
        return 1

    def can_apply(self, premises):
        return len(premises) == 1

    def construct_conclusion(self, premises):
        if len(premises) != 1:
            raise ValueError("QuantifierNegation-2需要恰好1个前提")
        expr = premises[0]
        # 情形1：前提为 ¬∃x P(x)，则结论为 ∀x ¬P(x)
        if z3.is_not(expr):
            inner = expr.arg(0)
            if z3.is_quantifier(inner) and inner.is_exists():
                try:
                    num_vars = inner.num_vars()
                    sorts = [inner.var_sort(i) for i in range(num_vars)]
                    body = inner.body()
                    vars_list = [z3.Const(f"x{i}{random.randint(1, 1_000_000)}", s) for i, s in enumerate(sorts)]
                    return z3.ForAll(vars_list, z3.Not(body))
                except Exception:
                    pass
        # 情形2：前提为 ∀x ¬P(x)，则结论为 ¬∃x P(x)
        if z3.is_quantifier(expr) and expr.is_forall():
            try:
                num_vars = expr.num_vars()
                sorts = [expr.var_sort(i) for i in range(num_vars)]
                body = expr.body()
                if z3.is_not(body):
                    inner_body = body.arg(0)
                    vars_list = [z3.Const(f"x{i}{random.randint(1, 1_000_000)}", s) for i, s in enumerate(sorts)]
                    return z3.Not(z3.Exists(vars_list, inner_body))
            except Exception:
                pass
        # 默认返回新变量
        return self.create_conclusion_variable()

    def generate_premises(self, conclusion_expr, max_premises=1):
        # 如果结论是 ∀x ¬P(x)，生成前提 ¬∃x P(x)
        if z3.is_quantifier(conclusion_expr) and conclusion_expr.is_forall():
            try:
                num_vars = conclusion_expr.num_vars()
                sorts = [conclusion_expr.var_sort(i) for i in range(num_vars)]
                body = conclusion_expr.body()
                if z3.is_not(body):
                    inner_body = body.arg(0)
                    vars_list = [z3.Const(f"x{i}{random.randint(1, 1_000_000)}", s) for i, s in enumerate(sorts)]
                    return [z3.Not(z3.Exists(vars_list, inner_body))]
            except Exception:
                pass
        # 如果结论是 ¬∃x P(x)，生成前提 ∀x ¬P(x)
        if z3.is_not(conclusion_expr):
            inner = conclusion_expr.arg(0)
            if z3.is_quantifier(inner) and inner.is_exists():
                try:
                    num_vars = inner.num_vars()
                    sorts = [inner.var_sort(i) for i in range(num_vars)]
                    body = inner.body()
                    vars_list = [z3.Const(f"x{i}{random.randint(1, 1_000_000)}", s) for i, s in enumerate(sorts)]
                    return [z3.ForAll(vars_list, z3.Not(body))]
                except Exception:
                    pass
        # 默认随机生成
        sort = z3.IntSort()
        var = z3.Const(f"x{random.randint(1, 1_000_000)}", sort)
        pred = _fresh_pred("P")
        if random.choice([True, False]):
            return [z3.Not(z3.Exists([var], pred(var)))]
        else:
            return [z3.ForAll([var], z3.Not(pred(var)))]

    def explain_step(self, premises, conclusion):
        premise = premises[0]
        return f"QuantifierNegation-2: 根据 ¬∃x P(x) ↔ ∀x ¬P(x) 将 {premise} 转换为 {conclusion}"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "¬∃x P(x) 等价于 ∀x ¬P(x)",
            "formal": "¬∃x P(x) ↔ ∀x ¬P(x)",
            "example": "不存在某人喜欢音乐 等价于 所有人都不喜欢音乐",
            "variables": ["量化变量x", "谓词P", "前提", "结论"],
        }


class QuantifierNegation2AltRule(RuleVariableMixin):
    """量词否定变换规则 2 的另一形式：∀x ¬P(x) ⟷ ¬∃x P(x)。

    本规则与 QuantifierNegation2Rule 形式上相同，但强调从 "∀x ¬P(x)" 推出 "¬∃x P(x)" 及其逆，
    作为独立的等价规则存在。
    """

    def __init__(self):
        self.name = "QuantifierNegation-2-alt"
        self.description = "量词否定 2 alt：∀x ¬P(x) ⟷ ¬∃x P(x)"

    def num_premises(self):
        return 1

    def can_apply(self, premises):
        return len(premises) == 1

    def construct_conclusion(self, premises):
        if len(premises) != 1:
            raise ValueError("QuantifierNegation-2-alt需要恰好1个前提")
        expr = premises[0]
        # 前提为 ∀x ¬P(x) → 结论为 ¬∃x P(x)
        if z3.is_quantifier(expr) and expr.is_forall():
            try:
                num_vars = expr.num_vars()
                sorts = [expr.var_sort(i) for i in range(num_vars)]
                body = expr.body()
                if z3.is_not(body):
                    inner_body = body.arg(0)
                    vars_list = [z3.Const(f"x{i}{random.randint(1, 1_000_000)}", s) for i, s in enumerate(sorts)]
                    return z3.Not(z3.Exists(vars_list, inner_body))
            except Exception:
                pass
        # 前提为 ¬∃x P(x) → 结论为 ∀x ¬P(x)
        if z3.is_not(expr):
            inner = expr.arg(0)
            if z3.is_quantifier(inner) and inner.is_exists():
                try:
                    num_vars = inner.num_vars()
                    sorts = [inner.var_sort(i) for i in range(num_vars)]
                    body = inner.body()
                    vars_list = [z3.Const(f"x{i}{random.randint(1, 1_000_000)}", s) for i, s in enumerate(sorts)]
                    return z3.ForAll(vars_list, z3.Not(body))
                except Exception:
                    pass
        return self.create_conclusion_variable()

    def generate_premises(self, conclusion_expr, max_premises=1):
        # 如果结论是 ¬∃x P(x)，前提应为 ∀x ¬P(x)
        if z3.is_not(conclusion_expr):
            inner = conclusion_expr.arg(0)
            if z3.is_quantifier(inner) and inner.is_exists():
                try:
                    num_vars = inner.num_vars()
                    sorts = [inner.var_sort(i) for i in range(num_vars)]
                    body = inner.body()
                    vars_list = [z3.Const(f"x{i}{random.randint(1, 1_000_000)}", s) for i, s in enumerate(sorts)]
                    return [z3.ForAll(vars_list, z3.Not(body))]
                except Exception:
                    pass
        # 如果结论是 ∀x ¬P(x)，前提应为 ¬∃x P(x)
        if z3.is_quantifier(conclusion_expr) and conclusion_expr.is_forall():
            try:
                num_vars = conclusion_expr.num_vars()
                sorts = [conclusion_expr.var_sort(i) for i in range(num_vars)]
                body = conclusion_expr.body()
                if z3.is_not(body):
                    inner_body = body.arg(0)
                    vars_list = [z3.Const(f"x{i}{random.randint(1, 1_000_000)}", s) for i, s in enumerate(sorts)]
                    return [z3.Not(z3.Exists(vars_list, inner_body))]
            except Exception:
                pass
        # 默认：随机选择一种形式
        sort = z3.IntSort()
        var = z3.Const(f"x{random.randint(1, 1_000_000)}", sort)
        pred = _fresh_pred("P")
        if random.choice([True, False]):
            return [z3.ForAll([var], z3.Not(pred(var)))]
        else:
            return [z3.Not(z3.Exists([var], pred(var)))]

    def explain_step(self, premises, conclusion):
        premise = premises[0]
        return f"QuantifierNegation-2-alt: 根据 ∀x ¬P(x) ↔ ¬∃x P(x) 将 {premise} 转换为 {conclusion}"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "∀x ¬P(x) 等价于 ¬∃x P(x)",
            "formal": "∀x ¬P(x) ↔ ¬∃x P(x)",
            "example": "所有人都不喜欢音乐 等价于 不存在某人喜欢音乐",
            "variables": ["量化变量x", "谓词P", "前提", "结论"],
        }


class ExistentialDistributionRule(RuleVariableMixin):
    """存在量词与析取分配规则：∃x (P(x)∨Q(x)) ⟷ (∃x P(x)) ∨ (∃x Q(x))。"""

    def __init__(self):
        self.name = "ExistentialDistribution"
        self.description = "存在量词对析取的分配：∃x (P∨Q) ⟷ ∃x P ∨ ∃x Q"

    def num_premises(self):
        return 1

    def can_apply(self, premises):
        return len(premises) == 1

    def construct_conclusion(self, premises):
        if len(premises) != 1:
            raise ValueError("ExistentialDistribution需要恰好1个前提")
        expr = premises[0]
        # 情形1：前提为 ∃x (P ∨ Q)，则结论为 (∃x P) ∨ (∃x Q)
        if z3.is_quantifier(expr) and expr.is_exists():
            try:
                # 只处理单变量且体部是析取的情况
                if expr.num_vars() == 1:
                    var_sort = expr.var_sort(0)
                    body = expr.body()
                    if z3.is_or(body) and len(body.children()) == 2:
                        left, right = body.children()
                        # 创建新的量化变量
                        var = z3.Const(f"x{random.randint(1, 1_000_000)}", var_sort)
                        # 将绑定变量替换为常量以实例化各部分
                        inst_left = z3.substitute_vars(left, var)
                        inst_right = z3.substitute_vars(right, var)
                        return z3.Or(z3.Exists([var], inst_left), z3.Exists([var], inst_right))
            except Exception:
                pass
        # 情形2：前提为 (∃x P) ∨ (∃x Q)，则结论为 ∃x (P ∨ Q)
        if z3.is_or(expr) and len(expr.children()) == 2:
            a, b = expr.children()
            if z3.is_quantifier(a) and a.is_exists() and z3.is_quantifier(b) and b.is_exists():
                try:
                    # 假设两个量化变量有相同的数量和类型
                    if a.num_vars() == 1 and b.num_vars() == 1:
                        sort_a = a.var_sort(0)
                        sort_b = b.var_sort(0)
                        if sort_a == sort_b:
                            var_sort = sort_a
                            var = z3.Const(f"x{random.randint(1, 1_000_000)}", var_sort)
                            # 获取各自体部并实例化
                            body_a = z3.substitute_vars(a.body(), var)
                            body_b = z3.substitute_vars(b.body(), var)
                            return z3.Exists([var], z3.Or(body_a, body_b))
                except Exception:
                    pass
        return self.create_conclusion_variable()

    def generate_premises(self, conclusion_expr, max_premises=1):
        # 如果结论是 ∃x (P∨Q)，生成前提 (∃x P) ∨ (∃x Q)
        if z3.is_quantifier(conclusion_expr) and conclusion_expr.is_exists():
            try:
                if conclusion_expr.num_vars() == 1:
                    var_sort = conclusion_expr.var_sort(0)
                    body = conclusion_expr.body()
                    if z3.is_or(body) and len(body.children()) == 2:
                        left, right = body.children()
                        var = z3.Const(f"x{random.randint(1, 1_000_000)}", var_sort)
                        inst_left = z3.substitute_vars(left, var)
                        inst_right = z3.substitute_vars(right, var)
                        return [z3.Or(z3.Exists([var], inst_left), z3.Exists([var], inst_right))]
            except Exception:
                pass
        # 如果结论是 (∃x P) ∨ (∃x Q)，生成前提 ∃x (P∨Q)
        if z3.is_or(conclusion_expr) and len(conclusion_expr.children()) == 2:
            a, b = conclusion_expr.children()
            if z3.is_quantifier(a) and a.is_exists() and z3.is_quantifier(b) and b.is_exists():
                try:
                    if a.num_vars() == 1 and b.num_vars() == 1:
                        sort_a = a.var_sort(0)
                        sort_b = b.var_sort(0)
                        if sort_a == sort_b:
                            var_sort = sort_a
                            var = z3.Const(f"x{random.randint(1, 1_000_000)}", var_sort)
                            body_a = z3.substitute_vars(a.body(), var)
                            body_b = z3.substitute_vars(b.body(), var)
                            return [z3.Exists([var], z3.Or(body_a, body_b))]
                except Exception:
                    pass
        # 默认随机
        sort = z3.IntSort()
        var = z3.Const(f"x{random.randint(1, 1_000_000)}", sort)
        pred1 = _fresh_pred("P")
        pred2 = _fresh_pred("Q")
        if random.choice([True, False]):
            return [z3.Exists([var], z3.Or(pred1(var), pred2(var)))]
        else:
            return [z3.Or(z3.Exists([var], pred1(var)), z3.Exists([var], pred2(var)))]

    def explain_step(self, premises, conclusion):
        premise = premises[0]
        return f"ExistentialDistribution: 根据分配律，将 {premise} 转换为 {conclusion}"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "∃x (P∨Q) 等价于 (∃x P)∨(∃x Q)",
            "formal": "∃x (P∨Q) ↔ (∃x P)∨(∃x Q)",
            "example": "存在一个人是学生或老师 等价于 存在一个人是学生 或 存在一个人是老师",
            "variables": ["量化变量x", "谓词P", "谓词Q", "前提", "结论"],
        }


class ExistentialConjunctionRule(RuleVariableMixin):
    """存在量词与合取分配规则：∃x (P(x)∧Q(x)) ⊢ (∃x P(x)) ∧ (∃x Q(x))。"""

    def __init__(self):
        self.name = "ExistentialConjunction"
        self.description = "存在量词对合取的一向分配：∃x (P∧Q) ⊢ ∃x P ∧ ∃x Q"

    def num_premises(self):
        return 1

    def can_apply(self, premises):
        return len(premises) == 1

    def construct_conclusion(self, premises):
        if len(premises) != 1:
            raise ValueError("ExistentialConjunction需要恰好1个前提")
        expr = premises[0]
        # 如果前提是 ∃x (P∧Q)，则结论为 (∃x P) ∧ (∃x Q)
        if z3.is_quantifier(expr) and expr.is_exists():
            try:
                if expr.num_vars() == 1:
                    var_sort = expr.var_sort(0)
                    body = expr.body()
                    if z3.is_and(body) and len(body.children()) == 2:
                        left, right = body.children()
                        var = z3.Const(f"x{random.randint(1, 1_000_000)}", var_sort)
                        inst_left = z3.substitute_vars(left, var)
                        inst_right = z3.substitute_vars(right, var)
                        return z3.And(z3.Exists([var], inst_left), z3.Exists([var], inst_right))
            except Exception:
                pass
        return self.create_conclusion_variable()

    def generate_premises(self, conclusion_expr, max_premises=1):
        # 如果结论是 (∃x P) ∧ (∃x Q)，生成前提 ∃x (P∧Q)
        if z3.is_and(conclusion_expr) and len(conclusion_expr.children()) == 2:
            a, b = conclusion_expr.children()
            if z3.is_quantifier(a) and a.is_exists() and z3.is_quantifier(b) and b.is_exists():
                try:
                    if a.num_vars() == 1 and b.num_vars() == 1:
                        sort_a = a.var_sort(0)
                        sort_b = b.var_sort(0)
                        if sort_a == sort_b:
                            var_sort = sort_a
                            var = z3.Const(f"x{random.randint(1, 1_000_000)}", var_sort)
                            body_a = z3.substitute_vars(a.body(), var)
                            body_b = z3.substitute_vars(b.body(), var)
                            return [z3.Exists([var], z3.And(body_a, body_b))]
                except Exception:
                    pass
        # 默认随机生成
        sort = z3.IntSort()
        var = z3.Const(f"x{random.randint(1, 1_000_000)}", sort)
        pred1 = _fresh_pred("P")
        pred2 = _fresh_pred("Q")
        if random.choice([True, False]):
            return [z3.Exists([var], z3.And(pred1(var), pred2(var)))]
        else:
            return [z3.And(z3.Exists([var], pred1(var)), z3.Exists([var], pred2(var)))]

    def explain_step(self, premises, conclusion):
        premise = premises[0]
        return f"ExistentialConjunction: 根据存在量词对合取的一向分配，将 {premise} 转换为 {conclusion}"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "∃x (P∧Q) 推出 (∃x P)∧(∃x Q)",
            "formal": "∃x (P∧Q) ⊢ (∃x P)∧(∃x Q)",
            "example": "存在一个人同时是学生且老师，则存在一个人是学生且存在一个人是老师",
            "variables": ["量化变量x", "谓词P", "谓词Q", "前提", "结论"],
        }