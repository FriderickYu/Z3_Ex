# 量词操作

"""
* ExistentialGeneralization（存在泛化）：`P(a) ⊢ ∃x P(x)`
* UniversalInstantiationRule（全称实例化）：`∀x P(x) ⊢ P(a)` **已存在**
* UniversalGeneralization（全称泛化，侧条件）：`P(x) ⊢ ∀x P(x) （侧条件：x 不自由出现在未解除前提中）`
* ExistentialElimination（存在消去，证明规则）： `∃x P(x) ⇒ 取 fresh 常量 c，在子证明中以假设 P(c) 推出目标 φ，且 c 不出现在 φ/任何开放前提 中，则得 ⊢ φ`
"""

import z3
import random
from utils.variable_manager import RuleVariableMixin


def _fresh_var(name_prefix="v"):
    """生成一个新的整数变量，确保名称不易重复。"""
    return z3.Int(f"{name_prefix}{random.randint(1, 1_000_000)}")


def _fresh_pred(name_prefix="P"):
    """生成一个新的一元谓词函数``Int -> Bool``。"""
    return z3.Function(f"{name_prefix}{random.randint(1, 1_000_000)}", z3.IntSort(), z3.BoolSort())


class ExistentialGeneralizationRule(RuleVariableMixin):
    """存在泛化规则 (Existential Generalization)

    根据前提 P(a) 推出 ∃x P(x)。在此实现中，忽略前提的具体结构，
    使用新的谓词与变量构造一个存在量化式作为结论。
    """

    def __init__(self):
        self.name = "ExistentialGeneralization"
        self.description = "存在泛化：P(a) ⊢ ∃x P(x)"

    def num_premises(self):
        return 1

    def can_apply(self, premises):
        return len(premises) == 1

    def construct_conclusion(self, premises):
        if len(premises) != 1:
            raise ValueError("ExistentialGeneralization需要恰好1个前提")
        premise = premises[0]
        # 如果前提是一个谓词应用 P(a)，则保持同一谓词并推广为 ∃x P(x)
        if z3.is_app(premise) and premise.num_args() >= 1:
            try:
                pred = premise.decl()
                # 仅处理一元谓词
                if premise.num_args() == 1:
                    arg = premise.arg(0)
                    # 使用相同的参数类型创建新的受限变量
                    var_sort = arg.sort()
                    var = z3.Const(f"x{random.randint(1, 1_000_000)}", var_sort)
                    body = pred(var)
                    return z3.Exists([var], body)
            except Exception:
                pass
        # 默认：生成新的谓词和量化变量
        var = _fresh_var("x")
        pred = _fresh_pred("P")
        body = pred(var)
        return z3.Exists([var], body)

    def generate_premises(self, conclusion_expr, max_premises=1):
        """返回一个简单的前提，忽略结论结构。"""
        # 如果结论是 ∃x P(x) 形式，则返回对应实例 P(c)
        if z3.is_quantifier(conclusion_expr) and conclusion_expr.is_exists():
            try:
                body = conclusion_expr.body()
                # 获取受限变量的类型
                if conclusion_expr.num_vars() == 1:
                    sort = conclusion_expr.var_sort(0)
                    const = z3.Const(f"c{random.randint(1, 1_000_000)}", sort)
                    # 将体部中的受限变量替换为常量
                    inst_body = z3.substitute_vars(body, const)
                    return [inst_body]
            except Exception:
                pass
        # 默认：返回一个新的前提变量
        return [self.create_premise_variable()]

    def explain_step(self, premises, conclusion):
        premise = premises[0]
        return f"ExistentialGeneralization: 由于 {premise}，可推出存在 x 使得某谓词成立，即 {conclusion}"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "如果某个特定对象满足 P，则存在某个 x 满足 P",
            "formal": "P(a) ⊢ ∃x P(x)",
            "example": "如果某人是学生，那么存在一个人是学生",
            "variables": ["特定个体a", "谓词P", "存在量词x"],
        }


class UniversalInstantiationRule(RuleVariableMixin):
    """全称实例化规则 (Universal Instantiation)

    根据前提 ∀x P(x) 推出 P(a)。本实现忽略具体谓词结构，
    从全称量化表达式生成一个新的具体实例作为结论。
    """

    def __init__(self):
        self.name = "UniversalInstantiationRule"
        self.description = "全称实例化：∀x P(x) ⊢ P(a)"

    def num_premises(self):
        return 1

    def can_apply(self, premises):
        return len(premises) == 1

    def construct_conclusion(self, premises):
        if len(premises) != 1:
            raise ValueError("UniversalInstantiationRule需要恰好1个前提")
        premise = premises[0]
        # 若前提为 ∀x P(x)，则实例化为 P(c)
        if z3.is_quantifier(premise) and premise.is_forall():
            try:
                # 只处理单变量情况
                if premise.num_vars() == 1:
                    var_sort = premise.var_sort(0)
                    body = premise.body()
                    # 创建新的常量
                    const = z3.Const(f"c{random.randint(1, 1_000_000)}", var_sort)
                    # 使用 substitute_vars 替换绑定变量
                    instantiated = z3.substitute_vars(body, const)
                    return instantiated
            except Exception:
                pass
        # 默认：生成新的谓词实例
        const = _fresh_var("c")
        pred = _fresh_pred("P")
        return pred(const)

    def generate_premises(self, conclusion_expr, max_premises=1):
        """根据结论生成一个全称量化前提。"""
        # 如果结论是谓词实例 P(a)，则生成 ∀x P(x)
        if z3.is_app(conclusion_expr) and conclusion_expr.num_args() >= 1:
            try:
                pred = conclusion_expr.decl()
                if conclusion_expr.num_args() == 1:
                    arg = conclusion_expr.arg(0)
                    var_sort = arg.sort()
                    var = z3.Const(f"x{random.randint(1, 1_000_000)}", var_sort)
                    body = pred(var)
                    return [z3.ForAll([var], body)]
            except Exception:
                pass
        # 默认：构造一个新的全称量化表达式 ∀x P(x)
        var = _fresh_var("x")
        pred = _fresh_pred("P")
        body = pred(var)
        return [z3.ForAll([var], body)]

    def explain_step(self, premises, conclusion):
        premise = premises[0]
        return f"UniversalInstantiationRule: 由 {premise} 可得到某个具体实例 {conclusion}"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "从 ∀x P(x) 可以推出 P(a)",
            "formal": "∀x P(x) ⊢ P(a)",
            "example": "所有人都是学生，因此某个特定人是学生",
            "variables": ["量化变量x", "特定个体a", "谓词P"],
        }


class UniversalGeneralizationRule(RuleVariableMixin):
    """全称泛化规则 (Universal Generalization)

    根据前提 P(x) 推出 ∀x P(x)，但需满足侧条件：变量 x 在其他开放前提中不自由出现。
    在此简化实现中，不检查侧条件，直接根据前提生成全称量化式。
    """

    def __init__(self):
        self.name = "UniversalGeneralization"
        self.description = "全称泛化：P(x) ⊢ ∀x P(x)（侧条件：x 不自由出现在未解除前提中）"

    def num_premises(self):
        return 1

    def can_apply(self, premises):
        return len(premises) == 1

    def construct_conclusion(self, premises):
        if len(premises) != 1:
            raise ValueError("UniversalGeneralization需要恰好1个前提")
        premise = premises[0]
        # 如果前提是谓词实例 P(a)，推广为 ∀x P(x)
        if z3.is_app(premise) and premise.num_args() >= 1:
            try:
                pred = premise.decl()
                if premise.num_args() == 1:
                    arg = premise.arg(0)
                    var_sort = arg.sort()
                    var = z3.Const(f"x{random.randint(1, 1_000_000)}", var_sort)
                    body = pred(var)
                    return z3.ForAll([var], body)
            except Exception:
                pass
        # 默认：构造新的全称量化式
        var = _fresh_var("x")
        pred = _fresh_pred("P")
        body = pred(var)
        return z3.ForAll([var], body)

    def generate_premises(self, conclusion_expr, max_premises=1):
        """根据结论生成前提。"""
        # 如果结论是 ∀x P(x)，则实例化为 P(c)
        if z3.is_quantifier(conclusion_expr) and conclusion_expr.is_forall():
            try:
                if conclusion_expr.num_vars() == 1:
                    var_sort = conclusion_expr.var_sort(0)
                    body = conclusion_expr.body()
                    const = z3.Const(f"c{random.randint(1, 1_000_000)}", var_sort)
                    inst = z3.substitute_vars(body, const)
                    return [inst]
            except Exception:
                pass
        # 默认生成新的谓词实例 P(c)
        c = _fresh_var("c")
        pred = _fresh_pred("P")
        return [pred(c)]

    def explain_step(self, premises, conclusion):
        premise = premises[0]
        return f"UniversalGeneralization: 由于 {premise}，推广得到 {conclusion}"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "如果 P(x) 对任意 x 成立，则 ∀x P(x) 成立",
            "formal": "P(x) ⊢ ∀x P(x)",
            "example": "若某个属性对任意对象成立，则所有对象都具有该属性",
            "variables": ["变量x", "谓词P", "全称量化结论"]
        }


class ExistentialEliminationRule(RuleVariableMixin):
    """存在消去规则 (Existential Elimination)

    从存在量化的前提 ∃x P(x) 推出某结论 φ，需要在子证明中引入一个新的常量并利用假设 P(c) 证明 φ。
    本实现省略子证明的细节，直接返回一个新的结论变量作为推理结果。
    """

    def __init__(self):
        self.name = "ExistentialElimination"
        self.description = "存在消去：∃x P(x) ⇒ 通过假设 P(c) 推出 φ 后得到 φ"

    def num_premises(self):
        return 1

    def can_apply(self, premises):
        return len(premises) == 1

    def construct_conclusion(self, premises):
        if len(premises) != 1:
            raise ValueError("ExistentialElimination需要恰好1个前提")
        # 不深入展开存在消去的子证明，直接生成新的结论变量
        return self.create_conclusion_variable()

    def generate_premises(self, conclusion_expr, max_premises=1):
        """生成一个存在量化前提。"""
        # 构造 ∃x P(x) 的形式
        var = _fresh_var("x")
        pred = _fresh_pred("P")
        body = pred(var)
        return [z3.Exists([var], body)]

    def explain_step(self, premises, conclusion):
        premise = premises[0]
        return f"ExistentialElimination: 从 {premise} 引入新常量并假设，其子证明推出 {conclusion}"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "如果存在 x 满足 P(x)，且从假设 P(c) 推出 φ，则可以得出 φ",
            "formal": "∃x P(x) ⇒ 取新常量 c, P(c) ⊢ φ → φ",
            "example": "存在某人是学生，假设某人是学生可推出命题 φ，则命题 φ 成立",
            "variables": ["谓词P", "常量c", "目标φ"],
        }