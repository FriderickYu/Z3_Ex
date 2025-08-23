# 命题逻辑基础规则

"""
本模块实现命题逻辑中的一些基础推理规则。

当前包含的规则列表：

* **ModusTollens（否定后件）**：`¬Q, P→Q ⊢ ¬P`。
* **DisjunctiveSyllogism（析取三段论）**：`P∨Q, ¬P ⊢ Q`。
* **ConjunctionIntroductionRule（合取引入）**：`P, Q ⊢ P∧Q`（已存在于其它模块）。
* **ConjunctionEliminationRule（合取消除）**：`P∧Q ⊢ P`（已存在于其它模块）。
* **DisjunctionIntroductionRule（析取引入）**：`P ⊢ P∨Q`（已存在于其它模块）。
* **BiconditionalEliminationRule（双条件消除）**：`P↔Q ⊢ (P→Q)∧(Q→P)`（已存在于其它模块）。
* **ConsequentStrengthening（后件强化）**：`P→Q ⊢ P→(P∧Q)`。

新引入的规则类均继承自 ``RuleVariableMixin``，该混入类提供了统一的变量生成接口，以便在构造前提和结论时获得新的 Z3 布尔变量。
"""

import z3
import random
from utils.variable_manager import RuleVariableMixin


class ModusTollensRule(RuleVariableMixin):
    """否定后件规则 (Modus Tollens)

    根据前提 ¬Q 和 P→Q，推出结论 ¬P。
    """

    def __init__(self):
        self.name = "ModusTollens"
        self.description = "否定后件：¬Q, P→Q ⊢ ¬P"

    def num_premises(self):
        """返回该规则需要的前提数量。否定后件需要两个前提。"""
        return 2

    def _z3_equal(self, expr1, expr2):
        """判断两个 Z3 表达式是否等价，若简化失败则退化为字符串比较。"""
        try:
            return z3.simplify(expr1) == z3.simplify(expr2)
        except Exception:
            return str(expr1) == str(expr2)

    def can_apply(self, premises):
        """判断该规则是否可以应用于给定的前提集合。

        条件：恰好两个前提，并且一条是蕴含表达式，另一条是该蕴含后件的否定。
        """
        if len(premises) != 2:
            return False

        p1, p2 = premises

        # 情况1：p1 为蕴含，p2 为否定
        if z3.is_implies(p1) and z3.is_not(p2):
            consequent = p1.arg(1)
            negated = p2.arg(0)
            if self._z3_equal(consequent, negated):
                return True

        # 情况2：p2 为蕴含，p1 为否定
        if z3.is_implies(p2) and z3.is_not(p1):
            consequent = p2.arg(1)
            negated = p1.arg(0)
            if self._z3_equal(consequent, negated):
                return True

        return False

    def construct_conclusion(self, premises):
        """根据前提构造结论。若识别出 ¬Q 和 P→Q 的形式，则返回 ¬P；否则返回新的结论变量的否定。"""
        if len(premises) != 2:
            raise ValueError("ModusTollens需要恰好2个前提")

        p1, p2 = premises

        # 确定哪一个是蕴含，哪一个是否定
        if z3.is_implies(p1) and z3.is_not(p2):
            implication = p1
            negation = p2
        elif z3.is_implies(p2) and z3.is_not(p1):
            implication = p2
            negation = p1
        else:
            # 前提不满足形态，直接生成新的结论变量的否定
            return z3.Not(self.create_conclusion_variable())

        antecedent = implication.arg(0)
        consequent = implication.arg(1)

        # 若否定对象等于蕴含的后件，则结论为否定前件
        if self._z3_equal(negation.arg(0), consequent):
            return z3.Not(antecedent)
        else:
            # 否则生成新的结论变量的否定
            return z3.Not(self.create_conclusion_variable())

    def generate_premises(self, conclusion_expr, max_premises=2):
        """根据给定的结论反向生成前提。

        如果目标结论是 ¬P，则生成 ¬Q 和 P→Q，其中 Q 为新变量；否则随机生成两个变量形成前提。
        """
        # 尝试从结论中提取前件
        if z3.is_not(conclusion_expr):
            antecedent = conclusion_expr.arg(0)
            # 创建一个新的中间变量作为蕴含的后件
            intermediate = self.create_intermediate_variable()
            premise1 = z3.Not(intermediate)
            premise2 = z3.Implies(antecedent, intermediate)
            # 随机打乱前提顺序
            return [premise1, premise2] if random.choice([True, False]) else [premise2, premise1]
        else:
            # 结论不是简单的否定，生成随机前提
            P = self.create_premise_variable()
            Q = self.create_intermediate_variable()
            premise1 = z3.Not(Q)
            premise2 = z3.Implies(P, Q)
            return [premise1, premise2] if random.choice([True, False]) else [premise2, premise1]

    def explain_step(self, premises, conclusion):
        """生成该推理步骤的人类可读解释。"""
        if len(premises) == 2:
            return f"ModusTollens: 由于 {premises[0]} 和 {premises[1]}，可以推出 {conclusion}"
        return f"ModusTollens: 基于前提 {premises}，推出 {conclusion}"

    def get_rule_template(self):
        """返回规则的模板信息，供 UI 或文档使用。"""
        return {
            "name": self.name,
            "pattern": "如果非Q成立，且 P 蕴含 Q，那么非P成立",
            "formal": "¬Q, P→Q ⊢ ¬P",
            "example": "如果地面不湿，并且天下雨意味着地面湿，那么天下不雨",
            "variables": ["命题P", "命题Q", "蕴含关系P→Q", "否定后件¬Q", "结论¬P"]
        }


class DisjunctiveSyllogismRule(RuleVariableMixin):
    """析取三段论规则 (Disjunctive Syllogism)

    根据前提 P∨Q 和 ¬P，推出 Q。
    """

    def __init__(self):
        self.name = "DisjunctiveSyllogism"
        self.description = "析取三段论：P∨Q, ¬P ⊢ Q"

    def num_premises(self):
        """返回该规则需要的前提数量，析取消除需要两个前提。"""
        return 2

    def _z3_equal(self, expr1, expr2):
        try:
            return z3.simplify(expr1) == z3.simplify(expr2)
        except Exception:
            return str(expr1) == str(expr2)

    def can_apply(self, premises):
        """判断该规则是否可以应用。

        条件：两条前提，其中一条是析取表达式，另一条是该析取某一项的否定。
        """
        if len(premises) != 2:
            return False

        p1, p2 = premises

        # p1 为析取，p2 为否定
        if z3.is_or(p1) and z3.is_not(p2):
            disjuncts = list(p1.children())
            for d in disjuncts:
                if self._z3_equal(d, p2.arg(0)):
                    return True

        # p2 为析取，p1 为否定
        if z3.is_or(p2) and z3.is_not(p1):
            disjuncts = list(p2.children())
            for d in disjuncts:
                if self._z3_equal(d, p1.arg(0)):
                    return True

        return False

    def construct_conclusion(self, premises):
        """根据前提构造结论。如果存在某项被否定的析取，则返回另一项作为结论。"""
        if len(premises) != 2:
            raise ValueError("DisjunctiveSyllogism需要恰好2个前提")

        p1, p2 = premises

        # 确定析取表达式与否定表达式
        if z3.is_or(p1) and z3.is_not(p2):
            or_expr = p1
            neg_expr = p2
        elif z3.is_or(p2) and z3.is_not(p1):
            or_expr = p2
            neg_expr = p1
        else:
            # 模式不匹配时返回新的结论变量
            return self.create_conclusion_variable()

        disjuncts = list(or_expr.children())
        negated = neg_expr.arg(0)
        # 寻找与否定项不同的析取项
        others = [d for d in disjuncts if not self._z3_equal(d, negated)]

        if others:
            # 如果有多个其他项，随机选择其中之一作为结论
            return random.choice(others) if len(others) > 1 else others[0]
        else:
            # 理论上不会只有一项析取，此处为保险处理
            return self.create_conclusion_variable()

    def generate_premises(self, conclusion_expr, max_premises=2):
        """根据期望的结论反向生成前提。

        生成的前提包括一个包含结论与新变量的析取表达式，以及该新变量的否定。
        """
        # 创建与结论不同的新变量作为被排除项
        other = self.create_premise_variable()
        # 尽量避免与结论变量重名
        if isinstance(conclusion_expr, z3.BoolRef) and self._z3_equal(other, conclusion_expr):
            other = self.create_premise_variable()
        # 构造析取表达式（顺序随机）
        or_expr = z3.Or(conclusion_expr, other) if random.choice([True, False]) else z3.Or(other, conclusion_expr)
        not_expr = z3.Not(other)
        # 随机决定前提顺序
        return [or_expr, not_expr] if random.choice([True, False]) else [not_expr, or_expr]

    def explain_step(self, premises, conclusion):
        """生成该推理步骤的人类可读解释。"""
        if len(premises) == 2:
            return f"DisjunctiveSyllogism: 由于 {premises[0]} 和 {premises[1]}，因此可得 {conclusion}"
        return f"DisjunctiveSyllogism: 基于前提 {premises} 推出 {conclusion}"

    def get_rule_template(self):
        """返回规则模板信息。"""
        return {
            "name": self.name,
            "pattern": "如果 P 或 Q 成立，并且非P成立，那么 Q 成立",
            "formal": "P∨Q, ¬P ⊢ Q",
            "example": "如果天在下雨或者天气晴朗，并且天不在下雨，那么天气晴朗",
            "variables": ["析取项P", "析取项Q", "否定项¬P", "结论Q"]
        }


class ConsequentStrengtheningRule(RuleVariableMixin):
    """后件强化规则 (Consequent Strengthening)

    根据前提 P→Q，推出结论 P→(P∧Q)。
    """

    def __init__(self):
        self.name = "ConsequentStrengthening"
        self.description = "后件强化：P→Q ⊢ P→(P∧Q)"

    def num_premises(self):
        """后件强化仅需要一个前提。"""
        return 1

    def _z3_equal(self, expr1, expr2):
        try:
            return z3.simplify(expr1) == z3.simplify(expr2)
        except Exception:
            return str(expr1) == str(expr2)

    def can_apply(self, premises):
        """判断该规则是否可以应用。

        条件：恰好一个前提且该前提是蕴含表达式。
        """
        return len(premises) == 1 and z3.is_implies(premises[0])

    def construct_conclusion(self, premises):
        """根据前提构造结论。

        如果前提是 P→Q，则结论为 P→(P∧Q)；否则生成新的蕴含表达式。
        """
        if len(premises) != 1:
            raise ValueError("ConsequentStrengthening需要恰好1个前提")

        premise = premises[0]
        if z3.is_implies(premise):
            antecedent = premise.arg(0)
            consequent = premise.arg(1)
            # 后件强化：P→Q 推出 P→(P∧Q)
            return z3.Implies(antecedent, z3.And(antecedent, consequent))
        else:
            # 如果前提不是蕴含，生成新的随机蕴含表达式
            antecedent = self.create_premise_variable()
            conclusion_part = z3.And(self.create_premise_variable(), self.create_conclusion_variable())
            return z3.Implies(antecedent, conclusion_part)

    def generate_premises(self, conclusion_expr, max_premises=1):
        """根据期望的结论反向生成前提。

        如果结论形如 P→(P∧Q)，则生成 P→Q；否则生成随机的蕴含。"""
        # 检查结论是否为蕴含
        if z3.is_implies(conclusion_expr):
            antecedent = conclusion_expr.arg(0)
            consequent = conclusion_expr.arg(1)
            # 尝试识别后件是否为合取且包含前件
            if z3.is_and(consequent):
                children = list(consequent.children())
                q_part = None
                for child in children:
                    if not self._z3_equal(child, antecedent):
                        q_part = child
                        break
                if q_part is None and children:
                    q_part = children[0]
                if q_part is not None:
                    return [z3.Implies(antecedent, q_part)]
        # 默认生成随机蕴含作为前提
        P = self.create_premise_variable()
        Q = self.create_conclusion_variable()
        return [z3.Implies(P, Q)]

    def explain_step(self, premises, conclusion):
        """生成该推理步骤的人类可读解释。"""
        if premises:
            return f"ConsequentStrengthening: 由于 {premises[0]}，因此可以推出 {conclusion}（后件强化）"
        return f"ConsequentStrengthening: 基于前提 {premises}，推出 {conclusion}"

    def get_rule_template(self):
        """返回规则的模板信息。"""
        return {
            "name": self.name,
            "pattern": "如果 P 蕴含 Q，那么 P 蕴含 (P 且 Q)",
            "formal": "P→Q ⊢ P→(P∧Q)",
            "example": "如果天下雨意味着地面湿，那么天下雨意味着（天下雨且地面湿）",
            "variables": ["前件P", "后件Q", "蕴含关系P→Q", "强化结论P→(P∧Q)"]
        }


# -----------------------------------------------------------------------------
# 下面的四个规则此前已在其他模块中实现。为了不影响现有系统运行，
# 现将它们补充到此文件中，实现与其它规则一致的接口。


class ConjunctionIntroductionRule(RuleVariableMixin):
    """合取引入规则 (Conjunction Introduction)

    根据前提 P, Q 推出结论 P ∧ Q。此实现与已有模块保持一致。
    """

    def __init__(self):
        self.name = "ConjunctionIntroduction"
        self.description = "合取引入：P, Q ⊢ P ∧ Q"

    def num_premises(self):
        # 允许 2 到 4 个前提，两个及以上前提合取
        return random.randint(2, 4)

    def can_apply(self, premises):
        return len(premises) >= 2

    def construct_conclusion(self, premises):
        if len(premises) < 2:
            raise ValueError("ConjunctionIntroduction需要至少2个前提")
        if len(premises) == 2:
            return z3.And(premises[0], premises[1])
        else:
            return z3.And(*premises)

    def generate_premises(self, conclusion_expr, max_premises=4):
        if z3.is_and(conclusion_expr):
            return list(conclusion_expr.children())
        else:
            num_premises = random.randint(2, min(max_premises, 3))
            premises = [conclusion_expr]
            for _ in range(num_premises - 1):
                premises.append(self.create_premise_variable())
            return premises

    def explain_step(self, premises, conclusion):
        premise_strs = [str(p) for p in premises]
        return f"ConjunctionIntroduction: 由于 {' 和 '.join(premise_strs)} 都成立，因此 {conclusion} 成立"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "如果 P 成立且 Q 成立，那么 P 且 Q 成立",
            "formal": "P, Q ⊢ P ∧ Q",
            "example": "如果天在下雨且风在刮，那么天在下雨且风在刮",
            "variables": ["前提P", "前提Q", "合取结论P∧Q"]
        }


class ConjunctionEliminationRule(RuleVariableMixin):
    """合取消除规则 (Conjunction Elimination)

    根据前提 P ∧ Q 推出 P（或 Q）。此实现与已有模块保持一致。
    """

    def __init__(self):
        self.name = "ConjunctionElimination"
        self.description = "合取消除：P ∧ Q ⊢ P"

    def num_premises(self):
        return 1

    def can_apply(self, premises):
        if len(premises) != 1:
            return False
        return z3.is_and(premises[0])

    def construct_conclusion(self, premises):
        if len(premises) != 1:
            raise ValueError("ConjunctionElimination需要恰好1个前提")
        premise = premises[0]
        if not z3.is_and(premise):
            # 如果前提不是合取，则生成新的结论变量
            return self.create_conclusion_variable()
        conjuncts = list(premise.children())
        if conjuncts:
            return random.choice(conjuncts)
        else:
            return premise.arg(0) if random.choice([True, False]) else premise.arg(1)

    def generate_premises(self, conclusion_expr, max_premises=1):
        num_additional = random.randint(1, 3)
        additional_terms = []
        for _ in range(num_additional):
            additional_terms.append(self.create_premise_variable())
        all_terms = [conclusion_expr] + additional_terms
        conjunction = z3.And(*all_terms)
        return [conjunction]

    def explain_step(self, premises, conclusion):
        premise = premises[0]
        return f"ConjunctionElimination: 由于 {premise} 成立，其中包含 {conclusion}，因此 {conclusion} 成立"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "如果 P 且 Q 成立，那么 P 成立（或 Q 成立）",
            "formal": "P ∧ Q ⊢ P",
            "example": "如果天在下雨且风在刮，那么天在下雨",
            "variables": ["合取前提P∧Q", "结论P"]
        }


class DisjunctionIntroductionRule(RuleVariableMixin):
    """析取引入规则 (Disjunction Introduction)

    根据前提 P 推出 P ∨ Q。此实现与已有模块保持一致。
    """

    def __init__(self):
        self.name = "DisjunctionIntroduction"
        self.description = "析取引入：P ⊢ P ∨ Q"

    def num_premises(self):
        return 1

    def can_apply(self, premises):
        return len(premises) == 1

    def construct_conclusion(self, premises):
        if len(premises) != 1:
            raise ValueError("DisjunctionIntroduction需要恰好1个前提")
        premise = premises[0]
        additional_var = self.create_premise_variable()
        # 随机决定是 P ∨ Q 还是 Q ∨ P
        if random.choice([True, False]):
            return z3.Or(premise, additional_var)
        else:
            return z3.Or(additional_var, premise)

    def generate_premises(self, conclusion_expr, max_premises=1):
        if z3.is_or(conclusion_expr):
            disjuncts = list(conclusion_expr.children())
            if disjuncts:
                chosen_premise = random.choice(disjuncts)
                return [chosen_premise]
            else:
                # 回退处理
                return [conclusion_expr.arg(0)] if random.choice([True, False]) else [conclusion_expr.arg(1)]
        else:
            premise = self.create_premise_variable()
            return [premise]

    def explain_step(self, premises, conclusion):
        premise = premises[0]
        return f"DisjunctionIntroduction: 由于 {premise} 成立，因此 {conclusion} 也成立（析取弱化）"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "如果 P 成立，那么 P 或 Q 成立",
            "formal": "P ⊢ P ∨ Q",
            "example": "如果天在下雨，那么天在下雨或者天气晴朗",
            "variables": ["前提P", "附加变量Q", "析取结论P∨Q"],
            "note": "这是逻辑弱化的体现：从强的条件可以推出弱的条件"
        }


class BiconditionalEliminationRule(RuleVariableMixin):
    """双条件消除规则 (Biconditional Elimination)

    根据前提 P ↔ Q，推出结论 (P → Q) ∧ (Q → P)。此实现与已有模块保持一致。
    """

    def __init__(self):
        self.name = "BiconditionalElimination"
        self.description = "双条件消除：P ↔ Q ⊢ (P → Q) ∧ (Q → P)"

    def num_premises(self):
        return 1

    def can_apply(self, premises):
        if len(premises) != 1:
            return False
        premise = premises[0]
        return self._is_biconditional(premise)

    def _is_biconditional(self, expr):
        expr_str = str(expr).lower()
        return '↔' in expr_str or 'iff' in expr_str or self._is_equivalence(expr)

    def _is_equivalence(self, expr):
        try:
            return z3.is_eq(expr) and z3.is_bool(expr.arg(0))
        except Exception:
            return False

    def construct_conclusion(self, premises):
        if len(premises) != 1:
            raise ValueError("BiconditionalElimination需要恰好1个前提")
        # 创建两个变量代表双条件的两边
        var_p = self.create_premise_variable()
        var_q = self.create_premise_variable()
        # 构造 (P → Q) ∧ (Q → P)
        implication1 = z3.Implies(var_p, var_q)
        implication2 = z3.Implies(var_q, var_p)
        return z3.And(implication1, implication2)

    def generate_premises(self, conclusion_expr, max_premises=1):
        bicond_var = self.create_premise_variable()
        return [bicond_var]

    def explain_step(self, premises, conclusion):
        premise = premises[0]
        return f"BiconditionalElimination: 由于 {premise} 是双条件关系，因此 {conclusion} 成立（双向蕴含）"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "如果P当且仅当Q，那么P蕴含Q且Q蕴含P",
            "formal": "P ↔ Q ⊢ (P → Q) ∧ (Q → P)",
            "example": "如果天下雨当且仅当地面湿，那么天下雨蕴含地面湿，且地面湿蕴含天下雨",
            "variables": ["命题P", "命题Q", "双条件关系↔", "蕴含关系→"]
        }