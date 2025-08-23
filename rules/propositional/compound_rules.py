# 命题逻辑复合规则

"""
本模块实现命题逻辑中的复合推理规则。

包含的规则列表：

* **ConstructiveDilemma**：`(P→Q)∧(R→S), P∨R ⊢ Q∨S`
* **DestructiveDilemma**：`(P→Q)∧(R→S), ¬Q∨¬S ⊢ ¬P∨¬R`
* **ResolutionRule（归结）**：`P∨Q, ¬P∨R ⊢ Q∨R`
* **ModusPonensRule（肯定前件）**：`P, P→Q ⊢ Q`（已存在，现拷贝实现）
* **HypotheticalSyllogismRule（假言三段论）**：`P→Q, Q→R ⊢ P→R`（已存在，现拷贝实现）

为了避免影响现有系统运行，本文件中的规则实现与其它模块中的实现保持一致接口，
包括前提数量、应用条件、结论构造、反向前提生成、解释以及规则模板信息。
"""

import z3
import random
from utils.variable_manager import RuleVariableMixin


class ConstructiveDilemmaRule(RuleVariableMixin):
    """建构两难（Constructive Dilemma）

    根据前提 (P→Q)∧(R→S) 和 P∨R 推出结论 Q∨S。
    """

    def __init__(self):
        self.name = "ConstructiveDilemma"
        self.description = "建构两难：((P→Q)∧(R→S), P∨R ⊢ Q∨S)"

    def num_premises(self):
        return 2

    def _z3_equal(self, expr1, expr2):
        try:
            return z3.simplify(expr1) == z3.simplify(expr2)
        except Exception:
            return str(expr1) == str(expr2)

    def can_apply(self, premises):
        if len(premises) != 2:
            return False
        p1, p2 = premises
        # 一个前提是合取包含蕴含，另一个前提是析取
        return (z3.is_and(p1) and z3.is_or(p2)) or (z3.is_and(p2) and z3.is_or(p1))

    def _extract_implications(self, conj_expr):
        """从合取表达式中提取所有蕴含项。"""
        if not z3.is_and(conj_expr):
            return []
        return [c for c in conj_expr.children() if z3.is_implies(c)]

    def construct_conclusion(self, premises):
        if len(premises) != 2:
            raise ValueError("ConstructiveDilemma需要恰好2个前提")
        # 找到合取前提和析取前提
        if z3.is_and(premises[0]) and z3.is_or(premises[1]):
            conj, disj = premises[0], premises[1]
        elif z3.is_and(premises[1]) and z3.is_or(premises[0]):
            conj, disj = premises[1], premises[0]
        else:
            return self.create_conclusion_variable()
        implications = self._extract_implications(conj)
        disjuncts = list(disj.children()) if z3.is_or(disj) else []
        consequents = []
        # 尝试匹配每个析取项与蕴含的前件，收集对应的后件
        for d in disjuncts:
            matched = False
            for imp in implications:
                antecedent = imp.arg(0)
                consequent = imp.arg(1)
                if self._z3_equal(d, antecedent):
                    consequents.append(consequent)
                    matched = True
                    break
            # 如果没有匹配项，则忽略该析取项
        # 构造结论
        if not consequents:
            return self.create_conclusion_variable()
        if len(consequents) == 1:
            return consequents[0]
        else:
            return z3.Or(*consequents)

    def generate_premises(self, conclusion_expr, max_premises=2):
        """根据目标结论反向生成前提。"""
        # 解析结论中的析取项作为后件
        if z3.is_or(conclusion_expr):
            parts = list(conclusion_expr.children())
            if len(parts) >= 2:
                q_part, s_part = parts[0], parts[1]
            elif len(parts) == 1:
                q_part = parts[0]
                s_part = self.create_conclusion_variable()
            else:
                q_part = self.create_conclusion_variable()
                s_part = self.create_conclusion_variable()
        else:
            # 结论不是析取，则生成两个新结论变量作为后件
            q_part = self.create_conclusion_variable()
            s_part = self.create_conclusion_variable()
        # 创建前件变量
        p_var = self.create_premise_variable()
        r_var = self.create_premise_variable()
        # 构造两个蕴含并合取
        conj = z3.And(z3.Implies(p_var, q_part), z3.Implies(r_var, s_part))
        # 构造析取前提
        disj = z3.Or(p_var, r_var)
        return [conj, disj] if random.choice([True, False]) else [disj, conj]

    def explain_step(self, premises, conclusion):
        return f"ConstructiveDilemma: 由 {premises[0]} 和 {premises[1]} 推出 {conclusion}"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "如果 (P→Q) 且 (R→S) 成立，并且 P 或 R 成立，那么 Q 或 S 成立",
            "formal": "(P→Q)∧(R→S), P∨R ⊢ Q∨S",
            "example": "如果下雨意味着湿润，且下雪意味着寒冷，并且下雨或下雪，那么湿润或寒冷",
            "variables": ["前件P", "后件Q", "前件R", "后件S", "蕴含关系", "析取关系"]
        }


class DestructiveDilemmaRule(RuleVariableMixin):
    """破坏两难（Destructive Dilemma）

    根据前提 (P→Q)∧(R→S) 和 ¬Q∨¬S 推出结论 ¬P∨¬R。
    """

    def __init__(self):
        self.name = "DestructiveDilemma"
        self.description = "破坏两难：((P→Q)∧(R→S), ¬Q∨¬S ⊢ ¬P∨¬R)"

    def num_premises(self):
        return 2

    def _z3_equal(self, expr1, expr2):
        try:
            return z3.simplify(expr1) == z3.simplify(expr2)
        except Exception:
            return str(expr1) == str(expr2)

    def can_apply(self, premises):
        if len(premises) != 2:
            return False
        p1, p2 = premises
        # 一个前提是合取包含蕴含，另一个前提是析取的否定项
        # 简化检测：一个为合取，一个为析取
        return (z3.is_and(p1) and z3.is_or(p2)) or (z3.is_and(p2) and z3.is_or(p1))

    def _extract_implications(self, conj_expr):
        if not z3.is_and(conj_expr):
            return []
        return [c for c in conj_expr.children() if z3.is_implies(c)]

    def construct_conclusion(self, premises):
        if len(premises) != 2:
            raise ValueError("DestructiveDilemma需要恰好2个前提")
        # 确定合取和析取
        if z3.is_and(premises[0]) and z3.is_or(premises[1]):
            conj, disj = premises[0], premises[1]
        elif z3.is_and(premises[1]) and z3.is_or(premises[0]):
            conj, disj = premises[1], premises[0]
        else:
            return self.create_conclusion_variable()
        implications = self._extract_implications(conj)
        disjuncts = list(disj.children())
        negated_antecedents = []
        # 对于析取的每个否定项 ¬Q 或 ¬S，找到对应的蕴含的后件 Q 或 S，并取其前件 P 或 R
        for lit in disjuncts:
            if z3.is_not(lit):
                target = lit.arg(0)
                for imp in implications:
                    antecedent = imp.arg(0)
                    consequent = imp.arg(1)
                    if self._z3_equal(consequent, target):
                        negated_antecedents.append(z3.Not(antecedent))
                        break
        if not negated_antecedents:
            return self.create_conclusion_variable()
        if len(negated_antecedents) == 1:
            return negated_antecedents[0]
        else:
            return z3.Or(*negated_antecedents)

    def generate_premises(self, conclusion_expr, max_premises=2):
        """根据结论反向生成前提。"""
        # 如果结论是析取形式的否定，如 ¬P∨¬R
        if z3.is_or(conclusion_expr):
            parts = list(conclusion_expr.children())
            # 寻找两个否定项
            negs = [p for p in parts if z3.is_not(p)]
            if len(negs) >= 2:
                # 取前件变量
                p_neg = negs[0].arg(0)
                r_neg = negs[1].arg(0)
            elif len(negs) == 1:
                p_neg = negs[0].arg(0)
                r_neg = self.create_premise_variable()
            else:
                p_neg = self.create_premise_variable()
                r_neg = self.create_premise_variable()
        else:
            p_neg = self.create_premise_variable()
            r_neg = self.create_premise_variable()
        # 创建后件变量
        q_var = self.create_conclusion_variable()
        s_var = self.create_conclusion_variable()
        # 构造合取前提 (P→Q) ∧ (R→S)
        conj = z3.And(z3.Implies(p_neg, q_var), z3.Implies(r_neg, s_var))
        # 构造析取前提 ¬Q∨¬S
        disj = z3.Or(z3.Not(q_var), z3.Not(s_var))
        return [conj, disj] if random.choice([True, False]) else [disj, conj]

    def explain_step(self, premises, conclusion):
        return f"DestructiveDilemma: 由 {premises[0]} 和 {premises[1]} 推出 {conclusion}"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "如果 (P→Q) 且 (R→S) 成立，并且非Q或非S成立，那么非P或非R成立",
            "formal": "(P→Q)∧(R→S), ¬Q∨¬S ⊢ ¬P∨¬R",
            "example": "如果天在下雨意味着地面会湿，且天降雪意味着天气会冷，并且地面不湿或天气不冷，那么天不下雨或天不降雪",
            "variables": ["前件P", "后件Q", "前件R", "后件S", "蕴含关系", "否定析取"]
        }


class ResolutionRule(RuleVariableMixin):
    """归结规则 (Resolution Rule)

    根据前提 P∨Q 和 ¬P∨R 推出结论 Q∨R。
    """

    def __init__(self):
        self.name = "ResolutionRule"
        self.description = "归结：P∨Q, ¬P∨R ⊢ Q∨R"

    def num_premises(self):
        return 2

    def _z3_equal(self, expr1, expr2):
        try:
            return z3.simplify(expr1) == z3.simplify(expr2)
        except Exception:
            return str(expr1) == str(expr2)

    def _is_complementary(self, a, b):
        """判断两个字句是否互补（一个是另一个的否定）。"""
        if z3.is_not(a) and self._z3_equal(a.arg(0), b):
            return True
        if z3.is_not(b) and self._z3_equal(b.arg(0), a):
            return True
        return False

    def can_apply(self, premises):
        if len(premises) != 2:
            return False
        p1, p2 = premises
        if not (z3.is_or(p1) and z3.is_or(p2)):
            return False
        disj1 = list(p1.children())
        disj2 = list(p2.children())
        for d1 in disj1:
            for d2 in disj2:
                if self._is_complementary(d1, d2):
                    return True
        return False

    def construct_conclusion(self, premises):
        if len(premises) != 2:
            raise ValueError("ResolutionRule需要恰好2个前提")
        p1, p2 = premises
        if not (z3.is_or(p1) and z3.is_or(p2)):
            return self.create_conclusion_variable()
        disj1 = list(p1.children())
        disj2 = list(p2.children())
        # 搜索互补对，构造结论
        for d1 in disj1:
            for d2 in disj2:
                if self._is_complementary(d1, d2):
                    rest1 = [x for x in disj1 if not self._z3_equal(x, d1)]
                    rest2 = [x for x in disj2 if not self._z3_equal(x, d2)]
                    combined = rest1 + rest2
                    if not combined:
                        # 如果没有其余项，则返回新变量
                        return self.create_conclusion_variable()
                    if len(combined) == 1:
                        return combined[0]
                    else:
                        return z3.Or(*combined)
        return self.create_conclusion_variable()

    def generate_premises(self, conclusion_expr, max_premises=2):
        """根据结论反向生成前提。"""
        # 如果结论是析取，则取两个项作为 Q, R
        if z3.is_or(conclusion_expr):
            parts = list(conclusion_expr.children())
            if len(parts) >= 2:
                q_part, r_part = parts[0], parts[1]
            elif len(parts) == 1:
                q_part = parts[0]
                r_part = self.create_conclusion_variable()
            else:
                q_part = self.create_conclusion_variable()
                r_part = self.create_conclusion_variable()
        else:
            q_part = conclusion_expr
            r_part = self.create_conclusion_variable()
        # 创建一个新变量 P
        p_var = self.create_premise_variable()
        # 前提1：P ∨ Q；前提2：¬P ∨ R
        premise1 = z3.Or(p_var, q_part)
        premise2 = z3.Or(z3.Not(p_var), r_part)
        return [premise1, premise2] if random.choice([True, False]) else [premise2, premise1]

    def explain_step(self, premises, conclusion):
        return f"Resolution: 由 {premises[0]} 和 {premises[1]} 归结得到 {conclusion}"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "若有 P∨Q 和 ¬P∨R，则可归结出 Q∨R",
            "formal": "P∨Q, ¬P∨R ⊢ Q∨R",
            "example": "如果有命题a或b，并且有非a或c，那么可以推出b或c",
            "variables": ["字句P", "字句Q", "字句R"]
        }


class ModusPonensRule(RuleVariableMixin):
    """肯定前件推理规则 (Modus Ponens)

    规则形式：如果有 P 和 P → Q，则可以推出 Q。
    此实现为现有规则的拷贝。
    """

    def __init__(self):
        self.name = "ModusPonens"
        self.description = "肯定前件：P, P→Q ⊢ Q"

    def num_premises(self):
        return 2

    def _z3_equal(self, expr1, expr2):
        try:
            return z3.simplify(expr1) == z3.simplify(expr2)
        except Exception:
            return str(expr1) == str(expr2)

    def can_apply(self, premises):
        if len(premises) != 2:
            return False
        p1, p2 = premises
        # 情况1：p1是P，p2是P→Q
        if z3.is_implies(p2):
            antecedent = p2.arg(0)
            if self._z3_equal(p1, antecedent):
                return True
        # 情况2：p2是P，p1是P→Q
        if z3.is_implies(p1):
            antecedent = p1.arg(0)
            if self._z3_equal(p2, antecedent):
                return True
        return False

    def construct_conclusion(self, premises):
        if len(premises) != 2:
            raise ValueError("ModusPonens需要恰好2个前提")
        p1, p2 = premises
        # 找到蕴含关系和对应的前件
        if z3.is_implies(p1):
            implication = p1
            premise = p2
        elif z3.is_implies(p2):
            implication = p2
            premise = p1
        else:
            return self.create_conclusion_variable()
        antecedent = implication.arg(0)
        consequent = implication.arg(1)
        if self._z3_equal(premise, antecedent):
            return consequent
        else:
            return self.create_conclusion_variable()

    def generate_premises(self, conclusion_expr, max_premises=2):
        # 创建前提变量
        premise_var = self.create_premise_variable()
        # 构造蕴含关系
        implication = z3.Implies(premise_var, conclusion_expr)
        return [premise_var, implication]

    def explain_step(self, premises, conclusion):
        if len(premises) != 2:
            return f"ModusPonens: 无法解释，前提数量不正确"
        p1, p2 = premises
        if z3.is_implies(p1):
            return f"ModusPonens: 由于有 {p2} 且有 {p1}，因此可以推出 {conclusion}"
        elif z3.is_implies(p2):
            return f"ModusPonens: 由于有 {p1} 且有 {p2}，因此可以推出 {conclusion}"
        else:
            return f"ModusPonens: 基于前提 {premises}，推出 {conclusion}"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "如果 P 成立，并且 P 蕴含 Q，那么 Q 成立",
            "formal": "P, P → Q ⊢ Q",
            "example": "如果天下雨，并且天下雨意味着地面会湿，那么地面会湿",
            "variables": ["前提P", "蕴含关系P→Q", "结论Q"]
        }


class HypotheticalSyllogismRule(RuleVariableMixin):
    """假言三段论规则 (Hypothetical Syllogism)

    根据前提 P→Q 和 Q→R 推出结论 P→R。
    此实现为现有规则的拷贝。
    """

    def __init__(self):
        self.name = "HypotheticalSyllogism"
        self.description = "假言三段论：P → Q, Q → R ⊢ P → R"

    def num_premises(self):
        return 2

    def _z3_equal(self, expr1, expr2):
        try:
            return z3.simplify(expr1) == z3.simplify(expr2)
        except Exception:
            return str(expr1) == str(expr2)

    def can_apply(self, premises):
        if len(premises) != 2:
            return False
        p1, p2 = premises
        if not (z3.is_implies(p1) and z3.is_implies(p2)):
            return False
        antecedent1 = p1.arg(0)
        consequent1 = p1.arg(1)
        antecedent2 = p2.arg(0)
        consequent2 = p2.arg(1)
        # 检查传递性连接
        if self._z3_equal(consequent1, antecedent2):
            return True
        if self._z3_equal(consequent2, antecedent1):
            return True
        return False

    def construct_conclusion(self, premises):
        if len(premises) != 2:
            raise ValueError("HypotheticalSyllogism需要恰好2个前提")
        p1, p2 = premises
        if not (z3.is_implies(p1) and z3.is_implies(p2)):
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
        return self._create_new_implication()

    def _create_new_implication(self):
        antecedent = self.create_premise_variable()
        consequent = self.create_conclusion_variable()
        return z3.Implies(antecedent, consequent)

    def generate_premises(self, conclusion_expr, max_premises=2):
        if not z3.is_implies(conclusion_expr):
            return self._generate_random_premises()
        antecedent = conclusion_expr.arg(0)
        consequent = conclusion_expr.arg(1)
        # 创建中间变量
        intermediate_var = self.create_intermediate_variable()
        # 构造前提：P → Q 和 Q → R
        premise1 = z3.Implies(antecedent, intermediate_var)
        premise2 = z3.Implies(intermediate_var, consequent)
        return [premise1, premise2]

    def _generate_random_premises(self):
        P = self.create_premise_variable()
        Q = self.create_intermediate_variable()
        R = self.create_conclusion_variable()
        return [z3.Implies(P, Q), z3.Implies(Q, R)]

    def explain_step(self, premises, conclusion):
        if len(premises) == 2:
            p1, p2 = premises
            return f"HypotheticalSyllogism: 由于有 {p1} 且有 {p2}，通过传递性可以推出 {conclusion}"
        else:
            return f"HypotheticalSyllogism: 基于前提 {premises}，通过传递性推出 {conclusion}"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "如果 P 蕴含 Q，且 Q 蕴含 R，那么 P 蕴含 R",
            "formal": "P → Q, Q → R ⊢ P → R",
            "example": "如果天下雨就会积水，积水就会影响交通，那么天下雨就会影响交通",
            "variables": ["前提P", "中间项Q", "结论R", "蕴含关系"],
            "logical_property": "传递性"
        }