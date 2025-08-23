# AssociativeLaws（结合律）：`(P∧Q)∧R ⟷ P∧(Q∧R)；(P∨Q)∨R ⟷ P∨(Q∨R)`

import z3
import random
from utils.variable_manager import RuleVariableMixin


class AssociativeLawsRule(RuleVariableMixin):
    """结合律规则 (Associative Laws)

    对合取和析取表达式应用结合律，重排嵌套结构。
    """

    def __init__(self):
        self.name = "AssociativeLaws"
        self.description = "结合律：P ∧ (Q ∧ R) ≡ (P ∧ Q) ∧ R；P ∨ (Q ∨ R) ≡ (P ∨ Q) ∨ R"

    def num_premises(self):
        return 1

    def _z3_equal(self, expr1, expr2):
        try:
            return z3.simplify(expr1) == z3.simplify(expr2)
        except Exception:
            return str(expr1) == str(expr2)

    def can_apply(self, premises):
        return len(premises) == 1

    def _flatten(self, expr, op):
        """递归展开同一运算符的子表达式，返回平坦化的子列表。"""
        flat = []
        for child in expr.children():
            if op(child):
                flat.extend(self._flatten(child, op))
            else:
                flat.append(child)
        return flat

    def construct_conclusion(self, premises):
        if len(premises) != 1:
            raise ValueError("AssociativeLaws需要恰好1个前提")
        expr = premises[0]
        # 如果是合取
        if z3.is_and(expr):
            # 平坦化所有合取项
            flat = self._flatten(expr, z3.is_and)
            # 少于两个子项时无法应用结合律
            if len(flat) < 3:
                # 两个元素交换顺序等同于交换律，此处直接返回原表达式
                return expr
            # 随机选择左或右结合
            if random.choice([True, False]):
                # 左结合：(A ∧ B) ∧ C ∧ ...
                sub = z3.And(flat[0], flat[1])
                for c in flat[2:]:
                    sub = z3.And(sub, c)
                return sub
            else:
                # 右结合：A ∧ (B ∧ C) ∧ ...
                sub = z3.And(flat[-2], flat[-1])
                for c in reversed(flat[:-2]):
                    sub = z3.And(c, sub)
                return sub
        # 如果是析取
        if z3.is_or(expr):
            flat = self._flatten(expr, z3.is_or)
            if len(flat) < 3:
                return expr
            if random.choice([True, False]):
                # 左结合
                sub = z3.Or(flat[0], flat[1])
                for c in flat[2:]:
                    sub = z3.Or(sub, c)
                return sub
            else:
                # 右结合
                sub = z3.Or(flat[-2], flat[-1])
                for c in reversed(flat[:-2]):
                    sub = z3.Or(c, sub)
                return sub
        # 不匹配合取或析取则生成新变量
        return self.create_conclusion_variable()

    def generate_premises(self, conclusion_expr, max_premises=1):
        """根据结论生成一个嵌套的合取或析取表达式作为前提。"""
        # 如果结论是合取或析取，则重新嵌套它的子项
        if z3.is_and(conclusion_expr) or z3.is_or(conclusion_expr):
            op = z3.And if z3.is_and(conclusion_expr) else z3.Or
            # 收集所有子项
            children = []
            for child in conclusion_expr.children():
                children.append(child)
            # 少于三个子项，无法展示结合律嵌套，直接随机插入一个新变量
            if len(children) < 3:
                # 引入新变量以构造结合律情形
                new_var = self.create_premise_variable()
                children.append(new_var)
            # 随机选择结合方向
            if random.choice([True, False]):
                # (A ⋅ B) ⋅ C ⋯
                sub = op(children[0], children[1])
                for c in children[2:]:
                    sub = op(sub, c)
            else:
                # A ⋅ (B ⋅ C) ⋯
                sub = op(children[-2], children[-1])
                for c in reversed(children[:-2]):
                    sub = op(c, sub)
            return [sub]
        # 若结论不是合取或析取，则随机生成三个命题并构造嵌套表达式
        p = conclusion_expr if isinstance(conclusion_expr, z3.BoolRef) else self.create_premise_variable()
        q = self.create_premise_variable()
        r = self.create_premise_variable()
        if random.choice([True, False]):
            # 生成合取嵌套
            if random.choice([True, False]):
                return [z3.And(z3.And(p, q), r)]
            else:
                return [z3.And(p, z3.And(q, r))]
        else:
            # 生成析取嵌套
            if random.choice([True, False]):
                return [z3.Or(z3.Or(p, q), r)]
            else:
                return [z3.Or(p, z3.Or(q, r))]

    def explain_step(self, premises, conclusion):
        premise = premises[0]
        return f"AssociativeLaws: 根据结合律，将 {premise} 重新组合为 {conclusion}"

    def get_rule_template(self):
        return {
            "name": self.name,
            "pattern": "(P ∧ Q) ∧ R 等价于 P ∧ (Q ∧ R)；(P ∨ Q) ∨ R 等价于 P ∨ (Q ∨ R)",
            "formal": "(P ∧ Q) ∧ R ↔ P ∧ (Q ∧ R)；(P ∨ Q) ∨ R ↔ P ∨ (Q ∨ R)",
            "example": "(天在下雨且风在刮)且天气晴朗 等价于 天在下雨且(风在刮且天气晴朗)",
            "variables": ["命题P", "命题Q", "命题R", "结合前提", "结合结论"]
        }