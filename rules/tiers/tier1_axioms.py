from typing import List, Dict, Any, Set
import random
from rules.base.rule import BaseRule, RuleType, LogicalFormula, LogicalOperator


class ModusPonensRule(BaseRule):
    """
    肯定前件规则 (Modus Ponens)
    如果 P → Q 且 P，则 Q
    """

    def __init__(self):
        super().__init__("modus_ponens", RuleType.AXIOM, tier=1)
        self.template_patterns = [
            {"premises": ["{P} → {Q}", "{P}"], "conclusion": "{Q}"},
            {"premises": ["If {P} then {Q}", "{P}"], "conclusion": "{Q}"}
        ]

    def can_apply(self, premises: List[LogicalFormula]) -> bool:
        """检查是否存在 P→Q 和 P 的组合"""
        if len(premises) < 2:
            return False

        # 寻找蕴含关系和对应的前件
        implications = []
        atomic_formulas = []

        for formula in premises:
            if LogicalOperator.IMPLIES in formula.operators:
                implications.append(formula)
            else:
                atomic_formulas.append(formula)

        # 检查是否存在匹配的蕴含和前件
        for impl in implications:
            antecedent = self._extract_antecedent(impl.expression)
            for atomic in atomic_formulas:
                if self._formulas_match(antecedent, atomic.expression):
                    return True

        return False

    def apply(self, premises: List[LogicalFormula]) -> List[LogicalFormula]:
        """应用肯定前件规则"""
        if not self.can_apply(premises):
            raise ValueError("无法应用肯定前件规则")

        # 找到匹配的蕴含和前件
        for impl_formula in premises:
            if LogicalOperator.IMPLIES not in impl_formula.operators:
                continue

            antecedent = self._extract_antecedent(impl_formula.expression)
            consequent = self._extract_consequent(impl_formula.expression)

            for atomic_formula in premises:
                if (atomic_formula != impl_formula and
                        self._formulas_match(antecedent, atomic_formula.expression)):
                    # 生成结论
                    conclusion = LogicalFormula(
                        expression=consequent,
                        variables=self._extract_variables(consequent),
                        operators=[],
                        complexity=atomic_formula.complexity + 1
                    )

                    self.logger.debug(f"应用肯定前件: {antecedent} → {consequent}, {antecedent} ⊢ {consequent}")
                    return [conclusion]

        raise RuntimeError("应用肯定前件规则失败")

    def get_template(self) -> Dict[str, Any]:
        """获取规则模板"""
        return {
            "rule_name": "modus_ponens",
            "patterns": self.template_patterns,
            "premise_patterns": ["{P} → {Q}", "{P}"],
            "conclusion_patterns": ["{Q}"],
            "variables": ["P", "Q"],
            "complexity_range": (1, 3)
        }

    def _extract_antecedent(self, expression: str) -> str:
        """提取蕴含关系的前件"""
        parts = expression.split('→')
        if len(parts) >= 2:
            return parts[0].strip()

        # 处理 "If ... then ..." 格式
        if "If " in expression and " then " in expression:
            start = expression.find("If ") + 3
            end = expression.find(" then ")
            return expression[start:end].strip()

        return expression

    def _extract_consequent(self, expression: str) -> str:
        """提取蕴含关系的后件"""
        parts = expression.split('→')
        if len(parts) >= 2:
            return parts[1].strip()

        # 处理 "If ... then ..." 格式
        if " then " in expression:
            start = expression.find(" then ") + 6
            return expression[start:].strip()

        return expression

    def _formulas_match(self, expr1: str, expr2: str) -> bool:
        """检查两个表达式是否匹配"""
        return expr1.strip() == expr2.strip()

    def _extract_variables(self, expression: str) -> Set[str]:
        """从表达式中提取变量"""
        # 简化实现：假设单个字母为变量
        import re
        variables = set(re.findall(r'\b[A-Z]\b', expression))
        return variables


class ModusTollensRule(BaseRule):
    """
    否定后件规则 (Modus Tollens)
    如果 P → Q 且 ¬Q，则 ¬P
    """

    def __init__(self):
        super().__init__("modus_tollens", RuleType.AXIOM, tier=1)

    def can_apply(self, premises: List[LogicalFormula]) -> bool:
        """检查是否存在 P→Q 和 ¬Q 的组合"""
        if len(premises) < 2:
            return False

        implications = []
        negations = []

        for formula in premises:
            if LogicalOperator.IMPLIES in formula.operators:
                implications.append(formula)
            elif LogicalOperator.NOT in formula.operators:
                negations.append(formula)

        # 检查是否存在匹配的蕴含和否定后件
        for impl in implications:
            consequent = self._extract_consequent(impl.expression)
            for neg in negations:
                neg_content = self._extract_negated_content(neg.expression)
                if self._formulas_match(consequent, neg_content):
                    return True

        return False

    def apply(self, premises: List[LogicalFormula]) -> List[LogicalFormula]:
        """应用否定后件规则"""
        if not self.can_apply(premises):
            raise ValueError("无法应用否定后件规则")

        for impl_formula in premises:
            if LogicalOperator.IMPLIES not in impl_formula.operators:
                continue

            antecedent = self._extract_antecedent(impl_formula.expression)
            consequent = self._extract_consequent(impl_formula.expression)

            for neg_formula in premises:
                if (LogicalOperator.NOT in neg_formula.operators and
                        neg_formula != impl_formula):

                    neg_content = self._extract_negated_content(neg_formula.expression)
                    if self._formulas_match(consequent, neg_content):
                        # 生成否定前件的结论
                        conclusion_expr = f"¬{antecedent}"
                        conclusion = LogicalFormula(
                            expression=conclusion_expr,
                            variables=self._extract_variables(antecedent),
                            operators=[LogicalOperator.NOT],
                            complexity=impl_formula.complexity + 1
                        )

                        self.logger.debug(f"应用否定后件: {antecedent} → {consequent}, ¬{consequent} ⊢ ¬{antecedent}")
                        return [conclusion]

        raise RuntimeError("应用否定后件规则失败")

    def get_template(self) -> Dict[str, Any]:
        """获取规则模板"""
        return {
            "rule_name": "modus_tollens",
            "patterns": [
                {"premises": ["{P} → {Q}", "¬{Q}"], "conclusion": "¬{P}"}
            ],
            "premise_patterns": ["{P} → {Q}", "¬{Q}"],
            "conclusion_patterns": ["¬{P}"],
            "variables": ["P", "Q"],
            "complexity_range": (2, 4)
        }

    def _extract_antecedent(self, expression: str) -> str:
        """提取蕴含关系的前件"""
        return ModusPonensRule._extract_antecedent(self, expression)

    def _extract_consequent(self, expression: str) -> str:
        """提取蕴含关系的后件"""
        return ModusPonensRule._extract_consequent(self, expression)

    def _formulas_match(self, expr1: str, expr2: str) -> bool:
        """检查两个表达式是否匹配"""
        return expr1.strip() == expr2.strip()

    def _extract_negated_content(self, expression: str) -> str:
        """提取否定表达式的内容"""
        if expression.startswith('¬'):
            return expression[1:].strip()
        elif expression.startswith('not '):
            return expression[4:].strip()
        return expression

    def _extract_variables(self, expression: str) -> Set[str]:
        """从表达式中提取变量"""
        import re
        variables = set(re.findall(r'\b[A-Z]\b', expression))
        return variables


class HypotheticalSyllogismRule(BaseRule):
    """
    假言三段论规则 (Hypothetical Syllogism)
    如果 P → Q 且 Q → R，则 P → R
    """

    def __init__(self):
        super().__init__("hypothetical_syllogism", RuleType.AXIOM, tier=1)

    def can_apply(self, premises: List[LogicalFormula]) -> bool:
        """检查是否存在可连接的蕴含关系"""
        if len(premises) < 2:
            return False

        implications = [f for f in premises if LogicalOperator.IMPLIES in f.operators]

        if len(implications) < 2:
            return False

        # 检查是否存在可连接的蕴含关系
        for i, impl1 in enumerate(implications):
            consequent1 = self._extract_consequent(impl1.expression)
            for j, impl2 in enumerate(implications):
                if i != j:
                    antecedent2 = self._extract_antecedent(impl2.expression)
                    if self._formulas_match(consequent1, antecedent2):
                        return True

        return False

    def apply(self, premises: List[LogicalFormula]) -> List[LogicalFormula]:
        """应用假言三段论规则"""
        if not self.can_apply(premises):
            raise ValueError("无法应用假言三段论规则")

        implications = [f for f in premises if LogicalOperator.IMPLIES in f.operators]

        for i, impl1 in enumerate(implications):
            antecedent1 = self._extract_antecedent(impl1.expression)
            consequent1 = self._extract_consequent(impl1.expression)

            for j, impl2 in enumerate(implications):
                if i != j:
                    antecedent2 = self._extract_antecedent(impl2.expression)
                    consequent2 = self._extract_consequent(impl2.expression)

                    if self._formulas_match(consequent1, antecedent2):
                        # 生成新的蕴含关系
                        conclusion_expr = f"{antecedent1} → {consequent2}"
                        variables = self._extract_variables(antecedent1) | self._extract_variables(consequent2)

                        conclusion = LogicalFormula(
                            expression=conclusion_expr,
                            variables=variables,
                            operators=[LogicalOperator.IMPLIES],
                            complexity=max(impl1.complexity, impl2.complexity) + 1
                        )

                        self.logger.debug(
                            f"应用假言三段论: {antecedent1} → {consequent1}, {antecedent2} → {consequent2} ⊢ {conclusion_expr}")
                        return [conclusion]

        raise RuntimeError("应用假言三段论规则失败")

    def get_template(self) -> Dict[str, Any]:
        """获取规则模板"""
        return {
            "rule_name": "hypothetical_syllogism",
            "patterns": [
                {"premises": ["{P} → {Q}", "{Q} → {R}"], "conclusion": "{P} → {R}"}
            ],
            "premise_patterns": ["{P} → {Q}", "{Q} → {R}"],
            "conclusion_patterns": ["{P} → {R}"],
            "variables": ["P", "Q", "R"],
            "complexity_range": (2, 5)
        }

    def _extract_antecedent(self, expression: str) -> str:
        """提取蕴含关系的前件"""
        return ModusPonensRule._extract_antecedent(self, expression)

    def _extract_consequent(self, expression: str) -> str:
        """提取蕴含关系的后件"""
        return ModusPonensRule._extract_consequent(self, expression)

    def _formulas_match(self, expr1: str, expr2: str) -> bool:
        """检查两个表达式是否匹配"""
        return expr1.strip() == expr2.strip()

    def _extract_variables(self, expression: str) -> Set[str]:
        """从表达式中提取变量"""
        import re
        variables = set(re.findall(r'\b[A-Z]\b', expression))
        return variables


class DisjunctiveSyllogismRule(BaseRule):
    """
    析取三段论规则 (Disjunctive Syllogism)
    如果 P ∨ Q 且 ¬P，则 Q
    """

    def __init__(self):
        super().__init__("disjunctive_syllogism", RuleType.AXIOM, tier=1)

    def can_apply(self, premises: List[LogicalFormula]) -> bool:
        """检查是否存在析取和对应否定的组合"""
        if len(premises) < 2:
            return False

        disjunctions = [f for f in premises if LogicalOperator.OR in f.operators]
        negations = [f for f in premises if LogicalOperator.NOT in f.operators]

        if not disjunctions or not negations:
            return False

        # 检查析取的一个分量是否被否定
        for disj in disjunctions:
            disjuncts = self._extract_disjuncts(disj.expression)
            for neg in negations:
                neg_content = self._extract_negated_content(neg.expression)
                if any(self._formulas_match(d, neg_content) for d in disjuncts):
                    return True

        return False

    def apply(self, premises: List[LogicalFormula]) -> List[LogicalFormula]:
        """应用析取三段论规则"""
        if not self.can_apply(premises):
            raise ValueError("无法应用析取三段论规则")

        disjunctions = [f for f in premises if LogicalOperator.OR in f.operators]
        negations = [f for f in premises if LogicalOperator.NOT in f.operators]

        for disj in disjunctions:
            disjuncts = self._extract_disjuncts(disj.expression)

            for neg in negations:
                neg_content = self._extract_negated_content(neg.expression)

                # 找到被否定的析取分量
                for i, disjunct in enumerate(disjuncts):
                    if self._formulas_match(disjunct, neg_content):
                        # 其他分量为结论
                        remaining_disjuncts = [d for j, d in enumerate(disjuncts) if j != i]

                        if len(remaining_disjuncts) == 1:
                            conclusion_expr = remaining_disjuncts[0]
                        else:
                            conclusion_expr = " ∨ ".join(remaining_disjuncts)

                        variables = self._extract_variables(conclusion_expr)
                        operators = [LogicalOperator.OR] if len(remaining_disjuncts) > 1 else []

                        conclusion = LogicalFormula(
                            expression=conclusion_expr,
                            variables=variables,
                            operators=operators,
                            complexity=disj.complexity
                        )

                        self.logger.debug(f"应用析取三段论: {disj.expression}, {neg.expression} ⊢ {conclusion_expr}")
                        return [conclusion]

        raise RuntimeError("应用析取三段论规则失败")

    def get_template(self) -> Dict[str, Any]:
        """获取规则模板"""
        return {
            "rule_name": "disjunctive_syllogism",
            "patterns": [
                {"premises": ["{P} ∨ {Q}", "¬{P}"], "conclusion": "{Q}"},
                {"premises": ["{P} ∨ {Q}", "¬{Q}"], "conclusion": "{P}"}
            ],
            "premise_patterns": ["{P} ∨ {Q}", "¬{P}"],
            "conclusion_patterns": ["{Q}"],
            "variables": ["P", "Q"],
            "complexity_range": (2, 4)
        }

    def _extract_disjuncts(self, expression: str) -> List[str]:
        """提取析取表达式的分量"""
        # 分割 ∨ 操作符
        parts = expression.split('∨')
        return [part.strip() for part in parts]

    def _extract_negated_content(self, expression: str) -> str:
        """提取否定表达式的内容"""
        if expression.startswith('¬'):
            return expression[1:].strip()
        elif expression.startswith('not '):
            return expression[4:].strip()
        return expression

    def _formulas_match(self, expr1: str, expr2: str) -> bool:
        """检查两个表达式是否匹配"""
        return expr1.strip() == expr2.strip()

    def _extract_variables(self, expression: str) -> Set[str]:
        """从表达式中提取变量"""
        import re
        variables = set(re.findall(r'\b[A-Z]\b', expression))
        return variables


class ConjunctionIntroductionRule(BaseRule):
    """
    合取引入规则 (Conjunction Introduction)
    如果 P 且 Q，则 P ∧ Q
    """

    def __init__(self):
        super().__init__("conjunction_introduction", RuleType.AXIOM, tier=1)

    def can_apply(self, premises: List[LogicalFormula]) -> bool:
        """检查是否有多个独立的前提可以合取"""
        # 至少需要两个前提
        if len(premises) < 2:
            return False

        # 检查是否有非合取的独立前提
        independent_premises = [f for f in premises if LogicalOperator.AND not in f.operators]
        return len(independent_premises) >= 2

    def apply(self, premises: List[LogicalFormula]) -> List[LogicalFormula]:
        """应用合取引入规则"""
        if not self.can_apply(premises):
            raise ValueError("无法应用合取引入规则")

        # 选择两个或多个独立前提进行合取
        independent_premises = [f for f in premises if LogicalOperator.AND not in f.operators]

        # 随机选择2-3个前提进行合取
        num_to_combine = min(3, random.randint(2, len(independent_premises)))
        selected_premises = random.sample(independent_premises, num_to_combine)

        # 构建合取表达式
        expressions = [p.expression for p in selected_premises]
        conjunction_expr = " ∧ ".join(expressions)

        # 合并变量和操作符
        all_variables = set()
        all_operators = [LogicalOperator.AND]
        max_complexity = 0

        for premise in selected_premises:
            all_variables.update(premise.variables)
            all_operators.extend(premise.operators)
            max_complexity = max(max_complexity, premise.complexity)

        conclusion = LogicalFormula(
            expression=conjunction_expr,
            variables=all_variables,
            operators=all_operators,
            complexity=max_complexity + 1
        )

        premise_exprs = [p.expression for p in selected_premises]
        self.logger.debug(f"应用合取引入: {', '.join(premise_exprs)} ⊢ {conjunction_expr}")
        return [conclusion]

    def get_template(self) -> Dict[str, Any]:
        """获取规则模板"""
        return {
            "rule_name": "conjunction_introduction",
            "patterns": [
                {"premises": ["{P}", "{Q}"], "conclusion": "{P} ∧ {Q}"},
                {"premises": ["{P}", "{Q}", "{R}"], "conclusion": "{P} ∧ {Q} ∧ {R}"}
            ],
            "premise_patterns": ["{P}", "{Q}"],
            "conclusion_patterns": ["{P} ∧ {Q}"],
            "variables": ["P", "Q", "R"],
            "complexity_range": (1, 3)
        }


class ConjunctionEliminationRule(BaseRule):
    """
    合取消除规则 (Conjunction Elimination)
    如果 P ∧ Q，则 P 和 Q
    """

    def __init__(self):
        super().__init__("conjunction_elimination", RuleType.AXIOM, tier=1)

    def can_apply(self, premises: List[LogicalFormula]) -> bool:
        """检查是否存在合取表达式"""
        return any(LogicalOperator.AND in f.operators for f in premises)

    def apply(self, premises: List[LogicalFormula]) -> List[LogicalFormula]:
        """应用合取消除规则"""
        if not self.can_apply(premises):
            raise ValueError("无法应用合取消除规则")

        conclusions = []

        for premise in premises:
            if LogicalOperator.AND in premise.operators:
                # 分解合取表达式
                conjuncts = self._extract_conjuncts(premise.expression)

                for conjunct in conjuncts:
                    variables = self._extract_variables(conjunct)
                    # 确定操作符（去除AND）
                    operators = [op for op in premise.operators if op != LogicalOperator.AND]

                    conclusion = LogicalFormula(
                        expression=conjunct,
                        variables=variables,
                        operators=operators,
                        complexity=premise.complexity - 1
                    )
                    conclusions.append(conclusion)

                conjunct_strs = [c for c in conjuncts]
                self.logger.debug(f"应用合取消除: {premise.expression} ⊢ {', '.join(conjunct_strs)}")
                break  # 只处理第一个合取表达式

        return conclusions

    def get_template(self) -> Dict[str, Any]:
        """获取规则模板"""
        return {
            "rule_name": "conjunction_elimination",
            "patterns": [
                {"premises": ["{P} ∧ {Q}"], "conclusion": ["{P}", "{Q}"]},
                {"premises": ["{P} ∧ {Q} ∧ {R}"], "conclusion": ["{P}", "{Q}", "{R}"]}
            ],
            "premise_patterns": ["{P} ∧ {Q}"],
            "conclusion_patterns": ["{P}", "{Q}"],
            "variables": ["P", "Q", "R"],
            "complexity_range": (1, 2)
        }

    def _extract_conjuncts(self, expression: str) -> List[str]:
        """提取合取表达式的分量"""
        # 分割 ∧ 操作符
        parts = expression.split('∧')
        return [part.strip() for part in parts]

    def _extract_variables(self, expression: str) -> Set[str]:
        """从表达式中提取变量"""
        import re
        variables = set(re.findall(r'\b[A-Z]\b', expression))
        return variables