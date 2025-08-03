# 文件：rules/rules_pool.py
# 说明：真实逻辑规则池，包含8个基础逻辑推理规则

import random

# 导入所有真实的逻辑规则
from .modus_ponens import ModusPonensRule
from .conjunction_rules import ConjunctionIntroductionRule, ConjunctionEliminationRule
from .disjunction_introduction import DisjunctionIntroductionRule
from .hypothetical_syllogism import HypotheticalSyllogismRule
from .universal_instantiation import UniversalInstantiationRule
from .transitivity_rule import TransitivityRule
from .biconditional_elimination import BiconditionalEliminationRule


class RealLogicRulesPool:
    """
    真实逻辑规则池

    包含8个基础逻辑推理规则，能够生成有意义的推理链
    而不是之前的恒等式
    """

    def __init__(self):
        # 初始化所有真实逻辑规则
        self.rules = [
            ModusPonensRule(),  # 肯定前件：P, P→Q ⊢ Q
            ConjunctionIntroductionRule(),  # 合取引入：P, Q ⊢ P∧Q
            ConjunctionEliminationRule(),  # 合取消除：P∧Q ⊢ P
            DisjunctionIntroductionRule(),  # 析取引入：P ⊢ P∨Q
            HypotheticalSyllogismRule(),  # 假言三段论：P→Q, Q→R ⊢ P→R
            UniversalInstantiationRule(),  # 全称实例化：∀x P(x) ⊢ P(a)
            TransitivityRule(),  # 传递性：aRb, bRc ⊢ aRc
            BiconditionalEliminationRule(),  # 双条件消除：P↔Q ⊢ (P→Q)∧(Q→P)
        ]

        # 按规则类型分类，便于有针对性的选择
        self.rule_categories = {
            "implication": [ModusPonensRule(), HypotheticalSyllogismRule()],
            "conjunction": [ConjunctionIntroductionRule(), ConjunctionEliminationRule()],
            "disjunction": [DisjunctionIntroductionRule()],
            "quantifier": [UniversalInstantiationRule()],
            "relation": [TransitivityRule()],
            "biconditional": [BiconditionalEliminationRule()]
        }

        # 推理强度分类（用于构建推理链）
        self.strength_categories = {
            "strengthening": [ConjunctionIntroductionRule()],  # 结论更强
            "weakening": [DisjunctionIntroductionRule(), ConjunctionEliminationRule()],  # 结论更弱
            "preserving": [ModusPonensRule(), HypotheticalSyllogismRule(), TransitivityRule()],  # 强度保持
            "transforming": [BiconditionalEliminationRule(), UniversalInstantiationRule()]  # 形式转换
        }

    def sample_rule(self, category=None, strength=None):
        """
        从规则池中采样一个规则

        Args:
            category: 规则类别（可选）
            strength: 推理强度类别（可选）

        Returns:
            Rule: 采样的规则实例
        """
        if category and category in self.rule_categories:
            candidates = self.rule_categories[category]
        elif strength and strength in self.strength_categories:
            candidates = self.strength_categories[strength]
        else:
            candidates = self.rules

        return random.choice(candidates)

    def sample_rules(self, max_rules=None, ensure_diversity=True):
        """
        从规则池中采样多个规则

        Args:
            max_rules: 最大规则数量
            ensure_diversity: 是否确保规则类型多样性

        Returns:
            list: 规则列表
        """
        if max_rules is None:
            max_rules = len(self.rules)

        if ensure_diversity:
            # 确保从不同类别中选择规则
            selected_rules = []
            categories = list(self.rule_categories.keys())
            random.shuffle(categories)

            for category in categories[:max_rules]:
                rule = self.sample_rule(category=category)
                selected_rules.append(rule)

            # 如果还需要更多规则，随机补充
            while len(selected_rules) < max_rules:
                rule = self.sample_rule()
                if rule not in selected_rules:
                    selected_rules.append(rule)

            return selected_rules[:max_rules]
        else:
            return random.sample(self.rules, min(max_rules, len(self.rules)))

    def sample_rule_for_conclusion(self, conclusion_expr=None, goal_only=False):
        """
        根据结论表达式选择合适的规则

        Args:
            conclusion_expr: 目标结论表达式
            goal_only: 是否仅用于构造初始目标

        Returns:
            Rule: 适合的规则实例
        """
        if goal_only or conclusion_expr is None:
            # 构造初始目标时，偏向选择能产生复杂结论的规则
            preferred_rules = [
                HypotheticalSyllogismRule(),  # 产生蕴含关系
                ConjunctionIntroductionRule(),  # 产生合取
                BiconditionalEliminationRule()  # 产生复杂结构
            ]
            return random.choice(preferred_rules + self.rules)

        # 根据结论类型选择规则
        if conclusion_expr:
            import z3

            if z3.is_and(conclusion_expr):
                # 如果目标是合取，选择能产生合取的规则
                return random.choice([
                    ConjunctionIntroductionRule(),
                    BiconditionalEliminationRule()
                ])
            elif z3.is_or(conclusion_expr):
                # 如果目标是析取，选择析取引入
                return DisjunctionIntroductionRule()
            elif z3.is_implies(conclusion_expr):
                # 如果目标是蕴含，选择假言三段论
                return random.choice([
                    HypotheticalSyllogismRule(),
                    ModusPonensRule()
                ])

        # 默认随机选择
        return self.sample_rule()

    def get_rule_chain_for_depth(self, target_depth):
        """
        为指定深度生成推理规则链

        Args:
            target_depth: 目标深度

        Returns:
            list: 按深度排序的规则链
        """
        if target_depth < 2:
            return [self.sample_rule()]

        rule_chain = []

        # 第一层：选择能产生中间结论的规则
        first_rule = random.choice([
            ConjunctionIntroductionRule(),
            DisjunctionIntroductionRule(),
            UniversalInstantiationRule()
        ])
        rule_chain.append(first_rule)

        # 中间层：选择传递性规则
        for _ in range(target_depth - 2):
            middle_rule = random.choice([
                ModusPonensRule(),
                HypotheticalSyllogismRule(),
                TransitivityRule(),
                ConjunctionEliminationRule()
            ])
            rule_chain.append(middle_rule)

        # 最后一层：选择最终推理规则
        final_rule = random.choice([
            ModusPonensRule(),
            HypotheticalSyllogismRule(),
            ConjunctionEliminationRule()
        ])
        rule_chain.append(final_rule)

        return rule_chain

    def get_compatible_rules(self, existing_premises):
        """
        获取与现有前提兼容的规则

        Args:
            existing_premises: 现有前提列表

        Returns:
            list: 兼容的规则列表
        """
        compatible_rules = []

        for rule in self.rules:
            try:
                if rule.can_apply(existing_premises):
                    compatible_rules.append(rule)
            except:
                # 如果规则检查失败，跳过
                continue

        return compatible_rules if compatible_rules else [self.sample_rule()]

    def explain_all_rules(self):
        """
        解释所有规则的功能

        Returns:
            dict: 规则说明字典
        """
        explanations = {}

        for rule in self.rules:
            template = rule.get_rule_template()
            explanations[rule.name] = {
                "description": rule.description,
                "pattern": template.get("pattern", ""),
                "formal": template.get("formal", ""),
                "example": template.get("example", ""),
                "category": self._get_rule_category(rule)
            }

        return explanations

    def _get_rule_category(self, rule):
        """获取规则所属的类别"""
        for category, rules in self.rule_categories.items():
            if any(isinstance(rule, type(r)) for r in rules):
                return category
        return "unknown"

    def get_statistics(self):
        """
        获取规则池统计信息

        Returns:
            dict: 统计信息
        """
        return {
            "total_rules": len(self.rules),
            "categories": {cat: len(rules) for cat, rules in self.rule_categories.items()},
            "strength_types": {strength: len(rules) for strength, rules in self.strength_categories.items()},
            "rule_names": [rule.name for rule in self.rules]
        }

    def validate_all_rules(self):
        """
        验证所有规则的基本功能

        Returns:
            dict: 验证结果
        """
        validation_results = {}

        for rule in self.rules:
            try:
                # 测试基本功能
                premises_count = rule.num_premises()

                # 创建测试前提
                import z3
                test_premises = [z3.Bool(f"test_{i}") for i in range(premises_count)]

                # 测试构造结论
                if rule.can_apply(test_premises):
                    conclusion = rule.construct_conclusion(test_premises)
                    explanation = rule.explain_step(test_premises, conclusion)

                    validation_results[rule.name] = {
                        "status": "✅ 通过",
                        "premises_count": premises_count,
                        "can_construct": True,
                        "can_explain": bool(explanation)
                    }
                else:
                    validation_results[rule.name] = {
                        "status": "⚠️ 部分通过",
                        "premises_count": premises_count,
                        "can_construct": False,
                        "note": "can_apply返回False"
                    }

            except Exception as e:
                validation_results[rule.name] = {
                    "status": "❌ 失败",
                    "error": str(e)
                }

        return validation_results


# 创建全局规则池实例（向后兼容）
rule_pool = RealLogicRulesPool()


# 测试函数
def test_rules_pool():
    """测试新的规则池"""
    print("=== 🧪 测试真实逻辑规则池 ===")

    # 基本统计
    stats = rule_pool.get_statistics()
    print(f"📊 规则池统计: {stats}")

    # 验证所有规则
    print("\n🔍 验证所有规则:")
    validation = rule_pool.validate_all_rules()
    for rule_name, result in validation.items():
        print(f"  {rule_name}: {result['status']}")

    # 测试规则采样
    print(f"\n🎲 随机采样测试:")
    for i in range(3):
        rule = rule_pool.sample_rule()
        print(f"  采样 {i + 1}: {rule.name} - {rule.description}")

    # 测试类别采样
    print(f"\n📂 类别采样测试:")
    for category in ["implication", "conjunction", "disjunction"]:
        rule = rule_pool.sample_rule(category=category)
        print(f"  {category}: {rule.name}")

    # 测试规则链生成
    print(f"\n🔗 规则链生成测试:")
    chain = rule_pool.get_rule_chain_for_depth(4)
    for i, rule in enumerate(chain, 1):
        print(f"  层 {i}: {rule.name}")

    print("\n✅ 规则池测试完成")


if __name__ == "__main__":
    test_rules_pool()