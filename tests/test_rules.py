import unittest
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rules.base.rule import LogicalFormula, LogicalOperator
from rules.rule_pooling import StratifiedRulePool
from rules.tiers.tier1_axioms import (
    ModusPonensRule,
    ModusTollensRule,
    HypotheticalSyllogismRule,
    DisjunctiveSyllogismRule,
    ConjunctionIntroductionRule,
    ConjunctionEliminationRule
)


class TestTier1Rules(unittest.TestCase):
    """测试Tier 1基础公理规则"""

    def setUp(self):
        """测试前准备"""
        self.modus_ponens = ModusPonensRule()
        self.modus_tollens = ModusTollensRule()
        self.hypothetical_syllogism = HypotheticalSyllogismRule()
        self.disjunctive_syllogism = DisjunctiveSyllogismRule()
        self.conjunction_intro = ConjunctionIntroductionRule()
        self.conjunction_elim = ConjunctionEliminationRule()

    def test_modus_ponens_can_apply(self):
        """测试肯定前件规则的适用性检查"""
        # 准备前提：P → Q 和 P
        premise1 = LogicalFormula(
            expression="P → Q",
            variables={"P", "Q"},
            operators=[LogicalOperator.IMPLIES],
            complexity=2
        )
        premise2 = LogicalFormula(
            expression="P",
            variables={"P"},
            operators=[],
            complexity=1
        )

        premises = [premise1, premise2]

        # 测试规则适用性
        self.assertTrue(self.modus_ponens.can_apply(premises))

        # 测试不适用的情况
        wrong_premise = LogicalFormula(
            expression="R",
            variables={"R"},
            operators=[],
            complexity=1
        )
        wrong_premises = [premise1, wrong_premise]
        self.assertFalse(self.modus_ponens.can_apply(wrong_premises))

    def test_modus_ponens_apply(self):
        """测试肯定前件规则的应用"""
        # 准备前提
        premise1 = LogicalFormula(
            expression="P → Q",
            variables={"P", "Q"},
            operators=[LogicalOperator.IMPLIES],
            complexity=2
        )
        premise2 = LogicalFormula(
            expression="P",
            variables={"P"},
            operators=[],
            complexity=1
        )

        premises = [premise1, premise2]

        # 应用规则
        conclusions = self.modus_ponens.apply(premises)

        # 验证结果
        self.assertEqual(len(conclusions), 1)
        self.assertEqual(conclusions[0].expression, "Q")
        self.assertIn("Q", conclusions[0].variables)

    def test_modus_tollens_can_apply(self):
        """测试否定后件规则的适用性检查"""
        # 准备前提：P → Q 和 ¬Q
        premise1 = LogicalFormula(
            expression="P → Q",
            variables={"P", "Q"},
            operators=[LogicalOperator.IMPLIES],
            complexity=2
        )
        premise2 = LogicalFormula(
            expression="¬Q",
            variables={"Q"},
            operators=[LogicalOperator.NOT],
            complexity=1
        )

        premises = [premise1, premise2]
        self.assertTrue(self.modus_tollens.can_apply(premises))

    def test_hypothetical_syllogism_can_apply(self):
        """测试假言三段论规则的适用性检查"""
        # 准备前提：P → Q 和 Q → R
        premise1 = LogicalFormula(
            expression="P → Q",
            variables={"P", "Q"},
            operators=[LogicalOperator.IMPLIES],
            complexity=2
        )
        premise2 = LogicalFormula(
            expression="Q → R",
            variables={"Q", "R"},
            operators=[LogicalOperator.IMPLIES],
            complexity=2
        )

        premises = [premise1, premise2]
        self.assertTrue(self.hypothetical_syllogism.can_apply(premises))

    def test_disjunctive_syllogism_can_apply(self):
        """测试析取三段论规则的适用性检查"""
        # 准备前提：P ∨ Q 和 ¬P
        premise1 = LogicalFormula(
            expression="P ∨ Q",
            variables={"P", "Q"},
            operators=[LogicalOperator.OR],
            complexity=2
        )
        premise2 = LogicalFormula(
            expression="¬P",
            variables={"P"},
            operators=[LogicalOperator.NOT],
            complexity=1
        )

        premises = [premise1, premise2]
        self.assertTrue(self.disjunctive_syllogism.can_apply(premises))

    def test_conjunction_introduction_can_apply(self):
        """测试合取引入规则的适用性检查"""
        # 准备前提：P 和 Q
        premise1 = LogicalFormula(
            expression="P",
            variables={"P"},
            operators=[],
            complexity=1
        )
        premise2 = LogicalFormula(
            expression="Q",
            variables={"Q"},
            operators=[],
            complexity=1
        )

        premises = [premise1, premise2]
        self.assertTrue(self.conjunction_intro.can_apply(premises))

    def test_conjunction_elimination_can_apply(self):
        """测试合取消除规则的适用性检查"""
        # 准备前提：P ∧ Q
        premise = LogicalFormula(
            expression="P ∧ Q",
            variables={"P", "Q"},
            operators=[LogicalOperator.AND],
            complexity=2
        )

        premises = [premise]
        self.assertTrue(self.conjunction_elim.can_apply(premises))

    def test_rule_templates(self):
        """测试规则模板获取"""
        rules = [
            self.modus_ponens,
            self.modus_tollens,
            self.hypothetical_syllogism,
            self.disjunctive_syllogism,
            self.conjunction_intro,
            self.conjunction_elim
        ]

        for rule in rules:
            template = rule.get_template()
            self.assertIn('rule_name', template)
            self.assertIn('patterns', template)
            self.assertIsInstance(template['patterns'], list)

    def test_rule_statistics_update(self):
        """测试规则统计更新"""
        rule = self.modus_ponens

        # 初始状态
        self.assertEqual(rule.usage_count, 0)
        self.assertEqual(rule.success_rate, 0.0)

        # 更新成功案例
        rule.update_statistics(True)
        self.assertEqual(rule.usage_count, 1)
        self.assertEqual(rule.success_rate, 1.0)

        # 更新失败案例
        rule.update_statistics(False)
        self.assertEqual(rule.usage_count, 2)
        self.assertEqual(rule.success_rate, 0.5)


class TestStratifiedRulePool(unittest.TestCase):
    """测试分层规则池"""

    def setUp(self):
        """测试前准备"""
        self.rule_pool = StratifiedRulePool()
        self.modus_ponens = ModusPonensRule()
        self.modus_tollens = ModusTollensRule()

    def test_register_rule(self):
        """测试规则注册"""
        # 注册规则
        success = self.rule_pool.register_rule(self.modus_ponens)
        self.assertTrue(success)

        # 验证规则已注册
        retrieved_rule = self.rule_pool.get_rule_by_id("modus_ponens")
        self.assertIsNotNone(retrieved_rule)
        self.assertEqual(retrieved_rule.rule_id, "modus_ponens")

        # 重复注册应该失败
        duplicate_success = self.rule_pool.register_rule(self.modus_ponens)
        self.assertFalse(duplicate_success)

    def test_get_rules_by_tier(self):
        """测试按层级获取规则"""
        # 注册规则
        self.rule_pool.register_rule(self.modus_ponens)
        self.rule_pool.register_rule(self.modus_tollens)

        # 获取Tier 1规则
        tier1_rules = self.rule_pool.get_rules_by_tier(1)
        self.assertEqual(len(tier1_rules), 2)

        rule_ids = [rule.rule_id for rule in tier1_rules]
        self.assertIn("modus_ponens", rule_ids)
        self.assertIn("modus_tollens", rule_ids)

        # 获取不存在的层级
        tier5_rules = self.rule_pool.get_rules_by_tier(5)
        self.assertEqual(len(tier5_rules), 0)

    def test_get_applicable_rules(self):
        """测试获取可应用规则"""
        # 注册规则
        self.rule_pool.register_rule(self.modus_ponens)
        self.rule_pool.register_rule(self.modus_tollens)

        # 准备前提
        premise1 = LogicalFormula(
            expression="P → Q",
            variables={"P", "Q"},
            operators=[LogicalOperator.IMPLIES],
            complexity=2
        )
        premise2 = LogicalFormula(
            expression="P",
            variables={"P"},
            operators=[],
            complexity=1
        )

        premises = [premise1, premise2]

        # 获取可应用规则
        applicable_rules = self.rule_pool.get_applicable_rules(premises)

        # 至少应该包含modus_ponens
        rule_ids = [rule.rule_id for rule in applicable_rules]
        self.assertIn("modus_ponens", rule_ids)

    def test_selection_strategies(self):
        """测试规则选择策略"""
        # 注册规则
        self.rule_pool.register_rule(self.modus_ponens)
        self.rule_pool.register_rule(self.modus_tollens)

        rules = [self.modus_ponens, self.modus_tollens]

        # 测试随机选择
        self.rule_pool.set_selection_strategy("random")
        selected = self.rule_pool.select_rule(rules)
        self.assertIn(selected, rules)

        # 测试复杂度驱动选择
        self.rule_pool.set_selection_strategy("complexity_driven")
        selected = self.rule_pool.select_rule(rules, target_complexity=15)
        self.assertIn(selected, rules)

        # 测试平衡选择
        self.rule_pool.set_selection_strategy("balanced")
        selected = self.rule_pool.select_rule(rules)
        self.assertIn(selected, rules)

    def test_statistics(self):
        """测试统计信息"""
        # 注册规则
        self.rule_pool.register_rule(self.modus_ponens)
        self.rule_pool.register_rule(self.modus_tollens)

        # 获取统计信息
        stats = self.rule_pool.get_statistics()

        self.assertEqual(stats["total_rules"], 2)
        self.assertIn(1, stats["rules_by_tier"])
        self.assertEqual(stats["rules_by_tier"][1], 2)


if __name__ == '__main__':
    unittest.main()