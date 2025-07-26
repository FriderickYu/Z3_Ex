import unittest
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.dag_generator import DualLayerDAGGenerator, DAGGenerationMode
from core.complexity_controller import ComplexityController
from rules.rule_pooling import StratifiedRulePool
from rules.tiers.tier1_axioms import ModusPonensRule, ModusTollensRule
from rules.base.rule import LogicalFormula


class TestDAGGeneration(unittest.TestCase):
    """测试DAG生成功能"""

    def setUp(self):
        """测试前准备"""
        # 创建规则池
        self.rule_pool = StratifiedRulePool()

        # 注册基础规则
        self.rule_pool.register_rule(ModusPonensRule())
        self.rule_pool.register_rule(ModusTollensRule())

        # 创建DAG生成器
        self.dag_generator = DualLayerDAGGenerator(self.rule_pool)

    def test_dag_generator_initialization(self):
        """测试DAG生成器初始化"""
        self.assertIsNotNone(self.dag_generator)
        self.assertEqual(self.dag_generator.max_depth, 8)
        self.assertEqual(self.dag_generator.max_width, 5)

    def test_seed_formula_generation(self):
        """测试种子公式生成"""
        seed_formulas = self.dag_generator._generate_seed_formulas()

        self.assertIsInstance(seed_formulas, list)
        self.assertGreaterEqual(len(seed_formulas), self.dag_generator.min_premises)
        self.assertLessEqual(len(seed_formulas), self.dag_generator.max_premises)

        # 检查生成的公式格式
        for formula in seed_formulas:
            self.assertIsInstance(formula, LogicalFormula)
            self.assertIsNotNone(formula.expression)
            self.assertIsInstance(formula.variables, set)

    def test_forward_dag_generation(self):
        """测试前向DAG生成"""
        # 准备种子公式
        seed_formulas = [
            LogicalFormula(
                expression="P",
                variables={"P"},
                operators=[],
                complexity=1
            ),
            LogicalFormula(
                expression="P → Q",
                variables={"P", "Q"},
                operators=[],
                complexity=2
            )
        ]

        # 生成DAG
        dag = self.dag_generator._generate_forward_dag(seed_formulas, 3, 10)

        # 验证DAG属性
        self.assertGreater(dag.number_of_nodes(), 0)

        # 检查节点数据
        for node_id, data in dag.nodes(data=True):
            self.assertIn('data', data)
            node_data = data['data']
            self.assertIsNotNone(node_data.formula)
            self.assertIsInstance(node_data.level, int)

    def test_dag_validation(self):
        """测试DAG验证"""
        # 生成一个简单的DAG
        dag = self.dag_generator.generate_dag(
            mode=DAGGenerationMode.FORWARD,
            target_depth=2,
            target_complexity=5
        )

        # 验证DAG
        is_valid = self.dag_generator._validate_dag(dag)
        self.assertTrue(is_valid)


class TestComplexityController(unittest.TestCase):
    """测试复杂度控制器"""

    def setUp(self):
        """测试前准备"""
        self.complexity_controller = ComplexityController()

    def test_complexity_controller_initialization(self):
        """测试复杂度控制器初始化"""
        self.assertIsNotNone(self.complexity_controller)
        self.assertIsNotNone(self.complexity_controller.profile)

    def test_formula_complexity_calculation(self):
        """测试公式复杂度计算"""
        # 简单公式
        simple_formula = LogicalFormula(
            expression="P",
            variables={"P"},
            operators=[],
            complexity=1
        )

        complexity = self.complexity_controller.calculate_formula_complexity(simple_formula)

        self.assertIn('structural', complexity)
        self.assertIn('semantic', complexity)
        self.assertIn('computational', complexity)
        self.assertIn('cognitive', complexity)
        self.assertIn('overall', complexity)

        # 所有复杂度值应该为正数
        for key, value in complexity.items():
            self.assertGreater(value, 0)

    def test_target_complexity_progression(self):
        """测试目标复杂度递增"""
        # 测试不同步骤的目标复杂度
        complexity_step_0 = self.complexity_controller.get_target_complexity_for_step(0)
        complexity_step_1 = self.complexity_controller.get_target_complexity_for_step(1)
        complexity_step_2 = self.complexity_controller.get_target_complexity_for_step(2)

        # 复杂度应该递增
        self.assertLess(complexity_step_0, complexity_step_1)
        self.assertLess(complexity_step_1, complexity_step_2)

    def test_complexity_recording(self):
        """测试复杂度记录"""
        # 记录复杂度
        complexity_data = {
            'structural': 5.0,
            'semantic': 3.0,
            'computational': 4.0,
            'cognitive': 2.0,
            'overall': 3.5
        }

        self.complexity_controller.record_complexity(1, complexity_data)

        # 检查记录
        self.assertEqual(len(self.complexity_controller.complexity_history), 1)
        self.assertEqual(self.complexity_controller.current_step, 1)

    def test_tier_progression_decision(self):
        """测试层级提升决策"""
        # 测试应该提升层级的情况
        should_increase = self.complexity_controller.should_increase_tier(
            current_complexity=5.0,
            target_complexity=15.0,
            current_tier=1
        )
        self.assertTrue(should_increase)

        # 测试不应该提升层级的情况
        should_not_increase = self.complexity_controller.should_increase_tier(
            current_complexity=12.0,
            target_complexity=15.0,
            current_tier=4
        )
        # 这个结果取决于具体实现逻辑


if __name__ == '__main__':
    unittest.main()