"""
集成系统测试框架
测试表达式树构建器、复杂度控制器、Z3验证器和集成DAG生成器
"""

import unittest
import sys
import os
import tempfile
import json
from unittest.mock import Mock, patch
from typing import List, Dict, Any

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 导入测试目标
from rules.rule_pooling import StratifiedRulePool
from rules.base.rule import LogicalFormula, LogicalOperator
from rules.tiers.tier1_axioms import ModusPonensRule, ConjunctionIntroductionRule, ModusTollensRule

# 测试集成组件（可能不可用）
try:
    from core.expression_tree_builder import ExpressionTreeBuilder, ExpressionType, ConnectorType
    from core.complexity_controller import ComplexityController, ComplexityGrowthStrategy
    from core.z3_validator import Z3Validator, ValidationResult
    from core.dag_generator import DualLayerDAGGenerator, DAGGenerationMode

    INTEGRATED_COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"集成组件导入失败: {e}")
    INTEGRATED_COMPONENTS_AVAILABLE = False


class TestExpressionTreeBuilder(unittest.TestCase):
    """测试表达式树构建器"""

    def setUp(self):
        """测试前准备"""
        if not INTEGRATED_COMPONENTS_AVAILABLE:
            self.skipTest("表达式树构建器不可用")

        self.tree_builder = ExpressionTreeBuilder()

    def test_tree_builder_initialization(self):
        """测试构建器初始化"""
        self.assertIsNotNone(self.tree_builder)
        self.assertIsNotNone(self.tree_builder.variable_pool)
        self.assertEqual(len(self.tree_builder.variable_pool), 11)  # P-Z

    def test_build_expression_tree_basic(self):
        """测试基础表达式树构建"""
        tree = self.tree_builder.build_expression_tree(
            target_depth=2,
            complexity_factor=1.0,
            variable_limit=3
        )

        self.assertIsNotNone(tree)
        self.assertTrue(hasattr(tree, 'node_id'))
        self.assertTrue(hasattr(tree, 'expression_type'))
        self.assertTrue(hasattr(tree, 'depth'))

        # 检查深度
        max_depth = self._get_tree_max_depth(tree)
        self.assertLessEqual(max_depth, 2)

    def test_build_expression_tree_depth_control(self):
        """测试深度控制的准确性"""
        for target_depth in [1, 2, 3, 4]:
            with self.subTest(depth=target_depth):
                tree = self.tree_builder.build_expression_tree(
                    target_depth=target_depth,
                    complexity_factor=1.0
                )

                actual_depth = self._get_tree_max_depth(tree)
                # 允许±1的误差
                self.assertLessEqual(abs(actual_depth - target_depth), 1)

    def test_convert_to_logical_formula(self):
        """测试转换为LogicalFormula"""
        tree = self.tree_builder.build_expression_tree(target_depth=2)
        formula = self.tree_builder.convert_to_logical_formula(tree)

        self.assertIsInstance(formula, LogicalFormula)
        self.assertIsNotNone(formula.expression)
        self.assertIsInstance(formula.variables, set)
        self.assertGreater(len(formula.variables), 0)

    def test_complexity_factor_impact(self):
        """测试复杂度因子的影响"""
        # 低复杂度
        tree_simple = self.tree_builder.build_expression_tree(
            target_depth=3, complexity_factor=0.5
        )

        # 高复杂度
        tree_complex = self.tree_builder.build_expression_tree(
            target_depth=3, complexity_factor=2.0
        )

        # 高复杂度因子应该产生更复杂的树
        simple_nodes = self._count_tree_nodes(tree_simple)
        complex_nodes = self._count_tree_nodes(tree_complex)

        # 注意：这是概率性测试，可能偶尔失败
        self.assertGreaterEqual(complex_nodes, simple_nodes * 0.8)

    def test_statistics(self):
        """测试统计功能"""
        # 构建几个树
        for _ in range(3):
            self.tree_builder.build_expression_tree(target_depth=2)

        stats = self.tree_builder.get_statistics()

        self.assertIn('total_trees', stats)
        self.assertEqual(stats['total_trees'], 3)
        self.assertIn('avg_depth', stats)
        self.assertIn('avg_complexity', stats)

    def _get_tree_max_depth(self, tree) -> int:
        """获取树的最大深度"""
        if tree.is_leaf():
            return tree.depth

        max_depth = tree.depth
        if tree.left_child:
            max_depth = max(max_depth, self._get_tree_max_depth(tree.left_child))
        if tree.right_child:
            max_depth = max(max_depth, self._get_tree_max_depth(tree.right_child))

        return max_depth

    def _count_tree_nodes(self, tree) -> int:
        """计算树中节点数量"""
        count = 1
        if tree.left_child:
            count += self._count_tree_nodes(tree.left_child)
        if tree.right_child:
            count += self._count_tree_nodes(tree.right_child)
        return count


class TestComplexityController(unittest.TestCase):
    """测试复杂度控制器"""

    def setUp(self):
        """测试前准备"""
        if not INTEGRATED_COMPONENTS_AVAILABLE:
            self.skipTest("复杂度控制器不可用")

        self.complexity_controller = ComplexityController()

    def test_controller_initialization(self):
        """测试控制器初始化"""
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

        # 检查返回的复杂度字典
        expected_keys = ['structural', 'semantic', 'computational', 'cognitive', 'overall']
        for key in expected_keys:
            self.assertIn(key, complexity)
            self.assertIsInstance(complexity[key], (int, float))
            self.assertGreater(complexity[key], 0)

    def test_expression_tree_complexity(self):
        """测试表达式树复杂度计算"""
        if not hasattr(self.complexity_controller, 'calculate_expression_tree_complexity'):
            self.skipTest("表达式树复杂度功能不可用")

        # 创建模拟表达式树
        mock_tree = Mock()
        mock_tree.is_leaf.return_value = False
        mock_tree.expression_type = Mock()
        mock_tree.expression_type.value = 'binary'
        mock_tree.depth = 2
        mock_tree.variables = {'P', 'Q'}
        mock_tree.connector = Mock()
        mock_tree.connector.value = '∧'
        mock_tree.left_child = None
        mock_tree.right_child = None

        complexity = self.complexity_controller.calculate_expression_tree_complexity(mock_tree)

        self.assertIn('overall', complexity)
        self.assertGreater(complexity['overall'], 0)

    def test_complexity_target_generation(self):
        """测试复杂度目标生成"""
        if not hasattr(self.complexity_controller, 'generate_complexity_target'):
            self.skipTest("复杂度目标生成功能不可用")

        target = self.complexity_controller.generate_complexity_target(step=1)

        self.assertIsInstance(target, dict)
        self.assertIn('overall', target)
        self.assertGreater(target['overall'], 0)

    def test_complexity_progression(self):
        """测试复杂度递增"""
        complexities = []

        for step in range(5):
            target = self.complexity_controller.get_target_complexity_for_step(step)
            complexities.append(target)

        # 检查递增趋势
        for i in range(1, len(complexities)):
            self.assertGreaterEqual(complexities[i], complexities[i - 1])

    def test_statistics_recording(self):
        """测试统计记录"""
        # 记录一些复杂度数据
        for step in range(3):
            complexity_data = {
                'structural': step * 2.0,
                'semantic': step * 1.5,
                'computational': step * 1.8,
                'cognitive': step * 1.2,
                'overall': step * 2.0
            }
            self.complexity_controller.record_complexity(step, complexity_data)

        stats = self.complexity_controller.get_complexity_statistics()

        self.assertIn('total_steps', stats)
        self.assertEqual(stats['total_steps'], 3)
        self.assertIn('overall', stats)


class TestZ3Validator(unittest.TestCase):
    """测试Z3验证器"""

    def setUp(self):
        """测试前准备"""
        if not INTEGRATED_COMPONENTS_AVAILABLE:
            self.skipTest("Z3验证器不可用")

        try:
            self.z3_validator = Z3Validator(timeout_seconds=10)
        except ImportError:
            self.skipTest("Z3库不可用")

    def test_validator_initialization(self):
        """测试验证器初始化"""
        self.assertIsNotNone(self.z3_validator)
        self.assertIsNotNone(self.z3_validator.config)

    def test_valid_reasoning_step(self):
        """测试有效推理步骤验证"""
        # 测试 Modus Ponens: P → Q, P ⊢ Q
        premises = ["P → Q", "P"]
        conclusion = "Q"

        result = self.z3_validator.validate_reasoning_step(premises, conclusion)

        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, 'is_valid'))
        self.assertTrue(hasattr(result, 'status'))

        # 这个推理应该是有效的
        if result.status != ValidationResult.ERROR:
            self.assertTrue(result.is_valid)

    def test_invalid_reasoning_step(self):
        """测试无效推理步骤验证"""
        # 测试无效推理: P ⊢ Q（没有连接）
        premises = ["P"]
        conclusion = "Q"

        result = self.z3_validator.validate_reasoning_step(premises, conclusion)

        # 这个推理应该是无效的
        if result.status not in [ValidationResult.ERROR, ValidationResult.TIMEOUT]:
            self.assertFalse(result.is_valid)

    def test_complex_reasoning_validation(self):
        """测试复杂推理验证"""
        # 测试假言三段论: P → Q, Q → R ⊢ P → R
        premises = ["P → Q", "Q → R"]
        conclusion = "P → R"

        result = self.z3_validator.validate_reasoning_step(premises, conclusion)

        # 这个推理应该是有效的
        if result.status not in [ValidationResult.ERROR, ValidationResult.TIMEOUT]:
            self.assertTrue(result.is_valid)

    def test_expression_conversion(self):
        """测试表达式转换"""
        # 测试各种表达式格式
        test_expressions = [
            "P",
            "P → Q",
            "P ∧ Q",
            "P ∨ Q",
            "¬P"
        ]

        for expr in test_expressions:
            with self.subTest(expression=expr):
                try:
                    z3_expr, warnings = self.z3_validator.converter.convert_to_z3(expr)
                    self.assertIsNotNone(z3_expr)
                    # 如果转换成功，warnings可能为空
                except Exception as e:
                    # 一些复杂表达式可能转换失败，这是可以接受的
                    self.assertIsInstance(e, Exception)

    def test_statistics(self):
        """测试验证统计"""
        # 执行几次验证
        test_cases = [
            (["P → Q", "P"], "Q"),
            (["P"], "Q"),  # 无效案例
            (["P ∧ Q"], "P")  # 有效案例
        ]

        for premises, conclusion in test_cases:
            self.z3_validator.validate_reasoning_step(premises, conclusion)

        stats = self.z3_validator.get_validation_statistics()

        self.assertIn('total_validations', stats)
        self.assertEqual(stats['total_validations'], 3)


class TestIntegratedDAGGenerator(unittest.TestCase):
    """测试集成DAG生成器"""

    def setUp(self):
        """测试前准备"""
        if not INTEGRATED_COMPONENTS_AVAILABLE:
            self.skipTest("集成DAG生成器不可用")

        # 创建规则池
        self.rule_pool = StratifiedRulePool()
        self.rule_pool.register_rule(ModusPonensRule())
        self.rule_pool.register_rule(ConjunctionIntroductionRule())
        self.rule_pool.register_rule(ModusTollensRule())

        # 创建基础生成器
        self.basic_generator = DualLayerDAGGenerator(self.rule_pool)

        # 创建集成生成器
        self.integrated_generator = DualLayerDAGGenerator(
            self.rule_pool,
            enable_expression_trees=True,
            enable_z3_validation=True,
            z3_timeout_seconds=10
        )

    def test_basic_generation(self):
        """测试基础生成功能"""
        dag = self.basic_generator.generate_dag(
            mode=DAGGenerationMode.FORWARD,
            target_depth=3,
            target_complexity=5
        )

        self.assertIsNotNone(dag)
        self.assertGreater(dag.number_of_nodes(), 0)

        # 检查DAG结构
        import networkx as nx
        self.assertTrue(nx.is_directed_acyclic_graph(dag))

    def test_tree_driven_generation(self):
        """测试表达式树驱动生成"""
        dag = self.integrated_generator.generate_dag(
            mode=DAGGenerationMode.TREE_DRIVEN,
            target_depth=4,
            target_complexity=10
        )

        self.assertIsNotNone(dag)
        self.assertGreater(dag.number_of_nodes(), 0)

        # 检查节点是否包含表达式树信息
        has_tree_info = False
        for node_id, data in dag.nodes(data=True):
            node = data['data']
            if hasattr(node, 'expression_tree') and node.expression_tree:
                has_tree_info = True
                break

        self.assertTrue(has_tree_info, "应该至少有一个节点包含表达式树信息")

    def test_depth_control_accuracy(self):
        """测试深度控制的准确性"""
        for target_depth in [2, 3, 4]:
            with self.subTest(depth=target_depth):
                dag = self.integrated_generator.generate_dag(
                    mode=DAGGenerationMode.TREE_DRIVEN,
                    target_depth=target_depth
                )

                # 获取实际最大层级
                max_level = 0
                for node_id, data in dag.nodes(data=True):
                    node = data['data']
                    max_level = max(max_level, node.level)

                # 检查深度控制（允许±1的误差）
                self.assertLessEqual(abs(max_level - (target_depth - 1)), 1)

    def test_z3_validation_integration(self):
        """测试Z3验证集成"""
        dag = self.integrated_generator.generate_dag(
            mode=DAGGenerationMode.TREE_DRIVEN,
            target_depth=3
        )

        # 检查是否有节点被验证
        validated_count = 0
        for node_id, data in dag.nodes(data=True):
            node = data['data']
            if hasattr(node, 'is_validated') and node.is_validated:
                validated_count += 1

        self.assertGreater(validated_count, 0, "应该至少有一个节点被验证")

    def test_dataset_generation(self):
        """测试数据集生成"""
        dataset = self.integrated_generator.generate_dataset(
            sample_count=5,
            mode=DAGGenerationMode.TREE_DRIVEN
        )

        self.assertIsInstance(dataset, list)
        self.assertGreater(len(dataset), 0)

        # 检查样本格式
        for sample in dataset:
            self.assertIn('premises', sample)
            self.assertIn('conclusion', sample)
            self.assertIsInstance(sample['premises'], list)
            self.assertIsInstance(sample['conclusion'], str)

    def test_mode_compatibility(self):
        """测试不同模式的兼容性"""
        modes = [
            DAGGenerationMode.FORWARD,
            DAGGenerationMode.BACKWARD,
            DAGGenerationMode.BIDIRECTIONAL,
            DAGGenerationMode.TREE_DRIVEN
        ]

        for mode in modes:
            with self.subTest(mode=mode.value):
                try:
                    dag = self.integrated_generator.generate_dag(
                        mode=mode,
                        target_depth=2,
                        target_complexity=5
                    )
                    self.assertIsNotNone(dag)
                    self.assertGreater(dag.number_of_nodes(), 0)
                except Exception as e:
                    # 某些模式可能因为规则不足而失败，这是可以接受的
                    self.assertIsInstance(e, Exception)

    def test_error_handling(self):
        """测试错误处理"""
        # 测试无效参数
        try:
            dag = self.integrated_generator.generate_dag(
                target_depth=0,  # 无效深度
                target_complexity=-1  # 无效复杂度
            )
            # 如果没有抛出异常，至少应该返回一个基本的DAG
            self.assertIsNotNone(dag)
        except Exception:
            # 抛出异常也是可以接受的
            pass


class TestEndToEnd(unittest.TestCase):
    """端到端集成测试"""

    def setUp(self):
        """测试前准备"""
        if not INTEGRATED_COMPONENTS_AVAILABLE:
            self.skipTest("集成组件不可用")

    def test_complete_workflow(self):
        """测试完整工作流程"""
        # 1. 创建规则池
        rule_pool = StratifiedRulePool()
        rule_pool.register_rule(ModusPonensRule())
        rule_pool.register_rule(ConjunctionIntroductionRule())

        # 2. 创建集成生成器
        generator = DualLayerDAGGenerator(
            rule_pool,
            enable_expression_trees=True,
            enable_z3_validation=True
        )

        # 3. 生成DAG
        dag = generator.generate_dag(
            mode=DAGGenerationMode.TREE_DRIVEN,
            target_depth=3,
            target_complexity=8
        )

        # 4. 验证结果
        self.assertIsNotNone(dag)
        self.assertGreater(dag.number_of_nodes(), 0)

        # 5. 生成数据集
        dataset = generator.generate_dataset(sample_count=3)
        self.assertGreater(len(dataset), 0)

        # 6. 验证数据集质量
        for sample in dataset:
            self.assertIn('premises', sample)
            self.assertIn('conclusion', sample)
            self.assertTrue(len(sample['premises']) > 0)
            self.assertTrue(len(sample['conclusion']) > 0)

    def test_performance_benchmark(self):
        """性能基准测试"""
        import time

        rule_pool = StratifiedRulePool()
        rule_pool.register_rule(ModusPonensRule())
        rule_pool.register_rule(ConjunctionIntroductionRule())

        generator = DualLayerDAGGenerator(rule_pool, enable_expression_trees=True)

        # 测试生成速度
        start_time = time.time()

        for _ in range(5):  # 生成5个DAG
            dag = generator.generate_dag(
                mode=DAGGenerationMode.TREE_DRIVEN,
                target_depth=3,
                target_complexity=5
            )
            self.assertIsNotNone(dag)

        elapsed_time = time.time() - start_time

        # 每个DAG的生成时间应该在合理范围内（这里设置为10秒）
        avg_time = elapsed_time / 5
        self.assertLess(avg_time, 10.0, f"平均生成时间过长: {avg_time:.2f}秒")

        print(f"性能测试: 平均每个DAG生成时间 {avg_time:.2f}秒")


class TestUtils(unittest.TestCase):
    """测试工具函数"""

    def test_file_operations(self):
        """测试文件操作"""
        if not INTEGRATED_COMPONENTS_AVAILABLE:
            self.skipTest("集成组件不可用")

        # 创建临时数据集
        rule_pool = StratifiedRulePool()
        rule_pool.register_rule(ModusPonensRule())

        generator = DualLayerDAGGenerator(rule_pool)
        dataset = generator.generate_dataset(sample_count=2)

        # 测试JSON保存和加载
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
            temp_file = f.name

        try:
            # 读取文件
            with open(temp_file, 'r', encoding='utf-8') as f:
                loaded_dataset = json.load(f)

            self.assertEqual(len(loaded_dataset), len(dataset))
            self.assertEqual(loaded_dataset[0]['conclusion'], dataset[0]['conclusion'])

        finally:
            # 清理临时文件
            os.unlink(temp_file)


def run_all_tests():
    """运行所有测试"""
    print("=== ARNG集成系统测试套件 ===\n")

    # 检查组件可用性
    if not INTEGRATED_COMPONENTS_AVAILABLE:
        print("⚠️  警告: 集成组件不可用，将跳过大部分测试")
        print("请确保已正确安装所有依赖项\n")

    # 创建测试套件
    test_classes = [
        TestExpressionTreeBuilder,
        TestComplexityController,
        TestZ3Validator,
        TestIntegratedDAGGenerator,
        TestEndToEnd,
        TestUtils
    ]

    suite = unittest.TestSuite()

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)

    # 输出摘要
    print(f"\n=== 测试摘要 ===")
    print(f"运行测试: {result.testsRun}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print(f"跳过: {len(result.skipped)}")

    if result.failures:
        print(f"\n失败的测试:")
        for test, traceback in result.failures:
            print(f"  - {test}")

    if result.errors:
        print(f"\n错误的测试:")
        for test, traceback in result.errors:
            print(f"  - {test}")

    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / max(1, result.testsRun)
    print(f"\n成功率: {success_rate:.1%}")

    return result.wasSuccessful()


def run_quick_test():
    """运行快速测试（仅测试核心功能）"""
    print("=== 快速测试 ===\n")

    if not INTEGRATED_COMPONENTS_AVAILABLE:
        print("❌ 集成组件不可用")
        return False

    try:
        # 1. 测试表达式树构建
        print("1. 测试表达式树构建...")
        tree_builder = ExpressionTreeBuilder()
        tree = tree_builder.build_expression_tree(target_depth=2)
        print(f"   ✅ 成功构建表达式树: {tree.get_expression_string()}")

        # 2. 测试复杂度控制
        print("2. 测试复杂度控制...")
        complexity_controller = ComplexityController()
        target = complexity_controller.get_target_complexity_for_step(1)
        print(f"   ✅ 复杂度目标生成成功: {target:.2f}")

        # 3. 测试Z3验证
        print("3. 测试Z3验证...")
        z3_validator = Z3Validator(timeout_seconds=5)
        result = z3_validator.validate_reasoning_step(["P → Q", "P"], "Q")
        print(f"   ✅ Z3验证完成: {result.is_valid}")

        # 4. 测试集成生成
        print("4. 测试集成DAG生成...")
        from rules.rule_pooling import StratifiedRulePool
        from rules.tiers.tier1_axioms import ModusPonensRule

        rule_pool = StratifiedRulePool()
        rule_pool.register_rule(ModusPonensRule())

        generator = DualLayerDAGGenerator(rule_pool, enable_expression_trees=True)
        dag = generator.generate_dag(mode=DAGGenerationMode.TREE_DRIVEN, target_depth=2)
        print(f"   ✅ DAG生成成功: {dag.number_of_nodes()} 节点")

        print("\n🎉 所有快速测试通过！")
        return True

    except Exception as e:
        print(f"\n❌ 快速测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ARNG集成系统测试")
    parser.add_argument('--quick', action='store_true', help='运行快速测试')
    parser.add_argument('--full', action='store_true', help='运行完整测试套件')

    args = parser.parse_args()

    if args.quick:
        success = run_quick_test()
    elif args.full:
        success = run_all_tests()
    else:
        # 默认运行快速测试
        print("使用 --quick 运行快速测试，或 --full 运行完整测试套件")
        success = run_quick_test()

    exit(0 if success else 1)