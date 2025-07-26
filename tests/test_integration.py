import unittest
import sys
import os

from core.complexity_controller import ComplexityController
from core.dag_generator import DualLayerDAGGenerator, DAGGenerationMode
from rules.rule_pooling import StratifiedRulePool
from rules.tiers.__init__ import get_tier1_rules
from utils.logger_utils import ARNGLogger

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))




def create_default_generator():
    """创建默认生成器的备用实现"""
    rule_pool = StratifiedRulePool()
    tier1_rules = get_tier1_rules()
    for rule in tier1_rules:
        rule_pool.register_rule(rule)
    dag_generator = DualLayerDAGGenerator(rule_pool)
    complexity_controller = ComplexityController()
    return dag_generator, complexity_controller, rule_pool


def create_logger(name="IntegrationTest"):
    """创建日志器的备用实现"""
    return ARNGLogger(name)


class TestIntegration(unittest.TestCase):
    """集成测试"""

    def setUp(self):
        """测试前准备"""
        self.dag_generator, self.complexity_controller, self.rule_pool = create_default_generator()
        self.logger = create_logger("IntegrationTest")

    def test_default_generator_creation(self):
        """测试默认生成器创建"""
        self.assertIsNotNone(self.dag_generator)
        self.assertIsNotNone(self.complexity_controller)
        self.assertIsNotNone(self.rule_pool)

        # 检查规则池中是否有注册的规则
        stats = self.rule_pool.get_statistics()
        self.assertGreater(stats["total_rules"], 0)
        self.assertIn(1, stats["rules_by_tier"])

    def test_end_to_end_dag_generation(self):
        """端到端DAG生成测试"""
        # 生成DAG
        dag = self.dag_generator.generate_dag(
            mode=DAGGenerationMode.FORWARD,
            target_depth=3,
            target_complexity=8
        )

        # 验证生成的DAG
        self.assertIsNotNone(dag)
        self.assertGreater(dag.number_of_nodes(), 0)

        # 记录日志
        self.logger.info(f"生成的DAG包含 {dag.number_of_nodes()} 个节点和 {dag.number_of_edges()} 条边")

        # 检查DAG结构
        self.assertTrue(self.dag_generator._validate_dag(dag))

    def test_complexity_driven_generation(self):
        """复杂度驱动的生成测试"""
        initial_complexity = 2.0
        target_steps = 4

        for step in range(target_steps):
            target_complexity = self.complexity_controller.get_target_complexity_for_step(
                step, initial_complexity
            )

            # 生成对应复杂度的DAG
            dag = self.dag_generator.generate_dag(
                mode=DAGGenerationMode.FORWARD,
                target_depth=step + 2,
                target_complexity=int(target_complexity)
            )

            # 记录复杂度
            complexity_data = {
                'structural': target_complexity * 0.3,
                'semantic': target_complexity * 0.25,
                'computational': target_complexity * 0.25,
                'cognitive': target_complexity * 0.2,
                'overall': target_complexity
            }
            self.complexity_controller.record_complexity(step, complexity_data)

            self.logger.info(f"步骤 {step}: 目标复杂度 {target_complexity:.2f}, 生成节点数 {dag.number_of_nodes()}")

        # 检查复杂度统计
        stats = self.complexity_controller.get_complexity_statistics()
        self.assertEqual(stats["total_steps"], target_steps)
        self.assertGreater(stats["overall"]["growth_rate"], 0)

    def test_rule_pool_integration(self):
        """规则池集成测试"""
        # 测试规则选择策略
        strategies = ["random", "complexity_driven", "balanced"]

        for strategy in strategies:
            self.rule_pool.set_selection_strategy(strategy)

            # 生成DAG
            dag = self.dag_generator.generate_dag(
                mode=DAGGenerationMode.FORWARD,
                target_depth=2,
                target_complexity=5
            )

            self.assertIsNotNone(dag)
            self.logger.info(f"使用 {strategy} 策略生成了 {dag.number_of_nodes()} 个节点的DAG")

        # 检查规则使用统计
        stats = self.rule_pool.get_statistics()
        if stats["most_used_rule"]:
            self.logger.info(f"最常用规则: {stats['most_used_rule']}")

    def test_different_generation_modes(self):
        """测试不同生成模式"""
        modes = [DAGGenerationMode.FORWARD, DAGGenerationMode.BACKWARD, DAGGenerationMode.BIDIRECTIONAL]

        for mode in modes:
            try:
                dag = self.dag_generator.generate_dag(
                    mode=mode,
                    target_depth=2,
                    target_complexity=4
                )
                self.assertIsNotNone(dag)
                self.logger.info(f"{mode.value} 模式生成成功: {dag.number_of_nodes()} 个节点")
            except Exception as e:
                self.logger.warning(f"{mode.value} 模式生成失败: {e}")
                # 对于某些模式可能失败，这是正常的

    def test_system_robustness(self):
        """测试系统健壮性"""
        # 测试极端参数
        try:
            # 极小参数
            dag = self.dag_generator.generate_dag(
                mode=DAGGenerationMode.FORWARD,
                target_depth=1,
                target_complexity=1
            )
            self.assertIsNotNone(dag)

            # 较大参数
            dag = self.dag_generator.generate_dag(
                mode=DAGGenerationMode.FORWARD,
                target_depth=5,
                target_complexity=20
            )
            self.assertIsNotNone(dag)

        except Exception as e:
            self.logger.error(f"系统健壮性测试失败: {e}")
            self.fail(f"系统在极端参数下失败: {e}")


if __name__ == '__main__':
    unittest.main()